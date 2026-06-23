"""Interactive voice chat: you type, DeepSeek cloud LLM answers, dots.tts speaks it live.

DeepSeek exposes an OpenAI-compatible /chat/completions endpoint with SSE streaming.
Everything else (TTS double-streaming, jitter-buffer player, hold-back-last-token trick)
is identical to ollama_voice_chat.py — only the LLM transport layer differs.

Usage:
    python scripts/deepseek_voice_chat.py \
        --api-key sk-... \
        --prompt-audio apps/gradio/default_prompts/xyj_woman.wav \
        --prebuffer 1.5 --quiet

The API key can also be set via the DEEPSEEK_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    p = str(import_root)
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from loguru import logger  # noqa: E402

from dots_tts.runtime_double_streaming import DotsTtsRuntimeDoubleStreaming  # noqa: E402
from dots_tts.utils.logging import configure_logging  # noqa: E402
from dots_tts.utils.util import seed_everything  # noqa: E402

DEFAULT_SYSTEM_PROMPT = (
    "你是一个语音助手。"
    "【严格规则】每次回答必须控制在3句话以内，绝对不能超过3句。"
    "回答要口语化、自然，像说话一样简洁。"
    "禁止使用 Markdown、列表、编号、代码块、表情符号或任何格式符号，"
    "因为你的文字会被实时语音合成直接朗读出来，内容过长会导致明显卡顿。"
    "如果问题需要很长的解释，只说最核心的一两点，其余略去。"
)

DEEPSEEK_API_BASE = "https://api.deepseek.com"


def stream_deepseek_chat(api_key: str, model: str, messages: list[dict], base_url: str):
    """Yield text deltas from DeepSeek's OpenAI-compatible streaming chat endpoint."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        resp_obj = urllib.request.urlopen(req)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError(
                "DeepSeek API 返回 401 Unauthorized。\n"
                "  请检查 DEEPSEEK_API_KEY 是否已填写真实 key（不是占位符）。\n"
                "  当前传入的 key 开头: " + api_key[:8] + "..."
            ) from exc
        raise
    with resp_obj as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            payload_str = line[len("data:"):].strip()
            if payload_str == "[DONE]":
                break
            obj = json.loads(payload_str)
            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta:
                yield delta


def make_player(sample_rate: int, enable: bool, prebuffer_seconds: float = 1.0):
    """Return (write, drain, close) for buffered, gapless multi-turn playback."""
    if not enable:
        return (lambda _a: None), (lambda: None), (lambda: None)
    try:
        import threading

        import numpy as np
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001
        logger.warning("Live playback unavailable ({}); running text-only.", exc)
        return (lambda _a: None), (lambda: None), (lambda: None)

    state = {"buf": np.zeros(0, dtype=np.float32), "started": False}
    lock = threading.Lock()
    prebuffer = int(sample_rate * prebuffer_seconds)

    def callback(outdata, frames, _t, _s):
        with lock:
            buf = state["buf"]
            n = min(frames, len(buf))
            outdata[:n, 0] = buf[:n]
            if n < frames:
                outdata[n:, 0] = 0.0
            state["buf"] = buf[n:]

    stream = sd.OutputStream(
        samplerate=sample_rate, channels=1, dtype="float32", callback=callback
    )
    logger.info(
        "Live playback enabled: device={} prebuffer={:.2f}s",
        sd.query_devices(kind="output")["name"],
        prebuffer_seconds,
    )

    def write(arr):
        with lock:
            state["buf"] = np.concatenate([state["buf"], arr.astype(np.float32)])
            buffered = len(state["buf"])
        if not state["started"] and buffered >= prebuffer:
            stream.start()
            state["started"] = True

    def drain():
        if not state["started"]:
            with lock:
                has = len(state["buf"])
            if has == 0:
                return
            stream.start()
            state["started"] = True
        while True:
            with lock:
                remaining = len(state["buf"])
            if remaining <= 0:
                break
            time.sleep(0.05)
        time.sleep(0.1)
        try:
            stream.stop()
        except Exception:  # noqa: BLE001
            pass
        state["started"] = False

    def close():
        try:
            if state["started"]:
                stream.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass

    return write, drain, close


def speak_reply(runtime, write_audio, *, text_stream, prompt_audio, num_steps,
                guidance_scale):
    """Stream text_stream deltas into a fresh TTS session, printing + speaking live."""
    tok = runtime.model.tokenizer
    session = runtime.start_double_streaming(
        prompt_audio_path=prompt_audio,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
    )
    chunks: list[torch.Tensor] = []
    reply = ""
    pushed = 0

    def push(tid: int):
        chunk = session.push_text_token(int(tid))
        if chunk is not None:
            cpu_chunk = chunk.detach().cpu()
            chunks.append(cpu_chunk)
            write_audio(cpu_chunk.float().reshape(-1).numpy())

    for delta in text_stream:
        reply += delta
        print(delta, end="", flush=True)
        ids = tok.encode(reply, add_special_tokens=False)
        for tid in ids[pushed : max(pushed, len(ids) - 1)]:
            push(tid)
        pushed = max(pushed, len(ids) - 1)

    ids = tok.encode(reply, add_special_tokens=False)
    for tid in ids[pushed:]:
        push(tid)
    for chunk in session.finish_text():
        cpu_chunk = chunk.detach().cpu()
        chunks.append(cpu_chunk)
        write_audio(cpu_chunk.float().reshape(-1).numpy())
    print()
    return reply.strip()


def warmup(runtime, *, prompt_audio, num_steps, guidance_scale):
    tok = runtime.model.tokenizer
    t0 = time.time()
    session = runtime.start_double_streaming(
        prompt_audio_path=prompt_audio, num_steps=num_steps, guidance_scale=guidance_scale
    )
    for tid in tok.encode("你好啊", add_special_tokens=False):
        session.push_text_token(int(tid))
    for _ in session.finish_text():
        pass
    logger.info("Warmup completed in {:.2f}s.", time.time() - t0)


def main(argv=None):
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Interactive voice chat: type -> DeepSeek LLM -> dots.tts speaks live."
    )
    parser.add_argument("--model-name-or-path", default="rednote-hilab/dots.tts-mf")
    parser.add_argument(
        "--api-key",
        default=None,
        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)",
    )
    parser.add_argument(
        "--deepseek-model",
        default="deepseek-v4-pro",
        help="DeepSeek model name, e.g. deepseek-chat / deepseek-reasoner",
    )
    parser.add_argument(
        "--api-base",
        default=DEEPSEEK_API_BASE,
        help="Base URL of the API (default: https://api.deepseek.com)",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--prompt-audio", default=None, help="Reference wav for voice")
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.2)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument(
        "--prebuffer",
        type=float,
        default=1.0,
        help="Seconds of audio to buffer before playback starts (anti-stutter)",
    )
    parser.add_argument("--no-play", action="store_true", help="Disable live playback")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence per-turn INFO logs so the chat stays clean",
    )
    parser.add_argument(
        "--max-generate-length",
        type=int,
        default=128,
        help="Max audio patches to generate per turn (128=20s, 256=41s). Lower = less VRAM.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("[ERROR] 未提供 DeepSeek API Key。")
        print("  方法1: --api-key sk-...")
        print("  方法2: 设置环境变量 DEEPSEEK_API_KEY=sk-...")
        sys.exit(1)

    seed_everything(args.seed)
    runtime = DotsTtsRuntimeDoubleStreaming.from_pretrained(
        args.model_name_or_path,
        precision="bfloat16",
        optimize=args.optimize,
        max_generate_length=args.max_generate_length,
    )

    if not args.no_warmup:
        warmup(
            runtime,
            prompt_audio=args.prompt_audio,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
        )

    write_audio, drain_player, close_player = make_player(
        runtime.sample_rate, not args.no_play, args.prebuffer
    )

    if args.quiet:
        configure_logging(level="WARNING")

    messages = [{"role": "system", "content": args.system_prompt}]
    print("\n" + "=" * 60)
    print("  dots.tts 语音对话 (DeepSeek)  (输入问题回车; /quit 退出)")
    print("  model: {}  voice: {}".format(args.deepseek_model, args.prompt_audio or "default"))
    print("=" * 60)

    try:
        while True:
            try:
                user_text = input("\n你 > ").strip()
            except EOFError:
                break
            if not user_text:
                continue
            if user_text.lower() in {"/quit", "/exit", "quit", "exit"}:
                break

            messages.append({"role": "user", "content": user_text})
            print("助手 > ", end="", flush=True)
            t0 = time.time()
            text_stream = stream_deepseek_chat(
                api_key, args.deepseek_model, messages, args.api_base
            )
            reply = speak_reply(
                runtime,
                write_audio,
                text_stream=text_stream,
                prompt_audio=args.prompt_audio,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
            )
            drain_player()
            logger.info("turn done in {:.2f}s", time.time() - t0)
            if reply:
                messages.append({"role": "assistant", "content": reply})
    except KeyboardInterrupt:
        print("\n(interrupted)")
    finally:
        close_player()
        print("再见。")


if __name__ == "__main__":
    raise SystemExit(main())
