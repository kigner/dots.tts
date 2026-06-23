from __future__ import annotations

import argparse
import json
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

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from dots_tts.runtime_double_streaming import DotsTtsRuntimeDoubleStreaming  # noqa: E402
from dots_tts.utils.logging import configure_logging  # noqa: E402
from dots_tts.utils.util import seed_everything  # noqa: E402


def stream_ollama_text(host: str, model: str, prompt: str):
    """Yield text deltas from a local Ollama chat stream (no extra deps)."""
    payload = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/chat", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            obj = json.loads(line)
            delta = obj.get("message", {}).get("content", "")
            if delta:
                yield delta
            if obj.get("done"):
                break


def make_player(sample_rate: int, enable: bool, prebuffer_seconds: float = 0.5):
    """Return (write_fn, close_fn).

    Buffered callback playback: chunks are queued into a ring buffer and a
    sounddevice callback drains it. A small pre-buffer absorbs the jitter from
    slow/uneven LLM token arrival, so playback stays gapless instead of
    under-running (the stutter you get from a blocking stream.write).
    """
    if not enable:
        return (lambda _arr: None), (lambda: None)
    try:
        import threading

        import numpy as np
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001
        logger.warning("Live playback unavailable ({}); falling back to wav-only.", exc)
        return (lambda _arr: None), (lambda: None)

    state = {"buf": np.zeros(0, dtype=np.float32), "started": False, "underruns": 0}
    lock = threading.Lock()
    prebuffer = int(sample_rate * prebuffer_seconds)

    def callback(outdata, frames, _time_info, _status):
        with lock:
            buf = state["buf"]
            n = min(frames, len(buf))
            outdata[:n, 0] = buf[:n]
            if n < frames:
                outdata[n:, 0] = 0.0
                state["underruns"] += 1
            state["buf"] = buf[n:]

    stream = sd.OutputStream(
        samplerate=sample_rate, channels=1, dtype="float32", callback=callback
    )
    logger.info(
        "Live playback enabled (buffered): device={} prebuffer={:.2f}s",
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

    def close():
        if not state["started"]:
            stream.start()
            state["started"] = True
        while True:  # drain remaining buffer before stopping
            with lock:
                remaining = len(state["buf"])
            if remaining <= 0:
                break
            time.sleep(0.05)
        time.sleep(0.15)
        logger.info("Playback underruns (gaps): {}", state["underruns"])
        stream.stop()
        stream.close()

    return write, close


def warmup_session(runtime, *, prompt_audio, num_steps, guidance_scale):
    """Run a throwaway session to warm GPU kernels so first-packet latency is low."""
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
    parser = argparse.ArgumentParser(description="Ollama -> dots.tts duplex streaming demo.")
    parser.add_argument("--model-name-or-path", default="rednote-hilab/dots.tts-mf")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="qwen2.5:7b")
    parser.add_argument("--prompt", default="用一句话介绍你自己，不超过20个字。")
    parser.add_argument("--prompt-audio", default=None)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.2)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--play", action="store_true", help="Live audio playback via sounddevice")
    parser.add_argument(
        "--prebuffer",
        type=float,
        default=0.5,
        help="Seconds of audio to buffer before playback starts (anti-stutter)",
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="ollama_duplex.wav")
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    runtime = DotsTtsRuntimeDoubleStreaming.from_pretrained(
        args.model_name_or_path,
        precision="bfloat16",
        optimize=args.optimize,
        max_generate_length=1000,
    )
    tok = runtime.model.tokenizer

    if not args.no_warmup:
        warmup_session(
            runtime,
            prompt_audio=args.prompt_audio,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
        )

    write_audio, close_player = make_player(runtime.sample_rate, args.play, args.prebuffer)
    session = runtime.start_double_streaming(
        prompt_audio_path=args.prompt_audio,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
    )

    chunks: list[torch.Tensor] = []
    first_audio_at: float | None = None
    first_push_at: float | None = None
    full_text = ""
    pushed = 0

    def push(tid: int):
        nonlocal first_audio_at, first_push_at
        if first_push_at is None:
            first_push_at = time.time()
        chunk = session.push_text_token(int(tid))
        if chunk is not None:
            if first_audio_at is None:
                first_audio_at = time.time()
            cpu_chunk = chunk.detach().cpu()
            chunks.append(cpu_chunk)
            write_audio(cpu_chunk.float().reshape(-1).numpy())

    t_start = time.time()
    print("\n=== Ollama 文本流 ===")
    for delta in stream_ollama_text(args.ollama_host, args.ollama_model, args.prompt):
        full_text += delta
        print(delta, end="", flush=True)
        ids = tok.encode(full_text, add_special_tokens=False)
        for tid in ids[pushed : max(pushed, len(ids) - 1)]:  # hold back last token
            push(tid)
        pushed = max(pushed, len(ids) - 1)
    print("\n=== 文本流结束，收尾解码 ===")

    ids = tok.encode(full_text, add_special_tokens=False)
    for tid in ids[pushed:]:
        push(tid)
    for chunk in session.finish_text():
        cpu_chunk = chunk.detach().cpu()
        chunks.append(cpu_chunk)
        write_audio(cpu_chunk.float().reshape(-1).numpy())

    close_player()
    if not chunks:
        raise RuntimeError("No audio produced.")

    audio = torch.cat(chunks, dim=-1)
    out = Path(args.output)
    sf.write(out, audio.float().squeeze().numpy(), runtime.sample_rate)

    total = time.time() - t_start
    audio_secs = audio.shape[-1] / runtime.sample_rate
    first_packet = (first_audio_at - first_push_at) if first_audio_at and first_push_at else -1
    logger.info(
        "Duplex demo done: text='{}' chunks={} audio_seconds={:.2f} wall_seconds={:.2f} "
        "rtf={:.2f} first_packet_latency_s={:.3f} output={}",
        full_text.strip(),
        len(chunks),
        audio_secs,
        total,
        total / audio_secs if audio_secs else float("inf"),
        first_packet,
        out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
