# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**dots.tts** is a 2B-parameter, fully continuous, end-to-end autoregressive text-to-speech system. There are **no discrete audio tokens anywhere** in the pipeline — the backbone predicts a continuous VAE latent one patch at a time. The package ships inference, fine-tuning, and MeanFlow-distillation code, and an `src/` layout installable as `dots.tts` (import name `dots_tts`).

## Commands

```bash
# Install (Python 3.10–3.12). Constraints file pins recommended versions.
python -m pip install -e . -c constraints/recommended.txt
python -m pip install -e .[full] -c constraints/recommended.txt   # adds accelerate, tensorboard, ruff

# Lint / format (ruff; config in pyproject.toml, line-length 88, E501 ignored)
ruff check src
ruff format src

# Inference CLI (installed entry point `dots.tts` -> dots_tts.cli:main)
dots.tts --model-name-or-path rednote-hilab/dots.tts-soar \
  --text "..." --prompt-audio ref.wav --prompt-text "ref transcript" \
  --num-steps 10 --output out.wav
dots.tts --help     # full flag list

# Gradio web demo
python apps/gradio/app.py --model-name-or-path rednote-hilab/dots.tts-soar --optimize

# Duplex demo: stream a local Ollama LLM's text into double-streaming TTS (live playback)
python scripts/demo_ollama_duplex.py --ollama-model qwen2.5:1.5b \
  --prompt-audio ref.wav --play
# Key caveat: Ollama emits TEXT, not dots.tts tokens. The script re-encodes the LLM's
# streamed text with dots.tts's own tokenizer (hold-back-last-token to keep BPE merges
# stable), then pushes ids one-by-one into DoubleStreamingSession.push_text_token().
# --play uses a pre-buffered sounddevice callback (jitter buffer) to smooth uneven LLM
# token arrival; --no-warmup skips the GPU kernel warmup pass. Note: --optimize
# (torch.compile) needs Triton, which is unavailable on the Windows embedded-python
# bundle, so without compile TTS RTF stays >1 and truly gapless live playback isn't
# achievable on a single Windows GPU — generate-to-wav (omit --play) for stutter-free.

# Interactive voice chat: a multi-turn REPL where you type, a local Ollama LLM answers,
# and dots.tts speaks each reply live. The interactive sibling of demo_ollama_duplex.py.
python scripts/ollama_voice_chat.py --ollama-model qwen2.5:1.5b \
  --prompt-audio ref.wav --prebuffer 1.5
# This script IS the chat window (replaces `ollama run`): you cannot reliably tee an
# external interactive REPL's stdout on Windows without breaking it, so the script owns
# the input loop, streams each turn's reply token-by-token into a fresh
# DoubleStreamingSession (same hold-back-last-token re-encoding as the duplex demo), and
# plays it through a re-arming buffered jitter player. A built-in voice-assistant system
# prompt keeps replies short/spoken-friendly; --no-play for text-only, --system-prompt to
# override the persona. Same Triton/RTF caveat as above: voice lags slightly; raise
# --prebuffer for fewer gaps.

# Fine-tune from a released checkpoint
accelerate launch scripts/train_dots_tts.py --config configs/dots_tts.yaml

# MeanFlow distillation (student vs frozen flow-matching teacher)
accelerate launch --num_processes 2 --mixed_precision bf16 \
  scripts/train_dots_tts_meanflow.py --config configs/dots_tts_meanflow.yaml \
  --teacher-model-path pretrained_models/dots.tts-soar

# Build LJSpeech-48kHz smoke-test JSONL manifests
python scripts/prepare_train_jsonl_manifest.py --output-dir downloaded_data
```

`--model-name-or-path` accepts a local directory **or** an HF repo id; repo ids are fetched via `snapshot_download` and cached on first use.

**There is no automated test suite.** Validate changes by running the smoke configs (`configs/dots_tts.yaml`, `configs/dots_tts_meanflow.yaml` are tiny end-to-end pipeline checks), a CLI inference, or `scripts/example_double_streaming.py`.

## Architecture

The generation pipeline is a continuous AR loop (see `src/dots_tts/models/dots_tts/`):

- **`DotsTtsModel`** (`model.py`) — top-level `nn.Module`. Owns `from_pretrained`/`save_pretrained` (safetensors), warmup/`torch.compile` optimization, and `generate_audio`. Composes the frozen `AudioVAE` and `SpeakerXVectorFeatures` around the core.
- **`DotsTtsCore`** (`core.py`) — the AR backbone, three coupled components:
  - **LLM** — `Qwen2ForCausalLM` initialized from Qwen2.5-1.5B-Base, consumes BPE text directly (no phonemes), emits one hidden state per audio step.
  - **Semantic encoder** (`VAESemanticEncoder`, `modules/backbone/semantic_encoder.py`) — re-encodes each generated VAE patch into a compact embedding fed back to the LLM.
  - **Velocity field predictor** (`DiT`, `modules/backbone/dit.py`) — AR flow-matching head that denoises the next VAE latent patch, conditioned on the LLM hidden state, AR prefix, and CAM++ speaker x-vector.
  - Also: an EOS predictor head, and helpers `FlowMatchingHelper` / `CausalHelper` / `IOHelper` (latent normalization stats).
- **`AudioVAE`** (`modules/vocoder/bigvgan.py`) — frozen 48 kHz encoder + BigVGAN-style causal decoder, encodes/decodes the continuous latent. Vocoder sample rate drives `runtime.sample_rate`.
- **CAM++** (`modules/speaker/`) — frozen speaker x-vector encoder providing timbre conditioning.

### Two DiT modes

`DiT` runs in `"flow_matching"` (teacher / base / SCA checkpoints, multi-step ODE sampling) or `"meanflow"` (distilled student, adds a duration embedder, few-step `--num-steps 4`). The mode is selected from `config.meanflow.enabled` in `DotsTtsCore.__init__`. MeanFlow fuses CFG into the student, so `guidance_scale` only applies to flow-matching.

### Two sequence layouts

Controlled by template (`RUNTIME_TEMPLATE_BY_NAME` in `runtime.py`, `--template-name`):
- **plain** — full text prefix before the audio span (standard TTS).
- **1T1A interleaved** (`tts_interleave`) — alternates one BPE token with one audio step for low-latency duplex streaming. See `scripts/example_double_streaming.py` and `runtime_double_streaming.py`.

Special tokens delimiting the layout live in `utils/tokenizer.py` (`<|audio_gen_span|>`, `<|audio_comp_span|>`, `<|audio_gen_start|>`, `<|text_cond_end|>`); resolve them with `require_token_id`.

### Runtime entry points

`DotsTtsRuntime` (`runtime.py`) is the inference facade used by both the CLI and Gradio app: `from_pretrained(...)` then `generate(...)` (full clip) or `generate_stream(...)` (yields chunks). The CLI is a thin argparse wrapper in `cli.py`; the Gradio app lives under `apps/gradio/`.

## Config system

Training/data config is **pydantic + YAML**, not argparse. `AppConfig` (`config/app.py`) composes `DataConfig` + `TrainConfig` + `LossConfig`, loaded via `AppConfig.from_yaml`. `ConfigBase` (`config/base.py`) allows extra keys (`extra="allow"`) and exposes dict-like `.get()`; `StrictConfigBase` forbids extras. Model-architecture config (`ModelConfig`, `DiT`, `meanflow`, vocoder) lives in `models/dots_tts/config.py`. The YAML smoke configs in `configs/` are intended to be edited (paths, sources, `max_train_steps`) for real runs.

## Data pipeline

Streaming, resumable, multi-source. Read **`src/dots_tts/data/EXTENSION.md`** before touching data code — it is the authoritative guide. Flow: **source adapter** (reads raw, yields samples) → **sample pipeline** (strict 1:1 transform, no filtering/expansion) → **multi-source wrapper** (`WeightedMultiSourceAdapter` for train, `SequentialMultiSourceAdapter` for val) → `StreamingSampleDataset`/`DataLoader` → `OnlineBatcher` → `PadCollator`. New adapters/pipelines must be registered in `data/builders.py` (`_SOURCE_ADAPTER_CLASSES`, `_build_source_pipeline`) to be selectable by name in YAML. Manifest format is JSONL, one object per line with at least `fid`, `audio` (abs path), `text`.

## Conventions

- `from __future__ import annotations` at the top of modules; logging via `loguru`.
- `# region` / `# endregion` comments group long classes (e.g. `core.py`, `model.py`, training scripts) — keep them paired when editing.
- Checkpoints are safetensors directories (`model/` subdir), not single files; MeanFlow training saves only the student, never the frozen teacher.
