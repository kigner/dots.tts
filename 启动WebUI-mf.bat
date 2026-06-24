@echo off
chcp 65001 >nul
setlocal enableextensions
cd /d "%~dp0"

rem ASCII-only launcher (Chinese in .bat breaks cmd parsing on GBK systems).

set "PYTHON=%~dp0python-3.11.9\python.exe"

rem ---- HuggingFace: download from the OFFICIAL site (needs direct access / VPN) ----
rem  To use the China mirror instead, comment the line below and uncomment the mirror line.
set "HF_ENDPOINT=https://huggingface.co"
rem set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HOME=%~dp0hf_cache"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PYTHONUTF8=1"
set "GRADIO_ANALYTICS_ENABLED=False"

rem ---- TorchInductor settings so --optimize (torch.compile) is stable on this bundle ----
set "TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0"
set "TORCHINDUCTOR_CUDAGRAPHS=0"

rem ---- auto-open browser when the server is ready ----
set "DOTS_TTS_OPEN_BROWSER=1"

if not exist "%PYTHON%" (
  echo [ERROR] Embedded Python not found: %PYTHON%
  echo Make sure the python-3.11.9 folder sits next to this script.
  pause
  exit /b 1
)

echo ============================================================
echo   dots.tts WebUI  -  starting ...
echo.
echo   * Default model: rednote-hilab/dots.tts-mf  (4 steps, low VRAM)
echo   * First launch downloads the model (several GB) from huggingface.co.
echo     Progress shows in THIS window - please wait, do NOT close it.
echo   * When ready the browser opens at  http://127.0.0.1:7860
echo ============================================================
echo.

"%PYTHON%" "%~dp0apps\gradio\app.py" --model-name-or-path rednote-hilab/dots.tts-mf --default-num-steps 4 --max-generate-length 128 --optimize --host 127.0.0.1 --port 7860

echo.
echo WebUI stopped. Press any key to close this window.
pause >nul
