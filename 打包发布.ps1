# ============================================================================
#  打包发布.ps1  —  Build a clean, shippable copy of the dots.tts portable bundle.
#
#  What it does (NON-destructive: your dev folder is never modified):
#    1. robocopy this folder into a fresh staging dir, EXCLUDING dev cruft
#       (.git, .codegraph, .cursor, pack, __pycache__, HF xet download cache, ...).
#    2. In the STAGED COPY only:
#         - drop ALL sample voices under default_prompts (ship none).
#         - bundle the MSVC C++ runtime (msvcp140.dll, ...) next to python.exe so
#           it runs on a clean machine with no "VC++ Redistributable" install.
#         - scrub the personal DeepSeek API key  ->  sk-your-key-here
#         - inject  HF_HUB_OFFLINE=1  so the bundled models load with no network.
#    3. Compress the staging dir into a single .zip (7-Zip if present, else
#       Windows' built-in tar, else Compress-Archive).
#
#  Comments are intentionally ASCII/English: a non-BOM UTF-8 .ps1 with Chinese
#  text gets mis-decoded by Windows PowerShell 5.1 and can fail to parse.
#
#  Usage (from this folder):
#     powershell -ExecutionPolicy Bypass -File .\打包发布.ps1
#     powershell -ExecutionPolicy Bypass -File .\打包发布.ps1 -NoZip   # stage only
#
#  DISK: staging is a full copy (~18-20 GB) and the zip is roughly the same, so
#  make sure the target drive has ~40 GB free before running.
# ============================================================================
[CmdletBinding()]
param(
    [string]$OutDir     = (Join-Path (Split-Path $PSScriptRoot -Parent) "dots.tts-release"),
    [string]$FolderName = "dots.tts",
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"
$Src   = $PSScriptRoot
$Stage = Join-Path $OutDir $FolderName

Write-Host "============================================================"
Write-Host " dots.tts release packer"
Write-Host "   source : $Src"
Write-Host "   staging: $Stage"
Write-Host "============================================================"

# ---- 1) fresh staging dir --------------------------------------------------
if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

# ---- 2) copy with excludes -------------------------------------------------
# Root-only dirs: give full paths so we never drop a same-named dir deep inside
# a package. Recursive name patterns (caches) are matched anywhere.
$excludeDirs = @(
    (Join-Path $Src ".git"),
    (Join-Path $Src ".codegraph"),
    (Join-Path $Src ".cursor"),
    (Join-Path $Src ".ruff_cache"),
    (Join-Path $Src "pack"),
    "__pycache__",
    "xet",            # hf_cache\xet : HF download cache, not needed to run
    "*.egg-info"
)
$excludeFiles = @(
    "get-pip.py",
    "*.pyc",
    "ollama_duplex.wav",
    "录视频Demo.bat",          # personal/local use only, do not ship
    "2026-06-19-223506-readmemdfine-tuning.txt"
)

$rc = @($Src, $Stage, "/E", "/COPY:DAT", "/DCOPY:DAT", "/R:1", "/W:1", "/XJ", "/NP", "/NFL", "/NDL", "/NJH")
$rc += "/XD"; $rc += $excludeDirs
$rc += "/XF"; $rc += $excludeFiles

Write-Host "[1/3] copying (robocopy) ..."
robocopy @rc | Out-Null
# robocopy exit codes 0-7 = success, 8+ = real error
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE" }
$global:LASTEXITCODE = 0
Write-Host "      done."

# ---- drop ALL bundled sample voices from the staged copy -------------------
# Per request: no sample timbres ship. The WebUI handles an empty preset folder
# fine (it falls back to "No Preset"; users upload their own reference audio).
# NOTE: this clears EVERY audio file under default_prompts in the STAGED copy.
# If you want a specific demo voice to ship, keep it OUTSIDE default_prompts and
# point the .bat at it, or comment this block out.
$stagedPrompts = Join-Path $Stage "apps\gradio\default_prompts"
if (Test-Path $stagedPrompts) {
    $audioExt = @(".wav", ".mp3", ".flac", ".m4a", ".ogg")
    Get-ChildItem -LiteralPath $stagedPrompts -File |
        Where-Object { $audioExt -contains $_.Extension.ToLower() } |
        ForEach-Object {
            Remove-Item -LiteralPath $_.FullName -Force
            Write-Host "      dropped sample voice: $($_.Name)"
        }
}

# ---- bundle the MSVC C++ runtime so it runs on a CLEAN machine --------------
# torch / numpy need msvcp140.dll, which the embeddable Python does NOT include
# and torch does NOT bundle (only vcruntime140* ship). Without it a clean PC
# fails at "import torch" with "msvcp140.dll not found". These VC++ runtime DLLs
# are redistributable, so copy them next to python.exe -> fully self-contained.
$vcDlls = @("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "concrt140.dll", "vccorlib140.dll")
$sys32  = Join-Path $env:SystemRoot "System32"
$pyRoot = Join-Path $Stage "python-3.11.9"
foreach ($d in $vcDlls) {
    $srcDll = Join-Path $sys32 $d
    if (Test-Path $srcDll) {
        Copy-Item -LiteralPath $srcDll -Destination $pyRoot -Force
        Write-Host "      bundled VC++ runtime: $d"
    } else {
        Write-Host "      WARN: $d not found in System32 (skipped)"
    }
}
if (-not (Test-Path (Join-Path $pyRoot "msvcp140.dll"))) {
    throw "ABORT: msvcp140.dll not found in System32. Install the VC++ Redistributable on THIS build machine, then re-run."
}

# ---- 3) patch the staged .bat launchers ------------------------------------
# Write back as ASCII (no BOM) so cmd.exe still parses the .bat correctly.
function Write-Ascii([string]$Path, [string]$Text) {
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.Encoding]::ASCII)
}

Write-Host "[2/3] scrubbing key + forcing offline HF ..."
Get-ChildItem $Stage -Filter *.bat | ForEach-Object {
    $p = $_.FullName
    $t = [System.IO.File]::ReadAllText($p)
    $orig = $t

    # a) replace any real  sk-xxxxx  key with a placeholder
    $t = [regex]::Replace($t, 'set "DEEPSEEK_API_KEY=sk-[A-Za-z0-9]+"', 'set "DEEPSEEK_API_KEY=sk-your-key-here"')

    # b) models are bundled -> load offline, no huggingface.co round-trip
    if (($t -match 'set "HF_HOME=%~dp0hf_cache"') -and ($t -notmatch 'HF_HUB_OFFLINE')) {
        $t = $t -replace '(set "HF_HOME=%~dp0hf_cache")', "`$1`r`nset `"HF_HUB_OFFLINE=1`""
    }

    if ($t -ne $orig) { Write-Ascii $p $t; Write-Host "      patched: $($_.Name)" }
}

# sanity: fail loudly if any real key slipped through
$leak = Select-String -Path (Join-Path $Stage "*.bat") -Pattern 'DEEPSEEK_API_KEY=sk-(?!your-key-here)' -ErrorAction SilentlyContinue
if ($leak) { throw "ABORT: a real API key is still present in the staged copy: $($leak.Path)" }
Write-Host "      no API key leaked into the package. OK."

# ---- 4) compress -----------------------------------------------------------
if ($NoZip) {
    Write-Host "[3/3] -NoZip set; staging ready at: $Stage"
    return
}

$zip = Join-Path $OutDir "$FolderName-portable.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }

$sevenZip = (Get-Command 7z.exe -ErrorAction SilentlyContinue).Source
if (-not $sevenZip -and (Test-Path "$env:ProgramFiles\7-Zip\7z.exe")) { $sevenZip = "$env:ProgramFiles\7-Zip\7z.exe" }

Write-Host "[3/3] compressing -> $zip"
if ($sevenZip) {
    # -mx=1: fast; payload is mostly incompressible (.safetensors / .dll) anyway
    & $sevenZip a -tzip -mx=1 -mmt=on $zip $Stage | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "7-Zip failed with exit code $LASTEXITCODE" }
} elseif ((Get-Command tar.exe -ErrorAction SilentlyContinue)) {
    Push-Location $OutDir
    try { & tar.exe -a -c -f $zip $FolderName } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { throw "tar failed with exit code $LASTEXITCODE" }
} else {
    Write-Host "      (no 7-Zip/tar found; using Compress-Archive, this is slow)"
    Compress-Archive -Path $Stage -DestinationPath $zip -CompressionLevel Optimal
}

$gb = [math]::Round((Get-Item $zip).Length / 1GB, 2)
Write-Host "============================================================"
Write-Host " DONE.  $zip  ($gb GB)"
Write-Host " Distribute that single .zip. Recipients extract -> run a .bat."
Write-Host "============================================================"
