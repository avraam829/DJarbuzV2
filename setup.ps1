$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path .venv/Scripts/python.exe)) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }
}
& ./.venv/Scripts/python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir --keyring-provider disabled --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt --no-cache-dir --keyring-provider disabled --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'Dependencies installation failed.' }
& ./.venv/Scripts/python.exe tools/export_voice.py
if ($LASTEXITCODE -ne 0) { throw 'Voice export failed.' }
