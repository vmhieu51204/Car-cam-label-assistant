param(
    [int] $Port = 8000,
    [string] $EnvFile = '',
    [string] $KaggleCommand = '',
    [switch] $Kaggle
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 or later is required.' }
}
& $pythonPath -m pip install -r requirements-mock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if ($Kaggle) {
    & $pythonPath -m pip install -r requirements-kaggle.txt
    if ($LASTEXITCODE -ne 0) { throw 'Kaggle dependency installation failed.' }
}
$mockArgs = @('run_mock.py', '--port', $Port)
if ($EnvFile) { $mockArgs += @('--env-file', $EnvFile) }
if ($KaggleCommand) { $mockArgs += @('--kaggle-command', $KaggleCommand) }
& $pythonPath @mockArgs
