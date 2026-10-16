param(
    [int] $Port = 8000,
    [string] $EnvFile = '',
    [string] $KaggleCommand = '',
    [switch] $Kaggle
)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'start-mock.ps1') -Port $Port -EnvFile $EnvFile -KaggleCommand $KaggleCommand -Kaggle:$Kaggle
