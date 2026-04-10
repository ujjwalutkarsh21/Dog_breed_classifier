$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
if ($null -eq $PSScriptRoot) { $PSScriptRoot = Get-Location }

$PythonPath = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$ScriptPath = Join-Path $PSScriptRoot "scripts\live_detect.py"

Write-Host "Starting Dog Breed Live Detection..." -ForegroundColor Cyan

if (Test-Path $PythonPath) {
    & $PythonPath $ScriptPath
} else {
    Write-Error "Virtual environment 'venv' not found at: $PythonPath"
    Write-Host "Please ensure you have created the venv and installed requirements." -ForegroundColor Yellow
    Read-Host "Press Enter to continue"
}


# How to run: Type .\Run_Live_Detection.ps1 in terminal