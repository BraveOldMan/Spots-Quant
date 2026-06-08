param(
    [int]$IntervalMinutes = 15
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$RuntimeDir = Join-Path $RepoRoot "runtime"
$LogsDir = Join-Path $RepoRoot "logs"
$StopFile = Join-Path $RuntimeDir "api_football_today_daemon.stop"
$StdoutLog = Join-Path $LogsDir "api_football_today_daemon_launcher.log"
$StderrLog = Join-Path $LogsDir "api_football_today_daemon_launcher.err.log"

New-Item -ItemType Directory -Force -Path $RuntimeDir, $LogsDir | Out-Null
if (Test-Path -LiteralPath $StopFile) {
    Remove-Item -LiteralPath $StopFile -Force
}

$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}

if ($env:SPOTS_API_FOOTBALL_DAEMON_INTERVAL_MINUTES) {
    $IntervalMinutes = [int]$env:SPOTS_API_FOOTBALL_DAEMON_INTERVAL_MINUTES
}

$Arguments = @(
    "api_football_today_daemon.py",
    "--daemon",
    "--interval-minutes",
    "$IntervalMinutes"
)

$Command = (
    "cmd.exe /c `"cd /d `"$RepoRoot`" && " +
    "`"$PythonExe`" " +
    ($Arguments -join " ") +
    " >> `"$StdoutLog`" 2>> `"$StderrLog`"`""
)
$Shell = New-Object -ComObject WScript.Shell
$Shell.Run($Command, 0, $false) | Out-Null

Write-Host "Started Spots-Quant API-Football daemon."
