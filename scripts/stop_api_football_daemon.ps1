param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$RuntimeDir = Join-Path $RepoRoot "runtime"
$StopFile = Join-Path $RuntimeDir "api_football_today_daemon.stop"
$LockFile = Join-Path $RuntimeDir "api_football_today_daemon.lock"

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
Set-Content -LiteralPath $StopFile -Value "stop" -Encoding utf8

if ($Force -and (Test-Path -LiteralPath $LockFile)) {
    try {
        $Payload = Get-Content -LiteralPath $LockFile -Raw | ConvertFrom-Json
        if ($Payload.pid) {
            Stop-Process -Id ([int]$Payload.pid) -Force -ErrorAction SilentlyContinue
        }
    }
    catch {
        Write-Warning "Could not parse daemon lock file: $($_.Exception.Message)"
    }
}

Write-Host "Stop signal written for Spots-Quant API-Football daemon."
