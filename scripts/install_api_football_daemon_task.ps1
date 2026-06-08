param(
    [string]$TaskName = "SpotsQuant_API_Football_Daemon",
    [int]$IntervalMinutes = 15
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$StartScript = Join-Path $ScriptDir "start_api_football_daemon.ps1"

if (-not (Test-Path -LiteralPath $StartScript)) {
    throw "Missing start script: $StartScript"
}

$Argument = "-NoProfile -ExecutionPolicy Bypass -File `"$StartScript`" -IntervalMinutes $IntervalMinutes"
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $Argument -WorkingDirectory $RepoRoot
$Trigger = New-ScheduledTaskTrigger -AtLogOn
$Settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -AllowStartIfOnBatteries `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Start the Spots-Quant API-Football local data daemon at logon." `
    -Force | Out-Null

Write-Host "Installed scheduled task: $TaskName"
