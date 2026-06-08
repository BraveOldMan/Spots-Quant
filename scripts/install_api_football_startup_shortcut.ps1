param(
    [int]$IntervalMinutes = 15,
    [string]$ShortcutName = "SpotsQuant API-Football Daemon.lnk"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$StartScript = Join-Path $ScriptDir "start_api_football_daemon.ps1"
$StartupDir = [Environment]::GetFolderPath("Startup")
$ShortcutPath = Join-Path $StartupDir $ShortcutName

if (-not (Test-Path -LiteralPath $StartScript)) {
    throw "Missing start script: $StartScript"
}

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "powershell.exe"
$Shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$StartScript`" -IntervalMinutes $IntervalMinutes"
$Shortcut.WorkingDirectory = $RepoRoot
$Shortcut.WindowStyle = 7
$Shortcut.Description = "Start the Spots-Quant API-Football local data daemon at logon."
$Shortcut.Save()

Write-Host "Installed startup shortcut: $ShortcutPath"
