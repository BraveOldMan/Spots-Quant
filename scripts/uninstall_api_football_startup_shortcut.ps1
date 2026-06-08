param(
    [string]$ShortcutName = "SpotsQuant API-Football Daemon.lnk"
)

$ErrorActionPreference = "Stop"
$StartupDir = [Environment]::GetFolderPath("Startup")
$ShortcutPath = Join-Path $StartupDir $ShortcutName

if (Test-Path -LiteralPath $ShortcutPath) {
    Remove-Item -LiteralPath $ShortcutPath -Force
    Write-Host "Removed startup shortcut: $ShortcutPath"
}
else {
    Write-Host "Startup shortcut not found: $ShortcutPath"
}
