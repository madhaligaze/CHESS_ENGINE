# Magician installer. Run by MagicianSetup.exe from the folder it unpacks to (all files flat).
# Copies the program, writes magician.ini (where games and the memory of mistakes are kept),
# creates Start menu and desktop shortcuts and an entry in Windows "Apps" for uninstalling.
# No administrator rights are needed.
param([string]$InstallDir = "", [string]$DataDir = "", [switch]$Quiet)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
$src = $PSScriptRoot

# setup.cfg is written by scripts\package.ps1: project=<folder of the Magician project>.
$project = ""
$cfgPath = Join-Path $src "setup.cfg"
if (Test-Path $cfgPath) {
    $line = Get-Content $cfgPath | Where-Object { $_ -like "project=*" } | Select-Object -First 1
    if ($line) { $project = $line.Substring(8).Trim() }
}
$haveProject = $project -and (Test-Path (Join-Path $project "Cargo.toml"))
if (-not $InstallDir) {
    $InstallDir = if ($haveProject) { Join-Path $project "app" } else { Join-Path $env:LOCALAPPDATA "Programs\Magician" }
}
if (-not $DataDir) {
    $DataDir = if ($haveProject) { $project } else { $InstallDir }
}

try {
    foreach ($d in $InstallDir, "$InstallDir\nets", "$InstallDir\extension", $DataDir) {
        New-Item -ItemType Directory -Force $d | Out-Null
    }
    Copy-Item "$src\magician.exe", "$src\uninstall.ps1", "$src\magician.ico" $InstallDir -Force
    foreach ($n in "base.nn", "tal.nn", "carlsen.nn") { Copy-Item "$src\$n" "$InstallDir\nets" -Force }
    foreach ($f in "manifest.json", "background.js", "content.js") { Copy-Item "$src\$f" "$InstallDir\extension" -Force }
    [System.IO.File]::WriteAllText("$InstallDir\magician.ini", "home=$DataDir`r`n")

    $ws = New-Object -ComObject WScript.Shell
    $links = @(
        (Join-Path ([Environment]::GetFolderPath("Programs")) "Magician.lnk"),
        (Join-Path ([Environment]::GetFolderPath("Desktop")) "Magician.lnk")
    )
    foreach ($l in $links) {
        $s = $ws.CreateShortcut($l)
        $s.TargetPath = "$InstallDir\magician.exe"
        $s.Arguments = "play"
        $s.WorkingDirectory = $InstallDir
        $s.IconLocation = "$InstallDir\magician.ico"
        $s.Description = "Magician: шахматный движок, который играет как человек"
        $s.Save()
    }

    $key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Magician"
    New-Item -Path $key -Force | Out-Null
    $props = @{
        DisplayName = "Magician"
        DisplayVersion = "0.1"
        Publisher = "Magician"
        InstallLocation = $InstallDir
        DisplayIcon = "$InstallDir\magician.ico"
        UninstallString = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$InstallDir\uninstall.ps1`""
    }
    foreach ($k in $props.Keys) { Set-ItemProperty -Path $key -Name $k -Value $props[$k] }
    Set-ItemProperty -Path $key -Name NoModify -Value 1 -Type DWord
    Set-ItemProperty -Path $key -Name NoRepair -Value 1 -Type DWord

    if (-not $Quiet) {
        $msg = "Magician установлен в`n$InstallDir`n`nПартии и память ошибок: $DataDir`nЯрлык Magician на рабочем столе и в меню Пуск.`n`nЗапустить сейчас?"
        $r = [System.Windows.Forms.MessageBox]::Show($msg, "Magician", "YesNo", "Information")
        if ($r -eq "Yes") { Start-Process "$InstallDir\magician.exe" -ArgumentList "play" -WorkingDirectory $InstallDir }
    }
} catch {
    if (-not $Quiet) { [System.Windows.Forms.MessageBox]::Show("Не удалось установить Magician:`n$_", "Magician", "OK", "Error") | Out-Null }
    throw
}
