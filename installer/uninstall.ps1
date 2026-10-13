# Removes Magician: shortcuts, the entry in Windows "Apps" and the program folder.
# Games and the memory of mistakes live in the data folder (see magician.ini) and are kept.
$dir = $PSScriptRoot
Add-Type -AssemblyName System.Windows.Forms
$data = ""
if (Test-Path "$dir\magician.ini") {
    $line = Get-Content "$dir\magician.ini" | Where-Object { $_ -like "home=*" } | Select-Object -First 1
    if ($line) { $data = $line.Substring(5).Trim() }
}
$ok = [System.Windows.Forms.MessageBox]::Show("Удалить Magician из`n$dir ?`n`nСохранённые партии останутся в $data\games.", "Magician", "YesNo", "Question")
if ($ok -ne "Yes") { exit }
foreach ($l in (Join-Path ([Environment]::GetFolderPath("Programs")) "Magician.lnk"), (Join-Path ([Environment]::GetFolderPath("Desktop")) "Magician.lnk")) {
    Remove-Item $l -ErrorAction SilentlyContinue
}
Remove-Item "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Magician" -Recurse -ErrorAction SilentlyContinue
# The folder holds this script, so it is removed after the script exits.
Start-Process cmd.exe -ArgumentList "/c timeout /t 2 /nobreak >nul & rmdir /s /q `"$dir`"" -WindowStyle Hidden
