# Builds the Windows distribution into dist\:
#   dist\MagicianSetup.exe        installer (made with IExpress, which ships with Windows)
#   dist\Magician-portable.zip    the same program without installation
#   powershell -ExecutionPolicy Bypass -File scripts\package.ps1
# The release build goes to target\dist-build, so a running magician.exe is not in the way.

param([switch]$SkipBuild)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
$target = Join-Path $root "target\dist-build"
if (-not $SkipBuild) {
    cargo build --release --target-dir $target
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
}
$exe = Join-Path $target "release\magician.exe"
$dist = Join-Path $root "dist"
$stage = Join-Path $dist "stage"
foreach ($d in $stage, "$dist\Magician-portable") { if (Test-Path $d) { Remove-Item $d -Recurse -Force } }
New-Item -ItemType Directory -Force $stage | Out-Null

# Files of the installer, flat (IExpress unpacks everything into one folder).
Copy-Item $exe $stage
Copy-Item "$root\nets\base.nn", "$root\nets\tal.nn", "$root\nets\carlsen.nn" $stage
Copy-Item "$root\tools\chesscom\extension\*" $stage
# Windows PowerShell 5.1 reads UTF-8 scripts correctly only with a BOM.
$bom = New-Object System.Text.UTF8Encoding $true
foreach ($s in "install.ps1", "uninstall.ps1") {
    [IO.File]::WriteAllText("$stage\$s", [IO.File]::ReadAllText("$root\installer\$s", [Text.Encoding]::UTF8), $bom)
}
[IO.File]::WriteAllText("$stage\setup.cfg", "project=$root`r`n")

# Icon: a white knight on a green circle, stored as PNG inside an .ico file.
Add-Type -AssemblyName System.Drawing
$size = 256
$bmp = New-Object System.Drawing.Bitmap $size, $size
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = "AntiAlias"
$g.TextRenderingHint = "AntiAlias"
$g.Clear([System.Drawing.Color]::Transparent)
$g.FillEllipse((New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(255, 129, 182, 76))), 4, 4, $size - 8, $size - 8)
$font = New-Object System.Drawing.Font("Segoe UI Symbol", 170, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
$fmt = New-Object System.Drawing.StringFormat
$fmt.Alignment = "Center"
$fmt.LineAlignment = "Center"
$g.DrawString([string][char]0x265E, $font, [System.Drawing.Brushes]::White, (New-Object System.Drawing.RectangleF(0, 8, $size, $size)), $fmt)
$g.Dispose()
$png = New-Object System.IO.MemoryStream
$bmp.Save($png, [System.Drawing.Imaging.ImageFormat]::Png)
$bytes = $png.ToArray()
$ico = New-Object System.IO.MemoryStream
$w = New-Object System.IO.BinaryWriter $ico
$w.Write([uint16]0); $w.Write([uint16]1); $w.Write([uint16]1)          # header: icon, 1 image
$w.Write([byte]0); $w.Write([byte]0); $w.Write([byte]0); $w.Write([byte]0)  # 256x256, no palette
$w.Write([uint16]1); $w.Write([uint16]32)                               # planes, bits per pixel
$w.Write([uint32]$bytes.Length); $w.Write([uint32]22)                   # size, offset
$w.Write($bytes)
$w.Flush()
[IO.File]::WriteAllBytes("$stage\magician.ico", $ico.ToArray())

# Portable folder and zip.
$portable = "$dist\Magician-portable"
New-Item -ItemType Directory -Force "$portable\nets", "$portable\extension" | Out-Null
Copy-Item "$stage\magician.exe", "$stage\magician.ico" $portable
Copy-Item "$stage\*.nn" "$portable\nets"
Copy-Item "$stage\manifest.json", "$stage\background.js", "$stage\content.js" "$portable\extension"
$zip = "$dist\Magician-portable.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "$portable\*" -DestinationPath $zip

# Installer with IExpress.
$setup = "$dist\MagicianSetup.exe"
$files = Get-ChildItem $stage -File | ForEach-Object { $_.Name }
$sed = @"
[Version]
Class=IEXPRESS
SEDVersion=3
[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=1
HideExtractAnimation=1
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=N
InstallPrompt=%InstallPrompt%
DisplayLicense=%DisplayLicense%
FinishMessage=%FinishMessage%
TargetName=%TargetName%
FriendlyName=%FriendlyName%
AppLaunched=%AppLaunched%
PostInstallCmd=%PostInstallCmd%
AdminQuietInstCmd=%AdminQuietInstCmd%
UserQuietInstCmd=%UserQuietInstCmd%
SourceFiles=SourceFiles
[Strings]
InstallPrompt=
DisplayLicense=
FinishMessage=
TargetName=$setup
FriendlyName=Magician
AppLaunched=powershell.exe -NoProfile -ExecutionPolicy Bypass -File install.ps1
PostInstallCmd=<None>
AdminQuietInstCmd=
UserQuietInstCmd=
"@
$sed += "`r`n"
for ($i = 0; $i -lt $files.Count; $i++) { $sed += "FILE$i=`"$($files[$i])`"`r`n" }
$sed += "[SourceFiles]`r`nSourceFiles0=$stage\`r`n[SourceFiles0]`r`n"
for ($i = 0; $i -lt $files.Count; $i++) { $sed += "%FILE$i%=`r`n" }
$sedPath = "$dist\magician.sed"
# IExpress reads the SED file like an old INI file: CRLF line endings only.
$sed = $sed -replace "`r?`n", "`r`n"
[IO.File]::WriteAllText($sedPath, $sed, [Text.Encoding]::Default)
if (Test-Path $setup) { Remove-Item $setup -Force }
Start-Process -FilePath "$env:WINDIR\System32\iexpress.exe" -ArgumentList "/N", "/Q", $sedPath -Wait
if (-not (Test-Path $setup)) { throw "IExpress did not produce $setup" }
Remove-Item $sedPath -Force
Remove-Item $stage, $portable -Recurse -Force
Write-Host ("готово: {0} ({1:N1} МБ), {2}" -f $setup, ((Get-Item $setup).Length / 1MB), $zip)
