# Runs after the base network has finished training: style fine-tunes, export, and the
# human-likeness benchmark on held-out games. Logs go to nets\*.log / nets\eval_*.txt.
#   powershell -ExecutionPolicy Bypass -File scripts\after_base.ps1 [-WaitPid <pid>]

param([int]$WaitPid = 0)

# Windows PowerShell 5.1 redirects (*>) to UTF-16 by default; keep the logs grep-able.
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root "training\.venv\Scripts\python.exe"
$exe = Join-Path $root "target\release\magician.exe"
# Train on CPU + DirectML GPU when torch-directml is installed (training\.venv-dml).
$trainPy = Join-Path $root "training\.venv-dml\Scripts\python.exe"
$gpuArgs = @("--gpu", "dml", "--gpu-frac", "0.5")
if (-not (Test-Path $trainPy)) { $trainPy = $py; $gpuArgs = @() }

if ($WaitPid -gt 0) {
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
}

Push-Location training
& $py export.py ../nets/base.pt ../nets/base.nn --name base --elo 2750,2700 *> ../nets/export.log
& $trainPy -u train.py --data ../data/bin/tal.bin:3 ../data/bin/elite.bin:1 --holdout 8 --init ../nets/base.pt `
    --out ../nets/tal.pt --steps 1200 --batch 256 --lr 3e-4 --warmup 50 --val-every 100 --val-batches 40 --keep-best `
    --elo-override 2700,2650 @gpuArgs *> ../nets/tal.log
& $py export.py ../nets/tal.pt ../nets/tal.nn --name tal *>> ../nets/export.log
& $trainPy -u train.py --data ../data/bin/carlsen.bin:3 ../data/bin/elite.bin:1 --holdout 8 --init ../nets/base.pt `
    --out ../nets/carlsen.pt --steps 2500 --batch 256 --lr 3e-4 --warmup 50 --val-every 250 --val-batches 40 --keep-best `
    --elo-override 2700,2650 @gpuArgs *> ../nets/carlsen.log
& $py export.py ../nets/carlsen.pt ../nets/carlsen.nn --name carlsen *>> ../nets/export.log
Pop-Location

# Human-likeness: how often does each mode play the move the human actually played?
$common = @("--holdout", "8", "--max", "600", "--nodes", "150000", "--min-ply", "8")
# Sample rates spread ~600 positions over all held-out games (the files are chronological).
foreach ($cfg in @(
        @("tal", "Tal", "engine", "0.06"), @("tal", "Tal", "human", "0.06"), @("tal", "Tal", "tal", "0.06"),
        @("carlsen", "Carlsen", "engine", "0.018"), @("carlsen", "Carlsen", "human", "0.018"), @("carlsen", "Carlsen", "carlsen", "0.018"))) {
    $file, $player, $style, $sample = $cfg
    & $exe humaneval --player $player --style $style --sample $sample @common "data\raw\$player.pgn" *> "nets\eval_${file}_$style.txt"
}
Write-Host "after_base done"
