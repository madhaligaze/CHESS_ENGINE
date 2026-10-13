# Learn from the games Magician has played (games\*.pgn):
#  1. analyse the new games: errors of both sides, the engine's mistakes go to learn.txt
#     (the engine stops repeating them at once) and training records to data\learned\,
#  2. once enough records have been collected, fine-tune the base network on them, mixed with
#     the original human games so it stays human-like, and export a new nets\base.nn
#     (the previous one is kept as nets\base_prev.nn).
#   powershell -ExecutionPolicy Bypass -File scripts\learn_from_games.ps1 [-MinRecords 300] [-Steps 400]

param([int]$MinRecords = 300, [int]$Steps = 400, [switch]$Gpu)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root "target\release\magician.exe"
if (-not (Test-Path $exe)) { $exe = Join-Path $root "app\magician.exe" }

& $exe analyze
$learned = Join-Path $root "data\learned"
$records = 0
foreach ($f in "engine_errors.bin", "opponent_moves.bin") {
    $p = Join-Path $learned $f
    if (Test-Path $p) { $records += (Get-Item $p).Length / 256 }
}
Write-Host "records collected: $records (need $MinRecords for fine-tuning)"
if ($records -lt $MinRecords) {
    Write-Host "not enough yet: the memory of mistakes (learn.txt) is already in use; play more games"
    exit 0
}

$py = Join-Path $root "training\.venv\Scripts\python.exe"
$gpuArgs = @()
if ($Gpu -and (Test-Path "$root\training\.venv-dml\Scripts\python.exe")) {
    $py = "$root\training\.venv-dml\Scripts\python.exe"
    $gpuArgs = @("--gpu", "dml")
}
$data = @("../data/bin/elite.bin:4")
foreach ($f in "engine_errors.bin", "opponent_moves.bin") {
    if (Test-Path (Join-Path $learned $f)) { $data += "../data/learned/${f}:1" }
}
Push-Location training
& $py -u train.py --data @data --holdout 64 --init ../nets/base.pt --out ../nets/base_ft.pt `
    --steps $Steps --batch 256 --lr 1e-4 --warmup 20 --val-every 100 --keep-best @gpuArgs
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "fine-tuning failed" }
Copy-Item ../nets/base.nn ../nets/base_prev.nn -Force
& $py export.py ../nets/base_ft.pt ../nets/base.nn --name base --elo 2750,2700
Pop-Location
Write-Host "new nets\base.nn is ready (previous one: nets\base_prev.nn). Restart Magician to use it."
