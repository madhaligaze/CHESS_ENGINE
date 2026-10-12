# Full pipeline: download games, build the engine, generate training data, train the base
# human-move network, fine-tune the Tal and Carlsen styles and export everything to nets/.
#
#   powershell -ExecutionPolicy Bypass -File scripts\train_all.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\train_all.ps1 -BaseSteps 30000 -Months 2025-07,2025-08,2025-09,2025-10
#
# On a laptop CPU the base training (10000 steps) takes about 3-4 hours. -Gpu adds any
# DirectX 12 GPU, even an integrated Intel one, through DirectML: each batch is split between
# CPU and GPU (on an i5-8350U + UHD 620: 389 -> 570 positions/s). With an NVIDIA GPU install
# the CUDA build of torch into training\.venv instead; train.py then uses it automatically and
# you can afford a bigger network: -Channels 128 -Blocks 10 -BaseSteps 60000.

param(
    [int]$BaseSteps = 10000,
    [int]$Channels = 64,
    [int]$Blocks = 6,
    [string[]]$Months = @("2025-09", "2025-10"),
    [switch]$SkipDownload,
    [switch]$Gpu
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root "training\.venv\Scripts\python.exe"
$exe = Join-Path $root "target\release\magician.exe"
New-Item -ItemType Directory -Force data\raw, data\bin, nets | Out-Null

if (-not (Test-Path $py)) {
    python -m venv training\.venv
    & $py -m pip install --upgrade pip
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    & $py -m pip install numpy chess tqdm
}
$trainPy = $py
$gpuArgs = @()
if ($Gpu) {
    # torch-directml pins its own torch version, so it lives in a separate environment.
    $trainPy = Join-Path $root "training\.venv-dml\Scripts\python.exe"
    if (-not (Test-Path $trainPy)) {
        python -m venv training\.venv-dml
        & $trainPy -m pip install --upgrade pip
        & $trainPy -m pip install torch-directml numpy
    }
    $gpuArgs = @("--gpu", "dml", "--gpu-frac", "0.5")
}

if (-not $SkipDownload) {
    foreach ($p in "Tal", "Carlsen") {
        if (-not (Test-Path "data\raw\$p.pgn")) {
            curl.exe -sSL -o "data\raw\$p.zip" "https://www.pgnmentor.com/players/$p.zip"
            tar -xf "data\raw\$p.zip" -C data\raw
        }
    }
    foreach ($m in $Months) {
        if (-not (Test-Path "data\raw\lichess_elite_$m.pgn")) {
            curl.exe -sSL -o "data\raw\elite_$m.zip" "https://database.nikonoel.fr/lichess_elite_$m.zip"
            tar -xf "data\raw\elite_$m.zip" -C data\raw
        }
    }
}

cargo build --release
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }

$elite = $Months | ForEach-Object { "data\raw\lichess_elite_$_.pgn" }
& $exe datagen --out data\bin\elite.bin --style 0 --min-elo 2200 --min-tc 2 --sample 0.15 --opening-sample 0.05 --seed 7 @elite
& $exe datagen --out data\bin\tal.bin --style 1 --player Tal --seed 11 data\raw\Tal.pgn
& $exe datagen --out data\bin\carlsen.bin --style 2 --player Carlsen --seed 12 data\raw\Carlsen.pgn

Push-Location training
# Base model: strong human players in general.
& $trainPy -u train.py --data ../data/bin/elite.bin --val-data ../data/bin/tal.bin ../data/bin/carlsen.bin `
    --out ../nets/base.pt --steps $BaseSteps --batch 512 --lr 1.5e-3 --warmup 300 --val-every 500 `
    --channels $Channels --blocks $Blocks @gpuArgs
# Styles: fine-tune on the player's own moves (with a quarter of general games mixed in).
& $trainPy -u train.py --data ../data/bin/tal.bin:3 ../data/bin/elite.bin:1 --holdout 8 --init ../nets/base.pt `
    --out ../nets/tal.pt --steps 1200 --batch 256 --lr 3e-4 --warmup 50 --val-every 100 --val-batches 40 --keep-best --elo-override 2700,2650 @gpuArgs
& $trainPy -u train.py --data ../data/bin/carlsen.bin:3 ../data/bin/elite.bin:1 --holdout 8 --init ../nets/base.pt `
    --out ../nets/carlsen.pt --steps 2500 --batch 256 --lr 3e-4 --warmup 50 --val-every 250 --val-batches 40 --keep-best --elo-override 2700,2650 @gpuArgs
& $py export.py ../nets/base.pt ../nets/base.nn --name base --elo 2750,2700
& $py export.py ../nets/tal.pt ../nets/tal.nn --name tal
& $py export.py ../nets/carlsen.pt ../nets/carlsen.nn --name carlsen
Pop-Location

Write-Host "done: nets\base.nn, nets\tal.nn, nets\carlsen.nn"
