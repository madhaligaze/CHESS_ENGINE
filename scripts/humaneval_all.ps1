# Human-likeness table from the README: for the held-out Tal and Carlsen games, how often
# each mode plays the move the player actually played. Results go to nets\humaneval\*.txt.
#   powershell -ExecutionPolicy Bypass -File scripts\humaneval_all.ps1 [-Parallel 2] [-Nodes 150000]

param([int]$Parallel = 2, [int]$Nodes = 150000)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root "target\release\magician.exe"
$out = Join-Path $root "nets\humaneval"
New-Item -ItemType Directory -Force $out | Out-Null

# Sample rates spread ~600 positions over all held-out games (the PGN files are chronological).
$players = @{ Tal = "0.06"; Carlsen = "0.018" }
$runs = @()
foreach ($player in "Tal", "Carlsen") {
    $runs += , @($player, "engine", "50")
    $runs += , @($player, "human", "50")
    foreach ($h in 0, 25, 50, 75, 100) { $runs += , @($player, $player.ToLower(), "$h") }
}

$procs = @()
foreach ($r in $runs) {
    $player, $style, $h = $r
    while (@($procs | Where-Object { -not $_.HasExited }).Count -ge $Parallel) { Start-Sleep -Milliseconds 500 }
    $name = "{0}_{1}_h{2}" -f $player.ToLower(), $style, $h
    $procs += Start-Process -FilePath $exe -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "$out\$name.txt" -RedirectStandardError "$out\$name.err" `
        -ArgumentList "humaneval", "--player", $player, "--style", $style, "--humanity", $h, "--sample", $players[$player], `
        "--holdout", "8", "--max", "600", "--nodes", "$Nodes", "--min-ply", "8", "data\raw\$player.pgn"
}
$procs | Wait-Process

foreach ($r in $runs) {
    $player, $style, $h = $r
    $t = Get-Content ("$out\{0}_{1}_h{2}.txt" -f $player.ToLower(), $style, $h)
    $match = ($t | Select-String "this mode matches") -replace ".*: ", ""
    $net = ($t | Select-String "network top-1") -replace "network top-1 matches human: ", ""
    $cost = ($t | Select-String "chosen differs") -replace "chosen differs from engine best in ", ""
    "{0,-8} {1,-8} humanity {2,3}: matches {3} | network {4} | differs {5}" -f $player, $style, $h, $match, $net, $cost
}
