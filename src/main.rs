use magician::board::Board;
use magician::datagen::{self, DatagenConfig};
use magician::perft::{divide, perft, PERFT_SUITE};
use magician::search::{SearchParams, Searcher, SEARCH_STACK};
use magician::uci::Uci;
use std::time::Instant;

const USAGE: &str = "usage:
  magician                              UCI mode (default)
  magician bench [depth]                fixed-depth search of standard positions
  magician perft <depth> [fen]
  magician perftsuite [maxdepth]
  magician datagen --out FILE [--style N] [--min-elo N] [--player NAME] [--sample P]
                   [--opening-sample P] [--min-tc N] [--min-clock SEC] [--max-games N]
                   [--seed N] INPUT.pgn...
  magician humaneval [--player NAME] [--holdout N] [--sample P] [--max N] [--min-ply N]
                   [--nodes N | --depth N | --movetime MS] [--style human|tal|carlsen|engine]
                   [--humanity N] [--variety N] [--max-loss CP] [--net-dir DIR]
                   [--eval-style human|tal|carlsen|engine] FILE.pgn";

fn run_humaneval(args: &[String]) {
    use magician::engine::{Options, StyleKind};
    use magician::humaneval::{run, HumanEvalConfig};
    use magician::search::Limits;
    let mut cfg = HumanEvalConfig {
        input: String::new(),
        player: None,
        holdout: 8,
        sample: 0.2,
        max_positions: 500,
        min_ply: 0,
        limits: Limits { nodes: Some(100_000), ..Default::default() },
        opts: Options { variety: 0, quiet: true, hash_mb: 32, ..Default::default() },
        seed: 99,
    };
    let mut i = 0;
    let value = |i: &mut usize| -> String {
        *i += 1;
        args.get(*i).cloned().unwrap_or_else(|| panic!("missing value for {}", args[*i - 1]))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--player" => cfg.player = Some(value(&mut i)),
            "--holdout" => cfg.holdout = value(&mut i).parse().expect("--holdout"),
            "--sample" => cfg.sample = value(&mut i).parse().expect("--sample"),
            "--max" => cfg.max_positions = value(&mut i).parse().expect("--max"),
            "--min-ply" => cfg.min_ply = value(&mut i).parse().expect("--min-ply"),
            "--nodes" => cfg.limits = Limits { nodes: Some(value(&mut i).parse().expect("--nodes")), ..Default::default() },
            "--depth" => cfg.limits = Limits { depth: Some(value(&mut i).parse().expect("--depth")), ..Default::default() },
            "--movetime" => {
                cfg.limits = Limits { movetime: Some(value(&mut i).parse().expect("--movetime")), ..Default::default() }
            }
            "--style" => cfg.opts.style = StyleKind::parse(&value(&mut i)).expect("--style"),
            "--eval-style" => cfg.opts.eval_style = Some(StyleKind::parse(&value(&mut i)).expect("--eval-style")),
            "--humanity" => cfg.opts.humanity = value(&mut i).parse().expect("--humanity"),
            "--variety" => cfg.opts.variety = value(&mut i).parse().expect("--variety"),
            "--max-loss" => cfg.opts.max_loss = value(&mut i).parse().expect("--max-loss"),
            "--net-dir" => cfg.opts.net_dir = Some(value(&mut i).into()),
            "--seed" => cfg.seed = value(&mut i).parse().expect("--seed"),
            s if s.starts_with("--") => panic!("unknown option {s}\n{USAGE}"),
            s => cfg.input = s.to_string(),
        }
        i += 1;
    }
    if cfg.input.is_empty() {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    if let Err(e) = run(&cfg) {
        eprintln!("humaneval failed: {e}");
        std::process::exit(1);
    }
}

fn main() {
    magician::bitboard::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("uci") => {
            let mut uci = Uci::new();
            if args.first().is_some() {
                uci.command("uci");
            }
            uci.run();
        }
        Some("bench") => {
            let depth = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(11);
            std::thread::Builder::new().stack_size(SEARCH_STACK).spawn(move || bench(depth)).unwrap().join().unwrap();
        }
        Some("perft") => {
            let depth = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
            run_perft(depth, args.get(2).cloned());
        }
        Some("perftsuite") => run_perft_suite(args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5)),
        Some("datagen") => run_datagen(&args[1..]),
        Some("humaneval") => {
            let rest = args[1..].to_vec();
            std::thread::Builder::new()
                .stack_size(SEARCH_STACK)
                .spawn(move || run_humaneval(&rest))
                .unwrap()
                .join()
                .unwrap();
        }
        Some("help" | "-h" | "--help") => println!("{USAGE}"),
        _ => eprintln!("{USAGE}"),
    }
}

const BENCH_FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2QKB1R w KQ - 0 8",
    "2r3k1/pp3ppp/4p3/3pP3/3P4/P4N2/1P3PPP/2R3K1 w - - 0 25",
    "r1b2rk1/2q1bppp/p2p1n2/np2p3/3PP3/5N1P/PPBN1PP1/R1BQR1K1 w - - 0 13",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "6k1/5pp1/7p/3n4/8/2B3P1/5PKP/8 w - - 0 40",
];

fn bench(depth: i32) {
    let mut s = Searcher::standalone(32);
    let t = Instant::now();
    let mut nodes = 0;
    for fen in BENCH_FENS {
        let b = Board::from_fen(fen).unwrap();
        s.clear();
        let r = s.search(&b, &[b.hash], &SearchParams { depth: Some(depth), candidate_margin: 150, ..Default::default() });
        println!("{:>10} nodes  d{} {:>8} {:<6} {fen}", r.nodes, r.depth, magician::search::uci_score(r.score), r.best.to_uci());
        nodes += r.nodes;
    }
    let ms = t.elapsed().as_millis().max(1) as u64;
    println!("\n{nodes} nodes {} nps ({ms} ms)", nodes * 1000 / ms);
}

fn run_perft(depth: u32, fen: Option<String>) {
    let b = match fen {
        Some(f) => Board::from_fen(&f).expect("bad FEN"),
        None => Board::startpos(),
    };
    let t = Instant::now();
    let mut total = 0;
    for (m, n) in divide(&b, depth) {
        println!("{m}: {n}");
        total += n;
    }
    let secs = t.elapsed().as_secs_f64();
    println!("\nNodes: {total}\nTime: {secs:.3}s\nNPS: {:.0}", total as f64 / secs.max(1e-9));
}

fn run_perft_suite(max_depth: usize) {
    for (fen, counts) in PERFT_SUITE {
        let b = Board::from_fen(fen).unwrap();
        for (i, &expected) in counts.iter().enumerate().take(max_depth) {
            let t = Instant::now();
            let n = perft(&b, i as u32 + 1);
            let ok = if n == expected { "ok" } else { "MISMATCH" };
            println!(
                "{ok:8} d{} {n:>12} (expected {expected:>12}) {:>7.2}s  {fen}",
                i + 1,
                t.elapsed().as_secs_f64()
            );
        }
    }
}

fn run_datagen(args: &[String]) {
    let mut cfg = DatagenConfig::default();
    let mut i = 0;
    let value = |i: &mut usize| -> String {
        *i += 1;
        args.get(*i).cloned().unwrap_or_else(|| panic!("missing value for {}", args[*i - 1]))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--out" => cfg.output = value(&mut i),
            "--style" => cfg.style = value(&mut i).parse().expect("--style"),
            "--min-elo" => cfg.min_elo = value(&mut i).parse().expect("--min-elo"),
            "--player" => cfg.player = Some(value(&mut i)),
            "--sample" => cfg.sample = value(&mut i).parse().expect("--sample"),
            "--opening-sample" => cfg.opening_sample = value(&mut i).parse().expect("--opening-sample"),
            "--min-tc" => cfg.min_time_class = value(&mut i).parse().expect("--min-tc"),
            "--min-clock" => cfg.min_clock = value(&mut i).parse().expect("--min-clock"),
            "--max-games" => cfg.max_games = value(&mut i).parse().expect("--max-games"),
            "--seed" => cfg.seed = value(&mut i).parse().expect("--seed"),
            s if s.starts_with("--") => panic!("unknown option {s}\n{USAGE}"),
            s => cfg.inputs.push(s.to_string()),
        }
        i += 1;
    }
    if cfg.output.is_empty() || cfg.inputs.is_empty() {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    let t = Instant::now();
    match datagen::run(&cfg) {
        Ok(s) => println!(
            "games read {}, used {}, filtered {}, parse errors {}, positions {} ({:.1}s)",
            s.games_read,
            s.games_used,
            s.games_filtered,
            s.parse_errors,
            s.positions,
            t.elapsed().as_secs_f64()
        ),
        Err(e) => {
            eprintln!("datagen failed: {e}");
            std::process::exit(1);
        }
    }
}
