//! Turns a search into a move the way a strong human would choose it.
//!
//! 1. The network predicts which moves a strong human (or Tal, or Carlsen) would consider.
//! 2. The alpha-beta search finds the engine's own best move; then the human candidates
//!    that could still win the choice get exact scores.
//! 3. The move is chosen by KL-regularised selection (the "piKL" rule from human-AI
//!    research): maximise  expected_score(eval) + beta * ln(human probability).
//!    Among moves that are about equally good the human one wins; a clearly better engine
//!    move still wins; anything losing more than `max_loss` centipawns is never played.
//! 4. Thinking time follows the position like a human's does: obvious moves (a forced
//!    recapture, a move every strong player would play) are made quickly, complex
//!    positions get more time.

use crate::board::Board;
use crate::eval::Style;
use crate::nn::{NetEval, Network};
use crate::search::{uci_score, Limits, RootMove, SearchParams, SearchPool, SearchResult, MATE_BOUND};
use crate::types::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StyleKind {
    /// Strong human player in general (base network).
    Human,
    Tal,
    Carlsen,
    /// Pure engine: no network, maximum strength, no human-like choices.
    Engine,
}

impl StyleKind {
    pub fn parse(s: &str) -> Option<StyleKind> {
        match s.to_ascii_lowercase().as_str() {
            "human" => Some(StyleKind::Human),
            "tal" => Some(StyleKind::Tal),
            "carlsen" => Some(StyleKind::Carlsen),
            "engine" => Some(StyleKind::Engine),
            _ => None,
        }
    }

    pub fn eval_style(self) -> Style {
        match self {
            StyleKind::Human => Style::HUMAN,
            StyleKind::Tal => Style::TAL,
            StyleKind::Carlsen => Style::CARLSEN,
            StyleKind::Engine => Style::NEUTRAL,
        }
    }

    pub fn net_file(self) -> &'static str {
        match self {
            StyleKind::Tal => "tal.nn",
            StyleKind::Carlsen => "carlsen.nn",
            _ => "base.nn",
        }
    }

    /// (humanity, max_loss) multipliers in percent: Tal trusts his intuition more and
    /// accepts objectively dubious moves (sacrifices); Carlsen trusts it but stays precise.
    fn choice_scale(self) -> (u32, i32) {
        match self {
            StyleKind::Tal => (130, 150),
            StyleKind::Carlsen => (110, 100),
            _ => (100, 100),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub hash_mb: usize,
    pub threads: usize,
    pub style: StyleKind,
    /// 0..100: how strongly the human prior pulls the choice away from the engine's best.
    pub humanity: u32,
    /// 0..100: randomness among the acceptable moves (0 = deterministic).
    pub variety: u32,
    /// Never play a move whose score is more than this many centipawns below the best.
    pub max_loss: i32,
    pub human_timing: bool,
    pub move_overhead: u64,
    pub net_dir: Option<PathBuf>,
    /// No UCI info output (batch evaluation).
    pub quiet: bool,
    /// Evaluation personality to use instead of the style's own (experiments).
    pub eval_style: Option<StyleKind>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            hash_mb: 64,
            threads: 1,
            style: StyleKind::Human,
            humanity: 25,
            variety: 20,
            max_loss: 60,
            human_timing: true,
            move_overhead: 30,
            net_dir: None,
            quiet: false,
            eval_style: None,
        }
    }
}

/// The game so far: current position, the hashes of every position (for repetitions) and
/// the moves played (the network looks at the last two).
#[derive(Clone)]
pub struct Game {
    pub board: Board,
    pub hashes: Vec<u64>,
    pub moves: Vec<Move>,
}

impl Game {
    pub fn new(board: Board) -> Game {
        Game { hashes: vec![board.hash], board, moves: Vec::new() }
    }

    pub fn play(&mut self, m: Move) {
        self.board = self.board.make_move(m);
        self.hashes.push(self.board.hash);
        self.moves.push(m);
    }

    /// (opponent's last move, our previous move)
    pub fn last_moves(&self) -> (Move, Move) {
        let n = self.moves.len();
        let last = if n >= 1 { self.moves[n - 1] } else { Move::NONE };
        let prev = if n >= 2 { self.moves[n - 2] } else { Move::NONE };
        (last, prev)
    }
}

/// Networks found on disk, by file name.
#[derive(Default, Clone)]
pub struct Nets {
    pub dir: Option<PathBuf>,
    pub base: Option<Arc<Network>>,
    pub tal: Option<Arc<Network>>,
    pub carlsen: Option<Arc<Network>>,
}

impl Nets {
    /// nets/ next to the executable, two levels up (cargo target dir) or in the working
    /// directory: the first one that contains base.nn.
    pub fn find_dir() -> Option<PathBuf> {
        let mut candidates = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            if let Some(d) = exe.parent() {
                candidates.push(d.join("nets"));
                candidates.push(d.join("../../nets"));
            }
        }
        candidates.push(PathBuf::from("nets"));
        candidates.into_iter().find(|d| d.join("base.nn").exists()).map(|d| normalize(&d))
    }

    /// Loads the networks from `dir`, or from `find_dir()` when none is given.
    pub fn load(dir: Option<&Path>) -> Nets {
        let found = match dir {
            Some(d) => Some(d.to_path_buf()).filter(|d| d.join("base.nn").exists()),
            None => Self::find_dir(),
        };
        let Some(dir) = found else {
            return Nets::default();
        };
        let load = |name: &str| match Network::load(&dir.join(name)) {
            Ok(n) => Some(Arc::new(n)),
            Err(e) => {
                if dir.join(name).exists() {
                    println!("info string cannot load {name}: {e}");
                }
                None
            }
        };
        Nets { base: load("base.nn"), tal: load("tal.nn"), carlsen: load("carlsen.nn"), dir: Some(dir) }
    }

    pub fn for_style(&self, style: StyleKind) -> Option<Arc<Network>> {
        match style {
            StyleKind::Engine => None,
            StyleKind::Tal => self.tal.clone().or_else(|| self.base.clone()),
            StyleKind::Carlsen => self.carlsen.clone().or_else(|| self.base.clone()),
            StyleKind::Human => self.base.clone(),
        }
    }
}

/// Resolves "." and ".." lexically (canonicalize would produce \\?\ paths on Windows).
fn normalize(p: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Where the engine keeps its own files (games/, learn.txt): $MAGICIAN_HOME, else the
/// `home=` line of magician.ini next to the executable (written by the installer), else the
/// folder that holds nets/ (the project or the installation folder), else the exe's folder.
pub fn data_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("MAGICIAN_HOME") {
        return PathBuf::from(home);
    }
    let ini = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join("magician.ini")));
    if let Some(text) = ini.and_then(|p| std::fs::read_to_string(p).ok()) {
        if let Some(home) = text.lines().find_map(|l| l.trim().trim_start_matches('\u{feff}').strip_prefix("home=")) {
            let home = PathBuf::from(home.trim());
            if home.is_dir() {
                return home;
            }
        }
    }
    if let Some(root) = Nets::find_dir().and_then(|d| d.parent().map(Path::to_path_buf)) {
        return root;
    }
    std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Soft (stop starting iterations) and hard (abort) time limits in milliseconds.
pub fn time_budget(limits: &Limits, stm: usize, overhead: u64) -> (Option<u64>, Option<u64>) {
    if let Some(mt) = limits.movetime {
        let t = mt.saturating_sub(overhead).max(1);
        return (Some(t), Some(t));
    }
    if limits.infinite {
        return (None, None);
    }
    if let Some(time) = limits.time[stm] {
        let inc = limits.inc[stm].unwrap_or(0);
        let mtg = limits.movestogo.map_or(28, |m| m.clamp(1, 50)) as u64;
        let avail = time.saturating_sub(overhead).max(1);
        let soft = (avail / mtg + inc * 3 / 4).min(avail * 2 / 5).max(1);
        let hard = (soft * 3).min(avail * 3 / 5).max(soft);
        return (Some(soft), Some(hard));
    }
    (None, None)
}

/// How long a human would think here, relative to an average move.
fn complexity_factor(net: &NetEval, game: &Game) -> f64 {
    let Some(&(top, p)) = net.policy.first() else { return 1.0 };
    let mut f: f64 = match p {
        p if p >= 0.85 => 0.35,
        p if p >= 0.65 => 0.6,
        p if p >= 0.40 => 0.9,
        p if p >= 0.25 => 1.15,
        _ => 1.4,
    };
    // An expected recapture is played almost at once.
    let (last, _) = game.last_moves();
    if last.is_some() && last.is_capture() && top.is_capture() && top.to() == last.to() && p >= 0.5 {
        f = f.min(0.4);
    }
    // Familiar opening moves come from memory.
    if game.moves.len() < 16 && p >= 0.3 {
        f *= 0.5;
    }
    f
}

/// Human probabilities -> the set of moves worth scoring exactly.
pub fn candidate_moves(net: &NetEval) -> Vec<Move> {
    let mut out = Vec::new();
    let mut cum = 0.0;
    for &(m, p) in &net.policy {
        if out.len() >= 6 || (cum >= 0.92 && !out.is_empty()) || (p < 0.02 && !out.is_empty()) {
            break;
        }
        out.push(m);
        cum += p;
    }
    out
}

#[inline]
fn expected_score(cp: i32) -> f64 {
    1.0 / (1.0 + (-(cp as f64) / 170.0).exp())
}

fn next_rand(s: &mut u64) -> f64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

pub struct Decision {
    pub mv: Move,
    pub explanation: String,
}

/// Probabilities below this count as this: the network is not calibrated that far out.
const PRIOR_FLOOR: f32 = 0.002;

/// Weight of ln(human probability) against expected score in the utility.
fn humanity_beta(opts: &Options) -> f64 {
    opts.humanity as f64 / 100.0 * 0.05
}

/// Sampling temperature over utilities (0 = always the best utility).
fn variety_tau(opts: &Options) -> f64 {
    opts.variety as f64 / 100.0 * 0.02
}

/// Inverse of `expected_score`.
fn score_for_expected(es: f64) -> i32 {
    (-170.0 * (1.0 / es - 1.0).ln()).round() as i32
}

/// Utility of a root move: how good it is plus how human it is.
fn utility(r: &RootMove, beta: f64) -> f64 {
    expected_score(r.score) + beta * (r.prior.max(PRIOR_FLOOR) as f64).ln()
}

/// After the main search: exact scores for the human candidates that could still win the
/// choice. A candidate is only worth searching if humans prefer it to the engine's move, and
/// it only needs to be searched down to the score at which its utility would lose anyway.
fn verify_human_candidates(pool: &mut SearchPool, game: &Game, result: &mut SearchResult, net: &NetEval, opts: &Options) {
    let Some(&best) = result.root_moves.first() else { return };
    if result.depth < 2 || best.score.abs() >= MATE_BOUND || opts.humanity == 0 {
        return;
    }
    let (beta, tau) = (humanity_beta(opts), variety_tau(opts));
    let p_best = best.prior.max(PRIOR_FLOOR) as f64;
    let mut checks = Vec::new();
    for m in candidate_moves(net) {
        let p = net.prob(m).max(PRIOR_FLOOR) as f64;
        if m == best.mv || p <= p_best {
            continue;
        }
        // Lowest score at which the candidate's utility still reaches the best move's
        // (minus a little slack so sampling can pick near-equal moves).
        let need = expected_score(best.score) - beta * (p.ln() - p_best.ln()) - 2.0 * tau;
        let threshold = if need <= 0.0 { i32::MIN / 2 } else { score_for_expected(need.min(0.999_999)) };
        let floor = threshold.max(best.score - opts.max_loss) - 1;
        if floor < best.score {
            checks.push((m, floor));
        }
    }
    if checks.is_empty() {
        return;
    }
    let depth = (result.depth - 1).max(1);
    let scores = pool.verify_root_moves(&game.board, &game.hashes, &checks, best.score + 1, depth);
    for (&(m, _), s) in checks.iter().zip(scores) {
        if let Some(rm) = result.root_moves.iter_mut().find(|r| r.mv == m) {
            rm.candidate = true;
            rm.exact = s.is_some();
            if let Some(v) = s {
                rm.score = v;
                rm.depth = depth;
            }
        }
    }
}

/// KL-regularised choice among the root moves of a finished search.
pub fn choose(result: &SearchResult, opts: &Options, rng: &mut u64) -> Decision {
    let Some(&best) = result.root_moves.first() else {
        return Decision { mv: result.best, explanation: String::new() };
    };
    if result.depth == 0 || best.score.abs() >= MATE_BOUND || opts.humanity == 0 {
        return Decision { mv: best.mv, explanation: "engine move (forced, mate or humanity 0)".into() };
    }
    let beta = humanity_beta(opts);
    let floor = best.score - opts.max_loss;
    let options: Vec<(RootMove, f64)> = result
        .root_moves
        .iter()
        .filter(|r| r.mv == best.mv || (r.candidate && r.exact && r.score >= floor))
        .map(|r| (*r, utility(r, beta)))
        .collect();
    let max_u = options.iter().map(|o| o.1).fold(f64::MIN, f64::max);
    let chosen = if opts.variety == 0 {
        options.iter().find(|o| o.1 == max_u).unwrap().0
    } else {
        let tau = variety_tau(opts);
        let weights: Vec<f64> = options.iter().map(|o| ((o.1 - max_u) / tau).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mut x = next_rand(rng) * total;
        let mut pick = options[0].0;
        for (o, w) in options.iter().zip(&weights) {
            if x < *w {
                pick = o.0;
                break;
            }
            x -= w;
        }
        pick
    };
    let describe = |r: &RootMove| format!("{} (human {:.0}%, {})", r.mv, r.prior * 100.0, uci_score(r.score));
    let alternatives: Vec<String> = options.iter().filter(|o| o.0.mv != chosen.mv).map(|o| describe(&o.0)).collect();
    let explanation = if chosen.mv == best.mv {
        format!("played engine best {}; human alternatives considered: [{}]", describe(&chosen), alternatives.join(", "))
    } else {
        format!("played human move {} instead of engine best {}", describe(&chosen), describe(&best))
    };
    Decision { mv: chosen.mv, explanation }
}

/// What the engine decided and why.
pub struct Thought {
    pub mv: Move,
    pub result: SearchResult,
    /// Network move probabilities in this position (empty without a network).
    pub policy: Vec<(Move, f32)>,
    /// Why this move was chosen (empty for the pure engine).
    pub explanation: String,
}

impl Thought {
    /// Score of the move actually played (falls back to the best score).
    pub fn played_score(&self) -> i32 {
        self.result.root_moves.iter().find(|r| r.mv == self.mv && r.exact).map_or(self.result.score, |r| r.score)
    }

    pub fn played_prior(&self) -> f32 {
        self.policy.iter().find(|(m, _)| *m == self.mv).map_or(0.0, |(_, p)| *p)
    }
}

/// Full move decision for `game`: network, search, human choice. Prints UCI info lines
/// unless `opts.quiet`. Moves in `avoid` are not played if anything else is legal.
pub fn think(pool: &mut SearchPool, game: &Game, limits: &Limits, opts: &Options, net: Option<&Network>, seed: u64, avoid: &[Move]) -> Thought {
    let style = opts.style;
    pool.main.eval_params.style = opts.eval_style.unwrap_or(style).eval_style();
    let (h_pct, l_pct) = style.choice_scale();
    let opts = &Options {
        humanity: (opts.humanity * h_pct / 100).min(100),
        max_loss: opts.max_loss * l_pct / 100,
        ..opts.clone()
    };
    let (last, prev) = game.last_moves();
    let net_eval = net.map(|n| n.evaluate(&game.board, last, prev));

    let human = net_eval.is_some() && style != StyleKind::Engine && opts.humanity > 0;
    let (mut soft, hard) = time_budget(limits, game.board.stm, opts.move_overhead);
    let mut params = SearchParams {
        depth: limits.depth,
        nodes: limits.nodes,
        soft_ms: soft,
        hard_ms: hard,
        verbose: !opts.quiet,
        excluded: avoid.to_vec(),
        ..Default::default()
    };
    if let Some(ne) = &net_eval {
        params.priors = ne.policy.clone();
        if opts.human_timing && limits.movetime.is_none() {
            if let Some(s) = soft {
                let f = complexity_factor(ne, game);
                soft = Some(((s as f64 * f) as u64).max(1));
            }
        }
        // Leave a quarter of the time for checking the human candidates afterwards.
        params.soft_ms = if human { soft.map(|s| (s * 3 / 4).max(1)) } else { soft };
        let top: Vec<String> = ne.policy.iter().take(5).map(|(m, p)| format!("{m} {:.0}%", p * 100.0)).collect();
        if !opts.quiet {
            println!(
            "info string {} net: {} | human W/D/L {:.0}/{:.0}/{:.0}",
            net.map_or("", |n| n.name.as_str()),
            top.join(", "),
            ne.wdl[0] * 100.0,
            ne.wdl[1] * 100.0,
            ne.wdl[2] * 100.0
            );
        }
    }
    // A single legal move needs no thought.
    let legal = crate::movegen::legal_moves(&game.board);
    if legal.len() == 1 && (limits.time[game.board.stm].is_some() || limits.movetime.is_some()) {
        params.depth = Some(1);
    }

    let mut result = pool.search(&game.board, &game.hashes, &params);
    let (mv, explanation) = if let (true, Some(ne)) = (human, &net_eval) {
        verify_human_candidates(pool, game, &mut result, ne, opts);
        let mut rng = seed | 1;
        let d = choose(&result, opts, &mut rng);
        if !d.explanation.is_empty() && !opts.quiet {
            println!("info string {}", d.explanation);
        }
        (d.mv, d.explanation)
    } else {
        (result.best, String::new())
    };
    Thought { mv, result, policy: net_eval.map(|e| e.policy).unwrap_or_default(), explanation }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rm(mv: &str, score: i32, prior: f32) -> RootMove {
        let b = Board::startpos();
        let m = crate::san::parse_uci(&b, mv).unwrap();
        RootMove { mv: m, score, exact: true, depth: 10, prior, candidate: true }
    }

    fn result(moves: Vec<RootMove>) -> SearchResult {
        SearchResult { best: moves[0].mv, score: moves[0].score, depth: 10, seldepth: 10, nodes: 0, time_ms: 0, pv: vec![], root_moves: moves }
    }

    #[test]
    fn prefers_human_move_when_nearly_equal() {
        let r = result(vec![rm("g1h3", 30, 0.01), rm("e2e4", 20, 0.5), rm("d2d4", 18, 0.3)]);
        let o = Options { variety: 0, ..Default::default() };
        assert_eq!(choose(&r, &o, &mut 1).mv.to_uci(), "e2e4");
    }

    #[test]
    fn keeps_clearly_better_engine_move() {
        let r = result(vec![rm("g1h3", 120, 0.01), rm("e2e4", 10, 0.5)]);
        let o = Options { variety: 0, ..Default::default() };
        assert_eq!(choose(&r, &o, &mut 1).mv.to_uci(), "g1h3");
    }

    #[test]
    fn never_exceeds_max_loss() {
        let r = result(vec![rm("g1h3", 100, 0.001), rm("e2e4", 30, 0.99)]);
        let o = Options { variety: 0, humanity: 100, max_loss: 60, ..Default::default() };
        assert_eq!(choose(&r, &o, &mut 1).mv.to_uci(), "g1h3");
    }
}
