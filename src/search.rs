//! Alpha-beta search: iterative deepening with aspiration windows, principal variation
//! search, transposition table, null-move pruning, reverse futility pruning, razoring,
//! late move reductions/pruning, SEE pruning, killer/counter/history move ordering and a
//! quiescence search.
//!
//! After the search, `verify_root_moves` gives exact scores to a few chosen root moves (the
//! ones a human would consider, from the neural network), so the human-like move choice
//! compares honest numbers without slowing down the main search.

use crate::board::Board;
use crate::eval::{evaluate, EvalParams};
use crate::movegen::{generate_legal, generate_noisy, MoveList, MAX_MOVES};
use crate::tt::*;
use crate::types::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, LazyLock};
use std::time::Instant;

/// Stack size for search threads (the search is recursive and keeps move lists on the stack).
pub const SEARCH_STACK: usize = 64 * 1024 * 1024;

pub const INF: i32 = 32000;
pub const MATE: i32 = 31000;
pub const MATE_BOUND: i32 = MATE - 1000;
pub const MAX_PLY: usize = 128;
const EVAL_NONE: i32 = -32001;
const OFF: usize = 4;

const TT_SCORE: i32 = 4_000_000;
const GOOD_NOISY: i32 = 2_000_000;
const KILLER_1: i32 = 1_000_000;
const KILLER_2: i32 = 999_000;
const COUNTER: i32 = 998_000;
const BAD_NOISY: i32 = -2_000_000;

static LMR: LazyLock<[[i32; 64]; 64]> = LazyLock::new(|| {
    let mut t = [[0; 64]; 64];
    for (d, row) in t.iter_mut().enumerate().skip(1) {
        for (m, r) in row.iter_mut().enumerate().skip(1) {
            *r = (0.8 + (d as f64).ln() * (m as f64).ln() / 2.3) as i32;
        }
    }
    t
});

/// What `go` asked for.
#[derive(Clone, Debug, Default)]
pub struct Limits {
    pub depth: Option<i32>,
    pub nodes: Option<u64>,
    pub movetime: Option<u64>,
    /// Remaining clock time per color, in milliseconds.
    pub time: [Option<u64>; 2],
    pub inc: [Option<u64>; 2],
    pub movestogo: Option<u32>,
    pub infinite: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct RootMove {
    pub mv: Move,
    pub score: i32,
    /// The score is exact (inside the search window), not just a bound.
    pub exact: bool,
    pub depth: i32,
    /// Probability that a strong human plays this move (from the network), if known.
    pub prior: f32,
    pub candidate: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SearchParams {
    pub depth: Option<i32>,
    pub nodes: Option<u64>,
    /// Stop starting new iterations after this many milliseconds.
    pub soft_ms: Option<u64>,
    /// Abort the search after this many milliseconds.
    pub hard_ms: Option<u64>,
    /// Network move probabilities for the root moves (used for move ordering and reporting).
    pub priors: Vec<(Move, f32)>,
    /// Root moves not to play (known mistakes), unless nothing else is legal.
    pub excluded: Vec<Move>,
    pub verbose: bool,
    /// First iteration depth (helper threads start at different depths).
    pub start_depth: i32,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub best: Move,
    pub score: i32,
    pub depth: i32,
    pub seldepth: usize,
    pub nodes: u64,
    pub time_ms: u64,
    pub pv: Vec<Move>,
    /// Root moves with scores from the last completed iteration, best first.
    pub root_moves: Vec<RootMove>,
}

#[derive(Clone, Copy)]
struct StackEntry {
    eval: i32,
    mv: Move,
    /// Coloured piece (color * 6 + type) that made `mv`; 12 for none / null move.
    piece: usize,
}

pub struct Searcher {
    pub tt: Arc<TranspositionTable>,
    pub eval_params: EvalParams,
    pub stop: Arc<AtomicBool>,
    /// Node count shared by all threads of a search (for reporting).
    pub shared_nodes: Arc<AtomicU64>,
    history: Vec<i32>,
    cont: Vec<i32>,
    counter: Vec<Move>,
    killers: [[Move; 2]; MAX_PLY + 2],
    stack: [StackEntry; MAX_PLY + OFF + 2],
    pv: Box<[[Move; MAX_PLY + 1]; MAX_PLY + 1]>,
    pv_len: [usize; MAX_PLY + 2],
    hashes: Vec<u64>,
    root_moves: Vec<RootMove>,
    pub nodes: u64,
    seldepth: usize,
    start: Instant,
    hard_ms: Option<u64>,
    node_limit: u64,
    stopped: bool,
}

#[inline]
pub fn score_to_tt(s: i32, ply: usize) -> i32 {
    if s >= MATE_BOUND {
        s + ply as i32
    } else if s <= -MATE_BOUND {
        s - ply as i32
    } else {
        s
    }
}

#[inline]
pub fn score_from_tt(s: i32, ply: usize) -> i32 {
    if s >= MATE_BOUND {
        s - ply as i32
    } else if s <= -MATE_BOUND {
        s + ply as i32
    } else {
        s
    }
}

/// UCI score string: "cp 23" or "mate -3".
pub fn uci_score(s: i32) -> String {
    if s >= MATE_BOUND {
        format!("mate {}", (MATE - s + 1) / 2)
    } else if s <= -MATE_BOUND {
        format!("mate {}", -(MATE + s) / 2)
    } else {
        format!("cp {s}")
    }
}

#[inline]
fn hist_update(h: &mut i32, bonus: i32) {
    *h += bonus - *h * bonus.abs() / 16384;
}

impl Searcher {
    /// A single-threaded searcher with its own transposition table.
    pub fn standalone(hash_mb: usize) -> Searcher {
        Searcher::new(Arc::new(TranspositionTable::new(hash_mb)), Arc::new(AtomicBool::new(false)))
    }

    pub fn new(tt: Arc<TranspositionTable>, stop: Arc<AtomicBool>) -> Searcher {
        Searcher {
            tt,
            eval_params: EvalParams::default(),
            stop,
            shared_nodes: Arc::new(AtomicU64::new(0)),
            history: vec![0; 2 * 64 * 64],
            cont: vec![0; 12 * 64 * 12 * 64],
            counter: vec![Move::NONE; 12 * 64],
            killers: [[Move::NONE; 2]; MAX_PLY + 2],
            stack: [StackEntry { eval: EVAL_NONE, mv: Move::NONE, piece: 12 }; MAX_PLY + OFF + 2],
            pv: Box::new([[Move::NONE; MAX_PLY + 1]; MAX_PLY + 1]),
            pv_len: [0; MAX_PLY + 2],
            hashes: Vec::with_capacity(1024),
            root_moves: Vec::new(),
            nodes: 0,
            seldepth: 0,
            start: Instant::now(),
            hard_ms: None,
            node_limit: 0,
            stopped: false,
        }
    }

    /// Forget everything learned (new game), including the transposition table.
    pub fn clear(&mut self) {
        self.tt.clear();
        self.clear_tables();
    }

    /// Forget move-ordering statistics only.
    pub fn clear_tables(&mut self) {
        self.history.fill(0);
        self.cont.fill(0);
        self.counter.fill(Move::NONE);
    }

    #[inline]
    fn count_node(&mut self) {
        self.nodes += 1;
        if self.nodes & 1023 == 0 {
            self.shared_nodes.fetch_add(1024, Relaxed);
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    #[inline]
    fn eval(&self, b: &Board) -> i32 {
        evaluate(b, &self.eval_params)
    }

    #[inline]
    fn draw_score(&self, ply: usize) -> i32 {
        // Positive contempt: the engine (side to move at even plies) dislikes draws.
        let c = self.eval_params.style.contempt;
        if ply % 2 == 0 {
            -c
        } else {
            c
        }
    }

    #[inline]
    fn check_stop(&mut self) -> bool {
        if !self.stopped && self.nodes & 1023 == 0 {
            if self.stop.load(Relaxed)
                || self.hard_ms.is_some_and(|h| self.elapsed_ms() >= h)
                || (self.node_limit > 0 && self.nodes >= self.node_limit)
            {
                self.stopped = true;
            }
        }
        self.stopped
    }

    fn is_repetition(&self, b: &Board) -> bool {
        let n = self.hashes.len();
        let limit = (b.halfmove as usize).min(n.saturating_sub(1));
        let mut i = 4;
        while i <= limit {
            if self.hashes[n - 1 - i] == b.hash {
                return true;
            }
            i += 2;
        }
        false
    }

    #[inline]
    fn cont_index(prev: StackEntry, piece: usize, to: usize) -> Option<usize> {
        if prev.piece >= 12 || prev.mv.is_none() {
            None
        } else {
            Some(((prev.piece * 64 + prev.mv.to()) * 12 + piece) * 64 + to)
        }
    }

    fn quiet_score(&self, b: &Board, m: Move, ply: usize) -> i32 {
        let piece = b.stm * 6 + b.piece_on(m.from());
        let mut s = self.history[(b.stm * 64 + m.from()) * 64 + m.to()];
        if let Some(i) = Self::cont_index(self.stack[ply + OFF - 1], piece, m.to()) {
            s += self.cont[i];
        }
        s
    }

    fn score_moves(&self, b: &Board, moves: &MoveList, tt_move: Move, ply: usize, out: &mut [i32; MAX_MOVES]) {
        let prev = self.stack[ply + OFF - 1];
        let counter = if prev.piece < 12 && prev.mv.is_some() {
            self.counter[prev.piece * 64 + prev.mv.to()]
        } else {
            Move::NONE
        };
        for (i, &m) in moves.as_slice().iter().enumerate() {
            out[i] = if m == tt_move {
                TT_SCORE
            } else if m.is_noisy() {
                let victim = if m.is_ep() { PAWN } else { b.piece_on(m.to()) };
                let victim_value = if victim == NO_PIECE { 0 } else { [1, 3, 3, 5, 9, 0][victim] };
                let promo = if m.is_promo() { 8 } else { 0 };
                let base = (victim_value + promo) * 16 - b.piece_on(m.from()) as i32;
                if b.see_ge(m, 0) {
                    GOOD_NOISY + base
                } else {
                    BAD_NOISY + base
                }
            } else if m == self.killers[ply][0] {
                KILLER_1
            } else if m == self.killers[ply][1] {
                KILLER_2
            } else if m == counter {
                COUNTER
            } else {
                self.quiet_score(b, m, ply)
            };
        }
    }

    #[inline]
    fn pick(moves: &mut MoveList, scores: &mut [i32; MAX_MOVES], i: usize) -> (Move, i32) {
        let n = moves.len();
        let mut best = i;
        for j in i + 1..n {
            if scores[j] > scores[best] {
                best = j;
            }
        }
        moves.as_mut_slice().swap(i, best);
        scores.swap(i, best);
        (moves.as_slice()[i], scores[i])
    }

    fn update_quiet_stats(&mut self, b: &Board, m: Move, ply: usize, depth: i32, others: &[Move]) {
        let bonus = (16 * depth * depth + 32 * depth).min(1500);
        if self.killers[ply][0] != m {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = m;
        }
        let prev = self.stack[ply + OFF - 1];
        if prev.piece < 12 && prev.mv.is_some() {
            self.counter[prev.piece * 64 + prev.mv.to()] = m;
        }
        let stm = b.stm;
        let apply = |s: &mut Self, mv: Move, bonus: i32| {
            hist_update(&mut s.history[(stm * 64 + mv.from()) * 64 + mv.to()], bonus);
            let piece = stm * 6 + b.piece_on(mv.from());
            if let Some(i) = Self::cont_index(prev, piece, mv.to()) {
                hist_update(&mut s.cont[i], bonus);
            }
        };
        apply(self, m, bonus);
        for &o in others {
            apply(self, o, -bonus);
        }
    }

    fn negamax(&mut self, b: &Board, mut alpha: i32, mut beta: i32, mut depth: i32, ply: usize, cut_node: bool) -> i32 {
        let pv_node = beta - alpha > 1;
        self.pv_len[ply] = 0;
        if self.check_stop() {
            return 0;
        }
        let in_check = b.in_check();
        if in_check {
            depth += 1;
        }
        if depth <= 0 {
            return self.qsearch(b, alpha, beta, ply);
        }
        self.count_node();
        self.seldepth = self.seldepth.max(ply);

        if b.halfmove >= 100 || b.is_insufficient_material() || self.is_repetition(b) {
            return self.draw_score(ply);
        }
        if ply >= MAX_PLY - 1 {
            return if in_check { 0 } else { self.eval(b) };
        }
        alpha = alpha.max(-MATE + ply as i32);
        beta = beta.min(MATE - ply as i32 - 1);
        if alpha >= beta {
            return alpha;
        }

        let tte = self.tt.probe(b.hash);
        let tt_move = tte.map_or(Move::NONE, |e| e.mv);
        let mut tt_score = None;
        if let Some(e) = tte {
            let s = score_from_tt(e.score, ply);
            tt_score = Some((s, e.bound));
            if !pv_node
                && e.depth >= depth
                && (e.bound == BOUND_EXACT
                    || (e.bound == BOUND_LOWER && s >= beta)
                    || (e.bound == BOUND_UPPER && s <= alpha))
            {
                return s;
            }
        }

        let static_eval = if in_check {
            EVAL_NONE
        } else {
            match tte {
                Some(e) if e.eval != EVAL_NONE => e.eval,
                _ => self.eval(b),
            }
        };
        // The TT score is a better estimate than the static eval when its bound allows.
        let eval = match tt_score {
            Some((s, bound)) if !in_check && s.abs() < MATE_BOUND => {
                if (bound == BOUND_LOWER && s > static_eval) || (bound == BOUND_UPPER && s < static_eval) || bound == BOUND_EXACT {
                    s
                } else {
                    static_eval
                }
            }
            _ => static_eval,
        };
        self.stack[ply + OFF].eval = static_eval;
        let prev2 = self.stack[ply + OFF - 2].eval;
        let improving = !in_check && prev2 != EVAL_NONE && static_eval > prev2;
        self.killers[ply + 1] = [Move::NONE; 2];

        if !pv_node && !in_check {
            // Reverse futility pruning.
            if depth <= 8 && eval < MATE_BOUND && eval - 75 * (depth - improving as i32) >= beta {
                return (eval + beta) / 2;
            }
            // Razoring.
            if depth <= 3 && eval + 200 + 150 * depth <= alpha {
                let s = self.qsearch(b, alpha, alpha + 1, ply);
                if s <= alpha {
                    return s;
                }
            }
            // Null move pruning.
            if depth >= 3
                && eval >= beta
                && self.stack[ply + OFF - 1].mv.is_some()
                && b.has_non_pawn_material(b.stm)
                && beta > -MATE_BOUND
            {
                let r = 3 + depth / 3 + ((eval - beta) / 200).min(3);
                let nb = b.make_null();
                self.stack[ply + OFF].mv = Move::NONE;
                self.stack[ply + OFF].piece = 12;
                self.hashes.push(nb.hash);
                let s = -self.negamax(&nb, -beta, -beta + 1, depth - r, ply + 1, !cut_node);
                self.hashes.pop();
                if self.stopped {
                    return 0;
                }
                if s >= beta {
                    return if s >= MATE_BOUND { beta } else { s };
                }
            }
        }

        // Internal iterative reduction: no hash move means this node was never searched well.
        if depth >= 4 && tt_move.is_none() && (pv_node || cut_node) {
            depth -= 1;
        }

        let mut moves = MoveList::new();
        generate_legal(b, &mut moves);
        if moves.is_empty() {
            return if in_check { -MATE + ply as i32 } else { self.draw_score(ply) };
        }
        let mut scores = [0i32; MAX_MOVES];
        self.score_moves(b, &moves, tt_move, ply, &mut scores);

        let mut best = -INF;
        let mut best_move = Move::NONE;
        let mut quiets = [Move::NONE; 64];
        let mut n_quiets = 0;
        let mut searched = 0usize;
        let lmp_limit = ((3 + depth * depth) / (2 - improving as i32)) as usize;
        for i in 0..moves.len() {
            let (m, ms) = Self::pick(&mut moves, &mut scores, i);
            let quiet = !m.is_noisy();

            if best > -MATE_BOUND && !in_check {
                if quiet {
                    if depth <= 8 && searched >= lmp_limit {
                        continue;
                    }
                    if depth <= 8 && ms < COUNTER && static_eval + 100 + 90 * depth <= alpha {
                        continue;
                    }
                    if depth <= 8 && !b.see_ge(m, -50 * depth) {
                        continue;
                    }
                } else if depth <= 6 && ms < GOOD_NOISY && !b.see_ge(m, -30 * depth * depth) {
                    continue;
                }
            }

            let child = b.make_move(m);
            let gives_check = child.in_check();
            searched += 1;
            self.stack[ply + OFF].mv = m;
            self.stack[ply + OFF].piece = b.stm * 6 + b.piece_on(m.from());
            self.hashes.push(child.hash);
            let new_depth = depth - 1;
            let mut s;
            if searched == 1 {
                s = -self.negamax(&child, -beta, -alpha, new_depth, ply + 1, !pv_node && !cut_node);
            } else {
                let mut r = 0;
                if depth >= 3 && searched > 1 + pv_node as usize && (quiet || ms < 0) {
                    r = LMR[depth.min(63) as usize][searched.min(63)];
                    r -= pv_node as i32;
                    r += cut_node as i32;
                    r -= improving as i32;
                    r -= gives_check as i32;
                    if ms >= COUNTER {
                        r -= 1;
                    }
                    if quiet {
                        r -= (ms / 8000).clamp(-2, 2);
                    }
                    r = r.clamp(0, (new_depth - 1).max(0));
                }
                s = -self.negamax(&child, -alpha - 1, -alpha, new_depth - r, ply + 1, true);
                if s > alpha && r > 0 {
                    s = -self.negamax(&child, -alpha - 1, -alpha, new_depth, ply + 1, !cut_node);
                }
                if s > alpha && s < beta {
                    s = -self.negamax(&child, -beta, -alpha, new_depth, ply + 1, false);
                }
            }
            self.hashes.pop();
            if self.stopped {
                return 0;
            }

            if s > best {
                best = s;
                if s > alpha {
                    best_move = m;
                    alpha = s;
                    if pv_node {
                        self.pv[ply][0] = m;
                        let n = self.pv_len[ply + 1];
                        for j in 0..n {
                            self.pv[ply][j + 1] = self.pv[ply + 1][j];
                        }
                        self.pv_len[ply] = n + 1;
                    }
                    if s >= beta {
                        if quiet {
                            self.update_quiet_stats(b, m, ply, depth, &quiets[..n_quiets]);
                        }
                        break;
                    }
                }
            }
            if quiet && n_quiets < quiets.len() {
                quiets[n_quiets] = m;
                n_quiets += 1;
            }
        }

        let bound = if best >= beta {
            BOUND_LOWER
        } else if best_move.is_some() {
            BOUND_EXACT
        } else {
            BOUND_UPPER
        };
        self.tt.store(b.hash, best_move, score_to_tt(best, ply), static_eval, depth, bound);
        best
    }

    fn qsearch(&mut self, b: &Board, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.pv_len[ply] = 0;
        if self.check_stop() {
            return 0;
        }
        self.count_node();
        self.seldepth = self.seldepth.max(ply);
        if ply >= MAX_PLY - 1 {
            return self.eval(b);
        }
        if b.is_insufficient_material() {
            return self.draw_score(ply);
        }
        let in_check = b.in_check();
        let tte = self.tt.probe(b.hash);
        if let Some(e) = tte {
            let s = score_from_tt(e.score, ply);
            if e.bound == BOUND_EXACT
                || (e.bound == BOUND_LOWER && s >= beta)
                || (e.bound == BOUND_UPPER && s <= alpha)
            {
                return s;
            }
        }
        let static_eval;
        let mut best;
        if in_check {
            static_eval = EVAL_NONE;
            best = -INF;
        } else {
            static_eval = match tte {
                Some(e) if e.eval != EVAL_NONE => e.eval,
                _ => self.eval(b),
            };
            best = static_eval;
            if best >= beta {
                return best;
            }
            alpha = alpha.max(best);
        }

        let mut moves = MoveList::new();
        if in_check {
            generate_legal(b, &mut moves);
            if moves.is_empty() {
                return -MATE + ply as i32;
            }
        } else {
            generate_noisy(b, &mut moves);
        }
        let mut scores = [0i32; MAX_MOVES];
        self.score_moves(b, &moves, tte.map_or(Move::NONE, |e| e.mv), ply, &mut scores);

        let mut best_move = Move::NONE;
        for i in 0..moves.len() {
            let (m, _) = Self::pick(&mut moves, &mut scores, i);
            if !in_check {
                let victim = if m.is_ep() { PAWN } else { b.piece_on(m.to()) };
                let gain = crate::board::SEE_VALUES[victim] + if m.is_promo() { 800 } else { 0 };
                if static_eval + gain + 150 <= alpha {
                    continue;
                }
                if !b.see_ge(m, 0) {
                    continue;
                }
            }
            let child = b.make_move(m);
            let s = -self.qsearch(&child, -beta, -alpha, ply + 1);
            if self.stopped {
                return 0;
            }
            if s > best {
                best = s;
                if s > alpha {
                    alpha = s;
                    best_move = m;
                    if s >= beta {
                        break;
                    }
                }
            }
        }
        let bound = if best >= beta {
            BOUND_LOWER
        } else if best_move.is_some() {
            BOUND_EXACT
        } else {
            BOUND_UPPER
        };
        self.tt.store(b.hash, best_move, score_to_tt(best, ply), static_eval, 0, bound);
        best
    }

    fn root_search(&mut self, b: &Board, mut alpha: i32, beta: i32, depth: i32) -> i32 {
        self.pv_len[0] = 0;
        let mut best = -INF;
        for i in 0..self.root_moves.len() {
            let m = self.root_moves[i].mv;
            let child = b.make_move(m);
            self.stack[OFF].mv = m;
            self.stack[OFF].piece = b.stm * 6 + b.piece_on(m.from());
            self.hashes.push(child.hash);
            let lo = alpha;
            let mut s;
            if i == 0 {
                s = -self.negamax(&child, -beta, -alpha, depth - 1, 1, false);
            } else {
                s = -self.negamax(&child, -alpha - 1, -alpha, depth - 1, 1, true);
                if s > alpha && s < beta {
                    s = -self.negamax(&child, -beta, -alpha, depth - 1, 1, false);
                }
            }
            self.hashes.pop();
            if self.stopped {
                return best;
            }
            let rm = &mut self.root_moves[i];
            rm.score = s;
            rm.exact = s > lo && s < beta;
            rm.depth = depth;
            best = best.max(s);
            if s > alpha {
                alpha = s;
                self.pv[0][0] = m;
                let n = self.pv_len[1];
                for j in 0..n {
                    self.pv[0][j + 1] = self.pv[1][j];
                }
                self.pv_len[0] = n + 1;
                self.root_moves[..=i].rotate_right(1);
                if s >= beta {
                    break;
                }
            }
        }
        self.root_moves[1..].sort_by(|a, b| b.score.cmp(&a.score));
        best
    }

    /// After `search`: exact scores for some root moves. Each move is searched to `depth`
    /// with the window (floor, ceil); the result is None when it scored <= its floor or there
    /// was no time left, and scores >= ceil come back as ceil. Uses the search's deadline.
    pub fn verify_root_moves(&mut self, root: &Board, game_hashes: &[u64], moves: &[(Move, i32)], ceil: i32, depth: i32) -> Vec<Option<i32>> {
        self.stopped = false;
        self.node_limit = 0;
        self.hashes.clear();
        self.hashes.extend_from_slice(game_hashes);
        if self.hashes.last() != Some(&root.hash) {
            self.hashes.push(root.hash);
        }
        let mut out = Vec::with_capacity(moves.len());
        for &(m, floor) in moves {
            if self.stopped || floor >= ceil {
                out.push(None);
                continue;
            }
            let child = root.make_move(m);
            self.stack[OFF].mv = m;
            self.stack[OFF].piece = root.stm * 6 + root.piece_on(m.from());
            self.hashes.push(child.hash);
            let s = -self.negamax(&child, -ceil, -floor, depth - 1, 1, false);
            self.hashes.pop();
            out.push(if self.stopped || s <= floor { None } else { Some(s.min(ceil)) });
        }
        out
    }

    pub fn search(&mut self, root: &Board, game_hashes: &[u64], p: &SearchParams) -> SearchResult {
        self.start = Instant::now();
        self.stopped = false;
        self.nodes = 0;
        self.seldepth = 0;
        self.hard_ms = p.hard_ms;
        self.node_limit = p.nodes.unwrap_or(0);
        self.hashes.clear();
        self.hashes.extend_from_slice(game_hashes);
        if self.hashes.last() != Some(&root.hash) {
            self.hashes.push(root.hash);
        }
        self.tt.new_search();
        self.eval_params.engine_color = root.stm;
        self.killers = [[Move::NONE; 2]; MAX_PLY + 2];
        for e in self.stack.iter_mut() {
            *e = StackEntry { eval: EVAL_NONE, mv: Move::NONE, piece: 12 };
        }

        let mut legal = MoveList::new();
        generate_legal(root, &mut legal);
        let prior_of = |m: Move| p.priors.iter().find(|(pm, _)| *pm == m).map_or(0.0, |(_, pr)| *pr);
        let allowed = |m: &Move| !p.excluded.contains(m);
        let keep_all = !legal.as_slice().iter().any(allowed);
        self.root_moves = legal
            .as_slice()
            .iter()
            .filter(|m| keep_all || allowed(m))
            .map(|&m| RootMove {
                mv: m,
                score: -INF,
                exact: false,
                depth: 0,
                prior: prior_of(m),
                candidate: false,
            })
            .collect();
        // Initial order: hash move, then human prior, then captures.
        let tt_move = self.tt.probe(root.hash).map_or(Move::NONE, |e| e.mv);
        self.root_moves.sort_by(|a, b| {
            let key = |r: &RootMove| (r.mv == tt_move, (r.prior * 1e6) as i64, r.mv.is_noisy());
            key(b).cmp(&key(a))
        });

        let mut result = SearchResult {
            best: self.root_moves.first().map_or(Move::NONE, |r| r.mv),
            score: if root.in_check() && legal.is_empty() { -MATE } else { 0 },
            depth: 0,
            seldepth: 0,
            nodes: 0,
            time_ms: 0,
            pv: Vec::new(),
            root_moves: self.root_moves.clone(),
        };
        if legal.is_empty() {
            return result;
        }

        let max_depth = p.depth.unwrap_or(MAX_PLY as i32 - 8).min(MAX_PLY as i32 - 8);
        let mut prev_score = 0;
        let mut stable = 0;
        for depth in p.start_depth.max(1)..=max_depth {
            self.seldepth = 0;
            let mut window = 20;
            let (mut alpha, mut beta) = if depth >= 5 { (prev_score - window, prev_score + window) } else { (-INF, INF) };
            let mut score;
            loop {
                score = self.root_search(root, alpha, beta, depth);
                if self.stopped {
                    break;
                }
                if score <= alpha {
                    beta = (alpha + beta) / 2;
                    alpha = (score - window).max(-INF);
                } else if score >= beta {
                    beta = (score + window).min(INF);
                } else {
                    break;
                }
                window += window / 2;
            }
            if self.stopped && depth > 1 {
                break;
            }
            let best = self.root_moves[0].mv;
            stable = if best == result.best { stable + 1 } else { 0 };
            prev_score = score;
            result = SearchResult {
                best,
                score,
                depth,
                seldepth: self.seldepth,
                nodes: self.total_nodes(),
                time_ms: self.elapsed_ms(),
                pv: self.pv[0][..self.pv_len[0]].to_vec(),
                root_moves: self.root_moves.clone(),
            };
            if result.pv.first() != Some(&best) {
                result.pv = vec![best];
            }
            if p.verbose {
                self.print_info(&result);
            }
            if self.stopped {
                break;
            }
            if score.abs() >= MATE_BOUND && depth >= 2 * (MATE - score.abs()) + 4 {
                break;
            }
            if let Some(soft) = p.soft_ms {
                let factor = match stable {
                    0 if depth > 6 => 1.4,
                    0..=1 => 1.0,
                    2..=4 => 0.8,
                    _ => 0.6,
                };
                if self.elapsed_ms() as f64 >= soft as f64 * factor {
                    break;
                }
            }
        }
        result.nodes = self.total_nodes();
        result.time_ms = self.elapsed_ms();
        result
    }

    /// Nodes searched by all threads sharing this searcher's counter.
    fn total_nodes(&self) -> u64 {
        self.shared_nodes.load(Relaxed).max(self.nodes)
    }

    fn print_info(&self, r: &SearchResult) {
        let ms = r.time_ms.max(1);
        let pv: Vec<String> = r.pv.iter().map(|m| m.to_uci()).collect();
        println!(
            "info depth {} seldepth {} multipv 1 score {} nodes {} nps {} hashfull {} time {} pv {}",
            r.depth,
            r.seldepth,
            uci_score(r.score),
            r.nodes,
            r.nodes * 1000 / ms,
            self.tt.hashfull(),
            r.time_ms,
            pv.join(" ")
        );
    }
}

/// Lazy SMP: helper threads run the same iterative deepening on the shared transposition
/// table; the main thread's result is used and the helpers stop when it finishes.
pub struct SearchPool {
    pub tt: Arc<TranspositionTable>,
    pub main: Searcher,
    helpers: Vec<Searcher>,
    helper_stop: Arc<AtomicBool>,
    nodes: Arc<AtomicU64>,
}

impl SearchPool {
    pub fn new(hash_mb: usize, threads: usize, stop: Arc<AtomicBool>) -> SearchPool {
        let tt = Arc::new(TranspositionTable::new(hash_mb));
        let nodes = Arc::new(AtomicU64::new(0));
        let mut main = Searcher::new(tt.clone(), stop);
        main.shared_nodes = nodes.clone();
        let mut pool = SearchPool { tt, main, helpers: Vec::new(), helper_stop: Arc::new(AtomicBool::new(false)), nodes };
        pool.set_threads(threads);
        pool
    }

    pub fn set_threads(&mut self, threads: usize) {
        self.helpers.clear();
        for _ in 1..threads.max(1) {
            let mut h = Searcher::new(self.tt.clone(), self.helper_stop.clone());
            h.shared_nodes = self.nodes.clone();
            self.helpers.push(h);
        }
    }

    pub fn threads(&self) -> usize {
        self.helpers.len() + 1
    }

    pub fn resize_tt(&mut self, mb: usize) {
        self.tt = Arc::new(TranspositionTable::new(mb));
        self.main.tt = self.tt.clone();
        for h in &mut self.helpers {
            h.tt = self.tt.clone();
        }
    }

    pub fn clear(&mut self) {
        self.tt.clear();
        self.main.clear_tables();
        for h in &mut self.helpers {
            h.clear_tables();
        }
    }

    pub fn search(&mut self, root: &Board, game_hashes: &[u64], p: &SearchParams) -> SearchResult {
        self.nodes.store(0, Relaxed);
        if self.helpers.is_empty() {
            return self.main.search(root, game_hashes, p);
        }
        self.helper_stop.store(false, Relaxed);
        let helper_params = SearchParams { soft_ms: None, hard_ms: None, nodes: None, verbose: false, ..p.clone() };
        let eval_params = self.main.eval_params;
        let (main, helpers, helper_stop) = (&mut self.main, &mut self.helpers, &self.helper_stop);
        std::thread::scope(|s| {
            for (i, h) in helpers.iter_mut().enumerate() {
                h.eval_params = eval_params;
                let hp = SearchParams { start_depth: 1 + (i % 2) as i32, ..helper_params.clone() };
                std::thread::Builder::new()
                    .stack_size(SEARCH_STACK)
                    .spawn_scoped(s, move || {
                        h.search(root, game_hashes, &hp);
                    })
                    .expect("spawn helper thread");
            }
            let r = main.search(root, game_hashes, p);
            helper_stop.store(true, Relaxed);
            r
        })
    }

    /// See `Searcher::verify_root_moves` (runs on the main thread).
    pub fn verify_root_moves(&mut self, root: &Board, game_hashes: &[u64], moves: &[(Move, i32)], ceil: i32, depth: i32) -> Vec<Option<i32>> {
        self.main.verify_root_moves(root, game_hashes, moves, ceil, depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn search_fen(fen: &str, depth: i32) -> SearchResult {
        let b = Board::from_fen(fen).unwrap();
        let mut s = Searcher::standalone(16);
        s.search(&b, &[b.hash], &SearchParams { depth: Some(depth), ..Default::default() })
    }

    #[test]
    fn finds_mate_in_two() {
        // 1. Qxh7+? no: classic back-rank pattern, mate in 2 with Rd8+.
        let r = search_fen("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", 6);
        assert_eq!(r.best.to_uci(), "d1d8");
        assert!(r.score >= MATE_BOUND, "score {}", r.score);
    }

    #[test]
    fn wins_hanging_queen() {
        let r = search_fen("rnb1kbnr/pppp1ppp/8/4p1q1/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1", 6);
        assert_eq!(r.best.to_uci(), "c1g5");
    }

    #[test]
    fn stalemate_and_mate_detection() {
        let r = search_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 3);
        assert!(r.best.is_none() && r.score == 0);
        let r = search_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", 3);
        assert!(r.best.is_none() && r.score == -MATE);
    }
}
