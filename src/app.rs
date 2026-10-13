//! The Magician app (`magician play`): a small local web server with a board UI.
//! - play against Magician in any style and time control,
//! - watch Magician play Stockfish (if present) or another Magician style,
//! - let Magician play Chess.com computer bots in the user's own Edge window, through the
//!   extension in tools/chesscom/extension (it talks to /api/bridge/*),
//! - browse the saved games and the analysis of their errors.
//! Everything is stored in the data folder (games/, learn.txt) next to nets/.

use crate::board::Board;
use crate::engine::{data_dir, think, Game, Nets, Options, StyleKind};
use crate::gamelog::{result_for, EngineNote, GameLog};
use crate::learn::Learn;
use crate::movegen::legal_moves;
use crate::san::{parse_uci, to_san};
use crate::search::{Limits, SearchPool, SEARCH_STACK};
use crate::types::*;
use crate::uci::style_name;
use crate::ucicl::{find_stockfish, UciEngine};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const INDEX_HTML: &str = include_str!("../web/index.html");

/// Piece images: the cburnett set (Colin M.L. Burnett, CC BY-SA 3.0), as used by Lichess.
fn piece_svg(name: &str) -> Option<&'static str> {
    Some(match name {
        "wK.svg" => include_str!("../web/pieces/wK.svg"),
        "wQ.svg" => include_str!("../web/pieces/wQ.svg"),
        "wR.svg" => include_str!("../web/pieces/wR.svg"),
        "wB.svg" => include_str!("../web/pieces/wB.svg"),
        "wN.svg" => include_str!("../web/pieces/wN.svg"),
        "wP.svg" => include_str!("../web/pieces/wP.svg"),
        "bK.svg" => include_str!("../web/pieces/bK.svg"),
        "bQ.svg" => include_str!("../web/pieces/bQ.svg"),
        "bR.svg" => include_str!("../web/pieces/bR.svg"),
        "bB.svg" => include_str!("../web/pieces/bB.svg"),
        "bN.svg" => include_str!("../web/pieces/bN.svg"),
        "bP.svg" => include_str!("../web/pieces/bP.svg"),
        _ => return None,
    })
}

#[derive(Clone, Debug, PartialEq)]
enum Player {
    Human,
    Magician(StyleKind),
    Stockfish(u32),
}

impl Player {
    fn parse(s: &str) -> Option<Player> {
        let mut it = s.split(':');
        match it.next()? {
            "human" => Some(Player::Human),
            "magician" => Some(Player::Magician(StyleKind::parse(it.next().unwrap_or("human"))?)),
            "stockfish" => Some(Player::Stockfish(it.next().and_then(|e| e.parse().ok()).unwrap_or(2500).clamp(1320, 3190))),
            _ => None,
        }
    }

    fn name(&self) -> String {
        match self {
            Player::Human => "Вы".into(),
            Player::Magician(s) => format!("Magician ({})", style_name(*s)),
            Player::Stockfish(e) => format!("Stockfish 19 ({e})"),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Player::Human => "human",
            _ => "engine",
        }
    }
}

/// UI facts about one played move.
#[derive(Clone, Default)]
struct MoveInfo {
    uci: String,
    san: String,
    text: String,
}

struct LocalGame {
    id: u64,
    players: [Player; 2],
    log: GameLog,
    game: Game,
    clocks: [i64; 2],
    inc: i64,
    turn_start: Instant,
    infos: Vec<MoveInfo>,
    thinking: bool,
    human_move: Option<Move>,
    resign: bool,
    abort: bool,
    saved: Option<PathBuf>,
}

impl LocalGame {
    fn remaining(&self, side: usize) -> i64 {
        let running = !self.log.finished() && self.game.board.stm == side;
        self.clocks[side] - if running { self.turn_start.elapsed().as_millis() as i64 } else { 0 }
    }
}

/// A background process whose output the UI shows (Chess.com runs, analysis).
#[derive(Default)]
struct Job {
    child: Option<Child>,
    running: bool,
    log: VecDeque<String>,
}

struct State {
    game: Option<LocalGame>,
    next_id: u64,
    analysis: Arc<Mutex<Job>>,
}

/// The game Magician plays inside the user's own browser through the Edge extension
/// (Chess.com computer bots only). The extension reports what the board shows; the moves
/// are reconstructed here and Magician answers when it is its turn.
struct Bridge {
    log: Option<GameLog>,
    game: Game,
    our: usize,
    pool: SearchPool,
    learn: Learn,
    /// Magician's move that the board does not show yet, with its notes.
    pending: Option<(Move, EngineNote)>,
    status: String,
    last_text: String,
    saved: Option<PathBuf>,
}

struct App {
    state: Mutex<State>,
    cv: Condvar,
    bridge: Mutex<Bridge>,
    nets: Nets,
    dir: PathBuf,
    stockfish: Option<PathBuf>,
    port: u16,
}

pub fn run(port: u16, open_browser: bool) -> std::io::Result<()> {
    let dir = data_dir();
    let listener = (port..port + 20)
        .find_map(|p| TcpListener::bind(("127.0.0.1", p)).ok())
        .ok_or_else(|| std::io::Error::other("no free port"))?;
    let port = listener.local_addr()?.port();
    let app = Arc::new(App {
        state: Mutex::new(State { game: None, next_id: 1, analysis: Arc::new(Mutex::new(Job::default())) }),
        cv: Condvar::new(),
        bridge: Mutex::new(Bridge {
            log: None,
            game: Game::new(Board::startpos()),
            our: WHITE,
            pool: SearchPool::new(64, 2, Arc::new(AtomicBool::new(false))),
            learn: Learn::load(&dir.join("learn.txt")),
            pending: None,
            status: "ждёт партию на chess.com/play/computer".into(),
            last_text: String::new(),
            saved: None,
        }),
        nets: Nets::load(None),
        stockfish: find_stockfish(&dir),
        dir,
        port,
    });
    let url = format!("http://127.0.0.1:{port}/");
    println!("Magician работает: {url}");
    println!("Не закрывайте это окно, пока играете. Партии сохраняются в {}", app.dir.join("games").display());
    if open_browser {
        let _ = Command::new("cmd").args(["/C", "start", "", &url]).spawn();
    }
    for stream in listener.incoming().flatten() {
        let app = app.clone();
        std::thread::spawn(move || {
            let _ = handle(stream, &app);
        });
    }
    Ok(())
}

// ------------------------------------------------------------------ HTTP

struct Request {
    method: String,
    path: String,
    query: Vec<(String, String)>,
    host: String,
}

impl Request {
    fn q(&self, k: &str) -> Option<&str> {
        self.query.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str())
    }
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() => {
                if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                    out.push(v);
                    i += 2;
                }
            }
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn read_request(stream: &mut TcpStream) -> Option<Request> {
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 || buf.len() > 64 * 1024 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let text = String::from_utf8_lossy(&buf);
    let mut lines = text.split("\r\n");
    let mut first = lines.next()?.split_whitespace();
    let method = first.next()?.to_string();
    let target = first.next()?.to_string();
    let host = lines
        .find_map(|l| l.split_once(':').filter(|(k, _)| k.eq_ignore_ascii_case("host")).map(|(_, v)| v.trim().to_string()))
        .unwrap_or_default();
    let (path, qs) = target.split_once('?').unwrap_or((&target, ""));
    let query = qs
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (percent_decode(k), percent_decode(v))
        })
        .collect();
    Some(Request { method, path: path.to_string(), query, host })
}

fn respond(stream: &mut TcpStream, status: &str, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)
}

fn handle(mut stream: TcpStream, app: &Arc<App>) -> std::io::Result<()> {
    let Some(req) = read_request(&mut stream) else { return Ok(()) };
    // Only answer pages served from this machine (blocks DNS-rebinding tricks).
    let host_ok = req.host == format!("127.0.0.1:{}", app.port) || req.host == format!("localhost:{}", app.port);
    if !host_ok {
        return respond(&mut stream, "403 Forbidden", "text/plain", b"forbidden");
    }
    let json = |s: String| (String::from("200 OK"), "application/json; charset=utf-8", s.into_bytes());
    let (status, ctype, body) = match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") => ("200 OK".into(), "text/html; charset=utf-8", INDEX_HTML.as_bytes().to_vec()),
        ("GET", p) if p.starts_with("/pieces/") => match piece_svg(&p[8..]) {
            Some(svg) => ("200 OK".into(), "image/svg+xml", svg.as_bytes().to_vec()),
            None => ("404 Not Found".into(), "text/plain", b"not found".to_vec()),
        },
        ("GET", "/api/state") => json(state_json(app)),
        ("POST", "/api/new") => json(new_game(app, &req)),
        ("POST", "/api/move") => json(human_move(app, &req)),
        ("POST", "/api/resign") => json(resign(app)),
        ("GET", "/api/games") => json(games_json(app)),
        ("GET", "/api/game") => {
            let text = req.q("file").and_then(|f| game_file(app, f)).and_then(|p| std::fs::read_to_string(p).ok());
            match text {
                Some(t) => ("200 OK".into(), "text/plain; charset=utf-8", t.into_bytes()),
                None => ("404 Not Found".into(), "text/plain", b"not found".to_vec()),
            }
        }
        ("POST", "/api/analyze") => json(analyze_start(app)),
        ("POST", "/api/open") => json(open_external(app, req.q("what").unwrap_or(""))),
        ("POST", "/api/bridge/observe") => json(bridge_observe(app, &req)),
        ("POST", "/api/bridge/end") => json(bridge_end(app, &req)),
        _ => ("404 Not Found".into(), "text/plain", b"not found".to_vec()),
    };
    respond(&mut stream, &status, ctype, &body)
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => {}
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn ok(msg: &str) -> String {
    format!("{{\"ok\":true,\"message\":{}}}", esc(msg))
}

fn err(msg: &str) -> String {
    format!("{{\"ok\":false,\"message\":{}}}", esc(msg))
}

// ------------------------------------------------------------------ state

fn state_json(app: &Arc<App>) -> String {
    let st = app.state.lock().unwrap();
    let game = match &st.game {
        None => "null".to_string(),
        Some(g) => {
            let b = &g.game.board;
            let human_turn = !g.log.finished() && g.players[b.stm] == Player::Human;
            let legal: Vec<String> = if human_turn {
                legal_moves(b).as_slice().iter().map(|m| esc(&m.to_uci())).collect()
            } else {
                Vec::new()
            };
            let moves: Vec<String> = g
                .infos
                .iter()
                .map(|i| format!("{{\"uci\":{},\"san\":{},\"text\":{}}}", esc(&i.uci), esc(&i.san), esc(&i.text)))
                .collect();
            let players: Vec<String> =
                g.players.iter().map(|p| format!("{{\"name\":{},\"kind\":\"{}\"}}", esc(&p.name()), p.kind())).collect();
            let check = if b.in_check() { esc(&sq_name(b.king_sq(b.stm))) } else { "null".into() };
            format!(
                "{{\"id\":{},\"fen\":{},\"turn\":\"{}\",\"moves\":[{}],\"legal\":[{}],\"clocks\":[{},{}],\"players\":[{}],\"result\":{},\"termination\":{},\"thinking\":{},\"check\":{},\"saved\":{}}}",
                g.id,
                esc(&b.to_fen()),
                if b.stm == WHITE { "w" } else { "b" },
                moves.join(","),
                legal.join(","),
                g.remaining(WHITE).max(0),
                g.remaining(BLACK).max(0),
                players.join(","),
                esc(&g.log.result),
                esc(&g.log.termination),
                g.thinking,
                check,
                g.saved.as_ref().map_or("null".into(), |p| esc(&p.display().to_string()))
            )
        }
    };
    let job_json = |j: &Arc<Mutex<Job>>| {
        let j = j.lock().unwrap();
        let lines: Vec<String> = j.log.iter().map(|l| esc(l)).collect();
        format!("{{\"running\":{},\"log\":[{}]}}", j.running, lines.join(","))
    };
    let bridge = {
        let br = app.bridge.lock().unwrap();
        let moves: Vec<String> = br.log.as_ref().map_or(Vec::new(), |l| l.moves.iter().map(|m| esc(&m.san)).collect());
        format!(
            "{{\"active\":{},\"status\":{},\"moves\":[{}],\"last\":{},\"saved\":{}}}",
            br.log.is_some(),
            esc(&br.status),
            moves.join(","),
            esc(&br.last_text),
            br.saved.as_ref().map_or("null".into(), |p| esc(&p.display().to_string()))
        )
    };
    format!(
        "{{\"game\":{game},\"bridge\":{bridge},\"analysis\":{},\"stockfish\":{},\"data_dir\":{},\"extension_dir\":{}}}",
        job_json(&st.analysis),
        app.stockfish.is_some(),
        esc(&app.dir.display().to_string()),
        esc(&extension_dir(app).display().to_string())
    )
}

// ------------------------------------------------------------------ Chess.com bridge

fn placement(b: &Board) -> String {
    b.to_fen().split(' ').next().unwrap_or_default().to_string()
}

/// One or two legal moves that turn `b` into the shown placement.
fn find_path(b: &Board, shown: &str) -> Option<Vec<Move>> {
    for &m1 in legal_moves(b).as_slice() {
        let b1 = b.make_move(m1);
        if placement(&b1) == shown {
            return Some(vec![m1]);
        }
    }
    for &m1 in legal_moves(b).as_slice() {
        let b1 = b.make_move(m1);
        for &m2 in legal_moves(&b1).as_slice() {
            if placement(&b1.make_move(m2)) == shown {
                return Some(vec![m1, m2]);
            }
        }
    }
    None
}

fn bridge_save(app: &App, br: &mut Bridge) {
    if let Some(mut log) = br.log.take() {
        if !log.moves.is_empty() {
            br.saved = log.save(&app.dir.join("games")).ok();
        }
    }
    br.pending = None;
}

fn bridge_observe(app: &Arc<App>, req: &Request) -> String {
    let Some(shown) = req.q("board") else { return err("нет позиции") };
    let our = if req.q("flipped") == Some("1") { BLACK } else { WHITE };
    let style = req.q("style").and_then(StyleKind::parse).unwrap_or(StyleKind::Human);
    let opponent = req.q("opponent").filter(|o| !o.is_empty()).unwrap_or("бот Chess.com").to_string();
    let mut br = app.bridge.lock().unwrap();
    let start = Board::startpos();
    let at_start = shown == placement(&start);
    if at_start && (br.log.is_none() || !br.game.moves.is_empty() || br.our != our) {
        bridge_save(app, &mut br);
        let me = format!("Magician ({})", style_name(style));
        let them = format!("Chess.com: {opponent}");
        let (w, b) = if our == WHITE { (me, them) } else { (them, me) };
        let mut log = GameLog::new(start.clone(), &w, &b, "Chess.com vs computer");
        log.magician[our] = true;
        log.opponent_type = "engine".into();
        br.log = Some(log);
        br.game = Game::new(start);
        br.our = our;
        br.saved = None;
        br.status = "новая партия".into();
    }
    if br.log.is_none() {
        br.status = "начните новую партию: Magician подключается с первого хода".into();
        return format!("{{\"ok\":true,\"move\":null,\"status\":{}}}", esc(&br.status));
    }
    if placement(&br.game.board) != shown {
        let Some(path) = find_path(&br.game.board, shown) else {
            return format!("{{\"ok\":true,\"move\":null,\"status\":{}}}", esc("не понял позицию на доске, жду"));
        };
        for m in path {
            let note = br.pending.take().filter(|(pm, _)| *pm == m).map(|(_, n)| n);
            if let Some(log) = br.log.as_mut() {
                log.push(m, None, note);
            }
            br.game.play(m);
        }
        if let Some(log) = br.log.as_mut() {
            let _ = log.save(&app.dir.join("games"));
        }
        let finished = br.log.as_mut().is_some_and(|l| l.check_end());
        if finished {
            let result = br.log.as_ref().map(|l| format!("{} ({})", l.result, l.termination)).unwrap_or_default();
            bridge_save(app, &mut br);
            br.status = format!("партия окончена: {result}");
            return format!("{{\"ok\":true,\"move\":null,\"status\":{},\"over\":true}}", esc(&br.status));
        }
    }
    if br.game.board.stm != our {
        br.status = "ход соперника".into();
        return format!("{{\"ok\":true,\"move\":null,\"status\":{}}}", esc(&br.status));
    }
    if let Some((m, _)) = br.pending.as_ref() {
        // Already decided; the board just has not shown it yet.
        return format!("{{\"ok\":true,\"move\":{},\"status\":\"жду, пока ход появится на доске\"}}", esc(&m.to_uci()));
    }
    let clock = |k: &str| req.q(k).and_then(|v| v.parse::<u64>().ok()).filter(|&v| v > 0);
    let limits = match (clock("wclock"), clock("bclock")) {
        (Some(w), Some(b)) => Limits { time: [Some(w), Some(b)], ..Default::default() },
        _ => Limits { movetime: Some(1000), ..Default::default() },
    };
    let opts = Options { style, quiet: true, threads: 2, ..Default::default() };
    let net = app.nets.for_style(style);
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
    let game = br.game.clone();
    let avoid = br.learn.avoid(&game.board);
    let th = think(&mut br.pool, &game, &limits, &opts, net.as_deref(), seed, &avoid);
    let r = &th.result;
    let best = r.root_moves.first().map_or(th.mv, |m| m.mv);
    let score = th.played_score();
    let mut text = format!(
        "{}: оценка {} · глубина {} · {:.1} с",
        to_san(&game.board, th.mv),
        crate::gamelog::eval_text(if our == WHITE { score } else { -score }),
        r.depth,
        r.time_ms as f64 / 1000.0
    );
    if best != th.mv {
        text.push_str(&format!(" · человеческий ход вместо {}", to_san(&game.board, best)));
    }
    let note = EngineNote {
        score: Some(score),
        depth: r.depth,
        nodes: r.nodes,
        time_ms: r.time_ms,
        best,
        best_score: r.root_moves.first().map(|m| m.score),
        human_prob: (!th.policy.is_empty()).then(|| th.played_prior()),
    };
    br.pending = Some((th.mv, note));
    br.last_text = text.clone();
    br.status = "Magician сделал ход".into();
    format!("{{\"ok\":true,\"move\":{},\"status\":{}}}", esc(&th.mv.to_uci()), esc(&text))
}

/// The page reported the end of the game (resignation, time, a result on the board).
fn bridge_end(app: &Arc<App>, req: &Request) -> String {
    let mut br = app.bridge.lock().unwrap();
    if let Some(log) = br.log.as_mut() {
        if !log.finished() {
            let result = match req.q("result") {
                Some(r @ ("1-0" | "0-1" | "1/2-1/2")) => r,
                _ => "*",
            };
            log.set_result(result, req.q("reason").unwrap_or("Chess.com"));
        }
    }
    bridge_save(app, &mut br);
    br.status = "партия записана".into();
    ok(&br.status)
}

// ------------------------------------------------------------------ local games

fn new_game(app: &Arc<App>, req: &Request) -> String {
    let (Some(white), Some(black)) = (req.q("white").and_then(Player::parse), req.q("black").and_then(Player::parse)) else {
        return err("неизвестный игрок");
    };
    if (matches!(white, Player::Stockfish(_)) || matches!(black, Player::Stockfish(_))) && app.stockfish.is_none() {
        return err("Stockfish не найден: положите stockfish.exe в папку stockfish рядом с программой");
    }
    let base = req.q("base").and_then(|v| v.parse::<i64>().ok()).unwrap_or(60).clamp(10, 7200) * 1000;
    let inc = req.q("inc").and_then(|v| v.parse::<i64>().ok()).unwrap_or(0).clamp(0, 60) * 1000;
    let id = {
        let mut st = app.state.lock().unwrap();
        if let Some(g) = st.game.as_mut() {
            g.abort = true;
        }
        let id = st.next_id;
        st.next_id += 1;
        let start = Board::startpos();
        let mut log = GameLog::new(start.clone(), &white.name(), &black.name(), "Magician app");
        log.time_control = format!("{}+{}", base / 1000, inc / 1000);
        log.magician = [matches!(white, Player::Magician(_)), matches!(black, Player::Magician(_))];
        log.opponent_type = if white == Player::Human || black == Player::Human { "human" } else { "engine" }.into();
        st.game = Some(LocalGame {
            id,
            players: [white.clone(), black.clone()],
            log,
            game: Game::new(start),
            clocks: [base, base],
            inc,
            turn_start: Instant::now(),
            infos: Vec::new(),
            thinking: false,
            human_move: None,
            resign: false,
            abort: false,
            saved: None,
        });
        id
    };
    app.cv.notify_all();
    let app2 = app.clone();
    std::thread::Builder::new()
        .stack_size(SEARCH_STACK)
        .spawn(move || controller(app2, id, [white, black]))
        .expect("spawn game thread");
    ok("партия началась")
}

fn human_move(app: &Arc<App>, req: &Request) -> String {
    let mut st = app.state.lock().unwrap();
    let Some(g) = st.game.as_mut() else { return err("нет партии") };
    if g.log.finished() || g.players[g.game.board.stm] != Player::Human {
        return err("сейчас не ваш ход");
    }
    let Some(m) = req.q("uci").and_then(|u| parse_uci(&g.game.board, u)) else { return err("нелегальный ход") };
    g.human_move = Some(m);
    drop(st);
    app.cv.notify_all();
    ok("ход принят")
}

fn resign(app: &Arc<App>) -> String {
    let mut st = app.state.lock().unwrap();
    if let Some(g) = st.game.as_mut() {
        g.resign = true;
    }
    drop(st);
    app.cv.notify_all();
    ok("партия остановлена")
}

/// Seat-specific engines live in the game thread, never under the state lock.
enum Seat {
    Human,
    Magician(Box<SearchPool>, StyleKind),
    Stockfish(Box<UciEngine>),
}

fn controller(app: Arc<App>, id: u64, players: [Player; 2]) {
    let learn = Learn::load(&app.dir.join("learn.txt"));
    let stop = Arc::new(AtomicBool::new(false));
    let mut seats: Vec<Seat> = Vec::new();
    for p in &players {
        seats.push(match p {
            Player::Human => Seat::Human,
            Player::Magician(s) => Seat::Magician(Box::new(SearchPool::new(64, 1, stop.clone())), *s),
            Player::Stockfish(elo) => {
                let opts = [("UCI_LimitStrength", "true".to_string()), ("UCI_Elo", elo.to_string()), ("Threads", "1".into())];
                match app.stockfish.as_ref().map(|p| UciEngine::start(p, &opts)) {
                    Some(Ok(mut e)) => {
                        let _ = e.new_game();
                        Seat::Stockfish(Box::new(e))
                    }
                    _ => {
                        finish(&app, id, "*", "Stockfish не запустился");
                        return;
                    }
                }
            }
        });
    }
    loop {
        // Snapshot under the lock.
        let (side, game, clocks, inc) = {
            let st = app.state.lock().unwrap();
            let Some(g) = st.game.as_ref().filter(|g| g.id == id) else { return };
            if g.abort || g.log.finished() {
                return;
            }
            let b = &g.game.board;
            (b.stm, g.game.clone(), [g.remaining(WHITE), g.remaining(BLACK)], g.inc)
        };
        if clocks[side] <= 0 {
            finish(&app, id, result_for(Some(side ^ 1)), "время вышло");
            return;
        }
        let limits = Limits {
            time: [Some(clocks[0].max(1) as u64), Some(clocks[1].max(1) as u64)],
            inc: [Some(inc as u64), Some(inc as u64)],
            ..Default::default()
        };
        let mut note = None;
        let mut text = String::new();
        let mv = match &mut seats[side] {
            Seat::Human => {
                let st = app.state.lock().unwrap();
                let wait = Duration::from_millis(clocks[side].max(1) as u64);
                let (mut st, _) = app
                    .cv
                    .wait_timeout_while(st, wait, |s| {
                        s.game.as_ref().is_some_and(|g| g.id == id && !g.abort && !g.resign && g.human_move.is_none())
                    })
                    .unwrap();
                let Some(g) = st.game.as_mut().filter(|g| g.id == id) else { return };
                if g.abort {
                    return;
                }
                if g.resign {
                    drop(st);
                    finish(&app, id, result_for(Some(side ^ 1)), "сдача");
                    return;
                }
                match g.human_move.take() {
                    Some(m) => m,
                    None => continue, // timed out: the flag check above ends the game
                }
            }
            Seat::Magician(pool, style) => {
                set_thinking(&app, id, true);
                let opts = Options { style: *style, quiet: true, ..Default::default() };
                let net = app.nets.for_style(*style);
                let seed = SystemTime::now().duration_since(UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
                let avoid = learn.avoid(&game.board);
                let th = think(pool, &game, &limits, &opts, net.as_deref(), seed, &avoid);
                let r = &th.result;
                let best = r.root_moves.first().map_or(th.mv, |m| m.mv);
                let score = th.played_score();
                text = format!(
                    "оценка {} · глубина {} · {:.1} с",
                    crate::gamelog::eval_text(if side == WHITE { score } else { -score }),
                    r.depth,
                    r.time_ms as f64 / 1000.0
                );
                if !th.policy.is_empty() {
                    let top: Vec<String> =
                        th.policy.iter().take(3).map(|(m, p)| format!("{} {:.0}%", to_san(&game.board, *m), p * 100.0)).collect();
                    text.push_str(&format!(" · люди играют: {}", top.join(", ")));
                    if best != th.mv {
                        text.push_str(&format!(" · человеческий ход вместо {}", to_san(&game.board, best)));
                    }
                }
                note = Some(EngineNote {
                    score: Some(score),
                    depth: r.depth,
                    nodes: r.nodes,
                    time_ms: r.time_ms,
                    best,
                    best_score: r.root_moves.first().map(|m| m.score),
                    human_prob: (!th.policy.is_empty()).then(|| th.played_prior()),
                });
                th.mv
            }
            Seat::Stockfish(e) => {
                set_thinking(&app, id, true);
                let start_fen = Board::startpos().to_fen();
                let reply = e.go(&start_fen, &game.moves, [clocks[0].max(1) as u64, clocks[1].max(1) as u64], inc as u64);
                match reply.ok().and_then(|r| parse_uci(&game.board, &r.bestmove).map(|m| (m, r))) {
                    Some((m, r)) => {
                        text = format!(
                            "оценка {} · глубина {}",
                            r.score.map_or("?".into(), |s| crate::gamelog::eval_text(if side == WHITE { s } else { -s })),
                            r.depth
                        );
                        note = Some(EngineNote { score: r.score, depth: r.depth, nodes: r.nodes, ..Default::default() });
                        m
                    }
                    None => {
                        finish(&app, id, "*", "ошибка Stockfish");
                        return;
                    }
                }
            }
        };
        // Apply the move.
        let mut st = app.state.lock().unwrap();
        let Some(g) = st.game.as_mut().filter(|g| g.id == id) else { return };
        if g.abort {
            return;
        }
        g.thinking = false;
        let used = g.turn_start.elapsed().as_millis() as i64;
        g.clocks[side] -= used;
        if g.clocks[side] <= 0 {
            g.clocks[side] = 0;
            drop(st);
            finish(&app, id, result_for(Some(side ^ 1)), "время вышло");
            return;
        }
        g.clocks[side] += g.inc;
        g.turn_start = Instant::now();
        let san = to_san(&g.game.board, mv);
        if let Some(n) = note.as_mut() {
            n.time_ms = used as u64;
        }
        g.log.push(mv, Some(g.clocks[side] as u64), note);
        g.game.play(mv);
        g.infos.push(MoveInfo { uci: mv.to_uci(), san, text });
        g.saved = g.log.save(&app.dir.join("games")).ok();
        if g.log.check_end() {
            let (r, t) = (g.log.result.clone(), g.log.termination.clone());
            drop(st);
            finish(&app, id, &r, &t);
            return;
        }
    }
}

fn set_thinking(app: &Arc<App>, id: u64, v: bool) {
    if let Some(g) = app.state.lock().unwrap().game.as_mut().filter(|g| g.id == id) {
        g.thinking = v;
    }
}

fn finish(app: &Arc<App>, id: u64, result: &str, termination: &str) {
    let mut st = app.state.lock().unwrap();
    let Some(g) = st.game.as_mut().filter(|g| g.id == id) else { return };
    g.thinking = false;
    if !g.log.finished() {
        g.log.set_result(result, termination);
    }
    if !g.log.moves.is_empty() {
        g.saved = g.log.save(&app.dir.join("games")).ok();
    }
}

// ------------------------------------------------------------------ jobs

fn spawn_job(job: &Arc<Mutex<Job>>, mut cmd: Command) -> Result<(), String> {
    let mut j = job.lock().unwrap();
    if j.running {
        return Err("уже запущено".into());
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let mut child = cmd.spawn().map_err(|e| format!("не удалось запустить: {e}"))?;
    let out = child.stdout.take();
    let errs = child.stderr.take();
    j.log.clear();
    j.running = true;
    j.child = Some(child);
    for stream in [out.map(|o| Box::new(o) as Box<dyn Read + Send>), errs.map(|e| Box::new(e) as Box<dyn Read + Send>)]
        .into_iter()
        .flatten()
    {
        let job = job.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                let mut j = job.lock().unwrap();
                j.log.push_back(line);
                while j.log.len() > 400 {
                    j.log.pop_front();
                }
            }
        });
    }
    let job = job.clone();
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_millis(300));
        let mut j = job.lock().unwrap();
        let done = match j.child.as_mut() {
            Some(c) => c.try_wait().map_or(true, |s| s.is_some()),
            None => true,
        };
        if done {
            j.running = false;
            j.child = None;
            break;
        }
    });
    Ok(())
}

/// The extension folder: next to the installed program, or in the project.
fn extension_dir(app: &App) -> PathBuf {
    let installed = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join("extension")));
    match installed {
        Some(d) if d.join("manifest.json").exists() => d,
        _ => app.dir.join("tools").join("chesscom").join("extension"),
    }
}

/// Opens Chess.com's bots page, Edge's extensions page or the extension folder.
fn open_external(app: &Arc<App>, what: &str) -> String {
    let result = match what {
        "chesscom" => Command::new("cmd").args(["/C", "start", "", "https://www.chess.com/play/computer"]).spawn(),
        "extensions" => Command::new("cmd").args(["/C", "start", "msedge", "edge://extensions"]).spawn(),
        "extfolder" => Command::new("explorer").arg(extension_dir(app)).spawn(),
        _ => return err("неизвестно что открыть"),
    };
    match result {
        Ok(_) => ok("открыто"),
        Err(e) => err(&format!("не удалось открыть: {e}")),
    }
}

fn analyze_start(app: &Arc<App>) -> String {
    let Ok(exe) = std::env::current_exe() else { return err("не найден exe") };
    let mut cmd = Command::new(exe);
    cmd.arg("analyze").current_dir(&app.dir).env("MAGICIAN_HOME", &app.dir);
    let job = app.state.lock().unwrap().analysis.clone();
    match spawn_job(&job, cmd) {
        Ok(()) => ok("разбор запущен"),
        Err(e) => err(&e),
    }
}

/// Saved games, newest first, with the analysis report when there is one.
fn games_json(app: &Arc<App>) -> String {
    let dir = app.dir.join("games");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "pgn")).collect())
        .unwrap_or_default();
    files.sort();
    files.reverse();
    let items: Vec<String> = files
        .iter()
        .take(100)
        .map(|p| {
            let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
            let report = std::fs::read_to_string(dir.join("analysis").join(Path::new(&name).with_extension("txt"))).ok();
            format!("{{\"file\":{},\"report\":{}}}", esc(&name), report.map_or("null".into(), |r| esc(&r)))
        })
        .collect();
    format!("[{}]", items.join(","))
}

/// A game file name from the UI, resolved inside games/ (or games/analysis/) only.
fn game_file(app: &Arc<App>, name: &str) -> Option<PathBuf> {
    if name.contains("..") || name.contains('/') || name.contains('\\') {
        return None;
    }
    let dir = app.dir.join("games");
    [dir.join(name), dir.join("analysis").join(name)].into_iter().find(|p| p.exists())
}
