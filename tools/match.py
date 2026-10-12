"""Play a match between two UCI engines and estimate the Elo difference.

  python tools/match.py --engine1 target/release/magician.exe Style=Human \
                        --engine2 target/release/magician.exe Style=Engine \
                        --games 40 --tc 10+0.1 --pgn match.pgn

Each opening is played twice with colours reversed. Engine options are NAME=VALUE pairs.
"""

import argparse
import math
import random
import time

import chess
import chess.engine
import chess.pgn

# Common human openings (a few moves each) to diversify the games.
OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6",
    "e2e4 c7c5 b1c3 b8c6 g2g3",
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g7g6",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
    "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5",
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6",
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6",
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6",
    "d2d4 g8f6 c2c4 c7c5 d4d5 b7b5",
    "d2d4 f7f5 g2g3 g8f6 f1g2 g7g6",
    "d2d4 d7d5 g1f3 g8f6 c1f4 c7c5",
    "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5",
    "c2c4 c7c5 g1f3 g8f6 b1c3 b8c6",
    "g1f3 d7d5 g2g3 g8f6 f1g2 c7c6",
    "e2e4 e7e5 f2f4 e5f4 g1f3 g7g5",
    "e2e4 c7c5 c2c3 g8f6 e4e5 f6d5",
    "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6 b1c3 e6d5 c4d5 d7d6",
]


def parse_engine(spec):
    path, *opts = spec
    options = {}
    for o in opts:
        k, _, v = o.partition("=")
        options[k] = v
    return path, options


def open_engine(path, options):
    eng = chess.engine.SimpleEngine.popen_uci(path)
    cfg = {}
    for k, v in options.items():
        opt = eng.options.get(k)
        if opt is None:
            raise SystemExit(f"{path} has no option {k}")
        if opt.type == "spin":
            cfg[k] = int(v)
        elif opt.type == "check":
            cfg[k] = v.lower() == "true"
        else:
            cfg[k] = v
    if cfg:
        eng.configure(cfg)
    return eng


def play_game(engines, names, opening, tc, nodes, movetime, max_moves=250):
    board = chess.Board()
    for u in opening.split():
        board.push_uci(u)
    game = chess.pgn.Game()
    game.headers["White"], game.headers["Black"] = names
    game.setup(board)
    node = game
    base, inc = tc
    clocks = [base, base]
    game_key = object()  # a new key makes python-chess send ucinewgame
    while not board.is_game_over(claim_draw=True) and board.fullmove_number <= max_moves:
        side = 0 if board.turn == chess.WHITE else 1
        eng = engines[side]
        if nodes:
            limit = chess.engine.Limit(nodes=nodes)
        elif movetime:
            limit = chess.engine.Limit(time=movetime)
        else:
            limit = chess.engine.Limit(white_clock=clocks[0], black_clock=clocks[1], white_inc=inc, black_inc=inc)
        t = time.perf_counter()
        res = eng.play(board, limit, game=game_key, info=chess.engine.INFO_SCORE)
        used = time.perf_counter() - t
        if not nodes and not movetime:
            clocks[side] -= used
            if clocks[side] <= 0:
                game.headers["Termination"] = "time forfeit"
                return ("0-1" if side == 0 else "1-0"), game
            clocks[side] += inc
        node = node.add_variation(res.move)
        score = res.info.get("score")
        if score is not None:
            node.comment = f"{score.white()} {used:.2f}s"
        board.push(res.move)
    result = board.result(claim_draw=True)
    if result == "*":
        result = "1/2-1/2"
    return result, game


def elo_diff(score):
    score = min(max(score, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / score - 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine1", nargs="+", required=True)
    p.add_argument("--engine2", nargs="+", required=True)
    p.add_argument("--name1", default="engine1")
    p.add_argument("--name2", default="engine2")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--tc", default="10+0.1", help="seconds+increment per game")
    p.add_argument("--nodes", type=int, default=0)
    p.add_argument("--movetime", type=float, default=0)
    p.add_argument("--pgn", default=None)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    base, _, inc = args.tc.partition("+")
    tc = (float(base), float(inc or 0))
    e1 = open_engine(*parse_engine(args.engine1))
    e2 = open_engine(*parse_engine(args.engine2))
    openings = OPENINGS[:]
    random.Random(args.seed).shuffle(openings)
    w = d = l = 0
    pgn_out = open(args.pgn, "w", encoding="utf-8") if args.pgn else None
    try:
        for g in range(args.games):
            opening = openings[(g // 2) % len(openings)]
            first_white = g % 2 == 0
            engines = (e1, e2) if first_white else (e2, e1)
            names = (args.name1, args.name2) if first_white else (args.name2, args.name1)
            result, game = play_game(engines, names, opening, tc, args.nodes, args.movetime)
            game.headers["Result"] = result
            game.headers["Round"] = str(g + 1)
            if pgn_out:
                print(game, file=pgn_out, end="\n\n", flush=True)
            if result == "1/2-1/2":
                d += 1
            elif (result == "1-0") == first_white:
                w += 1
            else:
                l += 1
            n = w + d + l
            s = (w + 0.5 * d) / n
            print(f"game {n}: {names[0]} vs {names[1]} {result} | {args.name1} +{w} ={d} -{l} "
                  f"score {s:.3f} Elo {elo_diff(s):+.0f}", flush=True)
    finally:
        e1.quit()
        e2.quit()
        if pgn_out:
            pgn_out.close()
    n = w + d + l
    s = (w + 0.5 * d) / n
    # 95% interval from the per-game score variance.
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * (0 - s) ** 2) / max(1, n - 1)
    half = 1.96 * math.sqrt(var / n)
    lo, hi = elo_diff(max(1e-3, s - half)), elo_diff(min(1 - 1e-3, s + half))
    print(f"\n{args.name1} vs {args.name2}: +{w} ={d} -{l}, score {s:.3f}, Elo {elo_diff(s):+.0f} [{lo:+.0f}, {hi:+.0f}]")


if __name__ == "__main__":
    main()
