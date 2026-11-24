"""Scene generation for chess_dance_trace (v3, 10-question ladder).

3-9 coloured chess pieces (queen / king / rook / bishop / knight; types may
repeat, every colour is used once) share a 6x6, 7x7 or 8x8 board and move in
rounds: in each round 1-3 pieces glide to new squares at the same time, the
rest stay put, then the board rests before the next round. Five scenarios:

  dance    1-3 pieces per round, every move legal.
  rogue    like dance, but exactly one move (in a middle round) breaks the
           rules: the wrong pattern for its type or a slide through a piece.
  relay    one piece per round, faster rounds, more of them.
  capture  1-2 pieces per round; 2-3 moves land on a standing piece, which
           disappears the moment the mover lands (colour-blind captures).
  holes    3-9 squares are cut out of the board; nothing may land on a hole
           and slides cannot cross one (knights jump over them).

A move is legal when it follows the piece's movement pattern, lands on a
square that is free when the round begins (or, in `capture`, on a piece that
stands still that round) and does not slide through a square that is
occupied (or a hole) when the round begins; knights jump. Movers in one
round never share a destination or cross paths and stay clear of each other
on screen.

Everything is in frames (24 fps): the timing dict gives the intro, glide,
pause and closing hold; round r glides over frames [start_r, land_r].
python-chess is the geometric oracle here (Board.attacks on an 8x8 board,
with pawns standing in for pieces, holes and the squares outside a smaller
board); answers.py re-derives everything with its own geometry from the
renderer's per-frame track.
"""
from __future__ import annotations
import random

import chess

FPS = 24
W, H = 1280, 720
MAX_FRAMES = 240                     # 10.0 s
TYPES = "QKRBN"
NAME = {"Q": "queen", "K": "king", "R": "rook", "B": "bishop", "N": "knight"}
PLURAL = {"Q": "queens", "K": "kings", "R": "rooks", "B": "bishops", "N": "knights"}
PT = {"Q": chess.QUEEN, "K": chess.KING, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT}
FILES = "ABCDEFGH"
SCENARIOS = ("dance", "rogue", "relay", "capture", "holes")

# Saturated piece colours, pairwise >= 90 apart in RGB and far from every
# board / background colour, so they survive downscaling and blob detection.
COLORS = [("red", (214, 40, 40)), ("orange", (245, 124, 0)), ("yellow", (245, 206, 26)),
          ("lime", (150, 214, 40)), ("green", (30, 150, 60)), ("teal", (0, 160, 150)),
          ("sky-blue", (40, 190, 240)), ("blue", (37, 99, 235)), ("purple", (140, 60, 215)),
          ("magenta", (215, 40, 180)), ("pink", (255, 135, 195))]
BOARDS = [((240, 217, 181), (181, 136, 99)), ((222, 227, 230), (140, 154, 168)),
          ((238, 232, 213), (196, 180, 150)), ((214, 214, 210), (150, 150, 146)),
          ((214, 222, 204), (150, 163, 140)), ((205, 205, 218), (118, 118, 140))]
BACKS = [((245, 244, 240), False), ((232, 236, 240), False), ((240, 236, 228), False),
         ((226, 230, 226), False), ((36, 38, 44), True), ((28, 32, 46), True), ((46, 42, 40), True)]
BG_PATTERNS = ("plain", "dots", "stripes", "gradient")
PIECE_STYLES = ("disc", "tile", "figure")
LABEL_SIDES = ("LB", "RB", "LRB")    # rank numbers left / right / both, file letters below
LABEL_FONT = 34                       # cap height 25 px
LABEL_GAP = 28                        # label centre distance from the board edge (px)
TIMER_BOX = (20, 16, 200, 72)
TIMER_FONT = 36
SEP = 0.95        # moving pieces keep at least this many cells apart at every moment
BRUSH = 0.75      # sliding pieces keep this far (cells) from pieces standing still

SCEN = {
    "dance": dict(n=(4, 9), movers=(1, 3), rounds=(4, 6), move_f=(14, 22), pause_f=(9, 14)),
    "rogue": dict(n=(4, 9), movers=(1, 3), rounds=(4, 6), move_f=(14, 22), pause_f=(9, 14)),
    "relay": dict(n=(3, 7), movers=(1, 1), rounds=(6, 9), move_f=(12, 17), pause_f=(9, 12)),
    "capture": dict(n=(5, 9), movers=(1, 2), rounds=(4, 6), move_f=(14, 22), pause_f=(9, 14)),
    "holes": dict(n=(3, 8), movers=(1, 3), rounds=(4, 6), move_f=(14, 22), pause_f=(9, 14)),
}

STATS = {"tries": 0, "builds": 0}      # retry accounting (reporting only, never decides anything)


# ---------------------------------------------------------------- squares

def name(s) -> str:
    return FILES[s[0]] + str(s[1] + 1)


def sq(n: str):
    return FILES.index(n[0]), int(n[1:]) - 1


def label(scene, i) -> str:
    p = scene["pieces"][i]
    return f"{p['color']} {NAME[p['type']]}"


def _c(s) -> int:
    return chess.square(s[0], s[1])


def _uc(c: int):
    return chess.square_file(c), chess.square_rank(c)


def reach(t: str, o, blockers, N: int, holes=()) -> set:
    """Squares a `t` on `o` attacks on an N x N board: slides stop at the
    first square in `blockers` or `holes` (blockers are included, holes are
    not), squares outside the board and holes are never returned."""
    b = chess.Board.empty()
    b.set_piece_at(_c(o), chess.Piece(PT[t], chess.WHITE))
    walls = {s for s in blockers if s != o} | set(holes)
    walls |= {(f, r) for f in range(8) for r in range(8) if f >= N or r >= N}
    for s in walls:
        b.set_piece_at(_c(s), chess.Piece(chess.PAWN, chess.BLACK))
    out = {_uc(c) for c in b.attacks(_c(o))}
    return {s for s in out if s[0] < N and s[1] < N and s not in set(holes)}


def between(a, b) -> list:
    return sorted(_uc(c) for c in chess.SquareSet(chess.between(_c(a), _c(b))))


def cheb(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def classify(t, a, b, occupied, N, holes=(), capture_ok=False) -> str | None:
    """None if the move a->b is legal for a `t`, else why not:
    'occupied', 'pattern' (not a move of this type on an empty board) or
    'blocked' (slides through a piece or a hole)."""
    if b in set(holes):
        return "hole"
    if b in occupied and not capture_ok:
        return "occupied"
    if b not in reach(t, a, [], N):
        return "pattern"
    if b not in reach(t, a, occupied, N, holes):
        return "blocked"
    return None


# ---------------------------------------------------------------- timing

def round_frames(scene, r: int):
    """(start, land) frames of round r (1-based)."""
    tm = scene["timing"]
    s = tm["intro_f"] + (r - 1) * (tm["move_f"] + tm["pause_f"])
    return s, s + tm["move_f"]


def n_frames(scene) -> int:
    return round_frames(scene, scene["rounds"])[1] + scene["timing"]["hold_f"]


def timer_tenths(i: int) -> int:
    """What the on-screen clock shows at frame i, in tenths of a second."""
    return (10 * i) // FPS


def clock_text(tenths: int) -> str:
    return f"{tenths // 10}.{tenths % 10} s"


def frames_showing(tenths: int):
    """Frames whose clock reads `tenths` (inclusive range)."""
    lo = -(-tenths * FPS // 10)
    hi = -(-(tenths + 1) * FPS // 10) - 1
    return lo, hi


# ---------------------------------------------------------------- replay helpers

def positions(scene) -> list:
    """positions(scene)[r] = squares after round r (None = captured); r = 0 is the start."""
    cur = [sq(p["start"]) for p in scene["pieces"]]
    out = [list(cur)]
    for mv in scene["moves"]:
        for m in mv:
            cur[m["p"]] = sq(m["to"])
            if m.get("capture") is not None:
                cur[m["capture"]] = None
        out.append(list(cur))
    return out


def move_rounds(scene, i) -> list:
    return [r for r, mv in enumerate(scene["moves"], 1) if any(m["p"] == i for m in mv)]


def move_of(scene, i, r):
    return next((m for m in scene["moves"][r - 1] if m["p"] == i), None)


def captured_round(scene, i):
    for r, mv in enumerate(scene["moves"], 1):
        for m in mv:
            if m.get("capture") == i:
                return r
    return None


def attacks(scene, B, i) -> int:
    """How many other pieces piece i attacks on board B (list of squares / None)."""
    t = scene["pieces"][i]["type"]
    occ = [s for j, s in enumerate(B) if j != i and s is not None]
    hit = reach(t, B[i], occ, scene["N"], [sq(h) for h in scene["holes"]])
    return sum(1 for j, s in enumerate(B) if j != i and s is not None and s in hit)


def first_blocked(scene, block_sq, after_round: int, skip_piece=None, knights_jump=True):
    """With an obstacle standing on `block_sq` from round `after_round` on,
    the earliest round in which some move (not by skip_piece) lands on it or
    slides through it, and the movers affected in that round."""
    for r in range(after_round, scene["rounds"] + 1):
        hit = []
        for m in scene["moves"][r - 1]:
            if m["p"] == skip_piece:
                continue
            a, b = sq(m["from"]), sq(m["to"])
            jump = scene["pieces"][m["p"]]["type"] == "N"
            if b == block_sq or (not (jump and knights_jump) and block_sq in between(a, b)):
                hit.append(m["p"])
        if hit:
            return r, sorted(hit)
    return None, []


def never_moved_left(scene, x):
    """Re-simulate the capture scenario with piece x never moving: returns
    (pieces left at the end, ok) where ok is False if any other scheduled
    move would become impossible (blocked by x or by a piece x no longer
    captures)."""
    N, holes = scene["N"], [sq(h) for h in scene["holes"]]
    cur = [sq(p["start"]) for p in scene["pieces"]]
    for mv in scene["moves"]:
        occ = {s for s in cur if s is not None}
        movers = {m["p"] for m in mv}
        lands = []
        for m in mv:
            if m["p"] == x or cur[m["p"]] is None:
                continue
            a, b = sq(m["from"]), sq(m["to"])
            if cur[m["p"]] != a:
                return None, False
            victim = next((j for j, s in enumerate(cur) if s == b), None)
            if victim is not None and (victim in movers and victim != x):
                return None, False
            if classify(scene["pieces"][m["p"]]["type"], a, b, occ, N, holes, capture_ok=True):
                return None, False
            lands.append((m["p"], b, victim))
        dests = [b for _, b, _ in lands]
        if len(set(dests)) != len(dests):
            return None, False
        for p, b, victim in lands:
            if victim is not None:
                cur[victim] = None
            cur[p] = b
    return sum(1 for s in cur if s is not None), True


def fix_next(scene):
    """For the rogue move: if its piece had stayed put instead, the round of
    its next move and what that next move (to the same square) would be:
    None (legal) / 'pattern' / 'blocked' / 'occupied'. None if not askable."""
    ill = scene["illegal"]
    p, r = ill["p"], ill["round"]
    later = [q for q in move_rounds(scene, p) if q > r]
    if not later:
        return None
    r2 = later[0]
    a = sq(move_of(scene, p, r)["from"])
    P = positions(scene)
    # nothing else may land on or slide through a while p would still sit there
    for q in range(r, r2 + 1):
        for m in scene["moves"][q - 1]:
            if m["p"] == p:
                continue
            fa, fb = sq(m["from"]), sq(m["to"])
            jump = scene["pieces"][m["p"]]["type"] == "N"
            if fb == a or (not jump and a in between(fa, fb)):
                return None
    others = {s for j, s in enumerate(P[r2 - 1]) if j != p and s is not None}
    c = sq(move_of(scene, p, r2)["to"])
    if c == a:
        return None
    t = scene["pieces"][p]["type"]
    N = scene["N"]
    occupied = c in others
    pat = c not in reach(t, a, [], N)
    blk = (not pat) and c not in reach(t, a, others, N)
    kinds = [k for k, v in (("occupied", occupied), ("pattern", pat), ("blocked", blk)) if v]
    if len(kinds) > 1:
        return None
    return r2, (kinds[0] if kinds else None)


# ---------------------------------------------------------------- generation

def _illegal_cands(t, o, occupied, mover_origins, N):
    """(dest, kind) pairs that are clearly illegal for a `t` on `o`."""
    f, r = o
    out = []

    def add(df, dr, kind):
        ff, rr = f + df, r + dr
        if 0 <= ff < N and 0 <= rr < N and (ff, rr) not in occupied:
            out.append(((ff, rr), kind))

    diag = [(a, b) for a in (-1, 1) for b in (-1, 1)]
    orth = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if t == "R":
        for a, b in diag:
            for k in (1, 2, 3):
                add(a * k, b * k, "pattern")
    elif t == "B":
        for a, b in orth:
            for k in (1, 2, 3):
                add(a * k, b * k, "pattern")
    elif t == "Q":
        for a, b in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
            add(a, b, "pattern")
    elif t == "K":
        for a, b in orth + diag:
            add(2 * a, 2 * b, "pattern")
    elif t == "N":
        for a, b in orth + diag:
            add(a, b, "pattern")
            add(2 * a, 2 * b, "pattern")
    out = [(d, k) for d, k in out if not (set(between(o, d)) & occupied)]
    if t in "QRB":
        for d in sorted(reach(t, o, [], N)):
            hit = set(between(o, d)) & occupied
            if d not in occupied and hit and not (hit & mover_origins):
                out.append((d, "blocked"))
    return out


def _path(a, b, u, knight, arc):
    x = a[0] + (b[0] - a[0]) * u
    y = a[1] + (b[1] - a[1]) * u
    if knight:
        y += arc * 4 * u * (1 - u)        # same arc shape as the renderer, in cells
    return x, y


def _clear(moves, pos, types, arc):
    movers = {m["p"] for m in moves}
    victims = {m["capture"] for m in moves if m.get("capture") is not None}
    still = [pos[i] for i in range(len(pos)) if i not in movers and i not in victims and pos[i] is not None]
    for k in range(25):
        u = k / 24
        pts = [_path(sq(m["from"]), sq(m["to"]), u, types[m["p"]] == "N", arc) for m in moves]
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                if (pts[a][0] - pts[b][0]) ** 2 + (pts[a][1] - pts[b][1]) ** 2 < SEP ** 2:
                    return False
            m = moves[a]
            if m["illegal"] == "blocked":
                continue          # the rogue slide passes over a piece on purpose
            for s in still:
                if (pts[a][0] - s[0]) ** 2 + (pts[a][1] - s[1]) ** 2 < BRUSH ** 2:
                    return False
    return True


def _round(rng, sc, types, pos, prev, k, nmoved, want_illegal, want_capture, N, holes, arc):
    n = len(types)
    alive = [i for i in range(n) if pos[i] is not None]
    occupied = {pos[i] for i in alive}
    prefer = rng.choice(["pattern", "blocked"])
    for attempt in range(60):
        order = sorted(alive, key=lambda i: (nmoved[i], rng.random()))
        kk = min(k, len(alive))
        movers = sorted(rng.sample(order[:min(len(order), kk + 2)], kk))
        origins = {pos[i] for i in movers}
        ill, cap = None, None
        if want_illegal:
            kinds = {i: {c[1] for c in _illegal_cands(types[i], pos[i], occupied, origins - {pos[i]}, N)}
                     for i in movers}
            able = [i for i in movers if (prefer in kinds[i] if attempt < 40 else kinds[i])]
            if not able:
                continue
            ill = rng.choice(able)
        if want_capture:
            targets = {i: sorted(s for s in reach(types[i], pos[i], occupied, N, holes)
                                 if s in occupied and s not in origins) for i in movers}
            able = [i for i in movers if targets[i]]
            if not able:
                continue
            cap = rng.choice(able)
        used, moves, ok = set(), [], True
        for i in movers:
            o = pos[i]
            victim = None
            if i == ill:
                cands = [c for c in _illegal_cands(types[i], o, occupied, origins - {o}, N)
                         if c[1] == (prefer if prefer in kinds[i] else sorted(kinds[i])[0])]
            elif i == cap:
                cands = [(d, None) for d in targets[i]]
            else:
                cands = [(d, None) for d in sorted(reach(types[i], o, occupied, N, holes)) if d not in occupied]
            cands = [(d, kd) for d, kd in cands if d not in used and d != prev[i]
                     and not (set(between(o, d)) & used)]
            if not cands:
                ok = False
                break
            d, kd = rng.choice(cands)
            if i == cap:
                victim = next(j for j in alive if pos[j] == d)
            used |= {d} | set(between(o, d))
            moves.append({"p": i, "from": name(o), "to": name(d), "illegal": kd, "capture": victim})
        if ok and _clear(moves, pos, types, arc):
            return moves
    return None


def _timing(rng, sc):
    cfg = SCEN[sc]
    for _ in range(50):
        tm = {"intro_f": rng.randint(14, 24), "move_f": rng.randint(*cfg["move_f"]),
              "pause_f": rng.randint(*cfg["pause_f"]), "hold_f": rng.randint(14, 22)}
        rmax = (MAX_FRAMES - tm["intro_f"] - tm["hold_f"] + tm["pause_f"]) // (tm["move_f"] + tm["pause_f"])
        lo, hi = cfg["rounds"]
        if rmax >= lo:
            return tm, rng.randint(lo, min(hi, rmax))
    raise RuntimeError("no timing fits")


def _style(rng, N, sc):
    light, dark = rng.choice(BOARDS)
    bg, is_dark = rng.choice(BACKS)
    style_kind = rng.choice(PIECE_STYLES)
    cell = rng.randint(56, 72) if N == 8 else rng.randint(62, 78) if N == 7 else rng.randint(68, 86)
    side = rng.choice(LABEL_SIDES)
    bw = N * cell
    left_pad = 46 if side in ("LB", "LRB") else 0
    right_pad = 46 if side in ("RB", "LRB") else 0
    x_lo, x_hi = 236 + left_pad, W - 20 - right_pad - bw
    y_lo, y_hi = 24, H - 60 - bw
    return {"light": list(light), "dark": list(dark), "bg": list(bg), "bg_dark": is_dark,
            "bg_pattern": rng.choice(BG_PATTERNS), "edge_w": rng.randint(2, 5),
            "cell": cell, "x0": rng.randint(x_lo, max(x_lo, x_hi)), "y0": rng.randint(y_lo, max(y_lo, y_hi)),
            "piece_style": style_kind,
            "piece_scale": round(rng.uniform(0.9, 0.98) if style_kind == "figure" else rng.uniform(0.72, 0.84), 3), "arc": round(rng.uniform(0.35, 0.6), 3),
            "label_sides": side, "shadow": rng.random() < 0.5,
            "ink": [235, 235, 240] if is_dark else [30, 30, 36]}


def _labels(style, N):
    """Board coordinates: file letters under the board, rank numbers on the
    chosen side(s). Centre positions in pixels."""
    c, x0, y0 = style["cell"], style["x0"], style["y0"]
    out = []
    for f in range(N):
        out.append({"text": FILES[f], "x": x0 + (f + 0.5) * c, "y": y0 + N * c + LABEL_GAP, "size": LABEL_FONT})
    for r in range(N):
        y = y0 + (N - 1 - r + 0.5) * c
        if style["label_sides"] in ("LB", "LRB"):
            out.append({"text": str(r + 1), "x": x0 - LABEL_GAP, "y": y, "size": LABEL_FONT})
        if style["label_sides"] in ("RB", "LRB"):
            out.append({"text": str(r + 1), "x": x0 + N * c + LABEL_GAP, "y": y, "size": LABEL_FONT})
    for lb in out:
        lb["x"], lb["y"] = round(lb["x"], 1), round(lb["y"], 1)
    return out


def _holes(rng, N):
    k = rng.randint(3, 9 if N == 8 else 7)
    allsq = [(f, r) for f in range(N) for r in range(N)]
    for _ in range(50):
        hs = sorted(rng.sample(allsq, k))
        # no 2x2 block of holes, no hole touching three others (keeps the board readable)
        hset = set(hs)
        if any(sum((h[0] + a, h[1] + b) in hset for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b) > 2 for h in hs):
            continue
        return hs
    return sorted(rng.sample(allsq, 3))


def _try(rng, sc):
    STATS["tries"] += 1
    cfg = SCEN[sc]
    N = rng.choice([6, 7, 8, 8])
    n = rng.randint(*cfg["n"])
    if N == 6:
        n = min(n, 7)
    types = [rng.choice("QKRRBBNN") for _ in range(n)]
    if types.count("Q") > 2:
        return None
    cols = rng.sample(COLORS, n)
    holes = _holes(rng, N) if sc == "holes" else []
    free = sorted({(f, r) for f in range(N) for r in range(N)} - set(holes))
    pos = rng.sample(free, n)
    style = _style(rng, N, sc)
    tm, R = _timing(rng, sc)
    pieces = [{"type": t, "color": c, "rgb": list(rgb), "start": name(s)} for t, (c, rgb), s in zip(types, cols, pos)]
    ill_round = rng.randint(2, R - 1) if sc == "rogue" else None
    cap_rounds = set()
    if sc == "capture":
        cap_rounds = set(rng.sample(range(1, R + 1), rng.randint(2, min(3, R - 1))))
    nmoved, rounds, illegal = [0] * n, [], None
    prev = [None] * n
    pos = list(pos)
    for r in range(1, R + 1):
        lo, hi = cfg["movers"]
        k = rng.randint(lo, hi)
        mv = _round(rng, sc, types, pos, prev, k, nmoved, r == ill_round, r in cap_rounds, N, holes, style["arc"])
        if mv is None:
            return None
        for m in mv:
            prev[m["p"]] = pos[m["p"]]
            pos[m["p"]] = sq(m["to"])
            nmoved[m["p"]] += 1
            if m["capture"] is not None:
                pos[m["capture"]] = None
            if m["illegal"]:
                illegal = {"round": r, "p": m["p"], "kind": m["illegal"]}
        rounds.append(mv)
    if sc == "capture" and sum(1 for s in pos if s is not None) < 3:
        return None
    scene = {"scenario": sc, "N": N, "n": n, "rounds": R, "pieces": pieces, "holes": [name(h) for h in holes],
             "moves": rounds, "illegal": illegal, "timing": tm, "style": style, "labels": _labels(style, N),
             "timer": {"box": list(TIMER_BOX), "size": TIMER_FONT}}
    scene["n_frames"] = n_frames(scene)
    scene["duration"] = round(scene["n_frames"] / FPS, 6)
    return scene


# ---------------------------------------------------------------- askable questions

def viable(scene) -> dict:
    """Every (derivation_key, params) this scene supports, per level. Pure."""
    n, R, sc = scene["n"], scene["rounds"], scene["scenario"]
    P = positions(scene)
    mr = {i: move_rounds(scene, i) for i in range(n)}
    cnt = {i: len(mr[i]) for i in range(n)}
    types = [p["type"] for p in scene["pieces"]]
    holes = [sq(h) for h in scene["holes"]]
    capr = {i: captured_round(scene, i) for i in range(n)}
    v = {1: [], 2: [], 3: [], 4: [], 5: []}

    # L1 observation
    v[1].append(("n_pieces_start", {}))
    for t in sorted(set(types)):
        v[1].append(("n_type", {"type": t}))
    for i in range(n):
        if cnt[i] and P[R][i] is not None and P[R][i] != P[0][i]:
            v[1].append(("start_square", {"x": i}))
            v[1].append(("end_square", {"x": i}))
        if types.count(types[i]) == 1:
            v[1].append(("color_of_type", {"type": types[i]}))
    if sc == "holes":
        v[1].append(("n_holes", {}))
    if sc == "capture":
        v[1].append(("n_pieces_end", {}))

    # L2 relational
    first = {i: (mr[i][0] if mr[i] else 99) for i in range(n)}
    for i in range(n):
        later = [j for j in range(n) if first[j] > first[i]]
        if first[i] < 99 and len(later) >= 3:
            v[2].append(("first_mover", {"x": i}))
    if sc != "relay":
        for i in range(n):
            mates = sorted({m["p"] for r in mr[i] for m in scene["moves"][r - 1] if m["p"] != i})
            lonely = [j for j in range(n) if j != i and j not in mates and cnt[j]
                      and (capr[j] is None)]
            if len(mates) == 1 and len(lonely) >= 3:
                v[2].append(("moved_with", {"x": i}))
    for a in range(n):
        for b in range(a + 1, n):
            if cnt[a] and cnt[b] and cnt[a] != cnt[b]:
                v[2].append(("more_moves", {"pair": [a, b]}))
            if cnt[a] and cnt[b] and first[a] != first[b]:
                v[2].append(("moved_before", {"pair": [a, b]}))
    if sc == "capture":
        for i in range(n):
            if capr[i] is not None:
                v[2].append(("captured_by", {"x": i}))

    # L3 measurement
    for i in range(n):
        for k, r in enumerate(mr[i], 1):
            m = move_of(scene, i, r)
            if types[i] in "QRB" and m["illegal"] != "pattern":
                v[3].append(("slide_len", {"x": i, "k": k}))
            v[3].append(("land_time", {"x": i, "k": k}))
        if cnt[i] >= 1 and (capr[i] is None):
            v[3].append(("move_count", {"x": i}))
        for tenths in square_times(scene, i):
            v[3].append(("square_at_time", {"x": i, "t": tenths}))
    for tenths in moving_times(scene):
        v[3].append(("n_moving_at", {"t": tenths}))
    if sc == "capture":
        v[3].append(("n_captures", {}))

    # L4 integration
    v[4].append(("illegal_piece", {}))
    if scene["illegal"]:
        v[4].append(("illegal_time", {}))
    v[4].append(("total_moves", {}))
    top = max(cnt.values())
    if top >= 2 and list(cnt.values()).count(top) == 1:
        v[4].append(("most_moves", {}))
    for i in range(n):
        seen = {P[r][i] for r in range(R + 1) if P[r][i] is not None}
        if len(seen) >= 3:
            v[4].append(("never_on", {"x": i}))
        ms = [move_of(scene, i, r) for r in mr[i]]
        if types[i] in "QRB" and len(ms) >= 2 and not any(m["illegal"] == "pattern" for m in ms):
            v[4].append(("total_slide", {"x": i}))
        if len(ms) >= 2 and P[R][i] is not None and cheb(P[0][i], P[R][i]) >= 1:
            v[4].append(("end_distance", {"x": i}))
    if sc == "capture":
        caps = [sum(1 for mv in scene["moves"] for m in mv if m["p"] == i and m["capture"] is not None)
                for i in range(n)]
        if max(caps) >= 1 and caps.count(max(caps)) == 1 and max(caps) >= 2:
            v[4].append(("most_captures", {}))
        v[4].append(("first_captured", {}))

    # L5 counterfactual
    for i in range(n):
        for k, r in enumerate(mr[i], 1):
            m = move_of(scene, i, r)
            if m["capture"] is not None or m["illegal"]:
                continue
            Q = list(P[r])
            Q[i] = P[r - 1][i]
            if attacks(scene, Q, i) != attacks(scene, P[r], i):
                v[5].append(("cf_skip_attacks", {"x": i, "k": k}))
    if sc in ("dance", "relay", "holes", "rogue"):
        for i in range(n):
            if not cnt[i]:
                continue
            r, hit = first_blocked(scene, P[0][i], mr[i][0], skip_piece=i)
            if r is None or len(hit) == 1:
                v[5].append(("cf_never_moved_block", {"x": i}))
    if sc == "capture":
        for i in range(n):
            if cnt[i] and capr[i] is None:
                left, ok = never_moved_left(scene, i)
                if ok:
                    v[5].append(("cf_never_moved_left", {"x": i}))
    if sc == "holes":
        for f in range(scene["N"]):
            for rr in range(scene["N"]):
                s = (f, rr)
                if s in holes or s in P[0]:
                    continue
                r, hit = first_blocked(scene, s, 1, knights_jump=True)
                if r is not None and len(hit) == 1:
                    v[5].append(("cf_hole", {"sq": name(s)}))
    if scene["illegal"] and fix_next(scene):
        v[5].append(("cf_fix_next", {}))
    return v


def rests(scene, i):
    """Rest intervals of piece i: (first frame, last frame, square), while visible."""
    out = []
    cur, f0 = sq(scene["pieces"][i]["start"]), 0
    end = scene["n_frames"] - 1
    capr = captured_round(scene, i)
    for r in move_rounds(scene, i):
        s, land = round_frames(scene, r)
        out.append((f0, s, cur))
        cur, f0 = sq(move_of(scene, i, r)["to"]), land
    if capr is not None:
        end = round_frames(scene, capr)[1] - 1
    out.append((f0, end, cur))
    return out


def square_times(scene, i) -> list:
    """Clock readings (tenths) at which piece i rests on a square that is
    neither its start nor its end square, with every frame showing that
    reading at least 2 frames inside the rest."""
    P = positions(scene)
    R = scene["rounds"]
    out = []
    covered = [round_frames(scene, scene["illegal"]["round"])] if scene["illegal"] else []
    for a, b, s in rests(scene, i):
        if s == P[0][i] or s == P[R][i]:
            continue
        cands = []
        for tenths in range(timer_tenths(a) + 1, timer_tenths(b) + 1):
            lo, hi = frames_showing(tenths)
            if lo >= a + 2 and hi <= b - 2 and not any(g0 - 2 <= hi and lo <= g1 + 2 for g0, g1 in covered):
                cands.append(tenths)
        if cands:
            out.append(cands[len(cands) // 2])
    return out


def moving_times(scene) -> list:
    """One clock reading per round whose frames all lie >= 2 frames inside the glide."""
    out = []
    for r in range(1, scene["rounds"] + 1):
        s, land = round_frames(scene, r)
        cands = []
        for tenths in range(timer_tenths(s), timer_tenths(land) + 1):
            lo, hi = frames_showing(tenths)
            if lo >= s + 2 and hi <= land - 2:
                cands.append(tenths)
        if cands:
            out.append(cands[len(cands) // 2])
    return out


# ---------------------------------------------------------------- checks

def _rgb_dist(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def label_boxes(scene) -> list:
    """Approximate pixel boxes of the coordinate labels (font 34: cap height
    25 px, glyph width <= 26 px)."""
    out = []
    for lb in scene["labels"]:
        w = 26
        out.append((lb["x"] - w / 2 - 2, lb["y"] - 15, lb["x"] + w / 2 + 2, lb["y"] + 15))
    return out


def piece_extent_boxes(scene) -> list:
    """Conservative pixel boxes swept by every piece over the whole clip."""
    st = scene["style"]
    c, x0, y0, N = st["cell"], st["x0"], st["y0"], scene["N"]
    half = st["piece_scale"] * c / 2 + 2

    def centre(s):
        return x0 + (s[0] + 0.5) * c, y0 + (N - 1 - s[1] + 0.5) * c

    out = []
    for i, p in enumerate(scene["pieces"]):
        out.append(centre(sq(p["start"])) + (0.0,))
    for mv in scene["moves"]:
        for m in mv:
            a, b = centre(sq(m["from"])), centre(sq(m["to"]))
            lift = st["arc"] * c if scene["pieces"][m["p"]]["type"] == "N" else 0.0
            out.append((min(a[0], b[0]), min(a[1], b[1]), lift, max(a[0], b[0]), max(a[1], b[1])))
    boxes = []
    for e in out:
        if len(e) == 3:
            boxes.append((e[0] - half, e[1] - half, e[0] + half, e[1] + half))
        else:
            boxes.append((e[0] - half, e[1] - half - e[2], e[3] + half, e[4] + half))
    return boxes


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def check(scene) -> list:
    errs = []
    n, R, N, sc = scene["n"], scene["rounds"], scene["N"], scene["scenario"]
    cfg = SCEN[sc]
    if not (cfg["n"][0] <= n <= cfg["n"][1] and cfg["rounds"][0] <= R <= cfg["rounds"][1]):
        errs.append("piece or round count out of range")
    if len({p["color"] for p in scene["pieces"]}) != n:
        errs.append("piece colours are not unique")
    rgbs = [p["rgb"] for p in scene["pieces"]]
    st = scene["style"]
    for a in range(n):
        for b in range(a + 1, n):
            if _rgb_dist(rgbs[a], rgbs[b]) < 85:
                errs.append("two piece colours too close")
        for other in (st["light"], st["dark"], st["bg"]):
            if _rgb_dist(rgbs[a], other) < 85:
                errs.append("a piece colour is too close to the board or background")
    # timing: every event >= 0.5 s from both ends; clip <= 10 s
    if scene["n_frames"] > MAX_FRAMES or scene["n_frames"] != n_frames(scene):
        errs.append("clip longer than 10 s or frame count mismatch")
    if round_frames(scene, 1)[0] < 12 or scene["n_frames"] - round_frames(scene, R)[1] < 12:
        errs.append("an event is within 0.5 s of the start or the end")
    if scene["timing"]["pause_f"] < 2 * 2 or scene["timing"]["move_f"] < 4:
        errs.append("rounds too close together")
    # labels: inside the frame, clear of the board, the pieces, the timer and each other
    board = (st["x0"] - st["edge_w"], st["y0"] - st["edge_w"], st["x0"] + N * st["cell"] + st["edge_w"],
             st["y0"] + N * st["cell"] + st["edge_w"])
    lbs = label_boxes(scene)
    tb = scene["timer"]["box"]
    for k, a in enumerate(lbs):
        if a[0] < 0 or a[1] < 0 or a[2] > W or a[3] > H:
            errs.append("label outside the frame")
        if _overlap(a, board) or _overlap(a, tb):
            errs.append("label overlaps the board or the timer")
        for b in lbs[k + 1:]:
            if _overlap(a, b):
                errs.append("labels overlap")
    if _overlap(board, tb) or board[2] > W or board[3] > H:
        errs.append("board overlaps the timer or leaves the frame")
    for pb in piece_extent_boxes(scene):
        if _overlap(pb, tb) or any(_overlap(pb, a) for a in lbs):
            errs.append("a piece passes over a label or the timer")
            break
    # moves
    holes = [sq(h) for h in scene["holes"]]
    P = positions(scene)
    if len({s for s in P[0]}) != n or set(P[0]) & set(holes):
        errs.append("bad start squares")
    bad = []
    for r, mv in enumerate(scene["moves"], 1):
        lo, hi = cfg["movers"]
        if not lo <= len(mv) <= hi:
            errs.append(f"round {r}: {len(mv)} movers")
        occ = {s for s in P[r - 1] if s is not None}
        movers = {m["p"] for m in mv}
        used = set()
        for m in mv:
            a, b = sq(m["from"]), sq(m["to"])
            t = scene["pieces"][m["p"]]["type"]
            if P[r - 1][m["p"]] != a:
                errs.append(f"round {r}: piece {m['p']} doesn't start where it stood")
            victim = next((j for j, s in enumerate(P[r - 1]) if s == b), None)
            if (victim is not None) != (m["capture"] is not None) or (victim is not None and victim != m["capture"]):
                errs.append(f"round {r}: capture flag mismatch")
            if victim is not None and (sc != "capture" or victim in movers):
                errs.append(f"round {r}: lands on an occupied square")
            claim = {b} | set(between(a, b))
            if claim & used:
                errs.append(f"round {r}: movers' paths collide")
            used |= claim
            why = classify(t, a, b, occ, N, holes, capture_ok=sc == "capture")
            if (why is None) != (m["illegal"] is None) or (why and why != m["illegal"]):
                errs.append(f"round {r}: move {m['from']}-{m['to']} legality {why} doesn't match its flag")
            if m["illegal"]:
                bad.append((r, m["p"], m["illegal"]))
        live = [s for s in P[r] if s is not None]
        if len(set(live)) != len(live):
            errs.append(f"round {r}: two pieces share a square")
        if not _clear(mv, P[r - 1], [p["type"] for p in scene["pieces"]], st["arc"]):
            errs.append(f"round {r}: movers come too close on screen")
    want = [(scene["illegal"]["round"], scene["illegal"]["p"], scene["illegal"]["kind"])] if scene["illegal"] else []
    if bad != want or (sc == "rogue") != bool(want):
        errs.append(f"illegal moves {bad} != stored {want}")
    if sc == "capture":
        nc = sum(1 for mv in scene["moves"] for m in mv if m["capture"] is not None)
        if not 2 <= nc <= 3:
            errs.append("capture scenario needs 2-3 captures")
    import questions                      # lazy: questions imports sim
    C = questions.candidates(scene, random.Random(0))
    for lv in range(1, 6):
        if len(C[lv]) < 3 or len({c[0] for c in C[lv]}) < 2:
            errs.append(f"level {lv}: too few askable questions")
    return errs


def build(rng: random.Random) -> dict:
    sc = rng.choice(SCENARIOS)
    STATS["builds"] += 1
    while True:
        s = _try(rng, sc)
        if s is not None and not check(s):
            return s
