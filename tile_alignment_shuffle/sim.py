"""Scene generation for tile_alignment_shuffle (v3, 10-question ladder).

An N x N grid (N = 3-6) of tiles. Each tile shows ONE thing: a solid
coloured shape or a coloured symbol. There are N kinds of tile, each a
distinct shape/symbol in a distinct colour, with N copies each. About half
the clips also have 1-2 "false" tiles: the shape or symbol of a real kind in
a colour no other tile has, so they match nothing else (each one takes the
place of one copy of the kind it imitates).

The tiles are rearranged by S moves, one at a time. Five scenarios decide
what one move is:

    swap       two tiles anywhere on the grid trade places
    neighbour  two side-by-side tiles (same row or column, adjacent) trade places
    cycle      three tiles rotate places (A -> B -> C -> A) at once
    slide      a whole row or column slides one cell; the tile pushed off
               the end comes back in at the other end
    double     two separate pairs of tiles trade places at the same time

Every moving tile looks different from the tiles it moves with (a slide
only needs a visible, direction-readable change), so each move can be read
from the boards before and after it.

Lines. A complete line is a full row, column or diagonal of one kind. The
move list is built backward from a chosen final board:
  * ~75% of clips end with exactly one complete line. It first appears
    after move f, where f is the last move about a fifth of the time and
    otherwise 40-90% of the way through; later moves never touch its cells,
    so it stays put. No other board ever shows a complete line.
  * the other clips never show a complete line at all.
In some false-tile clips the final board has a near miss: a line of one
kind except for the false tile.

Timing is in frames (24 fps): `intro` frames on the start board, then per
move `move` frames of motion and `pause` frames of rest, then `hold` frames
on the final board; the whole clip is at most 240 frames (10 s).

Cells are (row, col), 0-based internally; questions say "row r, column c"
1-based, rows top to bottom, columns left to right. Geometry for the move
paths is in cell units (a cell centre is at its (row, col)).
"""
from __future__ import annotations
import math
import random

FPS = 24
MAX_FRAMES = 240
SCENARIOS = ("swap", "neighbour", "cycle", "slide", "double")

SHAPES = ["circle", "square", "triangle", "star", "heart", "hexagon", "diamond", "cross", "crescent",
          "pentagon"]
SYMBOLS = ["@", "&", "#", "?", "%", "$", "Ω", "Σ"]
# CIE76 dE between any two colours >= 40, so they stay apart after downscaling.
COLORS = {"red": (220, 30, 40), "blue": (20, 100, 235), "green": (20, 145, 50), "orange": (250, 130, 0),
          "purple": (120, 30, 150), "teal": (0, 170, 175), "pink": (245, 105, 200), "brown": (120, 70, 30),
          "yellow": (240, 205, 0), "black": (25, 25, 25)}
N_RANGE = {"swap": (3, 6), "neighbour": (3, 6), "cycle": (3, 6), "slide": (3, 6), "double": (4, 6)}
MOVES = {"swap": {3: (5, 8), 4: (6, 9), 5: (7, 10), 6: (8, 11)},
         "neighbour": {3: (6, 9), 4: (7, 10), 5: (8, 11), 6: (8, 11)},
         "cycle": {3: (4, 7), 4: (5, 8), 5: (5, 9), 6: (6, 9)},
         "slide": {3: (4, 6), 4: (4, 7), 5: (5, 8), 6: (5, 8)},
         "double": {4: (4, 7), 5: (5, 8), 6: (5, 9)}}
LINE_P = 0.75
LAST_P = 0.2              # chance the line forms on the very last move (~1 in 4 of 4 options)
FALSE_P = 0.55
NEAR_P = 0.5
SEP = 0.85                # moving tiles' centres stay >= SEP cells apart (tiles are ~0.9 cell wide)
BOW_TOTAL = 1.05          # a swapping pair passes each other this far apart (cells)

TIMER_BOX = (16, 14, 196, 66)
TIMER_SIZE = 36
LABEL_SIZE = 36           # row/column numbers: ~26 px tall digits
LABEL_GAP = 34            # label centre to board edge (px)

THEMES = {
    "paper": {"bg": (241, 239, 232), "slot": (222, 217, 204), "cell": (200, 194, 178), "ink": (40, 40, 40)},
    "slate": {"bg": (46, 52, 64), "slot": (72, 80, 96), "cell": (100, 110, 128), "ink": (238, 238, 238)},
    "sky": {"bg": (218, 231, 244), "slot": (192, 210, 230), "cell": (165, 186, 210), "ink": (25, 40, 70)},
    "sage": {"bg": (224, 235, 220), "slot": (198, 215, 193), "cell": (168, 190, 162), "ink": (30, 55, 35)},
    "sand": {"bg": (238, 226, 200), "slot": (220, 204, 172), "cell": (196, 178, 142), "ink": (70, 45, 20)},
    "night": {"bg": (22, 24, 32), "slot": (50, 54, 68), "cell": (80, 86, 104), "ink": (242, 242, 242)},
    "wood": {"bg": (150, 105, 66), "slot": (120, 82, 50), "cell": (98, 66, 40), "ink": (255, 248, 235)},
}
CARDS = {"ivory": ((252, 251, 246), (55, 55, 55)), "white": ((255, 255, 255), (30, 30, 30)),
         "cream": ((248, 243, 228), (80, 70, 55))}


# --------------------------------------------------------------------------- names and timing


def design_name(d: dict) -> str:
    return f"{d['color']} {d['kind']}" if d["kind"] in SHAPES else f"{d['color']} {d['kind']} symbol"


def cell_name(rc) -> str:
    return f"row {rc[0] + 1}, column {rc[1] + 1}"


def timer_text(f: int) -> str:
    t = (f * 10) // FPS
    return f"{t // 10}.{t % 10} s"


def move_start(scene, k: int) -> int:
    """First frame of move k (0-based); the tiles are still at rest on it."""
    tm = scene["timing"]
    return tm["intro"] + k * (tm["move"] + tm["pause"])


def move_end(scene, k: int) -> int:
    """First frame on which move k's tiles rest in their new cells."""
    return move_start(scene, k) + scene["timing"]["move"]


def stable_reading(f: int) -> bool:
    """The clock shows the same reading one frame before and after f."""
    return timer_text(f - 1) == timer_text(f) == timer_text(f + 1)


# --------------------------------------------------------------------------- boards


def lines(n: int) -> list:
    out = [("row", [(r, c) for c in range(n)]) for r in range(n)]
    out += [("column", [(r, c) for r in range(n)]) for c in range(n)]
    out += [("diagonal", [(i, i) for i in range(n)]), ("diagonal", [(i, n - 1 - i) for i in range(n)])]
    return out


def line_name(kind: str, cells) -> str:
    if kind == "row":
        return f"row {cells[0][0] + 1}"
    if kind == "column":
        return f"column {cells[0][1] + 1}"
    return "the diagonal from top-left to bottom-right" if tuple(cells[0]) == (0, 0) else \
        "the diagonal from top-right to bottom-left"


def alignments(tiles, grid) -> list:
    """Complete lines whose tiles all share one design: [(kind, cells, design)]."""
    out = []
    for kind, cells in lines(len(grid)):
        ds = {tiles[grid[r][c]] for r, c in cells}
        if len(ds) == 1:
            out.append((kind, cells, next(iter(ds))))
    return out


def apply(grid, move) -> list:
    g = [row[:] for row in grid]
    for (r1, c1), (r2, c2) in move["pairs"]:
        g[r2][c2] = grid[r1][c1]
    return g


def unapply(grid, move) -> list:
    g = [row[:] for row in grid]
    for (r1, c1), (r2, c2) in move["pairs"]:
        g[r1][c1] = grid[r2][c2]
    return g


def grids(scene) -> list:
    """Board (tile ids) at the start and after every move."""
    out = [[row[:] for row in scene["start"]]]
    for m in scene["moves"]:
        out.append(apply(out[-1], m))
    return out


def replay(scene, moves) -> list:
    g = [row[:] for row in scene["start"]]
    for m in moves:
        g = apply(g, m)
    return g


def where(grid, tid):
    for r, row in enumerate(grid):
        for c, t in enumerate(row):
            if t == tid:
                return (r, c)


def movers(grid, move) -> list:
    return [grid[r][c] for (r, c), _ in move["pairs"]]


def move_counts(scene) -> dict:
    gs = grids(scene)
    out = {t: 0 for t in range(len(scene["tiles"]))}
    for k, m in enumerate(scene["moves"]):
        for t in movers(gs[k], m):
            out[t] += 1
    return out


def first_moves(scene) -> dict:
    """tile -> 0-based index of its first move (tiles that never move are absent)."""
    out = {}
    gs = grids(scene)
    for k, m in enumerate(scene["moves"]):
        for t in movers(gs[k], m):
            out.setdefault(t, k)
    return out


def false_tiles(scene) -> list:
    return [t for t, d in enumerate(scene["tiles"]) if scene["designs"][d]["false"]]


# --------------------------------------------------------------------------- move geometry


def path_pos(n, move, i, e):
    """Centre (row, col) in cell units of the tile on move pair i at eased progress e.
    A slide's wrapping tile runs past the edge (the renderer draws its second copy)."""
    (r1, c1), (r2, c2) = move["pairs"][i]
    if move["kind"] == "slide":
        dr, dc = move["shift"]
        return (r1 + dr * e, c1 + dc * e)
    br, bc = move["bow"][i]
    s = bow_profile(e, move["prof"][i] if "prof" in move else "sqrt")
    return (r1 + (r2 - r1) * e + br * s, c1 + (c2 - c1) * e + bc * s)


def bow_profile(e: float, prof: str = "sqrt") -> float:
    """Sideways offset of a swapping tile (0 at both ends, 1 mid-way). "sqrt" opens the bow
    early, so two side-by-side tiles never come closer than about one cell; "sin" opens it
    gently, which lets a tile leave a corner along a diagonal without leaving the board."""
    s = max(0.0, math.sin(math.pi * e))
    return math.sqrt(s) if prof == "sqrt" else s


def _room(n, a, b, dr, dc, prof="sqrt") -> float:
    """Largest bow along (dr, dc) that keeps the tile centre inside the cell-centre box."""
    best = 1e9
    for j in range(1, 20):
        e = j / 20
        s = bow_profile(e, prof)
        for base, d in ((a[0] + (b[0] - a[0]) * e, dr), (a[1] + (b[1] - a[1]) * e, dc)):
            v = d * s
            if v > 1e-9:
                best = min(best, (n - 1 - base) / v)
            elif v < -1e-9:
                best = min(best, -base / v)
    return max(0.0, best)


PATHS = (("sqrt", BOW_TOTAL), ("sin", BOW_TOTAL), ("sin", 0.95), ("sin", 0.9))


def _pair_bows(n, a, b, prof, total):
    """Bows for a and b passing each other on opposite sides, inside the board, or None."""
    dr, dc = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dr, dc)
    pr, pc = -dc / L, dr / L
    ra, rb = _room(n, a, b, pr, pc, prof), _room(n, a, b, -pr, -pc, prof)
    small = min(total / 2, ra, rb)
    big = min(total - small, max(ra, rb))
    if small + big < total - 1e-9:
        return None
    ba, bb = (big, small) if ra >= rb else (small, big)
    return [round(pr * ba, 4), round(pc * ba, 4)], [round(-pr * bb, 4), round(-pc * bb, 4)]


def min_sep(n, move) -> float:
    """Smallest distance between two moving tile centres during the move (cells)."""
    if move["kind"] == "slide":
        return 1.0
    k = len(move["pairs"])
    best = 1e9
    for j in range(0, 41):
        e = j / 40
        ps = [path_pos(n, move, i, e) for i in range(k)]
        for x in range(k):
            for y in range(x + 1, k):
                best = min(best, math.hypot(ps[x][0] - ps[y][0], ps[x][1] - ps[y][1]))
    return best


def _swap_move(n, a, b, kind):
    """The first path style (PATHS) that keeps both tiles on the board and apart."""
    for prof, total in PATHS:
        bows = _pair_bows(n, a, b, prof, total)
        if bows is None:
            continue
        m = {"kind": kind, "pairs": [[list(a), list(b)], [list(b), list(a)]], "bow": [bows[0], bows[1]],
             "prof": [prof, prof]}
        if min_sep(n, m) >= SEP:
            return m
    return None


def propose(rng, scenario, n):
    """A random move of the scenario's kind (geometry only; designs are checked later)."""
    cells = [(r, c) for r in range(n) for c in range(n)]
    if scenario == "swap":
        a, b = rng.sample(cells, 2)
        return _swap_move(n, a, b, "swap")
    if scenario == "neighbour":
        a = rng.choice(cells)
        nb = [(a[0] + dr, a[1] + dc) for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
              if 0 <= a[0] + dr < n and 0 <= a[1] + dc < n]
        return _swap_move(n, a, rng.choice(nb), "swap")
    if scenario == "double":
        a, b, c, d = rng.sample(cells, 4)
        m1, m2 = _swap_move(n, a, b, "double"), _swap_move(n, c, d, "double")
        if m1 is None or m2 is None:
            return None
        m = {"kind": "double", "pairs": m1["pairs"] + m2["pairs"], "bow": m1["bow"] + m2["bow"],
             "prof": m1["prof"] + m2["prof"]}
        return m if min_sep(n, m) >= SEP else None
    if scenario == "cycle":
        a, b, c = rng.sample(cells, 3)
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0:
            return None                                    # collinear: paths would overlap
        m = {"kind": "cycle", "pairs": [[list(a), list(b)], [list(b), list(c)], [list(c), list(a)]],
             "bow": [[0.0, 0.0]] * 3}
        return m if min_sep(n, m) >= SEP else None
    # slide
    i, d = rng.randrange(n), rng.choice((1, -1))
    if rng.random() < 0.5:
        pairs = [[[i, c], [i, (c + d) % n]] for c in range(n)]
        shift = [0, d]
    else:
        pairs = [[[r, i], [(r + d) % n, i]] for r in range(n)]
        shift = [d, 0]
    return {"kind": "slide", "pairs": pairs, "shift": shift}


def move_cells(move) -> set:
    return {tuple(p[0]) for p in move["pairs"]}


def readable(tiles, grid, move) -> bool:
    """The move (acting on `grid`) can be read from the boards before and after it."""
    ds = [tiles[grid[r][c]] for (r, c), _ in move["pairs"]]
    if move["kind"] != "slide":
        return len(set(ds)) == len(ds)
    after = apply(grid, move)
    seq = lambda g: [tiles[g[r][c]] for (r, c), _ in move["pairs"]]  # noqa: E731
    back = {"kind": "slide", "pairs": [[q, p] for p, q in move["pairs"]]}
    return seq(after) != seq(grid) and seq(after) != seq(apply(grid, back))


# --------------------------------------------------------------------------- scene


def _final_board(rng, n):
    kinds = rng.sample(SHAPES, rng.randint(max(1, n - 3), n)) if rng.random() < 0.8 else []
    kinds = (kinds + rng.sample(SYMBOLS, n - len(kinds)))[:n]
    rng.shuffle(kinds)
    cols = rng.sample(list(COLORS), n + 2)
    designs = [{"kind": k, "color": c, "false": False} for k, c in zip(kinds, cols)]
    counts = [n] * n
    n_false = rng.choice([1, 1, 2]) if rng.random() < FALSE_P else 0
    mimic = rng.sample(range(n), n_false)
    for i, m in enumerate(mimic):
        designs.append({"kind": designs[m]["kind"], "color": cols[n + i], "false": True})
        counts[m] -= 1
    all_lines = lines(n)
    final = [[None] * n for _ in range(n)]
    want_line = rng.random() < LINE_P
    used, line = set(), None
    if want_line:
        a = rng.choice([i for i in range(n) if counts[i] >= n])
        line = rng.choice(all_lines)
        for r, c in line[1]:
            final[r][c] = a
        counts[a] -= n
        used |= set(line[1])
    near = None
    if n_false and rng.random() < NEAR_P:
        f = rng.randrange(n_false)
        b = mimic[f]
        opts = [cells for _, cells in all_lines if not used & set(cells)]
        if counts[b] >= n - 1 and opts:
            cells = rng.choice(opts)
            fc = rng.choice(cells)
            for r, c in cells:
                final[r][c] = n + f if (r, c) == fc else b
            counts[b] -= n - 1
            used |= set(cells)
            near = n + f
    pool = [d for d, k in enumerate(counts) for _ in range(k)]
    pool += [n + f for f in range(n_false) if n + f != near]
    rng.shuffle(pool)
    for r in range(n):
        for c in range(n):
            if final[r][c] is None:
                final[r][c] = pool.pop()
    return designs, final, line, near is not None


def _layout(rng, n):
    span = rng.randint(max(360, 76 * n), 600)
    for _ in range(200):
        x0 = rng.randint(60, 1280 - 24 - span)
        y0 = rng.randint(64, 720 - 22 - span)
        labels = label_list(n, x0, y0, span)
        if not layout_errors(n, x0, y0, span, labels):
            return {"x0": x0, "y0": y0, "span": span, "pad": 10}, labels
    return None, None


def label_list(n, x0, y0, span) -> list:
    cell = span / n
    out = [{"text": str(i + 1), "x": round(x0 - LABEL_GAP, 1), "y": round(y0 + cell * (i + 0.5), 1),
            "size": LABEL_SIZE, "role": "row"} for i in range(n)]
    out += [{"text": str(i + 1), "x": round(x0 + cell * (i + 0.5), 1), "y": round(y0 - LABEL_GAP, 1),
             "size": LABEL_SIZE, "role": "column"} for i in range(n)]
    return out


def label_box(lab) -> tuple:
    """Generous box around a 1-digit label (font-independent: 0.7 x 0.8 of the size, plus 3 px)."""
    w, h = 0.35 * lab["size"] + 3, 0.40 * lab["size"] + 3
    return (lab["x"] - w, lab["y"] - h, lab["x"] + w, lab["y"] + h)


def _overlap(a, b, m=0) -> bool:
    return a[0] - m < b[2] and b[0] - m < a[2] and a[1] - m < b[3] and b[1] - m < a[3]


def layout_errors(n, x0, y0, span, labels, pad=10) -> list:
    errs = []
    slot = (x0 - pad, y0 - pad, x0 + span + pad, y0 + span + pad)
    if slot[0] < 0 or slot[1] < 0 or slot[2] > 1280 or slot[3] > 720:
        errs.append("board leaves the frame")
    if _overlap(slot, TIMER_BOX, 8):
        errs.append("board touches the timer")
    boxes = [label_box(lab) for lab in labels]
    for i, b in enumerate(boxes):
        if b[0] < 0 or b[1] < 0 or b[2] > 1280 or b[3] > 720:
            errs.append("label leaves the frame")
        if _overlap(b, slot, 4) or _overlap(b, TIMER_BOX, 8):
            errs.append("label touches the board or the timer")
        if any(_overlap(b, o, 2) for o in boxes[i + 1:]):
            errs.append("labels overlap")
    return errs


def _moves_backward(rng, scenario, n, tiles, final, S, f, line_cells):
    """Moves 1..S (forward order) ending on `final`; boards obey the line rules."""
    want = alignments(tiles, final)
    g = [row[:] for row in final]
    back = []
    for k in range(S, 0, -1):                   # move k turns board k-1 into board k
        for _ in range(300):
            m = propose(rng, scenario, n)
            if m is None:
                continue
            if f is not None and k > f and move_cells(m) & line_cells:
                continue
            prev = unapply(g, m)
            if not readable(tiles, prev, m):
                continue
            al = alignments(tiles, prev)
            if (f is not None and k > f and al != want) or ((f is None or k <= f) and al):
                continue
            back.append(m)
            g = prev
            break
        else:
            return None, None
    return g, back[::-1]


def _try(rng, scenario):
    n = rng.randint(*N_RANGE[scenario])
    designs, final, line, near = _final_board(rng, n)
    # tile ids: one per cell of the final board, numbered in reading order
    tiles = [final[r][c] for r in range(n) for c in range(n)]
    end = [[r * n + c for c in range(n)] for r in range(n)]
    S = rng.randint(*MOVES[scenario][n])
    f = None
    if line:
        if scenario == "slide" and line[0] == "diagonal":
            f = S                               # every row and column crosses a diagonal
        elif rng.random() < LAST_P:
            f = S
        else:
            f = rng.randint(max(1, math.ceil(0.4 * S)), max(1, min(S - 1, math.floor(0.9 * S))))
    start, moves = _moves_backward(rng, scenario, n, tiles, end, S, f, set(line[1]) if line else set())
    if start is None:
        return None
    hold = rng.randint(14, 30)
    pmax = min(25, (MAX_FRAMES - 24 - hold) // S)
    if pmax < 16:
        return None
    period = rng.randint(16, pmax)
    mv = max(10, min(period - 5, round(period * rng.uniform(0.58, 0.72))))
    # intro 0.5-1.0 s; in line clips pick one where the clock reads the same one frame either
    # side of the moment the line completes, so its time is never on a 0.1 s boundary
    intros = list(range(12, 25))
    if f is not None:
        intros = [i for i in intros if stable_reading(i + (f - 1) * period + mv)]
    intro = rng.choice(intros)
    layout, labels = _layout(rng, n)
    if layout is None:
        return None
    return {
        "scenario": scenario, "n": n, "designs": designs, "tiles": tiles, "start": start, "moves": moves,
        "line": line is not None, "line_move": f, "near_miss": near,
        "timing": {"intro": intro, "move": mv, "pause": period - mv, "hold": hold},
        "n_frames": intro + S * period + hold,
        "layout": layout, "labels": labels, "timer": {"box": list(TIMER_BOX), "size": TIMER_SIZE},
        "theme": rng.choice(sorted(THEMES)), "card": rng.choice(sorted(CARDS)),
        "style": {"radius": round(rng.uniform(0.06, 0.2), 3), "gap": round(rng.uniform(0.05, 0.09), 3),
                  "scale": round(rng.uniform(0.27, 0.34), 3), "border": rng.choice([3, 4]),
                  "lift": round(rng.uniform(0.05, 0.1), 3)},
    }


def check(scene) -> list:
    errs = []
    n, T, D = scene["n"], scene["tiles"], scene["designs"]
    sc = scene["scenario"]
    if sc not in SCENARIOS or not N_RANGE[sc][0] <= n <= N_RANGE[sc][1]:
        return ["bad scenario or grid size"]
    gs = grids(scene)
    if sorted(t for row in scene["start"] for t in row) != list(range(n * n)):
        errs.append("start board isn't a permutation of the tiles")
    S = len(scene["moves"])
    if not MOVES[sc][n][0] <= S <= MOVES[sc][n][1]:
        errs.append("move count out of range")
    # line rules: no complete line except the final one, from move f on
    fin = alignments(T, gs[-1])
    f = scene["line_move"]
    if scene["line"]:
        if len(fin) != 1 or f is None or not 1 <= f <= S:
            errs.append("line clip must end with exactly one complete line")
        else:
            for k, g in enumerate(gs):
                al = alignments(T, g)
                if (k < f and al) or (k >= f and al != fin):
                    errs.append(f"board {k} breaks the line rule")
            for m in scene["moves"][f:]:
                if move_cells(m) & set(fin[0][1]):
                    errs.append("a move after the line forms touches the line")
    elif f is not None or any(alignments(T, g) for g in gs):
        errs.append("no-line clip shows a complete line")
    # designs
    if len({d["color"] for d in D}) != len(D):
        errs.append("two designs share a colour")
    real = [d for d in D if not d["false"]]
    if len({d["kind"] for d in real}) != len(real) or len(real) != n:
        errs.append("real kinds are not N distinct shapes/symbols")
    for i, d in enumerate(D):
        k = T.count(i)
        if d["false"] and (k != 1 or d["kind"] not in {x["kind"] for x in real}):
            errs.append("a false tile isn't a unique copy of a real kind's shape")
        if not d["false"] and k < n - 2:
            errs.append("a real kind has too few copies")
    # moves
    kind = {"swap": "swap", "neighbour": "swap"}.get(sc, sc)
    for k, m in enumerate(scene["moves"]):
        src = [tuple(p[0]) for p in m["pairs"]]
        dst = [tuple(p[1]) for p in m["pairs"]]
        if sorted(src) != sorted(dst) or len(set(src)) != len(src) or any(a == b for a, b in zip(src, dst)):
            errs.append(f"move {k + 1} isn't a proper rearrangement")
        if m["kind"] != kind:
            errs.append(f"move {k + 1} has the wrong kind")
        if sc == "neighbour" and abs(src[0][0] - src[1][0]) + abs(src[0][1] - src[1][1]) != 1:
            errs.append(f"move {k + 1} isn't between neighbours")
        if not readable(T, gs[k], m):
            errs.append(f"move {k + 1} can't be read from the boards")
        if min_sep(n, m) < SEP - 1e-9:
            errs.append(f"move {k + 1}: moving tiles come too close")
        if m["kind"] != "slide":
            for i in range(len(m["pairs"])):
                if any(not (-1e-6 <= v <= n - 1 + 1e-6) for j in range(21) for v in path_pos(n, m, i, j / 20)):
                    errs.append(f"move {k + 1}: a tile path leaves the board")
    # timing and layout
    tm = scene["timing"]
    if scene["n_frames"] != tm["intro"] + S * (tm["move"] + tm["pause"]) + tm["hold"]:
        errs.append("n_frames doesn't match the timing")
    if scene["n_frames"] > MAX_FRAMES:
        errs.append("clip longer than 10 s")
    if tm["intro"] < 12 or tm["hold"] < 14 or tm["pause"] < 5 or tm["move"] < 10:
        errs.append("timing too tight")
    lay = scene["layout"]
    if scene["labels"] != label_list(n, lay["x0"], lay["y0"], lay["span"]):
        errs.append("labels don't match the layout")
    errs += layout_errors(n, lay["x0"], lay["y0"], lay["span"], scene["labels"], lay["pad"])
    if lay["span"] / n < 70:
        errs.append("cells too small")
    return errs


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)          # fixed per seed, so retries don't skew the mix
    tries = 0
    while True:
        tries += 1
        s = _try(rng, scenario)
        if s is not None and not check(s):
            s["tries"] = tries
            return s
