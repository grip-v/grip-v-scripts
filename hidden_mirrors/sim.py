"""hidden_mirrors scene sampler and closed-form tracer.

Inverse optics (after Jane Street "Hall of Mirrors", Apr 2015): a board of
square cells hides two-sided diagonal mirrors ('/' or '\\') that are never
drawn. Light pulses are fired in from openings ("ports") on the board edge,
one colour per pulse; a pulse moves one cell per `step` seconds, leaving a
lit trail, and turns 90 degrees when it reaches the centre of a mirror's
cell. Where the mirrors are has to be worked out from how the pulses bend.
Each trail stays lit for HOLD_S after its pulse leaves, then fades out over
FADE_S, so the clip never shows all paths at once.

Scenarios (scene["scenario"]):
  classic   square board (4-7 cells a side), pulses fired one after another
            (a pulse may be fired before the previous one has left)
  rect      rectangular board (5-9 columns x 4-7 rows, never square)
  volley    pulses are fired in pairs from two ports at the same moment
  blocks    1-3 dark blocks (drawn) stop any pulse that reaches them; a
            stopped pulse leaves a ring on the block
  partial   some of the mirrors are drawn, the rest stay hidden

Coordinates: cell (c, r), c = column from the left, r = row from the top;
directions (dc, dr) with dr > 0 = down. Cells are named by the on-screen
labels: a column letter above the board and a row number left of it
("C4" = column C, row 4). A port is (edge, index): index is the column for
the top / bottom edge and the row for the left / right edge. Times are
seconds; frame f is shown at t = f / 24 and the clock shows f * 10 // 24
tenths. Pure function of the rng passed in.
"""
from __future__ import annotations
import math
import random

FPS = 24
MAX_FRAMES = 240
EDGE_S = 0.5                         # question events stay this far inside the clip
HOLD_S, FADE_S = 1.0, 0.6            # a trail stays lit HOLD_S after its pulse ends, then fades out
AREA = (250, 20, 1262, 704)          # the board, its markers and its labels live here (the clock is left of it)
CLOCK_BOX = (18, 16, 214, 84)
SCENARIOS = ("classic", "rect", "volley", "blocks", "partial")
LETTERS = "ABCDEFGHIJ"
LABEL_PX = 34                        # label font size (glyphs >= 24 px tall)
MARK_OUT = 0.42                      # port markers reach this far (in cells) outside the board edge
EDGES = ("top", "right", "bottom", "left")

# per theme: colour name -> RGB; the names are what the questions use
PULSES = {
    "dark": {"gold": (250, 220, 60), "blue": (70, 160, 255), "pink": (255, 100, 190), "green": (70, 225, 100),
             "purple": (175, 120, 255), "orange": (255, 135, 30), "red": (255, 55, 55), "white": (245, 245, 245)},
    "light": {"gold": (190, 160, 0), "blue": (20, 100, 230), "pink": (225, 40, 160), "green": (20, 150, 50),
              "purple": (125, 55, 215), "orange": (235, 105, 0), "red": (205, 25, 30), "black": (25, 25, 30)},
}
THEMES = {
    "dark": {"bg": [(16, 18, 26), (26, 20, 30), (14, 26, 28), (28, 28, 30), (20, 24, 38)],
             "board": [(38, 42, 54), (46, 40, 54), (34, 48, 50), (44, 44, 44)],
             "grid": [(78, 84, 100), (92, 84, 104), (70, 96, 94), (96, 96, 96)],
             "ink": (235, 235, 240), "block": (8, 8, 10), "mirror": [(150, 220, 255), (200, 200, 140)]},
    "light": {"bg": [(232, 230, 224), (220, 226, 234), (236, 232, 240), (226, 234, 226)],
              "board": [(250, 250, 246), (244, 247, 250), (248, 244, 236)],
              "grid": [(176, 172, 164), (160, 170, 186), (170, 184, 170)],
              "ink": (20, 20, 24), "block": (60, 60, 66), "mirror": [(40, 90, 150), (70, 70, 80)]},
}


# ------------------------------------------------------------------ the reflection rule
def reflect(kind, d):
    dc, dr = d
    return (-dr, -dc) if kind == "/" else (dr, dc)


def flipped(kind):
    return "\\" if kind == "/" else "/"


def entry(world, port):
    """(first cell, direction) of a pulse fired in at port (edge, index)."""
    W, H = world["cols"], world["rows"]
    e, i = port
    return {"left": ((0, i), (1, 0)), "right": ((W - 1, i), (-1, 0)),
            "top": ((i, 0), (0, 1)), "bottom": ((i, H - 1), (0, -1))}[e]


def all_ports(world):
    W, H = world["cols"], world["rows"]
    return [("top", i) for i in range(W)] + [("right", i) for i in range(H)] + \
        [("bottom", i) for i in range(W)] + [("left", i) for i in range(H)]


def trace_one(world, port, t0, mirrors=None, blocks=None):
    """One pulse fired in at `port` at time t0. Returns cells (in order), halves (cell, side) the trail
    covers, turns [(t, cell, din, dout)], pts [(t, x, y)] waypoints in grid units (cell centre =
    c + 0.5, r + 0.5) and end ("exit", edge, index, t) | ("stopped", cell, t)."""
    W, H, s = world["cols"], world["rows"], world["step"]
    mir = mirrors if mirrors is not None else {tuple(m["cell"]): m["kind"] for m in world["mirrors"]}
    blk = blocks if blocks is not None else {tuple(b) for b in world["blocks"]}
    cell, d = entry(world, port)
    pts = [(t0, cell[0] + 0.5 - 0.5 * d[0], cell[1] + 0.5 - 0.5 * d[1])]
    cells, halves, turns = [], [], []
    t = t0 + s / 2
    for _ in range(4 * W * H + 4):                 # a path from an edge always leaves (reversibility)
        c, r = cell
        if not (0 <= c < W and 0 <= r < H):
            t_exit = t - s / 2
            pts.append((t_exit, c + 0.5 - 0.5 * d[0], r + 0.5 - 0.5 * d[1]))
            edge = "left" if c < 0 else "right" if c >= W else "top" if r < 0 else "bottom"
            return {"cells": cells, "halves": halves, "turns": turns, "pts": pts,
                    "end": ("exit", edge, r if edge in ("left", "right") else c, t_exit)}
        cells.append(cell)
        halves.append((cell, (-d[0], -d[1])))
        pts.append((t, c + 0.5, r + 0.5))
        if cell in blk:
            return {"cells": cells, "halves": halves, "turns": turns, "pts": pts, "end": ("stopped", cell, t)}
        k = mir.get(cell)
        if k is not None:
            nd = reflect(k, d)
            turns.append((t, cell, d, nd))
            d = nd
        halves.append((cell, d))
        cell = (c + d[0], r + d[1])
        t += s
    raise RuntimeError("pulse never leaves")


def runs(scene_like, **kw):
    out = []
    for p in scene_like["pulses"]:
        r = trace_one(scene_like, (p["edge"], p["index"]), p["t0"], **kw)
        r["colour"] = p["colour"]
        r["port"] = (p["edge"], p["index"])
        r["t0"] = p["t0"]
        out.append(r)
    return out


def end_time(run):
    return run["end"][-1]


def visible_until(run):
    return end_time(run) + HOLD_S + FADE_S


def head_at(run, t):
    """(x, y) of a pulse's head at time t in grid units, or None before launch / after its end."""
    pts = run["pts"]
    if t < pts[0][0] - 1e-9 or t > pts[-1][0] + 1e-9:
        return None
    for (ta, xa, ya), (tb, xb, yb) in zip(pts, pts[1:]):
        if ta - 1e-9 <= t <= tb + 1e-9:
            u = min(1.0, max(0.0, (t - ta) / (tb - ta)))
            return xa + (xb - xa) * u, ya + (yb - ya) * u
    return pts[-1][1], pts[-1][2]


def trail_alpha(run, t):
    """1 while the pulse flies and HOLD_S after; then a linear fade to 0 over FADE_S."""
    if t < run["t0"]:
        return 0.0
    u = t - end_time(run) - HOLD_S
    return 1.0 if u <= 0 else max(0.0, 1.0 - u / FADE_S)


def n_frames(scene):
    return scene["frames"]


def duration(scene):
    return scene["frames"] / FPS


def clock_tenths(frame: int) -> int:
    return frame * 10 // FPS


def cell_name(cell):
    return f"{LETTERS[cell[0]]}{cell[1] + 1}"


# ------------------------------------------------------------------ geometry on screen
def label_off(cell):
    """Distance (px) from the board edge to a label's centre: past the port markers."""
    return MARK_OUT * cell + 8 + LABEL_PX / 2


def layout(rng, cols, rows):
    """Board origin and cell size (px): label strips above and left, marker margins on every side."""
    lab = LABEL_PX + 14
    w, h = AREA[2] - AREA[0] - lab, AREA[3] - AREA[1] - lab
    cmax = min(108.0, w / (cols + 2 * MARK_OUT + 0.1), h / (rows + 2 * MARK_OUT + 0.1))
    cell = int(rng.uniform(max(58.0, 0.78 * cmax), cmax))
    x_lo = AREA[0] + lab + MARK_OUT * cell + 8
    y_lo = AREA[1] + lab + MARK_OUT * cell + 8
    x_hi = AREA[2] - MARK_OUT * cell - 4 - cols * cell
    y_hi = AREA[3] - MARK_OUT * cell - 4 - rows * cell
    x0 = int(x_lo + rng.random() * max(0.0, x_hi - x_lo))
    y0 = int(y_lo + rng.random() * max(0.0, y_hi - y_lo))
    return x0, y0, cell


def labels(scene):
    """Every text label on screen except the clock: [{text, x, y}] (centre, px). Fixed for the clip."""
    st = scene["style"]
    x0, y0, cell = st["x0"], st["y0"], st["cell"]
    off = label_off(cell)
    out = [{"text": LETTERS[c], "x": x0 + (c + 0.5) * cell, "y": y0 - off} for c in range(scene["cols"])]
    out += [{"text": str(r + 1), "x": x0 - off, "y": y0 + (r + 0.5) * cell} for r in range(scene["rows"])]
    return out


# ------------------------------------------------------------------ generation
def _style(rng, cols, rows):
    theme = rng.choice(("dark", "dark", "light"))
    T = THEMES[theme]
    x0, y0, cell = layout(rng, cols, rows)
    tw = rng.randint(5, 9)                               # heads stay well wider than the trail they lead
    return {"theme": theme, "bg": list(rng.choice(T["bg"])), "board": list(rng.choice(T["board"])),
            "board_pat": rng.choice(("plain", "checker", "dots")), "grid": list(rng.choice(T["grid"])),
            "grid_style": rng.choice(("solid", "dashed", "dotted")), "ink": list(T["ink"]),
            "block": list(T["block"]), "mirror": list(rng.choice(T["mirror"])), "mirror_w": rng.randint(5, 8),
            "trail_w": tw, "head_px": rng.randint(max(22, 2 * tw + 10), 30),
            "head_shape": rng.choice(("disc", "square", "diamond")),
            "marker": rng.choice(("wedge", "bar", "notch")), "x0": x0, "y0": y0, "cell": cell}


def _board(rng, scenario):
    if scenario == "rect":
        while True:
            cols, rows = rng.randint(5, 9), rng.randint(4, 7)
            if cols != rows:
                break
    else:
        cols = rows = rng.randint(5, 7) if scenario == "volley" else rng.randint(4, 7)
    cells = [(c, r) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    k = max(3, int(round(rng.uniform(0.14, 0.32) * cols * rows)))
    mirrors = [{"cell": list(c), "kind": rng.choice(("/", "\\")), "shown": False} for c in cells[:k]]
    if scenario == "partial":
        for m in rng.sample(mirrors, rng.randint(1, max(1, k // 3))):
            m["shown"] = True
    blocks = []
    if scenario == "blocks":
        blocks = [list(c) for c in cells[k:k + rng.randint(1, 3)]]
    mirrors.sort(key=lambda m: (m["cell"][1], m["cell"][0]))
    blocks.sort(key=lambda b: (b[1], b[0]))
    return {"cols": cols, "rows": rows, "step": round(rng.uniform(0.17, 0.27), 3), "mirrors": mirrors,
            "blocks": blocks, "pulses": []}


def _frame_t(t):
    """Snap a launch time to a frame instant so the first frame showing a pulse is exact."""
    return math.ceil(t * FPS - 1e-9) / FPS


def _end_port(run):
    return (run["end"][1], run["end"][2]) if run["end"][0] == "exit" else None


def _clash(world, a, b):
    """Two pulses whose trails are on screen at the same time must not share a trail half, a port, or
    come within one cell of each other."""
    if a["t0"] > visible_until(b) or b["t0"] > visible_until(a):
        return False
    if set(a["halves"]) & set(b["halves"]):
        return True
    lo, hi = max(a["t0"], b["t0"]), min(end_time(a), end_time(b))
    f = math.floor(lo * FPS)
    while f / FPS <= hi + 1e-9:
        ha, hb = head_at(a, f / FPS), head_at(b, f / FPS)
        if ha is not None and hb is not None and math.dist(ha, hb) < 1.0 - 1e-9:
            return True
        f += 1
    return False


def _schedule(rng, world, scenario, palette):
    want = rng.randint(4, 7) if scenario == "volley" else rng.randint(3, 6)
    group = 2 if scenario == "volley" else 1
    colours = rng.sample(sorted(palette), min(len(palette), want))
    ports = all_ports(world)
    used, made = set(), []
    t = _frame_t(rng.uniform(0.6, 1.1))
    while len(made) < want:
        batch = []
        for _ in range(min(group, want - len(made))):
            rng.shuffle(ports)
            for port in ports:
                if port in used:
                    continue
                r = trace_one(world, port, t)
                ep = _end_port(r)
                if ep is not None and (ep in used or ep == port and rng.random() < 0.7):
                    continue
                if len(r["cells"]) < 2 or len(r["cells"]) * world["step"] > 3.4:
                    continue
                if not r["turns"] and (rng.random() < 0.75 or any(not q["turns"] for q in made + batch)):
                    continue
                if end_time(r) > MAX_FRAMES / FPS - HOLD_S - 0.1:
                    continue
                r.update(colour=colours[len(made) + len(batch)], port=port, t0=t)
                if any(_clash(world, r, q) for q in made + batch):
                    continue
                batch.append(r)
                used.add(port)
                if ep is not None:
                    used.add(ep)
                break
            else:
                return made
        made += batch
        longest = max(len(q["cells"]) for q in batch) * world["step"]
        gap = rng.uniform(0.6, longest + 0.3) if rng.random() < 0.65 else rng.uniform(longest + 0.1, longest + 0.6)
        t = _frame_t(t + max(0.7, gap))
    return made


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    for attempt in range(1, 3000):
        world = _board(rng, scenario)
        st = _style(rng, world["cols"], world["rows"])
        palette = PULSES[st["theme"]]
        made = _schedule(rng, world, scenario, palette)
        if len(made) < (4 if scenario == "volley" else 3):
            continue
        world["pulses"] = [{"colour": r["colour"], "edge": r["port"][0], "index": r["port"][1], "t0": r["t0"]}
                           for r in made]
        t_end = max(end_time(r) for r in made)
        T = t_end + rng.uniform(HOLD_S + 0.1, HOLD_S + 0.6)
        frames = min(MAX_FRAMES, int(T * FPS))
        if frames < 96:                                   # at least 4 s
            continue
        scene = {"scenario": scenario, **world, "frames": frames, "style": st, "build_attempts": attempt}
        scene["labels"] = labels(scene)
        if check(scene) or not _ladder_ok(scene):
            continue
        return scene
    raise RuntimeError("no valid board")


def _ladder_ok(scene):
    """Can this board fill the 10-question ladder? (A trial build with its own fixed rng.)"""
    import questions                                   # late import: questions.py imports this module
    try:
        questions.build(scene, random.Random(0))
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------ consistency check
def check(scene: dict) -> list[str]:
    errs = []
    rs = runs(scene)
    if not 96 <= scene["frames"] <= MAX_FRAMES:
        errs.append("clip length out of range")
    for i, a in enumerate(rs):
        if a["t0"] < EDGE_S + 0.05:
            errs.append(f"{a['colour']} pulse fired too early")
        if end_time(a) > (scene["frames"] - 1) / FPS - HOLD_S + 1e-9:      # the last frame shown is frames - 1
            errs.append(f"{a['colour']} pulse ends less than {HOLD_S} s before the last frame")
        if abs(a["t0"] * FPS - round(a["t0"] * FPS)) > 1e-6:
            errs.append(f"{a['colour']} pulse is not fired on a frame instant")
        for b in rs[i + 1:]:
            if _clash(scene, a, b):
                errs.append(f"{a['colour']} and {b['colour']} trails overlap or their heads come too close")
    ports = [r["port"] for r in rs] + [_end_port(r) for r in rs if _end_port(r) and _end_port(r) != r["port"]]
    if len(set(ports)) != len(ports):
        errs.append("two pulses share a port")
    if len({r["colour"] for r in rs}) != len(rs):
        errs.append("two pulses share a colour")
    cells = [tuple(m["cell"]) for m in scene["mirrors"]] + [tuple(b) for b in scene["blocks"]]
    if len(set(cells)) != len(cells):
        errs.append("two objects share a cell")
    st = scene["style"]
    if st["cell"] < 58 or st["head_px"] < 20 or st["trail_w"] < 3 or st["head_px"] > 0.5 * st["cell"] \
            or st["head_px"] < 2 * st["trail_w"] + 10:
        errs.append("objects too small or heads too large for the cells")
    x1, y1 = st["x0"] + scene["cols"] * st["cell"], st["y0"] + scene["rows"] * st["cell"]
    off = label_off(st["cell"]) + LABEL_PX / 2 + 4
    m = MARK_OUT * st["cell"]
    if st["x0"] - off < AREA[0] or st["y0"] - off < AREA[1] or x1 + m > AREA[2] or y1 + m > AREA[3]:
        errs.append("board, markers or labels leave their area")
    if scene["scenario"] == "blocks" and not any(r["end"][0] == "stopped" for r in rs):
        errs.append("no pulse is stopped by a block")
    if scene["scenario"] == "partial":
        shown = {tuple(m["cell"]) for m in scene["mirrors"] if m["shown"]}
        hidden_turns = {t[1] for r in rs for t in r["turns"]} - shown
        if not hidden_turns:
            errs.append("no pulse turns at a hidden mirror")
    if scene["scenario"] == "volley" and len({r["t0"] for r in rs}) == len(rs):
        errs.append("volley without a simultaneous pair")
    if sum(len(r["turns"]) for r in rs) < 3:
        errs.append("fewer than 3 turns in the clip")
    return errs
