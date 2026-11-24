"""Sandpile cascade: grains are dropped one at a time onto a board of cells;
a cell that holds T or more grains TOPPLES: T grains leave it, one hop to
each of its outgoing neighbours (T = number of outgoing hops). A grain sent
past the edge of the board (or into the hole) is lost. Toppling can set off
more toppling (a cascade); each cascade settles completely before the next
grain is dropped.

Scenarios (`scene["scenario"]`):
  row      side view: a row of 5-9 trays, grains stacked upwards in each tray;
           T = 2 (one grain to each side) or T = 4 (two to each side); grains
           sent past either end fall off the board
  grid     top view: a 3x3 - 4x5 board of square cells; T = 4, one grain to
           each of the four side neighbours; grains sent off an edge are lost
  ring     top view: 6-10 round cells on a ring, one position is a hole;
           T = 2 (one grain each way round) or T = 4 (two each way); a grain
           sent into the hole is lost
  pyramid  top view: rows of cells, each row one cell longer than the row
           above; T = 2 and both grains go DOWN, to the two cells just below
           (down-left, down-right); grains sent below the bottom row are lost

Timing (whole frames, 24 fps): a dropped grain falls for F frames and lands
on frame `start + F` (side view: straight down from above the frame; top
view: it comes down from DROP_LIFT cell pitches above its slot while
shrinking from 2.3x to its resting size). Toppling happens in waves: D frames after the last
change, every cell holding >= T grains topples at once; its grains hop for M
frames and land together; then the next wave (if any) starts D frames later.
The next grain starts to fall `gap` frames after the cascade has settled.
Lost grains: in the row they hop past the end and fall off the bottom of the
frame (FALL_OUT frames); in top view they hop off the board (or into the
hole) and shrink away (SHRINK frames).

The only text on screen is the clock; cells are named by position.
"""
from __future__ import annotations
import functools
import json
import math
import random

import numpy as np

W, H, FPS = 1280, 720, 24
MAX_FRAMES = 240
EDGE = 12                 # frames between the clip's start / end and any question-relevant event
TIMER_BOX = (14, 14, 196, 70)
TIMER_FONT = 36
CLOCK_CLEAR = 14          # px between the clock panel and anything else
X_MIN = TIMER_BOX[2] + CLOCK_CLEAR
SHRINK = 5                # frames a lost grain takes to shrink away (top view)
DROP_LIFT = 0.7           # top view: a dropped grain comes down from this far above its slot (cell pitches)
FALL_OUT = 8              # frames a lost grain takes to fall out of the frame (row)
MIN_GRAIN = 22            # px
SCENARIOS = ("row", "grid", "ring", "pyramid")

GRAINS = {"red": (214, 40, 40), "blue": (30, 92, 224), "green": (24, 150, 64), "orange": (240, 128, 16),
          "purple": (136, 58, 200), "magenta": (214, 40, 160), "teal": (0, 150, 150), "gold": (226, 176, 8)}
SHAPES = ("square", "rounded", "disc")
BACKGROUNDS = [  # (top, bottom, light?)
    ((238, 234, 224), (220, 214, 200), True), ((228, 234, 240), (204, 214, 226), True),
    ((232, 238, 228), (212, 220, 206), True), ((240, 232, 236), (222, 212, 220), True),
    ((26, 30, 42), (42, 48, 64), False), ((36, 36, 40), (58, 58, 64), False), ((22, 40, 46), (36, 60, 66), False),
]
CELLS = {True: [(196, 190, 176), (184, 196, 208), (200, 206, 188), (210, 196, 200), (190, 190, 196)],
         False: [(78, 84, 100), (88, 88, 96), (64, 86, 92), (92, 84, 76), (74, 80, 70)]}
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]


def tenths(f: int) -> int:
    return (10 * f) // FPS


def clock_text(k: int) -> str:
    return f"{k // 10}.{k % 10} s"


def frames_showing(k: int) -> tuple[int, int]:
    """First and last frame whose clock reads k tenths."""
    return -(-k * FPS // 10), -(-(k + 1) * FPS // 10) - 1


# ---------------------------------------------------------------- board layout (units: cell pitch = 1)

def _cells_units(sc):
    """[(x, y, kind)] in units (y down), kind 'cell' / 'hole'; and out[c] = list of targets
    (cell index >= 0, or -1-k for virtual (off-board) spot k); virtual spots [(x, y)]."""
    s, d = sc["scenario"], sc["dims"]
    cells, out, virt = [], [], []

    def vspot(x, y):
        virt.append((x, y))
        return -len(virt)

    if s == "row":
        n = d["n"]
        cells = [(i, 0.0, "cell") for i in range(n)]
        half = sc["T"] // 2
        vl, vr = vspot(-1, 0.0), vspot(n, 0.0)
        for i in range(n):
            out.append([i - 1 if i > 0 else vl] * half + [i + 1 if i < n - 1 else vr] * half)
    elif s == "grid":
        R, C = d["rows"], d["cols"]
        idx = {(r, c): r * C + c for r in range(R) for c in range(C)}
        cells = [(c, r, "cell") for r in range(R) for c in range(C)]
        vs = {}
        for r in range(R):
            for c in range(C):
                tg = []
                for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                    p = (r + dr, c + dc)
                    if p in idx:
                        tg.append(idx[p])
                    else:
                        if p not in vs:
                            vs[p] = vspot(c + 0.8 * dc, r + 0.8 * dr)
                        tg.append(vs[p])
                out.append(tg)
    elif s == "ring":
        k = d["k"]
        rad = 0.5 / math.sin(math.pi / k)
        for i in range(k):
            a = d["a0"] + 2 * math.pi * i / k                          # increasing angle = clockwise on screen
            cells.append((rad * math.cos(a), rad * math.sin(a), "hole" if i == 0 else "cell"))
        half = sc["T"] // 2
        for i in range(k):
            out.append([] if i == 0 else [(i - 1) % k] * half + [(i + 1) % k] * half)
    else:  # pyramid
        R, w0 = d["rows"], d["w0"]
        idx = {}
        for r in range(R):
            for c in range(w0 + r):
                idx[(r, c)] = len(cells)
                cells.append((c - (w0 + r - 1) / 2, r * 0.92, "cell"))
        vs = {}
        for r in range(R):
            for c in range(w0 + r):
                tg = []
                for cc in (c, c + 1):
                    if (r + 1, cc) in idx:
                        tg.append(idx[(r + 1, cc)])
                    else:
                        if cc not in vs:
                            vs[cc] = vspot(cc - (w0 + R - 1) / 2, R * 0.92 - 0.1)
                        tg.append(vs[cc])
                out.append(tg)
    return cells, out, virt


def cap(sc) -> int:
    """How many grains a cell can show (slots)."""
    if sc["scenario"] == "row":
        return 2 * sc["T"]
    return 9 if sc["T"] == 4 else 4


def _slot_offsets(sc, size):
    """Offsets (units) of the slots inside a top-view cell of the given size, fill order."""
    if cap(sc) == 9:
        q = size / 3.1
        return [((j - 1) * q, (i - 1) * q) for i in range(3) for j in range(3)]
    q = size / 4.4
    return [(-q, -q), (q, -q), (-q, q), (q, q)]


def cell_size(sc) -> float:
    return {"grid": 0.9, "ring": 0.86, "pyramid": 0.84}.get(sc["scenario"], sc["tray"])


def grain_units(sc) -> float:
    if sc["scenario"] == "row":
        return sc["gf"]
    return sc["gf"] * cell_size(sc) / (3.1 if cap(sc) == 9 else 2.2)


ROW_GAP = 0.07            # units between stacked grains in a tray
TRAY_T = 0.14             # tray thickness (units)


@functools.lru_cache(maxsize=64)
def _geom(key: str) -> dict:
    sc = json.loads(key)
    return geometry_at(sc, sc["S"], sc["ox"], sc["oy"])


def geometry(sc) -> dict:
    return _geom(json.dumps({k: sc[k] for k in ("scenario", "dims", "T", "gf", "tray", "S", "ox", "oy")},
                            sort_keys=True))


def geometry_at(sc, S, ox, oy) -> dict:
    """Pixel geometry: cells (centre, kind, slot centres), outgoing targets, virtual spots, grain size."""
    cells_u, out, virt_u = _cells_units(sc)
    g = grain_units(sc)
    cells = []
    for (x, y, kind) in cells_u:
        cx, cy = ox + S * x, oy + S * y
        if kind == "hole":
            slots = []
        elif sc["scenario"] == "row":
            slots = [(cx, cy - S * (TRAY_T / 2 + ROW_GAP + (k + 0.5) * g + k * ROW_GAP)) for k in range(cap(sc))]
        else:
            slots = [(cx + S * dx, cy + S * dy) for dx, dy in _slot_offsets(sc, cell_size(sc))]
        cells.append({"c": (cx, cy), "kind": kind, "slots": slots})
    if sc["scenario"] == "row":
        virt = [(ox + S * x, cells[0]["slots"][0][1]) for x, _ in virt_u]
    else:
        virt = [(ox + S * x, oy + S * y) for x, y in virt_u]
    return {"cells": cells, "out": out, "virt": virt, "g": g * S, "size": cell_size(sc) * S}


def bbox(sc, S=1.0, ox=0.0, oy=0.0) -> tuple:
    """Extent of everything drawn (cells, trays, slots, virtual spots, big falling grains)."""
    G = geometry_at(sc, S, ox, oy)
    g, sz = G["g"], G["size"]
    xs, ys = [], []
    for c in G["cells"]:
        cx, cy = c["c"]
        if sc["scenario"] == "row":
            xs += [cx - sz / 2, cx + sz / 2]
            ys += [cy + S * TRAY_T / 2, c["slots"][-1][1] - g / 2]
        else:
            r = sz / 2 + (0.6 * g if c["kind"] == "cell" else 0)
            xs += [cx - r, cx + r]
            ys += [cy - r, cy + r]
            if c["kind"] == "cell":                                     # a grain coming down big from above
                ys.append(min(p[1] for p in c["slots"]) - DROP_LIFT * S - 1.15 * g)
    for x, y in G["virt"]:
        xs += [x - 0.6 * g, x + 0.6 * g]
        ys += [y - 0.6 * g, y + 0.6 * g]
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------- the rule

def topology(sc):
    cells, out, virt = _cells_units(sc)
    return [k for _, _, k in cells], out


def relax_waves(h, kinds, out, T):
    """Parallel toppling. Returns the list of waves (each the list of cells toppling at once);
    h is updated in place. Grains sent to a hole or a virtual spot are lost."""
    waves = []
    while True:
        wave = [c for c in range(len(h)) if kinds[c] == "cell" and h[c] >= T]
        if not wave:
            return waves
        if len(waves) > 400:
            raise RuntimeError("no settling")
        for c in wave:
            h[c] -= T
        for c in wave:
            for t in out[c]:
                if t >= 0 and kinds[t] == "cell":
                    h[t] += 1
        waves.append(wave)


def play(sc) -> dict:
    """The clip as events (frames). Pure function of the scene (no pixels)."""
    kinds, out = topology(sc)
    T, tm = sc["T"], sc["timing"]
    F, D, M = tm["F"], tm["D"], tm["M"]
    h = list(sc["init"])
    ev, aval = [], []
    t = 0
    hmax = max(h)
    end_any = 0
    for i, dr in enumerate(sc["drops"]):
        start = t + dr["gap"]
        land = start + F
        c = dr["cell"]
        h[c] += 1
        hmax = max(hmax, h[c])
        ev.append({"kind": "land", "f": land, "start": start, "cell": c, "drop": i})
        tw, n_top, cells_hit, lost = land, 0, [], 0
        wave_i = 0
        while True:
            wave = [x for x in range(len(h)) if kinds[x] == "cell" and h[x] >= T]
            if not wave:
                break
            t0 = tw + D
            for x in wave:
                h[x] -= T
            for x in wave:
                for tg in out[x]:
                    if tg >= 0 and kinds[tg] == "cell":
                        h[tg] += 1
                    else:
                        lost += 1
                        end_any = max(end_any, t0 + M + (FALL_OUT if sc["scenario"] == "row" else SHRINK))
                ev.append({"kind": "topple", "f": t0, "cell": x, "wave": wave_i, "drop": i, "to": list(out[x])})
                n_top += 1
                cells_hit.append(x)
            hmax = max(hmax, max(h))
            tw = t0 + M
            wave_i += 1
        aval.append({"drop": i, "land": land, "end": tw, "topples": n_top, "waves": wave_i, "cells": cells_hit,
                     "lost": lost})
        t = tw
        end_any = max(end_any, tw)
    return {"events": ev, "aval": aval, "final": h, "hmax": hmax, "end": max(end_any, t)}


# ---------------------------------------------------------------- kinematics (px, per frame)

@functools.lru_cache(maxsize=16)
def _timeline(key: str) -> dict:
    sc = json.loads(key)
    return _build_timeline(sc)


def timeline(sc) -> dict:
    return _timeline(json.dumps(sc, sort_keys=True))


def _build_timeline(sc) -> dict:
    """poses[f, g] = (x, y, scale) of every grain on every frame (NaN when not drawn);
    plus the events of play() and, per frame, the grains at rest in every cell."""
    G = geometry(sc)
    kinds, out = topology(sc)
    T, tm = sc["T"], sc["timing"]
    F, D, M = tm["F"], tm["D"], tm["M"]
    row = sc["scenario"] == "row"
    n = sc["n_frames"]
    segs = []                                 # per grain: list of (f0, f1, kind, a, b) ; position after f1 = b
    stacks = [[] for _ in kinds]

    def slot(c, k):
        return G["cells"][c]["slots"][k]

    for c, hc in enumerate(sc["init"]):
        for k in range(hc):
            segs.append([(0, 0, "rest", slot(c, k), slot(c, k))])
            stacks[c].append(len(segs) - 1)
    t = 0
    for i, dr in enumerate(sc["drops"]):
        start = t + dr["gap"]
        land = start + F
        c = dr["cell"]
        p1 = slot(c, len(stacks[c]))
        if row:
            p0 = (p1[0], -G["g"])
        else:
            p0 = p1
        segs.append([(start, land, "drop", p0, p1)])
        stacks[c].append(len(segs) - 1)
        tw = land
        while True:
            wave = [x for x in range(len(kinds)) if kinds[x] == "cell" and len(stacks[x]) >= T]
            if not wave:
                break
            t0 = tw + D
            moving = []
            for x in wave:
                leaving = stacks[x][-T:]
                del stacks[x][-T:]
                base = len(stacks[x])
                for j, (gid, tg) in enumerate(zip(leaving, out[x])):
                    moving.append((gid, slot(x, base + j), tg))
            for gid, p0, tg in moving:
                if tg >= 0 and kinds[tg] == "cell":
                    p1 = slot(tg, len(stacks[tg]))
                    stacks[tg].append(gid)
                    segs[gid].append((t0, t0 + M, "hop", p0, p1))
                else:
                    p1 = G["cells"][tg]["c"] if tg >= 0 else G["virt"][-1 - tg]
                    segs[gid].append((t0, t0 + M, "hop", p0, p1))
                    if row:
                        segs[gid].append((t0 + M, t0 + M + FALL_OUT, "fallout", p1, (p1[0], H + G["g"])))
                    else:
                        segs[gid].append((t0 + M, t0 + M + SHRINK, "shrink", p1, p1))
            tw = t0 + M
        t = tw
    ng = len(segs)
    P = np.full((n, ng, 3), np.nan)
    lift = 0.55 * (G["size"] if not row else sc["S"])
    for gid, ss in enumerate(segs):
        si = 0
        for f in range(n):
            while si + 1 < len(ss) and ss[si + 1][0] <= f:
                si += 1
            f0, f1, kind, a, b = ss[si]
            if f < f0:
                continue
            if f > f1:
                if kind not in ("shrink", "fallout"):
                    P[f, gid] = (b[0], b[1], 1.0)
                continue
            if kind == "rest":
                pose = (a[0], a[1], 1.0)
            elif kind == "drop":
                u = (f - f0) / (f1 - f0)
                if row:
                    pose = (a[0], a[1] + (b[1] - a[1]) * u * u, 1.0)
                else:
                    pose = (b[0], b[1] - DROP_LIFT * sc["S"] * (1 - u), 1.0 + 1.3 * (1 - u))
            elif kind == "hop":                     # already moved on its first frame, arrives on f1
                u = (f - f0 + 1) / (f1 - f0 + 1)
                x = a[0] + (b[0] - a[0]) * u
                y = a[1] + (b[1] - a[1]) * u
                if row:
                    pose = (x, y - lift * 4 * u * (1 - u), 1.0)
                else:
                    pose = (x, y, 1.0 + 0.35 * 4 * u * (1 - u))
            elif kind == "fallout":
                u = (f - f0) / (f1 - f0)
                pose = (a[0], a[1] + (b[1] - a[1]) * u * u, 1.0)
            else:  # shrink
                pose = (a[0], a[1], 1.0 - (f - f0) / (f1 - f0))
            P[f, gid] = pose
    # off-frame (row: above the top before it falls in, or below the bottom) -> not drawn
    g2 = G["g"] / 2
    off = (P[:, :, 1] + g2 * P[:, :, 2] <= 0) | (P[:, :, 1] - g2 * P[:, :, 2] >= H) | (P[:, :, 2] <= 0.02)
    P[off] = np.nan
    res = play(sc)
    return {"poses": P, "events": res["events"], "aval": res["aval"], "segs": segs, "final": res["final"]}


def counts(sc) -> np.ndarray:
    """counts[f, c] = grains of cell c sitting still in its slots on frame f (from the rule, not the pixels)."""
    pl = play(sc)
    n = sc["n_frames"]
    h = np.zeros((n, len(sc["init"])), dtype=int)
    cur = list(sc["init"])
    changes = []
    tm = sc["timing"]
    kinds, out = topology(sc)
    for e in pl["events"]:
        if e["kind"] == "land":
            changes.append((e["f"], e["cell"], +1))
        else:
            changes.append((e["f"], e["cell"], -sc["T"]))
            for tg in e["to"]:
                if tg >= 0 and kinds[tg] == "cell":
                    changes.append((e["f"] + tm["M"], tg, +1))
    changes.sort()
    k = 0
    for f in range(n):
        while k < len(changes) and changes[k][0] <= f:
            cur[changes[k][1]] += changes[k][2]
            k += 1
        h[f] = cur
    return h


def covered(sc) -> np.ndarray:
    """covered[f, c]: a grain that is not resting on a slot overlaps cell c's slots on frame f."""
    tl = timeline(sc)
    P = tl["poses"]
    G = geometry(sc)
    g = G["g"]
    n, nc = sc["n_frames"], len(G["cells"])
    slots = np.array([s for c in G["cells"] for s in c["slots"]])
    boxes = []
    for c in G["cells"]:
        if not c["slots"]:
            boxes.append(None)
            continue
        xs = [p[0] for p in c["slots"]]
        ys = [p[1] for p in c["slots"]]
        boxes.append((min(xs) - g / 2 - 2, min(ys) - g / 2 - 2, max(xs) + g / 2 + 2, max(ys) + g / 2 + 2))
    X, Y, Z = P[:, :, 0], P[:, :, 1], P[:, :, 2]
    vis = ~np.isnan(X)
    Xs, Ys, Zs = np.nan_to_num(X, nan=-1e6), np.nan_to_num(Y, nan=-1e6), np.nan_to_num(Z, nan=0.0)
    d = (np.abs(Xs[:, :, None] - slots[None, None, :, 0]) + np.abs(Ys[:, :, None] - slots[None, None, :, 1])).min(axis=2)
    moving = vis & ~((np.abs(Zs - 1) < 1e-9) & (d < 0.05))
    r = g * Zs / 2 + 3
    out = np.zeros((n, nc), dtype=bool)
    for c, bx in enumerate(boxes):
        if bx:
            hit = moving & (Xs - r < bx[2]) & (bx[0] < Xs + r) & (Ys - r < bx[3]) & (bx[1] < Ys + r)
            out[:, c] = hit.any(axis=1)
    return out


# ---------------------------------------------------------------- naming (positions only)

def cell_desc(sc, c) -> list:
    s, d = sc["scenario"], sc["dims"]
    if s == "row":
        n = d["n"]
        return ["L", c + 1] if c < (n + 1) // 2 else ["R", n - c]
    if s == "grid":
        return [c // d["cols"] + 1, c % d["cols"] + 1]
    if s == "ring":
        return [c]
    r, k = 0, c
    while k >= d["w0"] + r:
        k -= d["w0"] + r
        r += 1
    return [r + 1, k + 1]


def desc_name(scenario, desc, w0_top=None) -> str:
    if scenario == "row":
        return f"the {ORD[desc[1] - 1]} tray from the {'left' if desc[0] == 'L' else 'right'}"
    if scenario == "grid":
        return f"the cell in row {desc[0]}, column {desc[1]}"
    if scenario == "ring":
        return f"the {ORD[desc[0] - 1]} cell clockwise from the hole"
    if desc[0] == 1 and w0_top == 1:
        return "the top cell"
    return f"the {ORD[desc[1] - 1]} cell from the left in the {ORD[desc[0] - 1]} row"


def cell_name(sc, c) -> str:
    return desc_name(sc["scenario"], cell_desc(sc, c), sc["dims"].get("w0"))


CONVENTION = {"grid": " (Rows are counted from the top, columns from the left.)",
              "pyramid": " (Rows are counted from the top.)", "row": "", "ring": ""}
UNIT = {"row": "tray", "grid": "cell", "ring": "cell", "pyramid": "cell"}


def rule_text(sc) -> str:
    T, s = sc["T"], sc["scenario"]
    if s == "row":
        each = "one grain" if T == 2 else "two grains"
        return (f"A tray holding {T} or more grains topples: it sends {each} to each neighbouring tray, and "
                f"grains sent past an end of the row fall off.")
    if s == "grid":
        return ("A cell holding 4 or more grains topples: it sends one grain to each of the four cells beside "
                "it, and grains sent off the edge of the board are lost.")
    if s == "ring":
        each = "one grain" if T == 2 else "two grains"
        return (f"A cell holding {T} or more grains topples: it sends {each} to each of its two neighbours on "
                f"the ring, and grains sent into the hole are lost.")
    return ("A cell holding 2 or more grains topples: it sends one grain to each of the two cells just below "
            "it, and grains sent below the bottom row are lost.")


# ---------------------------------------------------------------- generation

REJECTS: dict = {}


def _reject(why):
    REJECTS[why] = REJECTS.get(why, 0) + 1
    return None


def _dims(rng, s):
    if s == "row":
        return {"n": rng.randint(5, 9)}, rng.choice([2, 4])
    if s == "grid":
        return rng.choice([{"rows": 3, "cols": 3}, {"rows": 3, "cols": 4}, {"rows": 3, "cols": 5},
                           {"rows": 4, "cols": 4}, {"rows": 4, "cols": 5}, {"rows": 3, "cols": 6}]), 4
    if s == "ring":
        return {"k": rng.randint(6, 10), "a0": round(rng.choice([-0.5, 0.0, 0.5, 1.0, 1.5]) * math.pi, 6)}, \
            rng.choice([2, 4])
    return {"rows": rng.randint(3, 5), "w0": rng.choice([1, 1, 2, 3])}, 2


def _style(rng):
    top, bot, light = rng.choice(BACKGROUNDS)
    style = rng.choice(["flat", "gradient", "vignette"])
    if style == "flat":
        bot = top
    grain = rng.choice(list(GRAINS))
    return {"bg": [list(top), list(bot)], "bg_style": style, "light": light, "grain": grain,
            "shape": rng.choice(SHAPES), "cell": list(rng.choice(CELLS[light])),
            "outline": rng.choice([3, 4]), "mark": rng.choice(["none", "dot", "ring"])}


def _attempt(rng, s):
    dims, T = _dims(rng, s)
    sc = {"scenario": s, "dims": dims, "T": T, "labels": [],
          "gf": round(rng.uniform(0.5, 0.66) if s == "row" else rng.uniform(0.62, 0.78), 3),
          "tray": round(rng.uniform(0.74, 0.86), 3) if s == "row" else 0.0,
          "timing": {"F": rng.randint(8, 12), "D": rng.randint(4, 8), "M": rng.randint(8, 12)}}
    kinds, out = topology(sc)
    live = [c for c, k in enumerate(kinds) if k == "cell"]
    load = rng.choice([0.25, 0.4, 0.55, 0.7, 0.85])          # how many cells start one grain short of toppling
    init = []
    for c, k in enumerate(kinds):
        if k != "cell":
            init.append(0)
        elif rng.random() < load:
            init.append(T - 1)
        else:
            init.append(rng.randint(0, T - 1))
    sc["init"] = init
    nd = rng.randint(3, 9)
    focus = rng.sample(live, min(len(live), rng.randint(2, 5)))
    drops = []
    for i in range(nd):
        c = rng.choice(focus) if rng.random() < 0.6 else rng.choice(live)
        gap = rng.randint(EDGE, EDGE + 10) if i == 0 else rng.randint(4, 16)
        drops.append({"cell": c, "gap": gap})
    sc["drops"] = drops
    tail = rng.randint(EDGE + 2, EDGE + 14)
    # keep only the drops whose cascade fits in the clip
    while sc["drops"]:
        pl = play(sc)
        if pl["end"] + tail <= MAX_FRAMES and pl["hmax"] <= cap(sc):
            break
        sc["drops"] = sc["drops"][:-1]
    if len(sc["drops"]) < 2:
        return _reject("too long")
    pl = play(sc)
    n_top = sum(a["topples"] for a in pl["aval"])
    if n_top < 2:
        return _reject("too few topples")
    if sum(1 for a in pl["aval"] if a["topples"]) < 1:
        return _reject("no cascade")
    sc["n_frames"] = pl["end"] + tail
    return sc


def _frame_it(rng, sc) -> bool:
    x0, y0, x1, y1 = bbox(sc)
    avail_w, avail_h = W - 16 - X_MIN, H - 32
    s_max = min(avail_w / (x1 - x0), avail_h / (y1 - y0))
    if sc["scenario"] == "row":
        s_max = min(s_max, 150)
    s_min = MIN_GRAIN / grain_units(sc) + 0.5
    if s_min > s_max:
        return False
    S = round(rng.uniform(s_min + 0.45 * (s_max - s_min), s_max))      # fill the frame
    S = max(math.ceil(s_min), min(int(s_max), S))
    sc["S"] = S
    bx0, by0, bx1, by1 = (v * S for v in (x0, y0, x1, y1))
    lo_x, hi_x = X_MIN - bx0, W - 16 - bx1
    lo_y, hi_y = 16 - by0, H - 16 - by1
    if sc["scenario"] == "row":                       # trays low in the frame, room to fall off the bottom
        lo_y = max(lo_y, hi_y - 0.35 * (hi_y - lo_y))
    if lo_x > hi_x or lo_y > hi_y:
        return False
    sc["ox"] = round(rng.uniform(lo_x, hi_x), 1)
    sc["oy"] = round(rng.uniform(lo_y, hi_y), 1)
    return True


ATTEMPTS: list = []                          # attempts per built scene (for the stats report)


def build(rng: random.Random) -> dict:
    s = rng.choice(SCENARIOS)
    for k in range(4000):
        sc = _attempt(rng, s)
        if sc is None:
            continue
        if not _frame_it(rng, sc):
            _reject("does not fit")
            continue
        sc["style"] = _style(rng)
        sc = json.loads(json.dumps(sc))
        errs = check(sc)
        if errs:
            _reject("check: " + errs[0].split(":")[0])
            continue
        ATTEMPTS.append(k + 1)
        return sc
    raise RuntimeError(f"no {s} scene")


# ---------------------------------------------------------------- checks

def check(sc) -> list[str]:
    errs = []
    n = sc["n_frames"]
    if not 1 <= n <= MAX_FRAMES:
        errs.append(f"length: {n} frames")
    pl = play(sc)
    if pl["hmax"] > cap(sc):
        errs.append(f"capacity: a cell holds {pl['hmax']} > {cap(sc)}")
    if pl["end"] + EDGE >= n:
        errs.append("end: activity too close to the end")
    for e in pl["events"]:
        if e["f"] < EDGE or e["f"] > n - 1 - EDGE:
            errs.append(f"edge: {e['kind']} at frame {e['f']}")
        if e["kind"] == "land" and e["start"] < EDGE:
            errs.append(f"edge: fall starts at frame {e['start']}")
    fs = sorted({e["f"] for e in pl["events"]})
    tm = sc["timing"]
    for a, b in zip(fs, fs[1:]):
        if b - a < 2:
            errs.append(f"spacing: events {a} and {b}")
    for e in pl["events"]:
        if e["kind"] == "topple" and e["f"] + tm["M"] in fs:
            errs.append("spacing: an arrival on an event frame")
    if sum(a["topples"] for a in pl["aval"]) < 2:
        errs.append("dull: fewer than two topples")
    G = geometry(sc)
    if G["g"] < MIN_GRAIN - 1e-9:
        errs.append(f"size: grain {G['g']:.1f} px")
    x0, y0, x1, y1 = bbox(sc, sc["S"], sc["ox"], sc["oy"])
    if x0 < X_MIN - 1e-6 or x1 > W - 4 or y0 < 4 or y1 > H - 4:
        errs.append("frame: board outside the frame or near the clock")
    tl = timeline(sc)
    P = tl["poses"]
    g2 = G["g"] / 2
    tb = TIMER_BOX
    near = ((P[:, :, 0] + g2 * P[:, :, 2] > tb[0] - CLOCK_CLEAR) & (P[:, :, 0] - g2 * P[:, :, 2] < tb[2] + CLOCK_CLEAR)
            & (P[:, :, 1] + g2 * P[:, :, 2] > tb[1] - CLOCK_CLEAR) & (P[:, :, 1] - g2 * P[:, :, 2] < tb[3] + CLOCK_CLEAR))
    if near.any():
        errs.append("clock: a grain passes the clock panel")
    vis = ~np.isnan(P[:, :, 0])
    for gid in range(P.shape[1]):
        fr = np.nonzero(vis[:, gid])[0]
        if len(fr) and fr[-1] < n - 1 and fr[-1] > n - 6:
            errs.append(f"end: grain {gid} leaves the frame at {fr[-1]}")
    # what is at rest equals the rule's counts
    C = counts(sc)
    if (C < 0).any():
        errs.append("rule: negative count")
    if not np.array_equal(C[-1], np.array(pl["final"])):
        errs.append("rule: final counts")
    return errs
