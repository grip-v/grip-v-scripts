"""nested_shapes (v2, PIL): a nest of 3-7 outlines, one inside the other, that changes over time.

A clip (1280x720, 24 fps, 6-10 s) shows a nest of closed outlines (12 shape
types from common/shapes.py), each one strictly inside the one around it, with
a guaranteed gap at every rotation. Each outline has a border colour, a line
style (solid, dashed, dotted, double) and a width; some carry a small "tab"
dot on one corner so that their turning is visible. All timing is in whole
frames.

Scenarios (scene["scenario"]):
  assemble  the outlines appear one at a time (outside-in, inside-out or in a
            mixed order); in some clips one or two of them disappear again
  spin      every outline is there from the start; outlines turn clockwise,
            counterclockwise or stay still, and reverse / stop / start at
            event frames (rotation patterns)
  marks     small filled marks (hearts, stars, dots, diamonds) are placed on
            the outlines one at a time, and some are taken off again; the
            marks ride on their outline as it turns (fill-mark counting)
  recolor   every outline has its own shape type; borders change colour at
            event frames (attribute binding over time)
  drift     a small nest glides inside a box and bounces off its edges while
            its outlines turn

In every scenario the outlines that are not the focus may still turn at a
steady rate, so rotation questions exist everywhere. `check` rejects a scene
whose question-relevant events lack a margin; `build` retries (same rng,
deterministic) until one passes.

Geometry (pixels, y down): ring k is unit_shape(type, aspect) scaled by
`size`, turned by rot(f) degrees (positive = clockwise on screen) about its
centre nest(f) + (dx, dy). Containment holds at any rotation because each
ring's outer reach (size * max radius of the unit outline + half the line)
plus its centre offset plus a gap stays below the next ring out's inner reach
(size * min boundary distance of its unit outline - half the line).
"""
from __future__ import annotations
import functools
import hashlib
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.shapes import unit_shape  # noqa: E402

FPS = 24
W, H = 1280, 720
AREA = (24, 100, 1256, 704)          # everything stays inside; the timer plate sits above-left of it
SCENARIOS = ("assemble", "spin", "marks", "recolor", "drift")
EDGE_F = 12                          # no event within 0.5 s of either end
GAP_F = 8                            # any two question-relevant events are >= 8 frames (0.33 s) apart
MIN_RUN_F = 24                       # a turning / still stretch lasts >= 1 s

TYPES = ("circle", "ellipse", "square", "rectangle", "diamond", "triangle", "pentagon", "hexagon",
         "octagon", "star", "cross", "trapezoid")
FOLD = {"circle": 0, "ellipse": 2, "square": 4, "rectangle": 2, "diamond": 2, "triangle": 3, "pentagon": 5,
        "hexagon": 6, "octagon": 8, "star": 5, "cross": 4, "trapezoid": 1}
ALIAS = 0.4                          # a turn per sample must stay under 0.4 of the symmetry period
LINES = ("solid", "solid", "dashed", "dotted", "double")

COLORS = {"red": (222, 48, 44), "orange": (243, 128, 20), "yellow": (238, 214, 46), "green": (38, 160, 62),
          "teal": (18, 170, 172), "blue": (40, 88, 222), "purple": (142, 66, 204), "pink": (236, 92, 176),
          "black": (28, 28, 30), "white": (246, 246, 246)}
THEMES = {
    "light": {"bgs": {"cream": (250, 245, 232), "paper": (244, 244, 242), "mist": (230, 238, 247),
                      "mint": (233, 243, 232), "sand": (240, 232, 214)},
              "colors": ("red", "orange", "green", "teal", "blue", "purple", "pink", "black")},
    "dark": {"bgs": {"navy": (16, 22, 42), "charcoal": (28, 28, 32), "slate": (26, 34, 46),
                     "plum": (36, 22, 40), "forest": (16, 34, 26)},
             "colors": ("red", "orange", "yellow", "green", "teal", "blue", "purple", "pink", "white")},
}
TEXTURES = ("plain", "vignette", "dots", "stripes", "grid", "noise")
MARK_KINDS = {"heart": "heart", "star": "star", "dot": "circle", "diamond": "diamond"}
WALLS = ("left", "right", "top", "bottom")


# ---- timer -----------------------------------------------------------------------
def tenths(f: int) -> int:
    """The on-screen timer reading at frame f, in tenths of a second (truncated)."""
    return f * 10 // FPS


def fmt_reading(tn: int) -> str:
    return f"{tn // 10}.{tn % 10} s"


def timer_text(f: int) -> str:
    return fmt_reading(tenths(f))


def frame_of_reading(tn: int) -> int:
    """First frame whose timer shows `tn` tenths."""
    return -(-tn * FPS // 10)


# ---- unit-shape geometry -------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def extents(kind: str, aspect: float) -> tuple[float, float]:
    """(max vertex radius, min boundary distance) of the unit outline about its centre."""
    pts = unit_shape(kind, aspect)
    big = max(math.hypot(x, y) for x, y in pts)
    small = min(_seg_dist(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts)))
    return big, small


def _seg_dist(a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, -(ax * dx + ay * dy) / (dx * dx + dy * dy)))
    return math.hypot(ax + t * dx, ay + t * dy)


@functools.lru_cache(maxsize=None)
def perimeter(kind: str, aspect: float) -> tuple[tuple, tuple]:
    """(points, cumulative arc length normalised to [0, 1]) of the closed unit outline."""
    pts = unit_shape(kind, aspect)
    cum = [0.0]
    for i in range(len(pts)):
        cum.append(cum[-1] + math.dist(pts[i], pts[(i + 1) % len(pts)]))
    tot = cum[-1]
    return pts, tuple(c / tot for c in cum), tot


def unit_point(kind: str, aspect: float, u: float) -> tuple[float, float]:
    """Point a fraction u of the way round the unit outline (by arc length)."""
    pts, cum, _ = perimeter(kind, aspect)
    u %= 1.0
    lo, hi = 0, len(pts)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cum[mid] <= u:
            lo = mid
        else:
            hi = mid
    a, b = pts[lo], pts[(lo + 1) % len(pts)]
    t = (u - cum[lo]) / max(1e-12, cum[lo + 1] - cum[lo])
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def place(ring: dict, ux: float, uy: float, cx: float, cy: float, rot: float) -> tuple[float, float]:
    th = math.radians(rot)
    c, s = math.cos(th), math.sin(th)
    r = ring["size"]
    return cx + r * (ux * c - uy * s), cy + r * (ux * s + uy * c)


def line_in(ring) -> float:
    """How far the drawn border reaches inward of the outline path (px)."""
    w = ring["width"]
    return w / 2 + (w + 3 if ring["line"] == "double" else 0)


def reach_out(ring) -> float:
    return ring["size"] * extents(ring["shape"], ring["aspect"])[0] + ring["width"] / 2


def reach_in(ring) -> float:
    return ring["size"] * extents(ring["shape"], ring["aspect"])[1] - line_in(ring)


def pad(ring) -> float:
    """Room a tab or a mark needs on each side of the outline."""
    return max(ring.get("tab_r", 0) if ring["tab"] else 0, ring.get("mark_r", 0))


MIN_BULGE = 7.0                      # px between an outline's outermost and innermost points for turning to show


def fold(ring) -> int:
    """Rotational symmetry a viewer sees: 1 with a tab; 0 (turning invisible) for a
    circle, or for an outline so small / round that its corners stick out < 7 px."""
    if ring["tab"]:
        return 1
    M, m = extents(ring["shape"], ring["aspect"])
    return FOLD[ring["shape"]] if ring["size"] * (M - m) >= MIN_BULGE else 0


def period(ring) -> float:
    f = fold(ring)
    return 360.0 / f if f else 0.0


# ---- time functions ------------------------------------------------------------------
def _n_in(a: int, b: int | None, f: int) -> int:
    """Frames j in [max(a, 1), min(f, b - 1)]: increments of segment [a, b) up to frame f."""
    hi = f if b is None else min(f, b - 1)
    return max(0, hi - max(a, 1) + 1)


def rot(ring: dict, f: int) -> float:
    """Rotation (degrees, clockwise) at frame f. Segment [f0, next f0) of `spin`
    sets the per-frame increment from frame f0 - 1 to frame f0 onward."""
    a = ring["rot0"]
    sp = ring["spin"]
    for i, (f0, w) in enumerate(sp):
        nxt = sp[i + 1][0] if i + 1 < len(sp) else None
        a += w / FPS * _n_in(f0, nxt, f)
    return a


def omega(ring: dict, j: int) -> float:
    """Turning rate (deg/s) shown between frames j - 1 and j."""
    w = 0.0
    for f0, ww in ring["spin"]:
        if f0 <= j:
            w = ww
    return w


def visible(ring: dict, f: int) -> bool:
    return ring["appear"] <= f and (ring["vanish"] is None or f < ring["vanish"])


def color_at(ring: dict, f: int) -> str:
    c = ring["color"]
    for f0, cc in ring["recolor"]:
        if f0 <= f:
            c = cc
    return c


def _reflect(p0: float, v: float, lo: float, hi: float, t: float) -> float:
    span = hi - lo
    u = (p0 - lo + v * t) % (2 * span)
    return lo + (u if u <= span else 2 * span - u)


def nest_pos(scene: dict, f: int) -> tuple[float, float]:
    n = scene["nest"]
    if not scene["arena"]:
        return n["x"], n["y"]
    lo_x, lo_y, hi_x, hi_y = bounds(scene)
    return _reflect(n["x"], n["vx"], lo_x, hi_x, f), _reflect(n["y"], n["vy"], lo_y, hi_y, f)


def bounds(scene, drop: str | None = None):
    """Range of the nest centre inside the box (a missing wall = no limit)."""
    a, R = scene["arena"], scene["R"]
    big = 1e6
    return (a[0] + R if drop != "left" else -big, a[1] + R if drop != "top" else -big,
            a[2] - R if drop != "right" else big, a[3] - R if drop != "bottom" else big)


def hits(scene, drop: str | None = None, until: float | None = None) -> list[tuple[float, str]]:
    """(time in frames, wall) of every wall hit, from the closed-form path."""
    if not scene["arena"]:
        return []
    n = scene["nest"]
    lo_x, lo_y, hi_x, hi_y = bounds(scene, drop)
    T = scene["n_frames"] - 1 if until is None else until
    out = []
    for p0, v, lo, hi, walls in ((n["x"], n["vx"], lo_x, hi_x, ("left", "right")),
                                 (n["y"], n["vy"], lo_y, hi_y, ("top", "bottom"))):
        p, vv, t = p0, v, 0.0
        while True:
            if vv > 0:
                dt, wall, nxt = (hi - p) / vv, walls[1], hi
            else:
                dt, wall, nxt = (lo - p) / vv, walls[0], lo
            t += dt
            if t > T or abs(nxt) >= 1e5:
                break
            out.append((t, wall))
            p, vv = nxt, -vv
    return sorted(out)


def ring_centre(scene, k: int, f: int) -> tuple[float, float]:
    x, y = nest_pos(scene, f)
    r = scene["rings"][k]
    return x + r["dx"], y + r["dy"]


def mark_visible(m: dict, f: int) -> bool:
    return m["add"] <= f and (m["remove"] is None or f < m["remove"])


def mark_pos(scene, m: dict, f: int) -> tuple[float, float]:
    ring = scene["rings"][m["ring"]]
    ux, uy = unit_point(ring["shape"], ring["aspect"], m["u"])
    cx, cy = ring_centre(scene, m["ring"], f)
    return place(ring, ux, uy, cx, cy, rot(ring, f))


# ---- truth helpers used by questions and checks ----------------------------------------
def label(scene, k: int) -> str:
    r = scene["rings"][k]
    return r["shape"] if scene["scenario"] == "recolor" else f"{r['color']} {r['shape']}"


def sign(w: float) -> int:
    return 0 if abs(w) < 1e-9 else (1 if w > 0 else -1)


def runs(scene, k: int) -> list[tuple[int, int, int]]:
    """Turning runs of ring k while on screen: (first frame showing it, last frame, direction)
    with direction +1 clockwise, -1 counterclockwise, 0 still. Interval (j-1, j] is
    shown when frames j-1 and j are both on screen."""
    r = scene["rings"][k]
    nf = scene["n_frames"]
    lo = r["appear"] + 1
    hi = (nf if r["vanish"] is None else r["vanish"]) - 1
    out = []
    cuts = sorted({lo} | {f0 for f0, _ in r["spin"] if lo < f0 <= hi})
    for i, a in enumerate(cuts):
        b = cuts[i + 1] - 1 if i + 1 < len(cuts) else hi
        d = sign(omega(r, a)) if fold(r) else 0
        if out and out[-1][2] == d:
            out[-1] = (out[-1][0], b, d)
        else:
            out.append((a, b, d))
    return out


def changes(scene, k: int) -> list[int]:
    """Frames at which ring k visibly changes how it turns."""
    return [a for a, _, _ in runs(scene, k)[1:]]


def dir_at(scene, k: int, f: int) -> int | None:
    for a, b, d in runs(scene, k):
        if a <= f <= b:
            return d
    return None


def on_screen(scene, f: int) -> list[int]:
    return [k for k, r in enumerate(scene["rings"]) if visible(r, f)]


def bounce_frames(scene, drop=None) -> list[tuple[int, str]]:
    return [(round(t), w) for t, w in hits(scene, drop)]


def events(scene) -> list[tuple[int, str]]:
    """Every question-relevant event frame with its kind."""
    ev = []
    nf = scene["n_frames"]
    for k, r in enumerate(scene["rings"]):
        if r["appear"] > 0:
            ev.append((r["appear"], f"appear {k}"))
        if r["vanish"] is not None:
            ev.append((r["vanish"], f"vanish {k}"))
        ev += [(f, f"recolor {k}") for f, _ in r["recolor"]]
        ev += [(f, f"turn {k}") for f in changes(scene, k)]
    for i, m in enumerate(scene["marks"]):
        ev.append((m["add"], f"add {i}"))
        if m["remove"] is not None:
            ev.append((m["remove"], f"remove {i}"))
    ev += [(f, f"hit {w}") for f, w in bounce_frames(scene)]
    return sorted(e for e in ev if e[0] <= nf)


def signature(scene) -> str:
    def rnd(x):
        if isinstance(x, float):
            return round(x, 2)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items() if k != "tries"}
        if isinstance(x, list):
            return [rnd(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(rnd(scene), sort_keys=True).encode()).hexdigest()[:16]


# ---- scene building ----------------------------------------------------------------------
def _style(rng):
    theme = rng.choice(sorted(THEMES))
    return {"theme": theme, "bg_name": rng.choice(sorted(THEMES[theme]["bgs"])), "texture": rng.choice(TEXTURES),
            "tex_step": rng.randint(26, 64), "tex_amp": rng.randint(5, 11)}


def _free(rng, lo, hi, taken, gap=GAP_F, tries=60):
    """A frame in [lo, hi] at least `gap` from every frame in `taken` (None if none found)."""
    for _ in range(tries):
        if hi < lo:
            return None
        f = rng.randint(lo, hi)
        if all(abs(f - t) >= gap for t in taken):
            return f
    return None


ROUND = ("circle", "octagon", "hexagon", "pentagon", "square")       # inner/outer reach ratio >= 0.7


def _shapes(rng, n, unique):
    """Shape sequence, outer to inner. Narrow shapes (triangle, star, cross, diamond,
    rectangle, ellipse, trapezoid) cost a lot of room, so a deep nest gets fewer."""
    narrow_max = {3: 3, 4: 2, 5: 2, 6: 1, 7: 1}[n] + (max(0, n - 5) if unique else 0)
    pool = list(TYPES)
    seq = []
    for i in range(n):
        n_narrow = sum(1 for t in seq if t not in ROUND)
        cand = [t for t in pool if (not unique or t not in seq) and (t in ROUND or n_narrow < narrow_max)]
        if "star" in seq or "cross" in seq:
            cand = [t for t in cand if t not in ("star", "cross")]
        seq.append(rng.choice(cand))
    return seq


def _nest(rng, scenario, n, style, with_marks=False, tabs=0.0):
    """Ring geometry, outer to inner, or None. Up to 6 tries, each with a tighter fit."""
    unique = scenario == "recolor"
    need = {3: 0.12, 4: 0.18, 5: 0.24, 6: 0.3, 7: 0.36}[n]       # product of inner/outer reach ratios
    for t in range(8):
        for _ in range(30):
            seq = _shapes(rng, n, unique)
            prod = math.prod(extents(x, 1.45)[1] / extents(x, 1.45)[0] for x in seq[:-1])
            if prod >= need:
                break
        got = _nest_try(rng, scenario, seq, style, with_marks, tabs, 1 - t / 8)
        if got is not None:
            return got
    return None


def _nest_try(rng, scenario, seq, style, with_marks, tabs, slack):
    n = len(seq)
    pal = list(THEMES[style["theme"]]["colors"])
    R0 = rng.uniform(95 + 15 * (n - 3), 170) if scenario == "drift" else rng.uniform(150 + 25 * max(0, n - 3), 284)
    loose = 1 - rng.uniform(0, 0.24) * slack * (4 / max(4, n))    # share of the free room each inner ring takes
    gap = rng.uniform(6, 6 + 10 * slack) if scenario == "drift" else rng.uniform(4, 4 + 6 * slack) if with_marks \
        else rng.uniform(7, 7 + 17 * slack * 4 / max(4, n))
    offset = rng.random() < 0.4
    th = rng.uniform(0, 2 * math.pi)
    step = rng.uniform(3, 10) if offset else 0.0
    twist = rng.choice([0.0, 0.0, rng.uniform(5, 20)])
    mark_r = round(rng.uniform(12, 16), 1) if with_marks else 0
    width = rng.choice(["thin", "mixed", "thick"])
    rings, ids = [], set()
    for k, shp in enumerate(seq):
        aspect = round(rng.uniform(1.3, 1.65), 2) if shp in ("ellipse", "rectangle") else 1.0
        w = {"thin": rng.randint(5, 6), "mixed": rng.randint(5, 8), "thick": rng.randint(6, 8)}[width]
        tab = rng.random() < tabs
        r = {"shape": shp, "aspect": aspect, "width": w, "line": "solid", "tab": tab,
             "tab_r": max(7, w + 3) if tab else 0, "mark_r": mark_r,
             "dx": round(k * step * math.cos(th), 1), "dy": round(k * step * math.sin(th), 1),
             "rot0": round((rng.uniform(0, 360) if k == 0 else
                            rings[-1]["rot0"] + twist + rng.choice([0.0, 0.0, rng.uniform(0, 90)])) % 360, 1)}
        M, m = extents(shp, aspect)
        if k == 0:
            size = (R0 - w / 2) / M
        else:
            prev = rings[-1]
            room = reach_in(prev) - pad(prev) - pad(r) - gap - math.hypot(r["dx"] - prev["dx"], r["dy"] - prev["dy"])
            size = (room * (loose if k < n - 1 or room < 60 else min(loose, rng.uniform(0.55, 1.0))) - w / 2) / M
        r["size"] = round(size - 0.05, 2)          # rounding never breaks containment
        ro = reach_out(r)
        # broken lines need room to keep the shape readable; small outlines stay solid
        r["line"] = rng.choice(LINES if ro >= 70 else ("solid", "solid", "double") if ro >= 50 else ("solid",))
        if ro < 24 or reach_in(r) < 12:
            return None
        for _ in range(40):
            c = rng.choice(pal)
            if (c, shp) not in ids:
                break
        else:
            return None
        ids.add((c, shp))
        r["color"] = c
        rings.append(r)
    R = max(math.hypot(r["dx"], r["dy"]) + reach_out(r) + pad(r) for r in rings)
    return rings, R


def _rotations(rng, scene, p_turn, w_range=(28.0, 96.0)):
    """Steady turning (or none) for every ring that can visibly turn."""
    for r in scene["rings"]:
        if not fold(r) or rng.random() >= p_turn:
            r["spin"] = [[0, 0.0]]
        else:
            r["spin"] = [[0, round(rng.choice([-1, 1]) * rng.uniform(*w_range), 1)]]


def _spin_events(rng, scene, n_changes, taken, w_range=(28.0, 96.0)):
    nf = scene["n_frames"]
    turnable = [k for k, r in enumerate(scene["rings"]) if fold(r)]
    if not turnable:
        return False
    for _ in range(n_changes):
        k = rng.choice(turnable)
        r = scene["rings"][k]
        if len(r["spin"]) > 3:                 # at most 3 changes per outline
            continue
        last = r["spin"][-1][0]
        f = _free(rng, max(last + MIN_RUN_F, MIN_RUN_F), nf - MIN_RUN_F, taken)
        if f is None:
            continue
        w = r["spin"][-1][1]
        if w == 0:
            nw = round(rng.choice([-1, 1]) * rng.uniform(*w_range), 1)
        elif rng.random() < 0.62:
            nw = round(-w * rng.uniform(0.75, 1.3), 1)
        else:
            nw = 0.0
        r["spin"].append([f, nw])
        taken.append(f)
    return True


def _assemble(rng, scene, taken):
    rings, nf = scene["rings"], scene["n_frames"]
    n = len(rings)
    order = rng.choice(["outside-in", "inside-out", "mixed"])
    seq = list(range(n))
    if order == "inside-out":
        seq.reverse()
    elif order == "mixed":
        while seq == sorted(seq) or seq == sorted(seq, reverse=True):
            rng.shuffle(seq)
    span = nf - 2 * EDGE_F - MIN_RUN_F
    step = min(40, span // n)
    if step < 12:
        return False
    f = rng.randint(EDGE_F, EDGE_F + 10)
    for k in seq:
        rings[k]["appear"] = f
        taken.append(f)
        f += rng.randint(12, step)
    if f - 12 > nf - MIN_RUN_F:
        return False
    scene["order"] = order
    if n >= 4 and rng.random() < 0.45:
        for k in rng.sample(range(n), rng.randint(1, 2)):
            v = _free(rng, rings[k]["appear"] + MIN_RUN_F, nf - MIN_RUN_F, taken)
            if v is not None:
                rings[k]["vanish"] = v
                taken.append(v)
    return True


def _marks(rng, scene, taken):
    rings, nf = scene["rings"], scene["n_frames"]
    style = scene["style"]
    pal = list(THEMES[style["theme"]]["colors"])
    marks = []
    n_add = rng.randint(4, 10)
    roomy = [k for k, r in enumerate(rings) if reach_in(r) >= 2.5 * r["mark_r"]]   # a mark stays small next to its outline
    if len(roomy) < 2:
        return False
    hot = rng.sample(roomy, min(len(roomy), rng.randint(2, 4)))
    for _ in range(n_add):
        k = rng.choice(hot) if rng.random() < 0.8 else rng.choice(roomy)
        r = rings[k]
        mr = r["mark_r"]
        tot = perimeter(r["shape"], r["aspect"])[2] * r["size"]
        need = 2.8 * mr / tot                       # arc fraction between two marks on one outline
        used = [m["u"] for m in marks if m["ring"] == k]
        u = None
        for _ in range(40):
            cand = round(rng.random(), 4)
            if all(min(abs(cand - x), 1 - abs(cand - x)) >= need for x in used):
                u = cand
                break
        if u is None:
            continue
        f = _free(rng, EDGE_F, nf - MIN_RUN_F, taken)
        if f is None:
            continue
        kind = rng.choice(sorted(MARK_KINDS))
        col = rng.choice([c for c in pal if c != r["color"]])
        marks.append({"ring": k, "u": u, "kind": kind, "color": col, "add": f, "remove": None})
        taken.append(f)
    if rng.random() < 0.6:
        for m in rng.sample(marks, min(len(marks), rng.randint(1, 3))):
            v = _free(rng, m["add"] + MIN_RUN_F, nf - EDGE_F, taken)
            if v is not None:
                m["remove"] = v
                taken.append(v)
    marks.sort(key=lambda m: (m["add"], m["ring"]))
    scene["marks"] = marks
    return len(marks) >= 3


def _recolor(rng, scene, taken):
    rings, nf = scene["rings"], scene["n_frames"]
    pal = list(THEMES[scene["style"]["theme"]]["colors"])
    n_ch = rng.randint(3, 7)
    movers = rng.sample(range(len(rings)), min(len(rings), rng.randint(2, max(2, len(rings) - 1))))
    for _ in range(n_ch):
        k = rng.choice(movers)
        r = rings[k]
        last = r["recolor"][-1][0] if r["recolor"] else 0
        f = _free(rng, max(EDGE_F, last + MIN_RUN_F), nf - MIN_RUN_F, taken)
        if f is None:
            continue
        cur = color_at(r, f)
        r["recolor"].append([f, rng.choice([c for c in pal if c != cur])])
        taken.append(f)
    return sum(len(r["recolor"]) for r in rings) >= 3


def _drift(rng, scene, taken):
    R = scene["R"]
    x0 = AREA[0] + rng.randint(0, 70)
    y0 = AREA[1] + rng.randint(0, 40)
    x1 = AREA[2] - rng.randint(0, 70)
    y1 = AREA[3] - rng.randint(0, 30)
    if x1 - x0 < 3.6 * R or y1 - y0 < 2.5 * R:
        return False
    scene["arena"] = [x0, y0, x1, y1]
    nf = scene["n_frames"]
    for _ in range(40):
        sp = rng.uniform(110, 300) / FPS
        th = rng.uniform(0, 2 * math.pi)
        vx, vy = sp * math.cos(th), sp * math.sin(th)
        if min(abs(vx), abs(vy)) < 1.0:
            continue
        scene["nest"] = {"x": round(rng.uniform(x0 + R + 2, x1 - R - 2), 1),
                         "y": round(rng.uniform(y0 + R + 2, y1 - R - 2), 1), "vx": round(vx, 3), "vy": round(vy, 3)}
        hs = hits(scene)
        ts = [t for t, _ in hs]
        if 1 <= len(hs) <= 7 and all(EDGE_F <= t <= nf - EDGE_F and abs(t - math.floor(t) - 0.5) >= 0.12 for t in ts) \
                and all(b - a >= GAP_F for a, b in zip(ts, ts[1:])):
            for f, _ in bounce_frames(scene):
                taken.append(f)
            return True
    return False


def _attempt(rng, scenario):
    nf = rng.randint(156, 240)
    style = _style(rng)
    n = {"assemble": rng.randint(3, 7), "spin": rng.randint(3, 7), "marks": rng.randint(3, 5),
         "recolor": rng.randint(3, 6), "drift": rng.choice([3, 3, 4, 4, 5])}[scenario]
    tabs = {"assemble": 0.3, "spin": rng.choice([0.0, 0.5, 0.8]), "marks": 0.0, "recolor": 0.3, "drift": 0.4}[scenario]
    got = _nest(rng, scenario, n, style, with_marks=scenario == "marks", tabs=tabs)
    if got is None:
        return None
    rings, R = got
    for r in rings:
        r.update({"appear": 0, "vanish": None, "recolor": [], "spin": [[0, 0.0]]})
        if not r["tab"]:
            r.pop("tab_r")
        if scenario != "marks":
            r.pop("mark_r")
    scene = {"scenario": scenario, "n_frames": nf, "style": style, "rings": rings, "marks": [],
             "arena": None, "R": math.ceil(R * 10) / 10, "order": None}
    if scenario != "drift":
        cx = rng.uniform(AREA[0] + R + 2, AREA[2] - R - 2)
        cy = rng.uniform(AREA[1] + R + 2, AREA[3] - R - 2)
        if AREA[1] + R + 2 > AREA[3] - R - 2:
            return None
        scene["nest"] = {"x": round(cx, 1), "y": round(cy, 1), "vx": 0.0, "vy": 0.0}
    taken = []
    ok = True
    if scenario == "spin":
        _rotations(rng, scene, rng.choice([0.55, 0.75, 0.9]))
        ok = _spin_events(rng, scene, rng.randint(2, 6), taken)
    elif scenario == "assemble":
        _rotations(rng, scene, 0.55)
        ok = _assemble(rng, scene, taken)
    elif scenario == "marks":
        _rotations(rng, scene, 0.5, (18.0, 60.0))
        ok = _marks(rng, scene, taken)
    elif scenario == "recolor":
        _rotations(rng, scene, 0.5)
        ok = _recolor(rng, scene, taken)
    else:
        ok = _drift(rng, scene, taken)
        _rotations(rng, scene, 0.6)
        if ok and rng.random() < 0.35:
            _spin_events(rng, scene, 1, taken)
    if not ok:
        return None
    return scene


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    tries = 0
    while True:
        tries += 1
        scene = _attempt(rng, scenario)
        if scene is not None and not check(scene):
            scene["tries"] = tries
            return scene


# ---- checks ------------------------------------------------------------------------------
def check(scene: dict) -> list[str]:
    errs = []
    sc, nf, rings = scene["scenario"], scene["n_frames"], scene["rings"]
    n = len(rings)
    if sc not in SCENARIOS:
        errs.append("unknown scenario")
    if not 6 * FPS <= nf <= 10 * FPS:
        errs.append("clip length out of range")
    if not 3 <= n <= 7:
        errs.append("ring count out of range")
    labs = [label(scene, k) for k in range(n)]
    if len(set(labs)) != n:
        errs.append("two outlines share a name")
    if sc == "recolor" and len({r["shape"] for r in rings}) != n:
        errs.append("recolor outlines must have distinct shapes")
    pal = THEMES[scene["style"]["theme"]]["colors"]
    for r in rings:
        if r["color"] not in pal or any(c not in pal for _, c in r["recolor"]):
            errs.append("a border colour doesn't suit the background")
        if r["width"] < 3:
            errs.append("a line thinner than 3 px")
    # containment at any rotation, sizes, frame
    for a, b in zip(rings, rings[1:]):
        room = reach_in(a) - pad(a) - pad(b) - math.hypot(a["dx"] - b["dx"], a["dy"] - b["dy"])
        if reach_out(b) + 3 > room:
            errs.append("two outlines come too close")
    if reach_out(rings[-1]) < 24:
        errs.append("innermost outline too small")
    R = max(math.hypot(r["dx"], r["dy"]) + reach_out(r) + pad(r) for r in rings)
    if R > scene["R"] + 1e-6:
        errs.append("bounding radius understated")
    box = scene["arena"] or AREA
    for f in range(0, nf, 4):
        x, y = nest_pos(scene, f)
        if x - R < box[0] - 0.5 or x + R > box[2] + 0.5 or y - R < box[1] - 0.5 or y + R > box[3] + 0.5:
            errs.append("the nest leaves its area")
            break
    if scene["arena"] and (scene["arena"][0] < AREA[0] or scene["arena"][1] < AREA[1]
                           or scene["arena"][2] > AREA[2] or scene["arena"][3] > AREA[3]):
        errs.append("box outside the action area")
    # turning: visible at 24 fps, runs long enough
    for k, r in enumerate(rings):
        if not fold(r) and any(w for _, w in r["spin"]):
            errs.append(f"{labs[k]} turns but its turning can't be seen")
        P = period(r)
        for _, w in r["spin"]:
            if P and abs(w) / FPS > ALIAS * P:
                errs.append(f"{labs[k]} turns too fast for its symmetry")
        for a, b, _ in runs(scene, k):
            if b - a + 1 < MIN_RUN_F and not (a == r["appear"] + 1 and b - a + 1 >= 12):
                errs.append(f"{labs[k]}: a turning stretch shorter than 1 s")
        for _, w in r["spin"]:
            if w and not 12 <= abs(w) <= 130:
                errs.append(f"{labs[k]}: turning rate out of range")
    # appearance
    for k, r in enumerate(rings):
        if r["vanish"] is not None and r["vanish"] - r["appear"] < MIN_RUN_F:
            errs.append(f"{labs[k]} is on screen for under 1 s")
        if r["appear"] > nf - MIN_RUN_F:
            errs.append(f"{labs[k]} appears too late")
    if sc == "assemble" and not any(visible(r, nf - 1) for r in rings):
        errs.append("nothing left at the end")
    # marks
    for i, m in enumerate(scene["marks"]):
        r = rings[m["ring"]]
        if m["color"] == r["color"]:
            errs.append("a mark has its outline's colour")
        if reach_in(r) < 2.5 * r["mark_r"]:
            errs.append("a mark on an outline too small for it")
        if m["remove"] is not None and m["remove"] - m["add"] < MIN_RUN_F:
            errs.append("a mark stays under 1 s")
        tot = perimeter(r["shape"], r["aspect"])[2] * r["size"]
        for m2 in scene["marks"][i + 1:]:
            if m2["ring"] == m["ring"]:
                du = abs(m["u"] - m2["u"])
                if min(du, 1 - du) * tot < 2.6 * r["mark_r"]:
                    errs.append("two marks too close on one outline")
    # events: away from the ends and from each other
    ev = events(scene)
    for f, kind in ev:
        if not EDGE_F <= f <= nf - EDGE_F:
            errs.append(f"event '{kind}' too close to a clip end")
    for (f1, k1), (f2, k2) in zip(ev, ev[1:]):
        if f2 - f1 < GAP_F:
            errs.append(f"events '{k1}' and '{k2}' too close in time")
    # drift: clean hits (also for every missing-wall what-if)
    if scene["arena"]:
        hs = hits(scene)
        if not 1 <= len(hs) <= 7:
            errs.append("too few or too many wall hits")
        for drop in (None,):                 # a missing wall only drops later hits: same margins
            hd = hits(scene, drop)
            for t, _ in hd:
                if abs(t - math.floor(t) - 0.5) < 0.12:
                    errs.append("a wall hit halfway between frames")
                if not EDGE_F <= t <= nf - EDGE_F:
                    errs.append("a wall hit too close to a clip end")
            for (t1, _), (t2, _) in zip(hd, hd[1:]):
                if t2 - t1 < GAP_F:
                    errs.append("two wall hits too close in time")
    if not errs:
        import questions
        sup = questions.supply(scene)
        if any(s < 3 for s in sup):
            errs.append(f"too few candidate questions per level {sup}")
    return errs
