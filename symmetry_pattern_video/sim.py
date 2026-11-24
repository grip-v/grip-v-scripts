"""Scene generation for symmetry_pattern_video (v3, 10-question ladder).

A pattern of shapes is arranged under ONE symmetry group about a centre
point: rotational (2/3/4/5/6/8-fold), mirror (left-right, up-down, both) or
dihedral (3 or 4 turns plus as many mirror lines). The pattern is a few
"orbits": one base shape (outline, colour, fill, line style, size, turn)
carried to every position by every element of the group, so shapes turn
like a pinwheel under rotations and appear as true mirror images under
reflections. About half the scenes are broken: exactly one shape differs
from its orbit partners (fill, colour, outline, size, turn, place, or, in
the waves scenario, the way it is traced).

Four scenarios (`scene["scenario"]`):

    waves     the orbits are traced one after another; each shape's outline
              grows from a start corner with a dot riding the tip, all in one
              turning direction (one shape may go the other way, or stop short)
    reveal    the shapes pop in one at a time in random order, so the
              picture's symmetry changes as it fills
    assemble  every orbit glides in from scattered starting spots, turning
              as it goes, and settles into place; orbits start and settle at
              different times
    turns     the finished pattern sits on a disc with a white notch on its
              rim; the disc is turned (clockwise / counter-clockwise) and
              flipped (left-right / upside down) in 2-4 moves

Coordinates: pattern positions are pixels relative to the centre with y UP;
screen = (cx + x, cy - y). Time is in frames (24 fps); the clock on frame f
reads floor(10 f / 24) tenths of a second. Every timed event happens on a
whole frame.
"""
from __future__ import annotations
import functools
import json
import math
import random

W, H, FPS = 1280, 720, 24
MAX_FRAMES = 240
SCENARIOS = ("waves", "reveal", "assemble", "turns")
TIMER_BOX = (14, 14, 196, 70)
TIMER_FONT = 36
EDGE_PAD = 12                    # frames: no timed event closer than this to either end
GAP = 7                          # frames: compared / neighbouring events at least this far apart

SHAPE_TYPES = ("triangle", "square", "diamond", "pentagon", "hexagon", "star", "circle", "kite",
               "chevron", "bolt")
SHAPE_NAMES = {"triangle": "triangle", "square": "square", "diamond": "diamond", "pentagon": "pentagon",
               "hexagon": "hexagon", "star": "star", "circle": "circle", "kite": "kite",
               "chevron": "chevron", "bolt": "lightning bolt"}
PLURAL = {k: (v + "es" if v.endswith("x") else v + "s") for k, v in SHAPE_NAMES.items()}
COLORS = {"red": (235, 64, 52), "orange": (247, 146, 30), "yellow": (245, 214, 50), "lime": (152, 222, 48),
          "green": (34, 160, 80), "cyan": (40, 200, 225), "blue": (60, 110, 240), "purple": (155, 85, 225),
          "pink": (245, 125, 200), "white": (240, 240, 240)}
FILLS = ("outline", "solid", "dot", "double")
FILL_NAMES = {"outline": "an empty outline", "solid": "solid filled", "dot": "an outline with a centre dot",
              "double": "a double outline"}
BACKGROUNDS = [((22, 24, 30), (36, 40, 52)), ((18, 30, 34), (30, 52, 58)), ((34, 24, 40), (54, 38, 64)),
               ((40, 40, 44), (24, 24, 28)), ((214, 216, 222), (190, 194, 204)), ((228, 220, 200), (206, 196, 172))]

GROUPS = {   # name -> (rotation order, mirror lines) of the finished, unbroken pattern
    "rotational_2": (2, 0), "rotational_3": (3, 0), "rotational_4": (4, 0), "rotational_5": (5, 0),
    "rotational_6": (6, 0), "rotational_8": (8, 0), "mirror_left_right": (1, 1), "mirror_up_down": (1, 1),
    "mirror_both": (2, 2), "dihedral_3": (3, 3), "dihedral_4": (4, 4)}
N_ORBITS = {"waves": (2, 5), "reveal": (2, 4), "assemble": (2, 4), "turns": (2, 4)}
MAX_SHAPES = {"waves": 16, "reveal": 12, "assemble": 16, "turns": 16}
STATIC_BREAKS = ("fill", "color", "shape", "size", "rotation", "position")
BREAKS = {"waves": STATIC_BREAKS + ("direction", "incomplete"), "reveal": STATIC_BREAKS,
          "assemble": STATIC_BREAKS, "turns": STATIC_BREAKS}
BREAK_NAMES = {None: "every shape matches its partners", "fill": "one shape has a different fill",
               "color": "one shape has a different colour", "shape": "one shape has a different outline",
               "size": "one shape is a different size", "rotation": "one shape is turned differently",
               "position": "one shape is out of place", "direction": "one shape is traced the other way round",
               "incomplete": "one shape's outline is never finished"}
OCTANTS = ("right", "top-right", "top", "top-left", "left", "bottom-left", "bottom", "bottom-right")
TOL = 1.0                         # px, descriptor symmetry tolerance


# ---------------------------------------------------------------- time / clock

def tenths(f: int) -> int:
    return (10 * f) // FPS


def clock_text(k: int) -> str:
    return f"{k // 10}.{k % 10} s"


def frames_showing(k: int) -> tuple[int, int]:
    lo = -(-k * FPS // 10)
    hi = -(-(k + 1) * FPS // 10) - 1
    return lo, hi


# ---------------------------------------------------------------- group algebra (y up)

def group_elements(group: str) -> list[tuple[float, int]]:
    """(theta, flip): p -> R(theta) . F^flip . p, F: (x, y) -> (x, -y)."""
    if group.startswith("rotational_"):
        k = int(group.split("_")[1])
        return [(360.0 * i / k, 0) for i in range(k)]
    if group == "mirror_up_down":            # mirror line = horizontal axis
        return [(0.0, 0), (0.0, 1)]
    if group == "mirror_left_right":         # mirror line = vertical axis
        return [(0.0, 0), (180.0, 1)]
    if group == "mirror_both":
        return [(0.0, 0), (0.0, 1), (180.0, 1), (180.0, 0)]
    k = int(group.split("_")[1])
    return [(360.0 * i / k, 0) for i in range(k)] + [(360.0 * i / k, 1) for i in range(k)]


def mirror_axes(group: str) -> list[float]:
    return sorted({(t / 2.0) % 180.0 for t, f in group_elements(group) if f})


def elem_matrix(elem) -> list[list[float]]:
    th, fl = elem
    c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
    return [[c, s], [s, -c]] if fl else [[c, -s], [s, c]]


def matrix_elem(M) -> tuple[float, int]:
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    return (round(math.degrees(math.atan2(M[1][0], M[0][0])) % 360.0, 6), 1 if det < 0 else 0)


def matmul(A, B):
    return [[A[i][0] * B[0][j] + A[i][1] * B[1][j] for j in range(2)] for i in range(2)]


def apply(M, p):
    return (M[0][0] * p[0] + M[0][1] * p[1], M[1][0] * p[0] + M[1][1] * p[1])


def centroid(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def octant(x: float, y: float) -> str:
    a = math.degrees(math.atan2(y, x)) % 360.0
    return OCTANTS[int(((a + 22.5) % 360.0) // 45)]


def octant_margin(x: float, y: float) -> float:
    """Angular distance (deg) to the nearest octant boundary."""
    a = (math.degrees(math.atan2(y, x)) - 22.5) % 45.0
    return min(a, 45.0 - a)


# ---------------------------------------------------------------- shapes

def _regular(n, start=90.0):
    return [(math.cos(math.radians(start + i * 360.0 / n)), math.sin(math.radians(start + i * 360.0 / n)))
            for i in range(n)]


UNIT = {
    "triangle": _regular(3), "square": _regular(4, 45.0),
    "diamond": [(1.0, 0.0), (0.0, 0.62), (-1.0, 0.0), (0.0, -0.62)],
    "pentagon": _regular(5), "hexagon": _regular(6, 0.0),
    "star": [((1.0 if i % 2 == 0 else 0.46) * math.cos(math.radians(90 + 36 * i)),
              (1.0 if i % 2 == 0 else 0.46) * math.sin(math.radians(90 + 36 * i))) for i in range(10)],
    "circle": _regular(40, 0.0),
    "kite": [(1.0, 0.0), (0.0, 0.52), (-0.72, 0.0), (0.0, -0.52)],
    "chevron": [(1.0, 0.0), (-0.65, 0.78), (-0.18, 0.0), (-0.65, -0.78)],
    "bolt": [(0.28, 1.0), (-0.55, -0.08), (0.06, -0.08), (-0.28, -1.0), (0.55, 0.08), (-0.06, 0.08)],
}
# the turn after which an outline looks the same (degrees; circle: any)
SELF_TURN = {"triangle": 120, "square": 90, "diamond": 180, "pentagon": 72, "hexagon": 60, "star": 72,
             "circle": 0, "kite": 360, "chevron": 360, "bolt": 180}


def outline(s: dict) -> list[tuple[float, float]]:
    """Outline vertices (y up, relative to the centre), in the shape's base
    order (vertex 0 is where tracing starts)."""
    c, sn = math.cos(math.radians(s["rot"])), math.sin(math.radians(s["rot"]))
    out = []
    for ux, uy in UNIT[s["type"]]:
        if s["mirrored"]:
            uy = -uy
        out.append((s["x"] + s["size"] * (ux * c - uy * sn), s["y"] + s["size"] * (ux * sn + uy * c)))
    return out


def _sig(s: dict):
    return (s["type"], s["color"], s["fill"], s["dash"], round(s["size"], 3), round(s.get("cap", 1.0), 3))


def _match(a, b, tol):
    used = [False] * len(b)
    for p in a:
        for j, q in enumerate(b):
            if not used[j] and abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol:
                used[j] = True
                break
        else:
            return False
    return True


def _key_pts(s):
    return [(s["x"], s["y"])] if s["type"] == "circle" else outline(s)


def is_symmetry(shapes: list[dict], elem, tol: float = TOL) -> bool:
    """Applying `elem` (about the centre) reproduces the same picture."""
    M = elem_matrix(elem)
    items = [(_sig(s), _key_pts(s)) for s in shapes]
    used = [False] * len(items)
    for sig, pts in items:
        moved = [apply(M, p) for p in pts]
        for j, (sig2, pts2) in enumerate(items):
            if not used[j] and sig2 == sig and _match(moved, pts2, tol):
                used[j] = True
                break
        else:
            return False
    return True


def detect(shapes: list[dict], tol: float = TOL) -> tuple[int, int]:
    """(rotation order, number of mirror lines) about the centre."""
    if not shapes:
        return 1, 0
    order = 1
    for m in range(12, 1, -1):
        if is_symmetry(shapes, (360.0 / m, 0), tol):
            order = m
            break
    cands = set()
    for a in shapes:
        for b in shapes:
            if _sig(a) != _sig(b):
                continue
            aa, ab = math.degrees(math.atan2(a["y"], a["x"])), math.degrees(math.atan2(b["y"], b["x"]))
            cands.add(round(((aa + ab) / 2.0) % 180.0, 3))
            cands.add(round(((aa + ab) / 2.0 + 90.0) % 180.0, 3))
    axes = []
    for ax in sorted(cands):
        if any(min(abs(ax - o), 180 - abs(ax - o)) < 0.5 for o in axes):
            continue
        if is_symmetry(shapes, (2.0 * ax, 1), tol):
            axes.append(ax)
    return order, len(axes)


def min_turn(order: int) -> int:
    return 0 if order == 1 else round(360 / order)


# ---------------------------------------------------------------- pattern construction

def _lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _style(rng):
    bg = rng.choice(BACKGROUNDS)
    light = _lum(bg[0]) > 128
    return {"bg": [list(bg[0]), list(bg[1])], "light": light, "stroke": rng.choice([3, 4, 5]),
            "dot_r": rng.choice([5, 6, 7]), "bg_style": rng.choice(["flat", "gradient", "vignette"]),
            "disc": list(tuple(max(0, min(255, c + (-18 if light else 16))) for c in bg[0])),
            "rim": list((90, 96, 110) if light else (150, 156, 170)),
            "notch": list((30, 30, 34) if light else (250, 250, 250))}


def _place_centre(rng, rp):
    for _ in range(200):
        cx = rng.uniform(rp + 24, W - rp - 24)
        cy = rng.uniform(rp + 16, H - rp - 16)
        # nearest point of the timer box to the centre must be outside the pattern circle
        nx = min(max(cx, TIMER_BOX[0]), TIMER_BOX[2])
        ny = min(max(cy, TIMER_BOX[1]), TIMER_BOX[3])
        if math.hypot(cx - nx, cy - ny) > rp + 14:
            return round(cx, 1), round(cy, 1)
    return None


def _orbits(rng, group, n_orb, rp, colors, smin, smax):
    elems = group_elements(group)
    axes = mirror_axes(group)
    placed = []
    shapes = []
    types = [rng.choice(SHAPE_TYPES) for _ in range(n_orb)] if rng.random() < 0.75 else \
        [rng.choice(SHAPE_TYPES)] * n_orb
    for o in range(n_orb):
        size = round(rng.uniform(smin, smax), 2)
        rot = round(rng.uniform(0, 360), 2)
        fill = rng.choice(FILLS)
        dash = rng.random() < 0.3 and fill in ("outline", "dot")
        for _ in range(300):
            r = rng.uniform(max(0.22 * rp, size * 1.6 + 10), rp - size - 4)
            a = rng.uniform(0, 360)
            if any(min((a - x) % 180, 180 - (a - x) % 180) < 9.0 for x in axes):
                continue
            base = (r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
            cs = [apply(elem_matrix(e), base) for e in elems]
            ok = all(math.dist(c, p) >= size + pr + 22 for c in cs for p, pr in placed)
            ok = ok and all(math.dist(cs[i], cs[j]) >= 2 * size + 22 for i in range(len(cs))
                            for j in range(i + 1, len(cs)))
            if ok:
                break
        else:
            return None
        placed += [(c, size) for c in cs]
        for th, fl in elems:
            x, y = apply(elem_matrix((th, fl)), base)
            shapes.append({"orbit": o, "type": types[o], "color": colors[o], "fill": fill, "dash": dash,
                           "size": size, "x": round(x, 3), "y": round(y, 3),
                           "rot": round((th + (-rot if fl else rot)) % 360.0, 4), "mirrored": bool(fl),
                           "cap": 1.0, "dir": None})
    return shapes


def _break(rng, scenario, shapes, colors, k):
    # in 2-copy groups a recoloured / reshaped copy would leave a "group" of one
    kind = rng.choice([b for b in BREAKS[scenario] if k > 2 or b not in ("color", "shape")])
    idxs = list(range(len(shapes)))
    if kind == "rotation":
        idxs = [i for i in idxs if shapes[i]["type"] not in ("circle",)]
    if not idxs:
        return None
    k = rng.choice(idxs)
    s = dict(shapes[k])
    if kind == "fill":
        s["fill"] = rng.choice([f for f in FILLS if f != s["fill"]])
    elif kind == "color":
        s["color"] = rng.choice([c for c in colors["spare"]])
    elif kind == "shape":
        used = {x["type"] for x in shapes}
        s["type"] = rng.choice([t for t in SHAPE_TYPES if t not in used] or [t for t in SHAPE_TYPES if t != s["type"]])
    elif kind == "size":
        f = rng.uniform(0.58, 0.74)
        s["size"] = round(s["size"] * (f if rng.random() < 0.5 or s["size"] > 44 else 1 / f), 2)
    elif kind == "rotation":
        s["rot"] = round((s["rot"] + rng.choice([-1, 1]) * rng.uniform(20, 32)) % 360.0, 4)
    elif kind == "position":
        d, a = rng.uniform(0.6, 0.95) * s["size"] + 10, rng.uniform(0, 2 * math.pi)
        s["x"], s["y"] = round(s["x"] + d * math.cos(a), 3), round(s["y"] + d * math.sin(a), 3)
    elif kind == "incomplete":
        s["cap"] = round(rng.uniform(0.5, 0.8), 3)
    elif kind == "direction":
        pass                                    # set when directions are assigned
    shapes = shapes[:k] + [s] + shapes[k + 1:]
    return kind, k, shapes


# ---------------------------------------------------------------- timelines (frames)

def _waves_timeline(rng, n_orb):
    L = rng.randint(14, 28)
    gap = rng.randint(GAP, 16)
    intro = rng.randint(EDGE_PAD, 20)
    starts = []
    f = intro
    for _ in range(n_orb):
        starts.append(f)
        f += L + 4 + gap                        # trace, fill fade-in, pause
    end = starts[-1] + L + 4 + rng.randint(14, 24)
    return {"intro": intro, "trace": L, "fade": 4, "starts": starts}, end


def _reveal_timeline(rng, n):
    f = rng.randint(EDGE_PAD, 20)
    appear = []
    for _ in range(n):
        appear.append(f)
        f += rng.randint(GAP, 14)
    order = list(range(n))
    rng.shuffle(order)
    times = [0] * n
    for slot, i in enumerate(order):
        times[i] = appear[slot]
    end = appear[-1] + 3 + rng.randint(14, 24)
    return {"appear": times, "pop": 3}, end


def _assemble_timeline(rng, shapes, n_orb, centre, rp):
    for _ in range(200):
        intro = rng.randint(EDGE_PAD, 20)
        starts = sorted(rng.sample(range(intro, intro + 70), n_orb))
        glides = [rng.randint(18, 40) for _ in range(n_orb)]
        rng.shuffle(starts)
        settle = [s + g - 1 for s, g in zip(starts, glides)]
        ss = sorted(starts)
        st = sorted(settle)
        if all(b - a >= GAP for a, b in zip(ss, ss[1:])) and all(b - a >= GAP + 1 for a, b in zip(st, st[1:])):
            break
    else:
        return None
    cx, cy = centre
    frm = []
    for s in shapes:
        for _ in range(200):
            x = rng.uniform(40 - cx, W - 40 - cx)
            y = rng.uniform(-(H - 40 - cy), cy - 40)
            sx, sy = cx + x, cy - y
            if not (sx < TIMER_BOX[2] + 50 and sy < TIMER_BOX[3] + 50) and math.hypot(x - s["x"], y - s["y"]) > 140:
                break
        frm.append([round(x, 2), round(y, 2), round(s["rot"] + rng.choice([-1, 1]) * rng.uniform(60, 200), 2)])
    end = max(settle) + rng.randint(14, 24)
    return {"starts": starts, "glides": glides, "from": frm}, end


def _turns_timeline(rng):
    moves = []
    f = rng.randint(EDGE_PAD + 2, 22)
    speed = rng.uniform(75, 160)                  # deg / s
    n_moves = rng.randint(2, 4)
    for _ in range(n_moves):
        if rng.random() < 0.62:
            ang = rng.choice([45, 60, 72, 90, 120, 135, 144, 180]) * rng.choice([-1, 1])
            dur = max(12, round(abs(ang) / speed * FPS))
            moves.append({"kind": "turn", "angle": ang, "f0": f, "f1": f + dur - 1})
        else:
            dur = rng.randint(14, 22)
            moves.append({"kind": "flip", "axis": rng.choice(["vertical", "horizontal"]), "f0": f, "f1": f + dur - 1})
        f += dur + rng.randint(GAP + 1, 16)
    end = moves[-1]["f1"] + rng.randint(14, 24)
    return {"moves": moves, "notch": round(rng.uniform(0, 360), 2)}, end


# ---------------------------------------------------------------- per-frame state

def _smooth(u):
    return u * u * (3 - 2 * u)


def move_matrix(m, u):
    """Linear map of move m at progress u (0..1)."""
    if m["kind"] == "turn":
        th = m["angle"] * _smooth(u)
        return elem_matrix((th, 0))
    c = math.cos(math.pi * _smooth(u))
    return [[c, 0.0], [0.0, 1.0]] if m["axis"] == "vertical" else [[1.0, 0.0], [0.0, c]]


def pattern_matrix(scene, f):
    M = [[1.0, 0.0], [0.0, 1.0]]
    if scene["scenario"] != "turns":
        return M
    for m in scene["timeline"]["moves"]:
        if f < m["f0"]:
            break
        u = 1.0 if f >= m["f1"] else (f - m["f0"] + 1) / (m["f1"] - m["f0"] + 1)
        M = matmul(move_matrix(m, u), M)
    return M


def trace_order(pts, clockwise: bool):
    """Vertices starting at vertex 0, ordered so tracing turns clockwise
    on screen (y up here, so clockwise = negative signed area)."""
    area = 0.5 * sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                     for i in range(len(pts)))
    if (area < 0) != clockwise:
        pts = [pts[0]] + pts[1:][::-1]
    return pts


def shape_state(scene, i, f):
    """(points y-up rel. centre in trace order, drawn fraction, tip, fill alpha, scale) or None."""
    s = scene["shapes"][i]
    sc, tl = scene["scenario"], scene["timeline"]
    pts = outline(s)
    frac, tip, fill_a = s["cap"], None, 1.0
    if sc == "waves":
        f0 = tl["starts"][s["orbit"]]
        if f < f0:
            return None
        pts = trace_order(pts, s["dir"] == "clockwise")
        u = min(1.0, (f - f0 + 1) / tl["trace"])
        frac = min(u, s["cap"])
        if u < 1.0:
            tip = point_along(pts, frac)
        f_done = f0 + tl["trace"] - 1
        fill_a = 0.0 if f <= f_done else min(1.0, (f - f_done) / tl["fade"])
    elif sc == "reveal":
        f0 = tl["appear"][i]
        if f < f0:
            return None
        k = f - f0
        scale = 0.45 + 0.55 * min(1.0, (k + 1) / tl["pop"])
        if scale < 1.0:
            pts = [(s["x"] + (x - s["x"]) * scale, s["y"] + (y - s["y"]) * scale) for x, y in pts]
    elif sc == "assemble":
        o = s["orbit"]
        f0, g = tl["starts"][o], tl["glides"][o]
        u = 0.0 if f < f0 else min(1.0, (f - f0 + 1) / g)
        u = _smooth(u)
        if u < 1.0:
            x0, y0, r0 = tl["from"][i]
            moved = dict(s, x=x0 + (s["x"] - x0) * u, y=y0 + (s["y"] - y0) * u, rot=r0 + (s["rot"] - r0) * u)
            pts = outline(moved)
    elif sc == "turns":
        M = pattern_matrix(scene, f)
        pts = [apply(M, p) for p in pts]
    return pts, frac, tip, fill_a


def point_along(pts, frac):
    """Point at fraction `frac` of the closed outline's perimeter."""
    segs = [(pts[k], pts[(k + 1) % len(pts)]) for k in range(len(pts))]
    lens = [math.dist(a, b) for a, b in segs]
    goal = frac * sum(lens)
    for k, ((a, b), ln) in enumerate(zip(segs, lens)):
        if goal <= ln or k == len(segs) - 1:
            u = 0.0 if ln == 0 else min(1.0, goal / ln)
            return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)
        goal -= ln
    return pts[0]


def notch_point(scene, f):
    tl = scene["timeline"]
    a = math.radians(tl["notch"])
    r = scene["disc_r"]
    return apply(pattern_matrix(scene, f), (r * math.cos(a), r * math.sin(a)))


# ---------------------------------------------------------------- events (generator side)

def wave_frames(scene):
    """[(start, done)] per orbit: first frame traced, first frame fully traced."""
    tl = scene["timeline"]
    return [(f0, f0 + tl["trace"] - 1) for f0 in tl["starts"]]


def settle_frames(scene):
    tl = scene["timeline"]
    return [s + g - 1 for s, g in zip(tl["starts"], tl["glides"])]


def net_matrix(scene, moves=None):
    M = [[1.0, 0.0], [0.0, 1.0]]
    for m in (scene["timeline"]["moves"] if moves is None else moves):
        M = matmul(move_matrix(m, 1.0), M)
    return [[round(v, 9) for v in row] for row in M]


def final_shapes(scene):
    """Descriptors of the finished pattern (before any turns move)."""
    return scene["shapes"]


def moved_shapes(scene, M):
    """The pattern after the linear map M (an isometry)."""
    th, fl = matrix_elem(M)
    out = []
    for s in scene["shapes"]:
        x, y = apply(M, (s["x"], s["y"]))
        rot = (th + (-s["rot"] if fl else s["rot"])) % 360.0
        out.append(dict(s, x=x, y=y, rot=rot, mirrored=bool(s["mirrored"]) != bool(fl)))
    return out


def event_frames(scene) -> list[int]:
    sc, tl = scene["scenario"], scene["timeline"]
    if sc == "waves":
        return sorted({f for a, b in wave_frames(scene) for f in (a, b)})
    if sc == "reveal":
        return sorted(tl["appear"])
    if sc == "assemble":
        return sorted(set(tl["starts"]) | set(settle_frames(scene)))
    return sorted({f for m in tl["moves"] for f in (m["f0"], m["f1"])})


# ---------------------------------------------------------------- checks

def check(scene) -> list[str]:
    errs = []
    sc = scene.get("scenario")
    if sc not in SCENARIOS:
        return [f"unknown scenario {sc}"]
    nf = scene["n_frames"]
    if not 96 <= nf <= MAX_FRAMES:
        errs.append(f"{nf} frames")
    shapes = scene["shapes"]
    if len(shapes) > MAX_SHAPES[sc]:
        errs.append("too many shapes")
    if any(s["size"] < 20 for s in shapes):
        errs.append("shape too small")
    bg = scene["style"]["bg"][0]
    for s in shapes:
        if sum(abs(a - b) for a, b in zip(COLORS[s["color"]], bg)) < 200:
            errs.append(f"{s['color']} too close to the background")
    # orbit names (colour + outline) are unique, the odd shape differs
    names = {}
    for s in scene["base"]:
        names.setdefault((s["color"], s["type"]), set()).add(s["orbit"])
    if any(len(v) > 1 for v in names.values()):
        errs.append("two orbits share colour and outline")
    # geometry: separation, inside the frame and clear of the timer
    cx, cy = scene["center"]
    for i, a in enumerate(shapes):
        for b in shapes[i + 1:]:
            if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < a["size"] + b["size"] + 14:
                errs.append("two shapes too close")
                break
        if math.hypot(a["x"], a["y"]) + a["size"] > scene["rp"] + 2:
            errs.append("shape outside the pattern circle")
    reach = scene["disc_r"] + 8 if sc == "turns" else scene["rp"]
    if cx - reach < 0 or cx + reach > W or cy - reach < 0 or cy + reach > H:
        errs.append("pattern leaves the frame")
    nx, ny = min(max(cx, TIMER_BOX[0]), TIMER_BOX[2]), min(max(cy, TIMER_BOX[1]), TIMER_BOX[3])
    if math.hypot(cx - nx, cy - ny) <= reach + 6:
        errs.append("pattern reaches the timer")
    # symmetry: the unbroken pattern has exactly its group, a static break destroys it
    want = GROUPS[scene["group"]]
    # no near-symmetries: the same group even with a loose 6 px tolerance
    if detect(scene["base"]) != want or detect(scene["base"], 6.0) != want:
        errs.append("unbroken pattern does not measure as its group (or nearly has more symmetry)")
    br = scene["break"]
    if br and br["kind"] != "direction" and (detect(shapes) == want or detect(shapes, 4.0) == want):
        errs.append("break leaves the symmetry (nearly) intact")
    if br and br["kind"] != "direction" and detect(shapes, 4.0) != detect(shapes):
        errs.append("broken pattern has a near-symmetry")
    if br and br["kind"] == "direction":
        dirs = [s["dir"] for s in shapes]
        if sorted(dirs.count(d) for d in set(dirs)) != [1, len(shapes) - 1]:
            errs.append("direction break is not exactly one shape")
    # timing
    ev = event_frames(scene)
    if ev and (ev[0] < EDGE_PAD or ev[-1] > nf - 1 - EDGE_PAD):
        errs.append("event too near the start or end")
    if sc == "waves":
        wf = wave_frames(scene)
        if any(b[0] - a[1] - 4 < GAP for a, b in zip(wf, wf[1:])):
            errs.append("waves too close")
    if sc == "reveal":
        ap = sorted(scene["timeline"]["appear"])
        if any(b - a < GAP for a, b in zip(ap, ap[1:])):
            errs.append("two shapes appear too close in time")
    if sc == "assemble":
        st = sorted(settle_frames(scene))
        ss = sorted(scene["timeline"]["starts"])
        if any(b - a < GAP for a, b in zip(st, st[1:])) or any(b - a < GAP for a, b in zip(ss, ss[1:])):
            errs.append("orbits start or settle too close")
        for (x, y, _), s in zip(scene["timeline"]["from"], shapes):
            if not (s["size"] < cx + x < W - s["size"] and s["size"] < cy - y < H - s["size"]):
                errs.append("start spot outside the frame")
    if sc == "turns":
        ms = scene["timeline"]["moves"]
        if any(b["f0"] - a["f1"] < GAP for a, b in zip(ms, ms[1:])):
            errs.append("moves too close")
    return errs


# ---------------------------------------------------------------- build

def _attempt(rng, sc):
    group = rng.choice(list(GROUPS))
    k = len(group_elements(group))
    lo, hi = N_ORBITS[sc]
    lo = max(lo, 3 if k == 2 else lo)
    opts = [m for m in range(lo, hi + 1) if m * k <= MAX_SHAPES[sc]]
    if not opts:
        return None
    n_orb = rng.choice(opts)
    n = n_orb * k
    style = _style(rng)
    rp = rng.uniform(200, 320) if sc != "turns" else rng.uniform(210, 290)
    centre = _place_centre(rng, rp + (34 if sc == "turns" else 0))
    if centre is None:
        return None
    dense = n > 9
    smin, smax = (28, 40) if dense else (34, 58)
    scale = 0.8 + 0.2 * rp / 320
    smin, smax = max(24, smin * scale), max(30, smax * scale)
    pal = [c for c in COLORS if sum(abs(a - b) for a, b in zip(COLORS[c], style["bg"][0])) >= 200]
    if len(pal) < n_orb + 1:
        return None
    rng.shuffle(pal)
    colors = pal[:n_orb]
    shapes = _orbits(rng, group, n_orb, rp, colors, smin, smax)
    if shapes is None:
        return None
    base = [dict(s) for s in shapes]
    direction = rng.choice(["clockwise", "counter_clockwise"])
    br = None
    if rng.random() < 0.5:
        res = _break(rng, sc, shapes, {"spare": pal[n_orb:]}, k)
        if res is None:
            return None
        kind, idx, shapes = res
        br = {"kind": kind, "index": idx}
    for i, s in enumerate(shapes):
        s["dir"] = direction if sc == "waves" else None
        if br and br["kind"] == "direction" and i == br["index"]:
            s["dir"] = "counter_clockwise" if direction == "clockwise" else "clockwise"
    for s in base:
        s["dir"] = direction if sc == "waves" else None
    if sc == "waves":
        tl, end = _waves_timeline(rng, n_orb)
    elif sc == "reveal":
        tl, end = _reveal_timeline(rng, n)
    elif sc == "assemble":
        res = _assemble_timeline(rng, shapes, n_orb, centre, rp)
        if res is None:
            return None
        tl, end = res
    else:
        tl, end = _turns_timeline(rng)
    scene = {"scenario": sc, "group": group, "center": list(centre), "rp": round(rp, 2),
             "disc_r": round(rp + 24, 2), "shapes": shapes, "base": base, "break": br,
             "direction": direction if sc == "waves" else None, "timeline": tl, "n_frames": end + 1,
             "style": style, "timer": {"box": list(TIMER_BOX), "size": TIMER_FONT}, "labels": []}
    return scene


def build(rng: random.Random) -> dict:
    sc = SCENARIOS[rng.randrange(len(SCENARIOS))]
    for attempt in range(1, 3000):
        scene = _attempt(rng, sc)
        if scene is not None and not check(scene) and _askable(scene):
            scene["attempts"] = attempt
            return scene
    raise RuntimeError("no valid scene")


def _askable(scene) -> bool:
    import questions  # noqa: PLC0415  (questions imports sim)
    P = questions.pools(questions.Facts(json.loads(json.dumps(scene))), random.Random(7))
    av = [len(P[lv]) for lv in range(1, 6)]
    # every level can reach its maximum, so any drawn count combination can be filled
    return all(a >= hi for a, (lo, hi) in zip(av, questions.L.TEN_RANGES))


@functools.lru_cache(maxsize=8)
def _cached_detect(key: str):
    return detect(json.loads(key))


def detect_cached(shapes):
    return _cached_detect(json.dumps(shapes, sort_keys=True))
