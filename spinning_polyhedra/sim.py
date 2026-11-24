"""Scene generation for spinning_polyhedra (the image benchmark's polyhedron
category, made temporal).

One or two wireframe solids spin in front of the camera. All edges are drawn,
front and back, so a single frame is an ambiguous tangle of lines; the motion
(plus a depth cue that varies per clip) shows the 3D shape. Coloured markers
sit on some corners so turns, passes and directions can be followed.

Scenarios (`scene["scenario"]`):
  steady        one solid, constant spin about the upright axis
  reverse       one solid; it stops for a moment and spins back (1-2 times)
  speed_change  one solid about the upright axis, speeding up, slowing down,
                or speeding up and then slowing down (it never stops)
  pair          two different solids side by side, each spinning steadily
                about its own upright axis at clearly different rates
  axis_switch   one solid spins about the upright axis, stops for a moment,
                then spins about a left-right or a front-back axis

Motion: every body has a list of contiguous segments
{"f0", "f1", "axis", "w0", "w1"} (frames, axis "y"/"x"/"z", angular speed in
deg/s ramping linearly from w0 to w1). A pause is a segment with w0 = w1 = 0.
Every timing is an integer frame; the clip is frames 0 .. n_frames - 1.

SOLIDS lists each solid's corners (vertices), edges, faces and face shapes;
`check` confirms the vertex coordinates (through a convex hull) really give
those counts.
"""
from __future__ import annotations
import functools
import itertools
import json
import math
import random

import numpy as np
from scipy.spatial import ConvexHull

FPS = 24
W, H = 1280, 720
MAX_FRAMES = 240                      # 10 s
EDGE_F = 14                           # events >= 0.5 s + 2 frames from either end
SCENARIOS = ("steady", "reverse", "speed_change", "pair", "axis_switch")
PHI = (1 + 5 ** 0.5) / 2
MAX_TRIES = 400


# ---- timer --------------------------------------------------------------------
def tenths(f: int) -> int:
    """Timer value at frame f in tenths of a second (floor), as drawn on screen."""
    return (f * 10) // FPS


def fmt_t(tn: int) -> str:
    return f"{tn // 10}.{tn % 10}"


def timer_text(f: int) -> str:
    return f"{fmt_t(tenths(f))} s"


# ---- solids -------------------------------------------------------------------
def _ring(n, y, off=0.0):
    return [(math.cos(2 * math.pi * (k + off) / n), y, math.sin(2 * math.pi * (k + off) / n)) for k in range(n)]


def _prism(n):
    return _ring(n, -0.75) + _ring(n, 0.75)


def _pyramid(n):
    return _ring(n, -0.6) + [(0.0, 1.0, 0.0)]


def _bipyramid(n):
    return _ring(n, 0.0) + [(0.0, 1.2, 0.0), (0.0, -1.2, 0.0)]


def _antiprism(n):
    return _ring(n, -0.6) + _ring(n, 0.6, 0.5)


def _perm_signs(base, even_only=False):
    out = set()
    for perm in itertools.permutations(range(3)):
        if even_only and perm not in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            continue
        for signs in itertools.product((1, -1), repeat=3):
            out.add(tuple(round(signs[i] * base[perm[i]], 9) for i in range(3)))
    return sorted(out)


FAMILY = {  # family solids: (family, n) -- used by the "one more side" counterfactual
    "triangular prism": ("prism", 3), "pentagonal prism": ("prism", 5), "hexagonal prism": ("prism", 6),
    "square pyramid": ("pyramid", 4), "pentagonal pyramid": ("pyramid", 5), "hexagonal pyramid": ("pyramid", 6),
    "triangular bipyramid": ("bipyramid", 3), "pentagonal bipyramid": ("bipyramid", 5),
    "hexagonal bipyramid": ("bipyramid", 6),
    "square antiprism": ("antiprism", 4), "pentagonal antiprism": ("antiprism", 5),
    "hexagonal antiprism": ("antiprism", 6),
}
FAMILY_FN = {"prism": _prism, "pyramid": _pyramid, "bipyramid": _bipyramid, "antiprism": _antiprism}


def _verts(name):
    if name == "tetrahedron":
        return [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    if name == "cube":
        return list(itertools.product((-1, 1), repeat=3))
    if name == "octahedron":
        return [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    if name == "icosahedron":
        return sorted(set(_perm_signs((0, 1, PHI), even_only=True)))
    if name == "dodecahedron":
        return list(itertools.product((-1, 1), repeat=3)) + sorted(set(_perm_signs((0, 1 / PHI, PHI), True)))
    if name == "cuboctahedron":
        return _perm_signs((1, 1, 0))
    if name == "truncated tetrahedron":
        return [p for p in _perm_signs((3, 1, 1)) if (p[0] * p[1] * p[2]) > 0]
    if name == "truncated octahedron":
        return _perm_signs((0, 1, 2))
    if name == "rhombic dodecahedron":
        return list(itertools.product((-1, 1), repeat=3)) + _perm_signs((2, 0, 0))
    fam, n = FAMILY[name]
    return FAMILY_FN[fam](n)


# name: (corners, edges, faces, face shapes {sides: count})
SOLIDS = {
    "tetrahedron": (4, 6, 4, {3: 4}),
    "cube": (8, 12, 6, {4: 6}),
    "octahedron": (6, 12, 8, {3: 8}),
    "dodecahedron": (20, 30, 12, {5: 12}),
    "icosahedron": (12, 30, 20, {3: 20}),
    "triangular prism": (6, 9, 5, {3: 2, 4: 3}),
    "pentagonal prism": (10, 15, 7, {5: 2, 4: 5}),
    "hexagonal prism": (12, 18, 8, {6: 2, 4: 6}),
    "square pyramid": (5, 8, 5, {4: 1, 3: 4}),
    "pentagonal pyramid": (6, 10, 6, {5: 1, 3: 5}),
    "hexagonal pyramid": (7, 12, 7, {6: 1, 3: 6}),
    "triangular bipyramid": (5, 9, 6, {3: 6}),
    "pentagonal bipyramid": (7, 15, 10, {3: 10}),
    "hexagonal bipyramid": (8, 18, 12, {3: 12}),
    "square antiprism": (8, 16, 10, {3: 8, 4: 2}),
    "pentagonal antiprism": (10, 20, 12, {3: 10, 5: 2}),
    "hexagonal antiprism": (12, 24, 14, {3: 12, 6: 2}),
    "cuboctahedron": (12, 24, 14, {3: 8, 4: 6}),
    "truncated tetrahedron": (12, 18, 8, {3: 4, 6: 4}),
    "truncated octahedron": (24, 36, 14, {4: 6, 6: 8}),
    "rhombic dodecahedron": (14, 24, 12, {4: 12}),
}
SHAPE = {3: "triangles", 4: "four-sided faces", 5: "pentagons", 6: "hexagons"}


@functools.lru_cache(maxsize=None)
def geometry(name):
    """(unit-radius vertices centred on the origin, edges as index pairs, faces as vertex-index lists)."""
    v = np.array(_verts(name), dtype=float)
    v -= v.mean(axis=0)
    v /= np.linalg.norm(v, axis=1).max()
    hull = ConvexHull(v)
    planes = {}
    for simplex, eq in zip(hull.simplices, hull.equations):
        key = tuple(np.round(eq, 5))
        planes.setdefault(key, set()).update(simplex.tolist())
    faces = [sorted(s) for s in sorted(planes.values(), key=sorted)]
    edges = set()
    for simplex in hull.simplices:
        for a, b in itertools.combinations(simplex.tolist(), 2):
            if sum(1 for f in faces if a in f and b in f) == 2:
                edges.add((min(a, b), max(a, b)))
    return v, sorted(edges), faces


def degree(name, k) -> int:
    return sum(1 for a, b in geometry(name)[1] if k in (a, b))


# ---- rotations and motion -------------------------------------------------------
def rot(ax, a):
    c, s = math.cos(a), math.sin(a)
    if ax == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if ax == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def base_matrix(body) -> np.ndarray:
    """Starting orientation: a turn about the upright axis, then a tilt."""
    tx, tz = body["tilt"]
    return rot("z", tz) @ rot("x", tx) @ rot("y", math.radians(body["phase"]))


def seg_angle(seg, f) -> float:
    """Degrees turned inside segment `seg` by frame f (clamped to the segment)."""
    f0, f1 = seg["f0"], seg["f1"]
    if f <= f0:
        return 0.0
    f = min(f, f1)
    t, T = (f - f0) / FPS, (f1 - f0) / FPS
    return seg["w0"] * t + (seg["w1"] - seg["w0"]) * t * t / (2 * T)


def seg_total(seg) -> float:
    return (seg["w0"] + seg["w1"]) / 2 * (seg["f1"] - seg["f0"]) / FPS


def orientation(body, f) -> np.ndarray:
    R = base_matrix(body)
    for seg in body["motion"]:
        a = seg_angle(seg, f)
        if a:
            R = rot(seg["axis"], math.radians(a)) @ R
        if f <= seg["f1"]:
            break
    return R


def theta_y(body, f) -> float:
    """Cumulative degrees turned about the upright axis by frame f (all segments must be 'y')."""
    tot = 0.0
    for seg in body["motion"]:
        if seg["axis"] == "y":
            tot += seg_angle(seg, f)
        if f <= seg["f1"]:
            break
    return tot


def speed(body, f) -> float:
    """Signed angular speed (deg/s) at frame f (the segment that contains (f, f+1))."""
    for seg in body["motion"]:
        if seg["f0"] <= f < seg["f1"]:
            u = (f + 0.5 - seg["f0"]) / (seg["f1"] - seg["f0"])
            return seg["w0"] + (seg["w1"] - seg["w0"]) * u
    return 0.0


def max_speed(body, f0=0, f1=None) -> float:
    f1 = body["motion"][-1]["f1"] if f1 is None else f1
    out = 0.0
    for seg in body["motion"]:
        if seg["f1"] > f0 and seg["f0"] < f1:
            out = max(out, abs(seg["w0"]), abs(seg["w1"]))
    return out


def is_pause(seg) -> bool:
    return seg["w0"] == 0 and seg["w1"] == 0


def runs(body) -> list[list[dict]]:
    """Maximal stretches of turning (lists of segments), split by pauses."""
    out, cur = [], []
    for seg in body["motion"]:
        if is_pause(seg):
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(seg)
    if cur:
        out.append(cur)
    return out


def pauses(body) -> list[tuple[int, int]]:
    """(stop frame, last still frame) of every pause: the pose is the same on frames f1..f2."""
    return [(s["f0"], s["f1"]) for s in body["motion"] if is_pause(s)]


def stop_frames(body) -> list[int]:
    return [a for a, _ in pauses(body)]


def resume_frames(body) -> list[int]:
    """First frame that shows a new pose after each pause."""
    return [b + 1 for _, b in pauses(body)]


def y_only(body) -> bool:
    return all(s["axis"] == "y" for s in body["motion"])


def run_angle(run) -> float:
    return sum(seg_total(s) for s in run)


def n_frames(scene) -> int:
    return scene["n_frames"]


# ---- markers (generator-side, analytic) -----------------------------------------
def mark_base(scene, m) -> np.ndarray:
    mk = scene["marks"][m]
    body = scene["bodies"][mk["body"]]
    v = geometry(body["solid"])[0][mk["v"]]
    return base_matrix(body) @ v


def orbit(scene, m) -> tuple[float, float]:
    """(radius about the upright axis, starting azimuth in degrees; 0 = straight in front)."""
    p = mark_base(scene, m)
    return math.hypot(p[0], p[2]), math.degrees(math.atan2(p[0], p[2]))


def azimuth(scene, m, f, body=None) -> float:
    body = body or scene["bodies"][scene["marks"][m]["body"]]
    return orbit(scene, m)[1] + theta_y(body, f)


def passes(scene, m, body=None, f_end=None) -> list[int]:
    """Frames (<= f_end) on which marker m has just crossed straight in front of the upright axis
    (azimuth = 0 mod 360)."""
    body = body or scene["bodies"][scene["marks"][m]["body"]]
    f_end = scene["n_frames"] - 1 if f_end is None else f_end
    out = []
    prev = math.floor(azimuth(scene, m, 0, body) / 360)
    for f in range(1, f_end + 1):
        k = math.floor(azimuth(scene, m, f, body) / 360)
        if k != prev:
            out.append(f)
        prev = k
    return out


def turning_frames(body, n) -> list[int]:
    return sorted({0, n - 1} | set(stop_frames(body)))


def pass_safe(scene, m, body=None, margin=30.0, f_end=None) -> bool:
    """Counting / timing the passes of marker m (up to frame f_end) is unambiguous: no pass within EDGE_F
    frames of either end, and at the start, the end and every stop the marker is >= `margin` degrees from
    straight in front."""
    body = body or scene["bodies"][scene["marks"][m]["body"]]
    n = scene["n_frames"]
    f_end = n - 1 if f_end is None else f_end
    ps = passes(scene, m, body, f_end)
    if any(f < EDGE_F or f > n - 1 - EDGE_F or f > f_end - 2 for f in ps):
        return False
    for f in [x for x in turning_frames(body, n) if x <= f_end] + [f_end]:
        a = azimuth(scene, m, f, body) % 360
        if min(a, 360 - a) < margin:
            return False
    return True


def side_at(scene, m, f, body=None):
    """(left/right, near/far) of marker m at frame f, or None if too close to the axis or the side."""
    body = body or scene["bodies"][scene["marks"][m]["body"]]
    r, _ = orbit(scene, m)
    a = math.radians(azimuth(scene, m, f, body))
    x, z = r * math.sin(a), r * math.cos(a)
    cam = scene["camera"]["dist"]
    px = abs(x) * body["scale"] * cam / (cam + 1)
    if px < 25 or abs(z) < 0.3 * r:
        return None
    return ("left" if x < 0 else "right"), ("near" if z > 0 else "far")


def turns_value(angle_deg) -> tuple[int, float]:
    t = abs(angle_deg) / 360
    return int(math.floor(t)), t - math.floor(t)


def frac_ok(frac) -> bool:
    return 0.2 <= frac <= 0.8


# ---- palettes -------------------------------------------------------------------
BG_DARK = [("ink", (18, 20, 28)), ("navy", (14, 22, 46)), ("plum", (34, 18, 40)), ("forest", (14, 32, 24)),
           ("charcoal", (34, 34, 36)), ("deep teal", (8, 34, 40)), ("wine", (40, 14, 22))]
BG_LIGHT = [("paper", (242, 240, 232)), ("mist", (226, 233, 240)), ("sand", (238, 228, 206)),
            ("pale mint", (224, 240, 230)), ("lavender", (236, 232, 246)), ("pale grey", (222, 222, 222))]
EDGE_DARK = [("silver", (222, 228, 238)), ("pale blue", (150, 200, 255)), ("mint", (150, 236, 200)),
             ("cream", (246, 232, 184)), ("lilac", (212, 182, 255)), ("peach", (255, 202, 162))]
EDGE_LIGHT = [("charcoal", (38, 42, 50)), ("navy", (28, 48, 112)), ("teal", (16, 92, 98)), ("plum", (88, 36, 98)),
              ("brown", (98, 58, 26)), ("slate", (66, 78, 100))]
MARKS = [("red", (232, 36, 36)), ("blue", (36, 104, 255)), ("green", (24, 186, 56)), ("yellow", (250, 214, 0)),
         ("magenta", (228, 36, 200)), ("orange", (255, 136, 0)), ("cyan", (0, 206, 230))]
MARK_SHAPES = ("dot", "square", "diamond")
LINE_STYLES = ("shade", "dashed_back", "flat")
BG_STYLES = ("flat", "gradient", "vignette")
TIMER_CORNERS = ("tl", "tr", "bl", "br")


def cdist(a, b) -> int:
    return sum(abs(int(x) - int(y)) for x, y in zip(a, b))


def mix(a, b, u):
    return tuple(int(round(a[i] + (b[i] - a[i]) * u)) for i in range(3))


def edge_dim(body, bg):
    return mix(body["edge_rgb"], bg, 0.45)


def timer_box(scene) -> tuple[int, int, int, int]:
    """Conservative pixel box of the timer text (renderer anchors it at a corner)."""
    t = scene["timer"]
    w, h = int(t["size"] * 3.4), int(t["size"] * 1.15)
    x = 32 if t["corner"][1] == "l" else W - 32 - w
    y = 26 if t["corner"][0] == "t" else H - 26 - h
    return x, y, x + w, y + h


def body_radius(scene, b) -> float:
    """Upper bound (px) on how far any drawn point of body b gets from its centre."""
    cam = scene["camera"]["dist"]
    body = scene["bodies"][b]
    return body["scale"] * cam / (cam - 1) + body["edge_w"] * 2 + 18


# ---- scene drawing --------------------------------------------------------------
def _seg(f0, f1, axis, w0, w1=None):
    return {"f0": f0, "f1": f1, "axis": axis, "w0": round(w0, 6), "w1": round(w0 if w1 is None else w1, 6)}


def _ratio_pair(rng, lo, hi, equal_p=0.4):
    """Two speeds in [lo, hi]: equal (prob equal_p) or differing by a factor >= 1.6."""
    a = rng.uniform(lo, hi)
    if rng.random() < equal_p:
        return a, a
    for _ in range(50):
        r = rng.uniform(1.6, 2.4)
        b = a * r if rng.random() < 0.5 else a / r
        if lo <= b <= hi:
            return a, b
    return a, a


def _motion(rng, scenario) -> tuple[list[list[dict]], int]:
    """Per-body motion segments and the clip length in frames."""
    sgn = rng.choice((-1, 1))
    if scenario in ("steady", "pair"):
        n = rng.randint(144, MAX_FRAMES)
        out = []
        periods = []
        for _ in range(2 if scenario == "pair" else 1):
            for _ in range(100):
                p = rng.randint(24, 90) / 10                         # seconds per turn, exact tenths
                if all(max(p, q) / min(p, q) >= 1.5 for q in periods):
                    break
            periods.append(p)
            out.append([_seg(0, n - 1, "y", rng.choice((-1, 1)) * 360 / p)])
        return out, n
    if scenario == "speed_change":
        n = rng.randint(156, MAX_FRAMES)
        prof = rng.choice(("up", "down", "up_down"))
        lo = rng.uniform(30, 55)
        hi = lo * rng.uniform(2.6, 3.6)
        if prof == "up":
            return [[_seg(0, n - 1, "y", sgn * lo, sgn * hi)]], n
        if prof == "down":
            return [[_seg(0, n - 1, "y", sgn * hi, sgn * lo)]], n
        mid = int(round(n * rng.uniform(0.4, 0.6)))
        lo2 = lo * rng.uniform(0.85, 1.15)
        return [[_seg(0, mid, "y", sgn * lo, sgn * hi), _seg(mid, n - 1, "y", sgn * hi, sgn * lo2)]], n
    if scenario == "reverse":
        k = 1 if rng.random() < 0.6 else 2
        lens = [rng.randint(36, 84) for _ in range(k + 1)]
        ps = [rng.randint(12, 24) for _ in range(k)]
        while sum(lens) + sum(ps) > MAX_FRAMES - 1:
            i = lens.index(max(lens))
            lens[i] -= 4
        w1, w2 = _ratio_pair(rng, 50, 150)
        speeds = [w1, w2, w1 if rng.random() < 0.5 else w2][:k + 1]
        segs, f = [], 0
        for i in range(k + 1):
            s = sgn * (-1) ** i * speeds[i]
            segs.append(_seg(f, f + lens[i], "y", s))
            f += lens[i]
            if i < k:
                segs.append(_seg(f, f + ps[i], "y", 0.0))
                f += ps[i]
        return [segs], f + 1
    # axis_switch
    l1, pz, l2 = rng.randint(40, 100), rng.randint(12, 24), rng.randint(40, 100)
    while l1 + pz + l2 > MAX_FRAMES - 1:
        if l1 >= l2:
            l1 -= 4
        else:
            l2 -= 4
    w1, w2 = _ratio_pair(rng, 50, 150)
    ax = rng.choice(("x", "z"))
    segs = [_seg(0, l1, "y", sgn * w1), _seg(l1, l1 + pz, "y", 0.0),
            _seg(l1 + pz, l1 + pz + l2, ax, rng.choice((-1, 1)) * w2)]
    return [segs], l1 + pz + l2 + 1


def _draw(rng) -> dict:
    scenario = rng.choice(SCENARIOS)
    dark = rng.random() < 0.6
    bg_name, bg = rng.choice(BG_DARK if dark else BG_LIGHT)
    bg_style = rng.choice(BG_STYLES)
    shade = rng.randint(10, 26) * (1 if dark else -1)
    bg2 = tuple(max(0, min(255, c + shade)) for c in bg)
    cam = {"elev": round(rng.uniform(6, 28), 3), "dist": round(rng.uniform(3.2, 6.0), 3)}
    timer = {"corner": rng.choice(TIMER_CORNERS), "size": rng.randint(36, 44),
             "ink": [236, 238, 242] if dark else [24, 26, 32]}
    motions, n = _motion(rng, scenario)
    nb = len(motions)
    solids = rng.sample(sorted(SOLIDS), nb)
    edge_pool = list(EDGE_DARK if dark else EDGE_LIGHT)
    edge_cols = rng.sample(edge_pool, nb)
    line_style = rng.choice(LINE_STYLES)
    edge_w = rng.randint(3, 5)
    tint = round(rng.uniform(0.08, 0.16), 3) if rng.random() < 0.35 else 0.0
    corner_dots = rng.random() < 0.4
    bodies = []
    for b in range(nb):
        if nb == 1:
            R = rng.uniform(175, 290)
            cx, cy = 640 + rng.uniform(-120, 120), 360 + rng.uniform(-25, 25)
        else:
            R = rng.uniform(140, 225)
            cx = (rng.uniform(300, 390) if b == 0 else rng.uniform(890, 980))
            cy = 360 + rng.uniform(-30, 30)
        scale = R * (cam["dist"] - 1) / cam["dist"]
        bodies.append({"solid": solids[b], "cx": round(cx, 2), "cy": round(cy, 2), "scale": round(scale, 2),
                       "tilt": [round(rng.uniform(-0.55, 0.55), 4), round(rng.uniform(-0.55, 0.55), 4)],
                       "phase": round(rng.uniform(0, 360), 3), "edge_name": edge_cols[b][0],
                       "edge_rgb": list(edge_cols[b][1]), "edge_w": edge_w, "motion": motions[b]})
    scene = {"scenario": scenario, "n_frames": n, "theme": "dark" if dark else "light", "bg_name": bg_name,
             "bg": list(bg), "bg2": list(bg2), "bg_style": bg_style, "camera": cam, "timer": timer,
             "line_style": line_style, "tint": tint, "corner_dots": corner_dots, "bodies": bodies,
             "labels": []}
    # markers
    n_marks = rng.choice((1, 2, 3)) if nb == 1 else rng.choice((2, 2, 3))
    owners = [0] * n_marks if nb == 1 else ([0, 1] + [rng.randrange(2)])[:n_marks]
    cols = [c for c in MARKS if all(cdist(c[1], x) >= 200 for x in
                                   [bg, bg2] + [tuple(bd["edge_rgb"]) for bd in bodies] +
                                   [edge_dim(bd, bg) for bd in bodies])]
    rng.shuffle(cols)
    picked = []
    for c in cols:
        if all(cdist(c[1], d[1]) >= 200 for d in picked):
            picked.append(c)
    cols = picked
    n_marks = max(nb, min(n_marks, len(cols)))
    owners = owners[:n_marks]
    marks = []
    for i, ob in enumerate(owners):
        body = bodies[ob]
        verts = list(range(len(geometry(body["solid"])[0])))
        rng.shuffle(verts)
        marks.append({"body": ob, "v": verts[0], "colour": cols[i % len(cols)][0] if cols else "red",
                      "rgb": list(cols[i % len(cols)][1]) if cols else [232, 36, 36],
                      "shape": rng.choice(MARK_SHAPES), "r": rng.randint(11, 15), "_cands": verts})
    scene["marks"] = marks
    # move each marker to the first vertex that keeps it clear of the axis and of the other markers
    for i, mk in enumerate(marks):
        for v in mk["_cands"]:
            mk["v"] = v
            r, az = orbit(scene, i)
            p = mark_base(scene, i)
            ok = r >= 0.35
            for j in range(i):
                if marks[j]["body"] != mk["body"]:
                    continue
                r2, az2 = orbit(scene, j)
                d = abs((az - az2 + 180) % 360 - 180)
                ok = ok and d >= 50 and np.linalg.norm(p - mark_base(scene, j)) >= 0.45
            if ok:
                break
    for mk in marks:
        del mk["_cands"]
    return scene


def build(rng: random.Random) -> dict:
    for tries in range(1, MAX_TRIES + 1):
        scene = _draw(rng)
        if not check(scene):
            scene["tries"] = tries
            return scene
    raise RuntimeError("no valid scene")


# ---- checks ---------------------------------------------------------------------
def check(scene) -> list:
    errs = []
    n = scene["n_frames"]
    if not 120 <= n <= MAX_FRAMES:
        errs.append(f"clip length {n} frames outside 120-{MAX_FRAMES}")
    bodies = scene["bodies"]
    if scene["scenario"] not in SCENARIOS or len(bodies) != (2 if scene["scenario"] == "pair" else 1):
        errs.append("bad scenario / body count")
        return errs
    if len({b["solid"] for b in bodies}) != len(bodies):
        errs.append("the two solids are the same")
    bg = tuple(scene["bg"])
    tb = timer_box(scene)
    for i, body in enumerate(bodies):
        v, e, f = geometry(body["solid"])
        V, E, F, shapes = SOLIDS[body["solid"]]
        if (len(v), len(e), len(f)) != (V, E, F):
            errs.append(f"{body['solid']}: geometry gives {len(v)}/{len(e)}/{len(f)}, table says {V}/{E}/{F}")
        if sorted(len(x) for x in f) != sorted(k for k, c in shapes.items() for _ in range(c)):
            errs.append(f"{body['solid']}: face shapes differ from the table")
        segs = body["motion"]
        if segs[0]["f0"] != 0 or segs[-1]["f1"] != n - 1 or \
                any(a["f1"] != b["f0"] for a, b in zip(segs, segs[1:])):
            errs.append("motion segments do not tile the clip")
        for s in segs:
            if is_pause(s):
                if s["f1"] - s["f0"] < 12:
                    errs.append("pause shorter than 0.5 s")
            else:
                if s["f1"] - s["f0"] < 24 or min(abs(s["w0"]), abs(s["w1"])) < 25 or \
                        max(abs(s["w0"]), abs(s["w1"])) > 200 or s["w0"] * s["w1"] <= 0:
                    errs.append("bad turning segment")
        for f in stop_frames(body) + resume_frames(body):
            if f < EDGE_F or f > n - 1 - EDGE_F:
                errs.append("stop/resume too close to an end")
        R = body_radius(scene, i)
        cx, cy = body["cx"], body["cy"]
        if cx - R < 6 or cx + R > W - 6 or cy - R < 6 or cy + R > H - 6:
            errs.append(f"body {i} can leave the frame")
        nx, ny = min(max(cx, tb[0]), tb[2]), min(max(cy, tb[1]), tb[3])
        if math.hypot(cx - nx, cy - ny) < R + 12:
            errs.append(f"body {i} can touch the timer")
    if len(bodies) == 2:
        d = math.hypot(bodies[0]["cx"] - bodies[1]["cx"], bodies[0]["cy"] - bodies[1]["cy"])
        if d < body_radius(scene, 0) + body_radius(scene, 1) + 20:
            errs.append("the two solids can overlap")
        if not bodies[0]["cx"] < bodies[1]["cx"]:
            errs.append("body 0 must be the left one")
    marks = scene["marks"]
    if not marks:
        errs.append("no markers")
    if len({m["colour"] for m in marks}) != len(marks):
        errs.append("two markers share a colour")
    for i, mk in enumerate(marks):
        body = bodies[mk["body"]]
        if not 0 <= mk["v"] < SOLIDS[body["solid"]][0]:
            errs.append("bad marked corner")
            continue
        if mk["r"] < 11:
            errs.append("marker smaller than 22 px")
        others = [bg, tuple(scene["bg2"])] + [tuple(b["edge_rgb"]) for b in bodies] + \
            [edge_dim(b, bg) for b in bodies] + [tuple(m["rgb"]) for j, m in enumerate(marks) if j != i]
        if min(cdist(mk["rgb"], o) for o in others) < 200:
            errs.append(f"marker {mk['colour']} too close in colour to something else")
        r, az = orbit(scene, i)
        if r < 0.35:
            errs.append(f"marker {mk['colour']} too close to the upright axis")
        for j in range(i):
            if marks[j]["body"] == mk["body"]:
                d = abs((az - orbit(scene, j)[1] + 180) % 360 - 180)
                if d < 50 or np.linalg.norm(mark_base(scene, i) - mark_base(scene, j)) < 0.45:
                    errs.append("two markers too close together")
    if errs:
        return errs
    import questions  # lazy: questions imports sim
    sup = questions.supply(scene)
    for lv, k in enumerate(sup, 1):
        if k < 3:
            errs.append(f"level {lv}: only {k} distinct candidate questions")
    return errs


def signature(scene) -> str:
    """Hash input for the uniqueness check: rounded scene params without the retry counter."""
    def rnd(x):
        if isinstance(x, float):
            return round(x, 3)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items() if k != "tries"}
        if isinstance(x, list):
            return [rnd(v) for v in x]
        return x
    return json.dumps(rnd(scene), sort_keys=True)


# ---- counterfactual motions (the changed rule; both sides re-simulate with these) ----
def scale_body(body, k) -> dict:
    """Same body, every turning speed multiplied by k (same timings, same stops)."""
    out = dict(body)
    out["motion"] = [dict(s, w0=s["w0"] * k, w1=s["w1"] * k) for s in body["motion"]]
    return out


def no_stop_body(body, n) -> dict:
    """Same body turning about the upright axis at its starting speed for the whole clip."""
    out = dict(body)
    out["motion"] = [{"f0": 0, "f1": n - 1, "axis": "y", "w0": body["motion"][0]["w0"],
                      "w1": body["motion"][0]["w0"]}]
    return out


def swap_speed_body(body, other) -> dict:
    """Same body and direction, turning at the other body's (constant) speed."""
    out = dict(body)
    w = math.copysign(abs(other["motion"][0]["w0"]), body["motion"][0]["w0"])
    out["motion"] = [dict(s, w0=w, w1=w) for s in body["motion"]]
    return out


def upright_body(body) -> dict:
    """Same timings, but every stretch of turning is about the upright axis in the starting direction."""
    out = dict(body)
    sg = 1 if body["motion"][0]["w0"] > 0 else -1
    out["motion"] = [dict(s, axis="y", w0=sg * abs(s["w0"]), w1=sg * abs(s["w1"])) for s in body["motion"]]
    return out
