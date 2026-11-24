"""Overlapping shapes (v3): flat shapes on a board, overlapping each other,
with one of four scenarios picked per seed:

  arrive  translucent shapes appear one at a time; overlaps stay visible
          through the see-through fills
  paper   opaque paper cut-outs are laid down one at a time; a later shape
          covers the earlier ones where they overlap
  depart  every translucent shape is on the board from the start and some
          of them disappear one at a time
  glide   translucent shapes stay put while one or two extra shapes slide
          across the board in a straight line, passing over some of them

Geometry is exact shapely polygons in frame pixels (1280x720). Two shapes
"overlap" when their intersection is at least MIN_OVERLAP px^2; scenes in
which two shapes (or a larger group) share only a sliver, or in which two
non-overlapping shapes come closer than MIN_GAP px, are rejected, so every
overlap is clearly visible and clearly countable. Every shape has a unique
colour + type name ("the red star"); the only text on screen is the clock.

Time is kept in whole frames (24 fps). An arrival (or departure) that
starts at frame s is visible from frame s, fully drawn (or gone) from frame
s + anim - 1. All randomness comes from the rng passed to build()."""
from __future__ import annotations
import functools
import hashlib
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.shapes import TYPES, unit_shape  # noqa: E402

FPS = 24
W, H = 1280, 720
MAX_FRAMES = 240                 # 10 s
TIMER_BOX = (1084, 12, 1266, 72)
MIN_OVERLAP = 300.0              # px^2: smaller shared areas are slivers (rejected)
MIN_GAP = 15.0                   # px: shapes that don't overlap stay at least this far apart
EDGE_F = 12                      # frames (0.5 s): events stay this far from the clip's start and end
SEP_F = 8                        # frames (0.33 s): question events stay this far apart
SCENARIOS = ("arrive", "paper", "depart", "glide")
COLORS = {"red": "#E23B30", "orange": "#F08A1A", "yellow": "#E3BC12", "green": "#2E9C4E",
          "teal": "#0FA3A0", "blue": "#2F6FE0", "purple": "#8A4FD8", "pink": "#E8559C",
          "brown": "#8F5A2B", "gray": "#6B7280"}
STYLES = ("solid", "dashed", "dotted", "double")
FILLS = ("plain", "stripes", "dots", "hatch")
LAYOUTS = ("cluster", "chain", "ring", "grid", "spiral", "flower", "scatter", "twin", "diagonal")
GLIDE_LAYOUTS = ("scatter", "grid", "ring", "chain", "twin", "diagonal")
THEMES = (
    {"name": "paper", "board": [250, 249, 245], "bg": [232, 230, 224], "grid": None, "dark": False},
    {"name": "cream", "board": [247, 240, 222], "bg": [222, 214, 194], "grid": None, "dark": False},
    {"name": "mint", "board": [236, 246, 240], "bg": [214, 228, 220], "grid": None, "dark": False},
    {"name": "sky", "board": [234, 242, 250], "bg": [210, 222, 236], "grid": None, "dark": False},
    {"name": "graph", "board": [252, 252, 252], "bg": [226, 228, 232], "grid": "lines", "dark": False},
    {"name": "dotgrid", "board": [246, 244, 238], "bg": [220, 216, 206], "grid": "dots", "dark": False},
    {"name": "linen", "board": [240, 234, 224], "bg": [214, 206, 192], "grid": "noise", "dark": False},
    {"name": "slate", "board": [40, 44, 54], "bg": [22, 24, 30], "grid": None, "dark": True},
    {"name": "night", "board": [28, 36, 52], "bg": [14, 18, 28], "grid": "lines", "dark": True},
)
ENTER_ANIMS = ("grow", "fade", "spin", "drop")
EXIT_ANIMS = ("shrink", "fade", "spin", "lift")
TYPE_PLURAL = {"circle": "circles", "ellipse": "ellipses", "square": "squares", "rectangle": "rectangles",
               "diamond": "diamonds", "triangle": "triangles", "pentagon": "pentagons", "hexagon": "hexagons",
               "octagon": "octagons", "star": "stars", "cross": "crosses", "heart": "hearts",
               "crescent": "crescents", "trapezoid": "trapezoids", "semicircle": "semicircles",
               "parallelogram": "parallelograms"}


# ------------------------------------------------------------------ geometry
def outline_at(shape: dict, x: float, y: float, rot: float, scale: float = 1.0) -> list:
    th = math.radians(rot)
    c, s = math.cos(th), math.sin(th)
    r = shape["size"] * scale
    return [(x + r * (u * c - v * s), y + r * (u * s + v * c))
            for u, v in unit_shape(shape["type"], shape.get("aspect", 1.0))]


def outline(shape: dict) -> list:
    """Outline at the shape's resting place (start of the path for a mover)."""
    return outline_at(shape, shape["x"], shape["y"], shape["rot"])


def polygon(shape: dict) -> Polygon:
    return Polygon(outline(shape)).buffer(0)


def label(shape: dict) -> str:
    return f"{shape['color']} {shape['type']}"


def statics(scene: dict) -> list[int]:
    return [i for i, s in enumerate(scene["shapes"]) if "path" not in s]


def movers(scene: dict) -> list[int]:
    return [i for i, s in enumerate(scene["shapes"]) if "path" in s]


def pose(shape: dict, t: float) -> tuple[float, float, float]:
    """(x, y, rot) of a shape at time t; movers slide at constant speed between frames f0 and f1."""
    p = shape.get("path")
    if p is None:
        return shape["x"], shape["y"], shape["rot"]
    t0, t1 = p["f0"] / FPS, p["f1"] / FPS
    u = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
    return (shape["x"] + (p["x1"] - shape["x"]) * u, shape["y"] + (p["y1"] - shape["y"]) * u,
            shape["rot"] + p["spin"] * u * (t1 - t0))


def poly_at(shape: dict, t: float) -> Polygon:
    x, y, r = pose(shape, t)
    return Polygon(outline_at(shape, x, y, r))          # the unit outlines are simple polygons


def _key(scene: dict) -> str:
    return json.dumps(scene["shapes"], sort_keys=True)


@functools.lru_cache(maxsize=512)
def _analysis(key: str) -> dict:
    shapes = json.loads(key)
    idx = [i for i, s in enumerate(shapes) if "path" not in s]
    polys = {i: polygon(shapes[i]) for i in idx}
    inter = {}
    for i, j in itertools.combinations(idx, 2):
        inter[(i, j)] = polys[i].intersection(polys[j]).area if polys[i].intersects(polys[j]) else 0.0
    return {"polys": polys, "inter": inter, "pairs": sorted(p for p, a in inter.items() if a >= MIN_OVERLAP)}


def pairs(scene: dict, drop=()) -> list[tuple[int, int]]:
    """Overlapping pairs among the static shapes (movers excluded)."""
    return [p for p in _analysis(_key(scene))["pairs"] if p[0] not in drop and p[1] not in drop]


def neighbours(scene: dict, drop=()) -> dict:
    nb = {i: set() for i in statics(scene) if i not in drop}
    for i, j in pairs(scene, drop):
        nb[i].add(j)
        nb[j].add(i)
    return nb


def max_depth(scene: dict, members=None) -> tuple[int, float]:
    """Largest number of shapes sharing one region (cliques of the overlap
    graph with a common intersection of at least MIN_OVERLAP), and the
    smallest non-empty common-intersection area met (for the sliver gate)."""
    an = _analysis(_key(scene))
    polys = an["polys"]
    members = sorted(polys) if members is None else sorted(members)
    nb = neighbours(scene)
    best, smallest = (1 if members else 0), float("inf")
    mset = set(members)

    def grow(group, region, cands):
        nonlocal best, smallest
        for k in sorted(cands):
            if k <= group[-1]:
                continue
            reg = region.intersection(polys[k])
            if reg.area < 1.0:
                continue
            smallest = min(smallest, reg.area)
            if reg.area >= MIN_OVERLAP:
                best = max(best, len(group) + 1)
                grow(group + [k], reg, cands & nb[k])
    for i in members:
        grow([i], polys[i], nb[i] & mset)
    return best, smallest


# -------------------------------------------------------------- timeline
def events(scene: dict) -> list[dict]:
    """[{"shape": i, "f": start frame}] in time order (arrivals or departures)."""
    return scene["events"]


def anim_window(scene: dict, ev: dict) -> tuple[int, int]:
    """First and last frame in which the event's shape is changing."""
    return ev["f"], ev["f"] + scene["anim_f"] - 1


def on_board(scene: dict, f: int) -> list[int]:
    """Shapes fully on the board at frame f (arrive / paper / depart)."""
    sc = scene["scenario"]
    if sc == "glide":
        return list(range(len(scene["shapes"])))
    ev = {e["shape"]: e["f"] for e in scene["events"]}
    out = []
    for i in range(len(scene["shapes"])):
        if sc == "depart":
            if i not in ev or f < ev[i]:
                out.append(i)
        elif i in ev and f >= ev[i] + scene["anim_f"] - 1:
            out.append(i)
    return out


def stable_window(scene: dict, f: int) -> tuple[int, int] | None:
    """Frames [a, b] around f in which nothing changes; None if f is inside a change."""
    a, b = 0, scene["n_frames"] - 1
    for e in scene["events"]:
        s, t = anim_window(scene, e)
        if s <= f <= t:
            return None
        if t < f:
            a = max(a, t + 1)
        else:
            b = min(b, s - 1)
    return a, b


def fps_for(gap_s: float) -> float:
    """Sampling rate needed to land at least one frame inside a gap of gap_s seconds."""
    return math.ceil(100.0 / max(gap_s, 1e-3)) / 100.0


def timer_text(f: int) -> str:
    return f"{f / FPS:.1f} s"


# ---------------------------------------------------------------- glide
def _polys_at(shape: dict, ts) -> np.ndarray:
    """Vectorised outlines of a (moving) shape at many times -> array of shapely polygons."""
    U = np.array(unit_shape(shape["type"], shape.get("aspect", 1.0)))
    poses = np.array([pose(shape, t) for t in ts])
    th = np.radians(poses[:, 2])
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    r = shape["size"]
    xs = poses[:, 0:1] + r * (U[None, :, 0] * c - U[None, :, 1] * s)
    ys = poses[:, 1:2] + r * (U[None, :, 0] * s + U[None, :, 1] * c)
    return shapely.polygons(np.stack([xs, ys], axis=2))


@functools.lru_cache(maxsize=1024)
def _mover_static(mkey: str, skey: str) -> dict:
    """One mover against one still shape: the times (s) the shared area rises
    above and falls below MIN_OVERLAP, the first and last contact, the peak
    area and the closest approach, sampled 48 times a second and refined by
    bisection."""
    sh, st = json.loads(mkey), json.loads(skey)
    sp = polygon(st)
    p = sh["path"]
    t0, t1 = p["f0"] / FPS, p["f1"] / FPS
    step = 1.0 / (2 * FPS)
    ts = [t0 + k * step for k in range(int(math.ceil((t1 - t0) / step)) + 1)]
    ts[-1] = min(ts[-1], t1)
    rad_m = sh["size"] * max(math.hypot(u, v) for u, v in unit_shape(sh["type"], sh.get("aspect", 1.0)))
    rad_s = st["size"] * max(math.hypot(u, v) for u, v in unit_shape(st["type"], st.get("aspect", 1.0)))
    cen = np.array([pose(sh, t)[:2] for t in ts])
    near = np.hypot(cen[:, 0] - st["x"], cen[:, 1] - st["y"]) <= rad_m + rad_s + MIN_GAP + 2
    area = np.zeros(len(ts))
    dist = np.full(len(ts), MIN_GAP + 99.0)
    if near.any():
        idx = np.nonzero(near)[0]
        pms = _polys_at(sh, [ts[k] for k in idx])
        area[idx] = shapely.area(shapely.intersection(pms, sp))
        dist[idx] = np.where(area[idx] > 0, 0.0, shapely.distance(pms, sp))

    def cross(k, thr, rising):
        """Refine a crossing inside one sample step on a 64-point grid (0.33 ms)."""
        sub = np.linspace(ts[k - 1], ts[k], 65)
        a = shapely.area(shapely.intersection(_polys_at(sh, sub), sp))
        hit = (a >= thr) if rising else (a < thr)
        j = int(np.argmax(hit))
        return float(sub[j]) if rising else float(sub[j - 1])
    ups = [k for k in range(1, len(ts)) if area[k - 1] < MIN_OVERLAP <= area[k]]
    downs = [k for k in range(1, len(ts)) if area[k - 1] >= MIN_OVERLAP > area[k]]
    c_up = [k for k in range(1, len(ts)) if area[k - 1] <= 0 < area[k]]
    c_dn = [k for k in range(1, len(ts)) if area[k - 1] > 0 >= area[k]]
    rec = {"peak": float(area.max()), "closest": float(dist.min()), "start_area": float(area[0]),
           "end_area": float(area[-1])}
    if len(ups) == 1 and len(downs) == 1 and len(c_up) == 1 and len(c_dn) == 1:
        rec["enter"] = cross(ups[0], MIN_OVERLAP, True)
        rec["exit"] = cross(downs[0], MIN_OVERLAP, False)
        rec["touch"] = cross(c_up[0], 1e-6, True)
        rec["leave"] = cross(c_dn[0], 1e-6, False)
    return rec


@functools.lru_cache(maxsize=256)
def _mover_gap(akey: str, bkey: str) -> float:
    a, b = json.loads(akey), json.loads(bkey)
    f0 = min(a["path"]["f0"], b["path"]["f0"])
    f1 = max(a["path"]["f1"], b["path"]["f1"])
    ts = [f / FPS for f in range(f0, f1 + 1)]
    return float(shapely.distance(_polys_at(a, ts), _polys_at(b, ts)).min())


def _sk(shape):
    return json.dumps(shape, sort_keys=True)


def _glide(key: str) -> dict:
    shapes = json.loads(key)
    st = [i for i, s in enumerate(shapes) if "path" not in s]
    mv = [i for i, s in enumerate(shapes) if "path" in s]
    out = {"pairs": {f"{m},{s}": _mover_static(_sk(shapes[m]), _sk(shapes[s])) for m in mv for s in st},
           "mover_gap": float("inf")}
    for a, b in itertools.combinations(mv, 2):
        out["mover_gap"] = min(out["mover_gap"], _mover_gap(_sk(shapes[a]), _sk(shapes[b])))
    return out


def glide_info(scene: dict) -> dict:
    return _glide(_key(scene))


def touches(scene: dict, m: int) -> list[tuple[int, float, float]]:
    """(static, enter_s, exit_s) for every static the mover m overlaps, by enter time."""
    g = glide_info(scene)["pairs"]
    out = [(s, g[f"{m},{s}"]["enter"], g[f"{m},{s}"]["exit"]) for s in statics(scene)
           if "enter" in g[f"{m},{s}"]]
    return sorted(out, key=lambda r: r[1])


def glide_events(scene: dict) -> list[float]:
    """Every moment something changes in a glide clip (s)."""
    out = []
    for m in movers(scene):
        p = scene["shapes"][m]["path"]
        out += [p["f0"] / FPS, p["f1"] / FPS]
        for _, a, b in touches(scene, m):
            out += [a, b]
    return sorted(out)


# ------------------------------------------------------------- placement
def _centres(rng, layout, n, board):
    bx0, by0, bx1, by1 = board
    w, h = bx1 - bx0, by1 - by0
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    if layout == "cluster":
        return [(rng.gauss(cx, w * 0.17), rng.gauss(cy, h * 0.15)) for _ in range(n)]
    if layout == "scatter":
        return [(rng.uniform(bx0 + 90, bx1 - 90), rng.uniform(by0 + 80, by1 - 80)) for _ in range(n)]
    if layout == "chain":
        x0, x1 = bx0 + 100, bx1 - 100
        amp, ph = rng.uniform(0.05, 0.18) * h, rng.uniform(0, 2 * math.pi)
        return [(x0 + (x1 - x0) * k / (n - 1), cy + amp * math.sin(ph + 2.2 * math.pi * k / (n - 1)))
                for k in range(n)]
    if layout == "diagonal":
        sgn = rng.choice((-1, 1))
        return [(bx0 + 110 + (w - 220) * k / (n - 1), cy + sgn * (h / 2 - 90) * (2 * k / (n - 1) - 1))
                for k in range(n)]
    if layout == "ring":
        a0 = rng.uniform(0, 2 * math.pi)
        rx, ry = w * rng.uniform(0.28, 0.36), h * rng.uniform(0.26, 0.33)
        return [(cx + rx * math.cos(a0 + 2 * math.pi * k / n), cy + ry * math.sin(a0 + 2 * math.pi * k / n))
                for k in range(n)]
    if layout == "grid":
        rows = 2 if n <= 6 else 3
        cols = math.ceil(n / rows)
        sx = min(w * 0.2, (w - 220) / max(1, cols - 1))
        sy = (h - 200) / max(1, rows - 1)
        pts = [(cx + (c - (cols - 1) / 2) * sx, cy + (r - (rows - 1) / 2) * sy)
               for r in range(rows) for c in range(cols)]
        return rng.sample(pts, n)
    if layout == "spiral":
        a0 = rng.uniform(0, 2 * math.pi)
        return [(cx + (30 + 0.3 * w * k / n) * math.cos(a0 + 0.8 * k),
                 cy + (30 + 0.3 * h * k / n) * math.sin(a0 + 0.8 * k)) for k in range(n)]
    if layout == "twin":
        hubs = [(bx0 + w * 0.28, cy + rng.uniform(-0.1, 0.1) * h), (bx0 + w * 0.72, cy + rng.uniform(-0.1, 0.1) * h)]
        return [(rng.gauss(hubs[k % 2][0], w * 0.08), rng.gauss(hubs[k % 2][1], h * 0.12)) for k in range(n)]
    a0 = rng.uniform(0, 2 * math.pi)                                   # flower: a hub and petals
    rx, ry = w * 0.22, h * 0.26
    return [(cx, cy)] + [(cx + rx * math.cos(a0 + 2 * math.pi * k / (n - 1)),
                          cy + ry * math.sin(a0 + 2 * math.pi * k / (n - 1))) for k in range(n - 1)]


def _inside(poly: Polygon, board, margin=10.0) -> bool:
    x0, y0, x1, y1 = poly.bounds
    return (x0 >= board[0] + margin and y0 >= board[1] + margin and x1 <= board[2] - margin
            and y1 <= board[3] - margin)


def _fits(cand: Polygon, placed: list, board, allow_overlap=True) -> bool:
    if not _inside(cand, board, 12):
        return False
    for p in placed:
        a = cand.intersection(p).area if cand.intersects(p) else 0.0
        if a >= 1.0 and (not allow_overlap or a < MIN_OVERLAP * 1.4):
            return False
        if a < 1.0 and cand.distance(p) < MIN_GAP + 3:
            return False
    return True


def _new_shape(rng, kind, col, x, y, size, fills):
    sh = {"type": kind, "color": col, "style": rng.choice(STYLES), "fill": rng.choice(fills),
          "x": round(x, 1), "y": round(y, 1), "size": round(size, 1), "rot": round(rng.uniform(0, 360), 1)}
    if kind in ("square", "diamond"):          # keep a square square-on and a diamond point-up
        sh["rot"] = round(rng.uniform(-20, 20) % 360, 1)
    if kind in ("ellipse", "rectangle", "parallelogram"):
        sh["aspect"] = round(rng.uniform(1.4, 2.1), 2)
    return sh


def _place(rng, layout, kinds, colors, board, size_rng, fills, allow_overlap=True):
    pts = _centres(rng, layout, len(kinds), board)
    lo, hi = size_rng
    shapes, polys = [], []
    for kind, col, (x, y) in zip(kinds, colors, pts):
        for attempt in range(90):
            j = 6 + attempt * 2.0
            sh = _new_shape(rng, kind, col,
                            min(max(x + rng.uniform(-j, j), board[0] + 50), board[2] - 50),
                            min(max(y + rng.uniform(-j, j), board[1] + 50), board[3] - 50),
                            rng.uniform(lo, hi) * max(0.6, 1 - attempt / 150), fills)
            poly = polygon(sh)
            if _fits(poly, polys, board, allow_overlap):
                shapes.append(sh)
                polys.append(poly)
                break
        else:
            return None
    return shapes


def _names(rng, n, types, palette):
    kinds = list(types)
    while len(kinds) < n:                      # at most 4 shapes of one type
        t = rng.choice(types)
        if kinds.count(t) < 4:
            kinds.append(t)
    rng.shuffle(kinds)
    used, colors = set(), []
    for kind in kinds:
        free = [c for c in palette if (c, kind) not in used]
        col = rng.choice(free)
        used.add((col, kind))
        colors.append(col)
    return kinds, colors


def _timeline(rng, n_ev, intro_lo, intro_hi, anim):
    """Start frames of n_ev events, with gaps of at least anim + 9 frames."""
    hold = rng.randint(16, 32)
    intro = rng.randint(intro_lo, intro_hi)
    gmin = anim + 9
    gmax = min(30, (MAX_FRAMES - intro - anim - hold) // max(1, n_ev - 1))
    if gmax < gmin:
        return None
    fs = [intro]
    for _ in range(n_ev - 1):
        fs.append(fs[-1] + rng.randint(gmin, gmax))
    return fs, fs[-1] + anim + hold


def _theme(rng):
    th = dict(rng.choice(THEMES))
    th["seed"] = rng.randint(0, 10 ** 6)
    if th["grid"] == "lines":
        th["grid_px"] = rng.choice((32, 40, 48, 60))
    elif th["grid"] == "dots":
        th["grid_px"] = rng.choice((24, 30, 36))
    return th


def _attempt(rng, scenario):
    th = _theme(rng)
    board = [rng.randint(16, 150), rng.randint(TIMER_BOX[3] + 16, 130),
             rng.randint(1130, 1264), rng.randint(620, 704)]
    palette = [c for c in COLORS if not (th["dark"] and c in ("brown", "gray"))]
    zoom = rng.uniform(0.8, 1.2)
    fills = FILLS if rng.random() < 0.7 else ("plain",)
    scene = {"scenario": scenario, "theme": th, "board": board, "zoom": round(zoom, 3),
             "line_w": rng.randint(3, 5), "alpha": 255 if scenario == "paper" else rng.randint(78, 120)}
    if scenario == "glide":
        n_st = rng.randint(5, 8)
        n_mv = 2 if n_st >= 5 and rng.random() < 0.4 else 1
        n = n_st + n_mv
        k = rng.randint(max(2, math.ceil(n / 4)), min(n, 8))
        layout = rng.choice(GLIDE_LAYOUTS)
    else:
        n = rng.randint(*{"arrive": (4, 10), "paper": (4, 9), "depart": (5, 10)}[scenario])
        k = rng.randint(max(2, math.ceil(n / 4)), min(n, 10))
        layout = rng.choice(LAYOUTS if n >= 4 else GLIDE_LAYOUTS)
    scene["pattern"] = layout
    types = rng.sample(TYPES, k)
    kinds, colors = _names(rng, n, types, palette)
    hi = min(118.0, 62 + 330.0 / n) * zoom
    if scenario == "glide":
        hi = min(hi, 95.0)
    size_rng = (max(36.0, 0.58 * hi), max(44.0, hi))
    n_st = n - (n_mv if scenario == "glide" else 0)
    shapes = _place(rng, layout, kinds[:n_st], colors[:n_st], board, size_rng, fills)
    if shapes is None:
        return None
    scene["shapes"] = shapes
    if scenario == "glide":
        return _attempt_glide(rng, scene, kinds[n_st:], colors[n_st:], size_rng, fills)
    if check_geometry(scene):
        return None
    np_ = len(pairs(scene))
    if not (max(2, n // 3) <= np_ <= 2 * n):
        return None
    scene["anim"] = rng.choice(EXIT_ANIMS if scenario == "depart" else ENTER_ANIMS)
    scene["anim_f"] = rng.randint(6, 8)
    order = list(range(n))
    rng.shuffle(order)
    if scenario == "depart":
        m = rng.randint(3, n - 1)
        tl = _timeline(rng, m, 24, 40, scene["anim_f"])
        order = order[:m]
    else:
        tl = _timeline(rng, n, 14, 26, scene["anim_f"])
    if tl is None:
        return None
    fs, scene["n_frames"] = tl
    scene["events"] = [{"shape": i, "f": f} for i, f in zip(order, fs)]
    scene["duration"] = round(scene["n_frames"] / FPS, 4)
    return scene


def _attempt_glide(rng, scene, kinds, colors, size_rng, fills):
    board = scene["board"]
    if check_geometry(scene):
        return None
    scene["n_frames"] = rng.randint(168, MAX_FRAMES)
    scene["anim"] = "none"
    scene["anim_f"] = 0
    scene["events"] = []
    N = scene["n_frames"]
    placed = [polygon(s) for s in scene["shapes"]]
    for kind, col in zip(kinds, colors):
        ok = False
        for _ in range(80):
            size = rng.uniform(size_rng[0], min(size_rng[1], 85.0))
            ang = rng.uniform(0, 2 * math.pi)
            w, h = board[2] - board[0], board[3] - board[1]
            L = rng.uniform(0.4, 0.8) * math.hypot(w, h)
            dx, dy = L * math.cos(ang), L * math.sin(ang)
            mx = size * 1.2 + 14
            xs = (board[0] + mx - min(0, dx), board[2] - mx - max(0, dx))
            ys = (board[1] + mx - min(0, dy), board[3] - mx - max(0, dy))
            if xs[0] >= xs[1] or ys[0] >= ys[1]:
                continue
            x0, y0 = rng.uniform(*xs), rng.uniform(*ys)
            sh = _new_shape(rng, kind, col, x0, y0, size, fills)
            v = rng.uniform(110, 260)
            f0 = rng.randint(EDGE_F + 2, 36)
            f1 = f0 + int(round(L / v * FPS))
            if f1 > N - EDGE_F - 4:
                f1 = N - EDGE_F - 4
                if (f1 - f0) / FPS < L / 330:
                    continue
            spin = 0.0 if kind in ("square", "diamond") else rng.choice((0.0, 0.0, round(rng.uniform(-45, 45), 1)))
            sh["path"] = {"x1": round(x0 + dx, 1), "y1": round(y0 + dy, 1), "f0": f0, "f1": f1, "spin": spin}
            p_start = polygon(sh)
            p_end = poly_at(sh, f1 / FPS)
            if not (_inside(p_start, board, 12) and _inside(p_end, board, 12)):
                continue
            if any(p_start.distance(q) < MIN_GAP + 3 or p_end.distance(q) < MIN_GAP + 3 for q in placed):
                continue
            scene["shapes"].append(sh)
            placed.append(p_start)
            if _glide_errors(scene, partial=True):
                scene["shapes"].pop()
                placed.pop()
                continue
            ok = True
            break
        if not ok:
            return None
    if max(len(touches(scene, m)) for m in movers(scene)) < 2:
        return None
    scene["duration"] = round(N / FPS, 4)
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


# ----------------------------------------------------------------- checks
def check_geometry(scene: dict) -> list[str]:
    errs = []
    an = _analysis(_key(scene))
    sh = scene["shapes"]
    for i, p in an["polys"].items():
        if not _inside(p, scene["board"], 6):
            errs.append(f"{label(sh[i])} leaves the board")
    for (i, j), a in an["inter"].items():
        if 1.0 <= a < MIN_OVERLAP:
            errs.append(f"{label(sh[i])} and {label(sh[j])} share only a sliver")
        elif a < 1.0 and an["polys"][i].distance(an["polys"][j]) < MIN_GAP:
            errs.append(f"{label(sh[i])} and {label(sh[j])} nearly touch")
    if not errs:
        _, smallest = max_depth(scene)
        if smallest < MIN_OVERLAP:
            errs.append("a group of shapes shares only a sliver")
    return errs


def _paper_errors(scene: dict) -> list[str]:
    """Every paper shape stays mostly visible, and every overlap shows which one is on top."""
    errs = []
    polys = _analysis(_key(scene))["polys"]
    order = [e["shape"] for e in scene["events"]]
    pos = {i: k for k, i in enumerate(order)}
    for k, i in enumerate(order):
        later = [polys[j] for j in order[k + 1:]]
        vis = polys[i].difference(unary_union(later)) if later else polys[i]
        if vis.area < 0.5 * polys[i].area or vis.buffer(-10).is_empty:
            errs.append(f"paper {label(scene['shapes'][i])} is mostly hidden")
    for i, j in pairs(scene):
        top = max(pos[i], pos[j])
        above = [polys[q] for q in order[top + 1:]]
        reg = polys[i].intersection(polys[j])
        if above:
            reg = reg.difference(unary_union(above))
        if reg.area < MIN_OVERLAP:
            errs.append(f"overlap of {label(scene['shapes'][i])} and {label(scene['shapes'][j])} is hidden")
    return errs


def _glide_errors(scene: dict, partial=False) -> list[str]:
    errs = []
    g = glide_info(scene)
    sh = scene["shapes"]
    N = scene["n_frames"]
    for m in movers(scene):
        p = sh[m]["path"]
        if p["f0"] < EDGE_F or p["f1"] > N - EDGE_F:
            errs.append(f"moving {label(sh[m])} starts or stops too close to the clip edge")
        for s in statics(scene):
            r = g["pairs"][f"{m},{s}"]
            if r["start_area"] > 0 or r["end_area"] > 0:
                errs.append(f"moving {label(sh[m])} starts or stops on the {label(sh[s])}")
            if r["peak"] <= 0:
                if r["closest"] < MIN_GAP:
                    errs.append(f"moving {label(sh[m])} nearly touches the {label(sh[s])}")
                continue
            if "enter" not in r:
                errs.append(f"moving {label(sh[m])} overlaps the {label(sh[s])} more than once or only a sliver")
                continue
            if r["peak"] < 4 * MIN_OVERLAP:
                errs.append(f"moving {label(sh[m])} only grazes the {label(sh[s])}")
            if r["exit"] - r["enter"] < 0.5:
                errs.append(f"moving {label(sh[m])} crosses the {label(sh[s])} too briefly")
            if r["enter"] - r["touch"] > 0.15 or r["leave"] - r["exit"] > 0.15:
                errs.append(f"moving {label(sh[m])} lingers at the edge of the {label(sh[s])}")
    if g["mover_gap"] < MIN_GAP:
        errs.append("the two moving shapes come too close")
    if not errs:
        ev = glide_events(scene)
        if ev and (ev[0] < EDGE_F / FPS or ev[-1] > (N - EDGE_F) / FPS):
            errs.append("an event is too close to the clip edge")
        for a, b in zip(ev, ev[1:]):
            if b - a < SEP_F / FPS:
                errs.append(f"glide events at {a:.2f} s and {b:.2f} s are too close")
                break
    return errs


def check(scene: dict) -> list[str]:
    errs = check_geometry(scene)
    sh = scene["shapes"]
    if len({label(s) for s in sh}) != len(sh):
        errs.append("two shapes share a colour + type name")
    if scene["n_frames"] > MAX_FRAMES:
        errs.append("clip longer than 10 s")
    if abs(scene["duration"] - scene["n_frames"] / FPS) > 1e-3:
        errs.append("duration doesn't match n_frames")
    b = scene["board"]
    if b[1] <= TIMER_BOX[3] + 8 or b[0] < 0 or b[2] > W or b[3] > H:
        errs.append("board overlaps the clock or leaves the frame")
    sc = scene["scenario"]
    if sc == "glide":
        if not movers(scene) or len(statics(scene)) < 5:
            errs.append("glide scene needs a moving shape and at least 5 still ones")
        else:
            errs += _glide_errors(scene)
            if not errs and max(len(touches(scene, m)) for m in movers(scene)) < 2:
                errs.append("no moving shape overlaps two still shapes")
        return errs
    ev = scene["events"]
    fs = [e["f"] for e in ev]
    if fs != sorted(fs) or len({e["shape"] for e in ev}) != len(ev):
        errs.append("events out of order or repeated")
    if sc in ("arrive", "paper") and sorted(e["shape"] for e in ev) != list(range(len(sh))):
        errs.append("not every shape arrives exactly once")
    if sc == "depart" and not 3 <= len(ev) <= len(sh) - 1:
        errs.append("depart scene must remove 3..n-1 shapes")
    if fs and (fs[0] < EDGE_F or fs[-1] + scene["anim_f"] - 1 > scene["n_frames"] - 1 - EDGE_F):
        errs.append("an event is too close to the clip edge")
    for a, b2 in zip(fs, fs[1:]):
        if b2 - a < scene["anim_f"] + SEP_F:
            errs.append("two events are too close together")
    if sc == "paper":
        errs += _paper_errors(scene)
    return errs


def signature(scene: dict) -> str:
    """Hash of the rounded scene parameters (used for the uniqueness stats)."""
    def rnd(x):
        if isinstance(x, float):
            return round(x, 1)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in sorted(x.items()) if k != "tries"}
        if isinstance(x, list):
            return [rnd(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(rnd(scene), sort_keys=True).encode()).hexdigest()
