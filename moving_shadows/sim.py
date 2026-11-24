"""moving_shadows scene sampler (New Problem Types B5).

Side view. A light (a sun on an arc, or a lamp running along a rail) moves at
constant speed across the sky while 2-5 coloured objects float in the air or
stand on the ground. Every object casts a shadow onto the ground surface
strip; all shadows are the same grey and merge where they overlap, so which
object casts which shadow has to be read from how the shadows move. 4-7
marker plates sit in the ground strip, numbered 1..n left to right.

Scenarios (scene["scenario"]):

  floating  2-5 floating shapes, a sun on an arc
  posts     2-3 standing objects (post, pillar, hut) plus 0-2 floating shapes, a sun on an arc
  drifting  one floating shape drifts sideways at constant speed, a sun on an arc
  rising    one floating shape rises or sinks at constant speed, a sun on an arc
  lamp      a lamp runs along a straight rail near the top of the frame instead of a sun

Geometry: an object's shadow on the ground line y = G is the interval
between the ground projections of its outline vertices. The sun is
effectively at infinity, so its rays are parallel and set by its elevation
phi(t) (it sweeps the sky at constant angular speed, 28-152 degrees):
x' = vx - (G - vy) cot(phi). The lamp is a point L(t):
x' = Lx + (vx - Lx) * (G - Ly) / (vy - Ly). A marker at column m
(geometric position m + 0.5) is in shadow when it lies inside an interval.
Coverage events (a marker entering or leaving a shadow) are found on a
1/240 s grid and bisected.

Time: frame f is shown at t = f / 24; the clip has N frames (168-240, so
7-10 s) and the light moves from its start (f = 0) to its end (f = N - 1).
The clock shows f * 10 // 24 tenths of a second.
Pure function of the rng passed in; JSON-only output.
"""
from __future__ import annotations
import functools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.shapes import unit_shape  # noqa: E402

FPS = 24
SIZE = (1280, 720)
MIN_FRAMES, MAX_FRAMES = 168, 240          # 7-10 s
EDGE = 0.6                                 # every coverage event >= 0.6 s from the start and the end
GAP = 0.3                                  # min time between coverage events at one marker; min shading length
MIN_SPEED = 50.0                           # px/s, shadow edges crossing a marker move at least this fast
GRAZE = 4.0                                # px, a shadow edge that turns back must stay this far from a marker
END_GAP = 8.0                              # px, at the first and last frame every shadow edge is this far from a marker
KMAX = 3.4                                 # lamp: max ground-projection factor (keeps shadows a sane length)
ELEV_MIN = 28.0                            # sun: lowest elevation (deg); shadows are at most cot(28) = 1.9 x height
DT = 1 / 240
CLOCK_BOX = (18, 16, 214, 84)
LABEL_SIZE = 36                            # marker numbers: 26 px tall digits
SCENARIOS = ("floating", "posts", "drifting", "rising", "lamp")

COLOURS = {
    "red": (220, 48, 48), "blue": (45, 95, 225), "green": (40, 160, 70), "purple": (140, 70, 210),
    "teal": (0, 165, 165), "pink": (235, 95, 185), "brown": (140, 85, 40), "lime": (150, 200, 30),
}
FLOAT_SHAPES = ("circle", "ellipse", "square", "rectangle", "diamond", "triangle", "pentagon", "hexagon",
                "octagon", "star", "cross", "heart", "trapezoid", "semicircle", "parallelogram", "crescent")
STAND_KINDS = ("post", "pillar", "hut")
PATTERNS = ("solid", "striped", "dotted", "checked", "ringed")
STAND_PATTERNS = ("solid", "striped", "banded", "dotted")
SKIES = {
    "day": [(150, 196, 236), (226, 238, 248)], "pale": [(196, 212, 226), (238, 240, 242)],
    "dawn": [(236, 190, 170), (250, 232, 210)], "dusk": [(120, 120, 176), (226, 186, 190)],
    "haze": [(214, 222, 206), (244, 244, 232)], "indoor": [(206, 200, 190), (232, 226, 216)],
}
GROUNDS = {
    "sand": (196, 172, 120), "soil": (150, 116, 84), "grass": (110, 150, 80), "stone": (150, 150, 146),
    "clay": (176, 120, 96), "snow": (214, 220, 228),
}
SUNS = {"gold": (255, 206, 60), "amber": (255, 170, 40), "lemon": (250, 228, 40)}


# ------------------------------------------------------------------ light and outlines
def n_frames(scene) -> int:
    return scene["frames"]


def duration(scene) -> float:
    return scene["frames"] / FPS


def t_end(scene) -> float:
    """Time of the last frame; the light reaches the end of its path here."""
    return (scene["frames"] - 1) / FPS


def clock_tenths(f: int) -> int:
    return f * 10 // FPS


def elevation(scene, t):
    """Sun elevation (radians above the horizon, measured from the right) at time t."""
    lt = scene["light"]
    return math.radians(lt["phi0"] + (lt["phi1"] - lt["phi0"]) * t / t_end(scene))


def light(scene, t):
    """Where the light is drawn: the lamp bulb, or the sun disc on its (elliptical) sky arc at its elevation."""
    lt = scene["light"]
    u = t / t_end(scene)
    if lt["kind"] == "lamp":
        return lt["x0"] + (lt["x1"] - lt["x0"]) * u, lt["y"]
    ph = elevation(scene, t)
    return lt["cx"] + lt["Rx"] * math.cos(ph), lt["cy"] - lt["Ry"] * math.sin(ph)


def light_np(scene, ts):
    lt = scene["light"]
    u = ts / t_end(scene)
    if lt["kind"] == "lamp":
        return lt["x0"] + (lt["x1"] - lt["x0"]) * u, np.full_like(ts, float(lt["y"]))
    ph = np.radians(lt["phi0"] + (lt["phi1"] - lt["phi0"]) * u)
    return lt["cx"] + lt["Rx"] * np.cos(ph), lt["cy"] - lt["Ry"] * np.sin(ph)


def project(scene, t, pts):
    """Ground x of each point's shadow. Sun: parallel rays at the sun's elevation (a sun is effectively at
    infinity), x' = x - (G - y) cot(elevation). Lamp: rays from the bulb, x' = Lx + (x - Lx)(G - Ly)/(y - Ly)."""
    G = scene["ground"]
    if scene["light"]["kind"] == "lamp":
        lx, ly = light(scene, t)
        return [lx + (vx - lx) * (G - ly) / (vy - ly) for vx, vy in pts]
    ph = elevation(scene, t)
    cot = math.cos(ph) / math.sin(ph)
    return [vx - (G - vy) * cot for vx, vy in pts]


def light_dir(scene) -> int:
    """+1 if the light moves left to right, -1 if right to left."""
    a, b = light(scene, 0.0)[0], light(scene, t_end(scene))[0]
    return 1 if b > a else -1


def base_outline(o: dict) -> list:
    """Outline at the start position (t = 0), in px."""
    G = o["ground"]
    if o["kind"] == "stand":
        x, w, h = o["x"], o["w"], o["h"]
        if o["shape"] == "hut":
            r = o["roof"]
            return [(x - w / 2, G), (x - w / 2, G - h), (x, G - h - r), (x + w / 2, G - h), (x + w / 2, G)]
        return [(x - w / 2, G), (x - w / 2, G - h), (x + w / 2, G - h), (x + w / 2, G)]
    th = math.radians(o["rot"])
    c, s = math.cos(th), math.sin(th)
    r = o["s"]
    return [(o["x"] + r * (px * c - py * s), o["y"] + r * (px * s + py * c))
            for px, py in unit_shape(o["shape"], o.get("aspect", 1.0))]


def offset(o: dict, t: float):
    return o.get("vx", 0.0) * t, o.get("vy", 0.0) * t


def outline(o: dict, t: float) -> list:
    dx, dy = offset(o, t)
    return [(x + dx, y + dy) for x, y in base_outline(o)]


def bbox(o: dict, t: float):
    pts = outline(o, t)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def shadow(scene, i: int, t: float):
    xs = project(scene, t, outline(scene["casters"][i], t))
    return min(xs), max(xs)


def shadows(scene, t: float) -> list:
    return [shadow(scene, i, t) for i in range(len(scene["casters"]))]


def mx(scene, j: int) -> float:
    """Geometric x of marker j: the centre of its pixel column."""
    return scene["markers"][j] + 0.5


PATTERN_WORD = {"solid": "", "striped": "striped ", "dotted": "dotted ", "checked": "checked ", "ringed": "ringed ",
                "banded": "banded "}


def label(o: dict) -> str:
    """Unique per clip by colour; the pattern word only adds what is drawn."""
    return f"the {PATTERN_WORD[o['pattern']]}{o['color']} {o['shape']}"


def light_name(scene) -> str:
    return "lamp" if scene["light"]["kind"] == "lamp" else "sun"


# ------------------------------------------------------------------ coverage events
def _grid(scene):
    T = t_end(scene)
    n = int(math.ceil(T / DT))
    return np.linspace(0.0, T, n + 1)


def _bounds(scene, ts):
    """Per caster: arrays lo(t), hi(t) of its ground shadow."""
    lx, ly = light_np(scene, ts)
    G = scene["ground"]
    lt = scene["light"]
    out = []
    for o in scene["casters"]:
        base = base_outline(o)
        vx = np.array([p[0] for p in base])[None, :] + o.get("vx", 0.0) * ts[:, None]
        vy = np.array([p[1] for p in base])[None, :] + o.get("vy", 0.0) * ts[:, None]
        if lt["kind"] == "lamp":
            xs = lx[:, None] + (vx - lx[:, None]) * (G - ly[:, None]) / (vy - ly[:, None])
        else:
            ph = np.radians(lt["phi0"] + (lt["phi1"] - lt["phi0"]) * ts / t_end(scene))
            xs = vx - (G - vy) * (np.cos(ph) / np.sin(ph))[:, None]
        out.append((xs.min(axis=1), xs.max(axis=1)))
    return out


def _bisect(f, a, b, va):
    for _ in range(34):
        mid = (a + b) / 2
        if f(mid) == va:
            a = mid
        else:
            b = mid
    return (a + b) / 2


def _runs(on, ts, f, T):
    ivs, start = [], (0.0 if on[0] else None)
    for k in np.nonzero(on[1:] != on[:-1])[0]:
        x = _bisect(f, float(ts[k]), float(ts[k + 1]), bool(on[k]))
        if start is None:
            start = x
        else:
            ivs.append((start, x))
            start = None
    if start is not None:
        ivs.append((start, float(T)))
    return ivs


def _key(scene):
    return json.dumps({k: v for k, v in scene.items() if k in ("frames", "ground", "light", "casters", "markers")},
                      sort_keys=True)


@functools.lru_cache(maxsize=64)
def _events(key: str):
    scene = json.loads(key)
    T = t_end(scene)
    ts = _grid(scene)
    bnd = _bounds(scene, ts)
    nc = len(scene["casters"])
    per, union, graze = {}, {}, []
    for j in range(len(scene["markers"])):
        m = mx(scene, j)
        any_on = np.zeros_like(ts, dtype=bool)
        for i in range(nc):
            lo, hi = bnd[i]
            on = (lo <= m) & (m <= hi)
            any_on |= on
            per[(i, j)] = _runs(on, ts, lambda t, i=i, m=m: _covers(scene, i, m, t), T)
            d = np.minimum(m - lo, hi - m)               # signed depth of the marker inside the shadow
            dd = np.diff(d)
            ext = np.nonzero(np.sign(dd[1:]) * np.sign(dd[:-1]) < 0)[0] + 1
            graze += [float(abs(d[k])) for k in ext]
            graze += [float(abs(d[k])) + GRAZE - END_GAP for k in (0, -1)]   # near-miss at the start or end
        union[j] = _runs(any_on, ts, lambda t, m=m: any(_covers(scene, i, m, t) for i in range(nc)), T)
    return per, union, (min(graze) if graze else 1e9)


def _covers(scene, i, m, t) -> bool:
    lo, hi = shadow(scene, i, t)
    return lo <= m <= hi


def per_caster(scene):
    return _events(_key(scene))[0]


def union(scene):
    return _events(_key(scene))[1]


def min_graze(scene):
    return _events(_key(scene))[2]


def first_cover(scene, j):
    ivs = union(scene)[j]
    return ivs[0][0] if ivs else None


def shaded_time(scene, j) -> float:
    return sum(b - a for a, b in union(scene)[j])


def shaded_at(scene, j, t) -> bool:
    return any(a < t < b for a, b in union(scene)[j])


def edge_speed(scene, i, j, t) -> tuple[float, float]:
    """(speed px/s, signed distance px from the marker) of the shadow edge of caster i nearest marker j at t."""
    m = mx(scene, j)
    h = 1e-3
    lo, hi = shadow(scene, i, t)
    use = 0 if abs(lo - m) < abs(hi - m) else 1
    a, b = shadow(scene, i, t - h)[use], shadow(scene, i, t + h)[use]
    return abs(b - a) / (2 * h), (lo, hi)[use] - m


def edge_pos(scene, i, j, t_event, t) -> float:
    """Signed distance (px) at time t of the edge of caster i that crosses marker j at t_event."""
    m = mx(scene, j)
    lo, hi = shadow(scene, i, t_event)
    use = 0 if abs(lo - m) < abs(hi - m) else 1
    return shadow(scene, i, t)[use] - m


def events_at(scene, j):
    """All coverage edges (per caster and union) at marker j, sorted: [(t, caster)] (union edges have caster -1)."""
    out = [(x, i) for (i, jj), ivs in per_caster(scene).items() if jj == j for a, b in ivs for x in (a, b)]
    return sorted(out)


def without(scene, i: int) -> dict:
    return dict(scene, casters=[o for k, o in enumerate(scene["casters"]) if k != i])


def reversed_light(scene) -> dict:
    lt = dict(scene["light"])
    if lt["kind"] == "lamp":
        lt["x0"], lt["x1"] = lt["x1"], lt["x0"]
    else:
        lt["phi0"], lt["phi1"] = lt["phi1"], lt["phi0"]
    return dict(scene, light=lt)


def stilled(scene) -> dict:
    cs = [{k: v for k, v in o.items() if k not in ("vx", "vy")} for o in scene["casters"]]
    return dict(scene, casters=cs)


# ------------------------------------------------------------------ consistency check
def event_errors(scene) -> list:
    """Timing rules on coverage events. Also used on what-if scenes before asking about them."""
    errs = []
    T = t_end(scene)
    per, un = per_caster(scene), union(scene)
    for (i, j), ivs in per.items():
        for a, b in ivs:
            if a < EDGE:
                errs.append("a marker is in shadow at (or right after) the start")
            if b - a < GAP:
                errs.append("a shadow covers a marker too briefly")
            for x in (a, b):
                if x >= T - 1e-9:
                    continue
                if x > T - EDGE:
                    errs.append("a coverage event too near the end")
                sp, _ = edge_speed(scene, i, j, x)
                if sp < MIN_SPEED:
                    errs.append("a shadow edge crosses a marker too slowly")
    for j in range(len(scene["markers"])):
        ts = [t for t, _ in events_at(scene, j) if t < T - 1e-9]
        if any(b - a < GAP for a, b in zip(ts, ts[1:])):
            errs.append("two shadow edges cross one marker too close together in time")
        for (a, b), (c, d) in zip(un[j], un[j][1:]):
            if c - b < GAP:
                errs.append("a marker is out of shadow too briefly")
    if min_graze(scene) < GRAZE:
        errs.append("a shadow edge turns back right at a marker")
    return sorted(set(errs))


def geom_errors(scene, casters=None) -> list:
    """Frame layout: light, objects, clock and their clearances, sampled over the clip."""
    errs = []
    T = t_end(scene)
    G = scene["ground"]
    cs = scene["casters"] if casters is None else casters
    cx0, cy0, cx1, cy1 = CLOCK_BOX
    sr = scene["style"]["sun_r"]
    for t in np.linspace(0.0, T, 41):
        t = float(t)
        lx, ly = light(scene, t)
        if not (sr + 8 <= lx <= SIZE[0] - sr - 8 and sr + 8 <= ly):
            errs.append("the light leaves the frame")
        if lx - sr < cx1 + 14 and ly - sr < cy1 + 14:
            errs.append("the light touches the clock")
        boxes = []
        for o in cs:
            pts = outline(o, t)
            x0, y0, x1, y1 = bbox(o, t)
            if scene["light"]["kind"] == "lamp" and any(vy - ly < 30 or (G - ly) / (vy - ly) > KMAX for _, vy in pts):
                errs.append("the light is too close above an object")
            if x0 < 12 or x1 > SIZE[0] - 12 or y0 < 12:
                errs.append("an object leaves the frame")
            if x0 < cx1 + 14 and y0 < cy1 + 14:
                errs.append("an object touches the clock")
            if o["kind"] == "float" and y1 > G - 110:
                errs.append("a floating shape is too low")
            dx = max(x0 - lx, lx - x1, 0.0)
            dy = max(y0 - ly, ly - y1, 0.0)
            if math.hypot(dx, dy) < sr + 20:
                errs.append("the light passes too close to an object")
            boxes.append((x0, y0, x1, y1))
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                p, q = boxes[a], boxes[b]
                if not (p[2] + 26 <= q[0] or q[2] + 26 <= p[0] or p[3] + 26 <= q[1] or q[3] + 26 <= p[1]):
                    errs.append("two objects come too close")
        if errs:
            break
    return sorted(set(errs))


def check(scene) -> list:
    errs = []
    G = scene["ground"]
    cs = scene["casters"]
    if not MIN_FRAMES <= scene["frames"] <= MAX_FRAMES:
        errs.append("clip length out of range")
    if len({o["color"] for o in cs}) != len(cs) or not 2 <= len(cs) <= 5:
        errs.append("object count or colours wrong")
    errs += geom_errors(scene)
    mk = scene["markers"]
    if not 4 <= len(mk) <= 7:
        errs.append("marker count out of range")
    if mk != sorted(mk) or any(b - a < 90 for a, b in zip(mk, mk[1:])) or mk[0] < 50 or mk[-1] > SIZE[0] - 50:
        errs.append("markers too close together or to the frame edge")
    for o in cs:
        if o["kind"] == "stand" and any(abs(mx(scene, j) - o["x"]) < o["w"] / 2 + 30 for j in range(len(mk))):
            errs.append("a marker sits at the foot of a standing object")
    for lb, j in zip(scene["labels"], range(len(mk))):
        x0, y0, x1, y1 = lb["box"]
        if lb["text"] != str(j + 1) or y0 < G + scene["style"]["strip"] + 4 or y1 > SIZE[1] - 4:
            errs.append("a marker number is wrong or off the ground body")
    errs += event_errors(scene)
    un = union(scene)
    if sum(1 for j in range(len(mk)) if un[j]) < 3:
        errs.append("fewer than three markers are ever shaded")
    mv = [i for i, o in enumerate(cs) if o.get("vx") or o.get("vy")]
    if scene["scenario"] in ("drifting", "rising"):
        if len(mv) != 1 or not any(per_caster(scene)[(mv[0], j)] for j in range(len(mk))):
            errs.append("the moving shape's shadow never reaches a marker")
    elif mv:
        errs.append("a shape moves outside the drifting / rising scenarios")
    return sorted(set(errs))


# ------------------------------------------------------------------ sampling
def _r(x, n=2):
    return round(float(x), n)


def _style(rng: random.Random, scenario: str) -> dict:
    sky = "indoor" if scenario == "lamp" and rng.random() < 0.6 else rng.choice([k for k in SKIES if k != "indoor"])
    ground = rng.choice(list(GROUNDS))
    jit = lambda c, a: [max(0, min(255, v + rng.randint(-a, a))) for v in c]  # noqa: E731
    return {
        "sky": sky, "sky_top": jit(SKIES[sky][0], 10), "sky_bot": jit(SKIES[sky][1], 8),
        "sky_pattern": rng.choice(["gradient", "gradient", "flat", "bands"]),
        "ground": ground, "ground_rgb": jit(GROUNDS[ground], 10),
        "ground_tex": rng.choice(["plain", "speckle", "stripes", "bricks"]),
        "strip": rng.randint(24, 30),
        "plate": rng.choice(["rect", "diamond", "round"]), "plate_w": rng.randint(20, 26),
        "shade": _r(rng.uniform(0.40, 0.56)),
        "sun": rng.choice(list(SUNS)), "sun_r": rng.randint(17, 26),
        "sun_style": rng.choice(["disc", "rays", "ring"]),
        "outline": rng.choice([3, 3, 4]),
    }


def _caster(rng, color, kind, G, top_min):
    pat = rng.choice(STAND_PATTERNS if kind == "stand" else PATTERNS)
    if kind == "stand":
        shape = rng.choice(STAND_KINDS)
        if shape == "post":
            w, h, roof = rng.randint(20, 30), rng.randint(130, 250), 0
        elif shape == "pillar":
            w, h, roof = rng.randint(36, 66), rng.randint(110, 200), 0
        else:
            w, h, roof = rng.randint(44, 76), rng.randint(80, 150), rng.randint(26, 48)
        if G - h - roof < top_min:
            h = int(G - roof - top_min)
            if h < 90:
                return None
        return {"kind": "stand", "shape": shape, "color": color, "pattern": pat, "ground": G,
                "x": _r(rng.uniform(120, 1160), 1), "w": w, "h": h, "roof": roof}
    shape = rng.choice(FLOAT_SHAPES)
    s = rng.randint(22, 46)
    lo, hi = top_min + s, G - 150 - s
    if hi < lo:
        return None
    o = {"kind": "float", "shape": shape, "color": color, "pattern": pat, "ground": G,
         "x": _r(rng.uniform(110, 1170), 1), "y": _r(rng.uniform(lo, hi), 1), "s": s,
         "rot": 0 if shape == "circle" else rng.randint(0, 359)}
    if shape in ("ellipse", "rectangle"):
        o["aspect"] = _r(rng.uniform(1.4, 2.1))
    return o


def _move(rng, o, scenario, G, top_min, T):
    if scenario == "drifting":
        o["vx"] = _r(rng.uniform(16, 45) * rng.choice([-1, 1]))
        return True
    o["s"] = rng.randint(20, 30)
    lo, hi = top_min + o["s"], G - 130 - o["s"]          # range for the shape's centre
    v = min(rng.uniform(12, 28), (hi - lo) / T * 0.95)
    if v < 12:
        return False
    span = v * T
    up = rng.random() < 0.5
    y0 = rng.uniform(lo + span, hi) if up else rng.uniform(lo, hi - span)
    o["y"] = _r(y0, 1)
    o["vy"] = _r(-v if up else v)
    return True


def _light(rng, scenario):
    """(light, top_min): top_min is the highest an object's top may be (y px)."""
    if scenario == "lamp":
        y = rng.randint(70, 150)
        a, b = rng.uniform(260, 420), rng.uniform(860, 1180)
        if rng.random() < 0.5:
            a, b = b, a
        return {"kind": "lamp", "y": y, "x0": _r(a, 1), "x1": _r(b, 1)}, None
    top = rng.uniform(50, 95)                     # sun height at noon (y px)
    Rx, Ry = rng.uniform(520, 600), rng.uniform(260, 400)
    cx = rng.uniform(610, 670)
    a, b = rng.uniform(ELEV_MIN, 46), rng.uniform(134, 180 - ELEV_MIN)
    if rng.random() < 0.5:
        a, b = b, a
    return {"kind": "arc", "cx": _r(cx, 1), "cy": _r(top + Ry, 1), "Rx": _r(Rx, 1), "Ry": _r(Ry, 1),
            "phi0": _r(a), "phi1": _r(b)}, top + 110


def good_columns(scene):
    """For every ground column x in [60, 1220): whether a marker there would satisfy the timing rules
    (judged on the 1/240 s grid with a small extra margin), and which casters ever cover it."""
    T = t_end(scene)
    ts = _grid(scene)
    bnd = _bounds(scene, ts)
    xs = np.arange(60, SIZE[0] - 60) + 0.5
    nx = len(xs)
    events = [[] for _ in range(nx)]
    ok = np.ones(nx, dtype=bool)
    covers = np.zeros((len(scene["casters"]), nx), dtype=bool)
    early = ts < EDGE + 0.02
    for i, (lo, hi) in enumerate(bnd):
        dlo, dhi = np.abs(np.gradient(lo, ts)), np.abs(np.gradient(hi, ts))
        on = (lo[None, :] <= xs[:, None]) & (xs[:, None] <= hi[None, :])
        covers[i] = on.any(axis=1)
        ok &= ~on[:, early].any(axis=1)
        cc, kk = np.nonzero(on[:, 1:] != on[:, :-1])
        for c, k in zip(cc.tolist(), kk.tolist()):
            x = xs[c]
            sp = dlo[k] if abs(lo[k] - x) < abs(hi[k] - x) else dhi[k]
            t = float(ts[k])
            if t > T - EDGE - 0.02 or sp < MIN_SPEED * 1.1:
                ok[c] = False
            events[c].append(t)
        for b in (lo, hi):                        # turning points of a shadow edge: no marker close by
            ok &= (np.abs(xs - b[0]) >= END_GAP + 1) & (np.abs(xs - b[-1]) >= END_GAP + 1)
            db = np.diff(b)
            ext = np.nonzero(np.sign(db[1:]) * np.sign(db[:-1]) < 0)[0] + 1
            for k in ext.tolist():
                ok &= np.abs(xs - b[k]) >= GRAZE + 1
    for c in range(nx):
        ev = sorted(events[c])
        if ok[c] and any(b - a < GAP + 0.03 for a, b in zip(ev, ev[1:])):
            ok[c] = False
    for o in scene["casters"]:
        if o["kind"] == "stand":
            ok &= np.abs(xs - o["x"]) >= o["w"] / 2 + 34
    return xs, ok, covers


def _markers(rng, scene):
    xs, ok, covers = good_columns(scene)
    shaded = ok & covers.any(axis=0)
    dark = ok & ~covers.any(axis=0)
    mv = [i for i, o in enumerate(scene["casters"]) if o.get("vx") or o.get("vy")]
    n = rng.randint(4, 7)
    for _ in range(40):
        n_dark = min(int(dark.sum() > 0) * rng.choice([0, 0, 1, 1, 2]), n - 3)
        pool = [(1, c) for c in np.nonzero(shaded)[0].tolist()] + [(0, c) for c in np.nonzero(dark)[0].tolist()]
        if not pool:
            return None
        picked, want = [], {1: n - n_dark, 0: n_dark}
        if mv:
            cand = [c for c in np.nonzero(shaded & covers[mv[0]])[0].tolist()]
            if not cand:
                return None
            picked.append(rng.choice(cand))
            want[1] -= 1
        for kind in (1, 0):
            cols = [c for k, c in pool if k == kind]
            tries = 0
            while want[kind] > 0 and cols and tries < 200:
                tries += 1
                c = rng.choice(cols)
                if all(abs(c - p) >= 90 for p in picked):
                    picked.append(c)
                    want[kind] -= 1
        if want[1] <= 0 and want[0] <= 0:
            return sorted(int(xs[c] - 0.5) for c in picked)
    return None


def _labels(scene) -> list:
    from common.video5 import font
    f = font(LABEL_SIZE)
    G, strip = scene["ground"], scene["style"]["strip"]
    cy = G + strip + 34
    out = []
    for j, m in enumerate(scene["markers"]):
        txt = str(j + 1)
        x0, y0, x1, y1 = f.getbbox(txt, anchor="mm", stroke_width=3)
        cx = m + 0.5
        out.append({"text": txt, "x": cx, "y": cy, "size": LABEL_SIZE,
                    "box": [_r(cx + x0, 1), _r(cy + y0, 1), _r(cx + x1, 1), _r(cy + y1, 1)]})
    return out


def _attempt(rng: random.Random, scenario: str) -> dict | None:
    frames = rng.randint(MIN_FRAMES, MAX_FRAMES)
    T = (frames - 1) / FPS
    G = rng.randint(548, 610)
    lt, top_min = _light(rng, scenario)
    style = _style(rng, scenario)
    if top_min is None:                           # lamp: keep the projection factor below KMAX
        top_min = lt["y"] + (G - lt["y"]) / (KMAX - 0.25) + 4
    n = rng.randint(2, 5)
    if scenario == "posts":
        n_st = rng.randint(2, min(3, n))
    elif scenario == "floating":
        n_st = 0
    else:
        n_st = min(sum(rng.random() < 0.3 for _ in range(n)), n - 1)
    kinds = ["stand"] * n_st + ["float"] * (n - n_st)
    rng.shuffle(kinds)
    mover = kinds.index("float") if scenario in ("drifting", "rising") else -1
    cols = rng.sample(list(COLOURS), n)
    scene = {"scenario": scenario, "frames": frames, "ground": G, "light": lt, "style": style, "casters": []}
    for k in range(n):
        for _ in range(30):
            o = _caster(rng, cols[k], kinds[k], G, top_min)
            if o is None or (k == mover and not _move(rng, o, scenario, G, top_min, T)):
                continue
            if not geom_errors(scene, scene["casters"] + [o]):
                scene["casters"].append(o)
                break
        else:
            return None
    scene["casters"].sort(key=lambda o: o["x"])
    ms = _markers(rng, scene)
    if ms is None:
        return None
    scene["markers"] = ms
    scene["labels"] = _labels(scene)
    return scene


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    attempts = 0
    while True:
        attempts += 1
        scene = _attempt(rng, scenario)
        if scene is None:
            continue
        scene = json.loads(json.dumps(scene))
        if not check(scene):
            scene["build_attempts"] = attempts
            return scene
