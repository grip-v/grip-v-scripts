"""bouncing_ball_restitution scene sampler and bounce physics.

`build(rng)` returns a JSON-only scene: 3-5 objects (balls, blocks, hexagons,
diamonds) that fall under gravity and bounce on a floor, each bounce keeping a
fraction e of the impact speed (coefficient of restitution), so every rebound
rises to e^2 of the previous height. Four scenarios (`scene["scenario"]`):

    drop       3-5 objects released together from different heights
    staggered  3-5 objects hang in clamps and are released one at a time
    pads       3-5 objects fall onto coloured floor pads; the PAD sets e
    throw      2-3 objects slide sideways along stacked shelves while
               bouncing, and rebound off the left/right walls (elastic)

Physics is closed form and in pixel units (px, s, px/s^2): free fall between
contacts, v_out = e * v_in at each floor contact, and the "stop rule": when a
rebound would rise less than SETTLE_PX the object stops on the floor instead.
Walls in `throw` reverse the sideways velocity with no loss.

Everything here is a pure function of the stored (rounded) scene numbers, so
the validator and answers.py can work from the stored scene alone.
"""
from __future__ import annotations
import math
import random

W, H = 1280, 720
FPS = 24
SCENARIOS = ("drop", "staggered", "pads", "throw")
SETTLE_PX = 30.0          # rebounds lower than this end in the object coming to rest
BORDER = 1.12             # no rebound apex within [S/BORDER, S*BORDER] (borderline stop)
EDGE_S = 0.5              # events must be >= 0.5 s after start and before the end
AFTER_END_S = 0.3         # ... and no event in the first 0.3 s after the end either
SAME_OBJ_GAP = 3 / FPS    # events of one object at least 3 frames apart
TOP_Y = 100               # nothing drawn above this line except the timer
EXTRA_S = 8.0             # events are simulated this far past the end (for extrapolation)

COLORS = [  # name, rgb  (chosen to stay distinct when shrunk to 640x360)
    ("red", (222, 45, 45)), ("orange", (247, 138, 22)), ("yellow", (238, 206, 30)),
    ("green", (46, 168, 66)), ("cyan", (32, 196, 214)), ("blue", (44, 86, 224)),
    ("purple", (138, 64, 204)), ("pink", (240, 104, 184)), ("brown", (140, 88, 44)),
    ("lime", (160, 222, 40)),
]
PATTERNS = ("solid", "striped", "dotted", "ringed", "two-tone", "crossed")
SHAPES = ("circle", "square", "hexagon", "diamond")
NOUN = {"circle": "ball", "square": "block", "hexagon": "hexagon", "diamond": "diamond"}
BACKGROUNDS = [  # name, top rgb, bottom rgb, dark?
    ("paper", (246, 243, 234), (232, 228, 216), False),
    ("sky", (222, 236, 248), (246, 248, 252), False),
    ("sand", (243, 232, 212), (226, 212, 186), False),
    ("mint", (228, 244, 236), (212, 230, 222), False),
    ("slate", (52, 58, 70), (32, 36, 46), True),
    ("night", (24, 30, 58), (12, 16, 34), True),
    ("graphite", (70, 70, 74), (46, 46, 50), True),
]
OVERLAYS = ("none", "grid", "dots", "ruler", "lanes")
FLOORS = ("solid", "hatched", "tiles", "stripe")


def name_of(o: dict) -> str:
    return f"the {o['pattern']} {o['color']} {NOUN[o['shape']]}"


def r_min(o: dict) -> float:
    """Smallest size that keeps enough of the object's own colour visible
    after patterns, outlines and H.264 (a diamond covers half a square)."""
    return 20.0 if o["shape"] == "diamond" else 17.0


def half_w(o: dict) -> float:
    """Horizontal half-extent of the drawn shape (vertical half-extent is r)."""
    return o["r"] * (0.87 if o["shape"] == "hexagon" else 1.0)


def floor_y(scene: dict, o: dict) -> float:
    if scene["scenario"] == "throw":
        return scene["shelves"][o["lane"]]["floor_y"]
    return scene["floor_y"]


def restitution(scene: dict, o: dict) -> float:
    if scene["scenario"] == "pads":
        return scene["pads"][o["lane"]]["e"]
    return o["e"]


# ----------------------------------------------------------------- physics
def bounce(h0: float, e: float, g: float, release: float, t_max: float) -> dict:
    """Closed-form bounce history of an object dropped from rest at height
    h0 (bottom above the floor) at time `release`. Returns floor-contact
    times, the up-speed after each contact (0 = comes to rest there), the
    rebound apex heights and the rest time (None if still bouncing at t_max)."""
    t = release + math.sqrt(2.0 * h0 / g)
    v = math.sqrt(2.0 * g * h0)
    impacts, ups, apexes, rest = [], [], [], None
    while t <= t_max:
        impacts.append(t)
        v_up = e * v
        apex = v_up * v_up / (2.0 * g)
        apexes.append(apex)
        if apex < SETTLE_PX:
            ups.append(0.0)
            rest = t
            break
        ups.append(v_up)
        t += 2.0 * v_up / g
        v = v_up
    return {"impacts": impacts, "ups": ups, "apexes": apexes, "rest": rest}


def height_at(h0: float, g: float, release: float, hist: dict, t: float) -> float:
    if t <= release:
        return h0
    imp = hist["impacts"]
    if not imp or t < imp[0]:
        return max(0.0, h0 - 0.5 * g * (t - release) ** 2)
    k = 0
    while k + 1 < len(imp) and imp[k + 1] <= t:
        k += 1
    v = hist["ups"][k]
    tau = t - imp[k]
    return max(0.0, v * tau - 0.5 * g * tau * tau)


def x_track(x_start: float, vx: float, lo: float, hi: float):
    """Elastic reflection between x=lo and x=hi. Returns (x(t), wall_hits(t_max))."""
    span = hi - lo
    u0 = (x_start - lo) if vx > 0 else (hi - x_start)
    sp = abs(vx)

    def x_at(t: float) -> float:
        if sp == 0:
            return x_start
        u = u0 + sp * max(0.0, t)
        m = math.floor(u / span)
        frac = u - m * span
        fwd = (m % 2 == 0)
        if vx > 0:
            return lo + frac if fwd else hi - frac
        return hi - frac if fwd else lo + frac

    def hits(t_max: float) -> list[tuple[float, str]]:
        out, k = [], 1
        while sp > 0:
            t = (k * span - u0) / sp
            if t > t_max:
                break
            side = ("right" if k % 2 == 1 else "left") if vx > 0 else ("left" if k % 2 == 1 else "right")
            out.append((t, side))
            k += 1
        return out

    return x_at, hits


def x_limits(scene: dict, o: dict) -> tuple[float, float]:
    hw = half_w(o)
    return scene["wall_l"] + hw, scene["wall_r"] - hw


def history(scene: dict, o: dict, e: float | None = None, h0: float | None = None,
            t_max: float | None = None) -> dict:
    """Floor-contact history of object `o`, optionally with a changed e / h0."""
    return bounce(o["h0"] if h0 is None else h0, restitution(scene, o) if e is None else e,
                  scene["g"], o["release"], scene["duration"] + EXTRA_S if t_max is None else t_max)


def wall_hits(scene: dict, o: dict, vx: float | None = None, t_max: float | None = None):
    if scene["scenario"] != "throw":
        return []
    lo, hi = x_limits(scene, o)
    _, hits = x_track(o["x"], o["vx"] if vx is None else vx, lo, hi)
    return hits(scene["duration"] + EXTRA_S if t_max is None else t_max)


def center_at(scene: dict, o: dict, t: float, hist: dict | None = None) -> tuple[float, float]:
    """Drawn centre (px) of object o at time t."""
    hist = hist or history(scene, o)
    h = height_at(o["h0"], scene["g"], o["release"], hist, t)
    x = o["x"]
    if scene["scenario"] == "throw":
        lo, hi = x_limits(scene, o)
        x = x_track(o["x"], o["vx"], lo, hi)[0](t)
    return x, floor_y(scene, o) - o["r"] - h


def events(scene: dict, o: dict) -> list[tuple[float, str]]:
    """All of one object's events (release, floor contacts, wall hits) up to
    the end + EXTRA_S, sorted by time."""
    hist = history(scene, o)
    ev = [(t, "floor") for t in hist["impacts"]]
    if scene["scenario"] == "staggered":
        ev.append((o["release"], "release"))
    ev += [(t, f"wall_{s}") for t, s in wall_hits(scene, o)]
    return sorted(ev)


def frame_of(t: float) -> int:
    return int(round(t * FPS))


def timer_tenths(i: int) -> int:
    """What the on-screen clock shows at frame i, in tenths of a second."""
    return (10 * i) // FPS


def clean_time(t: float) -> bool:
    """True when both frames bracketing t show the same clock reading, and t
    is at least 0.03 s from a 0.1 s boundary (timer answers are then exact)."""
    a, b = math.floor(t * FPS), math.ceil(t * FPS)
    frac = t * 10 - math.floor(t * 10)
    return timer_tenths(a) == timer_tenths(b) and 0.3 <= frac <= 0.7


def clock_text(t: float) -> str:
    k = timer_tenths(frame_of(t))
    return f"{k // 10}.{k % 10} s"


def borderline(apexes) -> bool:
    return any(SETTLE_PX / BORDER <= a <= SETTLE_PX * BORDER for a in apexes)


# ----------------------------------------------------------------- sampling
def _contrast(rgb, bg) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, bg)))


def _style(rng: random.Random) -> dict:
    bg = rng.choice(BACKGROUNDS)
    return {"bg": bg[0], "bg_top": list(bg[1]), "bg_bottom": list(bg[2]), "dark": bg[3],
            "overlay": rng.choice(OVERLAYS), "floor": rng.choice(FLOORS),
            "floor_shade": round(rng.uniform(0.0, 1.0), 3), "outline": rng.choice([0, 2, 3])}


def _pick_colors(rng, style, n_obj, n_pads):
    bg = style["bg_top"]
    ok = [c for c in COLORS if _contrast(c[1], bg) >= 120 and _contrast(c[1], style["bg_bottom"]) >= 120]
    rng.shuffle(ok)
    return ok[:n_obj], ok[n_obj:n_obj + n_pads]


def _objects(rng, cols, n):
    objs = []
    for i, (cname, rgb) in enumerate(cols[:n]):
        objs.append({"id": i, "color": cname, "rgb": list(rgb), "pattern": rng.choice(PATTERNS),
                     "shape": rng.choice(SHAPES) if rng.random() < 0.7 else "circle"})
    return objs


def _sample_e(rng) -> float:
    return round(math.sqrt(rng.uniform(0.14, 0.82)), 3)


def _try_build(rng: random.Random, scenario: str) -> dict | None:
    style = _style(rng)
    if scenario == "throw":
        g, n = round(rng.uniform(380.0, 700.0), 1), rng.randint(2, 3)
    else:
        g, n = round(rng.uniform(650.0, 1500.0), 1), rng.randint(3, 5)
    cols, pad_cols = _pick_colors(rng, style, n, n if scenario == "pads" else 0)
    if len(cols) < n or (scenario == "pads" and len(pad_cols) < n):
        return None
    objs = _objects(rng, cols, n)
    scene = {"scenario": scenario, "g": g, "settle_px": SETTLE_PX, "style": style, "objects": objs}
    if scenario == "throw":
        _layout_throw(rng, scene)
    else:
        _layout_lanes(rng, scene, pad_cols)
    for o in objs:
        for _ in range(40):
            ok = _sample_motion(rng, scene, o)
            if ok:
                break
        else:
            return None
    if scenario == "staggered":
        ts, t = [], rng.uniform(0.6, 1.4)
        for _ in objs:
            ts.append(round(t, 3))
            t += rng.uniform(0.5, 1.3)
        order = list(range(n))
        rng.shuffle(order)
        for k, i in enumerate(order):
            objs[i]["release"] = ts[k]
    # duration: everyone at rest + a pause, capped at 10 s; then slide off any event
    rests = []
    for o in objs:
        hist = history(scene, o, t_max=60.0)
        rests.append(hist["rest"] if hist["rest"] is not None else 60.0)
    target = min(10.0, max(rests) + rng.uniform(0.8, 2.0)) if rng.random() < 0.6 else rng.uniform(5.5, 10.0)
    d = math.floor(target * FPS) / FPS
    scene["duration"] = round(d, 4)
    all_ev = sorted(t for o in objs for t, _ in events(scene, o))
    while d >= 4.0:
        if not any(d - EDGE_S - 1e-9 < t < d + AFTER_END_S for t in all_ev):
            break
        d -= 1.0 / FPS
    if d < 4.0:
        return None
    scene["duration"] = round(d, 4)
    return scene


def _pool_errors(scene) -> list[str]:
    import questions  # lazy: questions imports sim
    sizes = [len(v) for v in questions.candidates(scene, random.Random(0)).values()]
    lo = [1, 2, 2, 1, 1]
    if any(s < m for s, m in zip(sizes, lo)) or sum(min(s, 3) for s in sizes) < 10:
        return [f"not enough askable questions per level: {sizes}"]
    return []


def _layout_lanes(rng, scene, pad_cols):
    n = len(scene["objects"])
    fy = rng.randint(590, 680)
    width = rng.randint(max(560, 190 * n), 1180)
    x0 = rng.randint(50, W - 50 - width)
    lane = width / n
    order = list(range(n))
    rng.shuffle(order)
    rmax = min(40.0, lane / 2 - 22)
    for k, i in enumerate(order):
        o = scene["objects"][i]
        o["lane"] = k
        o["r"] = round(rng.uniform(r_min(o), max(r_min(o) + 1, rmax)), 1)
        o["x"] = round(x0 + lane * (k + 0.5) + rng.uniform(-0.12, 0.12) * lane, 1)
        o["release"] = 0.0
        o["vx"] = 0.0
    scene.update({"floor_y": fy, "wall_l": x0, "wall_r": x0 + width})
    if scene["scenario"] == "pads":
        pads = []
        for k in range(n):
            cname, rgb = pad_cols[k]
            pads.append({"color": cname, "rgb": list(rgb), "e": _sample_e(rng),
                         "x0": round(x0 + lane * k + 10, 1), "x1": round(x0 + lane * (k + 1) - 10, 1),
                         "pattern": rng.choice(["solid", "hatched", "dotted"])})
        scene["pads"] = pads
        for o in scene["objects"]:
            o["e"] = None


def _layout_throw(rng, scene):
    n = len(scene["objects"])
    top, bottom = TOP_Y + rng.randint(0, 20), rng.randint(660, 700)
    hgt = (bottom - top) / n
    x0 = rng.randint(40, 200)
    x1 = W - rng.randint(40, 200)
    shelves = [{"floor_y": round(top + hgt * (k + 1), 1), "top_y": round(top + hgt * k, 1)} for k in range(n)]
    order = list(range(n))
    rng.shuffle(order)
    rmax = min(36.0, hgt * 0.22)
    for k, i in enumerate(order):
        o = scene["objects"][i]
        o["lane"] = k
        o["r"] = round(rng.uniform(r_min(o), max(r_min(o) + 1, rmax)), 1)
        o["release"] = 0.0
        hw = half_w(o)
        o["x"] = round(rng.uniform(x0 + hw + 60, x1 - hw - 60), 1)
        o["vx"] = round(rng.choice([-1, 1]) * rng.uniform(110.0, 420.0), 1)
    scene.update({"shelves": shelves, "wall_l": x0, "wall_r": x1})


def _sample_motion(rng, scene, o) -> bool:
    if scene["scenario"] != "pads":
        o["e"] = _sample_e(rng)
    g = scene["g"]
    fy = floor_y(scene, o)
    ceiling = scene["shelves"][o["lane"]]["top_y"] + 8 if scene["scenario"] == "throw" else TOP_Y + 18
    hmax = fy - ceiling - 2 * o["r"]
    hmin = max(0.16 * g, 3.2 * SETTLE_PX)
    if hmax < hmin + 20:
        return False
    o["h0"] = round(rng.uniform(hmin, hmax), 1)
    hist = history(scene, o, t_max=60.0)
    return not borderline(hist["apexes"])


def build_with_attempts(rng: random.Random) -> tuple[dict, int]:
    scenario = rng.choice(SCENARIOS)
    for attempt in range(1, 500):
        scene = _try_build(rng, scenario)
        if scene is not None and not check(scene):
            return scene, attempt
    raise RuntimeError("could not build a valid scene")


def build(rng: random.Random) -> dict:
    return build_with_attempts(rng)[0]


# ----------------------------------------------------------------- checks
def check(scene: dict) -> list[str]:
    errs = []
    objs, d = scene["objects"], scene["duration"]
    if not 4.0 <= d <= 10.0 + 1e-9:
        errs.append(f"duration {d} outside 4-10 s")
    descs = [(o["color"], o["pattern"], o["shape"]) for o in objs]
    if len(set(o["color"] for o in objs)) != len(objs) or len(set(descs)) != len(descs):
        errs.append("objects are not uniquely identifiable")
    if scene["scenario"] == "pads":
        pc = [p["color"] for p in scene["pads"]]
        if len(set(pc)) != len(pc) or set(pc) & set(o["color"] for o in objs):
            errs.append("pad colours clash")
    if scene["scenario"] != "throw":
        xs = sorted(objs, key=lambda o: o["x"])
        for a, b in zip(xs, xs[1:]):
            if b["x"] - a["x"] < half_w(a) + half_w(b) + 20:
                errs.append("lanes too close")
    for o in objs:
        if o["r"] < r_min(o) - 1e-9:
            errs.append(f"{name_of(o)} too small")
        hist = history(scene, o)
        if borderline(hist["apexes"]):
            errs.append(f"{name_of(o)} has a borderline stop")
        if not 0 < restitution(scene, o) < 1:
            errs.append("bad restitution")
        ev = events(scene, o)
        for t, kind in ev:
            if t < EDGE_S or d - EDGE_S < t < d + AFTER_END_S:
                errs.append(f"{name_of(o)} {kind} at {t:.3f}s is too close to the start or end")
        if o["release"] > d - EDGE_S - 0.5:
            errs.append(f"{name_of(o)} is released too late to be seen falling")
        for (ta, _), (tb, _) in zip(ev, ev[1:]):
            if tb - ta < SAME_OBJ_GAP:
                errs.append(f"{name_of(o)} has two events {tb - ta:.3f}s apart")
        top = floor_y(scene, o) - o["h0"] - 2 * o["r"]
        lim = scene["shelves"][o["lane"]]["top_y"] if scene["scenario"] == "throw" else TOP_Y
        if top < lim:
            errs.append(f"{name_of(o)} starts above its space")
        if scene["scenario"] == "throw":
            lo, hi = x_limits(scene, o)
            if not lo <= o["x"] <= hi:
                errs.append(f"{name_of(o)} starts inside a wall")
    return errs or _pool_errors(scene)
