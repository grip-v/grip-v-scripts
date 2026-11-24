"""object_tracking_mot_v2: multiple-object tracking in a walled arena.

3-8 objects move inside a rectangular arena. Each is a disc for the physics
(drawn as a coloured shape inscribed in that disc). Scenarios
(scene["scenario"]):

  colours    unique objects (colour + shape + pattern) on straight paths that
             bounce elastically off the walls and off each other
  targets    identical objects; 1-3 are ringed while everything stands still at
             the start, then the rings vanish and all objects move (classic MOT)
  occluders  unique objects + 1-2 opaque screens drawn over them; an object is
             hidden while it overlaps a screen
  orbit      unique objects on smooth closed curves (no collisions; paths may
             cross, later objects drawn on top)

Faint unnumbered centre lines split the arena into four quadrants (the only
position vocabulary). The clock is the only text on screen.

Bouncing motion is exact and event driven: equal-mass elastic discs, walls
reflect, collisions exchange the normal velocity components. Times are
seconds; frame f is shown at t = f / 24, the clock shows f * 10 // 24 tenths.
Pure function of the rng passed in.
"""
from __future__ import annotations
import math
import random

FPS = 24
MAX_FRAMES = 240
EDGE_S = 0.5
AREA = (250, 20, 1262, 704)
CLOCK_BOX = (18, 16, 214, 84)
SCENARIOS = ("colours", "targets", "occluders", "orbit")
EV_GAP = 0.3            # s: an object's events (bounces / collisions) are this far apart
GAP_PX = 30.0           # discs never closer than this unless they collide (a miss never looks like a touch)

COLOURS = {
    "red": (222, 40, 40), "orange": (247, 130, 20), "yellow": (240, 200, 20), "green": (35, 165, 60),
    "teal": (0, 170, 170), "blue": (40, 90, 230), "purple": (140, 60, 210), "pink": (240, 100, 190),
    "brown": (135, 80, 35),
}
SHAPES = ("circle", "square", "hexagon", "diamond", "triangle")
PATTERNS = ("solid", "striped", "dotted", "ringed")
SCREENS = {"slate": (88, 96, 112), "sand": (196, 178, 140), "olive": (110, 118, 70), "plum": (110, 80, 110)}
THEMES = {
    "light": {"bg": [(236, 233, 226), (222, 228, 234), (232, 230, 238)], "arena": [(250, 250, 246), (244, 247, 250)],
              "wall": [(70, 70, 76), (60, 66, 80)], "line": [(200, 196, 188), (190, 198, 210)],
              "ring": (20, 20, 24), "same": [(70, 110, 200), (200, 90, 60), (60, 150, 90)]},
    "dark": {"bg": [(18, 20, 26), (26, 22, 30), (16, 26, 28)], "arena": [(36, 40, 50), (44, 40, 52), (34, 46, 48)],
             "wall": [(200, 200, 206), (190, 196, 210)], "line": [(70, 76, 90), (84, 78, 96)],
             "ring": (250, 250, 250), "same": [(120, 170, 255), (255, 150, 100), (120, 220, 150)]},
}


# ------------------------------------------------------------------ exact event-driven motion
def _wall_time(p, v, r, lo, hi):
    """Time until a disc at p moving at v touches the wall interval [lo + r, hi - r]."""
    if v > 1e-12:
        return (hi - r - p) / v
    if v < -1e-12:
        return (lo + r - p) / v
    return math.inf


def _pair_time(a, b):
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    dvx, dvy = b["vx"] - a["vx"], b["vy"] - a["vy"]
    R = a["r"] + b["r"]
    A = dvx * dvx + dvy * dvy
    B = 2 * (dx * dvx + dy * dvy)
    C = dx * dx + dy * dy - R * R
    if A < 1e-12 or B >= 0:
        return math.inf
    disc = B * B - 4 * A * C
    if disc < 0:
        return math.inf
    t = (-B - math.sqrt(disc)) / (2 * A)
    return t if t > 1e-9 else (0.0 if C < 1e-6 else math.inf)


def simulate(objs, box, t_end, start=0.0):
    """objs: [{id, x, y, vx, vy, r}] at t = start (objects may also carry "t0": they stay put until t0).
    box: (x0, y0, x1, y1). Returns {"segs": {id: [(t, x, y, vx, vy)]}, "events": [(t, kind, ids, where)]}
    with kind "wall" (where = left/right/top/bottom) or "hit" (ids = the pair)."""
    st = [dict(o) for o in objs]
    x0, y0, x1, y1 = box
    segs = {o["id"]: [(start, o["x"], o["y"], o["vx"], o["vy"])] for o in st}
    events = []
    t = start
    for _ in range(4000):
        best, what = t_end - t, None
        for i, o in enumerate(st):
            for axis, lo, hi in (("x", x0, x1), ("y", y0, y1)):
                dt = _wall_time(o[axis], o["v" + axis], o["r"], lo, hi)
                if dt < best - 1e-12:
                    best, what = max(dt, 0.0), ("wall", i, axis)
            for j in range(i + 1, len(st)):
                dt = _pair_time(o, st[j])
                if dt < best - 1e-12:
                    best, what = dt, ("hit", i, j)
        for o in st:
            o["x"] += o["vx"] * best
            o["y"] += o["vy"] * best
        t += best
        if what is None:
            break
        if what[0] == "wall":
            o = st[what[1]]
            ax = what[2]
            side = ("right" if o["vx"] > 0 else "left") if ax == "x" else ("bottom" if o["vy"] > 0 else "top")
            o["v" + ax] = -o["v" + ax]
            events.append((t, "wall", (o["id"],), side))
            segs[o["id"]].append((t, o["x"], o["y"], o["vx"], o["vy"]))
        else:
            a, b = st[what[1]], st[what[2]]
            nx, ny = b["x"] - a["x"], b["y"] - a["y"]
            n = math.hypot(nx, ny)
            nx, ny = nx / n, ny / n
            pa = a["vx"] * nx + a["vy"] * ny
            pb = b["vx"] * nx + b["vy"] * ny
            a["vx"] += (pb - pa) * nx
            a["vy"] += (pb - pa) * ny
            b["vx"] += (pa - pb) * nx
            b["vy"] += (pa - pb) * ny
            events.append((t, "hit", (a["id"], b["id"]), ""))
            segs[a["id"]].append((t, a["x"], a["y"], a["vx"], a["vy"]))
            segs[b["id"]].append((t, b["x"], b["y"], b["vx"], b["vy"]))
    return {"segs": segs, "events": events}


def seg_pos(segs, t):
    lo, hi = 0, len(segs) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if segs[mid][0] <= t:
            lo = mid
        else:
            hi = mid - 1
    t0, x, y, vx, vy = segs[lo]
    return x + vx * (t - t0), y + vy * (t - t0)


def orbit_pos(o, t):
    return (o["cx"] + o["ax"] * math.sin(o["wx"] * t + o["px"]), o["cy"] + o["ay"] * math.sin(o["wy"] * t + o["py"]))


# ------------------------------------------------------------------ scene-level helpers
def box_of(scene):
    a = scene["arena"]
    return (a["x0"], a["y0"], a["x1"], a["y1"])


def motion(scene):
    """{"segs", "events"} for the scene. Cached per scene object (identity), so a hot loop over
    position() costs one simulation; a copy of the scene simply simulates once more."""
    hit = _MEMO.get(id(scene))
    if hit is not None and hit[0] is scene:
        return hit[1]
    if scene["scenario"] == "orbit":
        res = {"segs": None, "events": []}
    else:
        objs = [{"id": o["id"], "x": o["x"], "y": o["y"], "vx": o["vx"], "vy": o["vy"], "r": o["r"]}
                for o in scene["objects"]]
        res = simulate(objs, box_of(scene), duration(scene) + 4.0, scene.get("start_t", 0.0))
    if len(_MEMO) > 64:
        _MEMO.clear()
    _MEMO[id(scene)] = (scene, res)                     # holding the scene keeps its id from being reused
    return res


_MEMO: dict = {}


def position(scene, oid, t):
    o = scene["objects"][oid]
    if scene["scenario"] == "orbit":
        return orbit_pos(o, t)
    if t <= scene.get("start_t", 0.0):
        return o["x"], o["y"]
    return seg_pos(motion(scene)["segs"][oid], t)


EXTENT = {"circle": (1, 1, 1, 1), "square": (0.72, 0.72, 0.72, 0.72), "diamond": (1, 1, 1, 1),
          "triangle": (0.866, 0.866, 1, 0.5), "hexagon": (1, 1, 0.866, 0.866)}   # left, right, up, down (x r)


def shape_box(o, x, y):
    """Box the drawn shape covers when its disc is centred at (x, y)."""
    l, r_, u, d = EXTENT[o["shape"]]
    r = o["r"]
    return x - l * r, y - u * r, x + r_ * r, y + d * r


def outline(o, x, y, n=48):
    """Points on the drawn shape's outline (same geometry as render.sprite), disc centred at (x, y)."""
    r, sh = o["r"], o["shape"]
    if sh == "circle":
        return [(x + r * math.cos(2 * math.pi * k / n), y + r * math.sin(2 * math.pi * k / n)) for k in range(n)]
    if sh == "square":
        h = 0.72 * r
        vs = [(-h, -h), (h, -h), (h, h), (-h, h)]
    elif sh == "diamond":
        vs = [(0, -r), (r, 0), (0, r), (-r, 0)]
    elif sh == "triangle":
        vs = [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a))) for a in (-90, 30, 150)]
    else:
        vs = [(r * math.cos(math.pi / 3 * k), r * math.sin(math.pi / 3 * k)) for k in range(6)]
    pts = []
    for (ax, ay), (bx, by) in zip(vs, vs[1:] + vs[:1]):
        for k in range(12):
            u = k / 12
            pts.append((x + ax + (bx - ax) * u, y + ay + (by - ay) * u))
    return pts


def hidden(scene, oid, t):
    """Some of the drawn shape is under a screen by more than 3 px (a graze of 3 px or less does not count)."""
    if not scene["screens"]:
        return False
    o = scene["objects"][oid]
    pts = outline(o, *position(scene, oid, t))
    d = 3.0
    return any(s["x0"] + d < px < s["x1"] - d and s["y0"] + d < py < s["y1"] - d
               for s in scene["screens"] for px, py in pts)


def n_frames(scene):
    return scene["frames"]


def duration(scene):
    return scene["frames"] / FPS


def clock_tenths(frame: int) -> int:
    return frame * 10 // FPS


def quadrant(scene, x, y):
    a = scene["arena"]
    cx, cy = (a["x0"] + a["x1"]) / 2, (a["y0"] + a["y1"]) / 2
    return ("top" if y < cy else "bottom") + "-" + ("left" if x < cx else "right")


def quad_margin(scene, x, y):
    a = scene["arena"]
    return min(abs(x - (a["x0"] + a["x1"]) / 2), abs(y - (a["y0"] + a["y1"]) / 2))


# ------------------------------------------------------------------ generation
def _style(rng):
    theme = rng.choice(("light", "light", "dark"))
    T = THEMES[theme]
    return {"theme": theme, "bg": list(rng.choice(T["bg"])), "arena": list(rng.choice(T["arena"])),
            "wall": list(rng.choice(T["wall"])), "wall_w": rng.randint(4, 8), "line": list(rng.choice(T["line"])),
            "line_style": rng.choice(("solid", "dashed", "dotted")), "floor": rng.choice(("plain", "grid", "dots")),
            "floor_step": rng.randint(40, 80), "ring": list(T["ring"]), "same": list(rng.choice(T["same"]))}


def _arena(rng):
    w, h = rng.randint(560, 1000), rng.randint(420, 670)
    x0 = rng.randint(AREA[0] + 6, AREA[2] - 6 - w)
    y0 = rng.randint(AREA[1] + 6, AREA[3] - 6 - h)
    return {"x0": x0, "y0": y0, "x1": x0 + w, "y1": y0 + h}


def _place(rng, n, box, radii, gap=24, avoid=()):
    pts = []
    x0, y0, x1, y1 = box
    for r in radii:
        for _ in range(200):
            p = (rng.uniform(x0 + r + 4, x1 - r - 4), rng.uniform(y0 + r + 4, y1 - r - 4))
            if any(s["x0"] - r - 12 < p[0] < s["x1"] + r + 12 and s["y0"] - r - 12 < p[1] < s["y1"] + r + 12
                   for s in avoid):
                continue
            if all(math.dist(p, q) > r + rr + gap for q, rr in pts):
                pts.append((p, r))
                break
        else:
            return None
    return [p for p, _ in pts]


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    for attempt in range(1, 20000):
        st = _style(rng)
        arena = _arena(rng)
        box = (arena["x0"], arena["y0"], arena["x1"], arena["y1"])
        n = rng.randint(5, 8) if scenario == "targets" else rng.randint(3, 7)
        frames = rng.randint(144, MAX_FRAMES)
        same_shape = rng.choice(SHAPES)
        cols = rng.sample(sorted(COLOURS), n) if scenario != "targets" else [None] * n
        radii = [rng.randint(18, 30) if scenario == "targets" else rng.randint(14, 30) for _ in range(n)]
        if scenario == "targets":
            radii = [radii[0]] * n
        screens = []
        if scenario == "occluders":
            names = rng.sample(sorted(SCREENS), rng.randint(1, 2))
            for k, nm in enumerate(names):
                vertical = rng.random() < 0.5
                if vertical:
                    w = rng.randint(70, 120)
                    sx = rng.uniform(box[0] + 80, box[2] - 80 - w)
                    sy0 = rng.uniform(box[1], box[1] + 0.3 * (box[3] - box[1]))
                    sy1 = rng.uniform(box[3] - 0.3 * (box[3] - box[1]), box[3])
                    s = {"name": nm, "x0": round(sx, 2), "x1": round(sx + w, 2), "y0": round(sy0, 2), "y1": round(sy1, 2)}
                else:
                    h = rng.randint(70, 110)
                    sy = rng.uniform(box[1] + 70, box[3] - 70 - h)
                    sx0 = rng.uniform(box[0], box[0] + 0.3 * (box[2] - box[0]))
                    sx1 = rng.uniform(box[2] - 0.3 * (box[2] - box[0]), box[2])
                    s = {"name": nm, "x0": round(sx0, 2), "x1": round(sx1, 2), "y0": round(sy, 2), "y1": round(sy + h, 2)}
                screens.append(s)
            if len(screens) == 2 and not (screens[0]["x1"] + 30 < screens[1]["x0"] or screens[1]["x1"] + 30 < screens[0]["x0"]
                                          or screens[0]["y1"] + 30 < screens[1]["y0"] or screens[1]["y1"] + 30 < screens[0]["y0"]):
                continue
        pts = _place(rng, n, box, radii, avoid=screens)          # nothing starts behind a screen
        if pts is None:
            continue
        objs = []
        for i in range(n):
            o = {"id": i, "r": radii[i], "colour": cols[i] if cols[i] else "same",
                 "shape": same_shape if scenario == "targets" else rng.choice(SHAPES),
                 "pattern": "solid" if scenario == "targets" else rng.choice(PATTERNS)}
            if scenario == "orbit":
                ax = rng.uniform(60, (box[2] - box[0]) / 2 - radii[i] - 6)
                ay = rng.uniform(50, (box[3] - box[1]) / 2 - radii[i] - 6)
                cx = rng.uniform(box[0] + ax + radii[i] + 4, box[2] - ax - radii[i] - 4)
                cy = rng.uniform(box[1] + ay + radii[i] + 4, box[3] - ay - radii[i] - 4)
                o.update(cx=round(cx, 3), cy=round(cy, 3), ax=round(ax, 3), ay=round(ay, 3),
                         wx=round(rng.uniform(0.5, 1.6), 4), wy=round(rng.uniform(0.5, 1.6), 4),
                         px=round(rng.uniform(0, 2 * math.pi), 4), py=round(rng.uniform(0, 2 * math.pi), 4))
            else:
                sp = rng.uniform(90, 260)
                a = rng.uniform(0.25, math.pi / 2 - 0.25) + rng.randrange(4) * math.pi / 2   # never near an axis
                o.update(x=round(pts[i][0], 3), y=round(pts[i][1], 3), vx=round(sp * math.cos(a), 3),
                         vy=round(sp * math.sin(a), 3))
            objs.append(o)
        targets = []
        start_t = 0.0
        if scenario == "targets":
            targets = sorted(rng.sample(range(n), rng.randint(1, min(3, n - 2))))
            start_t = round(rng.uniform(1.0, 1.6) * FPS) / FPS
        scene = {"scenario": scenario, "frames": frames, "arena": arena, "objects": objs, "screens": screens,
                 "targets": targets, "start_t": start_t, "style": st, "build_attempts": attempt}
        if check(scene) or not _ladder_ok(scene):
            continue
        return scene
    raise RuntimeError("no valid scene")


def _ladder_ok(scene):
    import questions                                   # late import: questions.py imports this module
    try:
        questions.build(scene, random.Random(0))
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------ consistency check
def clear_events(mo, t_end):
    """Every bounce and collision before t_end is clearly visible: a wall is hit at >= 40 px/s along its normal,
    a collision changes each object's velocity by >= max(40 px/s, 30 %)."""
    segs = mo["segs"]
    for t, kind, ids, side in mo["events"]:
        if t > t_end:
            continue
        for i in ids:
            k = next(k for k, sg in enumerate(segs[i]) if abs(sg[0] - t) < 1e-9)
            v0, v1 = segs[i][k - 1][3:5], segs[i][k][3:5]
            if kind == "wall":
                if abs(v0[0] if side in ("left", "right") else v0[1]) < 40.0:
                    return ["a wall graze (bounce under 40 px/s)"]
            elif math.hypot(v1[0] - v0[0], v1[1] - v0[1]) < max(40.0, 0.3 * math.hypot(*v0)):
                return ["a glancing collision"]
    return []



def check(scene: dict) -> list[str]:
    errs = []
    dur = duration(scene)
    objs = scene["objects"]
    n = len(objs)
    if scene["scenario"] != "targets":
        looks = [o["colour"] for o in objs]
        if len(set(looks)) != n:
            errs.append("two objects share a colour")
    if any(o["r"] < 10 for o in objs):
        errs.append("object under 20 px")
    if scene["scenario"] != "orbit":
        mo = motion(scene)
        ev = mo["events"]
        # per object: events EV_GAP apart (so each bounce / collision is a clear, separate event). The targets
        # scenario asks only where objects are, never about events, so its events need no spacing.
        spaced = scene["scenario"] != "targets"
        per = {}
        for t, kind, ids, _ in ev:
            for i in ids:
                per.setdefault(i, []).append(t)
        for i, ts in (per.items() if spaced else ()):
            ts = [t for t in ts if t <= dur + 1]
            if any(b - a < EV_GAP for a, b in zip(ts, ts[1:])):
                errs.append("an object has two events too close together")
                break
        errs += clear_events(mo, dur + 1)               # no wall grazes, no glancing collisions
        if scene["scenario"] == "targets":
            # identical objects: a ringed one must never bump an un-ringed one (a head-on bounce looks exactly
            # like the two passing through each other, so which one is the target would be ambiguous)
            tg = set(scene["targets"])
            if any(kind == "hit" and len(tg & set(ids)) == 1 and t <= dur + 0.5 for t, kind, ids, _ in ev):
                errs.append("a ringed object bumps an un-ringed one")
        # no two events (any objects) within 2 frames, and none within 2 frames of the clip ends
        allt = sorted(t for t, *_ in ev if t <= dur + 1)
        if spaced and any(b - a < 2 / FPS for a, b in zip(allt, allt[1:])):
            errs.append("two events within 2 frames")
        if spaced and any(abs(t - dur) < 2 / FPS or t < scene["start_t"] + 2 / FPS for t in allt):
            errs.append("an event at a clip edge")
        # no near-misses: every local minimum of a pair's gap is either a contact (a collision) or >= GAP_PX
        times = [k / 96 for k in range(int(dur * 96) + 1)]
        P = [[position(scene, i, t) for t in times] for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                R = objs[i]["r"] + objs[j]["r"]
                g = [math.dist(P[i][k], P[j][k]) - R for k in range(len(times))]
                for k in range(1, len(g) - 1):
                    if g[k] <= g[k - 1] and g[k] <= g[k + 1] and 0.5 < g[k] < GAP_PX:
                        if not any(abs(times[k] - e[0]) < 0.03 and set(e[2]) == {i, j} for e in ev if e[1] == "hit"):
                            errs.append("two objects nearly touch without colliding")
                            break
                if errs and errs[-1].startswith("two objects nearly"):
                    break
            if errs and errs[-1].startswith("two objects nearly"):
                break
    else:
        # orbits stay inside; crossings allowed
        a = scene["arena"]
        for o in objs:
            if o["cx"] - o["ax"] - o["r"] < a["x0"] or o["cx"] + o["ax"] + o["r"] > a["x1"] or \
                    o["cy"] - o["ay"] - o["r"] < a["y0"] or o["cy"] + o["ay"] + o["r"] > a["y1"]:
                errs.append("an orbit leaves the arena")
    if scene["scenario"] == "occluders":
        # nothing bounces or collides while hidden, so a hidden object moves straight and comes out where expected
        for t, kind, ids, _ in motion(scene)["events"]:
            # every object in an event is fully seen for 3 frames either side of it
            if t <= dur + 0.2 and any(hidden(scene, i, t + k / FPS) for i in ids for k in range(-3, 4)):
                errs.append("an event happens behind a screen")
                break
    a = scene["arena"]
    if a["x0"] < AREA[0] or a["y0"] < AREA[1] or a["x1"] > AREA[2] or a["y1"] > AREA[3]:
        errs.append("arena leaves its area")
    return errs
