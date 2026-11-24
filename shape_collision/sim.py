"""Scene generation for shape_collision (v3, 10-question ladder).

Shapes drift inside a walled box and bounce elastically off the walls and
off each other. Each shape is a disc for the physics (mass ~ r^2), drawn as
an outline (13 types) inside a thin ring that marks the disc, so every
contact is visible ring-to-ring. Every shape has its own colour (no two
alike), so "the blue star" is always unique.

Five scenarios (`scene["scenario"]`), picked per seed:

    swap    open box; every shape wears its own fill pattern and two shapes
            that collide SWAP patterns
    infect  open box; every shape starts solid except one that wears the
            scene's tag pattern; a collision between a tagged and an untagged
            shape tags the other one too (tags never go away)
    posts   like swap, plus 2-5 fixed round posts that shapes bounce off
    holes   like swap, plus 2-4 holes in the floor; a shape whose centre
            passes over a hole falls in and is gone for the rest of the clip
    rails   like swap, but the shapes ride on 2-4 horizontal rails and only
            meet shapes on their own rail (head-on 1-D collisions with
            unequal masses)

The motion is solved exactly, event by event (next wall / post / hole /
pair contact from closed-form straight-line motion), so `positions(scene, t)`
is exact at any time. Scenes are only kept when every event is clean: well
apart in time, not glancing, no near misses, nothing in the first or last
0.55 s of the clip (see `check`).

Time: frame f (0-based, 24 fps) shows the world at t = f / 24 and the clock
reads floor(10 f / 24) tenths of a second. An event at time t first shows its
result on frame ceil(24 t).
"""
from __future__ import annotations
import bisect
import functools
import json
import math
import random

W, H, FPS = 1280, 720, 24
MAX_FRAMES = 240                    # 10 s
MIN_FRAMES = 132                    # 5.5 s
SCENARIOS = ("swap", "infect", "posts", "holes", "rails")
SHAPES = ["circle", "square", "triangle", "pentagon", "hexagon", "octagon", "star", "cross", "heart",
          "diamond", "trapezoid"]       # no thin outlines: the pattern must stay readable at half size
COLORS = {"red": (232, 52, 48), "orange": (247, 142, 28), "yellow": (246, 218, 40), "lime": (150, 222, 46),
          "green": (30, 150, 70), "cyan": (40, 205, 230), "blue": (50, 100, 240), "purple": (150, 75, 220),
          "pink": (248, 120, 196), "white": (242, 242, 242)}
PATTERNS = ["solid", "hollow", "horizontal stripes", "vertical stripes", "diagonal stripes", "dots", "checks",
            "grid", "rings", "two-tone"]
TAGS = ["horizontal stripes", "vertical stripes", "diagonal stripes", "dots", "checks", "grid", "rings"]
FLOORS = [(28, 31, 40), (34, 42, 36), (42, 32, 46), (20, 36, 52), (54, 54, 58), (60, 46, 36),
          (200, 202, 196), (222, 212, 188), (176, 192, 204)]
N_RANGE = {"swap": (5, 8), "infect": (5, 8), "posts": (5, 7), "holes": (5, 8), "rails": (5, 8)}
COLL_RANGE = (3, 12)
TIMER_BOX = (14, 14, 196, 70)
TIMER_FONT = 36
BOX_TOP = 88                        # the box never reaches up to the timer panel

START_PAD = 0.55                    # no event in the first START_PAD s
END_PAD = 0.55                      # nor in the last END_PAD s before the last frame
AFTER_PAD = 0.3                     # nor in the AFTER_PAD s after the clip
GAP_BODY = 0.3                      # two collisions of one shape at least this far apart
GAP_ANY = 0.15                      # any two events (walls, posts too) of one shape
GAP_GLOBAL = 0.2                    # two collisions / captures (any shapes) at least this far apart
NEAR = 15                           # px: rings closer than this without touching = a near miss
MIN_NORMAL = 0.2                    # cos(angle) between closing velocity and the contact normal
MIN_WALL = 0.3                      # |normal speed| / speed for a wall / post bounce to be counted in a question
HOLE_DEEP = 0.8                     # a falling shape's straight path passes within this * hole radius
HOLE_CLEAR = 0.5                    # a shape that doesn't fall keeps its centre >= R + this * r away
EPS = 1e-9
MIN_SPEED = 40                      # px/s: every shape is visibly moving at the start
SIM_T = MAX_FRAMES / FPS + AFTER_PAD + 0.1   # physics horizon of a clip (s)
LONG_T = 40.0                       # a long run is cut into a clean clip window


# ---------------------------------------------------------------- small helpers

def label(b: dict) -> str:
    return f"the {b['color']} {b['shape']}"


def frame_of(t: float) -> int:
    """First frame that shows the result of an event at time t."""
    return int(math.ceil(t * FPS - 1e-7))


def tenths(f: int) -> int:
    """Clock reading (tenths of a second) on frame f."""
    return (10 * f) // FPS


def clock_text(k: int) -> str:
    return f"{k // 10}.{k % 10} s"


def frames_showing(k: int) -> tuple[int, int]:
    """Inclusive frame range whose clock reads k tenths."""
    lo = -(-k * FPS // 10)
    hi = -(-(k + 1) * FPS // 10) - 1
    return lo, hi


def t_last(scene) -> float:
    return (scene["n_frames"] - 1) / FPS


def _lum(c) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def second_colour(rgb) -> tuple:
    """The pattern's second colour: pale for dark colours, deep for light ones."""
    if _lum(rgb) < 165:
        return tuple(int(round(c + (255 - c) * 0.66)) for c in rgb)
    return tuple(int(round(c * 0.42)) for c in rgb)


# ---------------------------------------------------------------- exact event-driven physics

def _first_contact(dx, dy, ux, uy, rs):
    """Smallest s >= 0 with |d + u s| = rs while closing; inf if none."""
    b = dx * ux + dy * uy
    if b >= 0:
        return math.inf
    a = ux * ux + uy * uy
    c = dx * dx + dy * dy - rs * rs
    disc = b * b - a * c
    if disc < 0 or a < EPS:
        return math.inf
    return max(0.0, (-b - math.sqrt(disc)) / a)


def _wall_time(p, v, lo, hi):
    if v > EPS:
        return (hi - p) / v
    if v < -EPS:
        return (lo - p) / v
    return math.inf


def bounce(pi, vi, mi, pj, vj, mj):
    """Elastic disc-disc impulse along the line of centres."""
    dx, dy = pj[0] - pi[0], pj[1] - pi[1]
    d = math.hypot(dx, dy)
    nx, ny = dx / d, dy / d
    a = vi[0] * nx + vi[1] * ny
    b = vj[0] * nx + vj[1] * ny
    a2 = (a * (mi - mj) + 2 * mj * b) / (mi + mj)
    b2 = (b * (mj - mi) + 2 * mi * a) / (mi + mj)
    return ([vi[0] + (a2 - a) * nx, vi[1] + (a2 - a) * ny], [vj[0] + (b2 - b) * nx, vj[1] + (b2 - b) * ny])


def world_of(scene, drop=None, no_posts=False, no_holes=False) -> dict:
    """The physical world of a scene (optionally with shape `drop` removed,
    the posts or the holes taken away) as a hashable-by-json dict."""
    keep = [i for i in range(len(scene["bodies"])) if i != drop]
    return {"bodies": [[scene["bodies"][i][k] for k in ("x", "y", "vx", "vy", "r")] for i in keep],
            "ids": keep, "box": scene["box"],
            "posts": [] if no_posts else [[p["x"], p["y"], p["r"]] for p in scene["posts"]],
            "holes": [] if no_holes else [[h["x"], h["y"], h["r"]] for h in scene["holes"]],
            "T": scene.get("sim_T", SIM_T)}


def _wkey(world) -> str:
    return json.dumps(world, sort_keys=True)


@functools.lru_cache(maxsize=256)
def _simulate(key: str):
    """Exact motion. Returns (knots, events, nears) with body indices in the
    world's own numbering.
      knots  [(t, positions, velocities, alive)]
      events [(t, kind, i, k, quality)]  kind: 'c' pair (k = other body),
             'w' wall (k = axis), 'p' post (k = post), 'h' hole (k = hole);
             quality = cos of the approach for c/p, |v_n|/|v| for w,
             path-to-centre distance / R for h
      nears  [(t, kind, i, k, gap)] closest approaches that come near
    """
    world = json.loads(key)
    bs = world["bodies"]
    n, Tend = len(bs), world["T"]
    p = [[b[0], b[1]] for b in bs]
    v = [[b[2], b[3]] for b in bs]
    r = [b[4] for b in bs]
    m = [x * x for x in r]
    alive = [True] * n
    L, Tp, R, B = world["box"]
    posts, holes = world["posts"], world["holes"]
    t = 0.0
    knots = [(0.0, [q[:] for q in p], [q[:] for q in v], alive[:])]
    events, nears = [], []
    for _ in range(4000):
        best, ev = Tend - t, None
        for i in range(n):
            if not alive[i]:
                continue
            for ax, lo, hi in ((0, L + r[i], R - r[i]), (1, Tp + r[i], B - r[i])):
                dt = _wall_time(p[i][ax], v[i][ax], lo, hi)
                if dt < best:
                    best, ev = dt, ("w", i, ax)
            for k, (cx, cy, cr) in enumerate(posts):
                dt = _first_contact(cx - p[i][0], cy - p[i][1], -v[i][0], -v[i][1], r[i] + cr)
                if dt < best:
                    best, ev = dt, ("p", i, k)
            for k, (cx, cy, cr) in enumerate(holes):
                dt = _first_contact(cx - p[i][0], cy - p[i][1], -v[i][0], -v[i][1], cr)
                if dt < best:
                    best, ev = dt, ("h", i, k)
            for j in range(i + 1, n):
                if not alive[j]:
                    continue
                dt = _first_contact(p[j][0] - p[i][0], p[j][1] - p[i][1], v[j][0] - v[i][0], v[j][1] - v[i][1],
                                    r[i] + r[j])
                if dt < best:
                    best, ev = dt, ("c", i, j)
        _segment_nears(t, best, p, v, r, alive, posts, holes, nears)
        t += best
        for i in range(n):
            if alive[i]:
                p[i][0] += v[i][0] * best
                p[i][1] += v[i][1] * best
        if ev is None:
            break
        kind, i, k = ev
        sp = math.hypot(*v[i])
        if kind == "w":
            q = abs(v[i][k]) / sp
            v[i][k] = -v[i][k]
        elif kind == "p":
            cx, cy, _ = posts[k]
            dx, dy = p[i][0] - cx, p[i][1] - cy
            d = math.hypot(dx, dy)
            nx, ny = dx / d, dy / d
            vn = v[i][0] * nx + v[i][1] * ny
            q = -vn / sp
            v[i] = [v[i][0] - 2 * vn * nx, v[i][1] - 2 * vn * ny]
        elif kind == "h":
            cx, cy, cr = holes[k]
            ux, uy = v[i][0] / sp, v[i][1] / sp
            q = abs((cx - p[i][0]) * uy - (cy - p[i][1]) * ux) / cr
            alive[i] = False
            v[i] = [0.0, 0.0]
        else:
            j = k
            dx, dy = p[j][0] - p[i][0], p[j][1] - p[i][1]
            ux, uy = v[j][0] - v[i][0], v[j][1] - v[i][1]
            q = -(dx * ux + dy * uy) / (math.hypot(dx, dy) * math.hypot(ux, uy))
            v[i], v[j] = bounce(p[i], v[i], m[i], p[j], v[j], m[j])
        events.append((t, kind, i, k, q))
        knots.append((t, [q_[:] for q_ in p], [w[:] for w in v], alive[:]))
    else:
        events.append((t, "overflow", 0, 0, 0.0))
    return knots, events, nears


def _segment_nears(t0, dt, p, v, r, alive, posts, holes, out):
    """Closest approaches inside one straight-line segment [t0, t0 + dt]."""
    n = len(p)
    for i in range(n):
        if not alive[i]:
            continue
        objs = [(p[j][0] - p[i][0], p[j][1] - p[i][1], v[j][0] - v[i][0], v[j][1] - v[i][1], r[i] + r[j], "c", j)
                for j in range(i + 1, n) if alive[j]]
        objs += [(cx - p[i][0], cy - p[i][1], -v[i][0], -v[i][1], r[i] + cr, "p", k)
                 for k, (cx, cy, cr) in enumerate(posts)]
        objs += [(cx - p[i][0], cy - p[i][1], -v[i][0], -v[i][1], cr + HOLE_CLEAR * r[i], "h", k)
                 for k, (cx, cy, cr) in enumerate(holes)]
        for dx, dy, ux, uy, rs, kind, k in objs:
            a = ux * ux + uy * uy
            s = 0.0 if a < EPS else min(dt, max(0.0, -(dx * ux + dy * uy) / a))
            g = math.hypot(dx + ux * s, dy + uy * s) - rs
            lim = NEAR if kind != "h" else 0.0
            if g < lim:
                out.append((t0 + s, kind, i, k, g))


def _run(scene, **kw):
    world = world_of(scene, **kw)
    knots, events, nears = _simulate(_wkey(world))
    ids = world["ids"]
    ev = [(t, kind, ids[i], ids[k] if kind == "c" else k, q) for t, kind, i, k, q in events]
    nr = [(t, kind, ids[i], ids[k] if kind == "c" else k, g) for t, kind, i, k, g in nears]
    return knots, ev, nr, ids


# ---------------------------------------------------------------- event views (generator side)

def events(scene, **kw) -> list:
    """All events up to the end of the clip (+AFTER_PAD), original body ids."""
    lim = t_last(scene) + AFTER_PAD
    return [e for e in _run(scene, **kw)[1] if e[0] <= lim]


def shown(scene, **kw) -> list:
    """Events that happen inside the clip."""
    tl = t_last(scene)
    return [e for e in _run(scene, **kw)[1] if e[0] <= tl]


def collisions(scene, **kw) -> list:
    """[(t, i, j)] shape-shape collisions inside the clip, i < j."""
    return [(t, min(i, k), max(i, k)) for t, kind, i, k, _ in shown(scene, **kw) if kind == "c"]


def wall_hits(scene) -> list:
    return [(t, i) for t, kind, i, k, _ in shown(scene) if kind == "w"]


def post_hits(scene) -> list:
    return [(t, i, k) for t, kind, i, k, _ in shown(scene) if kind == "p"]


def captures(scene, **kw) -> list:
    return [(t, i, k) for t, kind, i, k, _ in shown(scene, **kw) if kind == "h"]


def positions(scene, t: float) -> list:
    knots = _run(scene)[0]
    k = max(0, bisect.bisect_right([kt for kt, *_ in knots], t) - 1)
    t0, p, v, _ = knots[k]
    dt = t - t0
    return [(q[0] + w[0] * dt, q[1] + w[1] * dt) for q, w in zip(p, v)]


def alive_at(scene, t: float) -> list:
    out = [True] * len(scene["bodies"])
    for tc, i, _ in captures(scene):
        if tc <= t:
            out[i] = False
    return out


def apply_rule(scene, state: list, i: int, j: int) -> None:
    """Pattern change at a collision of i and j (in place)."""
    if scene["rule"] == "swap":
        state[i], state[j] = state[j], state[i]
    else:
        tag = scene["tag"]
        if (state[i] == tag) != (state[j] == tag):
            state[i] = state[j] = tag


def start_patterns(scene) -> list:
    return [b["pattern"] for b in scene["bodies"]]


def replay(scene, skip=None, upto=None, start=None, coll=None) -> list:
    """Patterns after the collisions (index < upto, not index `skip`)."""
    pat = list(start or start_patterns(scene))
    for k, (_, i, j) in enumerate(coll if coll is not None else collisions(scene)):
        if upto is not None and k >= upto:
            break
        if k != skip:
            apply_rule(scene, pat, i, j)
    return pat


def patterns_at(scene, t: float) -> list:
    coll = collisions(scene)
    return replay(scene, upto=sum(1 for c in coll if c[0] <= t))


# ---------------------------------------------------------------- cleanliness

def problems(evs, nears) -> list[tuple[float, str]]:
    """Unclean moments of a run, as (time, message), in time order."""
    out = []
    if any(e[1] == "overflow" for e in evs):
        return [(0.0, "event overflow")]
    for t, kind, i, k, q in evs:
        if kind == "c" and q < MIN_NORMAL:
            out.append((t, f"glancing contact at {t:.2f}s"))
        if kind == "h" and q > HOLE_DEEP:
            out.append((t, f"shape grazes a hole at {t:.2f}s"))
    last, last_c = {}, {}
    for t, kind, i, k, _ in evs:
        for x in ((i, k) if kind == "c" else (i,)):
            if t - last.get(x, -9) < GAP_ANY:
                out.append((t, f"one shape has two events {t - last[x]:.2f}s apart"))
            if kind in "ch" and t - last_c.get(x, -9) < GAP_BODY:
                out.append((t, f"one shape collides twice {t - last_c[x]:.2f}s apart"))
            last[x] = t
            if kind in "ch":
                last_c[x] = t
    ts = [e[0] for e in evs if e[1] in "ch"]
    out += [(b, f"two collisions {b - a:.2f}s apart") for a, b in zip(ts, ts[1:]) if b - a < GAP_GLOBAL]
    contact = {}
    for t, kind, i, k, _ in evs:
        if kind in "cph":
            contact.setdefault((kind, i, k), []).append(t)
    for t, kind, i, k, g in nears:
        if kind == "h":
            if not contact.get(("h", i, k)):
                out.append((t, f"shape passes close to a hole at {t:.2f}s"))
        elif not any(abs(t - tc) < 0.35 for tc in contact.get((kind, i, k), [])):
            out.append((t, f"near miss ({kind}) at {t:.2f}s"))
    return sorted(out)


def event_errors(scene, evs, nears, tl) -> list[str]:
    """Rule breaks among events up to tl + AFTER_PAD."""
    lim = tl + AFTER_PAD
    errs = [m for t, m in problems(evs, nears) if t <= lim]
    for t, kind, i, k, q in evs:
        if t < START_PAD:
            errs.append(f"{kind} event at {t:.2f}s, too near the start")
        if kind in "ch" and tl - END_PAD < t <= lim:
            errs.append(f"{kind} event at {t:.2f}s, too near the end")
    return errs


def scenario_errors(scene, coll=None, caps=None) -> list[str]:
    errs = []
    coll = collisions(scene) if coll is None else coll
    n = len(scene["bodies"])
    if not COLL_RANGE[0] <= len(coll) <= COLL_RANGE[1]:
        errs.append(f"{len(coll)} collisions")
    sc = scene["scenario"]
    if sc == "infect":
        st = [b["pattern"] for b in scene["bodies"]]
        k = 0
        for _, i, j in coll:
            before = st.count(scene["tag"])
            apply_rule(scene, st, i, j)
            k += st.count(scene["tag"]) - before
        if k < 2:
            errs.append("fewer than 2 tags passed on")
    if sc == "posts" and len(post_hits(scene)) < 2:
        errs.append("fewer than 2 post bounces")
    if sc == "holes":
        c = len(captures(scene) if caps is None else caps)
        if not 1 <= c <= n - 3:
            errs.append(f"{c} shapes fall into holes")
    return errs


def check(scene) -> list[str]:
    errs = []
    bs = scene["bodies"]
    n = len(bs)
    sc = scene.get("scenario")
    if sc not in SCENARIOS:
        return [f"unknown scenario {sc}"]
    if not N_RANGE[sc][0] <= n <= N_RANGE[sc][1]:
        errs.append(f"{n} shapes")
    if len({b["color"] for b in bs}) != n:
        errs.append("colours repeat")
    if any(b["shape"] not in SHAPES or b["color"] not in COLORS or b["pattern"] not in PATTERNS for b in bs):
        errs.append("unknown shape, colour or pattern")
    if scene["rule"] == "swap" and len({b["pattern"] for b in bs}) != n:
        errs.append("patterns repeat")
    if scene["rule"] == "infect" and sorted(b["pattern"] for b in bs) != sorted(["solid"] * (n - 1) + [scene["tag"]]):
        errs.append("infect scene needs one tagged shape")
    if not MIN_FRAMES <= scene["n_frames"] <= MAX_FRAMES:
        errs.append(f"{scene['n_frames']} frames")
    if any(math.hypot(b["vx"], b["vy"]) < MIN_SPEED for b in bs):
        errs.append("a shape is (almost) at rest")
    if any(b["r"] < 28 for b in bs):
        errs.append("shape too small")
    L, T, R, B = scene["box"]
    if T < BOX_TOP or L < 8 or R > W - 8 or B > H - 8:
        errs.append("box outside its frame area")
    floor = scene["style"]["floor"]
    for b in bs:
        if sum(abs(a - c) for a, c in zip(COLORS[b["color"]], floor)) < 200:
            errs.append(f"{b['color']} too close to the floor colour")
    tl = t_last(scene)
    _, evs, nears, _ = _run(scene)
    errs += event_errors(scene, evs, nears, tl)
    errs += scenario_errors(scene)
    if sc == "rails":
        for b in bs:
            if b["vy"] != 0 or b["y"] not in scene["lanes"]:
                errs.append("a rail shape leaves its rail")
    return errs


# ---------------------------------------------------------------- generation

def _style(rng: random.Random) -> dict:
    floor = rng.choice(FLOORS)
    light = _lum(floor) > 128
    tex = rng.choice(["plain", "grid", "dots", "tiles", "planks"])
    shade = (-10 if light else 10) * rng.choice([1, 1.4])
    bgs = [((14, 16, 22), (40, 44, 56)), ((30, 22, 18), (70, 52, 40)), ((12, 26, 30), (34, 64, 70)),
           ((235, 236, 240), (196, 200, 210)), ((60, 64, 74), (20, 22, 28)), ((44, 30, 56), (16, 12, 24))]
    return {"floor": list(floor), "floor_tex": tex, "tex_step": rng.choice([36, 48, 60, 72]),
            "tex_shade": round(shade), "bg": [list(c) for c in rng.choice(bgs)],
            "wall": list(rng.choice([(150, 156, 172), (196, 170, 120), (120, 130, 150), (180, 92, 70),
                                     (90, 150, 140), (225, 225, 230)])),
            "wall_w": rng.randint(5, 12),
            "ring": list((205, 208, 218) if not light else (70, 74, 86)),
            "ring_style": rng.choice(["solid", "dashed"]),
            "edge_w": rng.choice([3, 4]),
            "post": list(rng.choice([(118, 118, 128), (140, 120, 100), (100, 120, 136)])),
            "hole_rim": list(rng.choice([(150, 150, 160), (176, 150, 96), (110, 140, 150)])),
            "rail": list((92, 98, 112) if not light else (150, 150, 158))}


def _box(rng: random.Random, sc: str) -> list:
    w = rng.randint(860, 1240)
    h = rng.randint(470, 610) if sc != "rails" else rng.randint(500, 610)
    x0 = rng.randint(max(16, (W - w) // 2 - 120), min(W - 16 - w, (W - w) // 2 + 120))
    y0 = rng.randint(BOX_TOP + 6, max(BOX_TOP + 6, H - 14 - h))
    return [x0, y0, x0 + w, y0 + h]


def _place(rng, r, box, margin, others):
    L, T, R, B = box
    for _ in range(300):
        x = rng.uniform(L + margin, R - margin)
        y = rng.uniform(T + margin, B - margin)
        if all(math.hypot(x - ox, y - oy) >= r + orr + 90 for ox, oy, orr in others):
            return round(x, 2), round(y, 2)
    return None


def _first_wall(x, y, vx, vy, r, box):
    L, T, R, B = box
    return min(_wall_time(x, vx, L + r, R - r), _wall_time(y, vy, T + r, B - r))


def _bodies(rng: random.Random, sc: str, box, n, posts, holes, lanes):
    L, T, R, B = box
    zoom = min((R - L) / 1200, (B - T) / 600)
    shapes = [rng.choice(SHAPES) for _ in range(n)]
    speed = rng.uniform(80, 190) * (0.85 + 0.15 * zoom)
    rmin, rmax = max(28, round(32 * zoom)), max(34, round(58 * zoom))
    if sc == "rails":
        rmax = min(rmax, int((lanes[1] - lanes[0] - 46) / 2) if len(lanes) > 1 else rmax)
    others = [(p["x"], p["y"], p["r"]) for p in posts] + [(h["x"], h["y"], h["r"]) for h in holes]
    bodies = []
    lane_of = []
    if sc == "rails":
        # two shapes per rail, the rest spread over rails that have room for a third
        per = [2] * len(lanes)
        for li in rng.sample(range(len(lanes)), n - 2 * len(lanes)):
            per[li] += 1
        lane_of = [li for li in range(len(lanes)) for _ in range(per[li])]
        rng.shuffle(lane_of)
    for k in range(n):
        r = rng.randint(rmin, rmax)
        s = speed * rng.uniform(0.75, 1.25)
        for _ in range(60):
            if sc == "rails":
                y = lanes[lane_of[k]]
                mates = [(b["x"], b["r"]) for b, li in zip(bodies, lane_of) if li == lane_of[k]]
                x = round(rng.uniform(L + r + 30, R - r - 30), 2)
                if any(abs(x - mx) < r + mr + 70 for mx, mr in mates):
                    continue
                vx, vy = round(s * rng.choice([-1, 1]), 3), 0.0
            else:
                pos = _place(rng, r, box, r + 16, others + [(b["x"], b["y"], b["r"]) for b in bodies])
                if pos is None:
                    return None
                x, y = pos
                # keep clear of the axes so wall bounces are never shallow at first
                a = math.radians(rng.choice([0, 90, 180, 270]) + rng.uniform(15, 75))
                vx, vy = round(s * math.cos(a), 3), round(s * math.sin(a), 3)
            soon = START_PAD + 0.15
            if _first_wall(x, y, vx, vy, r, box) <= soon:
                continue
            if any(_first_contact(b["x"] - x, b["y"] - y, b["vx"] - vx, b["vy"] - vy, r + b["r"]) <= soon
                   for b in bodies):
                continue
            if any(_first_contact(o["x"] - x, o["y"] - y, -vx, -vy, r + o["r"]) <= soon for o in posts) or \
                    any(_first_contact(h["x"] - x, h["y"] - y, -vx, -vy, h["r"]) <= soon for h in holes):
                continue
            break
        else:
            return None
        bodies.append({"shape": shapes[k], "color": None, "pattern": None, "r": r, "x": x, "y": y,
                       "vx": vx, "vy": vy})
    return bodies


def _fixtures(rng, sc, box, n):
    L, T, R, B = box
    posts, holes, lanes = [], [], []
    if sc == "posts":
        for _ in range(rng.randint(2, 5)):
            r = rng.randint(22, 40)
            for _ in range(200):
                x, y = rng.uniform(L + 150, R - 150), rng.uniform(T + 140, B - 140)
                if all(math.hypot(x - p["x"], y - p["y"]) > r + p["r"] + 170 for p in posts):
                    posts.append({"x": round(x, 2), "y": round(y, 2), "r": r})
                    break
    if sc == "holes":
        for _ in range(rng.randint(2, 3)):
            r = rng.randint(28, 40)
            for _ in range(200):
                x, y = rng.uniform(L + 110, R - 110), rng.uniform(T + 100, B - 100)
                if all(math.hypot(x - h["x"], y - h["y"]) > r + h["r"] + 190 for h in holes):
                    holes.append({"x": round(x, 2), "y": round(y, 2), "r": r})
                    break
    if sc == "rails":
        k = rng.choice([k for k in (2, 3, 4) if 2 * k <= n <= 3 * k])
        gap = (B - T) / k
        lanes = [round(T + gap * (i + 0.5), 2) for i in range(k)]
    return posts, holes, lanes


def _state_at(knots, t):
    k = max(0, bisect.bisect_right([kt for kt, *_ in knots], t) - 1)
    t0, p, v, alive = knots[k]
    return [(q[0] + w[0] * (t - t0), q[1] + w[1] * (t - t0), w[0], w[1]) for q, w in zip(p, v)], alive


def _windows(evs, probs):
    """(start, n_frames) of clean windows in a long run, longest first."""
    P = [t for t, _ in probs]
    C = [e[0] for e in evs if e[1] in "ch"]
    Cc = [e[0] for e in evs if e[1] == "c"]
    E = [e[0] for e in evs]
    out = []
    a = 0.0
    while a < LONG_T - MIN_FRAMES / FPS - 1:
        a = round(a + 0.05, 2)
        if bisect.bisect_left(E, a + START_PAD) != bisect.bisect_left(E, a):
            continue
        pk = bisect.bisect_left(P, a)
        p_next = P[pk] if pk < len(P) else LONG_T
        top = min(MAX_FRAMES, int((p_next - AFTER_PAD - a) * FPS) + 1, int((LONG_T - 1 - a) * FPS))
        for nf in range(top, MIN_FRAMES - 1, -1):
            tl = a + (nf - 1) / FPS
            if bisect.bisect_right(C, tl + AFTER_PAD) != bisect.bisect_right(C, tl - END_PAD):
                continue
            nc = bisect.bisect_right(Cc, tl) - bisect.bisect_left(Cc, a)
            if COLL_RANGE[0] <= nc <= COLL_RANGE[1]:
                out.append((nf, a))
                break
    out.sort(key=lambda w: (-w[0], w[1]))
    return [(a, nf) for nf, a in out]


def _attempt(rng: random.Random, sc: str) -> dict | None:
    box = _box(rng, sc)
    style = _style(rng)
    n = rng.randint(*N_RANGE[sc]) + (rng.randint(0, 2) if sc == "holes" else 0)
    posts, holes, lanes = _fixtures(rng, sc, box, n)
    bodies = _bodies(rng, sc, box, n, posts, holes, lanes)
    if bodies is None:
        return None
    floor = style["floor"]
    cols = [c for c in COLORS if sum(abs(a - b) for a, b in zip(COLORS[c], floor)) >= 200]
    longscene = {"box": box, "bodies": bodies, "posts": posts, "holes": holes, "sim_T": LONG_T}
    knots, evs, nears, _ = _run(longscene)
    wins = _windows(evs, problems(evs, nears))
    tag = rng.choice(TAGS) if sc == "infect" else None
    for a, nf in wins[:6]:
        state, alive = _state_at(knots, a)
        keep = [i for i in range(len(bodies)) if alive[i]]
        m = len(keep)
        if not N_RANGE[sc][0] <= m <= N_RANGE[sc][1] or len(cols) < m:
            continue
        if any(math.hypot(state[i][2], state[i][3]) < MIN_SPEED for i in keep):
            continue
        bs = []
        for i, c in zip(keep, rng.sample(cols, m)):
            x, y, vx, vy = state[i]
            bs.append({"shape": bodies[i]["shape"], "color": c, "pattern": None, "r": bodies[i]["r"],
                       "x": round(x, 2), "y": bodies[i]["y"] if sc == "rails" else round(y, 2),
                       "vx": round(vx, 3), "vy": 0.0 if sc == "rails" else round(vy, 3)})
        if sc == "infect":
            for b in bs:
                b["pattern"] = "solid"
            bs[rng.randrange(m)]["pattern"] = tag
        else:
            for b, p in zip(bs, rng.sample(PATTERNS, m)):
                b["pattern"] = p
        scene = {"scenario": sc, "rule": "infect" if sc == "infect" else "swap", "tag": tag,
                 "box": box, "bodies": bs, "posts": posts, "holes": holes, "lanes": lanes,
                 "style": style, "timer": {"box": list(TIMER_BOX), "size": TIMER_FONT}, "labels": [],
                 "n_frames": nf}
        if not check(scene):
            return scene
    return None


def _askable(scene) -> bool:
    """Every ladder level can be filled (probed with a fixed side rng, so the
    main rng stream is untouched)."""
    import questions  # noqa: PLC0415  (questions imports sim)
    try:
        questions.build(json.loads(json.dumps(scene)), random.Random(7))
    except RuntimeError:
        return False
    return True


def build(rng: random.Random) -> dict:
    sc = SCENARIOS[rng.randrange(len(SCENARIOS))]
    for attempt in range(1, 5000):
        scene = _attempt(rng, sc)
        if scene is not None and not check(scene) and _askable(scene):
            scene["attempts"] = attempt
            scene["duration"] = round(scene["n_frames"] / FPS, 6)
            return scene
    raise RuntimeError("no clean scene found")
