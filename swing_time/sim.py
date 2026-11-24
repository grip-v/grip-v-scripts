"""Swing Time (after Jane Street, Aug 2016): a particle crosses a board of
squares by hooking a rope on a post (a grid corner) and swinging around it
until it lands on the centre of another square the same distance from the
post. A swing

  * uses a rope at most `cap` units long (squares are 1 unit),
  * goes the shorter way round (either way on an exact half turn),
  * must keep the particle on the board,
  * may not sweep over another post (a post inside the swept wedge, or on
    its edge, snags the rope),
  * may not land on a hatched pit square (the "pits" scenario),
  * costs its rope length.

The particle always takes the cheapest route from its start square to the
checkered goal square. The generator only keeps boards whose cheapest route
is unique by a clear margin (`MARGIN` units cheaper than any other route),
so the route shown is the only right one.

Scenarios (scene["scenario"]): left_right (cross from the left column to
the right column), bottom_top (bottom row to top row), corner (corner to
opposite corner), dense (many posts, short ropes) and pits (hatched squares
the particle may not land on).

All geometry here is float/angle based; answers.py re-derives legality with
exact integer cross products, so the two solvers are independent. Times are
integer frames at 24 fps; the clock shows frame*10//24 tenths of a second.
Pure function of the rng passed in.
"""
from __future__ import annotations
import heapq
import math
import random

FPS = 24
MAX_FRAMES = 240                    # 10 s
MIN_EDGE = 12                       # >= 0.5 s before the first and after the last event
MIN_SWING = 12                      # a swing lasts >= 0.5 s
MIN_PAUSE = 8                       # rests between swings >= 1/3 s
MARGIN = 0.25                       # cheapest route beats every other route by this many units
SWINGS = (3, 6)
SCENARIOS = ("left_right", "bottom_top", "corner", "dense", "pits")

# board area on screen (the clock sits top-left, left of x = 240)
AREA = (240, 18, 1262, 702)
CLOCK_BOX = (18, 16, 214, 84)

POST_COLOURS = {
    "red": (222, 45, 38), "orange": (247, 135, 25), "yellow": (245, 205, 30), "green": (34, 160, 70),
    "teal": (0, 170, 170), "blue": (35, 90, 225), "purple": (135, 60, 200), "pink": (240, 110, 190),
    "brown": (135, 80, 35), "grey": (135, 135, 140),
}
POST_SHAPES = ("circle", "square", "triangle", "diamond")
THEMES = {
    "light": {"bg": [(226, 222, 212), (210, 218, 226), (232, 226, 236), (214, 226, 214)],
              "board": [((252, 250, 244), (240, 236, 226)), ((248, 250, 252), (232, 238, 244)),
                        ((250, 248, 240), (236, 240, 230))],
              "grid": [(178, 174, 166), (160, 170, 184), (190, 180, 170)],
              "outline": (30, 30, 34), "particle": [((255, 255, 255), (20, 20, 24)), ((20, 20, 24), (255, 255, 255))],
              "rope": [(95, 60, 30), (60, 60, 70), (40, 70, 110)],
              "trail": [(230, 110, 100), (110, 150, 230), (200, 120, 210)],
              "marker": [(150, 60, 130), (30, 110, 140), (95, 115, 35)],
              "start": [(170, 225, 185), (175, 210, 245), (215, 195, 240)],
              "goal": [((246, 206, 90), (190, 140, 30)), ((250, 180, 140), (200, 100, 60))],
              "pit": ((44, 44, 52), (205, 205, 215))},
    "dark": {"bg": [(24, 28, 36), (34, 26, 34), (20, 32, 30), (30, 30, 30)],
             "board": [((52, 60, 74), (44, 52, 66)), ((58, 54, 66), (50, 46, 58)), ((48, 62, 60), (40, 54, 52))],
             "grid": [(110, 120, 138), (128, 118, 138), (104, 130, 126)],
             "outline": (245, 245, 245), "particle": [((255, 255, 255), (20, 20, 24))],
             "rope": [(240, 220, 170), (220, 225, 235), (250, 190, 120)],
             "trail": [(255, 140, 120), (140, 190, 255), (230, 160, 240)],
             "marker": [(150, 230, 220), (190, 160, 250), (175, 225, 120)],
             "start": [(60, 130, 90), (60, 100, 160), (115, 85, 150)],
             "goal": [((214, 170, 40), (120, 90, 20)), ((220, 120, 70), (130, 60, 30))],
             "pit": ((10, 10, 12), (200, 70, 70))},
}


# ------------------------------------------------------------------ geometry (float, angle based)
def centre(sq):
    return sq[0] + 0.5, sq[1] + 0.5


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def sweep_of(src, dst, post, d):
    """Signed sweep angle (radians) for a swing src -> dst about post in direction d (+1 ccw, -1 cw)."""
    c, q = centre(src), centre(dst)
    a0 = math.atan2(c[1] - post[1], c[0] - post[0])
    a1 = math.atan2(q[1] - post[1], q[0] - post[0])
    return d * ((d * (a1 - a0)) % (2 * math.pi))


def _clear(src, post, sweep, posts, w, h):
    c = centre(src)
    r = math.dist(c, post)
    a0 = math.atan2(c[1] - post[1], c[0] - post[0])
    span = abs(sweep)
    d = 1 if sweep > 0 else -1

    def within(a):
        rel = (d * (a - a0)) % (2 * math.pi)
        return rel <= span + 1e-9 or rel >= 2 * math.pi - 1e-9

    for k in range(4):                                   # the arc's extreme points must stay on the board
        a = k * math.pi / 2
        if within(a):
            x, y = post[0] + r * math.cos(a), post[1] + r * math.sin(a)
            if x < -1e-9 or x > w + 1e-9 or y < -1e-9 or y > h + 1e-9:
                return False
    for o in posts:
        if o == post:
            continue
        if math.dist(o, post) > r + 1e-9:
            continue
        if within(math.atan2(o[1] - post[1], o[0] - post[0])):
            return False
    return True


def swing_dir(src, dst, post, posts, w, h):
    """+1 / -1 for a legal counter-clockwise / clockwise swing, 0 if illegal (rope length not checked)."""
    c, q = centre(src), centre(dst)
    a0 = math.atan2(c[1] - post[1], c[0] - post[0])
    a1 = math.atan2(q[1] - post[1], q[0] - post[0])
    s = _wrap(a1 - a0)
    dirs = (1, -1) if abs(abs(s) - math.pi) < 1e-9 else ((1,) if s > 0 else (-1,))
    for d in dirs:
        if _clear(src, post, sweep_of(src, dst, post, d), posts, w, h):
            return d
    return 0


def graph(world):
    """{square: [(dst, post, rope, dir)]} of every legal swing."""
    w, h, cap = world["w"], world["h"], world["cap"]
    posts = [tuple(p) for p in world["posts"]]
    pits = {tuple(p) for p in world["pits"]}
    squares = [(x, y) for x in range(w) for y in range(h) if (x, y) not in pits]
    out = {sq: [] for sq in squares}
    for p in posts:
        rings = {}
        for sq in squares:
            d2 = (2 * sq[0] + 1 - 2 * p[0]) ** 2 + (2 * sq[1] + 1 - 2 * p[1]) ** 2
            if d2 <= 4 * cap * cap:
                rings.setdefault(d2, []).append(sq)
        for d2, sqs in rings.items():
            r = math.sqrt(d2) / 2
            for a in sqs:
                for b in sqs:
                    if a != b:
                        d = swing_dir(a, b, p, posts, w, h)
                        if d:
                            out[a].append((b, p, r, d))
    return out


def _dijkstra(adj, src):
    dist, prev, heap = {src: 0.0}, {}, [(0.0, src)]
    while heap:
        c, u = heapq.heappop(heap)
        if c > dist[u] + 1e-12:
            continue
        for v, p, r, d in adj.get(u, ()):
            if c + r < dist.get(v, math.inf) - 1e-12:
                dist[v] = c + r
                prev[v] = (u, p, r, d)
                heapq.heappush(heap, (c + r, v))
    return dist, prev


def solve(world):
    """{"cost", "path": [(post, land, rope, dir)], "second"} or None if the goal can't be reached.
    `second` is the cost of the cheapest route that differs from `path` in at least one swing."""
    adj = graph(world)
    s, g = tuple(world["start"]), tuple(world["goal"])
    if s not in adj or g not in adj:
        return None
    dist, prev = _dijkstra(adj, s)
    if g not in dist:
        return None
    path, v = [], g
    while v != s:
        u, p, r, d = prev[v]
        path.append((u, v, p, r, d))
        v = u
    path.reverse()
    on = {(u, v, p) for u, v, p, _, _ in path}
    radj = {}
    for u, es in adj.items():
        for v, p, r, d in es:
            radj.setdefault(v, []).append((u, p, r, d))
    back, _ = _dijkstra(radj, g)
    second = math.inf
    for u, es in adj.items():
        if u not in dist:
            continue
        for v, p, r, d in es:
            if (u, v, p) not in on and v in back:
                second = min(second, dist[u] + r + back[v])
    return {"cost": dist[g], "second": second,
            "path": [(p, v, r, d) for u, v, p, r, d in path]}


def world_of(scene, drop=None, cap=None, pits=True):
    return {"w": scene["w"], "h": scene["h"], "cap": scene["cap"] if cap is None else cap,
            "posts": [p["xy"] for p in scene["posts"] if p["colour"] != drop],
            "pits": scene["pits"] if pits else [], "start": scene["start"], "goal": scene["goal"]}


def rope_values(limit=6.0):
    """Every rope length a swing can have (square centre to grid corner), ascending."""
    out = set()
    for a in range(1, 14, 2):
        for b in range(a, 14, 2):
            r = math.sqrt(a * a + b * b) / 2
            if r <= limit:
                out.add(round(r, 9))
    return sorted(out)


# ------------------------------------------------------------------ timing
def events(scene):
    """[(start_frame, land_frame)] per swing: the rope appears at start_frame and the particle
    lands (rope gone) at land_frame."""
    t = scene["timing"]
    f, out = t["intro"], []
    for k, dur in enumerate(t["dur"]):
        out.append((f, f + dur))
        f += dur + t["pause"]
    return out


def n_frames(scene):
    t = scene["timing"]
    return t["intro"] + sum(t["dur"]) + t["pause"] * (len(t["dur"]) - 1) + t["outro"]


def clock_tenths(frame: int) -> int:
    return frame * 10 // FPS


# ------------------------------------------------------------------ generation
def _layout(rng, scenario):
    if scenario == "bottom_top":
        w, h = rng.randint(5, 9), rng.randint(6, 8)
    elif scenario == "dense":
        w, h = rng.randint(7, 10), rng.randint(5, 7)
    else:
        w, h = rng.randint(7, 12), rng.randint(5, 8)
    n_posts = rng.randint(8, 10) if scenario == "dense" else rng.randint(5, 9)
    cap = 3.2 if scenario == "dense" else rng.choice((4.0, 4.6, 5.0))
    pts = set()
    while len(pts) < n_posts:
        pts.add((rng.randint(1, w - 1), rng.randint(1, h - 1)))
    posts = sorted(pts)
    rng.shuffle(posts)
    if scenario == "bottom_top":
        start, goal = (rng.randrange(w), 0), (rng.randrange(w), h - 1)
    elif scenario == "corner":
        cx, cy = rng.choice(((0, 0), (1, 0), (0, 1), (1, 1)))
        start = (cx * (w - 1), cy * (h - 1))
        goal = ((1 - cx) * (w - 1), (1 - cy) * (h - 1))
    else:
        start, goal = (0, rng.randrange(h)), (w - 1, rng.randrange(h))
    if rng.random() < 0.5 and scenario != "corner":       # mirror: right-to-left / top-to-bottom
        start, goal = goal, start
    pits = []
    if scenario == "pits":
        cand = [(x, y) for x in range(w) for y in range(h) if (x, y) not in (start, goal)]
        pits = sorted(rng.sample(cand, rng.randint(3, 8)))
    return {"w": w, "h": h, "cap": cap, "posts": [list(p) for p in posts], "pits": [list(p) for p in pits],
            "start": list(start), "goal": list(goal)}


def _style(rng, w, h, n_posts):
    theme = rng.choice(("light", "light", "dark"))
    T = THEMES[theme]
    hi = min(88.0, (AREA[2] - AREA[0]) / w, (AREA[3] - AREA[1]) / h)
    cell = int(rng.uniform(max(54.0, 0.78 * hi), hi))
    x0 = AREA[0] + rng.randint(0, int(AREA[2] - AREA[0] - w * cell))
    y0 = AREA[1] + rng.randint(0, int(AREA[3] - AREA[1] - h * cell))
    colours = rng.sample(sorted(POST_COLOURS), n_posts)
    shapes = [rng.choice(POST_SHAPES) for _ in range(n_posts)]
    pf, prim = rng.choice(T["particle"])
    ppx = rng.randint(26, 34)
    return {
        "theme": theme, "bg": list(rng.choice(T["bg"])), "board": [list(c) for c in rng.choice(T["board"])],
        "board_pat": rng.choice(("plain", "checker")), "grid": list(rng.choice(T["grid"])),
        "grid_style": rng.choice(("solid", "dashed", "dotted")), "cell": cell, "x0": x0, "y0": y0,
        "post_px": rng.randint(24, max(24, min(32, int(0.42 * cell)))), "colours": colours, "shapes": shapes,
        "particle": {"fill": list(pf), "rim": list(prim), "pattern": rng.choice(("solid", "bullseye", "ring")),
                     "px": ppx},
        "rope": {"col": list(rng.choice(T["rope"])), "w": rng.randint(4, 6),
                 "style": rng.choice(("solid", "dashed", "beaded"))},
        "trail": {"col": list(rng.choice(T["trail"])), "w": rng.randint(3, 5), "style": rng.choice(("solid", "dotted"))},
        "marker": {"col": list(rng.choice(T["marker"])), "kind": rng.choice(("ring", "box", "diamond")),
                   "px": ppx + rng.randint(12, 18)},
        "start": list(rng.choice(T["start"])), "goal": [list(c) for c in rng.choice(T["goal"])],
        "pit": [list(c) for c in T["pit"]], "outline": list(T["outline"]),
    }


def _timing(rng, arcs):
    n = len(arcs)
    intro, outro = rng.randint(MIN_EDGE + 2, 22), rng.randint(MIN_EDGE + 2, 24)
    pause = rng.randint(MIN_PAUSE, 11)
    base = rng.randint(MIN_SWING, 15)
    budget = MAX_FRAMES - intro - outro - pause * (n - 1) - base * n
    if budget < 0:
        return None
    fpu = min(rng.uniform(1.6, 4.2), budget / max(sum(arcs), 1e-9))
    dur = [max(MIN_SWING, base + int(a * fpu)) for a in arcs]
    t = {"intro": intro, "pause": pause, "outro": outro, "dur": dur}
    if intro + sum(dur) + pause * (n - 1) + outro > MAX_FRAMES:
        return None
    return t


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    for attempt in range(1, 1000):
        world = _layout(rng, scenario)
        res = solve(world)
        if res is None or not SWINGS[0] <= len(res["path"]) <= SWINGS[1]:
            continue
        if res["second"] - res["cost"] < MARGIN:
            continue
        here, arcs = tuple(world["start"]), []
        for p, land, r, d in res["path"]:
            arcs.append(r * abs(sweep_of(here, land, p, d)))
            here = land
        timing = _timing(rng, arcs)
        if timing is None:
            continue
        st = _style(rng, world["w"], world["h"], len(world["posts"]))
        name = {tuple(xy): st["colours"][i] for i, xy in enumerate(world["posts"])}
        scene = {
            "scenario": scenario, "w": world["w"], "h": world["h"], "cap": world["cap"],
            "posts": [{"colour": st["colours"][i], "shape": st["shapes"][i], "xy": xy}
                      for i, xy in enumerate(world["posts"])],
            "pits": world["pits"], "start": world["start"], "goal": world["goal"],
            "swings": [{"post": name[tuple(p)], "land": list(land), "dir": d} for p, land, r, d in res["path"]],
            "cost": round(res["cost"], 6), "margin": round(min(res["second"] - res["cost"], 99.0), 6),
            "timing": timing, "style": {k: v for k, v in st.items() if k not in ("colours", "shapes")},
            "build_attempts": attempt,
        }
        return scene
    raise RuntimeError("no valid board found")


# ------------------------------------------------------------------ consistency check
def check(scene: dict) -> list[str]:
    errs = []
    res = solve(world_of(scene))
    if res is None:
        return ["the goal can't be reached"]
    if abs(res["cost"] - scene["cost"]) > 1e-4:
        errs.append("stored cost isn't the cheapest route")
    if res["second"] - res["cost"] < MARGIN - 1e-9:
        errs.append(f"another route is within {MARGIN} units of the cheapest (near tie)")
    by_xy = {tuple(p["xy"]): p["colour"] for p in scene["posts"]}
    got = [{"post": by_xy[tuple(p)], "land": list(l), "dir": d} for p, l, r, d in res["path"]]
    if got != scene["swings"]:
        errs.append("shown swings aren't the unique cheapest route")
    if not SWINGS[0] <= len(scene["swings"]) <= SWINGS[1]:
        errs.append("swing count out of range")
    looks = [(p["colour"], p["shape"]) for p in scene["posts"]]
    if len({c for c, _ in looks}) != len(looks):
        errs.append("two posts share a colour")
    if len({tuple(p["xy"]) for p in scene["posts"]}) != len(scene["posts"]):
        errs.append("two posts share a corner")
    pits = {tuple(p) for p in scene["pits"]}
    if tuple(scene["start"]) in pits or tuple(scene["goal"]) in pits:
        errs.append("start or goal is a pit")
    t = scene["timing"]
    if t["intro"] < MIN_EDGE or t["outro"] < MIN_EDGE or t["pause"] < MIN_PAUSE or min(t["dur"]) < MIN_SWING:
        errs.append("timing too tight")
    if len(t["dur"]) != len(scene["swings"]) or n_frames(scene) > MAX_FRAMES:
        errs.append("timing doesn't fit the swings / 10 s")
    st = scene["style"]
    x1, y1 = st["x0"] + scene["w"] * st["cell"], st["y0"] + scene["h"] * st["cell"]
    if st["x0"] < AREA[0] or st["y0"] < AREA[1] or x1 > AREA[2] or y1 > AREA[3]:
        errs.append("board leaves its screen area")
    if st["post_px"] < 20 or st["particle"]["px"] < 20 or st["cell"] < 54:
        errs.append("objects too small")
    return errs
