"""Scene generation for colored_route_puzzle (v3, 10-question ladder).

A subway-style map: 4-6 stops (dark shape markers: circle, square, triangle,
diamond, star, cross; every shape at most once) sit on the nodes of a routing
grid, and 4-8 coloured routes join them. Every route is a grid path found by
a randomized Dijkstra search over nodes no earlier route uses, so routes are
node-disjoint: they never cross, touch or run along each other; two routes
only meet at a stop, each through its own side (up / down / left / right).
Routes appear over time (each grows from one of its ends), so the draw
order, the direction of growth and the timing exist only in the video.

Five scenarios (`scene["scenario"]`):

  sequential  5-8 routes drawn one at a time, random order and speeds.
  overlap     4-7 routes whose drawing overlaps in time: a new route starts
              every 8-20 frames while earlier ones are still growing, so the
              start order and the finish order differ.
  dead_ends   3-6 normal routes plus 1-2 dead ends: a route that leaves a stop
              and stops short at a crossbar, reaching no other stop.
  erase       5-7 routes drawn one at a time; 1-2 of them later shrink back
              and vanish, so the last frame no longer shows every route.
  traveller   4-7 routes drawn quickly, then a black ring that sits on a stop
              from the start rides 2-4 routes from stop to stop.

All randomness comes from the random.Random passed to build(); the scene is
a plain JSON-able dict that the renderer, the questions and the checks all
read. The only text on screen is the clock (scene["timer"]); scene["labels"]
is empty by design (stops are named by shape, routes by colour).
"""
from __future__ import annotations
import heapq
import math
import random

import numpy as np

FPS = 24
W, H = 1280, 720
MAX_FRAMES = 240                     # 10.0 s
EDGE_F = 12                          # events >= 0.5 s from both ends
GAP_F = 2                            # any two events >= 2 frames apart
CMP_F = 8                            # compared times differ by >= 8 frames (0.33 s)
TIMER_BOX = (20, 16, 200, 72)
TIMER_FONT = 36
AREA = (34, 120, 1246, 696)          # grid nodes live inside this box (clear of the clock)
SCENARIOS = ("sequential", "overlap", "dead_ends", "erase", "traveller")
SHAPES = ("circle", "square", "triangle", "diamond", "star", "cross")
PALETTE = (("red", (214, 40, 40)), ("orange", (242, 130, 20)), ("gold", (214, 172, 0)),
           ("green", (36, 150, 64)), ("lime", (140, 198, 40)), ("teal", (0, 140, 138)),
           ("sky blue", (64, 166, 236)), ("blue", (36, 70, 204)), ("purple", (128, 58, 190)),
           ("pink", (238, 108, 178)), ("brown", (132, 80, 36)))
BACKS = (("white", (250, 250, 248)), ("cream", (248, 243, 229)), ("pale grey", (234, 236, 239)),
         ("pale blue", (229, 237, 246)), ("pale green", (231, 241, 231)), ("sand", (243, 234, 216)))
INKS = ((66, 74, 104), (88, 62, 98), (56, 86, 88), (74, 74, 74))   # stop-marker colours
TOKEN_RGB = (12, 12, 12)
TINTS = ((120, 180, 120), (120, 160, 210), (170, 170, 170), (200, 170, 120))
STYLES = ("solid", "dashed", "dotted")
STYLE_WORD = {"solid": "solid", "dashed": "dashed", "dotted": "dotted"}
BG_PATTERNS = ("plain", "grid", "dots", "gradient")
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))   # up, right, down, left (y grows downwards)
COLOR_SEP = 85.0                     # min RGB distance between two route colours
SCEN = {
    "sequential": {"normal": (5, 8), "dead": (0, 0)},
    "overlap": {"normal": (4, 7), "dead": (0, 0)},
    "dead_ends": {"normal": (3, 6), "dead": (1, 2)},
    "erase": {"normal": (5, 7), "dead": (0, 0)},
    "traveller": {"normal": (4, 7), "dead": (0, 0)},
}
STATS = {"tries": 0, "builds": 0, "fail": {}}   # retry accounting (reporting only, never decides anything)


# ---------------------------------------------------------------- small helpers

def dist(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def mix(a, b, u):
    return tuple(int(round(a[k] + (b[k] - a[k]) * u)) for k in range(3))


def timer_tenths(i: int) -> int:
    """What the on-screen clock shows at frame i, in tenths of a second."""
    return (10 * i) // FPS


def clock_text(tenths: int) -> str:
    return f"{tenths // 10}.{tenths % 10} s"


def frames_showing(tenths: int):
    """Frames whose clock reads `tenths` (inclusive range)."""
    lo = -(-tenths * FPS // 10)
    hi = -(-(tenths + 1) * FPS // 10) - 1
    return lo, hi


def node_xy(scene, n):
    g = scene["grid"]
    return (g["x0"] + n[0] * g["step"], g["y0"] + n[1] * g["step"])


def stop_xy(scene, sid):
    return node_xy(scene, scene["stops"][sid]["node"])


def shape(scene, sid) -> str:
    return scene["stops"][sid]["shape"]


def corner_nodes(nodes):
    """The route's corner nodes (start, every turn, end)."""
    out = [tuple(nodes[0])]
    for k in range(1, len(nodes) - 1):
        a, b, c = nodes[k - 1], nodes[k], nodes[k + 1]
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            out.append(tuple(b))
    out.append(tuple(nodes[-1]))
    return out


def n_corners(route) -> int:
    return len(corner_nodes(route["nodes"])) - 2


def length_steps(route) -> int:
    return len(route["nodes"]) - 1


def ends(route):
    """(a, b) stop ids; b is None for a dead end."""
    return route["a"], route["b"]


def touches(route, sid) -> bool:
    return route["a"] == sid or route["b"] == sid


def is_normal(route) -> bool:
    return route["b"] is not None


# ---------------------------------------------------------------- timing helpers

def events(scene) -> list:
    """Every event frame: (frame, kind, id)."""
    out = []
    for r in scene["routes"]:
        out += [(r["draw"][0], "start", r["id"]), (r["draw"][1], "finish", r["id"])]
        if r["erase"]:
            out += [(r["erase"][0], "erase_start", r["id"]), (r["erase"][1], "gone", r["id"])]
    if scene["token"]:
        for k, rd in enumerate(scene["token"]["rides"]):
            out += [(rd["t"][0], "depart", k), (rd["t"][1], "arrive", k)]
    return sorted(out)


def event_frames(scene) -> list:
    return sorted(e[0] for e in events(scene))


def complete_at(route, i: int) -> bool:
    if i < route["draw"][1]:
        return False
    return not route["erase"] or i <= route["erase"][0]


def growing_at(route, i: int) -> bool:
    return route["draw"][0] < i < route["draw"][1]


def start_order(scene) -> list:
    return sorted(scene["routes"], key=lambda r: r["draw"][0])


def complete_ids(scene, i: int) -> list:
    return [r["id"] for r in scene["routes"] if complete_at(r, i)]


def reach(scene, rids, sid) -> set:
    """Stops reachable from sid along the normal routes in rids (changing routes only at stops)."""
    adj = {s["id"]: set() for s in scene["stops"]}
    for rid in rids:
        r = scene["routes"][rid]
        if is_normal(r):
            adj[r["a"]].add(r["b"])
            adj[r["b"]].add(r["a"])
    seen, todo = {sid}, [sid]
    while todo:
        for nb in adj[todo.pop()]:
            if nb not in seen:
                seen.add(nb)
                todo.append(nb)
    return seen


def stable_interval(scene, lo: int, hi: int) -> bool:
    """No event within GAP_F frames of [lo, hi]."""
    return all(not (lo - GAP_F <= f <= hi + GAP_F) for f in event_frames(scene))


def visible_growth(route):
    """Frames in which the route is clearly mid-growth (past its first quarter, >= 2 frames before done)."""
    s, e = route["draw"]
    return s + max(GAP_F, -(-(e - s) // 4)), e - GAP_F


# ---------------------------------------------------------------- layout

def _grid(rng):
    step = rng.randint(44, 62)
    nx_max = (AREA[2] - AREA[0]) // step + 1
    ny_max = (AREA[3] - AREA[1]) // step + 1
    nx = rng.randint(max(11, nx_max - 5), nx_max)
    ny = rng.randint(max(8, ny_max - 2), ny_max)
    x0 = AREA[0] + rng.randint(0, (AREA[2] - AREA[0]) - (nx - 1) * step)
    y0 = AREA[1] + rng.randint(0, (AREA[3] - AREA[1]) - (ny - 1) * step)
    return {"nx": nx, "ny": ny, "step": step, "x0": x0, "y0": y0}


def _style(rng, step):
    bg_name, bg = rng.choice(BACKS)
    ink = rng.choice(INKS)
    w = rng.randint(7, min(12, step // 4))
    mh = rng.randint(16, min(22, int(step * 0.4)))
    return {"bg": list(bg), "bg_name": bg_name, "pattern": rng.choice(BG_PATTERNS), "ink": list(ink),
            "w": w, "mh": mh, "corner": rng.choice(("sharp", "round")),
            "dash": [3 * w, int(round(1.6 * w))], "dot": [int(round(2.4 * w)), round(0.65 * w, 2)],
            "token_r": int(round(mh * 1.45)) + 10, "token_w": 7, "shadow": rng.random() < 0.5,
            "tint": list(rng.choice(TINTS)), "tint_u": round(rng.uniform(0.18, 0.28), 3)}


def zone(node):
    return {(node[0] + dx, node[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}


def _place_stops(rng, grid, n):
    for _ in range(20):
        cand = [(x, y) for x in range(2, grid["nx"] - 2) for y in range(2, grid["ny"] - 2)]
        rng.shuffle(cand)
        nodes = []
        for c in cand:
            if all(max(abs(c[0] - o[0]), abs(c[1] - o[1])) >= 4 for o in nodes):
                nodes.append(c)
                if len(nodes) == n:
                    return nodes
    return None


def _obstacles(rng, grid, stop_nodes):
    out = []
    for _ in range(rng.choice((0, 0, 1, 1, 2))):
        for _ in range(60):
            wn, hn = rng.randint(1, 3), rng.randint(1, 2)
            i0, j0 = rng.randint(1, grid["nx"] - 2 - wn), rng.randint(1, grid["ny"] - 2 - hn)
            box = (i0, j0, i0 + wn, j0 + hn)
            if any(box[0] - 2 <= s[0] <= box[2] + 2 and box[1] - 2 <= s[1] <= box[3] + 2 for s in stop_nodes):
                continue
            if any(not (box[2] + 2 <= o[0] or o[2] + 2 <= box[0] or box[3] + 2 <= o[1] or o[3] + 2 <= box[1])
                   for o in out):
                continue
            out.append(list(box))
            break
    return out


def obstacle_nodes(scene_or_obs):
    obs = scene_or_obs["obstacles"] if isinstance(scene_or_obs, dict) else scene_or_obs
    return {(i, j) for o in obs for i in range(o[0], o[2] + 1) for j in range(o[1], o[3] + 1)}


def _pairs(rng, n_stops, n_normal, n_dead):
    """Route end pairs: every stop gets >= 1 route, no stop more than 4, <= 2 routes per stop pair."""
    deg = [0] * n_stops
    order = list(range(n_stops))
    rng.shuffle(order)
    pairs = []
    for k in range(0, n_stops - 1, 2):
        pairs.append((order[k], order[k + 1]))
    if n_stops % 2:
        pairs.append((order[-1], rng.choice(order[:-1])))
    for a, b in pairs:
        deg[a] += 1
        deg[b] += 1
    tries = 0
    while len(pairs) < n_normal and tries < 200:
        tries += 1
        cand = [s for s in range(n_stops) if deg[s] < 4]
        if len(cand) < 2:
            break
        a, b = rng.sample(cand, 2)
        if sum(1 for p in pairs if {p[0], p[1]} == {a, b}) >= 2:
            continue
        if sum(1 for p in pairs if {p[0], p[1]} == {a, b}) == 1 and rng.random() < 0.6:
            continue
        pairs.append((a, b))
        deg[a] += 1
        deg[b] += 1
    for _ in range(n_dead):
        cand = [s for s in range(n_stops) if deg[s] < 4]
        a = rng.choice(cand)
        pairs.append((a, None))
        deg[a] += 1
    rng.shuffle(pairs)
    return [(a, b) if b is None or rng.random() < 0.5 else (b, a) for a, b in pairs]


class _Router:
    def __init__(self, rng, grid, stop_nodes, blocked):
        self.rng = rng
        self.nx, self.ny = grid["nx"], grid["ny"]
        self.stops = stop_nodes
        self.used = set(blocked)
        self.free_ports = {k: [(n[0] + d[0], n[1] + d[1]) for d in DIRS] for k, n in enumerate(stop_nodes)}
        self.reserved = set()
        for n in stop_nodes:
            self.reserved |= zone(n)

    def _ok(self, n):
        return 0 <= n[0] < self.nx and 0 <= n[1] < self.ny

    def _search(self, src, targets, wiggle, turn):
        cost = {}
        best = {(src, None): 0.0}
        prev = {}
        heap = [(0.0, src, -1)]
        while heap:
            d, node, came = heapq.heappop(heap)
            if node in targets:
                path, key = [node], (node, came)
                while key in prev:
                    key = prev[key]
                    path.append(key[0])
                return path[::-1]
            if d > best.get((node, came), float("inf")):
                continue
            for k, (dx, dy) in enumerate(DIRS):
                nb = (node[0] + dx, node[1] + dy)
                if not self._ok(nb) or nb in self.used or (nb in self.reserved and nb not in targets):
                    continue
                if nb not in cost:
                    cost[nb] = 1.0 + self.rng.random() * wiggle
                nd = d + cost[nb] + (turn if came not in (-1, k) else 0.0)
                if nd < best.get((nb, k), float("inf")):
                    best[(nb, k)] = nd
                    prev[(nb, k)] = (node, came)
                    heapq.heappush(heap, (nd, nb, k))
        return None

    def route(self, a, b, wiggle, turn, stub):
        ports = list(self.free_ports[a])
        self.rng.shuffle(ports)
        for pa in ports:
            if pa in self.used:
                continue
            if b is None:
                targets = self._stub_targets(pa, stub)
                if not targets:
                    continue
                path = self._search(pa, targets, wiggle, turn)
                if path and len(path) >= 3:
                    self.free_ports[a].remove(pa)
                    self.used.update(path)
                    return [self.stops[a]] + path
                continue
            tb = {p for p in self.free_ports[b] if p not in self.used and p != pa}
            if not tb:
                return None
            path = self._search(pa, tb, wiggle, turn)
            if path:
                self.free_ports[a].remove(pa)
                self.free_ports[b].remove(path[-1])
                self.used.update(path)
                return [self.stops[a]] + path + [self.stops[b]]
        return None

    def _stub_targets(self, port, stub):
        lo, hi = stub
        out = []
        for x in range(1, self.nx - 1):
            for y in range(1, self.ny - 1):
                d = abs(x - port[0]) + abs(y - port[1])
                n = (x, y)
                if lo <= d <= hi and n not in self.used and n not in self.reserved and \
                        all(max(abs(x - s[0]), abs(y - s[1])) >= 3 for s in self.stops):
                    out.append(n)
        if not out:
            return set()
        return set(self.rng.sample(out, min(4, len(out))))


def _colours(rng, n, style):
    bg, ink = style["bg"], style["ink"]
    tint = mix(bg, style["tint"], style["tint_u"])
    cand = [(nm, c) for nm, c in PALETTE if dist(c, bg) >= 125 and dist(c, ink) >= 100
            and dist(c, TOKEN_RGB) >= 100 and dist(c, tint) >= 120]
    rng.shuffle(cand)
    out = []
    for nm, c in cand:
        if all(dist(c, o[1]) >= COLOR_SEP for o in out):
            out.append((nm, c))
        if len(out) == n:
            return out
    return None


def _line_styles(rng, n):
    mode = rng.random()
    if mode < 0.3:
        return ["solid"] * n
    return [rng.choices(STYLES, weights=(0.5, 0.25, 0.25))[0] for _ in range(n)]


# ---------------------------------------------------------------- timing

def _sequential(rng, seq, t, dr, pr, er=(8, 14)):
    """seq: list of ("draw"|"erase", rid). Returns {rid: draw}, {rid: erase}, end frame."""
    draws, erases = {}, {}
    for kind, rid in seq:
        if kind == "draw":
            d = rng.randint(*dr)
            draws[rid] = [t, t + d]
            t += d + rng.randint(*pr)
        else:
            d = rng.randint(*er)
            erases[rid] = [t, t + d]
            t += d + rng.randint(*pr)
    return draws, erases, t


def _timing(rng, sc, n, normal_ids, token_rides):
    """Returns (draws, erases, rides timing, last event frame) or None."""
    order = list(range(n))
    rng.shuffle(order)
    intro = rng.randint(EDGE_F, 22)
    if sc == "overlap":
        for _ in range(40):
            draws, s = {}, intro
            for k, rid in enumerate(order):
                if k:
                    s += rng.randint(CMP_F, 20)
                draws[rid] = [s, s + rng.randint(18, 44)]
            fr = sorted(f for v in draws.values() for f in v)
            if all(b - a >= GAP_F for a, b in zip(fr, fr[1:])):
                return draws, {}, [], fr[-1]
        return None
    if sc == "erase":
        seq = [("draw", r) for r in order]
        k = 2 if n >= 6 and rng.random() < 0.5 else 1
        targets = rng.sample(order[:-1], k)
        for tg in targets:
            pos_min = max(next(i for i, x in enumerate(seq) if x == ("draw", tg)) + 1, 3)
            seq.insert(rng.randint(pos_min, len(seq)), ("erase", tg))
        draws, erases, t = _sequential(rng, seq, intro, (10, 20), (4, 9))
        return draws, erases, [], t - 1
    if sc == "traveller":
        draws, _, t = _sequential(rng, [("draw", r) for r in order], intro, (8, 13), (3, 6))
        t += rng.randint(4, 9)
        rides = []
        for k in range(token_rides):
            d = rng.randint(14, 24)
            rides.append([t, t + d])
            t += d + rng.randint(6, 12)
        return draws, {}, rides, rides[-1][1]
    draws, _, t = _sequential(rng, [("draw", r) for r in order], intro, (10, 24), (4, 11))
    return draws, {}, [], t - 1


def _trail(rng, routes, n_stops, want):
    """A walk along normal routes (each used once) of up to `want` rides; >= 2 needed."""
    best = None
    for _ in range(30):
        cur = rng.randrange(n_stops)
        used, rides = set(), []
        for _ in range(want):
            opts = [r for r in routes if is_normal(r) and r["id"] not in used and touches(r, cur)]
            if not opts:
                break
            r = rng.choice(opts)
            nxt = r["b"] if r["a"] == cur else r["a"]
            rides.append({"route": r["id"], "from": cur, "to": nxt})
            used.add(r["id"])
            cur = nxt
        if best is None or len(rides) > len(best):
            best = rides
        if len(best) == want:
            break
    return best if best and len(best) >= 2 else None


# ---------------------------------------------------------------- build

def _fail(why):
    STATS["fail"][why] = STATS["fail"].get(why, 0) + 1
    return None


def _try(rng, sc, counts):
    STATS["tries"] += 1
    grid = _grid(rng)
    style = _style(rng, grid["step"])
    n_stops, n_normal, n_dead = counts
    stop_nodes = _place_stops(rng, grid, n_stops)
    if stop_nodes is None:
        return _fail("stops")
    obstacles = _obstacles(rng, grid, stop_nodes)
    pairs = _pairs(rng, n_stops, n_normal, n_dead)
    wiggle, turn = round(rng.uniform(1.0, 3.5), 3), round(rng.uniform(0.2, 1.2), 3)
    paths = None
    for _ in range(8):                       # a few routing orders on the same layout
        router = _Router(rng, grid, stop_nodes, obstacle_nodes(obstacles))
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        paths = {}
        for i in idx:
            a, b = pairs[i]
            p = router.route(a, b, wiggle, turn, (3, 7))
            if p is None:
                paths = None
                break
            paths[i] = p
        if paths is not None:
            break
    if paths is None:
        return _fail("route")
    n = len(pairs)
    cols = _colours(rng, n, style)
    if cols is None:
        return _fail("colours")
    lstyles = _line_styles(rng, n)
    routes = [{"id": i, "color": cols[i][0], "rgb": list(cols[i][1]), "style": lstyles[i],
               "a": pairs[i][0], "b": pairs[i][1], "nodes": [list(x) for x in paths[i]],
               "grow": "a" if pairs[i][1] is None and rng.random() < 0.6 else rng.choice(("a", "b")),
               "draw": None, "erase": None} for i in range(n)]
    trail = None
    if sc == "traveller":
        trail = _trail(rng, routes, n_stops, rng.randint(2, 4))
        if trail is None:
            return _fail("trail")
    tm = _timing(rng, sc, n, [r["id"] for r in routes if is_normal(r)], len(trail) if trail else 0)
    if tm is None:
        return _fail("timing")
    draws, erases, rides, last = tm
    hold = rng.randint(EDGE_F + 1, 22)
    if last + hold > MAX_FRAMES:
        return _fail("too long")
    for r in routes:
        r["draw"] = draws[r["id"]]
        r["erase"] = erases.get(r["id"])
    token = None
    if trail:
        token = {"start": trail[0]["from"],
                 "rides": [dict(rd, t=rides[k]) for k, rd in enumerate(trail)]}
    scene = {"scenario": sc, "grid": grid, "style": style, "obstacles": obstacles,
             "stops": [{"id": k, "shape": s, "node": list(stop_nodes[k])}
                       for k, s in enumerate(rng.sample(SHAPES, n_stops))],
             "routes": routes, "token": token, "n_frames": last + hold,
             "timer": {"box": list(TIMER_BOX), "size": TIMER_FONT}, "labels": [],
             "router": {"wiggle": wiggle, "turn": turn}}
    errs = check(scene)
    if errs:
        return _fail("check: " + errs[0].split(":")[0])
    return scene


def build(rng: random.Random) -> dict:
    sc = rng.choice(SCENARIOS)
    cfg = SCEN[sc]
    n_stops = rng.randint(4, 6)
    counts = (n_stops, rng.randint(max(cfg["normal"][0], (n_stops + 1) // 2), cfg["normal"][1]),
              rng.randint(*cfg["dead"]))
    for k in range(600):            # retries keep the drawn counts, so big maps are not filtered out
        if k and k % 150 == 0 and counts[1] > max(cfg["normal"][0], (n_stops + 1) // 2):
            counts = (counts[0], counts[1] - 1, counts[2])     # very rare: one route fewer
            STATS["fail"]["fewer routes"] = STATS["fail"].get("fewer routes", 0) + 1
        scene = _try(rng, sc, counts)
        if scene is not None and _askable(scene):
            STATS["builds"] += 1
            return scene
        if scene is not None:
            _fail("too few questions")
    raise RuntimeError("could not build a scene")


def _askable(scene) -> bool:
    import questions  # local import: questions imports sim
    C = questions.candidates(scene, random.Random(0))
    return all(len({(c[0], str(c[1])) for c in C[lv]}) >= 3 for lv in range(1, 6))


# ---------------------------------------------------------------- checks

def route_px(scene, r):
    return [node_xy(scene, n) for n in r["nodes"]]


def _dense(pts, step=4.0):
    out = []
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        L = abs(x2 - x1) + abs(y2 - y1)
        k = max(1, int(L // step))
        out += [(x1 + (x2 - x1) * j / k, y1 + (y2 - y1) * j / k) for j in range(k)]
    out.append(pts[-1])
    return out


def drawing_extent(scene):
    """Bounding box of everything ever drawn except the background."""
    st = scene["style"]
    xs, ys = [], []
    pad = max(st["w"], st["mh"] * 1.25) + 3
    for r in scene["routes"]:
        for x, y in route_px(scene, r):
            xs.append(x)
            ys.append(y)
    for s in scene["stops"]:
        x, y = stop_xy(scene, s["id"])
        xs.append(x)
        ys.append(y)
    box = [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]
    if scene["token"]:
        R = st["token_r"] + 3
        for rd in scene["token"]["rides"]:
            for x, y in route_px(scene, scene["routes"][rd["route"]]):
                box = [min(box[0], x - R), min(box[1], y - R), max(box[2], x + R), max(box[3], y + R)]
    g = scene["grid"]
    for o in scene["obstacles"]:
        box = [min(box[0], g["x0"] + o[0] * g["step"] - g["step"] * 0.45), min(box[1], g["y0"] + o[1] * g["step"] - g["step"] * 0.45),
               max(box[2], g["x0"] + o[2] * g["step"] + g["step"] * 0.45), max(box[3], g["y0"] + o[3] * g["step"] + g["step"] * 0.45)]
    return box


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def check(scene) -> list:
    errs = []
    sc, st, g = scene["scenario"], scene["style"], scene["grid"]
    routes, stops = scene["routes"], scene["stops"]
    cfg = SCEN[sc]
    nn = sum(1 for r in routes if is_normal(r))
    nd = len(routes) - nn
    if not (4 <= len(stops) <= 6 and cfg["normal"][0] <= nn <= cfg["normal"][1] and cfg["dead"][0] <= nd <= cfg["dead"][1]):
        errs.append("counts: stop or route count out of range")
    if len({s["shape"] for s in stops}) != len(stops):
        errs.append("identity: stop shapes not unique")
    if len({r["color"] for r in routes}) != len(routes):
        errs.append("identity: route colours not unique")
    tint = mix(st["bg"], st["tint"], st["tint_u"])
    for k, r in enumerate(routes):
        if dict(PALETTE)[r["color"]] != tuple(r["rgb"]):
            errs.append("identity: colour name and rgb disagree")
        for o in routes[k + 1:]:
            if dist(r["rgb"], o["rgb"]) < COLOR_SEP:
                errs.append("identity: two route colours too close")
        if dist(r["rgb"], st["bg"]) < 125 or dist(r["rgb"], st["ink"]) < 100 or dist(r["rgb"], TOKEN_RGB) < 100 \
                or dist(r["rgb"], tint) < 120:
            errs.append("identity: route colour too close to background, stops, ring or park")
    if dist(st["ink"], TOKEN_RGB) < 90 or dist(st["ink"], st["bg"]) < 150:
        errs.append("identity: stop colour too close to ring or background")
    # sizes (1280x720): objects >= 20 px, lines >= 6 px (>= 3 px at half size)
    if st["w"] < 6 or 2 * st["mh"] < 30 or st["w"] * 4 > g["step"] or st["mh"] > 0.42 * g["step"]:
        errs.append("sizes: line or marker size out of range")
    if st["token_r"] - st["token_w"] < st["mh"] * 1.42 + 2:
        errs.append("sizes: ring would cover its stop marker")
    # grid paths: unit steps, ends at their stops through distinct ports, node-disjoint
    stop_nodes = {tuple(s["node"]): s["id"] for s in stops}
    zones = {}
    for s in stops:
        for z in zone(s["node"]):
            zones[z] = s["id"]
    used = {}
    blocked = obstacle_nodes(scene)
    ports = set()
    for r in routes:
        nodes = [tuple(n) for n in r["nodes"]]
        if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in zip(nodes, nodes[1:])):
            errs.append("path: route makes a non-unit step")
        if any(not (0 <= n[0] < g["nx"] and 0 <= n[1] < g["ny"]) for n in nodes):
            errs.append("path: route leaves the grid")
        if nodes[0] != tuple(stops[r["a"]]["node"]):
            errs.append("path: route doesn't start at its stop")
        if is_normal(r) and (nodes[-1] != tuple(stops[r["b"]]["node"]) or r["a"] == r["b"]):
            errs.append("path: route doesn't end at its stop")
        if not is_normal(r) and (nodes[-1] in zones or len(nodes) < 4):
            errs.append("path: dead end inside a stop zone or too short")
        inner = nodes[1:-1] if is_normal(r) else nodes[1:]
        for n in inner:
            if n in used:
                errs.append("path: two routes share a node")
            used[n] = r["id"]
            if n in blocked:
                errs.append("path: route runs through a park")
            if n in zones and n not in (nodes[1], nodes[-2] if is_normal(r) else None):
                errs.append("path: route passes through a stop zone")
        for p in ((nodes[0], nodes[1]),) + (((nodes[-1], nodes[-2]),) if is_normal(r) else ()):
            if p in ports:
                errs.append("path: two routes use the same side of a stop")
            ports.add(p)
    # independent pixel test: points of different routes stay >= 0.9 step apart away from stops
    rad = max(st["mh"] * 1.3 + st["w"], 0.75 * g["step"])
    spx = [stop_xy(scene, s["id"]) for s in stops]
    S = np.array(spx, dtype=float)
    pts = []
    for r in routes:
        P = np.array(_dense(route_px(scene, r)), dtype=float)
        dS = np.sqrt(((P[:, None, :] - S[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        pts.append(P[dS > rad])
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if len(pts[i]) and len(pts[j]):
                d = np.sqrt(((pts[i][:, None, :] - pts[j][None, :, :]) ** 2).sum(axis=2)).min()
                if d < 0.9 * g["step"] - 4:
                    errs.append("path: two routes come too close")
    # timing: <= 10 s, events >= 0.5 s from both ends and >= 2 frames apart
    nf = scene["n_frames"]
    fr = event_frames(scene)
    if nf > MAX_FRAMES:
        errs.append("timing: clip longer than 10 s")
    if fr[0] < EDGE_F or fr[-1] > nf - 1 - EDGE_F:
        errs.append("timing: an event within 0.5 s of the start or the end")
    if any(b - a < GAP_F for a, b in zip(fr, fr[1:])):
        errs.append("timing: two events closer than 2 frames")
    for r in routes:
        s, e = r["draw"]
        if e - s < 8:
            errs.append("timing: route drawn in under 8 frames")
        if r["erase"] and (r["erase"][0] - e < 6 or r["erase"][1] - r["erase"][0] < 6):
            errs.append("timing: erased route not shown long enough")
    if sc == "overlap":
        st0 = sorted(r["draw"][0] for r in routes)
        if any(b - a < CMP_F for a, b in zip(st0, st0[1:])):
            errs.append("timing: overlap starts too close")
        if not any(growing_at(a, i) and growing_at(b, i) for a in routes for b in routes if a is not b
                   for i in range(a["draw"][0], a["draw"][1])):
            errs.append("timing: overlap clip never draws two routes at once")
    else:
        spans = sorted([r["draw"] for r in routes] + [r["erase"] for r in routes if r["erase"]])
        if any(b[0] <= a[1] for a, b in zip(spans, spans[1:])):
            errs.append("timing: one-at-a-time drawing overlaps")
    if (sc == "erase") != any(r["erase"] for r in routes):
        errs.append("scenario: erasing only in the erase scenario")
    if sc == "erase" and sum(1 for r in routes if complete_at(r, nf - 1)) < 2:
        errs.append("scenario: fewer than 2 routes left at the end")
    tk = scene["token"]
    if (sc == "traveller") != bool(tk):
        errs.append("scenario: ring only in the traveller scenario")
    if tk:
        last_draw = max(r["draw"][1] for r in routes)
        cur, prev_t1, seen = tk["start"], last_draw, set()
        for rd in tk["rides"]:
            r = routes[rd["route"]]
            if not is_normal(r) or {r["a"], r["b"]} != {rd["from"], rd["to"]} or rd["from"] != cur \
                    or rd["route"] in seen:
                errs.append("ring: ride doesn't follow a route from where the ring is")
            if rd["t"][0] - prev_t1 < 4 or rd["t"][1] - rd["t"][0] < 10:
                errs.append("ring: ride too early or too short")
            seen.add(rd["route"])
            cur, prev_t1 = rd["to"], rd["t"][1]
    # layout: everything inside the frame and clear of the clock
    box = drawing_extent(scene)
    tb = scene["timer"]["box"]
    if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
        errs.append("layout: drawing leaves the frame")
    if _overlap(box, (tb[0] - 8, tb[1] - 8, tb[2] + 8, tb[3] + 8)):
        errs.append("layout: drawing reaches the clock")
    if scene["labels"]:
        errs.append("layout: no text labels expected besides the clock")
    return errs
