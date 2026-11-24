"""Blinking signal networks: coloured, patterned nodes joined by straight
links. A starting node flashes; a node that lights up sends a pulse down each
of its links except those that already brought it a pulse. A pulse reaching
a dark node lights it (and it fires in turn); a pulse reaching a lit node
fizzles. Every lit node keeps a bright halo, so the order has to be watched.

Five scenarios (scene["scenario"]):

  hidden_slow     one link, drawn like every other, carries pulses at half speed
  marked_speeds   dashed links carry pulses at half speed, solid links at full speed
  one_way         arrowed links carry pulses only in the arrow's direction
  two_sources     a second node flashes on its own a little after the first
  double_trigger  double-outlined nodes need two pulses before they light;
                  after the first one they show a half halo

Everything lives in pixel space (1280x720 frame): node centres in px, speed
in px/s, so the renderer, the generator and the independent answer checker
all share the same geometry. The event simulation here is an exact
priority-queue replay; answers.py re-derives everything with a separate
time-stepped simulation.
"""
from __future__ import annotations
import heapq
import itertools
import math
import random

W, H = 1280, 720
FPS = 24
SCENARIOS = ("hidden_slow", "marked_speeds", "one_way", "two_sources", "double_trigger")
LAYOUTS = ("scatter", "grid", "ring", "clusters")
MARGIN = 0.3            # s: min gap between any two events a question depends on
MIN_SEP = 0.125         # s: (3 frames) no two nodes ever light up closer than this
PAD = 0.5               # s: question events stay this far from the clip start and end
MAX_DUR = 10.0
TIMER_BOX = (1078, 12, 1266, 78)          # the clock; nodes never come near it
AREA = (100, 150, 1180, 650)              # where node centres may go
MIN_TRAVEL = 0.4        # s: every pulse is on screen at least this long

COLOURS = {
    "red": (228, 50, 50), "orange": (247, 142, 30), "yellow": (246, 216, 40),
    "green": (52, 178, 70), "cyan": (36, 200, 216), "blue": (54, 104, 236),
    "purple": (156, 84, 214), "pink": (242, 112, 184), "brown": (150, 94, 44),
}
SHAPES = ("circle", "square", "triangle", "diamond", "star")
PATTERNS = ("solid", "striped", "dotted", "ringed")
THEMES = {   # name: (background, dark?)
    "navy": ((16, 22, 40), True), "charcoal": ((28, 28, 32), True), "forest": ((16, 34, 28), True),
    "plum": ((34, 20, 42), True), "slate": ((32, 40, 52), True), "cocoa": ((38, 28, 22), True),
    "cream": ((238, 232, 216), False), "fog": ((222, 228, 236), False), "sky": ((216, 232, 244), False),
}
BG_STYLES = ("plain", "gradient", "grid", "dots", "hatch", "vignette")
DARK_LINKS = ((112, 122, 150), (140, 140, 152), (98, 132, 150), (150, 132, 112), (126, 112, 150))
LIGHT_LINKS = ((120, 120, 132), (104, 122, 140), (138, 116, 100), (112, 112, 112))


# ----------------------------------------------------------------- geometry
def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def seg_cross(p1, p2, p3, p4) -> bool:
    """Proper crossing of two segments (sharing an endpoint doesn't count)."""
    if len({tuple(p1), tuple(p2), tuple(p3), tuple(p4)}) < 4:
        return False
    return (_orient(p1, p2, p3) * _orient(p1, p2, p4) < 0 and
            _orient(p3, p4, p1) * _orient(p3, p4, p2) < 0)


def pt_seg(p, a, b) -> float:
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    L2 = (bx - ax) ** 2 + (by - ay) ** 2
    if L2 <= 0:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / L2))
    return math.dist(p, (ax + t * (bx - ax), ay + t * (by - ay)))


def node_map(scene) -> dict:
    return {n["id"]: n for n in scene["nodes"]}


def xy(scene, nid):
    n = node_map(scene)[nid]
    return (n["x"], n["y"])


def edge_len(scene, e) -> float:
    return math.dist(xy(scene, e["a"]), xy(scene, e["b"]))


def travel(scene, e, unslow=False) -> float:
    v = scene["speed"] * (0.5 if e["slow"] and not unslow else 1.0)
    return edge_len(scene, e) / v


def same_edge(e, pair) -> bool:
    return sorted([e["a"], e["b"]]) == sorted(pair)


# --------------------------------------------------------------- simulation
def simulate(scene, cut=None, starts=None, unslow=False, single=False, reverse=None) -> dict:
    """Exact event replay. Variants for counterfactuals: `cut` removes the
    link [a, b]; `starts` replaces the spontaneous flashes; `unslow` makes
    every link full speed; `single` makes every node light on its first
    pulse; `reverse` flips the one-way direction of link [a, b]."""
    nodes = node_map(scene)
    thr = {k: (1 if single else nodes[k]["threshold"]) for k in nodes}
    links = []
    for e in scene["edges"]:
        if cut and same_edge(e, cut):
            continue
        a, b = e["a"], e["b"]
        if e["oneway"] and reverse and same_edge(e, reverse):
            a, b = b, a
        links.append((a, b, e["oneway"], travel(scene, e, unslow)))
    starts = scene["starts"] if starts is None else starts
    heap, seq = [], 0
    for nid, t in starts:
        heap.append((float(t), seq, "start", nid, None))
        seq += 1
    heapq.heapify(heap)
    fire, parent, charge, got = {}, {}, {}, {k: set() for k in nodes}
    arrivals = {k: [] for k in nodes}
    pulses = []

    def fire_node(n, t, par):
        nonlocal seq
        fire[n] = t
        parent[n] = par
        for i, (a, b, ow, w) in enumerate(links):
            if n == a:
                other = b
            elif n == b and not ow:
                other = a
            else:
                continue
            if other in got[n]:
                continue
            pulses.append({"from": n, "to": other, "t0": t, "t1": t + w})
            heapq.heappush(heap, (t + w, seq, "arrive", other, n))
            seq += 1

    while heap:
        t, _, kind, n, src = heapq.heappop(heap)
        if kind == "start":
            arrivals[n].append([t, None])
            if n not in fire:
                fire_node(n, t, None)
            continue
        arrivals[n].append([t, src])
        if n in fire:
            continue
        got[n].add(src)
        charge[n] = charge.get(n, 0) + 1
        if charge[n] >= thr[n]:
            fire_node(n, t, src)
    r6 = lambda v: round(v, 6)  # noqa: E731
    out_p = []
    for p in pulses:
        to = p["to"]
        if to in fire and abs(fire[to] - p["t1"]) < 1e-9 and parent[to] == p["from"]:
            res = "lit"
        elif to in fire and fire[to] < p["t1"] - 1e-9:
            res = "fizzle"
        else:
            res = "charge"
        out_p.append({"from": p["from"], "to": to, "t0": r6(p["t0"]), "t1": r6(p["t1"]), "result": res})
    out_p.sort(key=lambda p: (p["t0"], p["t1"], p["from"], p["to"]))
    half = {}
    for k in sorted(nodes):
        if thr[k] > 1:
            ts = [a[0] for a in arrivals[k] if a[1] is not None]
            if ts and (k not in fire or ts[0] < fire[k] - 1e-9):
                half[k] = r6(ts[0])
    return {"fire": {k: r6(fire[k]) for k in sorted(fire)},
            "parent": {k: parent[k] for k in sorted(parent)},
            "arrivals": {k: [[r6(t), s] for t, s in sorted(arrivals[k], key=lambda a: a[0])]
                         for k in sorted(arrivals)},
            "half": half, "pulses": out_p}


# ------------------------------------------------------------------ layout
def _layout(rng, style, n, box, dmin):
    x0, y0, x1, y1 = box
    pts = []

    def ok(p):
        return (AREA[0] <= p[0] <= AREA[2] and AREA[1] <= p[1] <= AREA[3] and
                all(math.dist(p, q) >= dmin for q in pts))
    if style == "scatter":
        for _ in range(3000):
            p = (rng.uniform(x0, x1), rng.uniform(y0, y1))
            if ok(p):
                pts.append(p)
            if len(pts) == n:
                break
    elif style == "grid":
        cols = max(2, int((x1 - x0) // (dmin * 1.15)) + 1)
        rows = max(2, int((y1 - y0) // (dmin * 1.15)) + 1)
        cells = [(c, r) for c in range(cols) for r in range(rows)]
        rng.shuffle(cells)
        dx, dy = (x1 - x0) / max(1, cols - 1), (y1 - y0) / max(1, rows - 1)
        for c, r in cells:
            p = (x0 + c * dx + rng.uniform(-0.18, 0.18) * dx, y0 + r * dy + rng.uniform(-0.18, 0.18) * dy)
            if ok(p):
                pts.append(p)
            if len(pts) == n:
                break
    elif style == "ring":
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        inner = rng.randint(0, 2)
        k = n - inner
        a0 = rng.uniform(0, 2 * math.pi)
        for i in range(k):
            a = a0 + 2 * math.pi * i / k + rng.uniform(-0.12, 0.12)
            p = (cx + rx * math.cos(a), cy + ry * math.sin(a))
            if ok(p):
                pts.append(p)
        for _ in range(400):
            if len(pts) >= n:
                break
            p = (cx + rng.uniform(-0.45, 0.45) * rx, cy + rng.uniform(-0.45, 0.45) * ry)
            if ok(p):
                pts.append(p)
    else:   # clusters
        kc = rng.randint(2, 3)
        cents = [(rng.uniform(x0, x1), rng.uniform(y0, y1)) for _ in range(kc)]
        for i in range(4000):
            c = cents[i % kc]
            p = (c[0] + rng.gauss(0, dmin * 0.9), c[1] + rng.gauss(0, dmin * 0.9))
            if x0 - dmin <= p[0] <= x1 + dmin and ok(p):
                pts.append(p)
            if len(pts) == n:
                break
    if len(pts) < n:
        return None
    return [(round(p[0], 1), round(p[1], 1)) for p in pts]


def _angles_ok(pos, edges, min_deg=25.0) -> bool:
    for k in pos:
        angs = sorted(math.atan2(pos[b][1] - pos[a][1], pos[b][0] - pos[a][0]) if a == k else
                      math.atan2(pos[a][1] - pos[b][1], pos[a][0] - pos[b][0])
                      for a, b in edges if k in (a, b))
        if len(angs) < 2:
            continue
        gaps = [b - a for a, b in zip(angs, angs[1:])] + [angs[0] + 2 * math.pi - angs[-1]]
        if min(gaps) < math.radians(min_deg):
            return False
    return True


def _clear(pos, a, b, R) -> bool:
    return all(pt_seg(pos[c], pos[a], pos[b]) >= R + 26 for c in pos if c not in (a, b))


def _edges(rng, pos, R, dmin):
    names = sorted(pos)
    cand = sorted((math.dist(pos[a], pos[b]), a, b) for a, b in itertools.combinations(names, 2))
    comp = {x: x for x in names}

    def root(x):
        while comp[x] != x:
            x = comp[x]
        return x
    edges = []
    for d, a, b in cand:                       # Euclidean MST (never self-crossing)
        if root(a) != root(b):
            comp[root(a)] = root(b)
            edges.append((a, b))
    if any(not _clear(pos, a, b, R) for a, b in edges) or not _angles_ok(pos, edges):
        return None
    deg = {x: sum(x in e for e in edges) for x in names}
    extra = rng.randint(1, max(1, len(names) // 2))
    pool = [(a, b) for d, a, b in cand if d <= 2.6 * dmin and (a, b) not in edges]
    rng.shuffle(pool)
    for a, b in pool:
        if extra == 0:
            break
        if deg[a] >= 4 or deg[b] >= 4 or not _clear(pos, a, b, R):
            continue
        if any(seg_cross(pos[a], pos[b], pos[c], pos[d]) for c, d in edges):
            continue
        if not _angles_ok(pos, edges + [(a, b)]):
            continue
        edges.append((a, b))
        deg[a] += 1
        deg[b] += 1
        extra -= 1
    return edges


def _theme(rng):
    name = rng.choice(sorted(THEMES))
    bg, dark = THEMES[name]
    return {"name": name, "bg": list(bg), "dark": dark, "style": rng.choice(BG_STYLES),
            "link": list(rng.choice(DARK_LINKS if dark else LIGHT_LINKS)),
            "halo": [255, 255, 255] if dark else [24, 24, 30],
            "pulse": [255, 255, 255] if dark else [24, 24, 30],
            "link_w": rng.randint(3, 6), "rail": rng.random() < 0.25,
            "texture_seed": rng.randint(0, 10 ** 6)}


def _palette(theme):
    bg = theme["bg"]
    out = []
    for c in sorted(COLOURS):
        if not theme["dark"] and c in ("yellow", "cyan"):
            continue
        if math.dist(COLOURS[c], bg) >= 130:
            out.append(c)
    return out


def _try(rng, scenario, n):
    R = rng.randint(20, 30)
    layout = rng.choice(LAYOUTS)
    dmin = 4 * R + rng.uniform(28, 70)
    aw, ah = AREA[2] - AREA[0], AREA[3] - AREA[1]
    bw, bh = rng.uniform(0.6, 1.0) * aw, rng.uniform(0.66, 1.0) * ah
    bx, by = AREA[0] + rng.uniform(0, aw - bw), AREA[1] + rng.uniform(0, ah - bh)
    pts = _layout(rng, layout, n, (bx, by, bx + bw, by + bh), dmin)
    if pts is None:
        return None
    ids = [f"n{i}" for i in range(n)]
    pos = dict(zip(ids, pts))
    elist = _edges(rng, pos, R, dmin)
    if elist is None or len(elist) < 5:
        return None
    theme = _theme(rng)
    if scenario == "marked_speeds":
        theme["rail"] = False                      # dashed vs solid is the only line-style cue
    pal = _palette(theme)
    shapes = rng.sample(SHAPES, rng.randint(2, len(SHAPES)))
    combos = [(c, s) for c in pal for s in shapes]
    if len(combos) < n:
        return None
    pats = [p for p in PATTERNS if not (scenario == "double_trigger" and p == "ringed")]
    pat_mode = rng.random()            # some scenes all solid, some mixed
    picked = rng.sample(combos, n)
    nodes = []
    for nid, (c, s) in zip(ids, picked):
        p = "solid" if pat_mode < 0.3 else rng.choice(pats)
        name = f"{c} {s}" if p == "solid" else f"{p} {c} {s}"
        nodes.append({"id": nid, "x": pos[nid][0], "y": pos[nid][1], "colour": c, "rgb": list(COLOURS[c]),
                      "shape": s, "pattern": p, "threshold": 1, "name": name})
    edges = [{"a": a, "b": b, "slow": False, "oneway": False} for a, b in elist]
    src = rng.choice(ids)
    if scenario == "hidden_slow":
        rng.choice(edges)["slow"] = True
    elif scenario == "marked_speeds":
        for e in rng.sample(edges, rng.randint(2, max(2, min(4, len(edges) // 3)))):
            e["slow"] = True
    elif scenario == "one_way":
        frac = rng.uniform(0.6, 1.0)
        for e in edges:
            if rng.random() < frac:
                e["oneway"] = True
                if rng.random() < 0.5:
                    e["a"], e["b"] = e["b"], e["a"]
        if not any(e["oneway"] for e in edges):
            edges[0]["oneway"] = True
    elif scenario == "double_trigger":
        cand = [x for x in ids if x != src]
        for x in rng.sample(cand, rng.randint(2, min(4, len(cand) // 2 + 1))):
            node = next(nd for nd in nodes if nd["id"] == x)
            node["threshold"] = 2
            node["pattern"] = "solid"                 # the double outline is its marking; keep the colour clear
            node["name"] = f"{node['colour']} {node['shape']}"
    scene = {"scenario": scenario, "layout": layout, "theme": theme, "R": R, "nodes": nodes,
             "edges": edges, "speed": 1.0, "starts": [[src, 0.0]]}
    # pick the speed so the whole cascade takes a target time
    probe = simulate(scene)
    latest = max([p["t1"] for p in probe["pulses"]] + list(probe["fire"].values()))
    target = rng.uniform(3.0 + 0.4 * (n - 6), 7.4)
    speed = round(latest / target, 2)
    if not 85 <= speed <= 420:
        return None
    scene["speed"] = speed
    t0 = round(rng.uniform(0.6, 1.3), 2)
    scene["starts"] = [[src, t0]]
    if scenario == "two_sources":
        first = simulate(scene)["fire"]
        other = rng.choice([x for x in ids if x != src])
        delay = rng.uniform(0.5, 0.8) * (first.get(other, t0 + 3) - t0)
        scene["starts"].append([other, round(t0 + delay, 2)])
    log = simulate(scene)
    last = max([p["t1"] for p in log["pulses"]] + list(log["fire"].values()))
    end = round(last + rng.uniform(0.7, 1.2), 2)
    if end > MAX_DUR:
        return None
    scene.update(log)
    scene["end_t"] = end
    return scene


def build(rng: random.Random) -> dict:
    import questions                                  # lazy: questions imports sim
    scenario = rng.choice(SCENARIOS)                 # fixed per seed, so the mix stays uniform
    n = rng.randint(6, 11)                            # likewise the node count
    tries = 0
    while True:
        tries += 1
        scene = _try(rng, scenario, n)
        if scene is None or check(scene) or not questions.pool_ok(scene):
            continue
        scene["tries"] = tries
        return scene


# ------------------------------------------------------------------- check
def check(scene: dict) -> list[str]:
    errs = []
    log = simulate(scene)
    for k in ("fire", "parent", "arrivals", "half", "pulses"):
        if log[k] != scene.get(k):
            errs.append(f"stored {k} doesn't match a replay")
    if scene["scenario"] not in SCENARIOS:
        errs.append("unknown scenario")
    nodes, R = scene["nodes"], scene["R"]
    pos = {n["id"]: (n["x"], n["y"]) for n in nodes}
    if len({n["name"] for n in nodes}) != len(nodes) or len({(n["colour"], n["shape"]) for n in nodes}) != len(nodes):
        errs.append("two nodes share a description")
    if R < 20 or scene["theme"]["link_w"] < 3:
        errs.append("objects or lines too small")
    for n in nodes:
        if not (AREA[0] <= n["x"] <= AREA[2] and AREA[1] <= n["y"] <= AREA[3]):
            errs.append(f"{n['id']} outside the play area")
    for a, b in itertools.combinations(pos, 2):
        if math.dist(pos[a], pos[b]) < 4 * R + 20:
            errs.append(f"{a} and {b} too close")
    pairs = [(e["a"], e["b"]) for e in scene["edges"]]
    if len({tuple(sorted(p)) for p in pairs}) != len(pairs):
        errs.append("duplicate link")
    for (a, b), (c, d) in itertools.combinations(pairs, 2):
        if seg_cross(pos[a], pos[b], pos[c], pos[d]):
            errs.append("two links cross")
    for a, b in pairs:
        if not _clear(pos, a, b, R):
            errs.append(f"link {a}-{b} passes too close to a node")
    if not _angles_ok(pos, pairs):
        errs.append("two links leave a node at too small an angle")
    comp = {k: {k} for k in pos}
    for a, b in pairs:
        u = comp[a] | comp[b]
        for k in u:
            comp[k] = u
    if len(comp[nodes[0]["id"]]) != len(nodes):
        errs.append("network not connected")
    for e in scene["edges"]:
        if travel(scene, e) < MIN_TRAVEL:
            errs.append("a pulse crosses a link too fast to see")
    sc = scene["scenario"]
    nslow = sum(e["slow"] for e in scene["edges"])
    if (sc == "hidden_slow" and nslow != 1) or (sc == "marked_speeds" and nslow < 2) or \
            (sc not in ("hidden_slow", "marked_speeds") and nslow):
        errs.append("wrong number of slow links")
    if (sc == "one_way") != any(e["oneway"] for e in scene["edges"]):
        errs.append("one-way links in the wrong scenario")
    if (sc == "two_sources") != (len(scene["starts"]) == 2) or len(scene["starts"]) not in (1, 2):
        errs.append("wrong number of starting flashes")
    ndbl = sum(n["threshold"] == 2 for n in nodes)
    if (sc == "double_trigger") != (ndbl >= 2):
        errs.append("double-trigger nodes in the wrong scenario")
    fire, end = scene["fire"], scene["end_t"]
    for nid, t in scene["starts"]:
        if fire.get(nid) != round(t, 6) or node_map(scene)[nid]["threshold"] != 1:
            errs.append(f"starting node {nid} was pre-empted")
    if end > MAX_DUR + 1e-9:
        errs.append("clip longer than 10 s")
    if len(fire) < 5:
        errs.append("fewer than five nodes light up")
    times = sorted(fire.values())
    if any(b - a < MIN_SEP for a, b in zip(times, times[1:])):
        errs.append("two nodes light up too close together")
    events = list(times) + [t for arr in scene["arrivals"].values() for t, _ in arr]
    if events and (min(events) < PAD or max(events) > end - PAD):
        errs.append("an event is too close to the clip start or end")
    for k, arr in scene["arrivals"].items():
        ts = [t for t, _ in arr]
        if any(b - a < MARGIN for a, b in zip(ts, ts[1:])):
            errs.append(f"two pulses reach {k} too close together")
    return errs


def signature(scene: dict) -> str:
    """Hash input for the diversity check: rounded scene parameters."""
    import json
    key = {"scenario": scene["scenario"], "theme": scene["theme"]["name"], "R": scene["R"],
           "speed": round(scene["speed"], 1),
           "nodes": [[round(n["x"]), round(n["y"]), n["colour"], n["shape"], n["pattern"], n["threshold"]]
                     for n in scene["nodes"]],
           "edges": sorted([e["a"], e["b"], e["slow"], e["oneway"]] for e in scene["edges"]),
           "starts": [[s, round(t, 1)] for s, t in scene["starts"]]}
    return json.dumps(key, sort_keys=True)
