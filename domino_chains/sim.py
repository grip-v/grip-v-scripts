"""Branching domino chains, seen from above. Lettered dominoes sit at the
junctions of a left-to-right network of lanes; each lane is a straight row
of plain dominoes. One push knocks over A, and every falling domino
topples the next one in its lane if it stands within REACH. A lane can
fork (a junction domino starts every lane leaving it), two lanes can merge
into one junction (whichever chain arrives first knocks it over; the late
one just stops), and a "dud gap" (one missing domino, so the gap is about
twice the spacing) stops a lane dead.

Every link takes distance / SPEED seconds, so fall times are an exact
shortest-path computation over a directed graph of dominoes."""
from __future__ import annotations
import heapq
import math
import random

COLS, ROWS = 5, 3
SPACING = 0.5
REACH = 0.75
SPEED = 2.0                 # board units per second along a lane
PUSH_T = 1.0
LETTERS = "ABCDEFGHIJKL"


def node_xy(col: int, row: int) -> tuple[float, float]:
    return 1.0 + 3.5 * col, 1.2 + 2.8 * row


def _crosses(e1, e2) -> bool:
    (c1, r1), (_, r2) = e1
    (c3, r3), (_, r4) = e2
    return c1 == c3 and (r1 - r3) * (r2 - r4) < 0


def links(scene: dict, removed: str | None = None, filled: int | None = None) -> dict:
    """Directed links {domino id: [(next id, seconds)]}. `removed` takes a
    lettered domino off the board; `filled` puts the missing domino back
    into that lane's gap."""
    dom = {d["id"]: d for d in scene["dominoes"]}
    out = {i: [] for i in dom if i != removed}
    for k, lane in enumerate(scene["lanes"]):
        ids = [lane["u"]] + list(lane["ids"]) + [lane["v"]]
        pts = [(dom[i]["x"], dom[i]["y"]) for i in ids]
        if filled == k and lane["gap"] is not None:
            g = lane["gap"]                             # the gap sits between ids[g] and ids[g + 1]
            mid = ((pts[g][0] + pts[g + 1][0]) / 2, (pts[g][1] + pts[g + 1][1]) / 2)
            ids = ids[:g + 1] + ["fill"] + ids[g + 1:]
            pts = pts[:g + 1] + [mid] + pts[g + 1:]
            out.setdefault("fill", [])
        for (a, pa), (b, pb) in zip(zip(ids, pts), list(zip(ids, pts))[1:]):
            if a == removed or b == removed:
                continue
            dist = math.dist(pa, pb)
            if dist <= REACH:
                out.setdefault(a, []).append((b, dist / SPEED))
    return out


def fall_times(scene: dict, removed: str | None = None, filled: int | None = None) -> dict:
    g = links(scene, removed, filled)
    start = "A"
    if start == removed:
        return {}
    best = {start: PUSH_T}
    pq = [(PUSH_T, start)]
    while pq:
        t, u = heapq.heappop(pq)
        if t > best.get(u, 1e9):
            continue
        for v, dt in g.get(u, []):
            if t + dt < best.get(v, 1e9) - 1e-12:
                best[v] = t + dt
                heapq.heappush(pq, (t + dt, v))
    return {k: round(v, 6) for k, v in best.items()}


def triggers(scene: dict, fall: dict) -> dict:
    """For each fallen lettered domino, the lane that knocked it over and
    the arrival time of every lane that reached it."""
    dom = {d["id"]: d for d in scene["dominoes"]}
    res = {}
    for k, lane in enumerate(scene["lanes"]):
        last = lane["ids"][-1] if lane["ids"] else lane["u"]
        if last in fall:
            dist = math.dist((dom[last]["x"], dom[last]["y"]), (dom[lane["v"]]["x"], dom[lane["v"]]["y"]))
            if dist <= REACH:
                res.setdefault(lane["v"], []).append((round(fall[last] + dist / SPEED, 6), lane["u"], k))
    return {v: sorted(a) for v, a in res.items()}


def build(rng: random.Random) -> dict:
    want_merge = rng.random() < 0.6
    while True:
        cols = [[rng.randrange(ROWS)]]
        for c in range(1, COLS):
            cols.append(sorted(rng.sample(range(ROWS), rng.choice([1, 2, 2, 3, 3]))))
        edges = []
        for c in range(COLS - 1):
            for r in cols[c]:
                nxt = [r2 for r2 in cols[c + 1] if abs(r2 - r) <= 1]
                rng.shuffle(nxt)
                for r2 in nxt[:1 + (rng.random() < 0.4)]:
                    edges.append(((c, r), (c + 1, r2)))
            for r2 in cols[c + 1]:
                if not any(e[1] == (c + 1, r2) for e in edges):
                    src = [r for r in cols[c] if abs(r - r2) <= 1]
                    if src:
                        edges.append(((c, rng.choice(src)), (c + 1, r2)))
        nodes = sorted({e[0] for e in edges} | {e[1] for e in edges})
        if any(_crosses(a, b) for a in edges for b in edges if a < b):
            continue
        if not (6 <= len(nodes) <= 10):
            continue
        name = {n: LETTERS[i] for i, n in enumerate(nodes)}
        dominoes = [{"id": name[n], "x": node_xy(*n)[0], "y": node_xy(*n)[1], "label": name[n]} for n in nodes]
        lanes = []
        n_gap = rng.choice([1, 1, 2])
        gap_lanes = set(rng.sample(range(len(edges)), min(n_gap, len(edges) - 1)))
        for k, (a, b) in enumerate(sorted(edges)):
            (x0, y0), (x1, y1) = node_xy(*a), node_xy(*b)
            m = round(math.dist((x0, y0), (x1, y1)) / SPACING)
            pts = [(x0 + (x1 - x0) * i / m, y0 + (y1 - y0) * i / m) for i in range(1, m)]
            gap = None
            if k in gap_lanes:
                g = rng.randrange(2, len(pts) - 2)
                pts = pts[:g] + pts[g + 1:]
                gap = g                              # between chain positions g and g+1 (0 = the junction u)
            ids = []
            for i, (x, y) in enumerate(pts):
                did = f"{name[a]}{name[b]}{i}"
                dominoes.append({"id": did, "x": round(x, 4), "y": round(y, 4), "label": None})
                ids.append(did)
            lanes.append({"u": name[a], "v": name[b], "ids": ids, "gap": gap,
                          "dir": [round((x1 - x0) / math.dist((x0, y0), (x1, y1)), 6),
                                  round((y1 - y0) / math.dist((x0, y0), (x1, y1)), 6)]})
        scene = {"dominoes": dominoes, "lanes": lanes}
        fall = fall_times(scene)
        letters = [d["id"] for d in dominoes if d["label"]]
        fallen = [x for x in letters if x in fall]
        if len(fallen) < 4:
            continue
        trig = triggers(scene, fall)
        if any(len(a) > 1 and a[1][0] - a[0][0] < 0.35 for a in trig.values()):
            continue
        if want_merge and not any(len(a) > 1 for a in trig.values()):
            continue
        ft = sorted(fall[x] for x in fallen)
        if any(b - a < 0.2 for a, b in zip(ft, ft[1:])):
            continue
        if not any(abs(fall[x] * 2 - round(fall[x] * 2)) <= 0.3 for x in fallen if x != "A"):
            continue
        scene["fall"] = fall
        scene["end_t"] = round(max(fall.values()) + 1.6, 3)
        return scene


def fall_dir(scene: dict) -> dict:
    """Direction each fallen domino topples: along its lane; a junction
    domino topples along the lane that knocked it over (A: to the right)."""
    out = {"A": [1.0, 0.0]}
    for lane in scene["lanes"]:
        for i in lane["ids"]:
            out[i] = lane["dir"]
    for v, arr in triggers(scene, scene["fall"]).items():
        out[v] = scene["lanes"][arr[0][2]]["dir"]
    return out


def check(scene: dict) -> list[str]:
    errs = []
    if fall_times(scene) != scene["fall"]:
        errs.append("stored fall times don't match a replay")
    dom = {d["id"]: d for d in scene["dominoes"]}
    for lane in scene["lanes"]:
        ids = [lane["u"]] + lane["ids"] + [lane["v"]]
        gaps = [math.dist((dom[a]["x"], dom[a]["y"]), (dom[b]["x"], dom[b]["y"])) for a, b in zip(ids, ids[1:])]
        big = [g for g in gaps if g > REACH]
        if (lane["gap"] is None) != (not big) or any(0.6 < g < 0.9 for g in gaps):
            errs.append(f"lane {lane['u']}->{lane['v']} has an ambiguous or unexpected gap")
    return errs
