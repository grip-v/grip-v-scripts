"""Independent answers for domino_chains. Rebuilds each lane's chain from
the domino coordinates (ordering dominoes by their projection onto the
lane), then finds fall times by Bellman-Ford relaxation instead of the
generator's Dijkstra, and answers from those times."""
from __future__ import annotations
import math

REACH = 0.75
SPEED = 2.0
PUSH_T = 1.0


def _chains(scene, removed=None, filled=None):
    pos = {d["id"]: (d["x"], d["y"]) for d in scene["dominoes"]}
    for k, lane in enumerate(scene["lanes"]):
        u, v = pos[lane["u"]], pos[lane["v"]]
        dx, dy = v[0] - u[0], v[1] - u[1]
        members = [lane["u"], lane["v"]] + list(lane["ids"])
        pts = {m: pos[m] for m in members}
        if filled == k:
            order = sorted(members, key=lambda m: (pts[m][0] - u[0]) * dx + (pts[m][1] - u[1]) * dy)
            for a, b in zip(order, order[1:]):
                if math.dist(pts[a], pts[b]) > REACH:
                    pts["gapfill"] = ((pts[a][0] + pts[b][0]) / 2, (pts[a][1] + pts[b][1]) / 2)
                    members.append("gapfill")
                    break
        members = [m for m in members if m != removed]
        members.sort(key=lambda m: (pts[m][0] - u[0]) * dx + (pts[m][1] - u[1]) * dy)
        yield k, lane, [(m, pts[m]) for m in members]


def times(scene, removed=None, filled=None):
    edges = []
    for _, _, chain in _chains(scene, removed, filled):
        for (a, pa), (b, pb) in zip(chain, chain[1:]):
            if math.dist(pa, pb) <= REACH:
                edges.append((a, b, math.dist(pa, pb) / SPEED))
    t = {"A": PUSH_T} if removed != "A" else {}
    for _ in range(len(scene["dominoes"]) + 2):
        changed = False
        for a, b, dt in edges:
            if a in t and t[a] + dt < t.get(b, 1e9) - 1e-12:
                t[b] = t[a] + dt
                changed = True
        if not changed:
            break
    return t


def _letters(scene):
    return [d["id"] for d in scene["dominoes"] if d["label"]]


def _trigger(scene, t, x):
    best = None
    for _, lane, chain in _chains(scene):
        if lane["v"] != x:
            continue
        (a, pa), (b, pb) = chain[-2], chain[-1]
        if a in t and math.dist(pa, pb) <= REACH:
            arr = t[a] + math.dist(pa, pb) / SPEED
            if best is None or arr < best[0]:
                best = (arr, lane["u"])
    return best[1]


def answer(key: str, p: dict, scene: dict):
    t = times(scene)
    fallen = [x for x in _letters(scene) if x in t]
    if key == "n_fallen":
        return len(fallen)
    if key == "n_standing":
        return len(_letters(scene)) - len(fallen)
    if key == "fall_before":
        return min(p["pair"], key=lambda x: t[x])
    if key == "next_after":
        later = [x for x in fallen if t[x] > t[p["x"]]]
        return min(later, key=lambda x: t[x])
    if key == "fall_time":
        return f"{round(t[p['x']] * 2) / 2:.1f} s"
    if key == "trigger":
        return _trigger(scene, t, p["x"])
    if key == "path":
        path = [p["x"]]
        while path[-1] != "A":
            path.append(_trigger(scene, t, path[-1]))
        return " → ".join(reversed(path))
    if key == "cf_remove":
        t2 = times(scene, removed=p["x"])
        return sum(1 for x in _letters(scene) if x in t2)
    if key == "cf_fill":
        t2 = times(scene, filled=p["lane"])
        return sum(1 for x in _letters(scene) if x in t2)
    return None
