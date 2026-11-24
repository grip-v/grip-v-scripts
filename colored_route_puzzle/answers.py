"""Independent answers for colored_route_puzzle.

Never reads the generator's timing or graph fields (draw / erase windows,
route end ids, grow flags, ride lists, node paths). Everything is measured
from what the renderer draws:

  * render.track(scene, i) for every frame i: the drawn polyline of every
    route (in pixels, starting at the end it grows from) and the ring centre;
  * render.stop_marks(scene): the shape and pixel centre of every stop;
  * render.timer_text(i): the clock text on frame i.

From those, `extract` builds a timeline: per route its drawn length on every
frame, from which the start (last empty frame), finish (first full frame),
erase start (last full frame) and gone (first empty frame again) are read;
its two ends (a stop marker centre, or a dead end) and the end it grows
from; its corner count and pixel length; and for the ring, the stops it
rests on and the route it is on between them. `derive` answers every
question from such a timeline alone, re-simulating counterfactuals on it.
frame_check.py builds the same timeline from decoded video frames and calls
`derive` too.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render  # noqa: E402

SHAPE_ORDER = ("circle", "square", "triangle", "diamond", "star", "cross")
DEAD_END = "dead end"
NEVER = "it would never become possible"
NO_WAY = "it could not get there"
DEAD_TXT = "its dead end (the crossbar)"
EPS = 1e-6


def _plen(pts) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def _corners(pts) -> int:
    dirs = []
    for a, b in zip(pts, pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        if abs(dx) < EPS and abs(dy) < EPS:
            continue
        d = ((dx > EPS) - (dx < -EPS), (dy > EPS) - (dy < -EPS))
        if not dirs or dirs[-1] != d:
            dirs.append(d)
    return max(0, len(dirs) - 1)


def _seg_dist(p, a, b) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    L2 = vx * vx + vy * vy
    u = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L2))
    return math.hypot(p[0] - a[0] - u * vx, p[1] - a[1] - u * vy)


def on_path(p, pts, tol) -> bool:
    return any(_seg_dist(p, a, b) <= tol for a, b in zip(pts, pts[1:]))


def stop_at(stops, p, tol):
    for sh, (x, y) in stops.items():
        if math.hypot(p[0] - x, p[1] - y) <= tol:
            return sh
    return None


def events_from_lengths(lens, full):
    """(start, finish, erase_start, gone) from a per-frame drawn-length series."""
    nf = len(lens)
    first = next(i for i in range(nf) if lens[i] > EPS)
    e = next(i for i in range(nf) if lens[i] >= full - EPS)
    later = [i for i in range(e, nf) if lens[i] < full - EPS]
    g0 = later[0] - 1 if later else None
    g1 = next((i for i in range(later[0], nf) if lens[i] <= EPS), None) if later else None
    return first - 1, e, g0, g1


_CACHE: dict = {}


def extract(scene) -> dict:
    """Timeline of what the renderer draws (see module docstring)."""
    hit = _CACHE.get(id(scene))
    if hit is not None and hit[0] is scene:
        return hit[1]
    nf = scene["n_frames"]
    stops = {m["shape"]: (m["x"], m["y"]) for m in render.stop_marks(scene)}
    tracks = [render.track(scene, i) for i in range(nf)]
    routes = {}
    for k, r0 in enumerate(tracks[0]["routes"]):
        lens = [_plen(tr["routes"][k]["pts"]) for tr in tracks]
        full = max(lens)
        s, e, g0, g1 = events_from_lengths(lens, full)
        path = tracks[e]["routes"][k]["pts"]
        anchor = stop_at(stops, path[0], 0.5) or DEAD_END
        far = stop_at(stops, path[-1], 0.5) or DEAD_END
        routes[r0["color"]] = {"style": r0["style"], "path": path, "anchor": anchor, "far": far,
                               "len": full, "corners": _corners(path), "s": s, "e": e, "g0": g0, "g1": g1}
    token = None
    if tracks[0]["token"] is not None:
        token = token_rides([tr["token"] for tr in tracks], stops, routes, 0.5)
    tl = {"nf": nf, "stops": stops, "routes": routes, "token": token,
          "route_ids": [r["color"] for r in tracks[0]["routes"]],
          "stop_ids": [m["shape"] for m in render.stop_marks(scene)]}
    _CACHE.clear()
    _CACHE[id(scene)] = (scene, tl)
    return tl


def token_rides(pos, stops, routes, tol):
    """Ring start stop and rides [(route colour, from, to, t0 = last frame resting, t1 = first frame resting)]."""
    at = [stop_at(stops, p, tol) if p is not None else None for p in pos]
    runs = []
    for i, s in enumerate(at):
        if s is None:
            continue
        if runs and runs[-1][0] == s and runs[-1][2] == i - 1:
            runs[-1][2] = i
        else:
            runs.append([s, i, i])
    rides = []
    for a, b in zip(runs, runs[1:]):
        mid = pos[(a[2] + b[1]) // 2]
        on = [c for c, r in routes.items() if on_path(mid, r["path"], max(tol, 2.0))]
        rides.append({"route": on[0] if len(on) == 1 else None, "from": a[0], "to": b[0], "t0": a[2], "t1": b[1]})
    return {"start": runs[0][0], "rides": rides}


# ---------------------------------------------------------------- timeline queries

def comp(r, i) -> bool:
    return r["e"] <= i and (r["g0"] is None or i <= r["g0"])


def growing(r, i) -> bool:
    return r["s"] < i < r["e"]


def ends(r):
    return {r["anchor"], r["far"]} - {DEAD_END}


def is_link(r) -> bool:
    return r["anchor"] != DEAD_END and r["far"] != DEAD_END


def reach(tl, colours, src) -> set:
    adj = {s: set() for s in tl["stops"]}
    for c in colours:
        r = tl["routes"][c]
        if is_link(r):
            adj[r["anchor"]].add(r["far"])
            adj[r["far"]].add(r["anchor"])
    seen, todo = {src}, [src]
    while todo:
        for y in adj[todo.pop()]:
            if y not in seen:
                seen.add(y)
                todo.append(y)
    return seen


def complete_set(tl, i) -> list:
    return [c for c, r in tl["routes"].items() if comp(r, i)]


def frames_of(tl, t) -> list:
    txt = f"{t // 10}.{t % 10} s"
    return [i for i in range(tl["nf"]) if render.timer_text(i) == txt]


def same_at(tl, frames, f):
    vals = {repr(f(i)) for i in frames}
    if not frames or len(vals) != 1:
        return None
    return f(frames[0])


def start_order(tl) -> list:
    return sorted(tl["routes"], key=lambda c: tl["routes"][c]["s"])


def L(c) -> str:
    return f"the {c} route"


def S(sh) -> str:
    return f"the {sh} stop"


def pair_txt(a, b) -> str:
    a, b = sorted((a, b), key=SHAPE_ORDER.index)
    return f"{S(a)} and {S(b)}"


def first_link(tl, a, b, skip=None):
    for i in range(tl["nf"]):
        cs = [c for c in complete_set(tl, i) if c != skip]
        if b in reach(tl, cs, a):
            done = [c for c in cs if tl["routes"][c]["e"] == i]
            return done[0] if len(done) == 1 else None
    return NEVER


def bfs_rides(tl, colours, a, b):
    adj = {s: set() for s in tl["stops"]}
    for c in colours:
        r = tl["routes"][c]
        if is_link(r):
            adj[r["anchor"]].add(r["far"])
            adj[r["far"]].add(r["anchor"])
    d, todo = {a: 0}, [a]
    while todo:
        nxt = []
        for x in todo:
            for y in adj[x]:
                if y not in d:
                    d[y] = d[x] + 1
                    nxt.append(y)
        todo = nxt
    return d.get(b)


# ---------------------------------------------------------------- answers

def derive(key: str, p: dict, tl: dict):
    R, nf = tl["routes"], tl["nf"]
    rid = lambda j: tl["route_ids"][j]  # noqa: E731
    sid = lambda j: tl["stop_ids"][j]  # noqa: E731
    order = start_order(tl)
    last = nf - 1
    tk = tl["token"]

    if key == "n_routes":
        return len(R)
    if key == "n_stops":
        return len(tl["stops"])
    if key == "n_end":
        return len(complete_set(tl, last))
    if key == "n_style":
        return sum(1 for r in R.values() if r["style"] == p["style"])
    if key == "first_drawn":
        return L(order[0])
    if key == "last_started":
        return L(order[-1])
    if key == "erased_which":
        hits = [rid(j) for j in p["options"] if R[rid(j)]["g0"] is not None]
        return L(hits[0]) if len(hits) == 1 else None
    if key == "n_dead":
        return sum(1 for r in R.values() if not is_link(r))
    if key == "token_start":
        return S(tk["start"])
    if key == "token_end":
        return S(tk["rides"][-1]["to"])
    if key == "route_ends":
        r = R[rid(p["route"])]
        return pair_txt(r["anchor"], r["far"]) if is_link(r) else None
    if key == "grow_from":
        r = R[rid(p["route"])]
        return DEAD_TXT if r["anchor"] == DEAD_END else S(r["anchor"])
    if key == "earlier_of":
        return L(min((rid(j) for j in p["options"]), key=lambda c: R[c]["s"]))
    if key == "next_start":
        k = order.index(rid(p["route"]))
        return L(order[k + 1])
    if key == "first_complete":
        return L(min((rid(j) for j in p["options"]), key=lambda c: R[c]["e"]))
    if key == "degree":
        return sum(1 for c in complete_set(tl, last) if sid(p["stop"]) in ends(R[c]))
    if key == "pair_count":
        want = {sid(p["pair"][0]), sid(p["pair"][1])}
        return sum(1 for c in complete_set(tl, last) if is_link(R[c]) and {R[c]["anchor"], R[c]["far"]} == want)
    if key == "token_ride":
        return L(tk["rides"][p["k"] - 1]["route"])
    if key == "token_dest":
        return S(tk["rides"][p["k"] - 1]["to"])
    if key == "longest":
        return L(max(R, key=lambda c: R[c]["len"]))
    if key == "corners":
        return R[rid(p["route"])]["corners"]
    if key == "finish_clock":
        return render.timer_text(R[rid(p["route"])]["e"])
    if key == "drawing_at":
        fr = frames_of(tl, p["t"])
        hits = [rid(j) for j in p["options"] if all(growing(R[rid(j)], i) for i in fr)]
        return L(hits[0]) if len(hits) == 1 else None
    if key == "n_complete_at":
        return same_at(tl, frames_of(tl, p["t"]), lambda i: len(complete_set(tl, i)))
    if key == "n_growing_at":
        return same_at(tl, frames_of(tl, p["t"]), lambda i: sum(1 for r in R.values() if growing(r, i)))
    if key == "slowest":
        return L(max(R, key=lambda c: R[c]["e"] - R[c]["s"]))
    if key == "erase_clock":
        return render.timer_text(R[rid(p["route"])]["g1"])
    if key == "arrive_clock":
        hits = [rd for rd in tk["rides"] if rd["to"] == sid(p["stop"])]
        return render.timer_text(hits[0]["t1"]) if len(hits) == 1 else None
    if key == "connect_by":
        x = first_link(tl, sid(p["pair"][0]), sid(p["pair"][1]))
        return None if x in (None, NEVER) else L(x)
    if key == "n_before_style":
        x = next(c for c in order if R[c]["style"] == p["style"])
        return len(complete_set(tl, R[x]["s"]))
    if key == "first_at_stop":
        return L(next(c for c in order if sid(p["stop"]) in ends(R[c])))
    if key == "kth_start":
        return L(order[p["k"] - 1])
    if key == "reach_at":
        return same_at(tl, frames_of(tl, p["t"]), lambda i: len(reach(tl, complete_set(tl, i), sid(p["stop"]))) - 1)
    if key == "token_n_stops":
        return len({tk["start"]} | {rd["to"] for rd in tk["rides"]})
    if key == "token_corners":
        return sum(R[rd["route"]]["corners"] for rd in tk["rides"])
    if key == "max_complete":
        return max(len(complete_set(tl, i)) for i in range(nf))
    if key == "cf_skip_connect":
        x = first_link(tl, sid(p["pair"][0]), sid(p["pair"][1]), skip=rid(p["route"]))
        return None if x is None else (NEVER if x == NEVER else L(x))
    if key == "cf_reverse":
        rev = order[::-1]
        for k in range(len(rev)):
            if sid(p["pair"][1]) in reach(tl, rev[:k + 1], sid(p["pair"][0])):
                return L(rev[k])
        return None
    if key == "cf_remove_reach":
        cs = [c for c in complete_set(tl, last) if c != rid(p["route"])]
        return len(reach(tl, cs, sid(p["stop"]))) - 1
    if key == "cf_first":
        x = rid(p["route"])
        new = [x] + [c for c in order if c != x]
        return L(new[p["k"] - 1])
    if key == "cf_no_erase":
        cs = complete_set(tl, last) + [rid(p["route"])]
        return len(reach(tl, cs, sid(p["stop"]))) - 1
    if key == "cf_detour":
        cs = [c for c in complete_set(tl, last) if c != rid(p["route"])]
        k = bfs_rides(tl, cs, tk["start"], tk["rides"][-1]["to"])
        return NO_WAY if k is None else str(k)
    if key == "cf_style_gone":
        rest = [r for r in R.values() if r["style"] != p["style"]]
        return sum(1 for s in tl["stops"] if not any(s in ends(r) for r in rest))
    return None


def answer(key: str, p: dict, scene: dict):
    return derive(key, p, extract(scene))
