"""5-rung ladder for moving_points (see common/ladder5.py). Points are
named by their letters; times are read on the clock; coordinates are
whole numbers at every even second. Times are only asked away from a
rounding boundary, and comparisons only with a margin."""
from __future__ import annotations
import itertools
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402


def _xy(x, y):
    f = lambda v: f"{v:g}"
    return f"({f(x)}, {f(y)})"


def _safe_half(t):
    f = (t * 2) % 1
    return abs(f - 0.5) >= 0.2


def build(scene: dict, rng: random.Random) -> list[dict]:
    ps, T = scene["points"], scene["T"]
    nm = [p["name"] for p in ps]
    idx = list(range(len(ps)))
    qs = []

    # L1 observation
    quads = [i for i in idx if sim.quadrant(*sim.pos(ps[i], T))]
    key = rng.choice(["fastest", "fastest"] + (["quadrant_end"] if quads else []))
    if key == "fastest":
        i = max(idx, key=lambda i: sim.speed(ps[i]))
        opts, a = L.mc(rng, f"point {nm[i]}", [f"point {nm[k]}" for k in idx if k != i])
        qs.append(L.question(1, key, "Which point moves the fastest?", f"point {nm[i]}", opts, a, True))
    else:
        i = rng.choice(quads)
        gt = sim.quadrant(*sim.pos(ps[i], T))
        opts, a = L.mc(rng, gt, ["top right", "top left", "bottom left", "bottom right"])
        qs.append(L.question(1, key, f"At the end of the video, in which quarter of the plane is point {nm[i]}?", gt,
                             opts, a, True, {"x": i}))

    # L2 relational
    dists = {i: math.hypot(*sim.pos(ps[i], T)) for i in idx}
    near = [(x, y) for x, y in itertools.combinations(idx, 2) if abs(dists[x] - dists[y]) >= 0.8]
    yc = {i: sim.axis_cross(ps[i], "y", T) for i in idx}
    ycp = [(x, y) for x, y in itertools.combinations(idx, 2) if yc[x] and yc[y] and abs(yc[x] - yc[y]) >= 0.6]
    keys = (["closer_origin"] * 2 if near else []) + (["y_cross_first"] if ycp else [])
    key = rng.choice(keys or ["fastest2"])
    if key == "closer_origin":
        x, y = rng.choice(near)
        gt = f"point {nm[min((x, y), key=dists.get)]}"
        opts, a = L.mc(rng, gt, [f"point {nm[x]}", f"point {nm[y]}"])
        qs.append(L.question(2, key, f"At the end, which point is closer to the origin (0, 0): point {nm[x]} or point "
                             f"{nm[y]}?", gt, opts, a, True, {"x": x, "y": y}))
    elif key == "y_cross_first":
        x, y = rng.choice(ycp)
        gt = f"point {nm[min((x, y), key=yc.get)]}"
        opts, a = L.mc(rng, gt, [f"point {nm[x]}", f"point {nm[y]}"])
        qs.append(L.question(2, key, f"Both point {nm[x]} and point {nm[y]} cross the vertical axis. Which crosses it "
                             "first?", gt, opts, a, True, {"x": x, "y": y}))
    else:
        x, y = rng.sample(idx, 2)
        gt = f"point {nm[max((x, y), key=lambda i: sim.speed(ps[i]))]}"
        opts, a = L.mc(rng, gt, [f"point {nm[x]}", f"point {nm[y]}"])
        qs.append(L.question(2, "faster", f"Which moves faster: point {nm[x]} or point {nm[y]}?", gt, opts, a, True,
                             {"x": x, "y": y}))

    # L3 measurement
    key = rng.choice(["pos_at", "pos_at", "dist_end"])
    if key == "pos_at":
        i = rng.choice(idx)
        t = rng.choice(list(range(2, T + 1, 2)))
        x, y = sim.pos(ps[i], t)
        gt = _xy(x, y)
        wrong = [_xy(y, x), _xy(x + ps[i]["vx"] * 2, y + ps[i]["vy"] * 2), _xy(ps[i]["x"], ps[i]["y"]),
                 _xy(x - ps[i]["vx"] * 2, y - ps[i]["vy"] * 2), _xy(-x, y), _xy(x + 1, y - 1)]
        opts, a = L.mc(rng, gt, wrong)
        qs.append(L.question(3, key, f"Where is point {nm[i]} when the clock reads {t}.00 s?", gt, opts, a, True,
                             {"x": i, "t": t}))
    else:
        cands = [(x, y) for x, y in itertools.combinations(idx, 2)
                 if abs(math.dist(sim.pos(ps[x], T), sim.pos(ps[y], T)) % 1 - 0.5) >= 0.15]
        x, y = rng.choice(cands) if cands else (0, 1)
        gt = round(math.dist(sim.pos(ps[x], T), sim.pos(ps[y], T)))
        opts, a = L.mc(rng, gt, L.near_ints(gt, 0))
        qs.append(L.question(3, "dist_end", f"At the end, how far apart are point {nm[x]} and point {nm[y]}, to the "
                             "nearest whole unit?", gt, opts, a, True, {"x": x, "y": y}))

    # L4 integration
    cl = [(x, y, sim.closest(ps[x], ps[y])) for x, y in itertools.combinations(idx, 2)]
    cl = [(x, y, t) for x, y, t in cl if t is not None and 0.75 <= t <= T - 0.75 and _safe_half(t)]
    clean = [i for i in idx if sim.axis_clean(ps[i], T)]
    key = rng.choice((["closest_time"] * 2 if cl else []) + (["n_axis_cross"] if clean else []) or ["closest_time"])
    if key == "closest_time":
        x, y, t = rng.choice(cl)
        v = round(t * 2) / 2
        gt = f"{v:.1f} s"
        opts, a = L.mc(rng, gt, [f"{v + d:.1f} s" for d in (0.5, -0.5, 1.0, -1.0, 1.5) if 0 <= v + d <= T])
        qs.append(L.question(4, key, f"At what time on the clock are point {nm[x]} and point {nm[y]} closest together "
                             "(to the nearest 0.5 s)?", gt, opts, a, True, {"x": x, "y": y}))
    else:
        i = rng.choice(clean)
        gt = sum(1 for ax in ("x", "y") if sim.axis_cross(ps[i], ax, T))
        opts, a = L.mc(rng, gt, [0, 1, 2, 3])
        qs.append(L.question(4, "n_axis_cross", f"How many times does point {nm[i]} cross one of the two axes during "
                             "the video?", gt, opts, a, True, {"x": i}))

    # L5 counterfactual: one point at half speed
    i = rng.choice(idx)
    half = dict(ps[i], vx=ps[i]["vx"] / 2, vy=ps[i]["vy"] / 2)
    cands = [("cf_half_pos", i, None)]
    for k in idx:
        if k != i:
            t = sim.closest(ps[k], half)
            if t is not None and 0.75 <= t <= T - 0.75 and _safe_half(t):
                cands += [("cf_half_closest", i, k)] * 2
    kind, i, k = rng.choice(cands)
    if kind == "cf_half_pos":
        x, y = sim.pos(half, T)
        gt = _xy(x, y)
        fx, fy = sim.pos(ps[i], T)
        opts, a = L.mc(rng, gt, [_xy(fx, fy), _xy(ps[i]["x"], ps[i]["y"]), _xy(y, x), _xy(x + 1, y), _xy(x, y - 1)])
        qs.append(L.question(5, kind, f"If point {nm[i]} had moved along the same line at half its speed, where would "
                             "it be at the end of the video?", gt, opts, a, True, {"x": i}))
    else:
        v = round(sim.closest(ps[k], half) * 2) / 2
        gt = f"{v:.1f} s"
        opts, a = L.mc(rng, gt, [f"{v + d:.1f} s" for d in (0.5, -0.5, 1.0, -1.0, 1.5) if 0 <= v + d <= T])
        qs.append(L.question(5, kind, f"If point {nm[i]} had moved along the same line at half its speed, at what "
                             f"time would it be closest to point {nm[k]} (to the nearest 0.5 s)?", gt, opts, a, True,
                             {"x": i, "y": k}))
    return qs
