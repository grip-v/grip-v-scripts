"""5-rung ladder for phantom_traffic_jams (see common/ladder5.py). The four
coloured cars are named by colour ("the red car"); grey cars are only ever
counted. A "brake tap" is a coloured car braking to a stop by itself (its
brake lights flash); any other stop is a car caught in a jam."""
from __future__ import annotations
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402


def _secs(k):
    return f"{k * sim.STEP_S:.1f} s"


def build(scene: dict, rng: random.Random) -> list[dict]:
    col = sim.colored(scene)
    names = sim.COLORS
    car = {c: f"the {c} car" for c in names}
    st = {c: sim.stops(scene, col[c]) for c in names}
    first = {c: st[c][0][0] for c in names if st[c]}
    total = {c: sum(b for _, b in st[c]) for c in names}
    dist = {c: sim.distance(scene, col[c]) for c in names}
    stopped_all = sim.ever_stopped(scene)
    n = len(scene["pos"])
    tappers = [scene["colors"][t["car"]] for t in scene["taps"]]
    qs = []

    # L1 observation
    key = rng.choice(["n_taps", "stops_yesno", "stops_yesno", "n_col_stopped"])
    if key == "n_taps":
        gt = len(scene["taps"])
        opts, a = L.mc(rng, gt, L.near_ints(gt, 0, 4))
        qs.append(L.question(1, key, "How many times does a car flash its brake lights and brake to a stop by "
                             "itself?", gt, opts, a, True))
    elif key == "stops_yesno":
        c = rng.choice(names)
        gt = "yes" if st[c] else "no"
        opts, a = L.yes_no(rng, gt)
        qs.append(L.question(1, key, f"Does {car[c]} ever come to a complete stop?", gt, opts, a, True, {"c": c}))
    else:
        gt = len(first)
        opts, a = L.mc(rng, gt, L.near_ints(gt, 0, 4))
        qs.append(L.question(1, key, "How many of the four coloured cars come to a complete stop at some point?",
                             gt, opts, a, True))

    # L2 relational
    fpairs = [(x, y) for x, y in itertools.combinations(first, 2) if abs(first[x] - first[y]) >= 2]
    dpairs = [(x, y) for x, y in itertools.combinations(names, 2) if abs(dist[x] - dist[y]) >= 3]
    keys = (["first_stop"] * 2 if fpairs else []) + (["farther"] if dpairs else [])
    key = rng.choice(keys or ["stops_yesno"])
    if key == "first_stop":
        x, y = rng.choice(fpairs)
        if rng.random() < 0.5:
            x, y = y, x
        gt = car[x] if first[x] < first[y] else car[y]
        opts, a = L.mc(rng, gt, [car[x], car[y]])
        qs.append(L.question(2, key, f"Which comes to a complete stop first: {car[x]} or {car[y]}?", gt, opts, a,
                             True, {"x": x, "y": y}))
    elif key == "farther":
        x, y = rng.choice(dpairs)
        if rng.random() < 0.5:
            x, y = y, x
        gt = car[x] if dist[x] > dist[y] else car[y]
        opts, a = L.mc(rng, gt, [car[x], car[y]])
        qs.append(L.question(2, key, f"Which travels farther during the video: {car[x]} or {car[y]}?", gt, opts, a,
                             True, {"x": x, "y": y}))
    else:
        c = rng.choice(names)
        gt = "yes" if st[c] else "no"
        opts, a = L.yes_no(rng, gt)
        qs.append(L.question(2, "stops_yesno", f"Does {car[c]} ever come to a complete stop?", gt, opts, a, True,
                             {"c": c}))

    # L3 measurement
    stoppers = [c for c in names if st[c]]
    key = rng.choice(["laps"] + (["n_stops", "stopped_time"] if stoppers else []))
    if key == "laps":
        c = rng.choice(names)
        gt = sim.laps(scene, col[c])
        opts, a = L.mc(rng, gt, L.near_ints(gt, 0))
        qs.append(L.question(3, key, f"How many times does {car[c]} cross the start line (the white line across the "
                             "road)?", gt, opts, a, True, {"c": c}))
    elif key == "n_stops":
        c = rng.choice(stoppers)
        gt = len(st[c])
        opts, a = L.mc(rng, gt, L.near_ints(gt, 0))
        qs.append(L.question(3, key, f"How many separate times does {car[c]} come to a complete stop?", gt, opts, a,
                             True, {"c": c}))
    else:
        c = rng.choice(stoppers)
        gt = _secs(total[c])
        opts, a = L.mc(rng, gt, [_secs(total[c] + d) for d in (1, -1, 2, 3, -2) if total[c] + d > 0])
        qs.append(L.question(3, key, f"For how long in total is {car[c]} standing still (cars move in steps of "
                             f"{sim.STEP_S} s)?", gt, opts, a, True, {"c": c}))

    # L4 integration
    order = sorted(first, key=first.get)
    tops = sorted(total.values(), reverse=True)
    keys = ["n_ever_stopped"] + (["stop_order"] * 2 if len(order) >= 2 and len(set(first.values())) == len(first)
                                 else []) + (["most_stopped"] if tops[0] > tops[1] else [])
    key = rng.choice(keys)
    if key == "n_ever_stopped":
        gt = len(stopped_all)
        opts, a = L.mc(rng, gt, [n] + L.near_ints(gt, 0, n))
        qs.append(L.question(4, key, "Counting every car, grey ones included, how many cars come to a complete stop "
                             "at some point?", gt, opts, a, True))
    elif key == "stop_order":
        gt = ", then ".join(order)
        wrong = []
        for k in range(len(order) - 1):
            alt = list(order)
            alt[k], alt[k + 1] = alt[k + 1], alt[k]
            wrong.append(", then ".join(alt))
        wrong.append(", then ".join(order[::-1]))
        opts, a = L.mc(rng, gt, wrong + [", then ".join(rng.sample(order, len(order))) for _ in range(4)])
        qs.append(L.question(4, key, "In what order do the coloured cars come to a stop for the first time (only "
                             "cars that stop are listed)?", gt, opts, a, True))
    else:
        c = max(names, key=total.get)
        opts, a = L.mc(rng, car[c], [car[x] for x in names if x != c])
        qs.append(L.question(4, key, "Which coloured car spends the longest time standing still in total?", car[c],
                             opts, a, True))

    # L5 counterfactual: one brake tap never happens
    cands = []
    for tc in tappers:
        alt = sim.without_tap(scene, col[tc])
        ast = set(sim.ever_stopped(alt))
        for c in names:
            if c != tc and (col[c] in ast) != bool(st[c]):
                cands += [("cf_no_tap_yes", tc, c)] * 2
        cands.append(("cf_no_tap_count", tc, sum(1 for c in names if col[c] in ast)))
        cands.append(("cf_no_tap_all", tc, len(ast)))
    kind, tc, v = rng.choice(cands)
    if kind == "cf_no_tap_yes":
        alt = sim.without_tap(scene, col[tc])
        gt = "yes" if col[v] in sim.ever_stopped(alt) else "no"
        opts, a = L.yes_no(rng, gt)
        qs.append(L.question(5, kind, f"Suppose {car[tc]} had NOT braked by itself (everything else the same). "
                             f"Would {car[v]} ever have come to a complete stop?", gt, opts, a, True,
                             {"tap": tc, "c": v}))
    elif kind == "cf_no_tap_count":
        opts, a = L.mc(rng, v, [len(first)] + L.near_ints(v, 0, 4))
        qs.append(L.question(5, kind, f"Suppose {car[tc]} had NOT braked by itself (everything else the same). How "
                             "many of the four coloured cars would come to a complete stop at some point?", v, opts,
                             a, True, {"tap": tc}))
    else:
        opts, a = L.mc(rng, v, [len(stopped_all)] + L.near_ints(v, 0, n))
        qs.append(L.question(5, kind, f"Suppose {car[tc]} had NOT braked by itself (everything else the same). "
                             "Counting every car, how many would come to a complete stop at some point?", v, opts, a,
                             True, {"tap": tc}))
    return qs
