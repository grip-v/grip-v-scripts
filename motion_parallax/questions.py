"""5-rung ladder for motion_parallax (see common/ladder5.py). Balls are
named by colour. The camera always slides to the right, so every ball
drifts left, nearer balls faster. Clock answers are only asked away from a
rounding boundary; comparisons only with a margin."""
from __future__ import annotations
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

RATIOS = ["1.5 times", "2 times", "2.5 times", "3 times", "4 times", "1.25 times", "1.33 times", "1.67 times"]


def _ratio_text(r):
    for t in RATIOS:
        if abs(float(t.split()[0]) - r) < 0.02:
            return t
    return None


def _safe(x, step=0.1):
    f = (x / step) % 1
    return abs(f - 0.5) >= 0.3


def build(scene: dict, rng: random.Random) -> list[dict]:
    bs, T = scene["balls"], scene["T"]
    n = len(bs)
    nm = [f"the {b['color']} ball" for b in bs]
    idx = list(range(n))
    lt = [sim.leave_time(scene, b) for b in bs]
    gone = sorted([i for i in idx if lt[i] < T], key=lambda i: lt[i])
    qs = []

    # L1 observation
    key = rng.choice((["leaves_first"] * 2 if gone else []) + ["fastest"] * 2 + ["n_balls"])
    if key == "leaves_first":
        gt = nm[gone[0]]
        opts, a = L.mc(rng, gt, [nm[i] for i in idx if i != gone[0]])
        qs.append(L.question(1, key, "Which ball is the first to leave the screen?", gt, opts, a, True))
    elif key == "fastest":
        i = min(idx, key=lambda i: bs[i]["z"])
        opts, a = L.mc(rng, nm[i], [nm[k] for k in idx if k != i])
        qs.append(L.question(1, key, "Which ball slides across the screen the fastest?", nm[i], opts, a, True))
    else:
        opts, a = L.mc(rng, n, L.near_ints(n, 1))
        qs.append(L.question(1, key, "How many balls are there?", n, opts, a, False))

    # L2 relational: depth from motion, preferring pairs where size misleads
    pairs = [(x, y) for x, y in itertools.permutations(idx, 2) if bs[x]["z"] > 1.25 * bs[y]["z"]]
    tricky = [(x, y) for x, y in pairs if bs[x]["r"] > 1.2 * bs[y]["r"]]
    far, near = rng.choice(tricky if tricky and rng.random() < 0.75 else pairs)
    pair = [far, near]
    rng.shuffle(pair)
    opts, a = L.mc(rng, nm[near], [nm[far]])
    qs.append(L.question(2, "closer", f"The camera slides sideways at a steady speed. Which ball is closer to the "
                         f"camera: {nm[pair[0]]} or {nm[pair[1]]}?", nm[near], opts, a, True,
                         {"x": pair[0], "y": pair[1]}))

    # L3 measurement
    rats = [(x, y) for x, y in pairs if _ratio_text(bs[x]["m"] / bs[y]["m"])]
    ct = [i for i in idx if sim.center_time(scene, bs[i]) and sim.center_time(scene, bs[i]) < T - 0.5 and
          _safe(sim.center_time(scene, bs[i]))]
    keys = (["depth_ratio"] * 2 if rats else []) + (["center_time"] if ct else [])
    key = rng.choice(keys or ["depth_ratio"])
    if key == "depth_ratio" and rats:
        x, y = rng.choice(rats)
        gt = _ratio_text(bs[x]["m"] / bs[y]["m"])
        opts, a = L.mc(rng, gt, [t for t in RATIOS[:5] if t != gt])
        qs.append(L.question(3, "depth_ratio", f"How many times farther from the camera is {nm[x]} than {nm[y]}?",
                             gt, opts, a, True, {"x": x, "y": y}))
    else:
        i = rng.choice(ct)
        v = round(sim.center_time(scene, bs[i]), 1)
        gt = f"{v:.1f} s"
        opts, a = L.mc(rng, gt, [f"{v + d:.1f} s" for d in (0.1, -0.1, 0.2, -0.2, 0.3) if v + d > 0])
        qs.append(L.question(3, "center_time", f"On the clock, when does the centre of {nm[i]} cross the dashed centre "
                             "line (to the nearest 0.1 s)?", gt, opts, a, True, {"x": i}))

    # L4 integration
    rs = sorted(idx, key=lambda i: -sim.real_size(bs[i]))
    keys = (["depth_order"] * 2 if n <= 4 else []) + \
        (["true_largest"] * 2 if sim.real_size(bs[rs[0]]) >= 1.2 * sim.real_size(bs[rs[1]]) else [])
    key = rng.choice(keys or ["depth_order"])
    if key == "depth_order" and n <= 4:
        order = sorted(idx, key=lambda i: bs[i]["z"])
        seq = [bs[i]["color"] for i in order]
        gt = " → ".join(seq)
        wrong = [" → ".join(sorted(seq, key=lambda c: -next(b["r"] for b in bs if b["color"] == c)))]
        for k in range(len(seq) - 1):
            alt = list(seq)
            alt[k], alt[k + 1] = alt[k + 1], alt[k]
            wrong.append(" → ".join(alt))
        wrong.append(" → ".join(seq[::-1]))
        opts, a = L.mc(rng, gt, wrong)
        qs.append(L.question(4, "depth_order", "Order the balls from nearest to farthest from the camera.", gt, opts,
                             a, True))
    elif key == "true_largest":
        gt = nm[rs[0]]
        looks = max(idx, key=lambda i: bs[i]["r"])
        opts, a = L.mc(rng, gt, [nm[looks]] + [nm[i] for i in rs[1:]])
        qs.append(L.question(4, key, "Which ball is actually the largest in real size (not just how big it looks "
                             "on screen)?", gt, opts, a, True))
    else:
        i = min(idx, key=lambda i: bs[i]["z"])
        opts, a = L.mc(rng, nm[i], [nm[k] for k in idx if k != i])
        qs.append(L.question(4, "nearest", "Which ball is nearest to the camera?", nm[i], opts, a, True))

    # L5 counterfactual
    cands = []
    for i in idx:
        c = sim.center_time(scene, bs[i])
        if c and _safe(c / 2) and c / 2 >= 0.3:
            cands.append(("cf_double_speed", i))
    if len({round(sim.leave_time(dict(scene, v=-scene["v"]), b), 3) for b in bs}) == n:
        rt = [(1280 - sim.screen_x(scene, b, 0) + b["r"]) / sim.speed(scene, b) for b in bs]
        s = sorted(rt)
        if s[1] - s[0] >= 0.6:
            cands += [("cf_reverse_first", None)] * 2
    far_pairs = [(x, y) for x, y in itertools.permutations(idx, 2)
                 if max(2 * bs[x]["z"], bs[y]["z"]) >= 1.25 * min(2 * bs[x]["z"], bs[y]["z"])]
    if far_pairs:
        cands.append(("cf_twice_far", rng.choice(far_pairs)))
    kind, arg = rng.choice(cands)
    if kind == "cf_double_speed":
        v = round(sim.center_time(scene, bs[arg]) / 2, 1)
        gt = f"{v:.1f} s"
        orig = round(sim.center_time(scene, bs[arg]), 1)
        opts, a = L.mc(rng, gt, [f"{orig:.1f} s"] + [f"{v + d:.1f} s" for d in (0.2, -0.2, 0.4) if v + d > 0])
        qs.append(L.question(5, kind, f"If the camera had moved twice as fast, when would the centre of {nm[arg]} "
                             "cross the dashed centre line (to the nearest 0.1 s)?", gt, opts, a, True, {"x": arg}))
    elif kind == "cf_reverse_first":
        rt = [(1280 - sim.screen_x(scene, b, 0) + b["r"]) / sim.speed(scene, b) for b in bs]
        i = min(idx, key=lambda k: rt[k])
        opts, a = L.mc(rng, nm[i], [nm[k] for k in idx if k != i])
        qs.append(L.question(5, kind, "If the camera had slid the other way at the same speed (so every ball drifts "
                             "right), which ball would be the first to leave the screen?", nm[i], opts, a, True))
    else:
        x, y = arg
        gt = nm[x] if 2 * bs[x]["z"] < bs[y]["z"] else nm[y]
        opts, a = L.mc(rng, gt, [nm[x], nm[y]])
        qs.append(L.question(5, kind, f"If {nm[x]} were moved twice as far from the camera (everything else the same), "
                             f"which would slide across the screen faster: {nm[x]} or {nm[y]}?", gt, opts, a, True,
                             {"x": x, "y": y}))
    return qs
