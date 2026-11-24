"""5-rung ladder for object_tracking_mot (see common/ladder5.py)."""
from __future__ import annotations
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

QU = sim.QUADRANTS


def quadrant(d, t, c):
    x, y = sim.pos(d, t, c)
    return ("top" if y < c else "bottom") + "-" + ("left" if x < c else "right")


def times(scene):
    n = int(scene["duration"] * scene["sample_hz"]) + 1
    return [min(k / scene["sample_hz"], scene["duration"]) for k in range(n)]


def encounters(scene, i, j):
    """Number of separate intervals in which dots i and j are closer than the threshold."""
    c, n_int, inside = scene["canvas"] / 2, 0, False
    for t in times(scene):
        near = math.dist(sim.pos(scene["dots"][i], t, c), sim.pos(scene["dots"][j], t, c)) < scene["encounter_dist"]
        if near and not inside:
            n_int += 1
        inside = near
    return n_int


def midline_crossings(scene, k):
    c, n, prev = scene["canvas"] / 2, 0, None
    for t in [x / 60 for x in range(int(scene["duration"] * 60) + 1)]:
        s = sim.pos(scene["dots"][k], t, c)[0] < c
        if prev is not None and s != prev:
            n += 1
        prev = s
    return n


def moved(d, kind, val, t):
    """Dot d modified by a counterfactual, evaluated at time t."""
    if kind == "speed":
        return sim.pos({**d, "wx": d["wx"] * val, "wy": d["wy"] * val}, t)
    return sim.pos(d, min(t, val))          # freeze


def build(scene: dict, rng: random.Random) -> list[dict]:
    dots, T, c = scene["dots"], scene["duration"], scene["canvas"] / 2
    n = len(dots)
    qs = []

    # L1 observation
    if rng.random() < 0.5:
        opts, a = L.mc(rng, n, L.near_ints(n, 1))
        qs.append(L.question(1, "dot_count", "How many dots are shown in this video?", n, opts, a, False))
    else:
        k = rng.randrange(n)
        gt = quadrant(dots[k], 0.0, c)
        opts, a = L.mc(rng, gt, [q for q in QU if q != gt])
        qs.append(L.question(1, "start_quadrant", f"The dots are numbered 1-{n} at the very start. Which quadrant "
                             f"of the frame is dot {k + 1} in at the start of the video?", gt, opts, a, False,
                             {"dot": k}))

    # L2 relational
    if rng.random() < 0.5:
        i, j = sorted(rng.sample(range(n), 2))
        gt = "yes" if encounters(scene, i, j) > 0 else "no"
        opts, a = L.yes_no(rng, gt)
        qs.append(L.question(2, "pair_meet", f"Dots were numbered 1-{n} at the start only. Did the dot originally "
                             f"labelled {i + 1} ever touch (circles meet) the dot originally labelled {j + 1}? Answer yes "
                             f"or no.", gt, opts, a, True, {"i": i, "j": j}))
    else:
        k = rng.randrange(n)
        d = [(math.dist(sim.pos(dots[k], T, c), sim.pos(dots[m], T, c)), m) for m in range(n) if m != k]
        gt = f"dot {min(d)[1] + 1}"
        opts, a = L.mc(rng, gt, [f"dot {m + 1}" for m in range(n) if m != k and f"dot {m + 1}" != gt])
        qs.append(L.question(2, "closest_at_end", f"Following the dots by their original numbers, which dot is "
                             f"closest to dot {k + 1} at the end of the video?", gt, opts, a, True, {"dot": k}))

    # L3 measurement
    if rng.random() < 0.5:
        tot = sum(encounters(scene, i, j) for i in range(n) for j in range(i + 1, n))
        opts, a = L.mc(rng, tot, L.near_ints(tot, 0))
        qs.append(L.question(3, "encounter_count", "Counting every time any two dots touch or overlap "
                             "(their circles meet), how many separate touching encounters happen in total?", tot, opts, a, True))
    else:
        k = rng.randrange(n)
        cnt = midline_crossings(scene, k)
        opts, a = L.mc(rng, cnt, L.near_ints(cnt, 0))
        qs.append(L.question(3, "midline_crossings", f"How many times does dot {k + 1} cross the vertical centre "
                             f"line of the frame (left to right or right to left) during the video?",
                             cnt, opts, a, True, {"dot": k}))

    # L4 integration
    k = rng.randrange(n)
    gt = quadrant(dots[k], T, c)
    opts, a = L.mc(rng, gt, [q for q in QU if q != gt])
    qs.append(L.question(4, "final_quadrant", f"All dots look identical after the first couple of seconds. Tracking the dot "
                         f"originally labelled {k + 1} through every crossing, which quadrant is it in at the end of "
                         f"the video?", gt, opts, a, True, {"dot": k}))

    # L5 counterfactual
    k = rng.randrange(n)
    for _ in range(200):
        fz = rng.random() < 0.5
        v = rng.choice([0.5, 1.5, 2.0]) if not fz else round(rng.uniform(0.35, 0.65) * T, 1)
        x, y = moved(dots[k], "freeze" if fz else "speed", v, T)
        if min(abs(x - c), abs(y - c)) >= sim.BOUNDARY_MARGIN:
            break
        k = rng.randrange(n)
    x, y = moved(dots[k], "freeze" if fz else "speed", v, T)
    if not fz:
        key, txt, p = "cf_speed", (f"If dot {k + 1} had moved {v}x as fast along the same path for the whole video "
                                   f"(everything else unchanged), which quadrant would it be in at the end?"), \
            {"dot": k, "factor": v}
    else:
        key, txt, p = "cf_freeze", (f"If dot {k + 1} had stopped dead at t = {v} s and stayed put for the rest of "
                                    f"the video, which quadrant would it be in at the end?"), {"dot": k, "t": v}
    gt = ("top" if y < c else "bottom") + "-" + ("left" if x < c else "right")
    opts, a = L.mc(rng, gt, [q for q in QU if q != gt])
    qs.append(L.question(5, key, txt, gt, opts, a, True, p))
    return qs
