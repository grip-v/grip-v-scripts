"""5-rung ladder for domino_chains (see common/ladder5.py)."""
from __future__ import annotations
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402


def build(scene: dict, rng: random.Random) -> list[dict]:
    fall = scene["fall"]
    letters = [d["id"] for d in scene["dominoes"] if d["label"]]
    fallen = sorted((x for x in letters if x in fall), key=lambda x: fall[x])
    standing = [x for x in letters if x not in fall]
    qs = []

    # L1 observation
    if rng.random() < 0.6:
        v = len(fallen)
        opts, a = L.mc(rng, v, L.near_ints(v, 1, len(letters)))
        qs.append(L.question(1, "n_fallen", "How many of the lettered dominoes get knocked over?", v, opts, a, True))
    else:
        v = len(standing)
        opts, a = L.mc(rng, v, L.near_ints(v, 0, len(letters)))
        qs.append(L.question(1, "n_standing", "How many of the lettered dominoes are still standing at the end?",
                             v, opts, a, True))

    # L2 relational (time order)
    pairs = [(x, y) for i, x in enumerate(fallen) for y in fallen[i + 1:]
             if x != "A" and fall[y] - fall[x] >= 0.4] or [("A", y) for y in fallen[1:]]
    nexts = [x for i, x in enumerate(fallen[:-2]) if fall[fallen[i + 2]] - fall[fallen[i + 1]] >= 0.3
             and fall[fallen[i + 1]] - fall[x] >= 0.2]
    if nexts and rng.random() < 0.5:
        x = rng.choice(nexts)
        gt = fallen[fallen.index(x) + 1]
        opts, a = L.mc(rng, gt, rng.sample([y for y in letters if y != x], len(letters) - 1))
        qs.append(L.question(2, "next_after", f"Right after {x} falls, which lettered domino is the next one to "
                             f"fall?", gt, opts, a, True, {"x": x}))
    else:
        x, y = rng.choice(pairs)
        pair = [x, y]
        rng.shuffle(pair)
        opts, a = L.mc(rng, x, [y, "they fall at the same moment"])
        qs.append(L.question(2, "fall_before", f"Which falls first, {pair[0]} or {pair[1]}?", x, opts, a, True,
                             {"pair": pair}))

    # L3 measurement (clock top right)
    cands = [x for x in fallen if x != "A" and abs(fall[x] * 2 - round(fall[x] * 2)) <= 0.3]
    x = rng.choice(cands)
    base = round(fall[x] * 2) / 2
    gt = f"{base:.1f} s"
    opts, a = L.mc(rng, gt, [f"{v:.1f} s" for v in (base + 0.5, base - 0.5, base + 1.0, base - 1.0) if v > 0])
    qs.append(L.question(3, "fall_time", f"At about what time on the clock does domino {x} fall (to the nearest "
                         f"half second)?", gt, opts, a, True, {"x": x}))

    # L4 integration
    trig = sim.triggers(scene, fall)
    merges = [v for v, arr in trig.items() if len(arr) > 1]
    deep = [x for x in fallen if len(_path(trig, x)) >= 3] or [x for x in fallen if x != "A"]
    if merges and rng.random() < 0.6:
        x = rng.choice(merges)
        gt = trig[x][0][1]
        srcs = [s for _, s, _ in trig[x]]
        opts, a = L.mc(rng, gt, srcs + rng.sample([y for y in letters if y not in srcs and y != x], 2))
        qs.append(L.question(4, "trigger", f"More than one lane leads into {x}. Which lettered domino's lane "
                             f"actually knocks {x} over?", gt, opts, a, True, {"x": x}))
    else:
        x = rng.choice(deep)
        path = _path(trig, x)
        gt = " → ".join(path)
        wrong = []
        others = [y for y in letters if y not in path]
        for i in range(1, len(path) - 1):
            for y in others:
                alt = list(path)
                alt[i] = y
                wrong.append(" → ".join(alt))
        wrong.append(" → ".join(path[:1] + path[2:]))
        wrong.append(" → ".join(path[:1] + path[1:-1][::-1] + path[-1:]))
        rng.shuffle(wrong)
        opts, a = L.mc(rng, gt, wrong)
        qs.append(L.question(4, "path", f"The chain reaction that knocks {x} over runs through which lettered "
                             f"dominoes, in order?", gt, opts, a, True, {"x": x}))

    # L5 counterfactual
    gap_lanes = [k for k, lane in enumerate(scene["lanes"]) if lane["gap"] is not None]
    fillable = [k for k in gap_lanes if len([y for y in letters if y in sim.fall_times(scene, filled=k)])
                != len(fallen) or rng.random() < 0.3]
    if fillable and rng.random() < 0.5:
        k = rng.choice(fillable)
        lane = scene["lanes"][k]
        v = len([y for y in letters if y in sim.fall_times(scene, filled=k)])
        opts, a = L.mc(rng, v, [len(fallen)] + L.near_ints(v, 1, len(letters)))
        qs.append(L.question(5, "cf_fill", f"The lane from {lane['u']} to {lane['v']} has a gap where a domino is "
                             f"missing. If that gap were filled with one more domino, how many lettered dominoes "
                             f"would get knocked over in total?", v, opts, a, True, {"lane": k}))
    else:
        x = rng.choice([y for y in fallen if y != "A"])
        v = len([y for y in letters if y in sim.fall_times(scene, removed=x)])
        opts, a = L.mc(rng, v, [len(fallen) - 1] + L.near_ints(v, 0, len(letters)))
        qs.append(L.question(5, "cf_remove", f"If domino {x} were taken off the board before the push (leaving an "
                             f"empty spot), how many of the other lettered dominoes would get knocked over?",
                             v, opts, a, True, {"x": x}))
    return qs


def _path(trig, x):
    path = [x]
    while path[-1] != "A":
        path.append(trig[path[-1]][0][1])
    return path[::-1]
