"""5-rung ladder for shell_game_tracking_v2 (see common/ladder5.py). Slots are 'slot 1'..'slot N' from the left
(the numbers are drawn under the shelf)."""
from __future__ import annotations
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

NOTE = " Balls are named by colour and kind; slots are numbered from the left as shown under the shelf."


def lab(s: int) -> str:
    return f"slot {s + 1}"


def slot_opts(rng, n, gt):
    return L.mc(rng, lab(gt), [lab(s) for s in rng.sample([x for x in range(n) if x != gt], min(3, n - 1))])


def build(scene: dict, rng: random.Random) -> list[dict]:
    n, balls, swaps = scene["n_cups"], scene["balls"], scene["swaps"]
    ns = len(swaps)
    paths = [sim.track(b["slot"], swaps) for b in balls]     # paths[b][k] = slot after k swaps
    qs = []

    # L1 observation: read at the start
    v = rng.choice(["ball_start_slot", "start_slot_ball", "ball_count"])
    if v == "ball_start_slot":
        bi = rng.randrange(len(balls))
        opts, a = slot_opts(rng, n, balls[bi]["slot"])
        qs.append(L.question(1, v, f"At the start, {balls[bi]['name']} is shown under a raised cup. Which slot is "
                             f"that cup in?" + NOTE, lab(balls[bi]["slot"]), opts, a, False, {"ball": bi}))
    elif v == "start_slot_ball":
        k = rng.randrange(n)
        gt = next((b["name"] for b in balls if b["slot"] == k), "nothing")
        others = [b["name"] for b in balls if b["name"] != gt] + ["nothing"]
        opts, a = L.mc(rng, gt, [o for o in others if o != gt])
        qs.append(L.question(1, v, f"At the start, what is under the cup in {lab(k)}? Answer with a ball, or "
                             f"'nothing' if that cup is empty." + NOTE, gt, opts, a, False, {"slot": k}))
    else:
        c = len(balls)
        opts, a = L.mc(rng, c, L.near_ints(c, 1, 4))
        qs.append(L.question(1, v, "How many balls are shown at the start?", c, opts, a, False))

    # L2 relational
    want = "yes" if rng.random() < 0.5 else "no"
    v = rng.choice(["balls_adjacent", "swap_involves_ball"]) if len(balls) >= 2 else "swap_involves_ball"
    if v == "balls_adjacent":
        pairs = [(x, y) for x in range(len(balls)) for y in range(x + 1, len(balls))]
        good = [p for p in pairs if (abs(balls[p[0]]["slot"] - balls[p[1]]["slot"]) == 1) == (want == "yes")]
        x, y = rng.choice(good or pairs)
        gt = "yes" if abs(balls[x]["slot"] - balls[y]["slot"]) == 1 else "no"
        opts, a = L.yes_no(rng, gt)
        qs.append(L.question(2, v, f"At the start, are {balls[x]['name']} and {balls[y]['name']} under cups in "
                             f"neighbouring slots (next to each other)? Answer yes or no." + NOTE,
                             gt, opts, a, False, {"x": x, "y": y}))
    else:
        cand = [(b, k) for b in range(len(balls)) for k in range(ns)
                if (paths[b][k] in (swaps[k]["i"], swaps[k]["j"])) == (want == "yes")]
        b, k = rng.choice(cand or [(rng.randrange(len(balls)), rng.randrange(ns))])
        gt = "yes" if paths[b][k] in (swaps[k]["i"], swaps[k]["j"]) else "no"
        opts, a = L.yes_no(rng, gt)
        qs.append(L.question(2, v, f"During swap number {k + 1} (counting from the first), is the cup hiding "
                             f"{balls[b]['name']} one of the two cups that trade places? Answer yes or no.",
                             gt, opts, a, True, {"ball": b, "swap": k + 1}))

    # L3 measurement
    if rng.random() < 0.5:
        opts, a = L.mc(rng, ns, L.near_ints(ns, 1))
        qs.append(L.question(3, "swap_count", "How many swaps happen over the whole video?", ns, opts, a, True))
    else:
        b = rng.randrange(len(balls))
        moved = sum(1 for k in range(ns) if paths[b][k + 1] != paths[b][k])
        opts, a = L.mc(rng, moved, L.near_ints(moved, 0))
        qs.append(L.question(3, "ball_moves", f"How many times does the cup hiding {balls[b]['name']} change slots?",
                             moved, opts, a, True, {"ball": b}))

    # L4 integration
    if rng.random() < 0.5:
        b = rng.randrange(len(balls))
        gt_s = paths[b][-1]
        opts, a = slot_opts(rng, n, gt_s)
        qs.append(L.question(4, "ball_final_slot", f"The cups all look identical and the balls were only shown at "
                             f"the start. Following every swap, which slot holds {balls[b]['name']} at the end "
                             f"of the video?" + NOTE, lab(gt_s), opts, a, True, {"ball": b}))
    else:
        final = {paths[b][-1]: balls[b]["name"] for b in range(len(balls))}
        empty = [s for s in range(n) if s not in final]
        full = sorted(final)
        k = rng.choice(empty if rng.random() < 0.35 else full)
        gt = final.get(k, "nothing")
        others = [b["name"] for b in balls if b["name"] != gt] + ["nothing"]
        opts, a = L.mc(rng, gt, [o for o in others if o != gt])
        qs.append(L.question(4, "final_slot_ball", f"Following every swap, what is under the cup in {lab(k)} at the "
                             f"end of the video? Answer with a ball, or 'nothing' if that cup is empty." + NOTE,
                             gt, opts, a, True, {"slot": k}))

    # L5 counterfactual
    b = rng.randrange(len(balls))
    if rng.random() < 0.5:
        hit = [k for k in range(ns) if paths[b][k] in (swaps[k]["i"], swaps[k]["j"])]
        k = rng.choice(hit if hit and rng.random() < 0.7 else list(range(ns)))
        gt_s = sim.track(balls[b]["slot"], swaps[:k] + swaps[k + 1:])[-1]
        opts, a = slot_opts(rng, n, gt_s)
        qs.append(L.question(5, "cf_skip_swap", f"If swap number {k + 1} had never happened, which slot would hold "
                             f"{balls[b]['name']} at the end of the video?" + NOTE, lab(gt_s), opts, a, True,
                             {"ball": b, "swap": k + 1}))
    else:
        x, y = rng.sample(range(n), 2)
        gt_s = sim.track(balls[b]["slot"], swaps + [{"i": x, "j": y}])[-1]
        opts, a = slot_opts(rng, n, gt_s)
        qs.append(L.question(5, "cf_extra_swap", f"Suppose one more swap happened right after the video ends, "
                             f"between {lab(x)} and {lab(y)}. Which slot would then hold {balls[b]['name']}?" + NOTE,
                             lab(gt_s), opts, a, True, {"ball": b, "x": x, "y": y}))
    return qs
