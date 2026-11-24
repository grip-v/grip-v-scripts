"""5-rung ladder for theory_of_mind_belief (see common/ladder5.py)."""
from __future__ import annotations
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

W = "the watcher (the eye icon at the lower left)"


def _label(scene, i):
    n = len(scene["actors"])
    return ["left", "right"][i] + " shape" if n == 2 else ["left", "middle", "right"][i] + " shape"


def _attr(a):
    return a["attr"]


def _pool(a, gt):
    if a["attr"] == "number":
        return L.near_ints(gt, 0)
    return [x for x in (sim.SHAPES if a["attr"] == "shape" else list(sim.PALETTE)) if x != gt]


def _ask_val(rng, a, gt):
    opts, letter = L.mc(rng, gt, _pool(a, gt))
    return opts, letter


def build(scene: dict, rng: random.Random) -> list[dict]:
    acts, ws, t = scene["actors"], scene["windows"], scene["query_t"]
    n = len(acts)
    changes = [(a["index"], k, c) for a in acts for k, c in enumerate(a["changes"])]
    changes.sort(key=lambda x: x[2]["t"])
    stale = [a for a in acts if sim.belief_at(a, ws, t) != sim.value_at(a, t)]
    qs = []

    # L1 observation
    if rng.random() < 0.5:
        tot = len(changes)
        opts, a_ = L.mc(rng, tot, L.near_ints(tot, 0))
        qs.append(L.question(1, "total_changes", "Over the whole video, how many times does any tracked shape "
                             "change (a number ticks, a shape morphs or a colour swaps)?", tot, opts, a_, True))
    else:
        i = rng.randrange(n)
        gt = len(acts[i]["changes"])
        opts, a_ = L.mc(rng, gt, L.near_ints(gt, 0))
        qs.append(L.question(1, "actor_changes", f"Over the whole video, how many times does the "
                             f"{_label(scene, i)}'s {_attr(acts[i])} change?", gt, opts, a_, True, {"actor": i}))

    # L2 relational: the watcher vs. a moment
    if rng.random() < 0.5:
        edges = [w[k] for w in ws for k in ("start_s", "end_s")]
        want_away = rng.random() < 0.5
        for _ in range(500):
            t2 = round(rng.uniform(0.5, scene["duration_s"] - 0.5), 2)
            if all(abs(t2 - e) >= 0.3 for e in edges) and sim._away(t2, ws) == want_away:
                break
        gt = "no" if sim._away(t2, ws) else "yes"
        opts, a_ = L.yes_no(rng, gt)
        qs.append(L.question(2, "watching_at_t", f"At t={t2:.2f}s, is {W} watching the scene? Answer yes or no.",
                             gt, opts, a_, True, {"t": t2}))
    else:
        i, k, c = rng.choice(changes)
        gt = "no" if sim._away(c["t"], ws) else "yes"
        opts, a_ = L.yes_no(rng, gt)
        qs.append(L.question(2, "watching_at_change",
                             f"The {_label(scene, i)}'s {_attr(acts[i])} changes at t={c['t']:.2f}s. Is {W} "
                             f"watching at that moment? Answer yes or no.", gt, opts, a_, True,
                             {"actor": i, "k": k}))

    # L3 measurement: unseen changes
    unseen = [x for x in changes if sim._away(x[2]["t"], ws)]
    if rng.random() < 0.5:
        gt = len(unseen)
        opts, a_ = L.mc(rng, gt, L.near_ints(gt, 0))
        qs.append(L.question(3, "unseen_changes", f"How many changes happen while {W} is looking away "
                             f"(summed over all shapes)?", gt, opts, a_, True))
    else:
        i = rng.randrange(n)
        gt = sum(1 for x in unseen if x[0] == i)
        opts, a_ = L.mc(rng, gt, L.near_ints(gt, 0))
        qs.append(L.question(3, "unseen_changes_actor", f"How many changes to the {_label(scene, i)}'s "
                             f"{_attr(acts[i])} happen while {W} is looking away?", gt, opts, a_, True,
                             {"actor": i}))

    # L4 integration: belief
    if rng.random() < 0.5:
        a = rng.choice(stale)
        gt = sim.belief_at(a, ws, t)
        opts, a_ = _ask_val(rng, a, gt)
        qs.append(L.question(4, "belief_state",
                             f"At t={t:.2f}s, based only on what {W} has itself seen (nothing from while it "
                             f"looked away), what does it believe the {_label(scene, a['index'])}'s "
                             f"{_attr(a)} is?", gt, opts, a_, True, {"actor": a["index"]}))
    else:
        a = rng.choice(acts)
        gt = "yes" if sim.belief_at(a, ws, t) != sim.value_at(a, t) else "no"
        opts, a_ = L.yes_no(rng, gt)
        qs.append(L.question(4, "belief_stale",
                             f"At t={t:.2f}s, does {W} hold an outdated belief about the {_label(scene, a['index'])}'s "
                             f"{_attr(a)} (different from its actual value)? Answer yes or no.", gt, opts, a_, True,
                             {"actor": a["index"]}))

    # L5 counterfactual
    a = rng.choice(acts)
    if rng.random() < 0.5:
        k = rng.randrange(len(a["changes"]))
        ch = a["changes"][k]
        alt = {**a, "changes": [c for j, c in enumerate(a["changes"]) if j != k]}
        gt = sim.belief_at(alt, ws, t)
        opts, a_ = _ask_val(rng, a, gt)
        qs.append(L.question(5, "cf_remove_change",
                             f"Suppose the change to the {_label(scene, a['index'])}'s {_attr(a)} at t={ch['t']:.2f}s "
                             f"had never happened (everything else unchanged). At t={t:.2f}s, what would {W} "
                             f"believe the {_attr(a)} is?", gt, opts, a_, True, {"actor": a["index"], "k": k}))
    else:
        d = scene["cf_shift_s"]
        gt = sim.belief_at(a, sim._shifted(ws, d), t)
        opts, a_ = _ask_val(rng, a, gt)
        way = "later" if d > 0 else "earlier"
        qs.append(L.question(5, "cf_shift_windows",
                             f"Suppose every period when {W} looks away had started and ended {abs(d):.1f} s "
                             f"{way} than shown. At t={t:.2f}s, what would it believe the "
                             f"{_label(scene, a['index'])}'s {_attr(a)} is?", gt, opts, a_, True,
                             {"actor": a["index"]}))
    return qs
