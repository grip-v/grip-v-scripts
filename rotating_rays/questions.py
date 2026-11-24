"""5-rung ladder for rotating_rays (see common/ladder5.py). Directions are
as seen on screen. The rays "line up" when they point the same way; they
are "at a right angle" when 90 degrees apart."""
from __future__ import annotations
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

KIND = lambda a: "a right angle" if abs(a - 90) < 1e-6 else ("acute (less than 90°)" if a < 90 else
                                                            "obtuse (more than 90°)")


def _safe(t, step):
    f = (t / step) % 1
    return abs(f - 0.5) >= 0.25


def build(scene: dict, rng: random.Random) -> list[dict]:
    T, r, b = scene["T"], scene["red"], scene["blue"]
    al, ra = sim.events(scene, 0), sim.events(scene, 90)
    qs = []

    # L1 observation
    key = rng.choice(["faster", "red_dir", "n_turns"])
    if key == "faster":
        gt = "the red ray" if abs(r["w"]) > abs(b["w"]) else "the blue ray"
        opts, a = L.mc(rng, gt, ["the red ray", "the blue ray"])
        qs.append(L.question(1, key, "Which ray turns faster?", gt, opts, a, True))
    elif key == "red_dir":
        gt = "counter-clockwise" if r["w"] > 0 else "clockwise"
        opts, a = L.mc(rng, gt, ["clockwise", "counter-clockwise"])
        qs.append(L.question(1, key, "Which way does the red ray turn?", gt, opts, a, True))
    else:
        ray, word = rng.choice([(r, "red"), (b, "blue")])
        gt = int(abs(ray["w"]) * T // 360)
        opts, a = L.mc(rng, gt, L.near_ints(gt, 0))
        qs.append(L.question(1, key, f"How many complete turns does the {word} ray make?", gt, opts, a, True,
                             {"ray": word}))

    # L2 relational
    end = sim.between(scene, T)
    keys = ["same_dir"] + (["end_kind"] * 2 if abs(end - 90) >= 10 or abs(end - 90) < 1e-6 else [])
    key = rng.choice(keys)
    if key == "same_dir":
        gt = "yes" if (r["w"] > 0) == (b["w"] > 0) else "no"
        opts, a = L.yes_no(rng, gt)
        qs.append(L.question(2, key, "Do the two rays turn the same way (both clockwise or both counter-clockwise)?",
                             gt, opts, a, True))
    else:
        gt = KIND(end)
        opts, a = L.mc(rng, gt, ["acute (less than 90°)", "a right angle", "obtuse (more than 90°)"])
        qs.append(L.question(2, key, "At the end of the video, is the angle between the two rays acute, a right "
                             "angle, or obtuse?", gt, opts, a, True))

    # L3 measurement
    keys = (["end_angle"] if abs(end / 15 - round(end / 15)) < 1e-6 else []) + \
        (["first_align"] * 2 if _safe(al[0], 0.1) else [])
    key = rng.choice(keys or ["end_angle"])
    if key == "end_angle":
        v = int(round(end / 15) * 15)
        gt = f"{v}°"
        opts, a = L.mc(rng, gt, [f"{x}°" for x in (v + 15, v - 15, v + 30, v - 30, 180 - v) if 0 <= x <= 180])
        qs.append(L.question(3, key, "At the end of the video, what is the angle between the two rays? (The dial has "
                             "a tick every 15°.)", gt, opts, a, True))
    else:
        v = round(al[0], 1)
        gt = f"{v:.1f} s"
        opts, a = L.mc(rng, gt, [f"{v + d:.1f} s" for d in (0.1, -0.1, 0.2, -0.2, 0.3) if v + d > 0])
        qs.append(L.question(3, key, "On the clock, when do the two rays first line up, pointing the same way (to the "
                             "nearest 0.1 s)?", gt, opts, a, True))

    # L4 integration
    gap = 360 / abs(r["w"] - b["w"])
    keys = ["n_align", "n_right"] + (["align_gap"] if len(al) >= 2 and _safe(gap, 0.5) else [])
    key = rng.choice(keys)
    if key == "n_align":
        gt = len(al)
        opts, a = L.mc(rng, gt, [len(ra)] + L.near_ints(gt, 0))
        qs.append(L.question(4, key, "How many times do the two rays line up, pointing the same way?", gt, opts, a,
                             True))
    elif key == "n_right":
        gt = len(ra)
        opts, a = L.mc(rng, gt, [len(al)] + L.near_ints(gt, 0))
        qs.append(L.question(4, key, "How many times are the two rays at a right angle to each other?", gt, opts, a,
                             True))
    else:
        v = round(gap * 2) / 2
        gt = f"{v:.1f} s"
        opts, a = L.mc(rng, gt, [f"{v + d:.1f} s" for d in (0.5, -0.5, 1.0, -1.0, 2.0) if v + d > 0])
        qs.append(L.question(4, key, "How much time passes between one moment the rays line up and the next (to the "
                             "nearest 0.5 s)?", gt, opts, a, True))

    # L5 counterfactual
    cands = []
    rev = sim.events(scene, 0, blue=dict(b, w=-b["w"]))
    ev2 = sorted(rev + sim.events(scene, 90, blue=dict(b, w=-b["w"])))
    if not sim.on_event(scene, T, blue=dict(b, w=-b["w"])) and all(sim.EDGE <= t <= T - sim.EDGE for t in rev) and all(y - x >= sim.EV_GAP for x, y in zip(ev2, ev2[1:])):
        cands += [("cf_reverse_blue", len(rev))] * 2
    dbl = sim.events(scene, 0, red=dict(r, w=2 * r["w"]))
    if dbl and not sim.on_event(scene, T, red=dict(r, w=2 * r["w"])) and dbl[0] >= 0.3 and _safe(dbl[0], 0.1) and 2 * r["w"] != b["w"]:
        cands.append(("cf_double_red", dbl[0]))
    still = dict(b, w=0)
    st_ev = sim.events(scene, 0, blue=still)
    st_all = sorted(st_ev + sim.events(scene, 90, blue=still))
    if not sim.on_event(scene, T, blue=still) and all(sim.EDGE <= t <= T - sim.EDGE for t in st_ev) and \
            all(y - x >= sim.EV_GAP for x, y in zip(st_all, st_all[1:])):
        cands.append(("cf_still_blue", len(st_ev)))
    kind, v = rng.choice(cands) if cands else ("n_align", len(al))
    if kind == "cf_reverse_blue":
        opts, a = L.mc(rng, v, [len(al)] + L.near_ints(v, 0))
        qs.append(L.question(5, kind, "Suppose the blue ray had turned the other way at the same speed (everything "
                             "else the same). How many times would the rays line up?", v, opts, a, True))
    elif kind == "cf_still_blue":
        opts, a = L.mc(rng, v, [len(al)] + L.near_ints(v, 0))
        qs.append(L.question(5, kind, "Suppose the blue ray had stayed still at its starting position (the red ray "
                             "turning as before). How many times would the red ray pass over it?", v, opts, a, True))
    elif kind == "cf_double_red":
        x = round(v, 1)
        gt = f"{x:.1f} s"
        opts, a = L.mc(rng, gt, [f"{round(al[0], 1):.1f} s"] + [f"{x + d:.1f} s" for d in (0.2, -0.2, 0.4) if x + d > 0])
        qs.append(L.question(5, kind, "Suppose the red ray had turned twice as fast in the same direction (everything "
                             "else the same). When would the rays first line up (to the nearest 0.1 s)?", gt, opts,
                             a, True))
    else:
        opts, a = L.mc(rng, v, L.near_ints(v, 0))
        qs.append(L.question(5, "n_align", "How many times do the two rays line up, pointing the same way?", v, opts,
                             a, True))
    return qs
