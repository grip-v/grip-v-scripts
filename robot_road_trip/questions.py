"""5-rung ladder for robot_road_trip (see common/ladder5.py)."""
from __future__ import annotations
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402
from answers import _waits  # noqa: E402


def build(scene: dict, rng: random.Random) -> list[dict]:
    cars = [c["name"] for c in scene["cars"]]
    fin = scene["finish"]
    pulled = sorted({p["car"] for p in scene["pullovers"]})
    qs = []

    # L1 observation
    if rng.random() < 0.7:
        v = len(scene["pullovers"])
        opts, a = L.mc(rng, v, L.near_ints(v, 0))
        qs.append(L.question(1, "n_pullovers", "How many times does a car pull over onto the shoulder?",
                             v, opts, a, True))
    else:
        opts, a = L.mc(rng, len(cars), L.near_ints(len(cars), 2))
        qs.append(L.question(1, "n_cars", "How many cars drive along the road?", len(cars), opts, a, True))

    # L2 relational
    if rng.random() < 0.6:
        car = rng.choice(pulled)
        gt = next(p["passer"] for p in scene["pullovers"] if p["car"] == car)
        opts, a = L.mc(rng, gt, rng.sample([c for c in cars if c != car], len(cars) - 1))
        qs.append(L.question(2, "first_passer", f"The first time car {car} pulls onto the shoulder, which car is "
                             f"it making way for?", gt, opts, a, True, {"car": car}))
    else:
        gt = min(fin, key=fin.get)
        opts, a = L.mc(rng, gt, rng.sample(cars, len(cars)))
        qs.append(L.question(2, "first_finisher", "Which car crosses the finish line first?", gt, opts, a, True))

    # L3 measurement (the clock and the unit markers are on screen)
    waits = _waits(scene)
    clean = [c for c in pulled if abs(waits[c] * 2 - round(waits[c] * 2)) < 0.3 and waits[c] >= 0.4]
    if clean and rng.random() < 0.5:
        car = rng.choice(clean)
        v = round(waits[car] * 2) / 2
        f = lambda x: f"{x:.1f} s"
        opts, a = L.mc(rng, v, [x for x in (v + 0.5, v - 0.5, v + 1.0, v + 1.5) if x > 0], fmt=f)
        qs.append(L.question(3, "wait_time", f"In total, about how long does car {car} spend waiting on the "
                             f"shoulder (to the nearest half second)?", f(v), opts, a, True, {"car": car}))
    else:
        car = rng.choice(cars)
        v = next(c["speed"] for c in scene["cars"] if c["name"] == car)
        f = lambda x: f"{x:.2f}"
        opts, a = L.mc(rng, v, [round(v + s, 2) for s in (0.3, -0.3, 0.6, -0.6) if 0.6 <= v + s <= 2.6], fmt=f)
        qs.append(L.question(3, "speed", f"While it is driving in the lane, about how many road units per second "
                             f"does car {car} cover?", f(v), opts, a, True, {"car": car}))

    # L4 integration
    lost = {c["name"]: fin[c["name"]] - c["enter_t"] - scene["road"] / c["speed"] for c in scene["cars"]}
    top = sorted(lost.values(), reverse=True)
    if top[0] - top[1] >= 0.4 and rng.random() < 0.5:
        gt = max(lost, key=lost.get)
        opts, a = L.mc(rng, gt, rng.sample(cars, len(cars)))
        qs.append(L.question(4, "most_delayed",
                             "Compared with driving the whole road non-stop at its own speed, which car's trip "
                             "is held up the most (waiting to enter plus time on the shoulder)?", gt, opts, a, True))
    else:
        order = sorted(fin, key=fin.get)
        gt = " → ".join(order)
        wrong = []
        for i in range(len(order) - 1):
            alt = list(order)
            alt[i], alt[i + 1] = alt[i + 1], alt[i]
            wrong.append(" → ".join(alt))
        wrong.append(" → ".join(c["name"] for c in scene["cars"]))       # entry order
        opts, a = L.mc(rng, gt, rng.sample(wrong, len(wrong)))
        qs.append(L.question(4, "finish_order", "In what order do the cars cross the finish line?",
                             gt, opts, a, True))

    # L5 counterfactual: one car never enters
    car = rng.choice(cars)
    v = len(sim.simulate(scene["cars"], skip=car)["pullovers"])
    opts, a = L.mc(rng, v, [len(scene["pullovers"])] + L.near_ints(v, 0))
    qs.append(L.question(5, "cf_remove_car",
                         f"If car {car} had never entered the road (every other car entering and driving exactly "
                         f"as before), how many times in total would a car pull over?", v, opts, a, True,
                         {"car": car}))
    return qs
