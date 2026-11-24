"""Scene generation for phantom_traffic_jams (New Problem Types B18).

N cars (10-19) drive one way around a ring road of L = 60 cells under the
Nagel-Schreckenberg rule, with no randomness: every step each car
1. speeds up by 1 (up to VMAX = 3),
2. slows to the number of empty cells in front of it (no collisions),
3. if it is BRAKING (a scheduled brake tap holds it at a stop for 2-3
   steps), its speed is 0 and its brake lights flash,
and then all cars move forward together. Four cars are coloured (red,
blue, green, gold) and the rest are grey; 1-3 coloured cars tap their
brakes once each. A tap can start a queue of stopped cars that drifts
BACKWARD around the ring while every car keeps moving forward, catching
cars far behind the one that braked. Cars never overtake.

State is integer and exact. A step lasts STEP_S seconds on the clock; the
renderer slides each car smoothly between its positions.
"""
from __future__ import annotations
import functools
import json
import random

L = 60
VMAX = 3
STEP_S = 0.3
STEPS = (40, 52)
COLORS = ["red", "blue", "green", "gold"]


@functools.lru_cache(maxsize=64)
def _run(key: str):
    scene = json.loads(key)
    pos = list(scene["pos"])
    vel = list(scene["vel"])
    n = len(pos)
    taps = {(t["step"] + h, t["car"]) for t in scene["taps"] for h in range(t["hold"])}
    road = [-1] * L
    for i, p in enumerate(pos):
        road[p] = i
    P, V, B = [list(pos)], [list(vel)], [[False] * n]
    for s in range(scene["steps"]):
        new_v, brake = [], []
        for i in range(n):
            gap = 0
            while gap < VMAX + 1 and road[(pos[i] + gap + 1) % L] == -1:
                gap += 1
            v = min(vel[i] + 1, VMAX, gap)
            b = (s, i) in taps
            if b:
                v = 0
            new_v.append(v)
            brake.append(b)
        road = [-1] * L
        for i in range(n):
            pos[i] = (pos[i] + new_v[i]) % L
            road[pos[i]] = i
        vel = new_v
        P.append(list(pos))
        V.append(list(vel))
        B.append(brake)
    return P, V, B


def run(scene):
    return _run(json.dumps({k: scene[k] for k in ("pos", "vel", "taps", "steps")}, sort_keys=True))


def colored(scene) -> dict:
    """colour -> car index."""
    return {c: i for i, c in enumerate(scene["colors"]) if c != "gray"}


def stops(scene, i) -> list:
    """Stop episodes of car i: list of (first step stopped, number of steps stopped)."""
    _, V, _ = run(scene)
    out, cur = [], None
    for s in range(1, len(V)):
        if V[s][i] == 0:
            if cur is None:
                cur = [s, 0]
            cur[1] += 1
        elif cur is not None:
            out.append(tuple(cur))
            cur = None
    if cur is not None:
        out.append(tuple(cur))
    return out


def distance(scene, i) -> int:
    _, V, _ = run(scene)
    return sum(V[s][i] for s in range(1, len(V)))


def laps(scene, i) -> int:
    """Times car i crosses the start line (from cell L-1 side into cell 0 side)."""
    P, V, _ = run(scene)
    return sum(1 for s in range(1, len(P)) if P[s - 1][i] + V[s][i] >= L)


def without_tap(scene, car) -> dict:
    return dict(scene, taps=[t for t in scene["taps"] if t["car"] != car])


def ever_stopped(scene) -> list:
    _, V, _ = run(scene)
    return [i for i in range(len(scene["pos"])) if any(V[s][i] == 0 for s in range(1, len(V)))]


def check(scene) -> list:
    errs = []
    P, V, B = run(scene)
    n = len(scene["pos"])
    for s in range(len(P)):
        if len(set(P[s])) != n:
            errs.append("two cars share a cell")
            break
    col = colored(scene)
    if len(col) != 4:
        errs.append("need four coloured cars")
    if not 1 <= len(scene["taps"]) <= 3 or any(scene["colors"][t["car"]] == "gray" for t in scene["taps"]):
        errs.append("taps must be 1-3, by coloured cars")
    if any(not 2 <= t["hold"] <= 3 for t in scene["taps"]):
        errs.append("a tap must hold 2-3 steps")
    if len({t["car"] for t in scene["taps"]}) != len(scene["taps"]):
        errs.append("a car taps twice")
    stopped = ever_stopped(scene)
    if not stopped:
        errs.append("no car ever stops")
    if sum(1 for i in col.values() if i in stopped) == 0:
        errs.append("no coloured car stops")
    if all(t["car"] in stopped and len(stopped) <= 1 for t in scene["taps"]):
        errs.append("the jam never spreads")
    # every tap must change something visible (a car stops that wouldn't otherwise)
    for t in scene["taps"]:
        if set(ever_stopped(without_tap(scene, t["car"]))) == set(stopped):
            errs.append("a tap changes nothing")
    return errs


def build(rng: random.Random) -> dict:
    while True:
        n = rng.randint(10, 19)
        cells = sorted(rng.sample(range(L), n))
        # start fairly even: nudge towards equal spacing
        base = rng.randrange(L)
        cells = sorted({(base + round(k * L / n) + rng.choice([0, 0, 1])) % L for k in range(n)})
        if len(cells) != n:
            continue
        colors = ["gray"] * n
        picks = rng.sample(range(n), 4)
        for c, i in zip(COLORS, picks):
            colors[i] = c
        steps = rng.randint(*STEPS)
        k = rng.randint(1, 3)
        tappers = rng.sample(picks, k)
        taps = sorted(({"step": rng.randint(3, steps // 2), "car": c, "hold": rng.randint(2, 3)} for c in tappers),
                      key=lambda t: t["step"])
        if len({t["step"] for t in taps}) != len(taps) or any(b["step"] - a["step"] < 4 for a, b in zip(taps, taps[1:])):
            continue
        vel = [0] * n
        scene = {"steps": steps, "pos": cells, "vel": vel, "colors": colors, "taps": taps}
        # let traffic settle from a standing start before the clip: warm-up steps without taps
        P, V, _ = run(dict(scene, taps=[], steps=12))
        scene["pos"], scene["vel"] = P[-1], V[-1]
        if any(v == 0 for v in scene["vel"]):
            continue
        if not check(scene):
            return scene
