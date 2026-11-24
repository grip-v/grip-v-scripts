"""Minesweeper reveal trace (v2). A 4x4-12x12 board starts fully covered,
with the total mine count shown. Tiles are uncovered one per step; each
shows its number (blank for 0), and a few clips end by uncovering a mine.
In about 40% of clips exactly one revealed number is wrong, chosen so the
numbers become provably impossible (no mine layout fits them plus the
mine total) at some later step, not necessarily the step of the wrong
tile itself.

Rows and columns are numbered 1..n outside the board, and every position
in the stored scene and in the questions uses those 1-indexed labels.

Logical facts (satisfiable / certain mine / certain safe) come from
probability.py's exact engine, fed only the numbers a viewer has seen."""
from __future__ import annotations
import random

import probability as P

MAX_REVEALS = 40


def neighbours(r, c, n):
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0) and 1 <= r + dr <= n and 1 <= c + dc <= n]


def state(scene: dict, steps: int | None = None, skip: int | None = None, change: tuple | None = None) -> dict:
    """{(r, c): shown value} after `steps` reveals (1-indexed positions).
    `skip` drops one reveal (by step index); `change` = (step index, value)."""
    out = {}
    seq = scene["reveals"] if steps is None else scene["reveals"][:steps]
    for k, rv in enumerate(seq):
        if k == skip:
            continue
        v = change[1] if change and change[0] == k else rv["shown"]
        out[(rv["r"], rv["c"])] = v
    return out


def _engine_state(st):
    """probability.py uses 0-indexed cells."""
    return {(r - 1, c - 1): v for (r, c), v in st.items()}


def satisfiable(scene: dict, st: dict) -> bool:
    return P.is_satisfiable(scene["n"], _engine_state(st), scene["mines"])


def certainty(scene: dict, st: dict, cell: tuple) -> str:
    """'mine', 'safe' or 'unsure' for a covered cell."""
    p = P.compute_mine_probability(scene["n"], _engine_state(st), (cell[0] - 1, cell[1] - 1), scene["mines"])
    return "mine" if p == 1 else "safe" if p == 0 else "unsure"


def contradiction_step(scene: dict, **kw) -> int | None:
    """First step count after which no mine layout fits the shown numbers."""
    for s in range(1, len(scene["reveals"]) + 1):
        if not satisfiable(scene, state(scene, s, **kw)):
            return s
    return None


def _grow(rng, n, safe_set, k):
    """k safe tiles uncovered the way a player works outward: usually a
    covered safe tile touching what is already open, sometimes a fresh guess."""
    start = rng.choice(sorted(safe_set))
    order, opened = [start], {start}
    while len(order) < k:
        edge = sorted({nb for x in opened for nb in neighbours(*x, n)} & safe_set - opened)
        pool = edge if edge and rng.random() < 0.85 else sorted(safe_set - opened)
        x = rng.choice(pool)
        order.append(x)
        opened.add(x)
    return order


def build(rng: random.Random) -> dict:
    n = rng.randint(4, 12)
    while True:
        mines = max(2, round(rng.uniform(0.12, 0.2) * n * n))
        cells = [(r, c) for r in range(1, n + 1) for c in range(1, n + 1)]
        mine_set = set(rng.sample(cells, mines))
        truth = {x: sum(nb in mine_set for nb in neighbours(*x, n)) for x in cells if x not in mine_set}
        k = min(MAX_REVEALS, len(truth) - 3, max(4, round(rng.uniform(0.3, 0.75) * len(truth))))
        order = _grow(rng, n, set(truth), k)
        reveals = [{"r": r, "c": c, "shown": truth[(r, c)], "true": truth[(r, c)]} for r, c in order]
        if rng.random() < 0.12:
            r, c = rng.choice(sorted(mine_set))
            reveals.append({"r": r, "c": c, "shown": "mine", "true": "mine"})
        scene = {"n": n, "mines": mines, "mine_cells": [list(x) for x in sorted(mine_set)], "reveals": reveals,
                 "wrong_step": None}
        try:
            if rng.random() < 0.4:
                numeric = [i for i, rv in enumerate(reveals) if rv["shown"] != "mine" and i < len(reveals) - 2]
                done = False
                for i in rng.sample(numeric, min(12, len(numeric))):
                    for v in rng.sample([x for x in range(9) if x != reveals[i]["true"]], 3):
                        trial = dict(scene, reveals=[dict(x) for x in reveals])
                        trial["reveals"][i]["shown"] = v
                        s = contradiction_step(trial)
                        if s is not None and s < len(reveals):
                            scene, done = trial, True
                            scene["wrong_step"] = i
                            break
                    if done:
                        break
                if not done:
                    continue
            final = state(scene)
            covered = [x for x in cells if x not in final]
            if scene["wrong_step"] is None:
                scene["certain"] = {f"{r},{c}": certainty(scene, final, (r, c)) for r, c in covered}
                vals = list(scene["certain"].values())
                if vals.count("unsure") < 2 or (vals.count("mine") + vals.count("safe")) < 2:
                    continue
                if not any(w in vals and sum(v != w for v in vals) >= 3 for w in ("mine", "safe")):
                    continue
            scene["contradiction"] = contradiction_step(scene)
        except RuntimeError:                                    # a constraint component too large to enumerate
            continue
        return scene


def check(scene: dict) -> list[str]:
    errs = []
    n = scene["n"]
    mine_set = {tuple(x) for x in scene["mine_cells"]}
    if len(mine_set) != scene["mines"]:
        errs.append("stored mine count doesn't match the mine cells")
    seen = set()
    for k, rv in enumerate(scene["reveals"]):
        x = (rv["r"], rv["c"])
        if x in seen:
            errs.append(f"tile {x} is revealed twice")
        seen.add(x)
        if not (1 <= x[0] <= n and 1 <= x[1] <= n):
            errs.append(f"tile {x} is off the board")
        true = "mine" if x in mine_set else sum(nb in mine_set for nb in neighbours(*x, n))
        if rv["true"] != true:
            errs.append(f"tile {x} stores the wrong true value")
        if (rv["shown"] != rv["true"]) != (k == scene["wrong_step"]):
            errs.append(f"tile {x}: shown value differs from the truth on the wrong step")
    if contradiction_step(scene) != scene["contradiction"]:
        errs.append("stored contradiction step doesn't match")
    if (scene["wrong_step"] is None) != (scene["contradiction"] is None):
        errs.append("a wrong number must (and only it may) produce a contradiction")
    return errs
