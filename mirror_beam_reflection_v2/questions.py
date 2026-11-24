"""5-rung ladder for mirror_beam_reflection_v2 (see common/ladder5.py). Every rung has >= 2 variants."""
from __future__ import annotations
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

SIDE = {(0, 1): "left", (0, -1): "right", (1, 0): "top", (-1, 0): "bottom"}


def cl(cell):
    return f"row {cell[0]}, col {cell[1]}"


def boundary(n):
    return [(r, c) for r in range(n) for c in range(n) if r in (0, n - 1) or c in (0, n - 1)]


def key(cell):
    return f"{cell[0]}-{cell[1]}"


def cf_options(rng, n, gt_cell):
    gt = cl(gt_cell)
    pool = [cl(c) for c in boundary(n) if cl(c) != gt]
    return L.mc(rng, gt, rng.sample(pool, 3))


def build(scene: dict, rng: random.Random) -> list[dict]:
    n, mirrors, shown = scene["grid_n"], scene["mirrors"], scene["truncate_index"]
    start, d = scene["start"], scene["dir"]
    path, _, edge = sim.trace(mirrors, start, d, n)
    hit_idx = [i for i in range(1, len(path)) if key(path[i]) in mirrors]       # path index of each bounce event
    hits = [path[i] for i in hit_idx]
    shown_hits = [path[i] for i in hit_idx if i <= shown]
    exit_cell = path[-1]
    seen_cells = set(path[:shown + 1])
    qs = []

    # L1 observation
    v = rng.choice(["mirror_count", "slash_mirror_count", "backslash_mirror_count", "entry_side"])
    if v == "entry_side":
        gt = SIDE[tuple(d)]
        opts, a = L.mc(rng, gt, [e for e in sim.EDGES if e != gt])
        qs.append(L.question(1, v, "Through which side of the grid does the beam enter?", gt, opts, a, False))
    else:
        kind = {"mirror_count": None, "slash_mirror_count": "/", "backslash_mirror_count": "\\"}[v]
        gt = len(mirrors) if kind is None else sum(1 for m in mirrors.values() if m == kind)
        opts, a = L.mc(rng, gt, L.near_ints(gt, 0))
        text = ("How many mirrors are on the grid?" if kind is None else
                "How many of the mirrors are drawn as '/' (rising to the right)?" if kind == "/" else
                "How many of the mirrors are drawn as '\\' (falling to the right)?")
        qs.append(L.question(1, v, text, gt, opts, a, False))

    # L2 relational
    v = rng.choice(["bounces_shown", "first_mirror_hit", "third_mirror_hit"])
    if v == "bounces_shown":
        gt = len(shown_hits)
        opts, a = L.mc(rng, gt, L.near_ints(gt, 0))
        qs.append(L.question(2, v, "How many times does the beam bounce off a mirror in the video?",
                             gt, opts, a, True))
    else:
        k = 0 if v == "first_mirror_hit" else 2
        gt = cl(hits[k])
        others = [cl(tuple(int(x) for x in m.split("-"))) for m in mirrors]
        others = [o for o in others if o != gt]
        rng.shuffle(others)
        opts, a = L.mc(rng, gt, others)
        word = "first" if k == 0 else "third"
        qs.append(L.question(2, v, f"Which mirror does the beam hit {word} (counting each bounce in order)?",
                             gt, opts, a, True))

    # L3 measurement
    v = rng.choice(["cells_shown", "slash_hits_shown", "backslash_hits_shown"])
    if v == "cells_shown":
        gt = len(seen_cells)
        opts, a = L.mc(rng, gt, L.near_ints(gt, 1))
        qs.append(L.question(3, v, "How many different grid cells has the beam passed through by the end "
                             "of the video (counting the entry cell and the cell the beam head is in; a cell "
                             "crossed twice counts once)?", gt, opts, a, True))
    else:
        kind = "/" if v == "slash_hits_shown" else "\\"
        gt = sum(1 for b in shown_hits if mirrors[key(b)] == kind)
        opts, a = L.mc(rng, gt, L.near_ints(gt, 0))
        qs.append(L.question(3, v, f"How many of the beam's bounces in the video are off a '{kind}' mirror "
                             f"(a mirror hit twice counts twice)?", gt, opts, a, True))

    # L4 integration
    v = rng.choice(["exit_edge", "exit_cell"])
    if v == "exit_edge":
        opts, a = L.mc(rng, edge, [e for e in sim.EDGES if e != edge])
        qs.append(L.question(4, v, "The video stops before the beam leaves the grid. Following the "
                             "reflection rule, through which edge does it finally exit?", edge, opts, a, True))
    else:
        gt = cl(exit_cell)
        opts, a = cf_options(rng, n, exit_cell)
        qs.append(L.question(4, v, "The video stops before the beam leaves the grid. In which edge cell "
                             "is the beam when it exits?", gt, opts, a, True))

    # L5 counterfactual: only on cells the video never showed the beam in
    unseen_free = [c for c in path[shown + 1:] if key(c) not in mirrors and c not in seen_cells
                   and list(c) != list(start)]
    unseen_mirrors = [c for c in path[shown + 1:] if key(c) in mirrors and c not in seen_cells]
    base = (edge, cl(exit_cell))
    variants = ["cf_insert_exit_edge", "cf_insert_exit_cell", "cf_remove_exit_edge",
                "cf_flip_exit_edge", "cf_flip_exit_cell"]
    rng.shuffle(variants)
    for v in variants:
        if v.startswith("cf_insert"):
            pool = unseen_free if rng.random() < 0.7 else [
                (r, c) for r in range(n) for c in range(n) if key((r, c)) not in mirrors
                and (r, c) not in seen_cells and [r, c] != list(start)]
            if not pool:
                continue
            cell, kind = rng.choice(pool), rng.choice(["/", "\\"])
            new = {**mirrors, key(cell): kind}
            params = {"cell": list(cell), "kind": kind}
            what = f"Suppose a '{kind}' mirror were also placed at {cl(cell)}."
        elif v == "cf_remove_exit_edge":
            if not unseen_mirrors:
                continue
            cell = rng.choice(unseen_mirrors)
            new = {m: k for m, k in mirrors.items() if m != key(cell)}
            params = {"cell": list(cell)}
            what = f"Suppose the mirror at {cl(cell)} were removed."
        else:
            if not unseen_mirrors:
                continue
            cell = rng.choice(unseen_mirrors)
            flipped = "\\" if mirrors[key(cell)] == "/" else "/"
            new = {**mirrors, key(cell): flipped}
            params = {"cell": list(cell)}
            what = (f"Suppose the mirror at {cl(cell)} were flipped (a '/' becomes '\\' and vice versa).")
        out = sim.trace(new, start, d, n)
        if not out:
            continue
        p2, _, e2 = out
        is_edge = v.endswith("exit_edge")
        if (e2 if is_edge else cl(p2[-1])) == (base[0] if is_edge else base[1]) and rng.random() < 0.85:
            continue                                   # prefer counterfactuals that change the outcome
        if is_edge:
            opts, a = L.mc(rng, e2, [e for e in sim.EDGES if e != e2])
            qs.append(L.question(5, v, f"{what} Through which edge would the beam exit?", e2, opts, a, True, params))
        else:
            opts, a = cf_options(rng, n, p2[-1])
            qs.append(L.question(5, v, f"{what} In which edge cell would the beam exit?", cl(p2[-1]), opts, a,
                                 True, params))
        return qs
    # fallback: always-available insertion on any unseen free cell
    cell = next(c for c in [(r, c) for r in range(n) for c in range(n)]
                if key(c) not in mirrors and c not in seen_cells and list(c) != list(start)
                and sim.trace({**mirrors, key(c): "/"}, start, d, n))
    out = sim.trace({**mirrors, key(cell): "/"}, start, d, n)
    opts, a = L.mc(rng, out[2], [e for e in sim.EDGES if e != out[2]])
    qs.append(L.question(5, "cf_insert_exit_edge", f"Suppose a '/' mirror were also placed at {cl(cell)}. "
                         "Through which edge would the beam exit?", out[2], opts, a, True,
                         {"cell": list(cell), "kind": "/"}))
    return qs
