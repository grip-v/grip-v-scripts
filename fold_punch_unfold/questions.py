"""5-rung ladder for fold_punch_unfold (see common/ladder5.py). Cells are
named column letter + row number (A-H left to right, 1-8 top to bottom),
as labelled around the sheet. "Unfolded" means the sheet opened back to
its original 8 x 8 square."""
from __future__ import annotations
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

WAY = {("v", 1): "right to left", ("v", -1): "left to right", ("h", 1): "bottom to top", ("h", -1): "top to bottom"}
ALL = [f"{c}{r}" for r in range(1, 9) for c in sim.COLS]
ORD = ["first", "second", "third", "fourth"]


def _cells(hs):
    return [sim.cell(x, y) for x, y in hs]


def _sorted(cells):
    return ", ".join(sorted(cells, key=lambda c: (int(c[1:]), c[0])))


def _near(rng, cells, k):
    """k cells without holes, preferring neighbours and mirror images of hole cells."""
    near = set()
    for c in cells:
        x, y = sim.COLS.index(c[0]), int(c[1:]) - 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (7 - 2 * x, 0), (0, 7 - 2 * y)):
            if 0 <= x + dx < 8 and 0 <= y + dy < 8:
                near.add(f"{sim.COLS[x + dx]}{y + dy + 1}")
    pool = sorted(near - set(cells))
    rng.shuffle(pool)
    rest = [c for c in ALL if c not in cells and c not in pool]
    rng.shuffle(rest)
    return (pool + rest)[:k]


def _sane(folds) -> bool:
    try:
        cur = sim.shape([])
        for k, f in enumerate(folds):
            if cur.intersection(sim.half_plane(f, f["move"])).area < 1 or \
                    cur.intersection(sim.half_plane(f, -f["move"])).area < 1:
                return False
            cur = sim.shape(folds[:k + 1])
        return True
    except Exception:
        return False


def build(scene: dict, rng: random.Random) -> list[dict]:
    folds, punches = scene["folds"], [tuple(p) for p in scene["punches"]]
    hs = [tuple(h) for h in scene["holes"]]
    cells = _cells(hs)
    nf = len(folds)
    qs = []

    # L1 observation
    axial = [k for k, f in enumerate(folds, 1) if f["axis"] in ("v", "h")]
    key = rng.choice(["n_folds"] * 2 + (["fold_way"] * 2 if axial else []))
    if key == "n_folds":
        opts, a = L.mc(rng, nf, L.near_ints(nf, 1))
        qs.append(L.question(1, key, "How many times is the sheet folded?", nf, opts, a, True))
    else:
        k = rng.choice(axial)
        gt = WAY[(folds[k - 1]["axis"], folds[k - 1]["move"])]
        opts, a = L.mc(rng, gt, [w for w in WAY.values() if w != gt])
        qs.append(L.question(1, key, f"Which way does the paper move in the {ORD[k - 1]} fold?", gt, opts, a, True,
                             {"k": k}))

    # L2 relational
    lay = [sim.layers(folds, p) for p in punches]
    keys = ["has_hole"] * 2 + (["more_layers"] if len(punches) == 2 else [])
    key = rng.choice(keys)
    if key == "has_hole":
        c = rng.choice(cells)
        other = _near(rng, cells, 1)[0]
        pair = [c, other]
        rng.shuffle(pair)
        opts, a = L.mc(rng, c, [other])
        qs.append(L.question(2, key, f"After the sheet is unfolded, which of these two cells has a hole: {pair[0]} "
                             f"or {pair[1]}?", c, opts, a, True, {"cells": pair}))
    else:
        gt = "the first punch" if lay[0] > lay[1] else ("the second punch" if lay[1] > lay[0] else "the same number")
        opts, a = L.mc(rng, gt, ["the first punch", "the second punch", "the same number"])
        qs.append(L.question(2, key, "The folded paper is punched twice. Which punch goes through more layers of "
                             "paper?", gt, opts, a, True))

    # L3 measurement
    key = rng.choice(["n_layers", "n_holes"])
    if key == "n_layers":
        i = rng.randrange(len(punches))
        gt = lay[i]
        which = "the punch" if len(punches) == 1 else f"the {ORD[i]} punch"
        opts, a = L.mc(rng, gt, L.near_ints(gt, 1) + [gt * 2])
        qs.append(L.question(3, key, f"How many layers of paper does {which} go through?", gt, opts, a, True,
                             {"punch": i}))
    else:
        gt = len(hs)
        opts, a = L.mc(rng, gt, [len(punches), gt * 2] + L.near_ints(gt, 1))
        qs.append(L.question(3, key, "After the sheet is unfolded, how many holes does it have?", gt, opts, a,
                             True))

    # L4 integration
    rows = sorted({int(c[1:]) for c in cells})
    cols = sorted({c[0] for c in cells})
    keys = (["hole_set"] * 2 if len(cells) <= 6 else []) + ["row_count", "col_count"]
    key = rng.choice(keys)
    if key == "hole_set":
        gt = _sorted(cells)
        wrong = []
        for _ in range(12):
            alt = list(cells)
            j = rng.randrange(len(alt))
            alt[j] = _near(rng, cells, 3)[rng.randrange(3)]
            if len(set(alt)) == len(alt):
                wrong.append(_sorted(alt))
        if len(cells) > 1:
            wrong.append(_sorted(cells[:-1]))
        wrong.append(_sorted(cells + _near(rng, cells, 1)))
        opts, a = L.mc(rng, gt, wrong)
        qs.append(L.question(4, key, "After the sheet is unfolded, which cells have holes?", gt, opts, a, True))
    elif key == "row_count":
        r = rng.choice(rows)
        gt = sum(1 for c in cells if int(c[1:]) == r)
        opts, a = L.mc(rng, gt, L.near_ints(gt, 0, 8))
        qs.append(L.question(4, key, f"After the sheet is unfolded, how many holes are in row {r}?", gt, opts, a,
                             True, {"row": r}))
    else:
        col = rng.choice(cols)
        gt = sum(1 for c in cells if c[0] == col)
        opts, a = L.mc(rng, gt, L.near_ints(gt, 0, 8))
        qs.append(L.question(4, "col_count", f"After the sheet is unfolded, how many holes are in column {col}?",
                             gt, opts, a, True, {"col": col}))

    # L5 counterfactual: change one fold, same punch positions
    cands = []
    for k in range(1, nf + 1):
        flip = [dict(f) for f in folds]
        flip[k - 1]["move"] *= -1
        skip = [f for i, f in enumerate(folds, 1) if i != k]
        for kind, alt in (("cf_flip", flip), ("cf_skip", skip)):
            if not _sane(alt) or not all(sim.clear(alt, p) for p in punches):
                continue
            ah = sim.holes(alt, punches)
            if len(set(ah)) != len(ah):
                continue
            cands.append((kind, k, ah))
    if len(punches) == 1:
        spots = [(x + 0.5, y + 0.5) for x in range(sim.N) for y in range(sim.N)
                 if (x + 0.5, y + 0.5) != punches[0] and sim.clear(folds, (x + 0.5, y + 0.5))]
        diff = [q for q in spots if sim.layers(folds, q) != lay[0]]
        if spots:
            q = rng.choice(diff or spots)
            cands += [("cf_punch_at", q, sim.holes(folds, [q]))] * max(1, len(cands) // 2)
    if not cands:
        cands = [("n_holes_again", 0, hs)]
    kind, k, ah = rng.choice(cands)
    if kind == "cf_skip" and set(_cells(ah)) != set(cells) and rng.random() < 0.5:
        kind = "cf_skip_has"
    if kind == "cf_flip":
        gt = len(ah)
        opts, a = L.mc(rng, gt, [len(hs)] + L.near_ints(gt, 1))
        qs.append(L.question(5, kind, f"Suppose that in the {ORD[k - 1]} fold the OTHER side of the same crease "
                             "had been folded over instead, with everything else (the other folds and the punch "
                             "positions on the table) the same. How many holes would the unfolded sheet have?", gt,
                             opts, a, True, {"k": k}))
    elif kind == "cf_skip":
        gt = len(ah)
        opts, a = L.mc(rng, gt, [len(hs)] + L.near_ints(gt, 1))
        qs.append(L.question(5, kind, f"Suppose the {ORD[k - 1]} fold had been skipped, with everything else (the "
                             "other folds and the punch positions on the table) the same. How many holes would the "
                             "unfolded sheet have?", gt, opts, a, True, {"k": k}))
    elif kind == "cf_punch_at":
        gt = len(ah)
        c = sim.cell(*k)
        opts, a = L.mc(rng, gt, [len(hs)] + L.near_ints(gt, 1))
        qs.append(L.question(5, kind, f"Suppose the folded paper had been punched at cell {c} (as labelled around "
                             "the sheet) instead, with the same folds. How many holes would the unfolded sheet have?",
                             gt, opts, a, True, {"at": list(k)}))
    elif kind == "cf_skip_has":
        acells = _cells(ah)
        new = [c for c in acells if c not in cells] or acells
        c = rng.choice(new)
        others = [x for x in cells if x not in acells] + _near(rng, acells, 6)
        others = [x for x in others if x not in acells][:3]
        group = [c] + others
        rng.shuffle(group)
        opts, a = L.mc(rng, c, others)
        qs.append(L.question(5, kind, f"Suppose the {ORD[k - 1]} fold had been skipped, with everything else (the "
                             "other folds and the punch positions on the table) the same. Which of these cells "
                             f"would have a hole in the unfolded sheet: {', '.join(group)}?", c, opts, a, True,
                             {"k": k, "cells": group}))
    else:
        gt = len(hs)
        opts, a = L.mc(rng, gt, L.near_ints(gt, 1))
        qs.append(L.question(5, "n_holes", "If the sheet were unfolded, how many holes would it have?", gt, opts, a,
                             True))
    return qs
