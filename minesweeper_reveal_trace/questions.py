"""5-rung ladder for minesweeper_reveal_trace (see common/ladder5.py).
Every position is written "row r, column c" using the 1-based numbers
drawn outside the board; every tile a question calls covered is still
covered in the last frame."""
from __future__ import annotations
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402


def _at(x):
    return f"row {x[0]}, column {x[1]}"


def build(scene: dict, rng: random.Random) -> list[dict]:
    n, rv = scene["n"], scene["reveals"]
    final = sim.state(scene)
    tiles = [(x["r"], x["c"]) for x in rv]
    numeric = [(x["r"], x["c"]) for x in rv if x["shown"] != "mine"]
    covered = [(r, c) for r in range(1, n + 1) for c in range(1, n + 1) if (r, c) not in final]
    qs = []

    def other_tiles(correct, pool):
        return [_at(x) for x in rng.sample(pool, len(pool)) if _at(x) != correct]

    # L1 observation
    r = rng.random()
    if r < 0.35:
        v = len(rv)
        opts, a = L.mc(rng, v, L.near_ints(v, 1))
        qs.append(L.question(1, "n_revealed", "How many tiles are uncovered by the end of the video?", v, opts, a,
                             True))
    elif r < 0.8:
        x = rng.choice(numeric)
        v = final[x]
        gt = "blank" if v == 0 else str(v)
        pool = ["blank"] + [str(k) for k in range(1, 9)]
        opts, a = L.mc(rng, gt, rng.sample([p for p in pool if p != gt], 8))
        qs.append(L.question(1, "value_at", f"What does the uncovered tile at {_at(x)} show (a number, or blank "
                             f"for zero)?", gt, opts, a, False, {"cell": list(x)}))
    else:
        gt = "yes" if any(x["shown"] == "mine" for x in rv) else "no"
        opts, a = L.yes_no(rng, gt)
        qs.append(L.question(1, "mine_uncovered", "Is a mine uncovered at any point in the video?", gt, opts, a,
                             True))

    # L2 relational
    diff = [(a_, b_) for i, a_ in enumerate(numeric) for b_ in numeric[i + 1:] if final[a_] != final[b_]]
    if diff and rng.random() < 0.45:
        x, y = rng.choice(diff)
        pair = [x, y] if rng.random() < 0.5 else [y, x]
        gt = _at(x if final[x] > final[y] else y)
        opts, a = L.mc(rng, gt, [_at(pair[0]), _at(pair[1]), "they show the same number"])
        qs.append(L.question(2, "larger", f"Which uncovered tile shows the larger number: {_at(pair[0])} or "
                             f"{_at(pair[1])}?", gt, opts, a, False, {"a": list(pair[0]), "b": list(pair[1])}))
    else:
        i, j = sorted(rng.sample(range(len(tiles)), 2))
        pair = [tiles[i], tiles[j]] if rng.random() < 0.5 else [tiles[j], tiles[i]]
        gt = _at(tiles[i])
        opts, a = L.mc(rng, gt, [_at(pair[0]), _at(pair[1]), "they are uncovered at the same time"])
        qs.append(L.question(2, "earlier", f"Which tile is uncovered first: {_at(pair[0])} or {_at(pair[1])}?",
                             gt, opts, a, True, {"a": list(pair[0]), "b": list(pair[1])}))

    # L3 measurement
    r = rng.random()
    vals = sorted({final[x] for x in numeric if final[x] > 0})
    if vals and r < 0.35:
        v = rng.choice(vals)
        cnt = sum(1 for x in rv if x["shown"] == v)
        opts, a = L.mc(rng, cnt, L.near_ints(cnt, 1))
        qs.append(L.question(3, "count_value", f"How many uncovered tiles show the number {v}?", cnt, opts, a,
                             False, {"v": v}))
    elif r < 0.7:
        k = rng.randrange(1, len(tiles))
        x = tiles[k]
        opts, a = L.mc(rng, k + 1, L.near_ints(k + 1, 1, len(tiles)))
        qs.append(L.question(3, "reveal_step", f"On which reveal step (the step counter above the board) is the "
                             f"tile at {_at(x)} uncovered?", k + 1, opts, a, True, {"cell": list(x)}))
    else:
        x = rng.choice([t for t in tiles if any(y in covered for y in sim.neighbours(*t, n))] or tiles)
        v = sum(1 for y in sim.neighbours(*x, n) if y not in final)
        opts, a = L.mc(rng, v, L.near_ints(v, 0, 8))
        qs.append(L.question(3, "covered_neighbours", f"At the end, how many of the tiles touching {_at(x)} "
                             f"(including diagonals) are still covered?", v, opts, a, False, {"cell": list(x)}))

    # L4 integration: logic over everything uncovered
    if scene["wrong_step"] is None:
        cert = {tuple(int(v) for v in k.split(",")): s for k, s in scene["certain"].items()}
        kinds = [w for w in ("mine", "safe") if any(s == w for s in cert.values())
                 and sum(1 for s in cert.values() if s != w) >= 3]
        want = rng.choice(kinds)
        x = rng.choice([c for c, s in cert.items() if s == want])
        others = rng.sample([c for c, s in cert.items() if s != want], 3)
        cells = [x] + others
        gt = _at(x)
        opts, a = L.mc(rng, gt, [_at(c) for c in others])
        key = "certain_mine" if want == "mine" else "certain_safe"
        word = "must be a mine" if want == "mine" else "is certainly safe"
        qs.append(L.question(4, key, f"Using only the numbers shown and the mine total, which of these covered "
                             f"tiles {word}?", gt, opts, a, False, {"options": [list(c) for c in cells]}))
    else:
        v = scene["contradiction"]
        opts, a = L.mc(rng, v, L.near_ints(v, 1, len(rv)))
        qs.append(L.question(4, "contradiction_step", "One uncovered number is wrong. After which reveal step do "
                             "the numbers shown first stop fitting any possible mine layout (with the stated mine "
                             "total)?", v, opts, a, True))

    # L5 counterfactual (a what-if state can be too large for the exact engine: try the next candidate)
    def first_ok(cands, fn):
        for c in cands:
            try:
                return c, fn(c)
            except RuntimeError:
                continue
        return None, None

    if scene["wrong_step"] is not None:
        c = scene["contradiction"]
        pool = [k for k in range(c) if k != scene["wrong_step"]]
        rng.shuffle(pool)
        pool = [scene["wrong_step"]] + pool if rng.random() < 0.5 else pool + [scene["wrong_step"]]
        step, still = first_ok(pool, lambda k: sim.contradiction_step(scene, skip=k) is not None)
        gt = "yes" if still else "no"
        opts, a = L.yes_no(rng, gt)
        qs.append(L.question(5, "cf_skip", f"If reveal step {step + 1} (the tile at {_at(tiles[step])}) had been "
                             f"skipped, everything else the same, would the numbers still stop fitting any mine "
                             f"layout?", gt, opts, a, True, {"step": step}))
        return qs
    cert = {tuple(int(v) for v in k.split(",")): s_ for k, s_ in scene["certain"].items()}
    hide = [(x, tiles.index(x), y, s_) for y, s_ in cert.items() if s_ != "unsure"
            for x in sim.neighbours(*y, n) if x in final and final[x] != "mine"]
    rng.shuffle(hide)
    got = None
    if hide and rng.random() < 0.5:
        h, still = first_ok(hide, lambda h: sim.certainty(scene, sim.state(scene, skip=h[1]), h[2]) == h[3])
        if h is not None:
            x, k, y, s_ = h
            gt = "yes" if still else "no"
            opts, a = L.yes_no(rng, gt)
            word = "a mine" if s_ == "mine" else "safe"
            got = L.question(5, "cf_hide", f"The covered tile at {_at(y)} is certainly {word}. If the tile at "
                             f"{_at(x)} had never been uncovered (everything else the same), would you still know "
                             f"for sure that {_at(y)} is {word}?", gt, opts, a, False,
                             {"cell": list(x), "target": list(y), "was": s_})
    if got is None:
        pool = [t for t in numeric if final[t] < 8]
        rng.shuffle(pool)
        x, ok = first_ok(pool, lambda t: sim.satisfiable(scene, sim.state(scene, change=(tiles.index(t), final[t] + 1))))
        gt = "yes" if ok else "no"
        opts, a = L.yes_no(rng, gt)
        got = L.question(5, "cf_bump", f"If the tile at {_at(x)} showed one more than it does (everything else the "
                         f"same), could all the numbers still fit some mine layout with the stated mine total?",
                         gt, opts, a, False, {"cell": list(x)})
    qs.append(got)
    return qs
