"""Independent answers for cellular_automata (used only by the validator).

Nothing here reads scene["history"] or calls sim.step. The generations are
replayed with a SPARSE method (a dict of non-empty cells; neighbour counts
from a Counter over each filled cell's neighbour offsets, with explicit
bounds / modulo for the torus) instead of the generator's dense numpy
shifts. Moments are mapped to generations frame by frame through the
renderer's own per-frame state (render.gen_at_frame) and the timer text it
draws (render.timer_text); a moment whose frames are not all one still
generation gives None (= mismatch). The rule question is answered by
testing every rule of the scenario's pool against every transition.
Phrasing helpers (labels, option wording) come from questions.py; they are
formatting only.
"""
from __future__ import annotations
from collections import Counter

import questions as Q
import render
import sim

NEIGH = {
    "moore": [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)],
    "vn": [(-1, 0), (0, -1), (0, 1), (1, 0)],
    "elem": [(0, -1), (0, 1)],
}


def _rule_sets(rule):
    b, s = rule.split("/")
    return {int(x) for x in b[1:]}, {int(x) for x in s[1:]}


def _sparse(grid) -> dict:
    return {(r, c): v for r, row in enumerate(grid) for c, v in enumerate(row) if v}


def _step(cells: dict, scene, rule) -> dict:
    B, S = _rule_sets(rule)
    R, C = scene["rows"], scene["cols"]
    cnt = Counter()
    for (r, c), v in cells.items():
        if v != 1:
            continue
        for dr, dc in NEIGH[scene["nbhd"]]:
            rr, cc = r + dr, c + dc
            if scene["wrap"]:
                rr, cc = rr % R, cc % C
            elif not (0 <= rr < R and 0 <= cc < C):
                continue
            cnt[(rr, cc)] += 1
    out = {}
    for (r, c), v in cells.items():
        if v == 1:
            if cnt[(r, c)] in S:
                out[(r, c)] = 1
            elif scene["states"] == 3:
                out[(r, c)] = 2
    for rc, n in cnt.items():
        if rc not in cells and n in B:
            out[rc] = 1
    return out


def _replay(scene, rule=None, start=None, gens=None) -> list[dict]:
    g = _sparse(scene["start"] if start is None else start)
    out = [g]
    for _ in range(scene["gens"] if gens is None else gens):
        g = _step(g, scene, rule or scene["rule"])
        out.append(g)
    return out


def _n_live(g: dict) -> int:
    return sum(1 for v in g.values() if v == 1)


def _frames(scene) -> int:
    return int(round(render.duration(scene) * sim.FPS))


def _gen_of(scene, tok):
    if tok == "start":
        return 0
    if tok == "end":
        return scene["gens"]
    fr = [f for f in range(_frames(scene)) if render.timer_text(f) == f"{tok} s"]
    states = {render.gen_at_frame(scene, f) for f in fr}
    if len(states) != 1:
        return None
    k, u = states.pop()
    return k if u >= 1 else None


def _mcell(scene, i):
    r, c = scene["markers"][i]["cell"]
    return r, c


def _first_change_frame(scene, h, i):
    rc = _mcell(scene, i)
    s0 = h[0].get(rc, 0)
    for f in range(_frames(scene)):
        k, _ = render.gen_at_frame(scene, f)
        if h[k].get(rc, 0) != s0:
            return f, k
    return None, None


def answer(key: str, p: dict, scene: dict):
    h = _replay(scene)
    G = scene["gens"]
    live = [_n_live(g) for g in h]
    one_d = scene["scenario"] == "elementary"

    if key in ("count_live", "pop_at"):
        k = _gen_of(scene, p["t"])
        return None if k is None else live[k]
    if key == "count_fading":
        k = _gen_of(scene, p["t"])
        return None if k is None else sum(1 for v in h[k].values() if v == 2)
    if key == "markers_filled":
        k = _gen_of(scene, p["t"])
        if k is None:
            return None
        i, j = p["pair"]
        return Q.combo(scene, p["pair"], h[k].get(_mcell(scene, i)) == 1, h[k].get(_mcell(scene, j)) == 1)
    if key in ("more_live", "trend", "cell_change", "n_born", "n_lost"):
        a, b = _gen_of(scene, p["ta"]), _gen_of(scene, p["tb"])
        if a is None or b is None:
            return None
        if key == "more_live":
            if live[a] == live[b]:
                return "they have the same number"
            return Q.lab(scene, p["ta"] if live[a] > live[b] else p["tb"])
        if key == "trend":
            return "more" if live[b] > live[a] else "fewer" if live[b] < live[a] else "the same number"
        if key == "cell_change":
            if b != a + 1:
                return None
            rc = _mcell(scene, p["m"])
            return Q.outcome(scene, h[a].get(rc, 0), h[b].get(rc, 0))
        if b != a + 1:
            return None
        if key == "n_born":
            return sum(1 for rc, v in h[b].items() if v == 1 and rc not in h[a])
        return sum(1 for rc, v in h[a].items() if v == 1 and h[b].get(rc) != 1)
    if key == "marker_longer":
        i, j = p["pair"]
        ni, nj = (sum(1 for g in h if g.get(_mcell(scene, x)) == 1) for x in (i, j))
        return "they are equal" if ni == nj else Q.mshort(scene, i if ni > nj else j)
    if key == "marker_first":
        i, j = p["pair"]
        fi, fj = _first_change_frame(scene, h, i)[0], _first_change_frame(scene, h, j)[0]
        if fi is None and fj is None:
            return "neither ever changes"
        if fi == fj:
            return "they change at the same moment"
        first = i if fj is None or (fi is not None and fi < fj) else j
        return Q.mshort(scene, first)
    if key == "peak_moment":
        ks = [_gen_of(scene, t) for t in p["ts"]]
        if None in ks:
            return None
        best = max(live[k] for k in ks)
        win = [t for t, k in zip(p["ts"], ks) if live[k] == best]
        return Q.lab(scene, win[0]) if len(win) == 1 else None
    if key == "marker_count":
        rc = _mcell(scene, p["m"])
        return sum(1 for g in h if g.get(rc) == 1)
    if key == "first_change_time":
        f, _ = _first_change_frame(scene, h, p["m"])
        return None if f is None else render.timer_text(f)
    if key == "rule":
        pool = sim.POOLS[sim.SCENARIO_SETUP[scene["scenario"]][0]]
        fits = [r for r in pool if all(_step(a, scene, r) == b for a, b in zip(h, h[1:]))]
        return sim.describe(fits[0], scene["scenario"]) if len(fits) == 1 else None
    if key in ("always_filled", "ever_filled"):
        if one_d:
            cols = [{c for (r, c), v in g.items() if v == 1} for g in h]
        else:
            cols = [{rc for rc, v in g.items() if v == 1} for g in h]
        return len(set.intersection(*cols)) if key == "always_filled" else len(set.union(*cols))
    if key == "total_born":
        return sum(sum(1 for rc, v in b.items() if v == 1 and rc not in a) for a, b in zip(h, h[1:]))
    if key == "empty_time":
        for f in range(_frames(scene)):
            k, _ = render.gen_at_frame(scene, f)
            if k > 0 and not h[k]:
                return render.timer_text(f)
        return ("never: every row has a coloured cell" if one_d else "never: some cells remain to the end")
    if key == "biggest_change_time":
        diff = []
        for a, b in zip(h, h[1:]):
            diff.append(sum(1 for rc in set(a) | set(b) if a.get(rc, 0) != b.get(rc, 0)))
        best = max(diff)
        if diff.count(best) != 1:
            return None
        kb = diff.index(best) + 1
        f = next(f for f in range(_frames(scene)) if render.gen_at_frame(scene, f)[0] == kb)
        return render.timer_text(f)
    if key == "cf_next":
        more = _replay(scene, start=[[h[G].get((r, c), 0) for c in range(scene["cols"])]
                                     for r in range(scene["rows"])], gens=p["n"])
        return _n_live(more[-1])
    if key == "cf_markers_next":
        g = _step(h[G], scene, scene["rule"])
        i, j = p["pair"]
        return Q.combo(scene, p["pair"], g.get(_mcell(scene, i)) == 1, g.get(_mcell(scene, j)) == 1)
    if key == "cf_rule":
        return _n_live(_replay(scene, rule=p["rule"], gens=p["k"])[-1])
    if key == "cf_toggle":
        start = [list(row) for row in scene["start"]]
        r, c = _mcell(scene, p["m"])
        start[r][c] = 1 - start[r][c]
        return _n_live(_replay(scene, start=start, gens=p["k"])[-1])
    return None
