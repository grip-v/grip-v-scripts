"""10-question, 5-rung ladder for cellular_automata (common/ladder5.py, LADDER = "ten").

Nothing in the frame is labelled except the timer, so questions name things
by what is drawn:
  * colours: "navy blue cells" are the filled cells (live), "pink cells" the
    fading ones (generations scenario), "empty" cells are the plain ones;
  * marked cells: "the cell with the red ring" (2-D) or "the column under
    the red triangle" (elementary);
  * moments: "at the start", "at the end", or a timer reading such as
    "at 3.4 s" that falls >= 2 frames inside one still generation. In the
    elementary scenario a moment picks a row: "the top row", "the bottom
    row", "the lowest row drawn at 3.4 s".

Moments are stored in params as tokens ("start", "end" or "3.4"); answers.py
maps them back to generations with its own timeline. Every question stores
params["min_fps"] (sim.min_fps over the generations it relies on).

Templates per level (derivation keys):
  L1 count_live, markers_filled, count_fading
  L2 more_live, cell_change, trend, marker_longer, marker_first
  L3 pop_at, n_born, n_lost, peak_moment, marker_count, first_change_time
  L4 rule, always_filled, total_born, ever_filled, empty_time, biggest_change_time
  L5 cf_next, cf_markers_next, cf_rule, cf_toggle
"""
from __future__ import annotations
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
MAX_COUNT = 40


# ---- phrasing (shared with answers.py, formatting only) ---------------------
def is1d(scene) -> bool:
    return scene["scenario"] == "elementary"


def state_name(scene, v: int) -> str:
    st = scene["style"]
    return "empty" if v == 0 else st["live_name"] if v == 1 else st["dying_name"]


def lab(scene, tok: str) -> str:
    if is1d(scene):
        return {"start": "the top row", "end": "the bottom row"}.get(tok, f"the lowest row drawn at {tok} s")
    return {"start": "at the start", "end": "at the end"}.get(tok, f"at {tok} s")


def bare(scene, tok: str) -> str:
    """Label without the leading "at" (for "between ... and ..." phrasing)."""
    if is1d(scene):
        return lab(scene, tok)
    return {"start": "the start", "end": "the end"}.get(tok, f"{tok} s")


def mname(scene, i: int) -> str:
    m = scene["markers"][i]
    if is1d(scene):
        return f"the column under the {m['colour']} {m['shape']}"
    return f"the cell with the {m['colour']} {m['shape']}"


def mshort(scene, i: int) -> str:
    m = scene["markers"][i]
    return f"the {m['colour']} {m['shape']}"


def combo(scene, pair, a: bool, b: bool) -> str:
    if a and b:
        return "both"
    if not a and not b:
        return "neither"
    return f"only {mshort(scene, pair[0] if a else pair[1])}"


def combo_options(scene, pair) -> list[str]:
    return [combo(scene, pair, a, b) for a, b in ((True, False), (False, True), (True, True), (False, False))]


def outcome(scene, a: int, b: int) -> str:
    if a == b:
        return f"it stays {state_name(scene, a)}"
    return f"it turns from {state_name(scene, a)} to {state_name(scene, b)}"


def outcomes(scene) -> list[tuple[int, int]]:
    if scene["states"] == 3:
        return [(0, 0), (0, 1), (1, 1), (1, 2), (2, 0)]
    return [(0, 0), (0, 1), (1, 1), (1, 0)]


def t_of_frame(f: int) -> str:
    return f"{sim.fmt_t(sim.tenths(f))} s"


def rule_preamble(scene) -> str:
    sc = scene["scenario"]
    live, dying = scene["style"]["live_name"], scene["style"]["dying_name"]
    if sc == "elementary":
        return ("Each new row is worked out from the row above it; cells beyond the ends of a row count as empty. ")
    if sc == "torus":
        pre = "The grid wraps around: its left edge touches its right edge and its top touches its bottom. "
    else:
        pre = "Cells beyond the edge of the grid count as empty. "
    if sc == "von_neumann":
        pre += "Only the 4 side neighbours (up, down, left, right) count. "
    if sc == "generations":
        pre += (f"A {live} cell that does not stay {live} turns {dying} for one step and then empties; "
                f"{dying} cells are not counted as neighbours and cannot fill. ")
    return pre


# ---- helpers ------------------------------------------------------------------
def clear(a: int, b: int) -> bool:
    """Counts differ by a visible margin (>= 2 and >= 10 %)."""
    return abs(a - b) >= max(2, math.ceil(0.1 * max(a, b)))


class Ctx:
    def __init__(self, scene):
        self.s = scene
        self.h = [np.array(g) for g in scene["history"]]
        self.G = scene["gens"]
        self.live = [int((g == 1).sum()) for g in self.h]
        self.fading = [int((g == 2).sum()) for g in self.h]
        self.tok = ["start"] + [sim.time_ref(scene, k) for k in range(1, self.G)] + ["end"]
        self.all = list(range(self.G + 1))
        self.nm = len(scene["markers"])
        self.pairs = [(i, j) for i in range(self.nm) for j in range(i + 1, self.nm)]
        self.livename = scene["style"]["live_name"]

    def mstate(self, i, k):
        r, c = self.s["markers"][i]["cell"]
        return int(self.h[k][r, c])

    def fps(self, gens):
        return sim.min_fps(self.s, gens)

    def where(self, k):
        if is1d(self.s):
            return f"in {lab(self.s, self.tok[k])}"
        return lab(self.s, self.tok[k])

    def nxt(self, n=1, rule=None, start=None):
        sc = self.s
        g = self.h[-1] if start is None else np.array(start)
        for _ in range(n):
            g = sim.step(g, rule or sc["rule"], sc["nbhd"], sc["wrap"], sc["states"])
        return g


def _q(level, key, text, gt, opts, letter, motion, params):
    return L.question(level, key, text, gt, opts, letter, motion, params)


def _num(rng, gt, plausible):
    plaus = [p for p in plausible if p != gt and p >= 0]
    rng.shuffle(plaus)
    return L.mc(rng, gt, plaus[:2] + L.near_ints(gt, 0))


def _strs(rng, gt, others):
    others = [o for o in dict.fromkeys(others) if o != gt]
    rng.shuffle(others)
    opts, a = L.mc(rng, gt, others[:3])
    assert len(opts) == 4, (gt, others)
    return opts, a


# ---- level 1 -----------------------------------------------------------------
def level1(c: Ctx):
    s, out = c.s, []
    for k in (0, c.G):
        if 1 <= c.live[k] <= MAX_COUNT:
            def mk(rng, k=k):
                gt = c.live[k]
                o, a = _num(rng, gt, [c.live[1 if k == 0 else c.G - 1]])
                where = c.where(k)
                text = (f"How many {c.livename} cells are there {where}?" if not is1d(s)
                        else f"How many {c.livename} cells are {where}?")
                return _q(1, "count_live", text, gt, o, a, False, {"t": c.tok[k], "min_fps": c.fps([k])})
            out.append(("count_live", c.tok[k], mk))
        for pair in c.pairs:
            def mk(rng, k=k, pair=pair):
                a_, b_ = c.mstate(pair[0], k) == 1, c.mstate(pair[1], k) == 1
                gt = combo(s, pair, a_, b_)
                o, a = _strs(rng, gt, combo_options(s, pair))
                if is1d(s):
                    text = (f"In {lab(s, c.tok[k])}, which of these columns has a {c.livename} cell: "
                            f"{mname(s, pair[0])} or {mname(s, pair[1])}?")
                else:
                    text = (f"Which of these cells are {c.livename} {lab(s, c.tok[k])}: "
                            f"{mname(s, pair[0])} or {mname(s, pair[1])}?")
                return _q(1, "markers_filled", text, gt, o, a, False,
                          {"t": c.tok[k], "pair": list(pair), "min_fps": c.fps([k])})
            out.append(("markers_filled", (k, pair), mk))
    if s["states"] == 3 and 1 <= c.fading[c.G] <= MAX_COUNT:
        def mk(rng):
            gt = c.fading[c.G]
            o, a = _num(rng, gt, [c.live[c.G], c.fading[c.G - 1]])
            return _q(1, "count_fading", f"How many {s['style']['dying_name']} cells are there at the end?", gt, o, a,
                      False, {"t": "end", "min_fps": c.fps([c.G])})
        out.append(("count_fading", "end", mk))
    return out


# ---- level 2 -----------------------------------------------------------------
def level2(c: Ctx):
    s, out, G = c.s, [], c.G
    for x in range(G + 1):
        for y in range(x + 1, G + 1):
            if clear(c.live[x], c.live[y]) and max(c.live[x], c.live[y]) > 0:
                def mk(rng, x=x, y=y):
                    lx, ly = lab(s, c.tok[x]), lab(s, c.tok[y])
                    gt = lx if c.live[x] > c.live[y] else ly
                    o, a = _strs(rng, gt, [lx, ly, "they have the same number", "neither has any"])
                    text = (f"Which row has more {c.livename} cells: {lx} or {ly}?" if is1d(s) else
                            f"When are there more {c.livename} cells: {lx} or {ly}?")
                    return _q(2, "more_live", text, gt, o, a, True,
                              {"ta": c.tok[x], "tb": c.tok[y], "min_fps": c.fps([x, y])})
                out.append(("more_live", (x, y), mk))
    for i in range(c.nm):
        for k in range(G):
            a0, b0 = c.mstate(i, k), c.mstate(i, k + 1)
            if a0 == b0 and (i + k) % 3:      # keep most "stays the same" variants out
                continue
            def mk(rng, i=i, k=k, a0=a0, b0=b0):
                gt = outcome(s, a0, b0)
                o, a = _strs(rng, gt, [outcome(s, *p) for p in outcomes(s)])
                if is1d(s):
                    text = (f"In {mname(s, i)}, compare the cell in {lab(s, c.tok[k])} with the cell just below it "
                            f"(in {lab(s, c.tok[k + 1])}). Which describes the change?")
                else:
                    text = f"What happens to {mname(s, i)} between {bare(s, c.tok[k])} and {bare(s, c.tok[k + 1])}?"
                return _q(2, "cell_change", text, gt, o, a, True,
                          {"m": i, "ta": c.tok[k], "tb": c.tok[k + 1], "min_fps": c.fps([k, k + 1])})
            out.append(("cell_change", (i, k), mk))
    if clear(c.live[0], c.live[G]):
        def mk(rng):
            gt = "more" if c.live[G] > c.live[0] else "fewer"
            o, a = _strs(rng, gt, ["more", "fewer", "the same number", "none at either time"])
            text = (f"Compared with the top row, does the bottom row have more or fewer {c.livename} cells?"
                    if is1d(s) else f"Compared with the start, are there more or fewer {c.livename} cells at the end?")
            return _q(2, "trend", text, gt, o, a, True, {"ta": "start", "tb": "end", "min_fps": c.fps([0, G])})
        out.append(("trend", 0, mk))
    for p in c.pairs:
        na, nb = (sum(c.mstate(i, k) == 1 for k in c.all) for i in p)
        if na != nb:
            def mk(rng, p=p, na=na, nb=nb):
                gt = mshort(s, p[0] if na > nb else p[1])
                o, a = _strs(rng, gt, [mshort(s, p[0]), mshort(s, p[1]), "they are equal", "neither is ever filled"])
                text = (f"Over the whole diagram, which column holds more {c.livename} cells: {mname(s, p[0])} or "
                        f"{mname(s, p[1])}?" if is1d(s) else
                        f"Which cell is {c.livename} in more of the generations shown: {mname(s, p[0])} or "
                        f"{mname(s, p[1])}?")
                return _q(2, "marker_longer", text, gt, o, a, True, {"pair": list(p), "min_fps": c.fps(c.all)})
            out.append(("marker_longer", p, mk))
        fa, fb = (_first_change(c, i) for i in p)
        if fa != fb:
            def mk(rng, p=p, fa=fa, fb=fb):
                first = p[0] if (fb is None or (fa is not None and fa < fb)) else p[1]
                gt = mshort(s, first)
                o, a = _strs(rng, gt, [mshort(s, p[0]), mshort(s, p[1]), "they change at the same moment",
                                       "neither ever changes"])
                text = (f"Going down the diagram, which column is the first to show a cell of a different colour "
                        f"from its top cell: {mname(s, p[0])} or {mname(s, p[1])}?" if is1d(s) else
                        f"Which cell changes colour first: {mname(s, p[0])} or {mname(s, p[1])}?")
                return _q(2, "marker_first", text, gt, o, a, True,
                          {"pair": list(p), "min_fps": c.fps(list(range(max(x for x in (fa, fb) if x) + 1)))})
            out.append(("marker_first", p, mk))
    return out


def _first_change(c: Ctx, i):
    for k in range(1, c.G + 1):
        if c.mstate(i, k) != c.mstate(i, 0):
            return k
    return None


# ---- level 3 -----------------------------------------------------------------
def level3(c: Ctx):
    s, out, G = c.s, [], c.G
    for k in range(1, G):
        if 1 <= c.live[k] <= MAX_COUNT:
            def mk(rng, k=k):
                gt = c.live[k]
                o, a = _num(rng, gt, [c.live[k - 1], c.live[k + 1]])
                text = (f"How many {c.livename} cells are {c.where(k)}?" if is1d(s)
                        else f"How many {c.livename} cells are there {c.where(k)}?")
                return _q(3, "pop_at", text, gt, o, a, True, {"t": c.tok[k], "min_fps": c.fps([k])})
            out.append(("pop_at", k, mk))
    for k in range(G):
        born = int(((c.h[k] == 0) & (c.h[k + 1] == 1)).sum())
        lost = int(((c.h[k] == 1) & (c.h[k + 1] != 1)).sum())
        for key, val in (("n_born", born), ("n_lost", lost)):
            if 1 <= val <= 25:
                def mk(rng, k=k, key=key, val=val):
                    other = lost if key == "n_born" else born
                    o, a = _num(rng, val, [other, abs(c.live[k + 1] - c.live[k])])
                    a_, b_ = bare(s, c.tok[k]), bare(s, c.tok[k + 1])
                    if is1d(s):
                        text = (f"Compare {a_} with the row just below it ({b_}). In how many columns is the cell "
                                + (f"empty above but {c.livename} below?" if key == "n_born"
                                   else f"{c.livename} above but not {c.livename} below?"))
                    else:
                        text = (f"How many cells turn {c.livename} between {a_} and {b_}?" if key == "n_born" else
                                f"How many {c.livename} cells stop being {c.livename} between {a_} and {b_}?")
                    return _q(3, key, text, val, o, a, True,
                              {"ta": c.tok[k], "tb": c.tok[k + 1], "min_fps": c.fps([k, k + 1])})
                out.append((key, k, mk))
    if G + 1 >= 4:
        top = max(range(G + 1), key=lambda k: (c.live[k], -k))
        rest = [k for k in range(G + 1) if k != top and clear(c.live[top], c.live[k]) and c.live[k] < c.live[top]]
        if len(rest) >= 3:
            def mk(rng, top=top, rest=rest):
                ks = sorted([top] + rng.sample(rest, 3))
                gt = lab(s, c.tok[top])
                o, a = _strs(rng, gt, [lab(s, c.tok[k]) for k in ks])
                text = (f"Which of these rows has the most {c.livename} cells?" if is1d(s) else
                        f"At which of these moments are there the most {c.livename} cells?")
                return _q(3, "peak_moment", text, gt, o, a, True,
                          {"ts": [c.tok[k] for k in ks], "min_fps": c.fps(ks)})
            out.append(("peak_moment", 0, mk))
    for i in range(c.nm):
        n = sum(c.mstate(i, k) == 1 for k in c.all)
        def mk(rng, i=i, n=n):
            o, a = _num(rng, n, [c.G + 1 - n])
            text = (f"How many {c.livename} cells are there in {mname(s, i)}, counting every row?" if is1d(s) else
                    f"Counting the starting grid and every change after it, in how many of the {c.G + 1} "
                    f"generations shown is {mname(s, i)} {c.livename}?")
            return _q(3, "marker_count", text, n, o, a, True, {"m": i, "min_fps": c.fps(c.all)})
        out.append(("marker_count", i, mk))
        fk = _first_change(c, i)
        if fk is not None and G >= 4:
            def mk(rng, i=i, fk=fk):
                gt = t_of_frame(sim.change_frame(s, fk))
                others = [t_of_frame(sim.change_frame(s, k)) for k in range(1, G + 1) if k != fk]
                o, a = _strs(rng, gt, others)
                text = (f"At about what time is the first cell drawn in {mname(s, i)} whose colour differs from the "
                        f"top cell of that column?" if is1d(s) else
                        f"At about what time does {mname(s, i)} first change colour?")
                return _q(3, "first_change_time", text, gt, o, a, True,
                          {"m": i, "min_fps": c.fps(list(range(fk + 1)))})
            out.append(("first_change_time", i, mk))
    return out


# ---- level 4 -----------------------------------------------------------------
def level4(c: Ctx):
    s, out, G = c.s, [], c.G
    pool = sim.POOLS[sim.SCENARIO_SETUP[s["scenario"]][0]]

    def mk_rule(rng):
        gt = sim.describe(s["rule"], s["scenario"])
        b, sv = sim.parse(s["rule"])
        others = [r for r in pool if r != s["rule"]]
        near = [r for r in others if sim.parse(r)[0] == b or sim.parse(r)[1] == sv]
        rng.shuffle(near)
        far = [r for r in others if r not in near]
        rng.shuffle(far)
        picks = (near[:2] + far)[:3]
        o, a = _strs(rng, gt, [sim.describe(r, s["scenario"]) for r in picks])
        ask = "Which rule are the rows following?" if is1d(s) else "Which rule is the grid following?"
        return _q(4, "rule", rule_preamble(s) + ask, gt, o, a, True, {"min_fps": c.fps(c.all)})
    out.append(("rule", 0, mk_rule))
    if is1d(s):
        always = int(np.all(np.array([g[0] for g in c.h]) == 1, axis=0).sum())
        ever = int(np.any(np.array([g[0] for g in c.h]) == 1, axis=0).sum())
    else:
        always = int(np.logical_and.reduce([g == 1 for g in c.h]).sum())
        ever = int(np.logical_or.reduce([g == 1 for g in c.h]).sum())
    total_born = sum(int(((a == 0) & (b == 1)).sum()) for a, b in zip(c.h, c.h[1:]))
    texts = {
        "always_filled": (f"How many columns are {c.livename} in every row, from the top row to the bottom row?"
                          if is1d(s) else f"How many cells stay {c.livename} for the whole clip, from the start "
                                          f"to the end?"),
        "ever_filled": (f"How many columns contain at least one {c.livename} cell anywhere in the diagram?"
                        if is1d(s) else f"How many different cells are {c.livename} at some moment of the clip?"),
        "total_born": (f"Summed over every pair of neighbouring rows, how many times is a {c.livename} cell drawn "
                       f"directly below an empty cell?" if is1d(s) else
                       f"Over the whole clip, how many times does an empty cell turn {c.livename}?"),
    }
    for key, val, plaus in (("always_filled", always, [ever, c.live[0]]), ("ever_filled", ever, [always, max(c.live)]),
                            ("total_born", total_born, [ever, sum(c.live[1:]) - c.live[0]])):
        if 1 <= val <= MAX_COUNT:
            def mk(rng, key=key, val=val, plaus=plaus):
                o, a = _num(rng, val, plaus)
                return _q(4, key, texts[key], val, o, a, True, {"min_fps": c.fps(c.all)})
            out.append((key, 0, mk))
    empty = [k for k in range(1, G + 1) if not c.h[k].any()]
    if empty and G >= 4:
        def mk_empty(rng, ke=empty[0]):
            never = "never: some cells remain to the end" if not is1d(s) else "never: every row has a coloured cell"
            gt = t_of_frame(sim.change_frame(s, ke))
            others = [t_of_frame(sim.change_frame(s, k)) for k in range(1, G + 1) if k != ke]
            rng.shuffle(others)
            o, a = _strs(rng, gt, others[:2] + [never])
            text = ("At about what time is the first completely empty row drawn?" if is1d(s) else
                    "At about what time does the grid first become completely empty?")
            return _q(4, "empty_time", text, gt, o, a, True, {"min_fps": c.fps(list(range(ke + 1)))})
        out.append(("empty_time", 0, mk_empty))
    changed = [int((a != b).sum()) for a, b in zip(c.h, c.h[1:])]
    order = sorted(range(G), key=lambda k: -changed[k])
    if G >= 4 and clear(changed[order[0]], changed[order[1]]):
        def mk_big(rng, kb=order[0] + 1):
            gt = t_of_frame(sim.change_frame(s, kb))
            o, a = _strs(rng, gt, [t_of_frame(sim.change_frame(s, k)) for k in range(1, G + 1) if k != kb])
            text = ("At about what time is the row drawn that differs from the row above it in the most columns?"
                    if is1d(s) else "At about what time does the biggest change happen, with the most cells changing "
                                    "colour at once?")
            return _q(4, "biggest_change_time", text, gt, o, a, True, {"min_fps": c.fps(c.all)})
        out.append(("biggest_change_time", 0, mk_big))
    return out


# ---- level 5 -----------------------------------------------------------------
def level5(c: Ctx):
    s, out, G = c.s, [], c.G
    for n in (1, 2):
        val = sim.live_count(c.nxt(n))
        if val <= MAX_COUNT and (n == 1 or val != sim.live_count(c.nxt(1))):
            def mk(rng, n=n, val=val):
                o, a = _num(rng, val, [c.live[G], c.live[G - 1]])
                if is1d(s):
                    text = (f"If the clip went on and drew {'one more row' if n == 1 else 'two more rows'} under the "
                            f"same rule, how many {c.livename} cells would the {'new' if n == 1 else 'last new'} row "
                            f"have?")
                else:
                    text = (f"If the grid kept going under the same rule for {'one more step' if n == 1 else 'two more steps'}"
                            f" after the end, how many {c.livename} cells would there be?")
                return _q(5, "cf_next", text, val, o, a, True, {"n": n, "min_fps": c.fps(c.all)})
            out.append(("cf_next", n, mk))
    nx = c.nxt(1)
    for p in c.pairs:
        def mk(rng, p=p):
            (r1, c1), (r2, c2) = (s["markers"][i]["cell"] for i in p)
            gt = combo(s, p, nx[r1, c1] == 1, nx[r2, c2] == 1)
            o, a = _strs(rng, gt, combo_options(s, p))
            text = (f"If one more row were drawn under the same rule, which of these columns would have a "
                    f"{c.livename} cell in it: {mname(s, p[0])} or {mname(s, p[1])}?" if is1d(s) else
                    f"If the grid kept going for one more step under the same rule, which of these cells would be "
                    f"{c.livename}: {mname(s, p[0])} or {mname(s, p[1])}?")
            return _q(5, "cf_markers_next", text, gt, o, a, True, {"pair": list(p), "min_fps": c.fps(c.all)})
        out.append(("cf_markers_next", p, mk))
    pool = sim.POOLS[sim.SCENARIO_SETUP[s["scenario"]][0]]
    for alt in pool:
        if alt == s["rule"]:
            continue
        for k in (1, 2):
            val = sim.live_count(c.nxt(k, rule=alt, start=c.h[0]))
            if val != c.live[k] and val <= MAX_COUNT:
                def mk(rng, alt=alt, k=k, val=val):
                    o, a = _num(rng, val, [c.live[k]])
                    which = ("the second row" if k == 1 else "the third row") if is1d(s) else \
                        ("after the first step" if k == 1 else "after the second step")
                    text = (rule_preamble(s) + f"Suppose that, from the same start, the rule had instead been: "
                            f"{sim.describe(alt, s['scenario'])}. How many {c.livename} cells would there be "
                            f"{'in ' if is1d(s) else ''}{which}?")
                    return _q(5, "cf_rule", text, val, o, a, False, {"rule": alt, "k": k, "min_fps": c.fps([0])})
                out.append(("cf_rule", (alt, k), mk))
    for i in range(c.nm):
        r, cc = s["markers"][i]["cell"]
        if c.h[0][r, cc] not in (0, 1):
            continue
        st = c.h[0].copy()
        st[r, cc] = 1 - st[r, cc]
        for k in (1, 2):
            val = sim.live_count(c.nxt(k, start=st))
            if val != c.live[k] and val <= MAX_COUNT:
                def mk(rng, i=i, k=k, val=val, was=int(c.h[0][r, cc])):
                    o, a = _num(rng, val, [c.live[k]])
                    flip = "empty" if was else c.livename
                    if is1d(s):
                        text = (f"Suppose the top cell of {mname(s, i)} had been {flip} instead (everything else the "
                                f"same). How many {c.livename} cells would there be in "
                                f"{'the second row' if k == 1 else 'the third row'}?")
                    else:
                        text = (f"Suppose {mname(s, i)} had been {flip} at the start instead (everything else the "
                                f"same). How many {c.livename} cells would there be after the "
                                f"{'first' if k == 1 else 'second'} step?")
                    return _q(5, "cf_toggle", text, val, o, a, True, {"m": i, "k": k, "min_fps": c.fps(c.all)})
                out.append(("cf_toggle", (i, k), mk))
    return out


LEVELS = (level1, level2, level3, level4, level5)


def supply(scene) -> list[int]:
    """Number of distinct candidate questions per level (sim.check needs >= 3)."""
    c = Ctx(scene)
    return [len(f(c)) for f in LEVELS]


def _pick(rng, cands, n):
    bykey = {}
    for key, ident, mk in cands:
        bykey.setdefault(key, []).append(mk)
    keys = list(bykey)
    rng.shuffle(keys)
    chosen = []
    while len(chosen) < n:
        for key in keys:
            if bykey[key] and len(chosen) < n:
                chosen.append(bykey[key].pop(rng.randrange(len(bykey[key]))))
    return chosen


def build(scene: dict, rng: random.Random) -> list[dict]:
    counts = L.ten_counts(rng)
    c = Ctx(scene)
    qs = []
    for lvl, n in zip(LEVELS, counts):
        for mk in _pick(rng, lvl(c), n):
            qs.append(mk(rng))
    return L.number(qs)
