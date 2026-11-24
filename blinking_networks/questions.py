"""10-question, 5-rung ladder for blinking_networks (common/ladder5.py, LADDER = "ten").

`candidates(scene)` lists every question this scene supports (deterministic,
no rng); `build` draws the per-level counts, then picks templates round-robin
in a random order so the mix varies clip to clip. Every candidate already
satisfies the margin rules (>= 0.3 s between the events it relies on, times
>= 0.25 s from a rounding boundary, unique counterfactual winners), and
stores params["min_fps"] = the sampling rate needed to see those events.
"""
from __future__ import annotations
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
ORD = {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}
CF_OPTS = ("later than in the video", "at the same time as in the video", "earlier than in the video",
           "it would never light up")


def fps_for(gap: float) -> float:
    """Frames per second needed so that some sampled frame falls inside a gap."""
    if gap <= 0:
        return 24.0
    return max(0.5, math.ceil(100.0 / gap) / 100.0)


def ename(scene, pair) -> str:
    nm = sim.node_map(scene)
    a, b = sorted(nm[x]["name"] for x in pair)
    return f"{a} – {b}"


class Ctx:
    def __init__(self, scene):
        self.s = scene
        self.nm = sim.node_map(scene)
        self.ids = sorted(self.nm, key=lambda k: int(k[1:]))
        self.name = {k: self.nm[k]["name"] for k in self.ids}
        self.fire = scene["fire"]
        self.order = sorted(self.fire, key=self.fire.get)
        self.parent = scene["parent"]
        self.end = scene["end_t"]
        self.nb = {k: sorted({e["b"] if e["a"] == k else e["a"] for e in scene["edges"] if k in (e["a"], e["b"])},
                             key=lambda x: int(x[1:])) for k in self.ids}
        self.starts = [s for s, _ in scene["starts"]]

    def the(self, k):
        return "the " + self.name[k]

    def arr_gap(self, k):
        ts = [t for t, _ in self.s["arrivals"][k]]
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        return min(gaps) if gaps else 10.0

    def fire_gap(self, k):
        i = self.order.index(k)
        g = [self.fire[k], self.end - self.fire[k]]
        if i > 0:
            g.append(self.fire[k] - self.fire[self.order[i - 1]])
        if i + 1 < len(self.order):
            g.append(self.fire[self.order[i + 1]] - self.fire[k])
        return min(g)

    def chain(self, k):
        out = [k]
        while self.parent.get(out[-1]):
            out.append(self.parent[out[-1]])
        return out

    def slow_fps(self):
        if self.s["scenario"] != "hidden_slow":
            return 0.5
        e = next(e for e in self.s["edges"] if e["slow"])
        return fps_for(sim.edge_len(self.s, e) / self.s["speed"])

    def base_fps(self):
        f = self.slow_fps()
        if self.s["scenario"] == "two_sources":
            f = max(f, fps_for(self.arr_gap(self.starts[1])))
        return f

    def others(self, rng, exclude, k=3, prefer=()):
        pre = [x for x in prefer if x not in exclude]
        rest = [x for x in self.ids if x not in exclude and x not in pre]
        rng.shuffle(rest)
        return [self.name[x] for x in (pre + rest)][:k + 3]


def _unique_last(f):
    ts = sorted(f.values())
    return len(ts) >= 2 and ts[-1] - ts[-2] >= sim.MARGIN


def _q(level, key, params, text, gt, dist, motion=True, fmt=str, gap=None):
    """`gap`: the smallest time gap between the light-ups this question compares;
    candidates with gap < MARGIN are dropped (no near-ties)."""
    return {"level": level, "key": key, "params": params, "text": text, "gt": gt, "dist": dist,
            "motion": motion, "fmt": fmt, "gap": gap}


def _secs(v):
    return f"{v} s"


def _time_dist(gt):
    return lambda rng: [_secs(v) for v in (gt + 1, gt - 1, gt + 2, gt - 2, gt + 3) if v >= 0]


def _int_dist(gt, lo=0):
    return lambda rng: L.near_ints(gt, lo=lo)


def candidates(scene) -> list[dict]:
    c = Ctx(scene)
    s, sc, fire, order = scene, scene["scenario"], c.fire, c.order
    out = []
    n_lit = len(order)

    # ---------------- L1 observation
    out.append(_q(1, "first_lit", {"min_fps": fps_for(fire[order[1]] - fire[order[0]])},
                  "Which node is the very first to flash?", c.name[order[0]],
                  lambda rng: c.others(rng, {order[0]}, prefer=order[1:3]), gap=fire[order[1]] - fire[order[0]]))
    out.append(_q(1, "last_lit", {"min_fps": fps_for(min(fire[order[-1]] - fire[order[-2]],
                                                         c.end - fire[order[-1]]))},
                  "Which node is the last one to light up?", c.name[order[-1]],
                  lambda rng: c.others(rng, {order[-1]}, prefer=order[-3:-1][::-1]),
                  gap=fire[order[-1]] - fire[order[-2]]))
    out.append(_q(1, "count_nodes", {"min_fps": 0.5}, "How many nodes (shapes) are in the network?",
                  len(c.ids), _int_dist(len(c.ids), 1), motion=False))
    out.append(_q(1, "count_links", {"min_fps": 0.5}, "How many links (lines between nodes) are there?",
                  len(s["edges"]), _int_dist(len(s["edges"]), 1), motion=False))
    if sc == "marked_speeds":
        k = sum(e["slow"] for e in s["edges"])
        out.append(_q(1, "count_dashed", {"min_fps": 0.5}, "How many of the links are drawn dashed?", k,
                      _int_dist(k, 0), motion=False))
    if sc == "one_way":
        k = sum(e["oneway"] for e in s["edges"])
        out.append(_q(1, "count_arrows", {"min_fps": 0.5}, "How many links carry an arrowhead?", k,
                      _int_dist(k, 0), motion=False))
    if sc == "double_trigger":
        k = sum(n["threshold"] == 2 for n in s["nodes"])
        out.append(_q(1, "count_double", {"min_fps": 0.5}, "How many nodes have a double outline?", k,
                      _int_dist(k, 0), motion=False))
    if sc in ("one_way", "double_trigger"):
        k = len(c.ids) - n_lit
        out.append(_q(1, "count_dark", {"min_fps": fps_for(c.end - fire[order[-1]])},
                      "When the clip ends, how many nodes have never lit up (no full halo)?", k,
                      _int_dist(k, 0)))
    for x in c.ids:
        out.append(_q(1, "degree", {"x": x, "min_fps": 0.5}, f"How many links touch {c.the(x)}?",
                      len(c.nb[x]), _int_dist(len(c.nb[x]), 1), motion=False))

    # ---------------- L2 relational
    trios = [order[i:i + 3] for i in range(n_lit - 2)] + [order[i:i + 5:2] for i in range(n_lit - 4)]
    for trio in trios:
        tri = sorted(trio, key=lambda k: int(k[1:]))
        names = ", ".join(c.the(k) for k in tri[:2]) + f" and {c.the(tri[2])}"
        g = fire[trio[1]] - fire[trio[0]]
        out.append(_q(2, "first_of3", {"set": tri, "min_fps": fps_for(g)},
                      f"Of {names}, which one lights up first?", c.name[trio[0]],
                      lambda rng, t=trio: [c.name[t[1]], c.name[t[2]], "they all light up at the same moment"],
                      gap=g))
        g = fire[trio[2]] - fire[trio[1]]
        out.append(_q(2, "last_of3", {"set": tri, "min_fps": fps_for(g)},
                      f"Of {names}, which one lights up last?", c.name[trio[2]],
                      lambda rng, t=trio: [c.name[t[0]], c.name[t[1]], "they all light up at the same moment"],
                      gap=g))
    for x in c.ids:
        nbl = [y for y in c.nb[x] if y in fire]
        if len(nbl) >= 2:
            srt = sorted(nbl, key=fire.get)
            out.append(_q(2, "first_neighbor", {"x": x, "min_fps": fps_for(fire[srt[1]] - fire[srt[0]])},
                          f"Of the nodes directly linked to {c.the(x)}, which one lights up first?",
                          c.name[srt[0]], lambda rng, x=x, srt=srt: c.others(rng, {srt[0], x}, prefer=srt[1:]),
                          gap=fire[srt[1]] - fire[srt[0]]))
    for k in range(2, min(6, n_lit - 1) + 1):
        y = order[k - 1]
        out.append(_q(2, "nth_lit", {"k": k, "min_fps": fps_for(c.fire_gap(y))},
                      f"Which node is the {ORD[k]} one to light up?", c.name[y],
                      lambda rng, k=k, y=y: c.others(rng, {y}, prefer=[order[k - 2], order[k]]),
                      gap=c.fire_gap(y)))
    for i, x in enumerate(order[:-1]):
        y = order[i + 1]
        g = min(fire[y] - fire[x], c.fire_gap(y))
        out.append(_q(2, "next_after", {"x": x, "min_fps": fps_for(g)},
                      f"Which node lights up next, right after {c.the(x)}?", c.name[y],
                      lambda rng, i=i, x=x, y=y: c.others(rng, {x, y}, prefer=order[i + 2:i + 4]), gap=g))

    # ---------------- L3 measurement (the clock, top right)
    for x in order[1:]:
        t = fire[x]
        d = 0.5 - abs(t - round(t))
        if d >= 0.25:
            gt = int(round(t))
            out.append(_q(3, "fire_time", {"x": x, "min_fps": fps_for(d)},
                          f"To the nearest second, what does the clock read when {c.the(x)} lights up?",
                          _secs(gt), _time_dist(gt)))
    for x in order[1:]:
        p = c.parent[x]
        if p is None:
            continue
        dur = fire[x] - fire[p]
        d = 0.5 - abs(dur - round(dur))
        if dur >= 0.75 and d >= 0.3:
            gt = int(round(dur))
            out.append(_q(3, "travel_time", {"a": p, "b": x, "min_fps": fps_for(d / 2)},
                          f"To the nearest second, how long does the pulse that lights {c.the(x)} take to travel "
                          f"from {c.the(p)}?", _secs(gt), _time_dist(gt)))
    ts = sorted(fire.values())
    for T2 in range(2, int(c.end * 2)):
        T = T2 / 2
        if T > c.end - sim.PAD or T < ts[0]:
            continue
        before = [v for v in ts if v <= T]
        after = [v for v in ts if v > T]
        a = before[-1]
        b = after[0] if after else c.end
        if T - a >= sim.MARGIN and b - T >= sim.MARGIN:
            gt = len(before)
            out.append(_q(3, "count_lit_at", {"T": T, "min_fps": fps_for(b - a)},
                          f"When the clock reads {T:.1f} s, how many nodes have lit up (full halo)?", gt,
                          _int_dist(gt, 1)))
    span = ts[-1] - ts[0]
    d = 0.5 - abs(span - round(span))
    if d >= 0.3:
        gt = int(round(span))
        out.append(_q(3, "span", {"min_fps": fps_for(d / 2)},
                      "To the nearest second, how long is it from the very first flash until the last node "
                      "lights up?", _secs(gt), _time_dist(gt)))

    # ---------------- L4 integration
    for x in order:
        if x in c.starts or len(c.nb[x]) < 2:
            continue
        p = c.parent[x]
        g = min(c.arr_gap(x), sim.travel(s, next(e for e in s["edges"] if sim.same_edge(e, [p, x]))))
        word = "the pulse that finally makes" if s["nodes"][int(x[1:])]["threshold"] == 2 else "the pulse that lights"
        out.append(_q(4, "parent", {"x": x, "min_fps": fps_for(g)},
                      f"Which node sent {word} up {c.the(x)}?", c.name[p],
                      lambda rng, x=x, p=p: c.others(rng, {x, p}, prefer=[y for y in c.nb[x] if y != p])))
    for x in order:
        if x in c.starts:
            continue
        ch = c.chain(x)
        h = len(ch) - 1
        g = min(c.arr_gap(y) for y in ch)
        origin = "the node that started it" if sc == "two_sources" else "the first node to flash"
        out.append(_q(4, "hops", {"x": x, "min_fps": fps_for(g)},
                      f"Follow the chain of pulses that lit {c.the(x)} back to {origin}. How many links did the "
                      f"signal cross?", h, _int_dist(h, 1)))
    for x in order:
        if len(c.nb[x]) < 2:
            continue
        kids = [y for y in order if c.parent.get(y) == x]
        g = min([c.arr_gap(y) for y in c.nb[x]] + [10.0])
        out.append(_q(4, "children", {"x": x, "min_fps": fps_for(g)},
                      f"How many nodes are lit up directly by a pulse sent from {c.the(x)}?", len(kids),
                      _int_dist(len(kids), 0)))
    if sc == "hidden_slow":
        slow = next(e for e in s["edges"] if e["slow"])
        others = sorted(s["edges"], key=lambda e: abs(sim.edge_len(s, e) - sim.edge_len(s, slow)))
        out.append(_q(4, "slow_edge", {"min_fps": c.slow_fps()},
                      "All links look the same, but one of them carries pulses at half speed. Which link is it?",
                      ename(s, [slow["a"], slow["b"]]),
                      lambda rng: [ename(s, [e["a"], e["b"]]) for e in others if not e["slow"]][:5]))
    if sc == "two_sources":
        y = c.starts[1]
        out.append(_q(4, "spontaneous", {"min_fps": fps_for(c.arr_gap(y))},
                      "Apart from the very first node to flash, one other node flashes on its own before any pulse "
                      "reaches it. Which one?", c.name[y],
                      lambda rng: c.others(rng, {y, order[0]}, prefer=[k for k in order if k != y][1:3])))
        for x in order:
            if x in c.starts:
                continue
            ch = c.chain(x)
            g = min(c.arr_gap(k) for k in ch)
            out.append(_q(4, "origin", {"x": x, "min_fps": fps_for(g)},
                          f"Two nodes flash on their own. Which of them started the chain of pulses that lit "
                          f"{c.the(x)}?", c.name[ch[-1]],
                          lambda rng, ch=ch, x=x: [c.name[k] for k in c.starts if k != ch[-1]] +
                          c.others(rng, set(c.starts) | {x}, prefer=[ch[1]] if len(ch) > 2 else [])))
    dark = [k for k in c.ids if k not in fire]
    if len(dark) == 1:
        x = dark[0]
        out.append(_q(4, "never_lit", {"min_fps": fps_for(c.end - fire[order[-1]])},
                      "Exactly one node never lights up. Which one?", c.name[x],
                      lambda rng: c.others(rng, {x}, prefer=order[::-1][:2])))
    if sc == "double_trigger":
        dbl = sorted((k for k in order if c.nm[k]["threshold"] == 2), key=fire.get)
        if len(dbl) >= 2:
            g = min(fire[dbl[1]] - fire[dbl[0]], c.arr_gap(dbl[0]), c.arr_gap(dbl[1]))
            out.append(_q(4, "first_double", {"min_fps": fps_for(g)},
                          "Of the double-outlined nodes, which one lights up first?", c.name[dbl[0]],
                          lambda rng: [c.name[k] for k in dbl[1:]] + c.others(rng, set(dbl)), gap=g))

    # ---------------- L5 counterfactual (re-simulated)
    bf = c.base_fps()
    note = {"hidden_slow": " (the slow link stays slow)", "marked_speeds": " (dashed links stay slow)",
            "one_way": " (arrows still apply)", "two_sources": " (both nodes still flash on their own)",
            "double_trigger": " (double-outlined nodes still need two pulses)"}[sc]
    for e in s["edges"]:
        pair = [e["a"], e["b"]]
        r = sim.simulate(s, cut=pair)
        f2 = r["fire"]
        lab = f"the link between {c.the(pair[0])} and {c.the(pair[1])}"
        if len(f2) >= 2 and _unique_last(f2):
            last = max(f2, key=f2.get)
            out.append(_q(5, "cf_cut_last", {"edge": pair, "min_fps": bf},
                          f"If {lab} were removed{note}, which node would be the last to light up?", c.name[last],
                          lambda rng, last=last, f2=f2: c.others(rng, {last}, prefer=[order[-1]] +
                                                                 sorted(f2, key=f2.get)[-3:-1][::-1])))
        for x in order:
            if x in c.starts:
                continue
            if x not in f2:
                gt = CF_OPTS[3]
            elif abs(f2[x] - fire[x]) < 1e-6:
                gt = CF_OPTS[1]
            elif f2[x] - fire[x] >= sim.MARGIN:
                gt = CF_OPTS[0]
            elif fire[x] - f2[x] >= sim.MARGIN:
                gt = CF_OPTS[2]
            else:
                continue
            out.append(_q(5, "cf_cut_effect", {"edge": pair, "x": x, "min_fps": bf},
                          f"If {lab} were removed{note}, when would {c.the(x)} light up?", gt,
                          lambda rng, gt=gt: [o for o in CF_OPTS if o != gt]))
    if sc != "two_sources":
        t0 = s["starts"][0][1]
        for y in c.ids:
            if y == c.starts[0] or c.nm[y]["threshold"] != 1:
                continue
            f2 = sim.simulate(s, starts=[[y, t0]])["fire"]
            if len(f2) >= 2 and _unique_last(f2):
                last = max(f2, key=f2.get)
                out.append(_q(5, "cf_source_last", {"y": y, "min_fps": bf},
                              f"Suppose {c.the(y)} had flashed first instead, at the same moment{note}. Which node "
                              f"would be the last to light up?", c.name[last],
                              lambda rng, last=last: c.others(rng, {last}, prefer=[order[-1]])))
    if sc in ("hidden_slow", "marked_speeds"):
        f2 = sim.simulate(s, unslow=True)["fire"]
        if _unique_last(f2):
            last = max(f2, key=f2.get)
            what = "the one slow link" if sc == "hidden_slow" else "the dashed links"
            out.append(_q(5, "cf_unslow_last", {"min_fps": bf},
                          f"If {what} carried pulses at full speed like the others, which node would be the last "
                          f"to light up?", c.name[last],
                          lambda rng: c.others(rng, {last}, prefer=[order[-1]] + order[-3:-1])))
    if sc == "one_way":
        for e in s["edges"]:
            if e["oneway"]:
                pair = [e["a"], e["b"]]
                k = len(sim.simulate(s, reverse=pair)["fire"])
                out.append(_q(5, "cf_reverse_count", {"edge": pair, "min_fps": bf},
                              f"If the arrow on the link between {c.the(pair[0])} and {c.the(pair[1])} pointed the "
                              f"other way, how many nodes would light up in total?", k, _int_dist(k, 1)))
    if sc == "double_trigger":
        f2 = sim.simulate(s, single=True)["fire"]
        if _unique_last(f2):
            last = max(f2, key=f2.get)
            out.append(_q(5, "cf_single_last", {"min_fps": bf},
                          "If every node lit up on the first pulse that reached it, which node would be the last "
                          "to light up?", c.name[last],
                          lambda rng: c.others(rng, {last}, prefer=[order[-1]])))
        k = len(f2)
        out.append(_q(5, "cf_single_count", {"min_fps": bf},
                      "If every node lit up on the first pulse that reached it, how many nodes would light up in "
                      "total?", k, _int_dist(k, 1)))
    if sc == "two_sources":
        f2 = sim.simulate(s, starts=[s["starts"][0]])["fire"]
        for x in order:
            if x in c.starts:
                continue
            if x not in f2:
                gt = CF_OPTS[3]
            elif abs(f2[x] - fire[x]) < 1e-6:
                gt = CF_OPTS[1]
            elif f2[x] - fire[x] >= sim.MARGIN:
                gt = CF_OPTS[0]
            else:
                continue
            out.append(_q(5, "cf_no_second", {"x": x, "min_fps": bf},
                          f"Suppose {c.the(c.starts[1])} had not flashed on its own (it could still be lit by a "
                          f"pulse). When would {c.the(x)} light up?", gt,
                          lambda rng, gt=gt: [o for o in CF_OPTS if o != gt]))
        if _unique_last(f2):
            last = max(f2, key=f2.get)
            out.append(_q(5, "cf_no_second_last", {"min_fps": bf},
                          f"Suppose {c.the(c.starts[1])} had not flashed on its own (it could still be lit by a "
                          f"pulse). Which node would be the last to light up?", c.name[last],
                          lambda rng: c.others(rng, {last}, prefer=[order[-1]])))
    out = [q for q in out if q["gap"] is None or q["gap"] >= sim.MARGIN]
    for q in out:
        q["params"]["min_fps"] = round(q["params"]["min_fps"], 2)
    return out


def _by_level(cands):
    lv = {i: {} for i in range(1, 6)}
    for q in cands:
        lv[q["level"]].setdefault(q["key"], []).append(q)
    return lv


def pool_ok(scene) -> bool:
    """At least 3 distinct candidates on every level and 2+ templates on L2-L5."""
    lv = _by_level(candidates(scene))
    return all(sum(len(v) for v in lv[i].values()) >= 3 for i in lv) and all(len(lv[i]) >= 2 for i in (2, 3, 4, 5))


def build(scene: dict, rng: random.Random) -> list[dict]:
    lv = _by_level(candidates(scene))
    counts = L.ten_counts(rng)
    qs = []
    for level, k in zip(range(1, 6), counts):
        keys = sorted(lv[level])
        rng.shuffle(keys)
        pools = {key: list(lv[level][key]) for key in keys}
        picked, i = [], 0
        while len(picked) < k:
            key = keys[i % len(keys)]
            i += 1
            if pools[key]:
                picked.append(pools[key].pop(rng.randrange(len(pools[key]))))
        for q in picked:
            opts, a = L.mc(rng, q["gt"], q["dist"](rng), fmt=q["fmt"])
            qs.append(L.question(level, q["key"], q["text"], q["gt"], opts, a, q["motion"], q["params"]))
    return L.number(qs)
