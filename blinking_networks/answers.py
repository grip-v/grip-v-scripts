"""Independent answers for blinking_networks.

Reads only what the renderer draws as the static world (node positions,
shapes and outlines, links with their dash/arrow style, the hidden slow link,
the speed and the starting flashes) and re-runs the whole cascade with a
fixed-step (1/240 s) simulation that moves every pulse along its link in
pixel space. It never reads the generator's event log (fire / parent /
arrivals / pulses). Counterfactuals re-run the same stepper under the changed
rule. Every answer is then read off that independent replay.
"""
from __future__ import annotations
import math

DT = 1.0 / 240.0
T_MAX = 120.0
CF_OPTS = ("later than in the video", "at the same time as in the video", "earlier than in the video",
           "it would never light up")


def _names(scene):
    return {n["id"]: n["name"] for n in scene["nodes"]}


def _same(e, pair):
    return {e["a"], e["b"]} == set(pair)


def run(scene, cut=None, starts=None, unslow=False, single=False, reverse=None) -> dict:
    pos = {n["id"]: (n["x"], n["y"]) for n in scene["nodes"]}
    need = {n["id"]: (1 if single else n["threshold"]) for n in scene["nodes"]}
    out_links = {k: [] for k in pos}               # node -> [(neighbour, length px, speed px/s)]
    for e in scene["edges"]:
        if cut and _same(e, cut):
            continue
        a, b = e["a"], e["b"]
        if e["oneway"] and reverse and _same(e, reverse):
            a, b = b, a
        length = math.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1])
        v = scene["speed"] * (0.5 if e["slow"] and not unslow else 1.0)
        out_links[a].append((b, length, v))
        if not e["oneway"]:
            out_links[b].append((a, length, v))
    pending = sorted(([float(t), k] for k, t in (scene["starts"] if starts is None else starts)))
    lit, by, heard, hits = {}, {}, {k: set() for k in pos}, {k: [] for k in pos}
    active = []                                      # [from, to, travelled px, length, speed]

    def light(k, t, src, now):
        lit[k] = t
        by[k] = src
        for nb, length, v in out_links[k]:
            if nb not in heard[k]:
                active.append([k, nb, v * (now - t), length, v])

    step = 0
    while (active or pending) and step * DT < T_MAX:
        t0, t1 = step * DT, (step + 1) * DT
        events = []
        keep = []
        for p in active:
            p[2] += p[4] * DT
            if p[2] >= p[3]:
                events.append((t1 - (p[2] - p[3]) / p[4], p[1], p[0]))
            else:
                keep.append(p)
        active[:] = keep
        while pending and pending[0][0] <= t1:
            tt, k = pending.pop(0)
            events.append((tt, k, None))
        for te, k, src in sorted(events, key=lambda ev: ev[0]):
            hits[k].append((te, src))
            if k in lit:
                continue
            if src is None:
                light(k, te, None, t1)
                continue
            heard[k].add(src)
            if len(heard[k]) >= need[k]:
                light(k, te, src, t1)
        step += 1
    return {"lit": lit, "by": by, "hits": hits}


def _last(lit):
    return max(lit, key=lit.get)


def _order(lit):
    return sorted(lit, key=lit.get)


def _cmp(before, after, x):
    if x not in after:
        return CF_OPTS[3]
    d = after[x] - before[x]
    if abs(d) < 0.02:
        return CF_OPTS[1]
    return CF_OPTS[0] if d > 0 else CF_OPTS[2]


def answer(key: str, p: dict, scene: dict):
    nm = _names(scene)
    r = run(scene)
    lit, by = r["lit"], r["by"]
    order = _order(lit)
    nbrs = lambda x: {e["b"] if e["a"] == x else e["a"] for e in scene["edges"] if x in (e["a"], e["b"])}  # noqa
    if key == "first_lit":
        return nm[order[0]]
    if key == "last_lit":
        return nm[order[-1]]
    if key == "count_nodes":
        return len(scene["nodes"])
    if key == "count_links":
        return len(scene["edges"])
    if key == "count_dashed":
        return sum(1 for e in scene["edges"] if e["slow"])
    if key == "count_arrows":
        return sum(1 for e in scene["edges"] if e["oneway"])
    if key == "count_double":
        return sum(1 for n in scene["nodes"] if n["threshold"] == 2)
    if key == "count_dark":
        return len(scene["nodes"]) - len(lit)
    if key == "degree":
        return len(nbrs(p["x"]))
    if key == "first_of3":
        return nm[min(p["set"], key=lambda k: lit[k])]
    if key == "last_of3":
        return nm[max(p["set"], key=lambda k: lit[k])]
    if key == "first_neighbor":
        return nm[min((y for y in nbrs(p["x"]) if y in lit), key=lambda k: lit[k])]
    if key == "nth_lit":
        return nm[order[p["k"] - 1]]
    if key == "next_after":
        return nm[order[order.index(p["x"]) + 1]]
    if key == "fire_time":
        return f"{math.floor(lit[p['x']] + 0.5)} s"
    if key == "travel_time":
        return f"{math.floor(lit[p['b']] - lit[p['a']] + 0.5)} s"
    if key == "count_lit_at":
        return sum(1 for v in lit.values() if v <= p["T"])
    if key == "span":
        return f"{math.floor(lit[order[-1]] - lit[order[0]] + 0.5)} s"
    if key == "parent":
        return nm[by[p["x"]]]
    if key == "hops":
        h, x = 0, p["x"]
        while by[x] is not None:
            x, h = by[x], h + 1
        return h
    if key == "children":
        return sum(1 for k in lit if by[k] == p["x"])
    if key == "slow_edge":
        e = next(e for e in scene["edges"] if e["slow"])
        a, b = sorted([nm[e["a"]], nm[e["b"]]])
        return f"{a} – {b}"
    if key == "spontaneous":
        return nm[next(k for k in order[1:] if by[k] is None)]
    if key == "origin":
        x = p["x"]
        while by[x] is not None:
            x = by[x]
        return nm[x]
    if key == "never_lit":
        return nm[next(n["id"] for n in scene["nodes"] if n["id"] not in lit)]
    if key == "first_double":
        return nm[next(k for k in order if any(n["id"] == k and n["threshold"] == 2 for n in scene["nodes"]))]
    if key == "cf_cut_last":
        return nm[_last(run(scene, cut=p["edge"])["lit"])]
    if key == "cf_cut_effect":
        return _cmp(lit, run(scene, cut=p["edge"])["lit"], p["x"])
    if key == "cf_source_last":
        t0 = min(t for _, t in scene["starts"])
        return nm[_last(run(scene, starts=[[p["y"], t0]])["lit"])]
    if key == "cf_unslow_last":
        return nm[_last(run(scene, unslow=True)["lit"])]
    if key == "cf_reverse_count":
        return len(run(scene, reverse=p["edge"])["lit"])
    if key == "cf_single_last":
        return nm[_last(run(scene, single=True)["lit"])]
    if key == "cf_single_count":
        return len(run(scene, single=True)["lit"])
    if key == "cf_no_second":
        first = min(scene["starts"], key=lambda s: s[1])
        return _cmp(lit, run(scene, starts=[first])["lit"], p["x"])
    if key == "cf_no_second_last":
        first = min(scene["starts"], key=lambda s: s[1])
        return nm[_last(run(scene, starts=[first])["lit"])]
    return None
