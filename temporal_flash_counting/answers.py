"""Independent answers for temporal_flash_counting.

This never reads scene["flashes"], scene["pulses"], the shapes' rhythm tags
or the relay chain. It walks every frame of the renderer's drawing state
(render.state: centre, radius and fill colour of each shape), classifies
each fill as rest / lit / white by nearest colour, and turns runs of lit
frames into detected flashes (onset, offset, white or not) and runs of
enlarged radius into pulses. Grid rows / columns and the number of shapes
come from the drawn centres. Causes are inferred from timing alone: a flash
of B is a response to A when it starts 0.6-1.4 s after an onset of A;
counterfactuals re-run that rule with the removed shape's flashes gone
(through the middle shape for a chain of three).
"""
from __future__ import annotations
import functools
import json
import math
import statistics

import render
import sim

FPS = sim.FPS
LO, HI = math.ceil(0.6 * FPS), math.floor(1.4 * FPS)    # reaction window in frames


def _nearest(fill, pal):
    return min(pal, key=lambda k: sum((fill[j] - pal[k][j]) ** 2 for j in range(3)))


def events_from(modes, grown, idx, nf):
    """Detected flashes / pulses from per-frame looks, read only on the frames in
    `idx` (all frames, or every k-th frame for the frame-sampling check).
    modes[i][f] is "rest" / "lit" / "white", grown[i][f] a bool."""
    n = len(modes)
    ev = [[] for _ in range(n)]
    pul = [[] for _ in range(n)]
    for i in range(n):
        on, white, big = None, False, None
        for f in list(idx) + [nf]:
            m = modes[i][f] if f < nf else "rest"
            if m != "rest":
                if on is None:
                    on, white = f, False
                white = white or m == "white"
            elif on is not None:
                ev[i].append({"on": on, "off": f, "white": white})
                on = None
            g = grown[i][f] if f < nf else False
            if g and big is None:
                big = f
            elif not g and big is not None:
                pul[i].append((big, f))
                big = None
    return ev, pul


@functools.lru_cache(maxsize=8)
def _track(key: str):
    scene = json.loads(key)
    n, nf = len(scene["shapes"]), scene["n_frames"]
    pals = [render.palette(scene, i) for i in range(n)]
    sts = [render.state(scene, f) for f in range(nf)]
    cents = [(s["x"], s["y"]) for s in sts[0]]
    modes = [[_nearest(sts[f][i]["fill"], pals[i]) for f in range(nf)] for i in range(n)]
    grown = [[sts[f][i]["r"] > sts[0][i]["r"] * 1.05 for f in range(nf)] for i in range(n)]
    ev, pul = events_from(modes, grown, range(nf), nf)
    return ev, pul, cents


def track(scene):
    key = json.dumps({k: scene[k] for k in ("scenario", "n_frames", "style", "shapes", "flashes", "pulses")},
                     sort_keys=True)
    return _track(key)


def _lab(scene, i):
    s = scene["shapes"][i]
    return f"{s['color']} lamp" if scene["scenario"] == "signal" else f"{s['color']} {s['type']}"


def _long(e, cut):
    return (e["off"] - e["on"]) / FPS > cut


def _tn(f):
    return f * 10 // FPS


def _rd(tn):
    return f"{tn // 10}.{tn % 10} s"


def _frame_of(tn):
    return math.ceil(tn * FPS / 10)


def _argmax(vals: dict):
    """Key of the strict maximum, None on a tie."""
    best = max(vals.values())
    ks = [k for k, v in vals.items() if v == best]
    return ks[0] if len(ks) == 1 else None


def _caused(evs_trig, e):
    return any(LO <= e["on"] - g["on"] <= HI for g in evs_trig)


def _kept(trig_all, trig_alive, resp):
    """Flashes of the responder that still happen: a flash that reacts (starts
    0.6-1.4 s after) to some trigger flash disappears when none of the
    flashes it could be reacting to are left."""
    return [e for e in resp if not (_caused(trig_all, e) and not _caused(trig_alive, e))]


def answer(key: str, p: dict, scene: dict):
    ev, pul, cents = track(scene)
    n, nf = len(ev), scene["n_frames"]
    cnt = {i: len(ev[i]) for i in range(n)}
    lab = lambda i: None if i is None else _lab(scene, i)  # noqa: E731
    if key == "flash_count":
        return cnt[p["x"]]
    if key == "grid_rows":
        return len({round(y) for _, y in cents})
    if key == "grid_cols":
        return len({round(x) for x, _ in cents})
    if key == "n_shapes":
        return len(cents)
    if key == "first_flasher":
        return lab(min(range(n), key=lambda i: ev[i][0]["on"]))
    if key == "last_flasher":
        return lab(max(range(n), key=lambda i: ev[i][-1]["on"]))
    if key == "white_ever":
        hits = [i for i in p["opts"] if any(e["white"] for e in ev[i])]
        return lab(hits[0]) if len(hits) == 1 else None
    if key == "pulse_shape":
        hits = [i for i in p["opts"] if pul[i]]
        return lab(hits[0]) if len(hits) == 1 else None
    if key == "relation":
        a, b = p["pair"]
        x, y = min(((u, v) for u in ev[a] for v in ev[b]), key=lambda uv: abs(uv[0]["on"] - uv[1]["on"]))
        if x["on"] == y["on"]:
            return sim.RELATIONS[0]
        if min(x["off"], y["off"]) > max(x["on"], y["on"]):
            return sim.RELATIONS[1]
        if min(x["off"], y["off"]) < max(x["on"], y["on"]):
            return sim.RELATIONS[2]
        return sim.RELATIONS[3]
    if key == "more_by":
        a, b = p["pair"]
        return cnt[a] - cnt[b]
    if key == "earlier_of_four":
        return lab(min(p["opts"], key=lambda i: ev[i][0]["on"]))
    if key == "longest_flash":
        return lab(_argmax({i: max(e["off"] - e["on"] for e in ev[i]) for i in p["opts"]}))
    if key == "lit_during":
        big = max(ev[p["x"]], key=lambda e: e["off"] - e["on"])
        hits = [i for i in p["opts"] if any(big["on"] < e["on"] < big["off"] for e in ev[i])]
        return lab(hits[0]) if len(hits) == 1 else None
    if key == "long_count":
        return sum(_long(e, p["cut"]) for e in ev[p["x"]])
    if key == "white_count":
        return sum(e["white"] for e in ev[p["x"]])
    if key == "period":
        on = [e["on"] for e in ev[p["x"]]]
        g = statistics.mean(b - a for a, b in zip(on, on[1:])) / FPS
        return f"{round(g * 2) / 2:.1f} s"
    if key == "first_time":
        return _rd(_tn(ev[p["x"]][0]["on"]))
    if key == "count_window":
        fa, fb = _frame_of(p["a"]), _frame_of(p["b"])
        return sum(fa <= e["on"] < fb for i in range(n) for e in ev[i])
    if key == "most_frequent":
        return lab(_argmax(cnt))
    if key == "least_frequent":
        return lab(_argmax({i: -c for i, c in cnt.items()}))
    if key == "steady":
        def cv(i):
            on = [e["on"] for e in ev[i]]
            g = [b - a for a, b in zip(on, on[1:])]
            return statistics.pstdev(g) / statistics.mean(g)
        return lab(min(p["opts"], key=cv))
    if key == "trigger":
        resp = p["resp"]
        return lab(_argmax({k: sum(1 for g in ev[k] if any(LO <= r["on"] - g["on"] <= HI
                                                           for r in ev[resp])) for k in p["opts"]}))
    if key == "total_flashes":
        return sum(cnt.values())
    if key == "total_white":
        return sum(e["white"] for i in range(n) for e in ev[i])
    if key == "busiest_window":
        cuts = p["cuts"]
        edges = [0] + [_frame_of(c) for c in cuts] + [nf]
        wc = {k: sum(edges[k] <= e["on"] < edges[k + 1] for i in range(n) for e in ev[i]) for k in range(4)}
        k = _argmax(wc)
        lo = ([0] + cuts)[k]
        return f"from {_rd(lo)} to the end" if k == 3 else f"from {_rd(lo)} to {_rd(cuts[k])}"
    if key == "sequence":
        return ", ".join("long" if _long(e, p["cut"]) else "short" for e in ev[p["x"]])
    if key == "cf_no_trigger":
        return len(_kept(ev[p["trig"]], [], ev[p["resp"]]))
    if key == "cf_chain":
        mid = _kept(ev[p["trig"]], [], ev[p["via"]])
        return len(_kept(ev[p["via"]], mid, ev[p["resp"]]))
    if key == "cf_remove_most":
        top = _argmax(cnt)
        return lab(_argmax({i: c for i, c in cnt.items() if i != top}))
    if key == "cf_long_only":
        return lab(_argmax({i: sum(_long(e, p["cut"]) for e in ev[i]) for i in range(n)}))
    if key == "cf_white_only":
        return lab(_argmax({i: sum(e["white"] for e in ev[i]) for i in range(n)}))
    if key == "cf_no_white_most":
        return lab(_argmax({i: sum(not e["white"] for e in ev[i]) for i in range(n)}))
    if key == "cf_stop":
        ft = _frame_of(p["t"])
        return sum(e["on"] < ft for e in ev[p["x"]])
    return None
