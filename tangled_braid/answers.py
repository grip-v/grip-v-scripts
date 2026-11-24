"""Independent answers for tangled_braid.

Nothing here reads the generator's moves, logs or sim helpers. The clip is
measured from `render.track(scene, f)` for every frame f: the across-
coordinate where each strand leaves the loom bar, and the crossings the
renderer draws with the strand drawn under and the one drawn over.

  * start order: the strands sorted by their bar coordinate at frame 0;
  * a crossing is the first frame at which two neighbouring strands' bar
    coordinates meet (|du| < 1e-6) or pass; the over strand is the one the
    renderer draws on top at that crossing; the crossing's place is the
    neighbouring pair's place just before it;
  * clock readings come from the frame index (truncated tenths, as drawn).

Counterfactuals are re-played on the measured (place, side-on-top) list with
a separate swap loop.
"""
from __future__ import annotations
import functools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import render  # noqa: E402

FPS = 24
NTH = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]


def _clock(f: int) -> int:
    return (10 * f) // FPS


def _clock_text(t: int) -> str:
    return f"{t // 10}.{t % 10} s"


def _first_frame(t: int) -> int:
    return -(-t * FPS // 10)


def _last_frame(t: int) -> int:
    return -(-(t + 1) * FPS // 10) - 1


@functools.lru_cache(maxsize=64)
def _measure(key: str) -> dict:
    scene = json.loads(key)
    nf = scene["n_frames"]
    tr0 = render.track(scene, 0)
    order = sorted(tr0["head"], key=lambda n: tr0["head"][n])
    start = list(order)
    events = []                      # dict(f, place, low, high, over, under)
    orders_at = [list(order)]        # order after frame f (index f)
    for f in range(1, nf):
        tr = render.track(scene, f)
        h = tr["head"]
        swaps = [p for p in range(len(order) - 1) if h[order[p]] >= h[order[p + 1]] - 1e-6]
        for p in swaps:
            if p + 1 in swaps:
                raise ValueError("overlapping crossings")
            lo, hi = order[p], order[p + 1]
            drawn = [c for c in tr["crossings"] if {c["under"], c["over"]} == {lo, hi} and c["s_mid"] <= tr["s"] + 1e-6]
            c = max(drawn, key=lambda c: c["s_mid"])
            events.append({"f": f, "place": p, "low": lo, "high": hi, "over": c["over"], "under": c["under"]})
        for p in swaps:
            order[p], order[p + 1] = order[p + 1], order[p]
        orders_at.append(list(order))
    return {"start": start, "events": events, "orders_at": orders_at,
            "color": {s["name"]: s["color_name"] for s in scene["strands"]},
            "orient": scene["layout"]["orient"], "nf": nf}


def measure(scene) -> dict:
    return _measure(json.dumps(scene, sort_keys=True))


def _play(start, plan):
    """plan: list of (place, top_side) with top_side 0 = strand from the lower
    place on top, 1 = from the higher place. Returns (final order, crossings
    as (low, high, over))."""
    order = list(start)
    out = []
    for place, side in plan:
        lo, hi = order[place], order[place + 1]
        out.append((lo, hi, lo if side == 0 else hi))
        order[place], order[place + 1] = hi, lo
    return order, out


def _plan(M):
    return [(e["place"], 0 if e["over"] == e["low"] else 1) for e in M["events"]]


def _pairs(cr, a, b):
    return [k for k, c in enumerate(cr) if {c[0], c[1]} == {a, b}]


def _net(cr, x, y):
    v = 0
    for lo, hi, ov in cr:
        if {lo, hi} == {x, y}:
            v += 1 if ov == x else -1
    return v


def _signed(v):
    return f"{v:+d}" if v else "0"


def _place_text(M, p):
    return f"{NTH[p]} from the {'top' if M['orient'] == 'h' else 'left'}"


def _pair_text(M, a, b):
    x, y = sorted([M["color"][a], M["color"][b]])
    return f"{x} and {y}"


def answer(key: str, p: dict, scene: dict, M: dict | None = None):
    """M: a measurement to use instead of the renderer's track (checks.py
    passes one read off the decoded video frames)."""
    M = measure(scene) if M is None else M
    C = M["color"]
    start, ev = M["start"], M["events"]
    final, cr = _play(start, _plan(M))
    names = start
    inv = {x: [k for k, e in enumerate(ev) if x in (e["low"], e["high"])] for x in names}

    def other(e, x):
        return e["high"] if e["low"] == x else e["low"]

    def order_at_clock(t):
        lo, hi = _first_frame(t), _last_frame(t)
        o = M["orders_at"][lo]
        if any(M["orders_at"][f] != o for f in range(lo, hi + 1)):
            return None
        return o

    if key == "n_strands":
        return len(names)
    if key == "n_crossings":
        return len(ev)
    if key == "strand_crossings":
        return len(inv[p["x"]])
    if key == "start_edge":
        return C[start[0] if p["end"] == "low" else start[-1]]
    if key == "end_edge":
        return C[final[0] if p["end"] == "low" else final[-1]]
    if key == "first_pair":
        return _pair_text(M, ev[0]["low"], ev[0]["high"])
    if key == "never_cross":
        return C[next(x for x in names if not inv[x])]
    if key == "over_at":
        return C[ev[p["k"]]["over"]]
    if key == "first_cross_of":
        e = ev[inv[p["x"]][0]]
        y = other(e, p["x"])
        return f"it passes {'over' if e['over'] == p['x'] else 'under'} the {C[y]} strand"
    if key == "first_of":
        firsts = {x: ev[inv[x][0]]["f"] for x in p["opts"] if inv[x]}
        return C[min(firsts, key=firsts.get)]
    if key == "neighbor_at":
        o = order_at_clock(p["t"])
        i = o.index(p["x"]) + (-1 if p["side"] == "low" else 1)
        return C[o[i]]
    if key == "next_partner":
        ks = inv[p["x"]]
        j = next(i for i, k in enumerate(ks) if other(ev[k], p["x"]) == p["y"])
        return C[other(ev[ks[j + 1]], p["x"])]
    if key == "partner_first":
        return C[other(ev[inv[p["x"]][0]], p["x"])]
    if key == "same_time":
        k = next(k for k, e in enumerate(ev) if {e["low"], e["high"]} == {p["a"], p["b"]})
        e2 = next(e for j, e in enumerate(ev) if e["f"] == ev[k]["f"] and j != k)
        return _pair_text(M, e2["low"], e2["high"])
    if key == "pair_count":
        return len(_pairs(cr, p["a"], p["b"]))
    if key == "cross_time":
        k = next(k for k, e in enumerate(ev) if {e["low"], e["high"]} == {p["a"], p["b"]})
        return _clock_text(_clock(ev[k]["f"]))
    if key == "between":
        f0, f1 = _last_frame(p["t0"]), _first_frame(p["t1"])
        return sum(1 for e in ev if f0 < e["f"] < f1)
    if key == "displacement":
        return abs(final.index(p["x"]) - start.index(p["x"]))
    if key == "place_at":
        return _place_text(M, order_at_clock(p["t"]).index(p["x"]))
    if key == "count_by":
        f = _first_frame(p["t"])
        return sum(1 for k in inv[p["x"]] if ev[k]["f"] < f)
    if key == "net_over":
        return _signed(_net(cr, p['a'], p['b']))
    if key == "over_count":
        return sum(1 for c in cr if c[2] == p["x"])
    if key == "most_crossings":
        return C[max(names, key=lambda x: len(inv[x]))]
    if key == "over_most":
        return C[max(names, key=lambda x: sum(1 for c in cr if c[2] == x))]
    if key == "partner_seq":
        return ", then ".join(C[other(ev[k], p["x"])] for k in inv[p["x"]])
    if key == "home_count":
        return sum(1 for x in names if final.index(x) == start.index(x))
    if key == "where_started":
        return C[final[start.index(p["x"])]]
    plan = _plan(M)
    if key == "cf_skip_end":
        f2, _ = _play(start, plan[:p["k"]] + plan[p["k"] + 1:])
        return C[f2[p["p"]]]
    if key == "cf_skip_pair":
        _, c2 = _play(start, plan[:p["k"]] + plan[p["k"] + 1:])
        return len(_pairs(c2, p["a"], p["b"]))
    if key == "cf_flip_net":
        pl = list(plan)
        pl[p["k"]] = (pl[p["k"]][0], 1 - pl[p["k"]][1])
        _, c3 = _play(start, pl)
        return _signed(_net(c3, p['x'], p['y']))
    if key == "cf_mirror":
        _, c4 = _play(start, [(q, 1 - s) for q, s in plan])
        return sum(1 for c in c4 if c[2] == p["x"])
    if key == "cf_swap_start":
        s5 = list(start)
        i, j = s5.index(p["x"]), s5.index(p["y"])
        s5[i], s5[j] = s5[j], s5[i]
        f5, _ = _play(s5, plan)
        return C[f5[p["p"]]]
    return None
