"""Independent answers for multi_agent_chase v3. They never read the
generator's catch log or the balls' roles: everything comes from the
per-frame track (positions, radii, visibility: exactly what the renderer
draws) plus the drawn scenery (pillars, shaded circle). A catch is a ball that
is drawn on frame f-1 and gone on frame f; its catcher is the drawn ball whose
edge was closest to it on frame f-1. Clock readings use the renderer's clock
(floor(f * 10 / 24) tenths). What-ifs re-run the simulation with the change
and read the re-run's track the same way."""
from __future__ import annotations
import functools
import json
import math

import render
import sim


@functools.lru_cache(maxsize=64)
def _events(key: str, mods: tuple = ()):
    scene = json.loads(key)
    tr = sim.track(scene, mods)
    n = len(scene["balls"])
    ev = []
    for f in range(1, len(tr)):
        for j in range(n):
            if tr[f - 1][j][3] and not tr[f][j][3]:
                x, y, r, _ = tr[f - 1][j]
                h = min((i for i in range(n) if i != j and tr[f - 1][i][3]),
                        key=lambda i: math.dist((x, y), tr[f - 1][i][:2]) - tr[f - 1][i][2] - r)
                ev.append((f, h, j))
    return tr, ev


def _lbl(scene, i):
    b = scene["balls"][i]
    return f"the {b['color']} {b['pattern']} ball"


def _clock(f):
    return render.clock_tenths(f)


def _fmt(tenths):
    return f"{tenths / 10:.1f} s"


def _frames_showing(n_frames, tenths):
    return [f for f in range(n_frames) if _clock(f) == tenths]


def answer(key: str, p: dict, scene: dict):
    sk = json.dumps(scene, sort_keys=True)
    tr, ev = _events(sk)
    n = len(scene["balls"])
    alive = [i for i in range(n) if tr[-1][i][3]]
    by = {i: [v for _, h, v in ev if h == i] for i in range(n)}
    when = {v: f for f, _, v in ev}
    r0 = [tr[0][i][2] for i in range(n)]
    rf = [tr[-1][i][2] if tr[-1][i][3] else tr[when[i] - 1][i][2] for i in range(n)]
    big = lambda ids, rs: max(ids, key=lambda i: rs[i])  # noqa: E731
    if key == "n_start":
        return sum(r[3] for r in tr[0])
    if key == "n_end":
        return len(alive)
    if key == "n_catches":
        return len(ev)
    if key == "pattern_of":
        return scene["balls"][p["x"]]["pattern"]
    if key == "color_of":
        return scene["balls"][p["x"]]["color"]
    if key == "biggest_start":
        return _lbl(scene, big(range(n), r0))
    if key == "n_pillars":
        return len(scene["pillars"])
    if key == "zone_end":
        z = scene["zone"]
        return sum(math.dist(tr[-1][i][:2], (z["x"], z["y"])) + tr[-1][i][2] <= z["r"] for i in alive)
    if key == "caught_by":
        return _lbl(scene, next(h for _, h, v in ev if v == p["x"]))
    if key == "first_victim":
        return _lbl(scene, by[p["x"]][0])
    if key == "caught_first":
        return _lbl(scene, min((p["x"], p["y"]), key=when.get))
    if key == "bigger_end":
        return _lbl(scene, big((p["x"], p["y"]), rf))
    if key == "never_caught":
        return _lbl(scene, p["x"]) if p["x"] in alive else None
    if key == "is_catcher":
        return _lbl(scene, p["x"]) if by[p["x"]] else None
    if key == "first_caught":
        return _lbl(scene, ev[0][2])
    if key == "catch_time":
        return _fmt(_clock(when[p["x"]]))
    if key == "n_caught_by":
        return len(by[p["x"]])
    if key == "left_at":
        f = _frames_showing(len(tr), p["t"])[0]
        return sum(r[3] for r in tr[f])
    if key == "time_between":
        return _fmt(abs(_clock(when[p["x"]]) - _clock(when[p["y"]])))
    if key == "first_catch_time":
        return _fmt(_clock(ev[0][0]))
    if key == "left_after":
        f = when[p["x"]]
        return sum(r[3] for r in tr[f])
    if key == "catch_order":
        return ", then ".join(_lbl(scene, v) for _, _, v in ev)
    if key == "top_catcher":
        return _lbl(scene, max(range(n), key=lambda i: len(by[i])))
    if key == "nth_victim":
        return _lbl(scene, ev[p["k"]][2])
    if key == "caught_between":
        fa = _frames_showing(len(tr), p["t1"])[0]
        fb = _frames_showing(len(tr), p["t2"])[0]
        return sum(fa < f <= fb for f, _, _ in ev)
    if key == "biggest_end":
        return _lbl(scene, big(alive, rf))
    if key.startswith("cf_"):
        tag, x = p["tag"], p["x"]
        mods = ((tag,),) if x is None else ((tag, x),)
        ctr, cev = _events(sk, mods)
        if key == "cf_n_end":
            return sum(r[3] for r in ctr[-1])
        if key == "cf_first":
            return _lbl(scene, cev[0][2]) if cev else None
        if key == "cf_n_caught_by":
            return sum(h == x for _, h, _ in cev)
    return None
