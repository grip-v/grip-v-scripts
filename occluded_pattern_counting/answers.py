"""Independent answers for occluded_pattern_counting v3. They never read the
generator's hold list or per-hold logs: everything comes from the drawn
objects and the per-frame occluder rectangles (exactly what the renderer
draws). An object is hidden on a frame when its whole disc is inside a
rectangle, visible when it touches none. The occluder is "still" on frames
where no rectangle moved since the previous frame; still stretches are found
from the track itself. Clock readings use the renderer's clock
(floor(f * 10 / 24) tenths). What-ifs re-run the track with the change and
read it the same way."""
from __future__ import annotations
import functools
import json

import render
import sim

FAMILY_TEXT = {"grid": "a rectangular grid", "ring": "a ring", "triangle": "a triangle", "honeycomb":
               "a honeycomb (offset rows)"}


def _hidden(rc, o):
    return (rc[0] + o["r"] < o["x"] < rc[2] - o["r"]) and (rc[1] + o["r"] < o["y"] < rc[3] - o["r"])


@functools.lru_cache(maxsize=64)
def _read(key: str, mods: tuple = ()):
    scene = json.loads(key)
    tr = sim.track(scene, mods)
    objs = scene["objects"]
    hid = [frozenset(i for i, o in enumerate(objs) if any(_hidden(rc, o) for rc in rects)) for rects in tr]
    still = [f > 0 and tr[f] == tr[f - 1] for f in range(len(tr))]
    still[0] = len(tr) > 1 and tr[1] == tr[0]
    runs, f = [], 0
    while f < len(tr):
        if still[f]:
            g = f
            while g + 1 < len(tr) and still[g + 1]:
                g += 1
            runs.append((f, g))
            f = g + 1
        else:
            f += 1
    return tr, hid, runs


def _clock(f):
    return render.clock_tenths(f)


def _fmt(t):
    return f"{t / 10:.1f} s"


def _frame(n, t):
    return next(f for f in range(n) if _clock(f) == t)


def answer(key: str, p: dict, scene: dict):
    sk = json.dumps(scene, sort_keys=True)
    tr, hid, runs = _read(sk)
    objs = scene["objects"]
    n, nf = len(objs), len(tr)
    cols = [o["color"] for o in objs]
    base = max(sorted(set(cols)), key=cols.count)
    rh = [hid[(a + b) // 2] for a, b in runs]
    if key == "n_visible_start":
        return n - len(hid[0])
    if key == "family":
        return FAMILY_TEXT[scene["family"]]
    if key == "object_shape":
        return scene["shape"]
    if key == "main_color":
        return base
    if key == "n_occluders":
        return len(tr[0])
    if key == "start_vs_end":
        a, b = n - len(hid[0]), n - len(hid[-1])
        return "more at the start" if a > b else "more at the end" if b > a else "the same number"
    if key == "slide_dir":
        j = p["j"]
        f0 = runs[0][1]
        f1 = runs[1][0]
        dx = tr[f1][j][0] - tr[f0][j][0]
        dy = tr[f1][j][1] - tr[f0][j][1]
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"
    if key == "odd_end":
        return "hidden" if p["x"] in hid[-1] else "visible"
    if key == "odd_start":
        return "yes" if p["x"] in hid[0] else "no"
    if key == "after_first_move":
        a, b = len(rh[0]), len(rh[1])
        return "fewer" if b < a else "more" if b > a else "the same number"
    if key == "occ_more_end":
        cnt = [sum(_hidden(rc, o) for o in objs) for rc in tr[-1]]
        return f"the {scene['occluders'][cnt.index(max(cnt))]['color']} occluder"
    if key == "n_visible_at":
        return n - len(hid[_frame(nf, p["t"])])
    if key == "n_hidden_at":
        return len(hid[_frame(nf, p["t"])])
    if key == "first_move_time":
        return _fmt(_clock(runs[0][1] + 1))
    if key == "last_stop_time":
        return _fmt(_clock(runs[-1][0] - 1))           # arrival frame: last frame that differs from its predecessor
    if key == "first_hold_len":
        return _fmt(_clock(runs[0][1] + 1) - _clock(0))
    if key == "n_total":
        return n
    if key == "n_revealed":
        return len(hid[0] - hid[-1])
    if key == "n_covered":
        return len(hid[-1] - hid[0])
    if key == "n_ever_hidden":
        return len(frozenset().union(*rh))
    if key == "n_gaps":
        return len(scene["gaps"])
    if key == "n_odd":
        return sum(c != base for c in cols)
    if key == "max_hidden":
        return max(len(h) for h in rh)
    if key == "side_start":
        return _side(scene, hid[0])
    if key == "side_end":
        return _side(scene, hid[-1])
    if key == "most_stop":
        sizes = [len(h) for h in rh]
        return f"its {('first', 'second', 'third', 'fourth')[sizes.index(max(sizes))]} position"
    if key == "first_move_net":
        unc, cov = len(rh[0] - rh[1]), len(rh[1] - rh[0])
        return "it uncovers more" if unc > cov else "it covers more" if cov > unc else "the same number"
    if key.startswith("cf_"):
        mods = ((p["tag"],),) if p["x"] is None else ((p["tag"], p["x"]),)
        ctr, chid, _ = _read(sk, mods)
        if key == "cf_hidden_end":
            return len(chid[-1])
        if key == "cf_visible_end":
            return n - len(chid[-1])
        if key == "cf_never_seen":
            mids = [(a + b) // 2 for a, b in runs]
            return len(frozenset.intersection(*[chid[m] for m in mids]))
    return None


def _side(scene, hid):
    objs = scene["objects"]
    xs, ys = [o["x"] for o in objs], [o["y"] for o in objs]
    mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    dx = sum(objs[i]["x"] for i in hid) / len(hid) - mx
    dy = sum(objs[i]["y"] for i in hid) / len(hid) - my
    dx /= max(1.0, (max(xs) - min(xs)) / 2)
    dy /= max(1.0, (max(ys) - min(ys)) / 2)
    if abs(dx) > abs(dy):
        return "the right part" if dx > 0 else "the left part"
    return "the bottom part" if dy > 0 else "the top part"
