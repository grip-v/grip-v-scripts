"""Independent answers for translucent_occlusion v3. They never read the
generator's keyframes, kinds or change list: everything comes from the
per-frame track (centre, radius, colour, shape: exactly what the renderer
draws) and the drawn panels. "Completely hidden" is recomputed here by
testing the shape's disc against each panel; hide/reveal frames are the
flips of that test. Clock readings use the renderer's clock
(floor(f * 10 / 24) tenths). What-ifs re-run the track with the change and
read it the same way."""
from __future__ import annotations
import functools
import json
import math

import render
import sim

PANEL_NAMES = {1: ["the panel"], 2: ["the left panel", "the right panel"]}


def _inside(p, x, y, r):
    """True when the disc (x, y, r) lies entirely inside panel p."""
    if p["kind"] == "circle":
        return (x - p["cx"]) ** 2 + (y - p["cy"]) ** 2 <= (p["R"] - r) ** 2 and r <= p["R"]
    return min(x - p["x0"], p["x1"] - x, y - p["y0"], p["y1"] - y) >= r


def _touch(p, x, y, r):
    if p["kind"] == "circle":
        return math.hypot(x - p["cx"], y - p["cy"]) < p["R"] + r
    nx, ny = min(max(x, p["x0"]), p["x1"]), min(max(y, p["y0"]), p["y1"])
    return math.hypot(x - nx, y - ny) < r


@functools.lru_cache(maxsize=64)
def _read(key: str, mods: tuple = ()):
    scene = json.loads(key)
    tr = sim.track(scene, mods)
    ps = sim.panels_for(scene, mods)
    hid = [[a is not None and any(_inside(p, a[0], a[1], a[2]) for p in ps) for a in row] for row in tr]
    hides, revs = {}, {}
    for f in range(1, len(tr)):
        for i in range(len(tr[0])):
            if hid[f][i] and not hid[f - 1][i]:
                hides.setdefault(i, []).append(f)
            if hid[f - 1][i] and not hid[f][i]:
                revs.setdefault(i, []).append(f)
    return tr, ps, hid, hides, revs


def _name(tr, i):
    return f"the {tr[0][i][3]} {tr[0][i][4]}"


def _look(a):
    return f"a {a[3]} {a[4]}"


def _which(ps, a):
    for k, p in enumerate(ps):
        if _inside(p, a[0], a[1], a[2]):
            return k
    return None


def _clock(f):
    return render.clock_tenths(f)


def _fmt(t):
    return f"{t / 10:.1f} s"


def _frame_showing(n, tenths):
    return next(f for f in range(n) if _clock(f) == tenths)


def answer(key: str, p: dict, scene: dict):
    sk = json.dumps(scene, sort_keys=True)
    tr, ps, hid, hides, revs = _read(sk)
    n, nf = len(tr[0]), len(tr)
    names = PANEL_NAMES[len(ps)]
    if key == "n_total":
        return n
    if key == "n_clear_start":
        return sum(not any(_touch(q, *tr[0][i][:3]) for q in ps) for i in range(n))
    if key == "color_of":
        return tr[0][p["x"]][3]
    if key == "shape_of":
        return tr[0][p["x"]][4]
    if key == "panel_shape":
        return "rectangle" if ps[p["p"]]["kind"] == "rect" else "circle"
    if key == "n_panels":
        return len(ps)
    if key == "hid_first":
        return _name(tr, min((p["x"], p["y"]), key=lambda i: hides[i][0]))
    if key == "out_first":
        return _name(tr, min((p["x"], p["y"]), key=lambda i: revs[i][0]))
    if key == "exit_side":
        i = p["x"]
        f0 = revs[i][0]
        k = _which(ps, tr[f0 - 1][i])
        q = ps[k]
        f = next(g for g in range(f0, nf) if not _touch(q, *tr[g][i][:3]))
        x, y = tr[f][i][:2]
        if q["kind"] == "rect":
            if x <= q["x0"] or x >= q["x1"]:
                if q["y0"] <= y <= q["y1"]:
                    return "left" if x <= q["x0"] else "right"
            if y <= q["y0"] or y >= q["y1"]:
                if q["x0"] <= x <= q["x1"]:
                    return "top" if y <= q["y0"] else "bottom"
            return None
        ang = math.degrees(math.atan2(y - q["cy"], x - q["cx"])) % 360
        return ["right", "bottom", "left", "top"][int(((ang + 45) % 360) // 90)]
    if key == "went_behind":
        return _name(tr, p["x"]) if p["x"] in hides or hid[0][p["x"]] else None
    if key == "never_behind":
        return _name(tr, p["x"]) if not any(row[p["x"]] for row in hid) else None
    if key == "which_panel":
        return names[_which(ps, tr[hides[p["x"]][0]][p["x"]])]
    if key == "hide_time":
        return _fmt(_clock(hides[p["x"]][0]))
    if key == "reveal_time":
        return _fmt(_clock(revs[p["x"]][0]))
    if key == "hidden_for":
        i = p["x"]
        return _fmt(_clock(revs[i][0]) - _clock(hides[i][0]))
    if key == "n_hidden_at":
        return sum(hid[_frame_showing(nf, p["t"])])
    if key in ("change_size", "change_look", "before_look"):
        i = p["x"]
        f_in, f_out = hides[i][0], revs[i][0]
        a, b = tr[f_in - 1][i], tr[f_out][i]
        if key == "change_size":
            return "bigger" if b[2] > a[2] else "smaller" if b[2] < a[2] else "the same size"
        return _look(b) if key == "change_look" else _look(a)
    if key in ("reveal_order", "hide_order"):
        ev = revs if key == "reveal_order" else hides
        changed = {i for i in range(n) if tr[0][i][3:] != tr[-1][i][3:]}
        who = sorted((i for i in ev if i not in changed), key=lambda i: ev[i][0])
        return ", then ".join(_name(tr, i) for i in who)
    if key == "n_ever_hidden":
        return sum(any(row[i] for row in hid) for i in range(n))
    if key == "n_hidden_end":
        return sum(hid[-1])
    if key == "max_hidden":
        return max(sum(row) for row in hid)
    if key.startswith("cf_"):
        mods = (("nopanel",),) if p["tag"] == "nopanel" else (("delay", p["x"]),)
        ctr, cps, chid, chides, crevs = _read(sk, mods)
        if key == "cf_n_seen_end":
            return sum(a is not None and not any(_touch(q, *a[:3]) for q in cps) for a in ctr[-1])
        if key == "cf_n_hidden_at":
            return sum(chid[_frame_showing(nf, p["t"])])
        if key == "cf_n_hidden_end":
            return sum(chid[-1])
        if key == "cf_first_out":
            changed = {i for i in range(n) if tr[0][i][3:] != tr[-1][i][3:]}
            first = min(((f[0], i) for i, f in crevs.items() if i not in changed), default=None)
            return _name(tr, first[1]) if first else None
    return None
