"""Independent answers for rainbow_bagel, read off what the renderer draws.

Only render.track (each piece's drawn outline and fill colour on every frame)
and render.layout (bagel centre, ruler ticks, screen and tray rectangles) are
used; none of the plan fields (lift/land frames, trays, rings, the marked
index, the gold frame) are read.

  rings and slices    radii and angles of the outlines on the first cut frame
  units               outline radii divided by the ruler's tick spacing
  areas               pi * (R^2 - r^2) * span / 360 from those measurements
  events              lift = first frame a piece moves after the cut,
                      rest = first frame from which it never moves again,
                      gold = first frame its fill is gold
  trays               where each piece is drawn on the last frame
  counterfactuals     the same measurements with the changed rule applied
                      (pieces moved between trays, gold piece's flight shifted,
                      longer screen passes)"""
from __future__ import annotations
import math

import numpy as np

import render
import sim

_CACHE: dict = {}


class Obs:
    def __init__(self, scene):
        tr = render.track(scene)
        lay = render.layout(scene)
        self.nf = len(tr)
        n = len(tr[0])
        self.n = n
        cx, cy = lay["bagel_centre"]
        step = lay["ruler"]["step"]
        f0 = sim.CUT0           # first frame the slices are drawn, before they part
        self.R, self.r, self.mid, self.span = [], [], [], []
        for i in range(n):
            v = tr[f0][i]["poly"]
            rad = np.hypot(v[:, 0] - cx, v[:, 1] - cy)
            self.R.append(int(round(rad.max() / step)))
            self.r.append(int(round(rad.min() / step)))
            ang = np.degrees(np.arctan2(-(v[:, 1] - cy), v[:, 0] - cx)) % 360
            c = v.mean(0)
            self.mid.append(math.degrees(math.atan2(-(c[1] - cy), c[0] - cx)) % 360)
            rel = (ang - self.mid[-1] + 180) % 360 - 180
            self.span.append(int(round(rel.max() - rel.min())))
        radii = sorted(set(self.r))
        self.ring = [radii.index(x) + 1 for x in self.r]
        self.rings = len(radii)
        self.hole = min(self.r)
        self.outer = max(self.R)
        self.slices = int(round(360 / self.span[0]))
        names = sim.SLICES[self.slices]
        self.slice_name = [min(names, key=lambda m: abs((m[0] - a + 180) % 360 - 180))[1] for a in self.mid]
        cen = np.array([[tr[f][i]["poly"].mean(0) for i in range(n)] for f in range(self.nf)])
        moved = np.any(np.abs(np.diff(cen, axis=0)) > 1e-9, axis=2)          # (nf-1, n): moved into frame f+1
        self.lift = [int(sim.CUT1 + np.flatnonzero(moved[sim.CUT1:, i])[0]) for i in range(n)]
        self.rest = [int(np.flatnonzero(moved[:, i])[-1] + 1) for i in range(n)]
        gold = [[tuple(tr[f][i]["fill"]) == sim.GOLD for f in range(self.nf)] for i in range(n)]
        self.gold = next(i for i in range(n) if gold[i][-1])
        self.gold_frame = gold[self.gold].index(True)
        self.tray = []
        for i in range(n):
            x = cen[-1, i, 0]
            self.tray.append(next(t for t, (x0, y0, x1, y1) in lay["trays"].items() if x0 <= x <= x1))
        self.trays = list(lay["trays"])

    def area(self, i, slices=None):
        return math.pi * (self.R[i] ** 2 - self.r[i] ** 2) * (self.span[i] if slices is None else 360 / slices) / 360

    def tray_area(self, t, trays=None):
        trays = trays or self.tray
        return sum(self.area(i) for i in range(self.n) if trays[i] == t)

    def name(self, i):
        unit = sim.UNIT[self.slices]
        return f"ring {self.ring[i]}, {self.slice_name[i]} {unit}"

    def ref(self, i):
        return f"the {self.slice_name[i]} {sim.UNIT[self.slices]} of ring {self.ring[i]}"

    def resting(self, f, tray=None, skip=None):
        return sum(1 for i in range(self.n) if i != skip and self.rest[i] < f and (tray is None or self.tray[i] == tray))


def _obs(scene) -> Obs:
    key = id(scene)
    if key not in _CACHE or _CACHE[key][0] is not scene:
        _CACHE.clear()
        _CACHE[key] = (scene, Obs(scene))
    return _CACHE[key][1]


def _tenths(f: int) -> str:
    t = (f * 10) // sim.FPS
    return f"{t // 10}.{t % 10} s"


def _cmp(counts: dict) -> str:
    if len(counts) == 2:
        (ta, a), (tb, b) = counts.items()
        if a == b:
            return "same number"
        return f"{ta}, by {a - b}" if a > b else f"{tb}, by {b - a}"
    top = max(counts.values())
    lead = [t for t, c in counts.items() if c == top]
    return lead[0] if len(lead) == 1 else "a tie for the most"


def answer(key: str, p: dict, scene: dict):
    o = _obs(scene)
    n = o.n
    if key == "n_pieces":
        return n
    if key == "n_rings":
        return o.rings
    if key == "tray_count":
        return sum(1 for t in o.tray if t == p["tray"])
    if key == "tray_more_pieces":
        return _cmp({t: o.tray.count(t) for t in o.trays})
    if key == "tray_more_outer":
        return _cmp({t: sum(1 for i in range(n) if o.tray[i] == t and o.ring[i] == o.rings) for t in o.trays})
    if key == "ring_most":
        c = {f"ring {j}": sum(1 for i in range(n) if o.tray[i] == p["tray"] and o.ring[i] == j)
             for j in range(1, o.rings + 1)}
        top = max(c.values())
        lead = [r for r, v in c.items() if v == top]
        return lead[0] if len(lead) == 1 else "a tie for the most"
    if key == "first_lift":
        ids = p["pieces"]
        pick = min(ids, key=lambda i: o.lift[i]) if p["order"] == "first" else max(ids, key=lambda i: o.lift[i])
        return o.ref(pick)
    if key == "tray_area":
        return f"{o.tray_area(p['tray']):.1f}"
    if key == "gold_area":
        return f"{o.area(o.gold):.1f}"
    if key == "ruler_radius":
        return o.hole if p["which"] == "hole" else o.outer
    if key == "land_time":
        fr = sorted(o.rest[i] for i in range(n) if o.tray[i] == p["tray"])
        return _tenths(fr[0] if p["which"] == "first" else fr[-1])
    if key == "gold_time":
        return _tenths(o.gold_frame)
    if key == "count_at_time":
        sec, tenth = p["clock"].split()[0].split(".")
        x = int(sec) * 10 + int(tenth)
        f0 = next(f for f in range(o.nf) if (f * 10) // sim.FPS == x)
        return o.resting(f0 + 1, p["tray"])
    if key == "marked_origin":
        return o.name(o.gold)
    if key == "piece_tray":
        i = p["piece"]
        return f"{o.tray[i]} tray, {o.resting(o.rest[i], o.tray[i])} already there"
    if key == "first_land_origin":
        i = min((j for j in range(n) if o.tray[j] == p["tray"]), key=lambda j: o.rest[j])
        return o.name(i)
    if key == "gold_already":
        return o.resting(o.rest[o.gold], skip=o.gold)
    if key == "cf_move_ring":
        j = p["ring"]
        new = [(("right" if t == "left" else "left") if o.ring[i] == j else t) for i, t in enumerate(o.tray)]
        return f"{abs(o.tray_area('left', new) - o.tray_area('right', new)):.1f}"
    if key == "cf_count_move":
        return sum(1 for i in range(n) if o.tray[i] == p["to"] or o.ring[i] == p["ring"])
    if key == "cf_gold_other":
        new = [(p["to"] if i == o.gold else t) for i, t in enumerate(o.tray)]
        return f"{o.tray_area(p['to'], new):.1f}"
    if key == "cf_more_slices":
        return f"{o.area(o.gold, slices=p['slices']):.1f}"
    if key == "cf_gold_delay":
        new = o.rest[o.gold] + p["shift_frames"]
        return sum(1 for i in range(n) if i != o.gold and o.rest[i] < new)
    if key == "cf_screen_slow":
        return _tenths(max(o.rest) + p["extra_frames"])
    return None
