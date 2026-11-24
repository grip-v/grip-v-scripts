"""Independent answers for buoyancy_tank.

Nothing here reads the generator's event logs, densities or closed-form
profiles. Everything comes from what the renderer draws:

  * render.track(scene)   per-frame water level and object bottom heights
  * render.outline_cm(o)  the drawn outline of each object
  * render.attached / render.flow_on   whether a string/strap or stream is drawn

Events are found by scanning frames; submerged areas come from shapely
polygon clipping of the drawn outlines; densities are inferred from how
deep an object floats at rest (displaced area / own area), or bounded from
below when it is only ever seen resting on the floor. Counterfactuals are
re-solved by force balance: bisection on the water level with each
object's displaced area from polygon clipping.
"""
from __future__ import annotations
import functools
import json

import shapely
from shapely.geometry import Polygon

import render

STATES = ["floating, with its top above the water", "resting on the floor, completely under water",
          "resting on the floor, with its top above the water", "floating completely under water, off the floor"]
SHAPE_WORD = {"square": "square", "wide": "wide block", "tall": "tall block", "ball": "ball",
              "hexagon": "hexagon", "triangle": "triangle"}
FRACS = {0.25: "about a quarter", 0.5: "about half", 0.75: "about three quarters"}
PAT = {"plain": "plain, no pattern", "striped": "stripes", "spotted": "spots", "checked": "checks"}


@functools.lru_cache(maxsize=None)
def _poly(shape: str, w: float, h: float) -> Polygon:
    return Polygon(render.outline_cm({"shape": shape, "w": w, "h": h}))


def _shape(o) -> Polygon:
    return _poly(o["shape"], o["w"], o["h"])


def sub(o, y: float, level: float) -> float:
    """Drawn area of o (bottom at height y) below the water line."""
    if level <= y:
        return 0.0
    return shapely.clip_by_rect(_shape(o), -1, -1, o["w"] + 1, level - y).area


def area(o) -> float:
    return _shape(o).area


def desc(o) -> str:
    pat = "" if o["pattern"] == "plain" else o["pattern"] + " "
    return f"the {o['colour']} {pat}{SHAPE_WORD[o['shape']]}"


def _state(o, y, level) -> int:
    if y > 0:
        return 0
    return 1 if level >= o["h"] else 2


def _line(v, U):
    k = round(v / U)
    return k if k >= 1 and abs(v / U - k) <= 0.2 else None


class Obs:
    """Everything read off the drawn frames."""

    def __init__(self, scene):
        self.scene = scene
        self.objs = scene["objects"]
        self.n = len(self.objs)
        self.tw = scene["tank"]["w"]
        self.U = scene["style"]["grid"]
        self.tr = render.track(scene)
        self.nf = len(self.tr)
        self.L0, self.y0 = self.tr[0]
        self.Lf, self.yf = self.tr[-1]
        self.water0 = self.water(0)
        self.dwater = self.water(self.nf - 1) - self.water0
        sc = scene["scenario"]
        self.rho, self.lb = [], []
        for i, o in enumerate(self.objs):
            a = area(o)
            r = None
            if self.yf[i] > 0:
                r = sub(o, self.yf[i], self.Lf) / a
            elif sc in ("fill", "drain") and self.y0[i] > 0:
                r = sub(o, self.y0[i], self.L0) / a
            self.rho.append(r)
            grounded = [sub(o, 0.0, L) / a for L, ys in self.tr if ys[i] == 0.0]
            self.lb.append(max(grounded + [0.0]))

    def water(self, k):
        L, ys = self.tr[k]
        return self.tw * L - sum(sub(o, y, L) for o, y in zip(self.objs, ys))

    # ---- events, by scanning frames
    def _first_persistent(self, cond, start=0):
        vals = [cond(k) for k in range(self.nf)]
        for k in range(start, self.nf):
            if vals[k]:
                return k if all(vals[k:]) else None
        return None

    def event(self, i, et):
        o, sc = self.objs[i], self.scene["scenario"]
        y = lambda k: self.tr[k][1][i]                               # noqa: E731
        lv = lambda k: self.tr[k][0]                                 # noqa: E731
        if et == "release":
            for k in range(self.nf):
                if not render.attached(self.scene, i, k):
                    return k if k > 0 else None
            return None
        if et == "enter" and sc == "drop":
            return self._first_persistent(lambda k: lv(k) - y(k) > 1e-9)
        if et == "land" and sc == "drop":
            return self._first_persistent(lambda k: y(k) == 0.0)
        if et == "touchdown" and sc == "drain" and y(0) > 0:
            return self._first_persistent(lambda k: y(k) == 0.0)
        if et == "liftoff" and sc == "fill" and y(0) == 0.0:
            return self._first_persistent(lambda k: y(k) > 0.0)
        if et == "surface" and sc == "rise" and y(0) + o["h"] <= lv(0):
            return self._first_persistent(lambda k: y(k) + o["h"] > lv(k), 1)
        return None

    def floor_count(self, k):
        return sum(1 for y in self.tr[k][1] if y == 0.0)

    def flow_off_frame(self):
        on = [k for k in range(self.nf) if render.flow_on(self.scene, k)]
        return on[-1] + 1 if on else None

    # ---- force-balance settle
    def settle(self, objs, model, water):
        """model[j] = ("rho", r) for a known density, ("floor", lb) for a lower bound.
        Returns (level, bottoms) or None if a lower-bounded object might float."""
        def disp(o, m, L):
            s = sub(o, 0.0, L)
            if m[0] == "rho" and m[1] < 1 and s >= m[1] * area(o):
                return m[1] * area(o)
            return s

        lo, hi = 0.0, 1000.0
        for _ in range(48):
            mid = (lo + hi) / 2
            if self.tw * mid - water - sum(disp(o, m, mid) for o, m in zip(objs, model)) < 0:
                lo = mid
            else:
                hi = mid
        L = (lo + hi) / 2
        ys = []
        for o, m in zip(objs, model):
            a = area(o)
            if m[0] == "floor":
                if m[1] < 0.999 and sub(o, 0.0, L) / a > m[1] - 0.03:
                    return None
                ys.append(0.0)
            elif m[1] < 1 and sub(o, 0.0, L) >= m[1] * a:
                d0, d1 = 0.0, o["h"]
                for _ in range(40):
                    dm = (d0 + d1) / 2
                    if sub(o, 0.0, dm) < m[1] * a:
                        d0 = dm
                    else:
                        d1 = dm
                ys.append(L - (d0 + d1) / 2)
            else:
                ys.append(0.0)
        return L, ys

    def model(self, i, factor=1.0):
        if self.rho[i] is not None:
            return ("rho", self.rho[i] * factor)
        return ("floor", self.lb[i])

    def start_water(self, L0):
        sc = self.scene["scenario"]
        if sc == "drop":
            return self.tw * L0
        disp = 0.0
        for i, o in enumerate(self.objs):
            s = sub(o, 0.0, L0)
            m = self.model(i)
            if sc in ("fill", "drain") and m[0] == "rho" and m[1] < 1 and s >= m[1] * area(o):
                s = m[1] * area(o)
            elif sc in ("fill", "drain") and m[0] == "floor" and m[1] < 0.999 and s / area(o) > m[1] - 0.03:
                return None
            disp += s
        return self.tw * L0 - disp

    def counterfactual(self, kind, p):
        idx = list(range(self.n))
        model = [self.model(i) for i in idx]
        water = self.water0 + self.dwater
        if kind == "remove":
            idx.remove(p["obj"])
            model = [self.model(i) for i in idx]
        elif kind == "mass":
            model[p["obj"]] = self.model(p["obj"], p["factor"])
        elif kind == "water":
            w0 = self.start_water(self.L0 + p["dk"] * self.U)
            if w0 is None:
                return None
            water = w0 + self.dwater
        elif kind == "flow":
            dur = sum(1 for k in range(self.nf) if render.flow_on(self.scene, k))
            water = self.water0 + self.dwater * (dur + 24 * p["ds"]) / dur
        res = self.settle([self.objs[i] for i in idx], model, water)
        if res is None:
            return None
        return idx, model, res[0], res[1]


@functools.lru_cache(maxsize=8)
def _obs(key: str) -> Obs:
    return Obs(json.loads(key))


def answer(key: str, p: dict, scene: dict):
    ob = _obs(json.dumps(scene, sort_keys=True))
    objs, n, U = ob.objs, ob.n, ob.U
    Lf, yf = ob.Lf, ob.yf
    tops = [y + o["h"] for o, y in zip(objs, yf)]
    D = [desc(o) for o in objs]
    if key == "n_objects":
        return n
    if key == "n_float_end":
        return sum(1 for y in yf if y > 0)
    if key == "n_floor_end":
        return sum(1 for y in yf if y == 0.0)
    if key == "end_state":
        i = p["obj"]
        return STATES[_state(objs[i], yf[i], Lf)]
    if key == "shape_of":
        return SHAPE_WORD[objs[p["obj"]]["shape"]]
    if key == "pattern_of":
        return PAT[objs[p["obj"]]["pattern"]]
    if key == "first_event":
        fs = [(ob.event(i, p["event"]), i) for i in p["objs"]]
        fs = sorted(x for x in fs if x[0] is not None)
        return D[fs[0][1]] if fs else None
    if key == "pair_gap":
        fa, fb = ob.event(p["a"], p["event"]), ob.event(p["b"], p["event"])
        dt = (fb - fa) / 24
        first = p["a"] if dt > 0 else p["b"]
        return f"{D[first]}, by {'more' if abs(dt) > 1 else 'less'} than 1 second"
    if key == "pair_share":
        hf = [min(max(Lf - yf[i], 0), objs[i]["h"]) / objs[i]["h"] for i in (p["a"], p["b"])]
        return D[p["a"]] if hf[0] > hf[1] else D[p["b"]]
    if key == "pair_top":
        return D[p["a"]] if tops[p["a"]] > tops[p["b"]] else D[p["b"]]
    if key == "event_time":
        f = ob.event(p["obj"], p["event"])
        return None if f is None else render.timer_text(f)
    if key == "share_under":
        i = p["obj"]
        hf = (Lf - yf[i]) / objs[i]["h"]
        best = min(FRACS, key=lambda v: abs(v - hf))
        return FRACS[best]
    if key == "top_line":
        return _line(tops[p["obj"]], U)
    if key == "water_line":
        return _line(ob.L0 if p["when"] == "start" else Lf, U)
    if key == "extreme":
        v = {"highest_top": tops, "lowest_top": [-t for t in tops], "highest_bottom": list(yf)}[p["what"]]
        return D[max(range(n), key=lambda i: v[i])]
    if key == "least_dense":
        fl = [(ob.rho[i], i) for i in range(n) if yf[i] > 0]
        return D[min(fl)[1]]
    if key == "floor_count_at":
        f = ob.event(p["obj"], p["event"])
        return None if f is None else ob.floor_count(f)
    if key == "floor_count_flow":
        f = ob.flow_off_frame()
        return None if f is None else ob.floor_count(f)
    if key == "kth_event":
        fs = sorted((ob.event(i, p["event"]), i) for i in range(n) if ob.event(i, p["event"]) is not None)
        return D[fs[p["k"] - 1][1]]
    if key == "level_change":
        return _line(abs(Lf - ob.L0), U)
    if key.startswith("cf_"):
        kind = key[3:]
        res = ob.counterfactual(kind, p)
        if res is None:
            return None
        idx, model, Lc, ys = res
        if kind == "mass":
            i = p["obj"]
            return STATES[_state(objs[i], ys[i], Lc)]
        if p["ask"] == "line":
            return _line(Lc, U)
        return sum(1 for y in ys if y == 0.0)
    return None
