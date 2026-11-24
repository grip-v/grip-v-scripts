"""Independent answers for vessels_and_clocks.

Never reads sim.py's flow parameters (h0, q, k, K, valve times) or its
closed forms. Everything comes from a per-frame *track* of what is drawn:
the water height in px of every vessel, whether it overflows, whether a tap
pours into it, the width of its drain stream, and whether each valve handle
is open. The track is built from `render.state` (or, in frame_check.py,
from decoded MP4 pixels). The static drawing gives the rest: the vessels'
pixel widths (layout), their colours (names), which vessels a pipe joins,
and the ruler (the px positions of the "0" and "10" numbers give the bottom
and the px per mark).

From the track: levels = px / (px per mark); overflow / empty / valve
frames are the first frames showing them; the clock reading of frame f is
f * 10 // 24 tenths. Taps and drains are told apart by what is drawn (a
pouring tap, a drain stream); the drain law by whether the level falls
slower in the second half of its fall than in the first. What-ifs fit the
flow law to the track (rise / fall rate in marks per s, the slope of
sqrt(level) for a water clock, the exchange rate of a joined pair from the
decay of the level difference) and integrate it numerically (RK4) under
the change.
"""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FPS = 24
FULL = 10.0
DT = 1 / 2400
DEPTH = " (All vessels are equally deep, so a wider vessel holds more water per mark.)"
NONE_FIRST = "none of them"


def fmt_clock(tenths):
    return f"{tenths / 10:.1f} s"


def clock_of(frame):
    return frame * 10 // FPS


def track_from_render(scene) -> dict:
    import render
    F = scene["frames"]
    n = len(scene["vessels"])
    tr = {"water": [[] for _ in range(n)], "spill": [[] for _ in range(n)], "pour": [[] for _ in range(n)],
          "stream": [[] for _ in range(n)], "open": [[] for _ in scene["pairs"]]}
    for f in range(F):
        s = render.state(scene, f)
        for i, v in enumerate(s["vessels"]):
            tr["water"][i].append(v["water_px"])
            tr["spill"][i].append(v["spill"])
            tr["pour"][i].append(v["pour"] > 0)
            tr["stream"][i].append(v["stream"])
        for k, p in enumerate(s["pipes"]):
            tr["open"][k].append(p["open"])
    return tr


@functools.lru_cache(maxsize=32)
def _obs_cached(key: str):
    scene = json.loads(key)
    return Obs(scene, track_from_render(scene))


def obs(scene):
    return _obs_cached(json.dumps(scene, sort_keys=True))


def _rk4(rhs, h0, t_end, stop):
    """Integrate dh/dt = rhs(h) from h0; returns (time `stop(h)` first holds or None, h at t_end)."""
    h, t = h0, 0.0
    hit = None
    steps = int(math.ceil(t_end / DT))
    for _ in range(steps):
        k1 = rhs(h)
        k2 = rhs(h + DT / 2 * k1)
        k3 = rhs(h + DT / 2 * k2)
        k4 = rhs(h + DT * k3)
        hn = h + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if hit is None and stop(hn):
            lim = FULL if hn >= FULL else 0.0                    # interpolate the crossing inside the step
            hit = t + DT * (lim - h) / (hn - h) if hn != h else t + DT
            hn = lim
        h, t = hn, t + DT
        if hit is not None:
            break
    return hit, h


class Obs:
    def __init__(self, scene, tr):
        self.scene, self.tr = scene, tr
        self.F = len(tr["water"][0])
        self.n = len(tr["water"])
        labs = {lab["text"]: lab for lab in scene["labels"]}
        self.bottom = labs["0"]["y"]
        self.M = (labs["0"]["y"] - labs["10"]["y"]) / 10
        self.wpx = [x1 - x0 for x0, x1 in scene["layout"]["x"]]
        self.cols = [v["colour"] for v in scene["vessels"]]
        self.pipes = [tuple(p) for p in scene["pairs"]]
        self.tl = (self.F - 1) / FPS

    # ---- raw readings
    def lev(self, i, f):
        return self.tr["water"][i][f] / self.M

    def w(self, i):
        return self.wpx[i] / 100

    def role(self, i):
        if self.tr["pour"][i][0]:
            return "tap"
        if any(i in p for p in self.pipes):
            return "pair"
        if self.tr["stream"][i][0] > 0:
            return self.law(i)
        return "still"

    def partner(self, i):
        for a, b in self.pipes:
            if i in (a, b):
                return b if i == a else a
        return None

    def pipe_of(self, i):
        for k, (a, b) in enumerate(self.pipes):
            if i in (a, b):
                return k
        return None

    def first(self, seq):
        return next((f for f, x in enumerate(seq) if x), None)

    def ev_frame(self, i):
        """Frame first showing vessel i overflow (tap) or empty (drain); None if not in the clip."""
        if self.tr["pour"][i][0]:
            return self.first(self.tr["spill"][i])
        if self.tr["stream"][i][0] > 0:
            return self.first([x == 0 for x in self.tr["water"][i]])
        return None

    def valve_frame(self, a):
        return self.first(self.tr["open"][self.pipe_of(a)])

    def frame_of(self, ev):
        kind, i = ev
        return self.valve_frame(i) if kind == "valve" else self.ev_frame(i)

    def ev_kind(self, i):
        return "overflows" if self.tr["pour"][i][0] else "runs empty"

    # ---- fitted laws
    def _fall_span(self, i):
        """Last frame before the level of a drain drops under half a mark (or the last frame)."""
        fb = 0
        for f in range(self.F):
            if self.lev(i, f) >= 0.5:
                fb = f
            else:
                break
        return fb

    def law(self, i):
        fb = self._fall_span(i)
        m = max(1, fb // 2)
        early = (self.lev(i, 0) - self.lev(i, m)) / m
        late = (self.lev(i, m) - self.lev(i, 2 * m)) / m
        return "steady" if late >= 0.92 * early else "leaky"

    def rate(self, i):
        """Tap: rise in marks/s before overflowing; steady drain: fall in marks/s; leaky: fall of sqrt(level)/s."""
        role = self.role(i)
        if role == "tap":
            fe = self.ev_frame(i)
            fa = (fe - 2) if fe is not None else self.F - 1
            return (self.lev(i, fa) - self.lev(i, 0)) / (fa / FPS)
        fb = self._fall_span(i)
        if role == "steady":
            return (self.lev(i, 0) - self.lev(i, fb)) / (fb / FPS)
        return (math.sqrt(self.lev(i, 0)) - math.sqrt(self.lev(i, fb))) / (fb / FPS)

    def rhs(self, i, width_factor=1.0, rate_factor=1.0):
        role, r = self.role(i), self.rate(i) * rate_factor / width_factor
        if role == "tap":
            return lambda h: r if h < FULL else 0.0
        if role == "steady":
            return lambda h: -r if h > 0 else 0.0
        return lambda h: -2 * r * math.sqrt(max(h, 0.0))

    def cf_event(self, i, width_factor=1.0, rate_factor=1.0, dh=0.0, t_end=None):
        """(event time or None, level at t_end) for vessel i under the change."""
        stop = (lambda h: h >= FULL) if self.role(i) == "tap" else (lambda h: h <= 0)
        t_end = self.tl if t_end is None else t_end
        return _rk4(self.rhs(i, width_factor, rate_factor), self.lev(i, 0) + dh, t_end, stop)

    def settle(self, a, b, ha=None, hb=None):
        """Re-simulate a joined pair from the measured exchange rate until the levels agree."""
        fv = self.valve_frame(a)
        d1 = self.lev(a, fv) - self.lev(b, fv)
        d2 = self.lev(a, fv + 4) - self.lev(b, fv + 4)
        lam = math.log(d1 / d2) / (4 / FPS)
        wa, wb = self.w(a), self.w(b)
        K = lam / (1 / wa + 1 / wb)
        ha = self.lev(a, 0) if ha is None else ha
        hb = self.lev(b, 0) if hb is None else hb
        dt = 1 / 2400
        for _ in range(400000):
            flow = K * (ha - hb)
            ha, hb = ha - flow / wa * dt, hb + flow / wb * dt
            if abs(ha - hb) < 1e-7:
                break
        return (ha + hb) / 2

    # ---- naming
    def nm(self, i):
        return f"the {self.cols[i]} vessel"

    def desc(self, ev):
        kind, i = ev
        if kind == "valve":
            return f"the valve between the {self.cols[i]} and {self.cols[self.partner(i)]} vessels opens"
        return f"{self.nm(i)} {self.ev_kind(i)}"

    def short(self, ev):
        kind, i = ev
        if kind == "valve":
            return f"{self.cols[i]}-{self.cols[self.partner(i)]} valve opens"
        return f"{self.cols[i]} {'overflows' if self.ev_kind(i) == 'overflows' else 'empties'}"

    def vol(self, i, f):
        return self.lev(i, f) * self.w(i)

    def what(self, i):
        return "overflow" if self.tr["pour"][i][0] else "run empty"


def _cf_clock(o: Obs, i, t):
    if t is None:
        return f"it would not {o.what(i)} before the clip ends"
    if t > o.tl:
        return f"it would not {o.what(i)} before the clip ends"
    return fmt_clock(clock_of(math.ceil(t * FPS)))


def derive(key, p, o: Obs):
    n, F, last = o.n, o.F, o.F - 1
    if key == "n_overflow":
        return sum(1 for i in range(n) if any(o.tr["spill"][i]))
    if key == "n_empty_end":
        return sum(1 for i in range(n) if o.tr["water"][i][last] == 0)
    if key == "n_taps":
        return sum(1 for i in range(n) if o.tr["pour"][i][0])
    if key == "n_rising":
        return sum(1 for i in range(n) if any(o.tr["water"][i][f + 1] > o.tr["water"][i][f] + 1e-9
                                              for f in range(F - 1)))
    if key == "first_event":
        fr = {i: o.ev_frame(i) for i in range(n) if o.ev_frame(i) is not None}
        return o.nm(min(fr, key=fr.get)) if fr else NONE_FIRST
    if key == "widest":
        return o.nm(max(range(n), key=lambda i: o.wpx[i]))
    if key == "highest_start":
        return o.nm(max(range(n), key=lambda i: o.lev(i, 0)))
    if key == "steady_vs_leaky":
        x, y = p["x"], p["y"]
        laws = {i: o.law(i) for i in (x, y)}
        st = [i for i in (x, y) if laws[i] == "steady"]
        return o.nm(st[0]) if len(st) == 1 else None
    if key == "more_water_end":
        x, y = p["x"], p["y"]
        return o.nm(x if o.vol(x, last) > o.vol(y, last) else y)
    if key == "higher_end":
        x, y = p["x"], p["y"]
        return o.nm(x if o.lev(x, last) > o.lev(y, last) else y)
    if key == "faster_rise":
        x, y = p["x"], p["y"]
        return o.nm(x if o.rate(x) > o.rate(y) else y)
    if key == "earlier_event":
        a, b = tuple(p["a"]), tuple(p["b"])
        return o.desc(a if o.frame_of(a) < o.frame_of(b) else b)
    if key == "valve_flow":
        x, y = p["x"], p["y"]
        return o.nm(x if o.lev(x, last) > o.lev(x, 0) else y)
    if key == "level_end":
        return round(o.lev(p["x"], last))
    if key == "drop_marks":
        return round(o.lev(p["x"], 0) - o.lev(p["x"], last))
    if key == "event_clock":
        return fmt_clock(clock_of(o.ev_frame(p["x"])))
    if key == "valve_clock":
        return fmt_clock(clock_of(o.valve_frame(p["x"])))
    if key == "level_at_clock":
        f = next(f for f in range(F) if clock_of(f) == p["sec"] * 10)
        return round(o.lev(p["x"], f))
    if key == "rise_rate":
        return f"{round(2 * o.rate(p['x'])) / 2:.1f} marks per second"
    if key == "event_order":
        evs = [tuple(e) for e in p["ev"]]
        return " → ".join(o.short(e) for e in sorted(evs, key=o.frame_of))
    if key == "most_water_end":
        return o.nm(max(range(n), key=lambda i: o.vol(i, last)))
    if key == "shared_level":
        return round(o.settle(p["x"], p["y"]))
    if key == "time_between":
        a, b = tuple(p["a"]), tuple(p["b"])
        return fmt_clock(clock_of(o.frame_of(b)) - clock_of(o.frame_of(a)))
    if key == "most_lost":
        lost = {i: o.vol(i, 0) - o.vol(i, last) for i in range(n) if o.lev(i, 0) - o.lev(i, last) >= 0.5}
        return o.nm(max(lost, key=lost.get))
    if key == "cf_wider_time":
        t, _ = o.cf_event(p["x"], width_factor=2.0, t_end=o.tl + 1.0)
        return _cf_clock(o, p["x"], t)
    if key == "cf_fuller_time":
        t, _ = o.cf_event(p["x"], dh=p["dh"], t_end=o.tl + 1.0)
        return _cf_clock(o, p["x"], t)
    if key == "cf_wider_level":
        _, h = o.cf_event(p["x"], width_factor=2.0)
        return round(h)
    if key == "cf_shared_extra":
        x, y = p["x"], p["y"]
        a, b = (x, y) if o.partner(x) == y else (None, None)
        if a is None:
            return None
        return round(o.settle(x, y, ha=o.lev(x, 0) + p["dh"]))
    if key == "cf_valve_never":
        vol = [o.vol(i, 0) if o.partner(i) is not None else o.vol(i, last) for i in range(n)]
        return o.nm(max(range(n), key=lambda i: vol[i]))
    if key == "cf_faster_tap":
        x = p["x"]
        times = {}
        for i in range(n):
            if o.partner(i) is not None:
                continue
            if i == x:
                times[i] = o.cf_event(i, rate_factor=2.0, t_end=o.tl + 1.0)[0]
            else:
                fe = o.ev_frame(i)
                times[i] = fe / FPS if fe is not None else None      # frame resolution is enough (gap >= 0.3 s)
        times = {i: t for i, t in times.items() if t is not None and t <= o.tl}
        return o.nm(min(times, key=times.get)) if times else NONE_FIRST
    return None


def answer(key: str, p: dict, scene: dict):
    return derive(key, p, obs(scene))
