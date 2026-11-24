"""Independent answers for simple_harmonic_motion (validator, tests, frame check).

Nothing here reads sim.py's closed-form event lists. Factual answers come
from a per-frame track of what the renderer draws: each mass's centre in
pixels (render.state, or centres located in the decoded MP4 by
frame_check.py) and whether its stop post is still drawn. From the track:
rest crossings are sign changes of the offset from the rest mark
(interpolated between frames), turning points are where the per-frame step
changes sign (refined with a parabola through three frames), periods are
twice the mean gap between successive crossings, amplitudes are the largest
refined offset, and every clock value is the clock text of the relevant
frame.

Counterfactual answers re-simulate the changed system with a fixed-step
RK4 integrator of  m x'' + b x' + k x = 0  (springs) or
theta'' = -(g / L) theta  (pendulums) from the same starting state, and
read the answer off that numerical trajectory (questions.py uses the closed
form instead).
"""
from __future__ import annotations
import json
import math

import render

FPS = 24
_TRACKS: dict = {}
_CF: dict = {}

# ------------------------------------------------------------------ names / formats (shared with questions.py)
AXIS_WORDS = {"horizontal": ("right", "left", "right of", "left of", "right end"),
              "pendulum": ("right", "left", "right of", "left of", "right end"),
              "vertical": ("down", "up", "below", "above", "lowest point")}


def name(scene, colour):
    o = next(o for o in scene["objects"] if o["colour"] == colour)
    return f"the {colour} {o['shape']}"


def fmt_s(v):
    return f"{v:.1f} s"


def fmt_clock(tenths):
    return f"{tenths / 10:.1f} s"


def direction_options(scene):
    pos, neg, pside, nside, _ = AXIS_WORDS[scene["orient"]]
    return [f"moving {m}, {s} its rest mark" for m in (pos, neg) for s in (pside, nside)]


def direction_text(scene, vs, us):
    pos, neg, pside, nside, _ = AXIS_WORDS[scene["orient"]]
    return f"moving {pos if vs > 0 else neg}, {pside if us > 0 else nside} its rest mark"


def order_text(colours):
    return " → ".join(colours)


def clock_tenths(f):
    return f * 10 // FPS


def frame_of_clock(t):
    """First frame whose clock shows t (t in seconds, a multiple of 0.1)."""
    k = round(t * 10)
    f = (k * FPS + 9) // 10
    while clock_tenths(f) < k:
        f += 1
    return f


# ------------------------------------------------------------------ the rendered track
def track(scene):
    key = json.dumps(scene, sort_keys=True)
    if key not in _TRACKS:
        if len(_TRACKS) > 32:
            _TRACKS.clear()
        xy, held = {o["colour"]: [] for o in scene["objects"]}, {o["colour"]: [] for o in scene["objects"]}
        for f in range(render.n_frames(scene)):
            for s in render.state(scene, f):
                xy[s["colour"]].append(s["centre"])
                held[s["colour"]].append(s["held"])
        _TRACKS[key] = {"xy": xy, "held": held}
    return _TRACKS[key]


class Track:
    """Measurements from one track (rendered state, h = 0, or decoded video, h ~ 2 px of jitter).
    Crossings and turning points use a hysteresis band of h px, so jitter cannot fake an event."""

    def __init__(self, scene, tr, h=0.0):
        self.s, self.tr, self.h = scene, tr, h
        self.tick = scene["tick"]
        self.N = len(next(iter(tr["xy"].values())))
        self.u = {}
        for o in scene["objects"]:
            pts = tr["xy"][o["colour"]]
            if scene["orient"] == "vertical":
                self.u[o["colour"]] = [p[1] - o["rest"][1] for p in pts]
            else:
                self.u[o["colour"]] = [p[0] - o["rest"][0] for p in pts]
        self._cr, self._tu = {}, {}

    def release(self, c):
        """Last frame drawn with the stop post (None if it never had one)."""
        h = self.tr["held"][c]
        idx = [f for f, x in enumerate(h) if x]
        return idx[-1] if idx else None

    def crossings(self, c):
        """[(time in frames, first frame on the far side)] of every pass through the rest mark."""
        if c in self._cr:
            return self._cr[c]
        u, h, out = self.u[c], self.h, []
        sgn, last = 0, 0
        for f in range(self.N):
            if abs(u[f]) <= h:
                continue
            s = 1 if u[f] > 0 else -1
            if sgn and s != sgn:
                k = next(k for k in range(last + 1, f + 1) if u[k] == 0 or (u[k] > 0) != (sgn > 0))
                if u[k] == 0:
                    out.append((float(k), k + 1))
                else:
                    a, b = u[k - 1], u[k]
                    out.append((k - 1 + a / (a - b), k))
            sgn, last = s, f
        self._cr[c] = out
        return out

    def turns(self, c):
        """[(time in frames, signed offset)] where the mass stops and turns back (zigzag with band h)."""
        if c in self._tu:
            return self._tu[c]
        u, h, out = self.u[c], self.h, []
        d, ext = 0, 0
        for f in range(1, self.N):
            if d == 0:
                if u[f] - u[ext] > h:
                    d, ext = 1, f
                elif u[ext] - u[f] > h:
                    d, ext = -1, f
            elif d * (u[f] - u[ext]) > 0:
                ext = f
            elif d * (u[ext] - u[f]) > h:
                out.append(self._refine(u, ext))
                d, ext = -d, f
        self._tu[c] = out
        return out

    @staticmethod
    def _refine(u, f):
        if 0 < f < len(u) - 1:
            den = u[f - 1] - 2 * u[f] + u[f + 1]
            if den != 0:
                return f + (u[f - 1] - u[f + 1]) / (2 * den), u[f] - (u[f - 1] - u[f + 1]) ** 2 / (8 * den)
        return float(f), u[f]

    def period(self, c):
        """Twice the mean gap between rest crossings taken at a clear speed (>= 0.6 px per frame)."""
        u = self.u[c]
        cr = [t for t, fa in self.crossings(c) if 0 < fa < self.N and abs(u[fa] - u[fa - 1]) >= 0.6]
        return 2 * (cr[-1] - cr[0]) / (len(cr) - 1) / FPS

    def amplitude(self, c):
        vals = [abs(v) for _, v in self.turns(c)] + [abs(v) for v in self.u[c]]
        return max(vals)

    def moving_sign(self, c, f):
        d = self.u[c][f + 1] - self.u[c][f - 1]
        return 0 if abs(d) <= self.h else (d > 0) - (d < 0)


def _period_of_extreme(T: Track, scene, which):
    objs = scene["objects"]
    cs = [o["colour"] for o in objs]
    if which == "widest":
        c = max(cs, key=T.amplitude)
    elif which == "narrowest":
        c = min(cs, key=T.amplitude)
    elif which == "largest":
        c = max(objs, key=lambda o: o["size"])["colour"]
    elif which == "smallest":
        c = min(objs, key=lambda o: o["size"])["colour"]
    elif which == "longest":
        c = max(objs, key=lambda o: math.dist(o["anchor"], T.tr["xy"][o["colour"]][0]))["colour"]
    elif which == "shortest":
        c = min(objs, key=lambda o: math.dist(o["anchor"], T.tr["xy"][o["colour"]][0]))["colour"]
    elif which == "first_released":
        c = min(cs, key=T.release)
    else:
        c = max(cs, key=T.release)
    return fmt_s(T.period(c))


# ------------------------------------------------------------------ counterfactual re-simulation (RK4)
CF_DT = 1 / 1200


def changed(o, scene, change):
    """Physical parameters + start state after the change: (kind, params, t_start, x0, v0)."""
    if "L" in o:
        L, g = o["L"], scene["g"]
        th0 = math.asin(o["A"] / L)
        x0, v0 = th0 * math.cos(o["phi"]), -th0 * math.sqrt(g / L) * math.sin(o["phi"])
        if change == "length_x4":
            L *= 4
        elif change == "length_d4":
            L /= 4
        elif change == "amp_x2":
            x0, v0 = 2 * x0, 2 * v0
        return "pend", {"g": g, "L": L}, 0.0, x0, v0
    m, k, b = o["m"], o["k"], o["b"]
    if o["mode"] == "free":
        w = math.sqrt(k / m)
        t0, x0, v0 = 0.0, o["A"] * math.cos(o["phi"]), -o["A"] * w * math.sin(o["phi"])
    else:
        t0, x0, v0 = o["t_rel"], o["s"] * o["A"], 0.0
    if change == "mass_x4":
        m *= 4
    elif change == "mass_d4":
        m /= 4
    elif change == "stiff_x4":
        k *= 4
    elif change == "stiff_d4":
        k /= 4
    elif change == "amp_x2":
        x0, v0 = 2 * x0, 2 * v0
    elif change == "no_damping":
        b = 0.0
    elif change.startswith("later_"):
        t0 += float(change[6:])
    return "spring", {"m": m, "k": k, "b": b}, t0, x0, v0


def simulate(o, scene, change, t_end):
    """[(t, x)] sampled every CF_DT from the start state to t_end (x in px for springs, rad for pendulums)."""
    key = (json.dumps(o, sort_keys=True), scene.get("g"), change, round(t_end, 6))
    if key in _CF:
        return _CF[key]
    kind, p, t, x, v = changed(o, scene, change)
    if kind == "pend":
        acc = lambda x_, v_: -p["g"] / p["L"] * x_
    else:
        acc = lambda x_, v_: -(p["b"] * v_ + p["k"] * x_) / p["m"]
    h = CF_DT
    out = [(t, x)]
    while t < t_end:
        k1x, k1v = v, acc(x, v)
        k2x, k2v = v + h / 2 * k1v, acc(x + h / 2 * k1x, v + h / 2 * k1v)
        k3x, k3v = v + h / 2 * k2v, acc(x + h / 2 * k2x, v + h / 2 * k2v)
        k4x, k4v = v + h * k3v, acc(x + h * k3x, v + h * k3v)
        x += h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v += h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += h
        out.append((t, x))
    if len(_CF) > 64:
        _CF.clear()
    _CF[key] = out
    return out


def sim_crossings(path):
    out = []
    for (ta, a), (tb, b) in zip(path, path[1:]):
        if a != 0 and (b == 0 or (a > 0) != (b > 0)):
            out.append(ta + (tb - ta) * a / (a - b))
    return out


def cf_period(o, scene, change):
    t_end = 30.0
    cr = sim_crossings(simulate(o, scene, change, t_end))
    gaps = [b - a for a, b in zip(cr, cr[1:])]
    return 2 * sum(gaps) / len(gaps)


# ------------------------------------------------------------------ the dispatcher
def answer(key: str, p: dict, scene: dict, tr=None, h=0.0):
    """The answer to one question, from the rendered track (or a decoded track `tr` with jitter band h)."""
    T = Track(scene, tr if tr is not None else track(scene), h)
    objs = scene["objects"]
    cs = [o["colour"] for o in objs]
    by = {o["colour"]: o for o in objs}
    far = 1                                   # right / down

    if key == "n_objects":
        return len(T.tr["xy"])
    if key == "shape_of":
        return by[p["c"]]["shape"]
    if key == "pattern_of":
        return by[p["c"]]["pattern"]
    if key == "widest_swing":
        return name(scene, max(cs, key=T.amplitude))
    if key == "largest_object":
        return name(scene, max(objs, key=lambda o: o["size"])["colour"])
    if key == "longest_string":
        return name(scene, max(objs, key=lambda o: math.dist(o["anchor"], T.tr["xy"][o["colour"]][0]))["colour"])
    if key == "first_released":
        return name(scene, min(cs, key=T.release))
    if key == "n_right_at_t":
        f = frame_of_clock(p["t"])
        return sum(1 for c in cs if T.u[c][f] > 0)
    if key == "n_held_at_t":
        f = frame_of_clock(p["t"])
        return sum(1 for c in cs if T.tr["held"][c][f])
    if key == "direction_at_t":
        f = frame_of_clock(p["t"])
        return direction_text(scene, T.moving_sign(p["c"], f), T.u[p["c"]][f])
    if key == "farthest_at_t":
        f = frame_of_clock(p["t"])
        return name(scene, max(cs, key=lambda c: abs(T.u[c][f])))
    if key == "first_to_far_end":
        f = frame_of_clock(p["t"])
        first = {c: next((t for t, v in T.turns(c) if t > f and v * far > 0), math.inf) for c in cs}
        return name(scene, min(cs, key=first.get))
    if key == "n_faster_than":
        mine = T.period(p["c"])
        return sum(1 for c in cs if c != p["c"] and T.period(c) < mine)
    if key == "n_moving_far_at_t":
        f = frame_of_clock(p["t"])
        return sum(1 for c in cs if T.moving_sign(c, f) == far)
    if key == "released_before_count":
        mine = T.release(p["c"])
        return sum(1 for c in cs if T.release(c) < mine)
    if key == "period_of":
        return fmt_s(T.period(p["c"]))
    if key == "amplitude_ticks":
        return round(T.amplitude(p["c"]) / T.tick)
    if key == "peak_ticks":
        t = min(T.turns(p["c"]), key=lambda tv: abs(tv[0] / FPS - p["t"]))
        return round(abs(t[1]) / T.tick)
    if key == "crossing_time":
        f = frame_of_clock(p["t"])
        return fmt_clock(clock_tenths(next(fa for t, fa in T.crossings(p["c"]) if t > f)))
    if key == "crossings_count":
        return sum(1 for t, _ in T.crossings(p["c"]) if p["t0"] * FPS <= t <= p["t1"] * FPS)
    if key == "far_end_count":
        return sum(1 for t, v in T.turns(p["c"]) if v * far > 0 and p["t0"] * FPS <= t <= p["t1"] * FPS)
    if key == "release_time":
        return fmt_clock(clock_tenths(T.release(p["c"])))
    if key == "period_difference":
        return fmt_s(abs(T.period(p["a"]) - T.period(p["b"])))
    if key == "order_by_period":
        return order_text(sorted(p["cs"], key=T.period))
    if key == "total_crossings":
        return sum(1 for c in cs for t, _ in T.crossings(c) if p["t0"] * FPS <= t <= p["t1"] * FPS)
    if key == "period_of_extreme":
        return _period_of_extreme(T, scene, p["which"])
    if key == "return_time":
        r = T.release(p["c"])
        side = 1 if T.u[p["c"]][r] > 0 else -1
        back = next(t for t, v in T.turns(p["c"]) if t > r and v * side > 0)
        return fmt_s((back - r) / FPS)
    if key == "decay_count":
        return sum(1 for _, v in T.turns(p["c"]) if abs(v) >= p["n"] * T.tick)
    # ---------------- counterfactuals (re-simulated)
    if key == "cf_period":
        return fmt_s(cf_period(by[p["c"]], scene, p["change"]))
    if key == "cf_crossings":
        cr = sim_crossings(simulate(by[p["c"]], scene, p["change"], p["t1"] + 0.1))
        return sum(1 for t in cr if p["t0"] <= t <= p["t1"])
    if key == "cf_no_damping":
        path = simulate(by[p["c"]], scene, "no_damping", render.duration(scene))
        return round(max(abs(x) for _, x in path) / T.tick)
    if key == "cf_release_later":
        o = by[p["c"]]
        te = sim_crossings(simulate(o, scene, f"later_{p['d']}", o["t_rel"] + p["d"] + 4.0))[0]
        return fmt_clock(clock_tenths(math.ceil(te * FPS)))
    return None
