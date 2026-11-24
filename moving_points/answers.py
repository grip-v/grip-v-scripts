"""Independent answers for moving_points. Nothing uses the closed forms:
each point's velocity is measured from its positions at two sampled times
(as a viewer would), distances are minimised by a fine time scan with
golden-section refinement, and axis crossings are found by scanning for a
sign change."""
from __future__ import annotations
import math

import sim


def _vel(p, scale=1.0):
    (x0, y0), (x1, y1) = sim.pos(p, 0.0), sim.pos(p, 1.0)
    return x0, y0, (x1 - x0) * scale, (y1 - y0) * scale


def _at(p, t, scale=1.0):
    x0, y0, vx, vy = _vel(p, scale)
    return x0 + vx * t, y0 + vy * t


def _closest(a, b, T, sb=1.0):
    f = lambda t: math.dist(_at(a, t), _at(b, t, sb))
    best = min((k * T / 2000 for k in range(2001)), key=f)
    lo, hi = max(0.0, best - T / 1000), min(T, best + T / 1000)
    g = (5 ** 0.5 - 1) / 2
    for _ in range(80):
        m1, m2 = hi - g * (hi - lo), lo + g * (hi - lo)
        if f(m1) < f(m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


def _crosses(p, T):
    n = 0
    for axis in (0, 1):
        signs = [v > 0 for v in (_at(p, k * T / 4000)[axis] for k in range(4001)) if abs(v) > 1e-9]
        n += sum(1 for u, v in zip(signs, signs[1:]) if u != v)
    return n


def _first_y_cross(p, T):
    vals = [(k * T / 4000, _at(p, k * T / 4000)[0]) for k in range(4001)]
    vals = [(t, v > 0) for t, v in vals if abs(v) > 1e-9]
    return next(t for (t, u), (_, v) in zip(vals, vals[1:]) if u != v)


def _xy(x, y):
    return f"({round(x, 6):g}, {round(y, 6):g})"


def answer(key: str, p: dict, scene: dict):
    ps, T = scene["points"], scene["T"]
    pt = lambda i: f"point {ps[i]['name']}"
    sp = lambda i: math.hypot(*_vel(ps[i])[2:])
    if key == "fastest":
        return pt(max(range(len(ps)), key=sp))
    if key == "faster":
        return pt(max((p["x"], p["y"]), key=sp))
    if key == "quadrant_end":
        x, y = _at(ps[p["x"]], T)
        return ("top" if y > 0 else "bottom") + " " + ("right" if x > 0 else "left")
    if key == "closer_origin":
        return pt(min((p["x"], p["y"]), key=lambda i: math.hypot(*_at(ps[i], T))))
    if key == "y_cross_first":
        return pt(min((p["x"], p["y"]), key=lambda i: _first_y_cross(ps[i], T)))
    if key == "pos_at":
        return _xy(*_at(ps[p["x"]], p["t"]))
    if key == "dist_end":
        return round(math.dist(_at(ps[p["x"]], T), _at(ps[p["y"]], T)))
    if key == "closest_time":
        return f"{round(_closest(ps[p['x']], ps[p['y']], T) * 2) / 2:.1f} s"
    if key == "n_axis_cross":
        return _crosses(ps[p["x"]], T)
    if key == "cf_half_pos":
        return _xy(*_at(ps[p["x"]], T, 0.5))
    if key == "cf_half_closest":
        return f"{round(_closest(ps[p['y']], ps[p['x']], T, 0.5) * 2) / 2:.1f} s"
    return None
