"""Independent answers for rotating_rays. No closed forms: the two ray
directions are sampled every millisecond, the relative angle is unwrapped
into a continuous signal, and line-ups / right angles are counted where
that signal crosses a multiple of 360 (or 90 + 180k), with each crossing
refined by bisection. End angles are measured from the sampled
directions."""
from __future__ import annotations
import math

DT = 0.001


def _dir(ray, t, w=None):
    a = math.radians(ray["a0"] + (ray["w"] if w is None else w) * t)
    return math.cos(a), math.sin(a)


def _rel(scene, t, wr=None, wb=None):
    (x1, y1), (x2, y2) = _dir(scene["red"], t, wr), _dir(scene["blue"], t, wb)
    return math.degrees(math.atan2(x2 * y1 - y2 * x1, x1 * x2 + y1 * y2))      # red relative to blue, -180..180


def _unwrapped(scene, wr=None, wb=None):
    n = int(round(scene["T"] / DT))
    out, prev, acc = [], None, 0.0
    for k in range(n + 1):
        a = _rel(scene, k * DT, wr, wb)
        if prev is not None:
            d = a - prev
            if d > 180:
                d -= 360
            elif d < -180:
                d += 360
            acc += d
        else:
            acc = a
        prev = a
        out.append(acc)
    return out


def _crossings(vals, period, offset):
    times = []
    for k in range(len(vals) - 1):
        a, b = (vals[k] - offset) / period, (vals[k + 1] - offset) / period
        if math.floor(a) != math.floor(b):
            times.append((k + 0.5) * DT)
    return times


def answer(key: str, p: dict, scene: dict):
    T, r, b = scene["T"], scene["red"], scene["blue"]
    if key == "faster":
        return "the red ray" if abs(r["w"]) > abs(b["w"]) else "the blue ray"
    if key == "red_dir":
        (x0, y0), (x1, y1) = _dir(r, 0), _dir(r, 0.1)
        return "counter-clockwise" if x0 * y1 - y0 * x1 > 0 else "clockwise"
    if key == "n_turns":
        ray = r if p["ray"] == "red" else b
        return int(abs(ray["w"]) * T // 360)
    if key == "same_dir":
        return "yes" if (r["w"] > 0) == (b["w"] > 0) else "no"
    end = abs(_rel(scene, T))
    if key == "end_kind":
        return "a right angle" if abs(end - 90) < 1e-6 else ("acute (less than 90°)" if end < 90 else
                                                             "obtuse (more than 90°)")
    if key == "end_angle":
        return f"{int(round(end / 15) * 15)}°"
    vals = _unwrapped(scene)
    align = _crossings(vals, 360, 0)
    right = _crossings(vals, 180, 90)
    if key == "first_align":
        return f"{align[0]:.1f} s"
    if key == "n_align":
        return len(align)
    if key == "n_right":
        return len(right)
    if key == "align_gap":
        return f"{round((align[1] - align[0]) * 2) / 2:.1f} s"
    if key == "cf_reverse_blue":
        return len(_crossings(_unwrapped(scene, wb=-b["w"]), 360, 0))
    if key == "cf_still_blue":
        return len(_crossings(_unwrapped(scene, wb=0), 360, 0))
    if key == "cf_double_red":
        return f"{_crossings(_unwrapped(scene, wr=2 * r['w']), 360, 0)[0]:.1f} s"
    return None
