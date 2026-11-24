"""Independent answers for motion_parallax. Depth is never read from the
stored scene: each ball's on-screen drift is measured from its screen
position at two times (what a viewer sees), relative depth is the inverse
of that drift, and real size is the on-screen radius divided by the drift.
Clock times come from finding when the measured track crosses a line."""
from __future__ import annotations
import sim

RATIOS = ["1.5 times", "2 times", "2.5 times", "3 times", "4 times", "1.25 times", "1.33 times", "1.67 times"]


def _track(scene, b, t):
    return sim.screen_x(scene, b, t)


def _drift(scene, b):
    return _track(scene, b, 0.0) - _track(scene, b, 1.0)


def _cross(scene, b, line, v_scale=1.0):
    x0, d = _track(scene, b, 0.0), _drift(scene, b) * v_scale
    lo, hi = 0.0, 1000.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if x0 - d * mid > line:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def answer(key: str, p: dict, scene: dict):
    bs = scene["balls"]
    nm = lambda i: f"the {bs[i]['color']} ball"
    drift = [_drift(scene, b) for b in bs]
    size = [b["r"] / d for b, d in zip(bs, drift)]
    idx = range(len(bs))
    if key == "n_balls":
        return len(bs)
    if key == "fastest":
        return nm(max(idx, key=lambda i: drift[i]))
    if key == "leaves_first":
        return nm(min(idx, key=lambda i: _cross(scene, bs[i], -bs[i]["r"])))
    if key in ("closer",):
        return nm(max((p["x"], p["y"]), key=lambda i: drift[i]))
    if key == "nearest":
        return nm(max(idx, key=lambda i: drift[i]))
    if key == "depth_ratio":
        r = drift[p["y"]] / drift[p["x"]]
        return next(t for t in RATIOS if abs(float(t.split()[0]) - r) < 0.02)
    if key == "center_time":
        return f"{_cross(scene, bs[p['x']], sim.CX):.1f} s"
    if key == "depth_order":
        return " → ".join(bs[i]["color"] for i in sorted(idx, key=lambda i: -drift[i]))
    if key == "true_largest":
        return nm(max(idx, key=lambda i: size[i]))
    if key == "cf_double_speed":
        return f"{_cross(scene, bs[p['x']], sim.CX, 2.0):.1f} s"
    if key == "cf_reverse_first":
        t = [(1280 + bs[i]["r"] - _track(scene, bs[i], 0.0)) / drift[i] for i in idx]
        return nm(min(idx, key=lambda i: t[i]))
    if key == "cf_twice_far":
        return nm(max((p["x"], p["y"]), key=lambda i: drift[i] / (2 if i == p["x"] else 1)))
    return None
