"""Independent answer re-derivation for inertia_race.

Nothing here calls sim.py, and nothing reads the generator's per-racer facts
(t0, ramp, shape, k_shape, start, release_f, finish_f, violates). Every answer is
rebuilt from `render.layout` -- the same per-frame geometry `render.frame` draws --
plus the ruler labels, exactly as a viewer with perfect eyesight would:

  * a body's position comes from projecting its contact point onto the ramp axis
    through the "0.00" and "1.00" labels, then snapping it to a 0.05 gridline;
  * its release and crossing frames come from where that position starts and stops
    changing;
  * its turn count comes from unwrapping the drawn rotation angle, so a body that
    slips is measured as it is drawn rather than as rolling would imply;
  * its acceleration is fitted from the observed trajectory, and the
    counterfactuals are re-solved from that fitted value.
"""
from __future__ import annotations

import functools
import json
import math

import render

FPS = 24
GRID = 0.05
EPS = 1e-4
TAU = {"slider": 1.0, "solid_sphere": math.sqrt(1.4), "solid_disc": math.sqrt(1.5),
       "hollow_sphere": math.sqrt(5.0 / 3.0), "hoop": math.sqrt(2.0)}
SHAPE_WORD = {"slider": "sliding block", "solid_sphere": "solid ball",
              "solid_disc": "solid disc", "hollow_sphere": "hollow shell",
              "hoop": "thin ring"}


L6_OPTIONS = ("every body behaves consistently",
              "at least one body breaks the rules")


def _frac(x: float) -> str:
    return f"{x:.2f}"


def _one(x: float) -> str:
    return f"{x:.1f}"


def _snap(x: float) -> str:
    """Quantise a measured position to the ruler's 0.05 gridlines."""
    v = x / GRID
    k = round(v)
    if abs(v - k) > 0.22:
        raise ValueError(f"position {x:.4f} is not on a gridline")
    return _frac(round(k * GRID, 2))


@functools.lru_cache(maxsize=4)
def _track(key: str) -> dict:
    """Replay render.layout over every frame and read the clip back off it."""
    scene = json.loads(key)
    n = scene["n_frames"]
    ids = list(render.layout(scene, 0)["racers"])
    pos = {i: [] for i in ids}
    turns = {i: [] for i in ids}
    body = {}
    rpx = {}
    spin = {i: 0.0 for i in ids}
    last = {i: None for i in ids}
    for f in range(n):
        lay = render.layout(scene, f)
        for i in ids:
            st = lay["racers"][i]
            body[i] = st["body"]
            rpx[i] = st["r_px"]
            pos[i].append(render.to_fraction(scene, (st["contact_x"], st["contact_y"])))
            phi = st["phi"]
            if last[i] is not None:
                d = phi - last[i]
                if d < -180.0:            # the drawn angle wrapped past 360
                    d += 360.0
                spin[i] += max(0.0, d)
            last[i] = phi
            turns[i].append(spin[i] / 360.0)

    # the ramp's on-screen length is the gap between the 0.00 and 1.00 labels, so a
    # drawn radius converts to ruler units without reading the generator's view
    lab = {lb["text"]: (lb["x"], lb["y"]) for lb in scene["labels"]}
    run = math.dist(lab["0.00"], lab["1.00"])
    out = {"ids": ids, "pos": pos, "turns": turns, "body": body, "r_px": rpx,
           "r_frac": {i: rpx[i] / run for i in ids},
           "release": {}, "finish": {}, "start": {}, "accel": {}}
    for i in ids:
        s = pos[i]
        out["start"][i] = s[0]
        rel = next((f - 1 for f in range(1, n) if s[f] > s[0] + 1e-7), None)
        out["release"][i] = 0 if rel is None else rel
        # A finished body is drawn at the line on every later frame, so its measured
        # position stops changing exactly then. Testing for "stopped" rather than for
        # "near 1.0" avoids calling the finish a frame early when the last step in is
        # smaller than the threshold.
        out["finish"][i] = (next(f for f in range(n) if abs(s[f] - s[-1]) < 1e-9)
                            if s[-1] >= 1.0 - 1e-3 else None)
        # fit s(t) = s0 + a t^2 / 2 on the last frame before the body stops moving
        end = (out["finish"][i] - 1) if out["finish"][i] is not None else (n - 1)
        t = (end - out["release"][i]) / FPS
        out["accel"][i] = 2.0 * (s[end] - s[0]) / (t * t) if t > 0 else 0.0
    return out


def _tr(scene):
    return _track(json.dumps(scene, sort_keys=True))


LEAD_DEADBAND = 0.02             # a new leader counts only once clearly ahead
LEAD_HOLD = 3                    # frames it must hold that lead before it counts


def _cross(t, a, b):
    """Frames at which a and b swap places, measured off the drawn positions.

    The deadband stops two bodies running abreast from swapping order on noise,
    and the hold stops a single frame sitting exactly on the deadband from
    deciding the answer.
    """
    out, cur, run, sign = [], None, 0, None
    for f in range(len(t["pos"][a])):
        pa, pb = t["pos"][a][f], t["pos"][b][f]
        d = pa - pb
        s = 0 if min(pa, pb) >= 1.0 - EPS or abs(d) < LEAD_DEADBAND else (1 if d > 0 else -1)
        if s == 0:
            run, sign = 0, None
            continue
        run = run + 1 if s == sign else 1
        sign = s
        if run >= LEAD_HOLD and s != cur:
            if cur is not None:
                out.append(f)
            cur = s
    return out


def _time_to_line(t, i, *, start=None, mul=1.0, tau_to=None) -> float:
    """Seconds from release to the line, under an altered world."""
    a = t["accel"][i]
    s0 = t["start"][i] if start is None else start
    if tau_to is not None:
        a *= (TAU[t["body"][i]] / TAU[tau_to]) ** 2
    return math.sqrt(2.0 * max(0.0, 1.0 - s0) / (a * mul))


def _nm(scene, rid) -> str:
    return f"the {rid.split('_')[0]} body"


def answer(key: str, p: dict, scene: dict):
    t = _tr(scene)
    ids, pos, turns = t["ids"], t["pos"], t["turns"]
    tenths = lambda f: f"{(f * 10 // FPS) / 10:.1f}"

    # ---------------- L1
    if key == "shape_of":
        return SHAPE_WORD[t["body"][p["r"]]]
    if key == "n_racers":
        return len(ids)
    if key == "not_finished":
        miss = [i for i in ids if t["finish"][i] is None]
        return _nm(scene, miss[0]) if len(miss) == 1 else None
    if key == "widest":
        o = sorted(ids, key=lambda i: -t["r_px"][i])
        return _nm(scene, o[0]) if t["r_px"][o[0]] > t["r_px"][o[1]] else None
    if key == "starts_lowest":
        o = sorted(ids, key=lambda i: -t["start"][i])
        return _nm(scene, o[0]) if t["start"][o[0]] > t["start"][o[1]] else None

    # ---------------- L2
    if key == "first_of_pair":
        a, b = p["a"], p["b"]
        fa, fb = t["finish"][a], t["finish"][b]
        if fa is None or fb is None or fa == fb:
            return None
        return _nm(scene, a if fa < fb else b)
    if key == "same_finish":
        a, b = p["a"], p["b"]
        return "yes" if t["finish"][a] is not None and t["finish"][a] == t["finish"][b] else "no"
    if key == "ahead_at_t":
        a, b, f = p["a"], p["b"], p["f"]
        return _nm(scene, a if pos[a][f] > pos[b][f] else b)
    if key == "more_turns_by_t":
        a, b, f = p["a"], p["b"], p["f"]
        return _nm(scene, a if turns[a][f] > turns[b][f] else b)
    if key == "closing_at_t":
        a, b, f = p["a"], p["b"], p["f"]
        d0 = abs(pos[a][f] - pos[b][f])
        d1 = abs(pos[a][f + 1] - pos[b][f + 1])
        return "opening up" if d1 > d0 else "closing"
    if key == "passes_before_t":
        a, b, f = p["a"], p["b"], p["f"]
        return "yes" if any(g <= f for g in _cross(t, a, b)) else "no"

    # ---------------- L3
    if key == "pos_at_t":
        return _snap(pos[p["r"]][p["f"]])
    if key == "finish_time":
        f = t["finish"][p["r"]]
        return None if f is None else tenths(f)
    if key == "turns_by_t":
        return int(round(turns[p["r"]][p["f"]]))
    if key == "release_time":
        return tenths(t["release"][p["r"]])
    if key == "gap_at_t":
        return _snap(abs(pos[p["a"]][p["f"]] - pos[p["b"]][p["f"]]))

    # ---------------- L4
    if key in ("finish_order", "place_of"):
        done = sorted((t["finish"][i], i) for i in ids if t["finish"][i] is not None)
        if len(done) < 2 or any(b[0] - a[0] < 8 for a, b in zip(done, done[1:])):
            return None                      # a dead heat cannot be ranked
        seq = [i for _, i in done]
        if key == "finish_order":
            return " then ".join(i.split("_")[0] for i in seq)
        return seq.index(p["r"]) + 1 if p["r"] in seq else None
    if key == "win_margin":
        fs = sorted(t["finish"][i] for i in ids if t["finish"][i] is not None)
        return _one(round((fs[-1] - fs[0]) / FPS, 1))
    if key == "leader_at_t":
        f = p["f"]
        return _nm(scene, max(ids, key=lambda i: pos[i][f]))
    if key == "crossed_by_t":
        f = p["f"]
        return sum(1 for i in ids if t["finish"][i] is not None and t["finish"][i] <= f)

    # ---------------- L5, re-solved from the fitted acceleration
    if key == "cf_all_together":
        return _nm(scene, max(ids, key=lambda i: t["accel"][i]))
    if key in ("cf_gravity_time", "cf_steeper_time", "cf_from_top", "cf_as_shape"):
        i = p["r"]
        rel = t["release"][i] / FPS
        if key == "cf_as_shape":
            v = _time_to_line(t, i, tau_to=p["shape"])
        elif key == "cf_from_top":
            v = _time_to_line(t, i, start=0.0)
        else:
            v = _time_to_line(t, i, mul=p["x"])
        return _one(round(v + rel, 1))

    # ---------------- the separate universal diagnostic
    if key == "l6_consistency":
        return L6_OPTIONS[0] if _consistency(t) == "yes" else L6_OPTIONS[1]
    return None


def _consistency(t) -> str:
    """Does every body move the way its drawn shape and its own spin say it should?

    Two independent checks, both read off the drawing: the fitted acceleration
    must match the shape's inertia coefficient relative to the rest of the field,
    and the turns a rolling body makes must match the ground it covered.
    """
    ids = t["ids"]
    # spin against distance travelled: a rolling body turns once per circumference
    for i in ids:
        if t["body"][i] == "slider":
            continue
        f = t["finish"][i] or (len(t["pos"][i]) - 1)
        covered = t["pos"][i][f] - t["start"][i]
        expect = covered / (2.0 * math.pi * t["r_frac"][i])
        if expect > 0.5 and abs(t["turns"][i][f] - expect) / expect > 0.12:
            return "no"
    # acceleration against shape: a / a' must be tau'^2 / tau^2 for every pair
    for a in ids:
        for b in ids:
            if a >= b or min(t["accel"][a], t["accel"][b]) <= 0:
                continue
            got = t["accel"][a] / t["accel"][b]
            want = (TAU[t["body"][b]] / TAU[t["body"][a]]) ** 2
            if abs(got - want) / want > 0.06:
                return "no"
    return "yes"