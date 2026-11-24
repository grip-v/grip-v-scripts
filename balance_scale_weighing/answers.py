"""Independent answers for balance_scale_weighing.

Nothing here calls sim.py. Every answer is read off the renderer's
per-frame geometry (render.layout, the same data render.frame draws):

  * which weights sit on the scale in each frame, and on which side
    (contact point left or right of the pivot);
  * torque from the drawn geometry: mass x horizontal distance from the
    pivot (the pan post for a pan, the contact point on the lever);
  * the drawn beam angle; a "rest" is a run of >= 3 frames in which the
    angle does not change, and its position comes from the torque sign
    (cross-checked against the sign of the drawn angle);
  * event moments are the frames where a weight starts or stops resting,
    and times are the timer string drawn on that frame.

Counterfactuals edit the scene (remove a weight, flip its side, change a
mass, shift a peg, skip an event) and re-run the renderer's geometry.
"""
from __future__ import annotations
import copy
import functools
import json
import math

import render

DIR_TXT = {"left": "the left side is lower", "right": "the right side is lower", "level": "the beam is level"}
PAT_TXT = {"plain": "a single plain colour", "striped": "diagonal stripes", "ringed": "an inner outline ring",
           "two-tone": "a lighter lower half"}
ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
EPS = 1e-6


class Mismatch(Exception):
    pass


def _name(o):
    return f"the {o['colour']} {o['shape']}"


@functools.lru_cache(maxsize=32)
def _track_cached(key: str, full: bool):
    scene = json.loads(key)
    n = scene["n_frames"]
    frames = range(n) if full else [n - 1]
    return [render.layout(scene, f) for f in frames]


def _track(scene, full=True):
    return _track_cached(json.dumps(scene, sort_keys=True), full)


def _resting(lay):
    """{obj: side} for weights sitting still on the scale in this frame."""
    px = lay["pivot"][0]
    return {o: ("left" if st["x"] < px else "right") for o, st in lay["objs"].items() if st["state"] == "rest"}


def _along(lay, x, y):
    """Signed distance from the pivot measured along the drawn beam."""
    px, py = lay["pivot"]
    c, s = math.cos(math.radians(lay["angle"])), math.sin(math.radians(lay["angle"]))
    return (x - px) * c + (y - py) * s


def _arm(lay, oid):
    st = lay["objs"][oid]
    px = lay["pivot"][0]
    if lay["lever"]:
        return _along(lay, st["x"], st["y"])
    side = "left" if st["x"] < px else "right"
    return lay["pans"][side]["x"] - px


def _torque(scene, lay, masses=None):
    ms = masses or {o["id"]: o["mass"] for o in scene["objects"]}
    return sum(ms[o] * _arm(lay, o) for o in _resting(lay))


def _dir(tq):
    return "level" if abs(tq) < EPS else ("right" if tq > 0 else "left")


def _runs(scene, tr):
    """Rest runs: [(start, end_exclusive, dir)] where the drawn angle is constant for >= 3 frames."""
    out, start = [], 0
    n = len(tr)
    for f in range(1, n + 1):
        if f == n or tr[f]["angle"] != tr[f - 1]["angle"]:
            if f - start >= 3:
                mid = (start + f - 1) // 2
                d = _dir(_torque(scene, tr[mid]))
                a = tr[mid]["angle"]
                seen = "level" if abs(a) < 1e-9 else ("right" if a > 0 else "left")
                if seen != d:
                    raise Mismatch(f"drawn angle {a:.2f} disagrees with torque at frame {mid}")
                out.append((start, f, d))
            start = f
    return out


def _merged(runs):
    out = []
    for _, _, d in runs:
        if not out or out[-1] != d:
            out.append(d)
    return out


def _events(tr):
    """[(frame, obj, act, kind)] from rest/non-rest transitions; kind is drop, lift or hop."""
    evs = []
    objs = list(tr[0]["objs"])
    for o in objs:
        prev = tr[0]["objs"][o]["state"]
        left_at = None
        for f in range(1, len(tr)):
            s = tr[f]["objs"][o]["state"]
            if s == "rest" and prev != "rest":
                evs.append([f, o, "arrive", "hop" if left_at is not None else "drop"])
                left_at = None
            elif prev == "rest" and s != "rest":
                left_at = f
                evs.append([f, o, "leave", None])
            prev = s
        # a leave that never came back to rest is a lift
    for e in evs:
        if e[2] == "leave":
            back = [x for x in evs if x[1] == e[1] and x[2] == "arrive" and x[0] > e[0]]
            e[3] = "hop" if back else "lift"
    return sorted((tuple(e) for e in evs), key=lambda e: (e[0], e[1]))


def _find(evs, p):
    k = 0
    for e in evs:
        if e[1] == p["obj"] and e[2] == p["act"] and (e[2] == "arrive" or e[3] == "lift"):
            k += 1
            if k == p["occ"]:
                return e
    raise Mismatch(f"event {p} not seen")


def _tenths(text):
    a, b = text.split()[0].split(".")
    return int(a) * 10 + int(b)


def _fmt(v):
    return f"{v // 10}.{v % 10} s"


def _final_dir(scene, masses=None):
    lay = _track(scene, full=False)[-1]
    return _dir(_torque(scene, lay, masses))


def _without(scene, oid):
    s = copy.deepcopy(scene)
    s["initial"] = [it for it in s["initial"] if it["obj"] != oid]
    s["events"] = [e for e in s["events"] if e["obj"] != oid]
    return s


def _flip(scene, oid):
    s = copy.deepcopy(scene)
    for it in s["initial"] + s["events"]:
        if it["obj"] == oid and it.get("kind", "drop") in ("drop", "place"):
            it["side"] = "left" if it["side"] == "right" else "right"
    return s


def _side_mass(scene, lay, side):
    ms = {o["id"]: o["mass"] for o in scene["objects"]}
    return sum(ms[o] for o, s in _resting(lay).items() if s == side)


def answer(key: str, p: dict, scene: dict):
    try:
        return _answer(key, p, scene)
    except Mismatch:
        return None


def _answer(key, p, scene):
    objs = {o["id"]: o for o in scene["objects"]}
    tr = _track(scene)
    last = tr[-1]
    evs = _events(tr)
    if key == "final_count":
        return len(_resting(last))
    if key == "side_count":
        return sum(1 for s in _resting(last).values() if s == p["side"])
    if key == "end_dir":
        return DIR_TXT[_runs(scene, tr)[-1][2]]
    if key == "dots":
        return objs[p["obj"]]["mass"]
    if key == "shape":
        return objs[p["obj"]]["shape"]
    if key == "pattern":
        return PAT_TXT[objs[p["obj"]]["pattern"]]
    if key == "peg":
        st, side = last["objs"][p["obj"]], _resting(last)[p["obj"]]
        pegs = last["pegs"][side]
        k = min(range(4), key=lambda i: (pegs[i][0] - st["x"]) ** 2 + (pegs[i][1] - st["y"]) ** 2)
        return f"the {ORD[k + 1]} peg"
    arrivals = [e for e in evs if e[2] == "arrive" and e[3] == "drop"]
    if key == "first_arrival":
        return _name(objs[arrivals[0][1]])
    if key in ("first_of", "last_of"):
        fr = {e[1]: e[0] for e in arrivals}
        order = sorted(p["objs"], key=lambda o: fr[o])
        return _name(objs[order[0] if key == "first_of" else order[-1]])
    runs = _runs(scene, tr)
    if key == "dir_after":
        fe = _find(evs, p)[0]
        return DIR_TXT[next(r for r in runs if r[1] - 1 >= fe + 2)[2]]
    if key == "heaviest_of":
        return _name(objs[max(p["objs"], key=lambda o: objs[o]["mass"])])
    if key == "where_end":
        side = _resting(last).get(p["obj"])
        return {"left": "on the left pan", "right": "on the right pan", None: "lifted off the scale"}[side]
    if key == "event_time":
        return render.timer_text(_find(evs, p)[0])
    if key == "gap":
        fa, fb = _find(evs, p["a"])[0], _find(evs, p["b"])[0]
        return _fmt(_tenths(render.timer_text(fb)) - _tenths(render.timer_text(fa)))
    if key == "side_mass_end":
        return _side_mass(scene, last, p["side"])
    if key == "landings":
        return sum(1 for e in arrivals if _resting(tr[e[0]])[e[1]] == p["side"])
    if key == "lifts":
        return sum(1 for e in evs if e[2] == "leave" and e[3] == "lift")
    if key == "hops":
        return sum(1 for e in evs if e[2] == "arrive" and e[3] == "hop")
    if key == "hops_to":
        return sum(1 for e in evs if e[2] == "arrive" and e[3] == "hop" and _resting(tr[e[0]])[e[1]] == p["side"])
    if key == "max_side_count":
        return max(sum(1 for s in _resting(lay).values() if s == p["side"]) for lay in tr)
    if key == "changes":
        return len(_merged(runs)) - 1
    if key == "level_count":
        return sum(1 for d in _merged(runs)[1:] if d == "level")
    if key == "longest_dir":
        tot = {"left": 0, "right": 0, "level": 0}
        for a, b, d in runs:
            tot[d] += b - a
        return DIR_TXT[max(tot, key=lambda d: tot[d])]
    if key == "first_turn":
        k = next(i for i, r in enumerate(runs) if r[2] == p["dir"])
        cause = [e for e in evs if e[0] < runs[k][0] and not (e[2] == "leave" and e[3] == "hop")]
        return _name(objs[cause[-1][1]])
    if key == "mystery":
        mid = next(o for o in objs.values() if o["mystery"])["id"]
        for a, b, d in runs:
            lay = tr[(a + b - 1) // 2]
            if d == "level" and mid in _resting(lay):
                others = sum(objs[o]["mass"] * _arm(lay, o) for o in _resting(lay) if o != mid)
                m = -others / _arm(lay, mid)
                if abs(m - round(m)) > 1e-6:
                    raise Mismatch("non-integer mass")
                return int(round(m))
        raise Mismatch("no level rest with the dotless weight")
    if key == "max_side_mass":
        return max(_side_mass(scene, lay, p["side"]) for lay in tr)
    if key == "mass_after":
        return _side_mass(scene, tr[_find(evs, p)[0]], p["side"])
    if key == "torque_end":
        res = _resting(last)
        tot = 0
        for o, s in res.items():
            if s != p["side"]:
                continue
            pegs = last["pegs"][s]
            st = last["objs"][o]
            k = min(range(4), key=lambda i: abs(pegs[i][0] - st["x"]))
            tot += objs[o]["mass"] * (k + 1)
        return tot
    # ---- counterfactuals: edit the scene, re-run the geometry
    if key == "cf_remove":
        return DIR_TXT[_final_dir(_without(scene, p["obj"]))]
    if key == "cf_other_side":
        return DIR_TXT[_final_dir(_flip(scene, p["obj"]))]
    if key == "cf_heavier":
        ms = {o["id"]: o["mass"] for o in scene["objects"]}
        ms[p["obj"]] += p["delta"]
        return DIR_TXT[_final_dir(scene, ms)]
    if key == "cf_extra":
        tq = _torque(scene, last)
        px = last["pivot"][0]
        armx = (_along(last, *last["pegs"][p["side"]][p["peg"] - 1]) if last["lever"]
                else last["pans"][p["side"]]["x"] - px)
        return DIR_TXT[_dir(tq + p["mass"] * armx)]
    if key == "cf_changes_remove":
        alt = _without(scene, p["obj"])
        return len(_merged(_runs(alt, _track(alt)))) - 1
    if key in ("cf_skip_hop", "cf_no_lift"):
        kind = "hop" if key == "cf_skip_hop" else "lift"
        s = copy.deepcopy(scene)
        s["events"] = [e for e in s["events"] if not (e["obj"] == p["obj"] and e["kind"] == kind)]
        return DIR_TXT[_final_dir(s)]
    if key == "cf_shift_peg":
        s = copy.deepcopy(scene)
        for it in s["initial"] + s["events"]:
            if it["obj"] == p["obj"] and it["peg"]:
                it["peg"] += p["delta"]
        return DIR_TXT[_final_dir(s)]
    return None
