"""10-question ladder for buoyancy_tank (common/ladder5.py, LADDER = "ten").

Every template is enumerated into a deterministic candidate pool per level
(`pools`); only candidates whose answer has a clear visible margin enter
the pool. `build` draws the per-level counts with L5.ten_counts, then
picks from each pool (different templates first), and builds the options.
Counterfactual (L5) candidates are confirmed by re-running the ODE on the
changed scene before they are used.

Objects are named by colour + pattern + shape ("the red striped ball");
colours are unique within a clip. Lines on the back wall are counted up
from the floor (the lowest line above the floor is line 1). Times are the
clock reading on the first frame that shows the event.
"""
from __future__ import annotations
import functools
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
STATES = ["floating, with its top above the water", "resting on the floor, completely under water",
          "resting on the floor, with its top above the water", "floating completely under water, off the floor"]
PHRASE = {"release": "is let go", "enter": "touches the water", "land": "comes to rest on the tank floor",
          "surface": "pokes out above the water surface", "liftoff": "lifts off the tank floor",
          "touchdown": "comes to rest on the tank floor"}
INF = {"release": "be let go", "enter": "touch the water", "land": "come to rest on the tank floor",
       "surface": "poke out above the water surface", "liftoff": "lift off the tank floor",
       "touchdown": "come to rest on the tank floor"}
ORD = {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}
FACTOR = {0.5: "half as heavy", 2: "twice as heavy", 3: "three times as heavy", 4: "four times as heavy"}
FRACS = {0.25: "about a quarter", 0.5: "about half", 0.75: "about three quarters"}
FPS_STEPS = (1, 2, 3, 4, 6, 8, 12, 24)
LINES = "The lines on the back wall are evenly spaced; counting up from the floor, the lowest one is line 1."
FIRST = {"enter": "first touches the water", "surface": "first pokes out above the water surface"}
GAP = 10                                          # frames (> 0.4 s) between events a question compares


def fps_for(window_s: float) -> int:
    """Lowest sampling rate (frames/s) that is sure to put a frame inside the window."""
    for f in FPS_STEPS:
        if 1.0 / f <= window_s + 1e-9:
            return f
    return 24


def state_of(o, y, level) -> int:
    if y > 0:
        return 0
    return 1 if level >= o["h"] else 2


# ---------------------------------------------------------------- facts

@functools.lru_cache(maxsize=64)
def _facts(key: str) -> dict:
    scene = json.loads(key)
    objs = scene["objects"]
    tw = scene["tank"]["w"]
    S, U = scene["style"]["scale"], scene["style"]["grid"]
    nf = scene["n_frames"]
    L0, y0 = sim.initial_state(scene)
    Lf, yf = sim.final_state(scene)
    fr = sim.simulate(scene)["frames"][:nf]
    drafts = [sim.draft(o) for o in objs]
    # what a viewer can know about each density: exact (seen floating at rest) or a lower bound
    exact = [yf[i] > 0 or (scene["scenario"] in ("fill", "drain") and y0[i] > 0) for i in range(len(objs))]
    lb = []
    for i, o in enumerate(objs):
        a = sim.area(o)
        lb.append(max([sim.sub_area(o, Lk - ys[i]) / a for Lk, ys in fr if ys[i] == 0.0] + [0.0]))
    return {"scene": scene, "objs": objs, "n": len(objs), "tw": tw, "S": S, "U": U, "nf": nf,
            "L0": L0, "y0": y0, "Lf": Lf, "yf": yf, "drafts": drafts, "exact": exact, "lb": lb,
            "tops": [y + o["h"] for o, y in zip(objs, yf)],
            "hf": [min(max(Lf - y, 0.0), o["h"]) / o["h"] for o, y in zip(objs, yf)],
            "state": [state_of(o, y, Lf) for o, y in zip(objs, yf)],
            "desc": [sim.describe(o) for o in objs],
            "ev": sim.events(scene), "raw": sim.events(scene, raw=True), "floor": sim.floor_changes(scene),
            "nlines": int((scene["tank"]["h"] - 0.5) // U)}


def facts(scene: dict) -> dict:
    return _facts(json.dumps(scene, sort_keys=True))


def _ok_time(F, f) -> bool:
    return 12 <= f <= F["nf"] - 13


def _line(v: float, U: float):
    """Nearest line index if v is clearly near one (within 0.2 spacing), else None."""
    k = round(v / U)
    return k if k >= 1 and abs(v / U - k) <= 0.2 else None


def _cand(level, key, params, text, gt, dist, rm, fps, cf=None):
    return {"level": level, "key": key, "params": params, "text": text, "gt": gt, "dist": dist, "rm": rm,
            "fps": fps, "cf": cf}


def _line_dist(F, k):
    return L.near_ints(k, 1, max(F["nlines"], k + 2))


# ---------------------------------------------------------------- counterfactual helpers

def _cf_scene(F, kind, p) -> dict:
    """The changed scene for a counterfactual (used for the ODE re-run)."""
    sc = json.loads(json.dumps(F["scene"]))
    objs = sc["objects"]
    if kind == "remove":
        del objs[p["obj"]]
    elif kind == "mass":
        objs[p["obj"]]["rho"] = round(objs[p["obj"]]["rho"] * p["factor"], 6)
    elif kind == "water":
        sc["water0"] = _start_water(F, F["L0"] + p["dk"] * F["U"])
    elif kind == "flow":
        sc["flow"]["off"] += 24 * p["ds"]
    return sc


def _start_water(F, L0):
    sc = F["scene"]
    disp = 0.0
    for o, y, d in zip(F["objs"], F["y0"], F["drafts"]):
        if sc["scenario"] == "drop":
            continue
        if sc["scenario"] == "rise":
            disp += sim.sub_area(o, L0)
        else:
            disp += o["rho"] * sim.area(o) if (d is not None and L0 >= d) else sim.sub_area(o, L0)
    return F["tw"] * L0 - disp


def _cf_eq(F, kind, p):
    """Closed-form settled state of the counterfactual: (objects, level, bottoms) or None if invalid."""
    sc = F["scene"]
    objs = list(F["objs"])
    rhos = [o["rho"] for o in objs]
    water = sim.final_water(sc)
    if kind == "remove":
        del objs[p["obj"]]
        del rhos[p["obj"]]
    elif kind == "mass":
        rhos[p["obj"]] *= p["factor"]
    elif kind == "water":
        L0 = F["L0"] + p["dk"] * F["U"]
        if L0 < 1.5 or L0 > sc["tank"]["h"] - 4:
            return None
        if sc["scenario"] in ("fill", "drain"):
            for i, o in enumerate(F["objs"]):
                if not F["exact"][i] and F["lb"][i] < 0.999 and \
                        sim.sub_area(o, L0) / sim.area(o) > F["lb"][i] - 0.03:
                    return None
        water += _start_water(F, L0) - sc["water0"]
    elif kind == "flow":
        water += sc["flow"]["rate"] * p["ds"]
    if water <= 0:
        return None
    Lc, ys = sim.equilibrium(objs, water, F["tw"], rhos)
    if Lc > sc["tank"]["h"] - 2.5 or Lc < 1.0:
        return None
    if sc["scenario"] == "drop" and any(o["y0"] < Lc + 1.5 for o in objs):
        return None
    # every object whose density the viewer only bounds from below must stay grounded by that bound
    idx = [i for i in range(F["n"]) if not (kind == "remove" and i == p["obj"])]
    for j, i in enumerate(idx):
        if kind == "mass" and i == p["obj"]:
            continue
        if not F["exact"][i] and F["lb"][i] < 0.999:
            o = F["objs"][i]
            if sim.sub_area(o, Lc) / sim.area(o) > F["lb"][i] - 0.03:
                return None
    return objs, rhos, Lc, ys


def _robust_floor_count(F, objs, rhos, Lc, ys):
    S = F["S"]
    n = 0
    for o, r, y in zip(objs, rhos, ys):
        d = sim.draft(o, r)
        if y > 0:
            if y * S < sim.MARGIN_PX:
                return None
        else:
            n += 1
            if d is not None and d - Lc < 0.6:
                return None
            if abs(r - 1.0) < 0.05:
                return None
    return n


def verify_cf(F, kind, p) -> bool:
    """Re-run the ODE on the changed scene; it must settle onto the closed-form state."""
    eq = _cf_eq(F, kind, p)
    if eq is None:
        return False
    _, _, Lc, ys = eq
    sc = _cf_scene(F, kind, p)
    fr = sim.simulate(sc)["frames"]
    Lend, yend = fr[-1]
    if abs(Lend - Lc) > 0.05 or any(abs(a - b) > 0.05 for a, b in zip(yend, ys)):
        return False
    for Lk, yk in fr[-30:]:
        if abs(Lk - Lend) > 0.02 or any(abs(a - b) > 0.02 for a, b in zip(yk, yend)):
            return False
    th = sc["tank"]["h"]
    return all(Lk <= th - 1.5 for Lk, _ in fr)


# ---------------------------------------------------------------- pools

def pools(scene: dict) -> dict:
    F = facts(scene)
    objs, n, S, U, nf = F["objs"], F["n"], F["S"], F["U"], F["nf"]
    D = F["desc"]
    sc = scene["scenario"]
    P = {lv: [] for lv in range(1, 6)}
    colour = [o["colour"] for o in objs]

    # ---- L1 observation
    P[1].append(_cand(1, "n_objects", {}, "How many objects are in the tank scene?", n,
                      L.near_ints(n, 1, 9), False, 1))
    nfl = sum(1 for s in F["state"] if s == 0)
    P[1].append(_cand(1, "n_float_end", {}, "At the end of the clip, how many objects are floating (not touching "
                      "the tank floor)?", nfl, L.near_ints(nfl, 0, n), True, 1))
    P[1].append(_cand(1, "n_floor_end", {}, "At the end of the clip, how many objects are resting on the tank "
                      "floor?", n - nfl, L.near_ints(n - nfl, 0, n), True, 1))
    for i in range(n):
        P[1].append(_cand(1, "end_state", {"obj": i}, f"At the end of the clip, where is {D[i]}?",
                          STATES[F["state"][i]], [s for s in STATES if s != STATES[F["state"][i]]], True, 1))
        sw = sim.SHAPE_WORD[objs[i]["shape"]]
        others = [sim.SHAPE_WORD[k] for k in sim.SHAPES if sim.SHAPE_WORD[k] != sw]
        P[1].append(_cand(1, "shape_of", {"obj": i}, f"What shape is the {colour[i]} object?", sw,
                          others[i % 5:] + others[:i % 5], False, 1))
        pw = {"plain": "plain, no pattern", "striped": "stripes", "spotted": "spots", "checked": "checks"}
        P[1].append(_cand(1, "pattern_of", {"obj": i}, f"What pattern is on the {colour[i]} "
                          f"{sim.SHAPE_WORD[objs[i]['shape']]}?", pw[objs[i]["pattern"]],
                          [v for k, v in pw.items() if k != objs[i]["pattern"]], False, 1))

    # ---- L2 relational
    for et in sim.EVENTS[sc]:
        has = {i: F["ev"][i][et] for i in range(n) if et in F["ev"][i] and _ok_time(F, F["ev"][i][et])}
        if n >= 4 and len(has) >= 2:
            for combo in itertools.combinations(range(n), 4):
                ts = sorted((has[i], i) for i in combo if i in has)
                if len(ts) < 2 or ts[1][0] - ts[0][0] < GAP:
                    continue
                if any(i not in has and et in F["raw"][i] for i in combo):
                    continue                        # an option has the event, but too close to an end
                win = ix = ts[0][1]
                P[2].append(_cand(2, "first_event", {"event": et, "objs": list(combo)},
                                  f"Which of these objects {PHRASE[et]} first?", D[win],
                                  [D[i] for i in combo if i != ix], True, fps_for((ts[1][0] - ts[0][0]) / 24)))
        for a, b in itertools.combinations(sorted(has), 2):
            dt = (has[b] - has[a]) / 24
            if abs(dt) < 0.4 or abs(abs(dt) - 1.0) < 0.3:
                continue
            first, second = (a, b) if dt > 0 else (b, a)
            big = abs(dt) > 1.0
            opts = [f"{D[a]}, by more than 1 second", f"{D[a]}, by less than 1 second",
                    f"{D[b]}, by less than 1 second", f"{D[b]}, by more than 1 second"]
            gt = f"{D[first]}, by {'more' if big else 'less'} than 1 second"
            P[2].append(_cand(2, "pair_gap", {"event": et, "a": a, "b": b},
                              f"Which {PHRASE[et]} first, {D[a]} or {D[b]}, and by how much?", gt,
                              [o for o in opts if o != gt], True,
                              fps_for(min(abs(dt), abs(abs(dt) - 1.0)))))
    for a, b in itertools.combinations(range(n), 2):
        if F["state"][a] == 0 and F["state"][b] == 0 and abs(F["hf"][a] - F["hf"][b]) >= 0.12:
            gt = D[a] if F["hf"][a] > F["hf"][b] else D[b]
            P[2].append(_cand(2, "pair_share", {"a": a, "b": b}, f"At the end, which has the larger share of its "
                              f"own height under water: {D[a]} or {D[b]}?", gt,
                              [D[b] if gt == D[a] else D[a], "both have the same share under water",
                               "neither: both are completely out of the water"], True, 1))
        ta, tb = F["tops"][a], F["tops"][b]
        if abs(ta - tb) * S >= sim.MARGIN_PX and (ta > F["Lf"] or tb > F["Lf"]):
            gt = D[a] if ta > tb else D[b]
            P[2].append(_cand(2, "pair_top", {"a": a, "b": b}, f"At the end, whose top is higher above the tank "
                              f"floor: {D[a]} or {D[b]}?", gt,
                              [D[b] if gt == D[a] else D[a], "their tops are level",
                               "neither: both tops are under the water"], True, 1))

    # ---- L3 measurement
    for i in range(n):
        for et, f in F["ev"][i].items():
            if not _ok_time(F, f):
                continue
            gt = sim.timer_text(f)
            dist = [sim.timer_text(f + 24 * j) for j in (-1, 1, -2, 2, -3, 3) if 0 <= f + 24 * j <= nf - 1]
            if len(dist) >= 3:
                P[3].append(_cand(3, "event_time", {"obj": i, "event": et}, f"What does the clock read when "
                                  f"{D[i]} {FIRST.get(et, PHRASE[et])}?", gt, dist, True, 3))
        if objs[i]["shape"] in sim.RECTS and F["state"][i] == 0:
            for fv, word in FRACS.items():
                if abs(F["hf"][i] - fv) <= 0.04:
                    P[3].append(_cand(3, "share_under", {"obj": i}, f"At the end, how much of the height of {D[i]} "
                                      f"is under water?", word,
                                      [w for w in list(FRACS.values()) + ["all of it"] if w != word], True, 1))
        k = _line(F["tops"][i], U)
        if k is not None:
            P[3].append(_cand(3, "top_line", {"obj": i}, f"{LINES} At the end, which line is the top of "
                              f"{D[i]} closest to?", k, _line_dist(F, k), True, 1))
    for when, Lv in (("start", F["L0"]), ("end", F["Lf"])):
        k = _line(Lv, U)
        if k is not None:
            P[3].append(_cand(3, "water_line", {"when": when}, f"{LINES} When the clip {when}s, which line is "
                              f"the water surface closest to?", k, _line_dist(F, k), when == "end", 1))

    # ---- L4 integration
    if n >= 4:
        rank = {"highest_top": sorted(range(n), key=lambda i: -F["tops"][i]),
                "lowest_top": sorted(range(n), key=lambda i: F["tops"][i]),
                "highest_bottom": sorted(range(n), key=lambda i: -F["yf"][i])}
        val = {"highest_top": F["tops"], "lowest_top": F["tops"], "highest_bottom": F["yf"]}
        txt = {"highest_top": "At the end, which object's top is highest above the tank floor?",
               "lowest_top": "At the end, which object's top is lowest?",
               "highest_bottom": "At the end, which object's bottom is highest above the tank floor?"}
        for what, r in rank.items():
            v = val[what]
            if abs(v[r[0]] - v[r[1]]) * S >= sim.MARGIN_PX:
                P[4].append(_cand(4, "extreme", {"what": what}, txt[what], D[r[0]], [D[i] for i in r[1:4]], True, 1))
    fl = sorted((objs[i]["rho"], i) for i in range(n) if F["state"][i] == 0)
    if len(fl) >= 2 and n >= 4 and fl[1][0] - fl[0][0] >= 0.12:
        hf_sorted = sorted((F["hf"][i], i) for _, i in fl)
        if hf_sorted[0][1] == fl[0][1] and hf_sorted[1][0] - hf_sorted[0][0] >= 0.08:
            win = fl[0][1]
            others = [i for _, i in fl[1:]] + [i for i in range(n) if F["state"][i] != 0]
            P[4].append(_cand(4, "least_dense", {}, "Which object is made of the least dense material?", D[win],
                              [D[i] for i in others[:3]], True, 1))
    fc = F["floor"]
    fr = sim.simulate(scene)["frames"]

    def floor_count(k):
        return sum(1 for y in fr[k][1] if y == 0.0)

    for i in range(n):
        for et in ("release", "enter", "surface"):
            f = F["ev"][i].get(et)
            if f is None or not _ok_time(F, f):
                continue
            gap = min([abs(f - c) for c in fc] + [99])
            if gap < 12:
                continue
            c = floor_count(f)
            P[4].append(_cand(4, "floor_count_at", {"obj": i, "event": et}, f"At the moment {D[i]} {FIRST.get(et, PHRASE[et])}, "
                              f"how many objects are touching the tank floor?", c, L.near_ints(c, 0, n), True,
                              fps_for(gap / 24)))
    if scene.get("flow"):
        f = scene["flow"]["off"]
        gap = min([abs(f - c) for c in fc] + [99])
        if gap >= 12 and _ok_time(F, f):
            c = floor_count(f)
            what = "the tap is turned off" if sc == "fill" else "the drain stops"
            P[4].append(_cand(4, "floor_count_flow", {}, f"At the moment {what}, how many objects are touching the "
                              "tank floor?", c, L.near_ints(c, 0, n), True, fps_for(gap / 24)))
    for et in sim.EVENTS[sc]:
        ts = sorted((F["ev"][i][et], i) for i in range(n) if et in F["ev"][i] and _ok_time(F, F["ev"][i][et]))
        if len(ts) != sum(1 for i in range(n) if et in F["raw"][i]):
            continue                                # some crossing is unclear or too near an end
        for k in range(2, len(ts) + 1):
            t = ts[k - 1][0]
            gaps = [t - ts[k - 2][0]] + ([ts[k][0] - t] if k < len(ts) else [])
            if min(gaps) < GAP:
                continue
            win = ts[k - 1][1]
            others = sorted((i for i in range(n) if i != win),
                            key=lambda i: (0, abs(F["ev"][i][et] - t)) if et in F["ev"][i] else (1, i))
            P[4].append(_cand(4, "kth_event", {"event": et, "k": k}, f"Which object is the {ORD[k]} to {INF[et]}?",
                              D[win], [D[i] for i in others[:3]], True, fps_for(min(gaps) / 24)))
    k = _line(abs(F["Lf"] - F["L0"]), U)
    if k is not None:
        P[4].append(_cand(4, "level_change", {}, "Between the start and the end of the clip, by about how many "
                          "back-wall line spacings does the water surface move?", k, L.near_ints(k, 0, 12), True, 1))

    # ---- L5 counterfactual
    def cf_asks(kind, p, lead):
        eq = _cf_eq(F, kind, p)
        if eq is None:
            return
        o2, r2, Lc, ys = eq
        k = _line(Lc, U)
        if k is not None:
            P[5].append(_cand(5, "cf_" + kind, dict(p, ask="line"), f"{LINES} {lead}, which line would the water "
                              f"surface end closest to?", k, _line_dist(F, k), True, 1, (kind, p)))
        c = _robust_floor_count(F, o2, r2, Lc, ys)
        if c is not None:
            P[5].append(_cand(5, "cf_" + kind, dict(p, ask="floor"), f"{lead}, how many objects would end up "
                              f"resting on the tank floor?", c, L.near_ints(c, 0, len(o2)), True, 1, (kind, p)))

    for i in range(n):
        cf_asks("remove", {"obj": i}, f"If {D[i]} had not been in the tank at all (everything else the same)")
    for dk in (1, -1):
        cf_asks("water", {"dk": dk}, f"If the water had started one back-wall line spacing "
                f"{'higher' if dk > 0 else 'lower'} (everything else the same)")
    if scene.get("flow"):
        for ds in (-1, 1):
            what = "the tap had been turned off" if sc == "fill" else "the drain had been closed"
            cf_asks("flow", {"ds": ds}, f"If {what} one second {'earlier' if ds < 0 else 'later'}")
    for i in range(n):
        if not F["exact"][i]:
            continue
        for fac, word in FACTOR.items():
            r = objs[i]["rho"] * fac
            if abs(r - 1.0) < 0.08:
                continue
            p = {"obj": i, "factor": fac}
            eq = _cf_eq(F, "mass", p)
            if eq is None:
                continue
            o2, r2, Lc, ys = eq
            o, y = objs[i], ys[i]
            d = sim.draft(o, r)
            if y > 0 and (y * S < sim.MARGIN_PX or (y + o["h"] - Lc) * S < 12):
                continue
            if y == 0 and ((d is not None and d - Lc < 0.6) or abs(o["h"] - Lc) * S < 12):
                continue
            st = STATES[state_of(o, y, Lc)]
            P[5].append(_cand(5, "cf_mass", p, f"If {D[i]} were {word} (same size and shape, everything else the "
                              f"same), where would it be at the end?", st, [s for s in STATES if s != st], True, 1,
                              ("mass", p)))
    for lv in P:                                   # exactly four distinct options everywhere
        P[lv] = [c for c in P[lv] if len({str(d) for d in c["dist"]} - {str(c["gt"])}) >= 3]
    return P


def feasible(scene: dict) -> bool:
    P = pools(scene)
    if any(len(P[lv]) < 3 for lv in range(1, 5)):
        return False
    F = facts(scene)
    ok = 0
    for c in P[5]:
        if verify_cf(F, *c["cf"]):
            ok += 1
            if ok == 3:
                return True
    return False


def build(scene: dict, rng: random.Random) -> list[dict]:
    P = pools(scene)
    F = facts(scene)
    counts = L.ten_counts(rng)
    qs = []
    for lv, cnt in zip(range(1, 6), counts):
        pool = list(P[lv])
        rng.shuffle(pool)
        chosen, keys = [], set()
        for second_pass in (False, True):
            for c in pool:
                if len(chosen) == cnt:
                    break
                if c in chosen or (not second_pass and c["key"] in keys):
                    continue
                if c["cf"] is not None and not verify_cf(F, *c["cf"]):
                    continue
                chosen.append(c)
                keys.add(c["key"])
        if len(chosen) < cnt:
            raise RuntimeError(f"level {lv}: only {len(chosen)} usable questions")
        for c in chosen:
            opts, a = L.mc(rng, c["gt"], c["dist"])
            if len(opts) != 4:
                raise RuntimeError(f"{c['key']}: {len(opts)} options")
            qs.append(L.question(lv, c["key"], c["text"], c["gt"], opts, a, c["rm"],
                                 dict(c["params"], min_fps=c["fps"])))
    return L.number(qs)
