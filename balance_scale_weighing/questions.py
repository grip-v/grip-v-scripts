"""10-question ladder for balance_scale_weighing (common/ladder5.py, LADDER = "ten").

Per clip: 1-3 / 2-3 / 2-3 / 1-3 / 1-3 questions on levels 1-5 (L5.ten_counts),
drawn from several templates per level. Within a clip no template repeats
with the same params; distinct templates are used before any repeats.

Weights are named by colour and shape ("the red hexagon"); colours are
unique within a clip. Mass = number of dots. Times are read off the
on-screen timer. Every answer here comes from sim.timeline; answers.py
re-derives it from the renderer's per-frame geometry instead.
"""
from __future__ import annotations
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
DIR_TXT = {"left": "the left side is lower", "right": "the right side is lower", "level": "the beam is level"}
NEVER = "it keeps rocking and never comes to rest"     # 4th option of every tilt question; never correct
PAT_TXT = {"plain": "a single plain colour", "striped": "diagonal stripes", "ringed": "an inner outline ring",
           "two-tone": "a lighter lower half"}
ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
GAP_STEP = 12                     # tenths of a second between gap-question options
TIME_SEP = 8                      # tenths: event-time options are at least 0.8 s from the answer


def name(o) -> str:
    return f"the {o['colour']} {o['shape']}"


def side_txt(scene, side) -> str:
    return f"{side} half of the beam" if scene["scenario"] == "lever" else f"{side} pan"


def mf(frames: float) -> float:
    """min_fps for a window of `frames` frames: one sample must fall inside it."""
    return math.ceil(100 * sim.FPS / max(1, frames)) / 100


class Ctx:
    def __init__(self, scene):
        self.scene = scene
        self.objs = scene["objects"]
        self.states = sim.timeline(scene)
        self.rest = sim.rest_states(self.states)
        self.final = self.states[-1]
        self.min_win = min(s["rest"][1] - s["rest"][0] for s in self.rest)
        self.final_win = self.final["rest"][1] - self.final["rest"][0]
        # events as (obj, act, occ) refs with their question frame
        self.refs = []
        occ, hops = {}, {}
        for i, ev in enumerate(scene["events"]):
            act = "leave" if ev["kind"] == "lift" else "arrive"
            k = (ev["obj"], act)
            occ[k] = occ.get(k, 0) + 1
            if ev["kind"] == "hop":
                hops[ev["obj"]] = hops.get(ev["obj"], 0) + 1
            self.refs.append({"i": i, "obj": ev["obj"], "act": act, "occ": occ[k], "kind": ev["kind"],
                              "hop": hops.get(ev["obj"], 0), "f": sim.moment(ev)})
        self.n_occ = occ
        self.n_hops = hops          # hops are numbered among hops only, not among all arrivals

    def phrase(self, r, base=False) -> str:
        """'the red square lands on the scale' (base=True: '... land on the scale', for 'does ...?')."""
        o = self.objs[r["obj"]]
        if r["kind"] == "lift":
            return f"{name(o)} {'get' if base else 'is'} lifted off the scale"
        if r["kind"] == "hop":
            extra = ""
            if self.n_hops[r["obj"]] > 1:
                extra = " for the first time" if r["hop"] == 1 else " for the second time"
            return f"{name(o)} {'land' if base else 'lands'} on the other pan{extra}"
        return f"{name(o)} {'land' if base else 'lands'} on the scale"

    def state_after(self, r):
        return next(s for s in self.states if s["event"] == r["i"] and not s["leaving"])

    def present_span(self, oid) -> int:
        """Longest stretch (frames) the weight sits still on the scale."""
        best, start = 0, 0 if any(it["obj"] == oid for it in self.scene["initial"]) else None
        for r in self.refs:
            if r["obj"] != oid:
                continue
            if r["act"] == "arrive":
                start = r["f"]
            if r["kind"] in ("lift", "hop"):
                ev = self.scene["events"][r["i"]]
                if start is not None:
                    best = max(best, ev["f0"] - start)
                start = ev["f1"] if r["kind"] == "hop" else None
        if start is not None:
            best = max(best, self.scene["n_frames"] - start)
        return best

    def ref_params(self, r):
        return {"obj": r["obj"], "act": r["act"], "occ": r["occ"]}


def _dir_q(rng, lv, key, text, gt, rm, params, fps):
    opts, a = L.mc(rng, DIR_TXT[gt], [DIR_TXT[d] for d in ("left", "right", "level") if d != gt] + [NEVER])
    return L.question(lv, key, text, DIR_TXT[gt], opts, a, rm, dict(params, min_fps=fps))


def _int_q(rng, lv, key, text, gt, rm, params, fps, lo=0, extra=()):
    opts, a = L.mc(rng, gt, list(extra) + L.near_ints(gt, lo))
    return L.question(lv, key, text, gt, opts, a, rm, dict(params, min_fps=fps))


def _name_q(rng, lv, key, text, gt_obj, others, rm, params, fps, c, pad=False):
    gt = name(c.objs[gt_obj])
    pool = [name(c.objs[o]) for o in others if o != gt_obj]
    rng.shuffle(pool)
    if pad:                                   # top up with other weights in the clip
        rest = [name(x) for x in c.objs if x["id"] != gt_obj and x["id"] not in others]
        rng.shuffle(rest)
        pool += rest
    opts, a = L.mc(rng, gt, pool)
    return L.question(lv, key, text, gt, opts, a, rm, dict(params, min_fps=fps))


# --------------------------------------------------------------------------- candidates


def candidates(c: Ctx) -> dict:
    sc = c.scene
    scen = sc["scenario"]
    lever = scen == "lever"
    ids = [o["id"] for o in c.objs]
    dotted = [o["id"] for o in c.objs if not o["mystery"]]
    drops = [r for r in c.refs if r["kind"] in ("drop", "place")]
    cand = {1: [], 2: [], 3: [], 4: [], 5: []}
    cand[1] += [("final_count", {}), ("end_dir", {})] + [("side_count", {"side": s}) for s in sim.SIDES]
    cand[1] += [("dots", {"obj": o}) for o in dotted] + [("shape", {"obj": o}) for o in ids]
    cand[1] += [("pattern", {"obj": o}) for o in ids]
    if lever:
        cand[1] += [("peg", {"obj": o}) for o in ids]
    if len(drops) >= 3:
        cand[1].append(("first_arrival", {}))
    if len(drops) >= 4:
        for k in range(3):
            pick = sorted(rng_sample(drops, min(4, len(drops)), k))
            cand[2].append(("first_of", {"objs": pick}))
            cand[2].append(("last_of", {"objs": pick}))
    cand[2] += [("dir_after", c.ref_params(r)) for r in c.refs]
    for k in range(3):
        pick = sorted(rng_sample([{"obj": o} for o in dotted], min(4, len(dotted)), k + 7))
        if len(pick) == 4:
            cand[2].append(("heaviest_of", {"objs": pick}))
    if scen in ("add_remove", "transfer"):
        movers = {r["obj"] for r in c.refs if r["kind"] in ("lift", "hop")}
        cand[2] += [("where_end", {"obj": o}) for o in ids if o in movers or scen == "add_remove"]
    cand[3] += [("event_time", c.ref_params(r)) for r in c.refs]
    cand[3] += [("gap", {"a": c.ref_params(r1), "b": c.ref_params(r2)})
                for k, r1 in enumerate(c.refs) for r2 in c.refs[k + 1:k + 3]]
    cand[3] += [("side_mass_end", {"side": s}) for s in sim.SIDES]
    if scen in ("drop", "mystery", "lever", "add_remove"):
        cand[3] += [("landings", {"side": s}) for s in sim.SIDES]
    if scen == "add_remove":
        cand[3].append(("lifts", {}))
    if scen == "transfer":
        cand[3] += [("hops", {})] + [("hops_to", {"side": s}) for s in sim.SIDES]
    if scen in ("add_remove", "transfer"):
        cand[3] += [("max_side_count", {"side": s}) for s in sim.SIDES]
    cand[4] += [("changes", {}), ("level_count", {}), ("longest_dir", {})]
    cand[4] += [("first_turn", {"dir": d}) for d in ("left", "right")]
    if scen == "mystery":
        cand[4] += [("mystery", {})] * 1
    if scen in ("add_remove", "transfer"):
        cand[4] += [("max_side_mass", {"side": s}) for s in sim.SIDES]
    cand[4] += [("mass_after", dict(c.ref_params(r), side=s)) for r in c.refs for s in sim.SIDES]
    if lever:
        cand[4] += [("torque_end", {"side": s}) for s in sim.SIDES]
    cand[5] += [("cf_remove", {"obj": o}) for o in ids]
    if scen != "transfer":
        cand[5] += [("cf_other_side", {"obj": o}) for o in ids]
    cand[5] += [("cf_heavier", {"obj": o, "delta": d}) for o in ids for d in (1, 2)]
    for s in sim.SIDES:
        for m in (1, 2, 3, 4):
            if lever:
                cand[5] += [("cf_extra", {"side": s, "mass": m, "peg": p}) for p in range(1, 5)]
            else:
                cand[5].append(("cf_extra", {"side": s, "mass": m}))
    if scen in ("drop", "mystery", "lever", "add_remove"):
        cand[5] += [("cf_changes_remove", {"obj": o}) for o in ids]
    if scen == "transfer":
        cand[5] += [("cf_skip_hop", {"obj": o}) for o in ids]
    if scen == "add_remove":
        cand[5] += [("cf_no_lift", {"obj": r["obj"]}) for r in c.refs if r["kind"] == "lift"]
    if lever:
        cand[5] += [("cf_shift_peg", {"obj": o, "delta": d}) for o in ids for d in (1, -1)]
    return cand


def rng_sample(items, k, salt):
    """Deterministic, rng-free subset choice used only to enumerate candidate groups."""
    ids = [it["obj"] for it in items]
    if len(ids) <= k:
        return ids
    step = salt % len(ids)
    rot = ids[step:] + ids[:step]
    return rot[:k]


# --------------------------------------------------------------------------- builders


def build_one(c: Ctx, rng, lv, key, p):
    sc, objs = c.scene, c.objs
    scen = sc["scenario"]
    fin = c.final
    if key == "final_count":
        return _int_q(rng, lv, key, "How many weights are on the scale when the video ends?", len(fin["on"]),
                      False, p, mf(c.final_win))
    if key == "end_dir":
        return _dir_q(rng, lv, key, "When the video ends, which side of the beam is lower?", fin["dir"], False, p,
                      mf(c.final_win))
    if key == "side_count":
        n = sum(1 for s, _ in fin["on"].values() if s == p["side"])
        return _int_q(rng, lv, key, f"How many weights are on the {side_txt(sc, p['side'])} when the video ends?",
                      n, False, p, mf(c.final_win))
    if key == "dots":
        o = objs[p["obj"]]
        return _int_q(rng, lv, key, f"How many dots are on {name(o)}?", o["mass"], False, p,
                      mf(c.present_span(o["id"])), lo=1)
    if key == "shape":
        o = objs[p["obj"]]
        others = [s for s in sim.SHAPES if s != o["shape"]]
        rng.shuffle(others)
        opts, a = L.mc(rng, o["shape"], others)
        return L.question(lv, key, f"What shape is the {o['colour']} weight?", o["shape"], opts, a, False,
                          dict(p, min_fps=mf(c.present_span(o["id"]))))
    if key == "pattern":
        o = objs[p["obj"]]
        gt = PAT_TXT[o["pattern"]]
        opts, a = L.mc(rng, gt, [v for k, v in PAT_TXT.items() if k != o["pattern"]])
        return L.question(lv, key, f"What is the surface pattern of {name(o)}?", gt, opts, a, False,
                          dict(p, min_fps=mf(c.present_span(o["id"]))))
    if key == "peg":
        o = objs[p["obj"]]
        side, peg = fin["on"][o["id"]]
        gt = f"the {ORD[peg]} peg"
        opts, a = L.mc(rng, gt, [f"the {ORD[k]} peg" for k in range(1, 5) if k != peg])
        return L.question(lv, key, f"Counting the pegs outward from the pivot, which peg is {name(o)} sitting on?",
                          gt, opts, a, False, dict(p, min_fps=mf(c.present_span(o["id"]))))
    if key == "first_arrival":
        drops = [r for r in c.refs if r["kind"] in ("drop", "place")]
        return _name_q(rng, lv, key, "Which weight is the first to land on the scale?", drops[0]["obj"],
                       [r["obj"] for r in drops[1:]], True, p, mf(drops[1]["f"] - drops[0]["f"]), c, pad=True)
    if key in ("first_of", "last_of"):
        fr = {r["obj"]: r["f"] for r in c.refs if r["kind"] in ("drop", "place")}
        order = sorted(p["objs"], key=lambda o: fr[o])
        gt = order[0] if key == "first_of" else order[-1]
        nb = order[1] if key == "first_of" else order[-2]
        word = "first" if key == "first_of" else "last"
        names = ", ".join(name(objs[o]) for o in p["objs"])
        return _name_q(rng, lv, key, f"Of these weights ({names}), which one lands on the scale {word}?", gt,
                       p["objs"], True, p, mf(abs(fr[gt] - fr[nb])), c)
    if key == "dir_after":
        r = _find(c, p)
        st = c.state_after(r)
        return _dir_q(rng, lv, key, f"Right after {c.phrase(r)}, once the beam has come to rest, which side "
                      f"is lower?", st["dir"], True, p, mf(st["rest"][1] - st["rest"][0]))
    if key == "heaviest_of":
        ms = [objs[o]["mass"] for o in p["objs"]]
        if ms.count(max(ms)) != 1:
            return None
        gt = p["objs"][ms.index(max(ms))]
        names = ", ".join(name(objs[o]) for o in p["objs"])
        span = min(c.present_span(o) for o in p["objs"])
        return _name_q(rng, lv, key, f"Each dot is one unit of mass. Which of these weights is the heaviest: "
                       f"{names}?", gt, p["objs"], False, p, mf(span), c)
    if key == "where_end":
        o = objs[p["obj"]]
        txt = {"left": "on the left pan", "right": "on the right pan", None: "lifted off the scale"}
        gt = txt[fin["on"][o["id"]][0] if o["id"] in fin["on"] else None]
        opts, a = L.mc(rng, gt, [v for v in txt.values() if v != gt] + ["on the floor beside the scale"])
        return L.question(lv, key, f"Where is {name(o)} when the video ends?", gt, opts, a, True,
                          dict(p, min_fps=mf(c.final_win)))
    if key == "event_time":
        r = _find(c, p)
        gt_t = sim.timer_tenths(r["f"])
        cands = [sim.timer_tenths(x["f"]) for x in c.refs if x is not r]
        cands += [gt_t + d for d in (10, -10, 15, -15, 20, -20)]
        rng.shuffle(cands)
        chosen = []
        for v in cands:
            if 5 <= v <= sc["n_frames"] * 10 // sim.FPS and abs(v - gt_t) >= TIME_SEP and \
                    all(abs(v - w) >= 3 for w in chosen):
                chosen.append(v)
        if len(chosen) < 3:
            return None
        sep = min(abs(v - gt_t) for v in chosen[:3]) / 10
        opts, a = L.mc(rng, _tenths(gt_t), [_tenths(v) for v in chosen])
        return L.question(lv, key, f"According to the timer, at what time does {c.phrase(r, base=True)}?", _tenths(gt_t),
                          opts, a, True, dict(p, min_fps=math.ceil(200 / sep) / 100))
    if key == "gap":
        ra, rb = _find(c, p["a"]), _find(c, p["b"])
        g = sim.timer_tenths(rb["f"]) - sim.timer_tenths(ra["f"])
        ds = [g + GAP_STEP, g - GAP_STEP, g + 2 * GAP_STEP, g - 2 * GAP_STEP, g + 3 * GAP_STEP]
        ds = [d for d in ds if d >= 2]
        opts, a = L.mc(rng, _tenths(g), [_tenths(d) for d in ds])
        return L.question(lv, key, f"According to the timer, how much time passes between the moment "
                          f"{c.phrase(ra)} and the moment {c.phrase(rb)}?", _tenths(g), opts, a, True,
                          dict(p, min_fps=math.ceil(400 / (GAP_STEP / 10)) / 100))
    if key == "side_mass_end":
        on = [o for o, (s, _) in fin["on"].items() if s == p["side"]]
        if any(objs[o]["mystery"] for o in on):
            return None
        m = sum(objs[o]["mass"] for o in on)
        return _int_q(rng, lv, key, f"Each dot is one unit of mass. What is the total mass on the "
                      f"{side_txt(sc, p['side'])} when the video ends?", m, False, p, mf(c.final_win))
    if key in ("landings", "lifts", "hops", "hops_to"):
        if key == "landings":
            rs = [r for r in c.refs if r["kind"] in ("drop", "place") and sc["events"][r["i"]]["side"] == p["side"]]
            text = (f"How many weights land on the {side_txt(sc, p['side'])} during the video (not counting any "
                    f"that were there from the start)?")
        elif key == "lifts":
            rs = [r for r in c.refs if r["kind"] == "lift"]
            text = "How many weights are lifted off the scale during the video?"
        elif key == "hops":
            rs = [r for r in c.refs if r["kind"] == "hop"]
            text = "How many times does a weight jump from one pan to the other during the video?"
        else:
            rs = [r for r in c.refs if r["kind"] == "hop" and sc["events"][r["i"]]["side"] == p["side"]]
            text = f"How many times does a weight jump onto the {p['side']} pan from the other pan?"
        fs = [r["f"] for r in rs]
        gap = min([b - a for a, b in zip(fs, fs[1:])] or [c.min_win * 2])
        return _int_q(rng, lv, key, text, len(rs), True, p, mf(min(gap, 2 * c.min_win)))
    if key == "max_side_count":
        n = max(sum(1 for s, _ in st["on"].values() if s == p["side"]) for st in c.states)
        return _int_q(rng, lv, key, f"What is the largest number of weights that are on the "
                      f"{p['side']} pan at the same time during the video?", n, True, p, mf(c.min_win))
    if key == "changes":
        return _int_q(rng, lv, key, "Each time the beam comes to rest, it rests in one of three positions: left "
                      "side lower, right side lower, or level. Counting from the start of the video, how many times "
                      "does the resting position change?", sim.n_changes(c.states), True, p, mf(c.min_win))
    if key == "level_count":
        md = sim.merged_dirs(c.states)
        n = sum(1 for d in md[1:] if d == "level")
        return _int_q(rng, lv, key, "Not counting how it starts, how many separate times does the beam come to "
                      "rest perfectly level during the video?", n, True, p, mf(c.min_win))
    if key == "longest_dir":
        tot = {"left": 0, "right": 0, "level": 0}
        for st in c.rest:
            tot[st["dir"]] += st["rest"][1] - st["rest"][0]
        top = sorted(tot.values(), reverse=True)
        if top[0] - top[1] < 18:
            return None
        gt = max(tot, key=lambda d: tot[d])
        return _dir_q(rng, lv, key, "Over the whole video, which resting position does the beam spend the most "
                      "time in?", gt, True, p, mf(c.min_win))
    if key == "first_turn":
        d = p["dir"]
        if c.rest[0]["dir"] == d:
            return None
        k = next((i for i, st in enumerate(c.rest) if st["dir"] == d), None)
        if k is None:
            return None
        st = c.rest[k]
        r = next(x for x in c.refs if x["i"] == st["event"])
        movers = sorted({x["obj"] for x in c.refs})
        if len(movers) < 3 or len(objs) < 4:
            return None
        win = min(st["rest"][1] - st["rest"][0], c.rest[k - 1]["rest"][1] - c.rest[k - 1]["rest"][0])
        return _name_q(rng, lv, key, f"Which weight, by landing, being lifted off or moving pans, is the first "
                       f"to make the {d} side of the beam the lower side?", r["obj"], movers, True, p, mf(win), c, pad=True)
    if key == "mystery":
        o = next(x for x in objs if x["mystery"])
        lvl = next(st for st in c.rest if st["dir"] == "level" and o["id"] in st["on"])
        return _int_q(rng, lv, key, f"{name(o).capitalize()} has no dots. Counting each dot as one unit of mass, "
                      f"what is its mass?", o["mass"], True, p, mf(lvl["rest"][1] - lvl["rest"][0]), lo=1)
    if key == "max_side_mass":
        m = max(sum(objs[o]["mass"] for o, (s, _) in st["on"].items() if s == p["side"]) for st in c.states)
        return _int_q(rng, lv, key, f"Each dot is one unit of mass. What is the largest total mass on the "
                      f"{p['side']} pan at any moment during the video?", m, True, p, mf(c.min_win))
    if key == "mass_after":
        r = _find(c, p)
        st = c.state_after(r)
        on = [o for o, (s, _) in st["on"].items() if s == p["side"]]
        if any(objs[o]["mystery"] for o in on):
            return None
        m = sum(objs[o]["mass"] for o in on)
        return _int_q(rng, lv, key, f"Each dot is one unit of mass. Right after {c.phrase(r)}, what is the total "
                      f"mass on the {side_txt(sc, p['side'])}?", m, True, p, mf(st["rest"][1] - st["rest"][0]))
    if key == "torque_end":
        t = sum(objs[o]["mass"] * peg for o, (s, peg) in fin["on"].items() if s == p["side"])
        return _int_q(rng, lv, key, f"Each dot is one unit of mass, and the pegs are numbered 1-4 outward from the "
                      f"pivot. For the weights on the {p['side']} half of the beam at the end, what is the sum of "
                      f"mass x peg number?", t, False, p, mf(c.final_win))
    # ---- counterfactuals (re-simulated with sim.timeline)
    if key == "cf_remove":
        o = objs[p["obj"]]
        st = sim.timeline(sc, drop_obj=o["id"])[-1]
        return _dir_q(rng, lv, key, f"Suppose {name(o)} had never been on the scale at all, and everything else "
                      f"happened the same way. Which side would be lower at the end?", st["dir"], True, p,
                      mf(c.final_win))
    if key == "cf_other_side":
        o = objs[p["obj"]]
        if scen == "lever" and _peg_clash(c, flip=o["id"]):
            return None
        alt = sim.timeline(sc, side_flip=o["id"])
        if scen != "lever" and any(sum(1 for s, _ in x["on"].values() if s == side) > sim.PAN_CAP
                                   for x in alt for side in sim.SIDES):
            return None
        st = alt[-1]
        where = "on the matching peg of the other half of the beam" if scen == "lever" else "on the other pan"
        return _dir_q(rng, lv, key, f"Suppose {name(o)} had been put {where} instead, and everything else "
                      f"happened the same way. Which side would be lower at the end?", st["dir"], True, p,
                      mf(c.final_win))
    if key == "cf_heavier":
        o = objs[p["obj"]]
        ms = {x["id"]: x["mass"] for x in objs}
        ms[o["id"]] += p["delta"]
        st = sim.timeline(sc, masses=ms)[-1]
        more = "unit" if p["delta"] == 1 else "units"
        return _dir_q(rng, lv, key, f"Suppose {name(o)} were {p['delta']} {more} of mass heavier (one dot = one "
                      f"unit), and everything else happened the same way. Which side would be lower at the end?",
                      st["dir"], True, p, mf(c.final_win))
    if key == "cf_extra":
        on = dict(fin["on"])
        if scen == "lever":
            if [p["side"], p["peg"]] in on.values():
                return None
            where = f"on the empty {ORD[p['peg']]} peg (counting outward from the pivot) of the {p['side']} half"
        else:
            if sum(1 for s, _ in on.values() if s == p["side"]) >= sim.PAN_CAP:
                return None
            where = f"on the {p['side']} pan"
        tq = fin["torque"] + (1 if p["side"] == "right" else -1) * p["mass"] * (p.get("peg") or 1)
        dots = "dot" if p["mass"] == 1 else "dots"
        return _dir_q(rng, lv, key, f"Suppose one more weight with {p['mass']} {dots} were put {where} after the "
                      f"video ends. Which side would be lower once the beam settles?", sim.direction(tq), True, p,
                      mf(c.final_win))
    if key == "cf_changes_remove":
        o = objs[p["obj"]]
        st = sim.timeline(sc, drop_obj=o["id"])
        return _int_q(rng, lv, key, f"Suppose {name(o)} had never been on the scale at all, and everything else "
                      f"happened at the same times. Counting from the start, how many times would the beam's resting "
                      f"position (left side lower, right side lower, or level) change?", sim.n_changes(st), True,
                      p, mf(c.min_win), extra=[sim.n_changes(c.states)])
    if key == "cf_skip_hop":
        rs = [r for r in c.refs if r["obj"] == p["obj"] and r["kind"] == "hop"]
        if len(rs) != 1:
            return None
        o = objs[p["obj"]]
        st = sim.timeline(sc, skip=[rs[0]["i"]])[-1]
        return _dir_q(rng, lv, key, f"Suppose {name(o)} had stayed on its pan instead of jumping to the other "
                      f"one, and everything else happened the same way. Which side would be lower at the end?",
                      st["dir"], True, p, mf(c.final_win))
    if key == "cf_no_lift":
        r = next(r for r in c.refs if r["obj"] == p["obj"] and r["kind"] == "lift")
        o = objs[p["obj"]]
        st = sim.timeline(sc, skip=[r["i"]])[-1]
        if sum(1 for s, _ in st["on"].values() if s == st["on"][o["id"]][0]) > sim.PAN_CAP:
            return None
        return _dir_q(rng, lv, key, f"Suppose {name(o)} had never been lifted off, and everything else happened "
                      f"the same way. Which side would be lower at the end?", st["dir"], True, p, mf(c.final_win))
    if key == "cf_shift_peg":
        o = objs[p["obj"]]
        side, peg = fin["on"][o["id"]]
        if not 1 <= peg + p["delta"] <= 4 or _peg_clash(c, shift={o["id"]: p["delta"]}):
            return None
        st = sim.timeline(sc, peg_shift={o["id"]: p["delta"]})[-1]
        how = "one peg further from the pivot" if p["delta"] > 0 else "one peg closer to the pivot"
        return _dir_q(rng, lv, key, f"Suppose {name(o)} had been put {how}, on the same side, and everything else "
                      f"happened the same way. Which side would be lower at the end?", st["dir"], True, p,
                      mf(c.final_win))
    raise KeyError(key)


def _peg_clash(c, flip=None, shift=None) -> bool:
    for st in sim.timeline(c.scene, side_flip=flip, peg_shift=shift):
        used = [tuple(v) for v in st["on"].values()]
        if len(set(used)) != len(used):
            return True
    return False


def _tenths(v: int) -> str:
    return f"{v // 10}.{v % 10} s"


def _find(c, p):
    return next(r for r in c.refs if r["obj"] == p["obj"] and r["act"] == p["act"] and r["occ"] == p["occ"])


def build(scene: dict, rng: random.Random) -> list[dict]:
    c = Ctx(scene)
    counts = L.ten_counts(rng)
    cand = candidates(c)
    qs = []
    for lv, need in zip(range(1, 6), counts):
        by_key = {}
        for key, p in cand[lv]:
            by_key.setdefault(key, []).append(p)
        keys = list(by_key)
        rng.shuffle(keys)
        got, seen = [], set()
        # pass 1: one question per template (template first, then its params);
        # pass 2: further params of the same templates if the level still needs more
        for _pass in range(2):
            for key in keys:
                params = list(by_key[key])
                rng.shuffle(params)
                for p in params:
                    if len(got) == need:
                        break
                    sig = (key, repr(sorted(p.items())))
                    if sig in seen:
                        continue
                    seen.add(sig)
                    q = build_one(c, rng, lv, key, p)
                    if q is not None:
                        got.append(q)
                        break
        if len(got) != need:
            raise RuntimeError(f"level {lv}: only {len(got)} of {need} questions")
        qs += got
    return L.number(qs)
