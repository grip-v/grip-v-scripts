"""Scene generation for balance_scale_weighing.

A balance (two pans on posts, or a plain beam with pegs) receives weights
over a clip of at most 10 s. Every weight is a coloured shape whose mass is
the number of dots printed on it (one dot = one unit of mass). The beam's
resting angle is closed-form: k degrees per unit of torque difference
(right minus left), clamped at +-max_deg, and the beam eases to each new
target over `trans_f` frames (smoothstep). Right heavier -> right side lower.

Scenarios (scene["scenario"]):
  drop        weights fall one by one onto the left or right pan
  add_remove  some weights start on the pans; new ones fall on, others are lifted off
  transfer    weights start on the pans and hop over to the other pan
  lever       a beam with 4 pegs per side; weights fall onto pegs, torque = dots x peg number
  mystery     like drop, but one weight has no dots; its mass can be worked out
              because the beam comes to rest exactly level at some point

Everything is in integer frames (24 fps); times shown to the viewer come
from the on-screen timer, which shows floor(t * 10) / 10 seconds. Every
moment a question names (a landing, a lift-off) is put on a frame where the
exact time is within 0.045 s above its tenth, so the timer reading and the
true time agree to 0.1 s.

All randomness comes from the random.Random passed to build(); the scene is
plain JSON (lists, dicts, ints, floats rounded to 3 decimals).
"""
from __future__ import annotations
import random

FPS = 24
MAX_FRAMES = 240                       # 10 s
SCENARIOS = ("drop", "add_remove", "transfer", "lever", "mystery")
COLOURS = ("red", "orange", "yellow", "green", "teal", "blue", "purple", "pink", "brown")
SHAPES = ("circle", "square", "hexagon", "triangle", "pentagon")
PATTERNS = ("plain", "striped", "ringed", "two-tone")
SIDES = ("left", "right")
PAN_CAP = 4                            # slots per pan
PEGS = 4                               # pegs per side on the lever
MIN_DEG = 4.0                          # any tilted rest state is at least this steep
WINDOW_F = 11                          # every rest state lasts >= 11 frames (0.46 s)
FINAL_HOLD = (22, 34)                  # frames the final rest state is held
EDGE_F = 12                            # question events stay >= 0.5 s from both ends


def aligned(f: int) -> bool:
    """Frame whose exact time is within 0.045 s above a tenth (timer == true time to 0.1 s)."""
    return (f * 10) % FPS <= 10


def next_aligned(f: int) -> int:
    while not aligned(f):
        f += 1
    return f


def timer_tenths(f: int) -> int:
    return (f * 10) // FPS


def display(f: int) -> str:
    t = timer_tenths(f)
    return f"{t // 10}.{t % 10} s"


# --------------------------------------------------------------------------- physics


def arm(scene, side, peg):
    """Torque arm in scene units: 1 for a pan, the peg number on the lever."""
    return peg if scene["scenario"] == "lever" else 1


def torque(scene, on: dict, masses=None) -> int:
    """Right-minus-left torque of the weights in `on` ({obj: [side, peg]})."""
    ms = masses or {o["id"]: o["mass"] for o in scene["objects"]}
    tq = 0
    for oid, (side, peg) in on.items():
        tq += (1 if side == "right" else -1) * ms[oid] * arm(scene, side, peg)
    return tq


def target_deg(scene, tq: int) -> float:
    a = scene["deg_per_unit"] * tq
    return max(-scene["max_deg"], min(scene["max_deg"], a))


def direction(tq: int) -> str:
    return "right" if tq > 0 else "left" if tq < 0 else "level"


def change_frames(ev: dict) -> list[int]:
    """Frames at which an event changes what is on the scale."""
    if ev["kind"] in ("drop", "place"):
        return [ev["f1"]]
    if ev["kind"] == "lift":
        return [ev["f0"]]
    return [ev["f0"], ev["f1"]]          # hop: leaves, then lands on the other side


def moment(ev: dict) -> int:
    """The frame a question refers to: landing, lift-off, or landing after a hop."""
    return ev["f0"] if ev["kind"] == "lift" else ev["f1"]


def timeline(scene: dict, masses=None, drop_obj=None, side_flip=None, skip=None, peg_shift=None) -> list[dict]:
    """States of the scale: the initial one, then one per change point.

    Optional counterfactual edits: drop_obj (object never on the scale),
    side_flip (object placed on the other side), skip (event indices left
    out), peg_shift ({obj: delta} on the lever). Each state: on, torque, dir,
    angle, fc (change frame), event (index or None), leaving (hop departure),
    rest (start, end) frames during which the beam is still."""
    skip = set(skip or ())
    def place(oid, side, peg):
        if side_flip == oid:
            side = "left" if side == "right" else "right"
        if peg_shift and oid in peg_shift and peg is not None:
            peg = peg + peg_shift[oid]
        return [side, peg]

    on = {}
    for it in scene["initial"]:
        if it["obj"] != drop_obj:
            on[it["obj"]] = place(it["obj"], it["side"], it["peg"])
    tq = torque(scene, on, masses)
    states = [{"on": dict(on), "torque": tq, "dir": direction(tq), "angle": target_deg(scene, tq),
               "fc": 0, "event": None, "leaving": False}]
    for i, ev in enumerate(scene["events"]):
        if i in skip or ev["obj"] == drop_obj:
            continue
        oid = ev["obj"]
        if ev["kind"] in ("drop", "place"):
            on[oid] = place(oid, ev["side"], ev["peg"])
        elif ev["kind"] == "lift":
            on.pop(oid, None)
        else:                                            # hop: leave, then land on the other side
            side = on.pop(oid)[0]
            tq = torque(scene, on, masses)
            states.append({"on": dict(on), "torque": tq, "dir": direction(tq), "angle": target_deg(scene, tq),
                           "fc": ev["f0"], "event": i, "leaving": True})
            on[oid] = ["left" if side == "right" else "right", None]
        tq = torque(scene, on, masses)
        states.append({"on": dict(on), "torque": tq, "dir": direction(tq), "angle": target_deg(scene, tq),
                       "fc": ev["f0"] if ev["kind"] == "lift" else ev["f1"], "event": i, "leaving": False})
    # rest windows: the beam is still from the end of the transition until the next change
    cur = states[0]["angle"]
    moving_until = 0
    for k, st in enumerate(states):
        if k > 0 and st["angle"] != cur:
            moving_until = st["fc"] + scene["trans_f"]
        cur = st["angle"]
        start = max(st["fc"], moving_until)
        end = states[k + 1]["fc"] if k + 1 < len(states) else scene["n_frames"]
        st["rest"] = [start, end]
    return states


def rest_states(states: list[dict]) -> list[dict]:
    """States the beam actually comes to rest in (hop departures are only in flight)."""
    return [s for s in states if not s["leaving"]]


def merged_dirs(states: list[dict]) -> list[str]:
    out = []
    for s in rest_states(states):
        if not out or out[-1] != s["dir"]:
            out.append(s["dir"])
    return out


def n_changes(states) -> int:
    return len(merged_dirs(states)) - 1


# --------------------------------------------------------------------------- checks


def check(scene: dict) -> list[str]:
    errs = []
    nf = scene["n_frames"]
    if nf > MAX_FRAMES:
        errs.append(f"clip too long: {nf} frames")
    objs = scene["objects"]
    if [o["id"] for o in objs] != list(range(len(objs))):
        errs.append("object ids not 0..n-1")
    if len({o["colour"] for o in objs}) != len(objs):
        errs.append("two weights share a colour")
    if len({(o["colour"], o["shape"], o["pattern"]) for o in objs}) != len(objs):
        errs.append("two weights look the same")
    states = timeline(scene)
    for k, s in enumerate(states):
        if s["dir"] != "level" and abs(s["angle"]) < MIN_DEG - 1e-9:
            errs.append(f"tilt of {s['angle']:.2f} deg too small to see")
        if s["leaving"]:
            if s["rest"][1] > s["rest"][0] and s["angle"] != states[k - 1]["angle"]:
                errs.append("beam comes to rest while a weight is in flight")
            continue
        ln = s["rest"][1] - s["rest"][0]
        need = FINAL_HOLD[0] if s is states[-1] else WINDOW_F
        if ln < need:
            errs.append(f"rest window of {ln} frames after frame {s['fc']} is too short")
    for s in states:
        sides = {"left": 0, "right": 0}
        pegs = set()
        for oid, (side, peg) in s["on"].items():
            sides[side] += 1
            if peg is not None:
                if (side, peg) in pegs or not 1 <= peg <= PEGS:
                    errs.append(f"peg clash or bad peg {side}{peg}")
                pegs.add((side, peg))
        if scene["scenario"] != "lever" and max(sides.values()) > PAN_CAP:
            errs.append("more than 4 weights on one pan")
    last = 0
    for ev in scene["events"]:
        m = moment(ev)
        if not aligned(m):
            errs.append(f"event frame {m} is not timer-aligned")
        if not EDGE_F <= m <= nf - EDGE_F:
            errs.append(f"event frame {m} too close to the clip edges")
        for fc in change_frames(ev):
            if fc - last < 2:
                errs.append(f"events at frames {last} and {fc} are too close")
            last = fc
        if ev["f0"] < 1 or ev["f1"] <= ev["f0"]:
            errs.append(f"bad event frames {ev['f0']}..{ev['f1']}")
    if scene["scenario"] == "mystery":
        mo = [o for o in objs if o["mystery"]]
        if len(mo) != 1:
            errs.append("mystery scene needs exactly one dotless weight")
        else:
            mid = mo[0]["id"]
            if not any(s["dir"] == "level" and mid in s["on"] for s in rest_states(states)):
                errs.append("the dotless weight's mass can't be worked out")
    elif any(o["mystery"] for o in objs):
        errs.append("dotless weight outside the mystery scenario")
    if n_changes(states) < 2:
        errs.append("fewer than 2 changes of resting position")
    return errs


# --------------------------------------------------------------------------- build


def _theme(rng):
    base = rng.choice([
        # bg top, bg bottom, floor, stand, beam, pan, peg, ink(outline)
        ((236, 233, 224), (214, 208, 196), (150, 118, 86), (92, 88, 96), (70, 74, 84), (176, 180, 188)),
        ((222, 232, 240), (196, 210, 224), (120, 128, 140), (70, 78, 92), (52, 60, 74), (200, 206, 214)),
        ((38, 42, 54), (22, 24, 32), (70, 62, 56), (150, 156, 168), (196, 200, 210), (120, 126, 138)),
        ((242, 238, 246), (220, 214, 232), (128, 112, 140), (84, 78, 98), (64, 58, 78), (188, 182, 200)),
        ((228, 240, 226), (204, 222, 200), (110, 128, 96), (72, 84, 70), (58, 66, 56), (184, 196, 180)),
        ((50, 44, 40), (30, 26, 24), (96, 80, 64), (168, 160, 150), (210, 200, 188), (130, 122, 112)),
        ((250, 246, 236), (236, 228, 210), (176, 160, 130), (110, 100, 90), (88, 80, 72), (196, 190, 178)),
        ((30, 46, 52), (18, 30, 34), (52, 72, 76), (150, 172, 176), (190, 210, 212), (110, 132, 136)),
    ])
    jit = lambda c: [max(0, min(255, v + rng.randint(-10, 10))) for v in c]
    names = ("bg_top", "bg_bot", "floor", "stand", "beam", "pan")
    th = {n: jit(c) for n, c in zip(names, base)}
    th["bg_pattern"] = rng.choice(["plain", "grid", "stripes", "dots", "tiles"])
    th["floor_pattern"] = rng.choice(["plain", "planks", "checker"])
    th["floor_y"] = rng.randint(600, 640)
    th["stand_style"] = rng.choice(["pillar", "a_frame", "column"])
    th["pan_style"] = rng.choice(["tray", "bowl", "plate"])
    return th


def _objects(rng, n, scenario):
    cols = rng.sample(COLOURS, n)
    objs, looks = [], set()
    lever = scenario == "lever"
    for i in range(n):
        shape, pat = rng.choice(SHAPES), rng.choice(PATTERNS)
        objs.append({"id": i, "colour": cols[i], "shape": shape, "pattern": pat,
                     "mass": rng.randint(1, 4 if lever else 5),
                     "size": rng.randint(54, 62) if lever else rng.randint(58, 80), "mystery": False})
    return objs


def _schedule(rng, kinds, trans_f, fall_f, hop_f, lift_f):
    """Frames for a list of event kinds; None if the clip would run past 10 s."""
    f = next_aligned(rng.randint(20, 30))
    evs = []
    for k, kind in enumerate(kinds):
        if k:
            f = next_aligned(f + rng.randint(0, 6))
        if kind in ("drop", "place"):
            ev = {"kind": kind, "f0": f - fall_f, "f1": f}
            last = f
        elif kind == "lift":
            ev = {"kind": kind, "f0": f, "f1": f + lift_f}
            last = f
        else:
            land = f + hop_f
            while not aligned(land):
                land -= 1
            if land - f < 6:
                return None
            ev = {"kind": kind, "f0": f, "f1": land}
            last = land
        evs.append(ev)
        f = last + trans_f + WINDOW_F
    end = last + trans_f + rng.randint(*FINAL_HOLD)
    if end > MAX_FRAMES or end - evs[-1]["f1"] < EDGE_F:
        return None
    return evs, end


def _pick_sides(rng, scene, events, initial_on):
    """Assign sides / pegs to the events in order, respecting pan capacity and peg use."""
    on = {k: list(v) for k, v in initial_on.items()}
    lever = scene["scenario"] == "lever"
    for ev in events:
        oid = ev["obj"]
        if ev["kind"] in ("drop", "place"):
            if lever:
                free = [(s, p) for s in SIDES for p in range(1, PEGS + 1)
                        if [s, p] not in on.values()]
                s, p = rng.choice(free)
                ev["side"], ev["peg"] = s, p
            else:
                free = [s for s in SIDES if sum(1 for v in on.values() if v[0] == s) < PAN_CAP]
                if not free:
                    return False
                ev["side"], ev["peg"] = rng.choice(free), None
            on[oid] = [ev["side"], ev["peg"]]
        elif ev["kind"] == "lift":
            ev["side"], ev["peg"] = on.pop(oid)[0], None
        else:
            src = on[oid][0]
            dst = "left" if src == "right" else "right"
            if sum(1 for v in on.values() if v[0] == dst) >= PAN_CAP:
                return False
            ev["side"], ev["peg"] = dst, None
            on[oid] = [dst, None]
    return True


def _try(rng, scenario):
    lever = scenario == "lever"
    trans_f = rng.randint(10, 14)
    fall_f = rng.randint(7, 12)
    hop_f = rng.randint(7, min(10, trans_f - 2))
    lift_f = rng.randint(8, 12)
    initial_n = {"drop": 0, "mystery": 0, "lever": rng.randint(0, 2), "add_remove": rng.randint(2, 4),
                 "transfer": rng.randint(3, 6)}[scenario]
    if scenario in ("drop", "mystery"):
        kinds = ["drop"] * rng.randint(4, 7)
    elif scenario == "lever":
        kinds = ["place"] * rng.randint(4, 6)
    elif scenario == "add_remove":
        nd = rng.randint(2, 4)
        nl = rng.randint(2, min(4, initial_n + nd - 1))
        kinds = ["drop"] * nd + ["lift"] * nl
        rng.shuffle(kinds)
    else:
        nh = rng.randint(3, 6)
        kinds = ["hop"] * nh + ["drop"] * rng.randint(0, min(2, 7 - nh))
        rng.shuffle(kinds)
    sched = _schedule(rng, kinds, trans_f, fall_f, hop_f, lift_f)
    if sched is None:
        return None
    events, n_frames = sched
    n_new = sum(1 for k in kinds if k in ("drop", "place"))
    n_obj = initial_n + n_new
    if n_obj > len(COLOURS):
        return None
    objs = _objects(rng, n_obj, scenario)
    scene = {"scenario": scenario, "n_frames": n_frames, "duration": round(n_frames / FPS, 3),
             "trans_f": trans_f, "fall_f": fall_f, "lift_f": lift_f,
             "deg_per_unit": round(rng.uniform(2.6, 4.2) if lever else rng.uniform(4.0, 6.0), 2),
             "max_deg": round(rng.uniform(12.0, 16.0), 1),
             "camera": {"cx": rng.randint(600, 680), "cy": rng.randint(405, 445) if not lever
                        else rng.randint(390, 440), "sc": round(rng.uniform(0.9, 1.06), 3)},
             "theme": _theme(rng), "objects": objs, "initial": [], "events": []}
    # initial placements
    on = {}
    for oid in range(initial_n):
        if lever:
            free = [(s, p) for s in SIDES for p in range(1, PEGS + 1) if [s, p] not in on.values()]
            s, p = rng.choice(free)
        else:
            free = [s for s in SIDES if sum(1 for v in on.values() if v[0] == s) < PAN_CAP - 1]
            s, p = rng.choice(free), None
        on[oid] = [s, p]
        scene["initial"].append({"obj": oid, "side": s, "peg": p})
    # who does what
    nxt = initial_n
    present = list(range(initial_n))
    lifted = []
    for ev, kind in zip(events, kinds):
        if kind in ("drop", "place"):
            ev["obj"] = nxt
            present.append(nxt)
            nxt += 1
        elif kind == "lift":
            if not present:
                return None
            oid = rng.choice(present)
            present.remove(oid)
            ev["obj"] = oid
        else:
            hoppers = [o for o in present if sum(1 for e in events if e.get("obj") == o and e["kind"] == "hop") < 2]
            if not hoppers:
                return None
            ev["obj"] = rng.choice(hoppers)
    for ev in events:
        ev.update({"side": None, "peg": None})
    if not _pick_sides(rng, scene, events, on):
        return None
    scene["events"] = [{"kind": e["kind"], "obj": e["obj"], "side": e["side"], "peg": e["peg"],
                        "f0": e["f0"], "f1": e["f1"]} for e in events]
    if scenario == "mystery":
        mid = rng.randrange(n_obj - 1)                  # never the last weight to land
        objs[mid]["mystery"] = True
        ok = []
        for m in range(1, 7):
            objs[mid]["mass"] = m
            if not check(scene):
                ok.append(m)
        if not ok:
            return None
        objs[mid]["mass"] = rng.choice(ok)
    return scene


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    for attempt in range(1, 400):
        scene = _try(rng, scenario)
        if scene is not None and not check(scene):
            scene["gen_attempts"] = attempt
            return scene
    raise RuntimeError(f"no valid {scenario} scene")
