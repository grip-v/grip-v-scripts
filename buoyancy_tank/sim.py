"""Buoyancy tank: scene generation, physics and consistency checks.

A 2-D side view of a water tank (everything per unit depth, lengths in cm,
areas in cm^2). 3-6 solid objects of six shapes (square, wide block, tall
block, ball, hexagon, triangle) and hidden densities sit in or above the
tank. Anything lighter than water floats with exactly (density x its area)
of water displaced; anything heavier rests on the floor. The water level
is (water area + displaced area) / tank width.

Four scenarios (scene["scenario"]):

  drop   objects hang on strings above the water and are let go one at a
         time (a string vanishes); they fall in, splash down, then float
         or sink to the floor.
  rise   objects are strapped to the floor under deep water and the straps
         vanish one at a time; floaters shoot up and break the surface.
  fill   objects stand on the floor of a shallow tank; a tap pours water
         in for a few seconds; floaters lift off when the water is deep
         enough for them.
  drain  objects float or rest in a deep tank; a drain pipe empties water
         for a few seconds; floaters sink with the surface and come to
         rest on the floor.

Motion is one coupled ODE for every scenario: gravity, buoyancy from the
current submerged area, linear water drag, and the water level re-solved
from the displaced area every sub-step (Newton step on the level equation).
Sub-steps are 1/240 s; the renderer samples every 10th (24 fps).

Everything a question uses is checked here or in questions.py to have a
visible margin: final positions >= 15 px apart, events >= 0.5 s from the
clip ends, settled at least 1 s before the end, and so on.
"""
from __future__ import annotations
import functools
import json
import math
import random

FPS = 24
SUB = 10                       # physics sub-steps per video frame
DT = 1.0 / (FPS * SUB)
G = 600.0                      # cm/s^2 (slowed down for viewing)
GAMMA = 12.0                   # water drag force coefficient (per unit area density)
SIM_FRAMES = 288               # simulate 12 s; clips use the first <= 240 frames
MAX_FRAMES = 240               # 10 s at 24 fps
HOLD_FRAMES = 24               # everything is still for >= 1 s before the end
SETTLE_TOL = 0.03              # cm
MARGIN_PX = 15.0

SCENARIOS = ("drop", "rise", "fill", "drain")
SHAPES = ("square", "wide", "tall", "ball", "hexagon", "triangle")
SHAPE_WORD = {"square": "square", "wide": "wide block", "tall": "tall block", "ball": "ball",
              "hexagon": "hexagon", "triangle": "triangle"}
RECTS = ("square", "wide", "tall")
COLOURS = {"red": "#D62828", "orange": "#F28A1E", "yellow": "#F6D743", "green": "#2E9E44",
           "purple": "#7D3C98", "pink": "#F27FC0", "black": "#262626", "white": "#FFFFFF"}
PATTERNS = ("plain", "striped", "spotted", "checked")
BACKGROUNDS = ("#F3F0E8", "#E9EEF2", "#EFE9F3", "#EAF1E6", "#F5ECE4", "#E6E6E6", "#F2EEDC")
WALLS = ("#3A3F4B", "#4B3A2E", "#2F4858", "#555555", "#3E4A3A")
BACKWALLS = ("#FBFBF8", "#F4F7FA", "#FAF6F0", "#F2F2F2", "#F7F9F2")
WATERS = ((70, 150, 225), (60, 170, 200), (80, 130, 210), (70, 165, 175))


# ---------------------------------------------------------------- geometry

def sub_area(o: dict, d: float) -> float:
    """Area of object o below a water line d cm above its bottom."""
    h, w = o["h"], o["w"]
    if d <= 0.0:
        return 0.0
    if d > h:
        d = h
    k = o["shape"]
    if k in RECTS:
        return w * d
    if k == "ball":
        r = h / 2
        return r * r * math.acos(max(-1.0, min(1.0, (r - d) / r))) - (r - d) * math.sqrt(max(0.0, 2 * r * d - d * d))
    if k == "triangle":
        return w * (d - d * d / (2 * h))
    if k == "hexagon":
        half = h / 2
        if d <= half:
            return w * d / 2 + w * d * d / (2 * h)
        a = w * half / 2 + w * half * half / (2 * h)
        e = d - half
        return a + w * e - w * e * e / (2 * h)
    raise ValueError(k)


def waterline_width(o: dict, d: float) -> float:
    h, w = o["h"], o["w"]
    if d <= 0.0 or d >= h:
        return 0.0
    k = o["shape"]
    if k in RECTS:
        return w
    if k == "ball":
        r = h / 2
        return 2 * math.sqrt(max(0.0, d * (2 * r - d)))
    if k == "triangle":
        return w * (1 - d / h)
    if k == "hexagon":
        return w / 2 + w * (d / h if d <= h / 2 else (h - d) / h)
    raise ValueError(k)


def area(o: dict) -> float:
    return sub_area(o, o["h"])


def draft(o: dict, rho: float | None = None) -> float | None:
    """Depth below the surface at which o floats, or None if it sinks."""
    rho = o["rho"] if rho is None else rho
    if rho >= 1.0:
        return None
    target = rho * area(o)
    lo, hi = 0.0, o["h"]
    for _ in range(60):
        mid = (lo + hi) / 2
        if sub_area(o, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def equilibrium(objs: list, water: float, tw: float, rhos: list | None = None) -> tuple[float, list]:
    """Settled water level and object bottom heights for a given water area."""
    rhos = rhos or [o["rho"] for o in objs]
    drafts = [draft(o, r) for o, r in zip(objs, rhos)]

    def excess(L):
        disp = 0.0
        for o, r, d in zip(objs, rhos, drafts):
            disp += r * area(o) if (d is not None and L >= d) else sub_area(o, L)
        return tw * L - water - disp

    lo, hi = 0.0, 1000.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if excess(mid) < 0:
            lo = mid
        else:
            hi = mid
    L = (lo + hi) / 2
    return L, [L - d if (d is not None and L >= d) else 0.0 for d in drafts]


def level_for(objs: list, ys: list, water: float, tw: float) -> float:
    lo, hi = 0.0, 1000.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if tw * mid - water - sum(sub_area(o, mid - y) for o, y in zip(objs, ys)) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------- physics

def water_at(scene: dict, t: float) -> float:
    f = scene.get("flow")
    if not f:
        return scene["water0"]
    on, off = f["on"] / FPS, f["off"] / FPS
    return scene["water0"] + f["rate"] * min(max(t - on, 0.0), off - on)


def initial_state(scene: dict) -> tuple[float, list]:
    objs, tw = scene["objects"], scene["tank"]["w"]
    if scene["scenario"] in ("fill", "drain"):
        return equilibrium(objs, scene["water0"], tw)
    ys = [o["y0"] for o in objs]
    return level_for(objs, ys, scene["water0"], tw), ys


def _key(scene: dict) -> str:
    keep = {k: scene[k] for k in ("scenario", "tank", "water0", "flow", "objects")}
    return json.dumps(keep, sort_keys=True)


@functools.lru_cache(maxsize=512)
def _simulate(key: str) -> dict:
    scene = json.loads(key)
    objs, tw = scene["objects"], scene["tank"]["w"]
    n = len(objs)
    L, y = initial_state(scene)
    y = list(y)
    v = [0.0] * n
    rel = [o.get("release") for o in objs]
    rho = [o["rho"] for o in objs]
    A = [area(o) for o in objs]
    hs = [o["h"] for o in objs]
    frames = []
    state = [(y[i] == 0.0, L - y[i] > 1e-9, y[i] + hs[i] > L) for i in range(n)]
    trans = [[(0, state[i])] for i in range(n)]
    steps = SIM_FRAMES * SUB
    last_sched = max([r for r in rel if r is not None] + ([scene["flow"]["off"]] if scene.get("flow") else [0]))
    for s in range(steps):
        if s % SUB == 0:
            frames.append((round(L, 4), tuple(round(q, 4) for q in y)))
            k = len(frames) - 1
            if k > last_sched + 24 and k % 12 == 0 and max(abs(q) for q in v) < 1e-4:
                ref = frames[-24]
                if ref == frames[-1]:                   # at rest to 4 decimals for a whole second
                    break
        for i in range(n):
            if rel[i] is not None and s < rel[i] * SUB:
                continue
            o = objs[i]
            d = L - y[i]
            sub = 0.0 if d <= 0.0 else sub_area(o, d)
            a = G * (sub / (rho[i] * A[i]) - 1.0) - (GAMMA / rho[i]) * (sub / A[i]) * v[i]
            v[i] += a * DT
            y[i] += v[i] * DT
            if y[i] <= 0.0:
                y[i], v[i] = 0.0, max(v[i], 0.0)
        W = water_at(scene, (s + 1) * DT)
        for _ in range(2):                         # Newton on tw*L = W + sum(sub)
            f = tw * L - W
            fp = tw
            for o, yy in zip(objs, y):
                d = L - yy
                if d > 0.0:
                    f -= sub_area(o, d)
                    fp -= waterline_width(o, d)
            L -= f / fp
        for i in range(n):
            st = (y[i] == 0.0, L - y[i] > 1e-9, y[i] + hs[i] > L)
            if st != state[i]:
                state[i] = st
                trans[i].append((s + 1, st))
    else:
        frames.append((round(L, 4), tuple(round(q, 4) for q in y)))
    while len(frames) < SIM_FRAMES + 1:            # resting: hold the last state
        frames.append(frames[-1])
    return {"frames": tuple(frames), "trans": tuple(tuple(t) for t in trans)}


def simulate(scene: dict) -> dict:
    return _simulate(_key(scene))


def frame_of(substep: int) -> int:
    """First video frame that shows the state after `substep` sub-steps."""
    return -(-substep // SUB)


def timer_text(k: int) -> str:
    tenths = (10 * k) // FPS
    return f"{tenths // 10}.{tenths % 10} s"


def settle_frame(scene: dict) -> int:
    fr = simulate(scene)["frames"]
    Lf, yf = fr[-1]
    last = 0
    for k, (L, ys) in enumerate(fr):
        if abs(L - Lf) > SETTLE_TOL or any(abs(a - b) > SETTLE_TOL for a, b in zip(ys, yf)):
            last = k
    return last + 1


def final_water(scene: dict) -> float:
    return water_at(scene, 1e9)


def final_state(scene: dict, objs: list | None = None, rhos: list | None = None, water: float | None = None):
    objs = objs if objs is not None else scene["objects"]
    return equilibrium(objs, final_water(scene) if water is None else water, scene["tank"]["w"], rhos)


# ---------------------------------------------------------------- events

EVENTS = {"drop": ("release", "enter", "land"), "rise": ("release", "surface"),
          "fill": ("liftoff",), "drain": ("touchdown",)}


def events(scene: dict, raw: bool = False) -> dict:
    """{object index: {event type: frame}} for persistent, question-usable events.
    raw=True: the first crossing of every event that happens at all, unfiltered."""
    sim = simulate(scene)
    out = {}
    for i, (o, tr) in enumerate(zip(scene["objects"], sim["trans"])):
        ev = {}
        rel = o.get("release")
        if rel is not None:
            ev["release"] = rel

        def first(idx, val, after=0):
            hits = [s for s, st in tr if st[idx] == val and s >= after]
            if not hits:
                return None
            s0 = hits[0]
            if not raw and any(st[idx] != val for s, st in tr if s > s0):
                return None                         # not persistent
            return frame_of(s0)

        sc = scene["scenario"]
        if sc == "drop":
            e = first(1, True)
            if e is not None:
                ev["enter"] = e
            e = first(0, True)
            if e is not None:
                ev["land"] = e
        elif sc == "rise":
            if tr[0][1][2] is False:
                e = first(2, True, 1)
                if e is not None:
                    ev["surface"] = e
        elif sc == "fill":
            if tr[0][1][0]:
                e = first(0, False, 1)
                if e is not None:
                    ev["liftoff"] = e
        elif sc == "drain":
            if not tr[0][1][0]:
                e = first(0, True, 1)
                if e is not None:
                    ev["touchdown"] = e
        # keep an event only if it is a clean crossing between two drawn frames
        fr = sim["frames"]
        for et in [e for e in ev if e != "release" and not raw]:
            f = ev[et]
            if f < 1 or f >= len(fr):
                del ev[et]
                continue
            (La, ya), (Lb, yb) = fr[f - 1], fr[f]
            a, b = ya[i], yb[i]
            if et == "enter":
                ok = La - a <= -0.005 and Lb - b >= 0.005
            elif et in ("land", "touchdown"):
                ok = a >= 0.005 and b == 0.0
            elif et == "liftoff":
                ok = a == 0.0 and b >= 0.005
            else:
                ok = a + o["h"] - La <= -0.005 and b + o["h"] - Lb >= 0.005
            if not ok:
                del ev[et]
        out[i] = ev
    return out


def floor_changes(scene: dict) -> list[int]:
    """Frames at which any object's floor contact changes."""
    out = []
    for tr in simulate(scene)["trans"]:
        prev = tr[0][1][0]
        for s, st in tr[1:]:
            if st[0] != prev:
                out.append(frame_of(s))
                prev = st[0]
    return sorted(out)


def describe(o: dict) -> str:
    pat = "" if o["pattern"] == "plain" else o["pattern"] + " "
    return f"the {o['colour']} {pat}{SHAPE_WORD[o['shape']]}"


# ---------------------------------------------------------------- checks

def check(scene: dict) -> list[str]:
    errs = []
    objs, tw, th = scene["objects"], scene["tank"]["w"], scene["tank"]["h"]
    st = scene["style"]
    S = st["scale"]
    sim = simulate(scene)
    fr = sim["frames"]
    nf = scene["n_frames"]
    sf = settle_frame(scene)
    sched = [o["release"] for o in objs if o.get("release") is not None]
    if scene.get("flow"):
        sched += [scene["flow"]["on"], scene["flow"]["off"]]
    want = max(sf, max(sched, default=0)) + HOLD_FRAMES + 1
    if nf != want:
        errs.append(f"n_frames {nf} != {want}")
    if nf > MAX_FRAMES:
        errs.append("clip longer than 10 s")
    if min(sched, default=99) < 12:
        errs.append("a scheduled event is less than 0.5 s after the start")
    descs = [(o["colour"], o["pattern"], o["shape"]) for o in objs]
    if len(set(descs)) != len(descs) or len({o["colour"] for o in objs}) != len(objs):
        errs.append("two objects share a description")
    # settled state = equilibrium
    L, bottoms = final_state(scene)
    Lf, yf = fr[nf - 1]
    if abs(Lf - L) > 0.05 or any(abs(a - b) > 0.05 for a, b in zip(yf, bottoms)):
        errs.append("simulation does not settle onto the equilibrium")
    # layout
    bs = sorted(objs, key=lambda o: o["x"])
    if bs[0]["x"] < 2 or bs[-1]["x"] + bs[-1]["w"] > tw - (8 if scene.get("flow") else 2):
        errs.append("object touches a wall")
    if any(q["x"] - (p["x"] + p["w"]) < 3 for p, q in zip(bs, bs[1:])):
        errs.append("objects too close side by side")
    if any(min(o["w"], o["h"]) * S < 26 for o in objs):
        errs.append("object smaller than 26 px")
    # unambiguous end state for every object
    for o, yb in zip(objs, bottoms):
        d = draft(o)
        if yb > 0:
            if yb * S < MARGIN_PX:
                errs.append(f"{describe(o)} floats too close to the floor")
            if (yb + o["h"] - L) * S < 12:
                errs.append(f"{describe(o)} barely shows above the water")
        else:
            if d is not None and (d - L) < 0.6:
                errs.append(f"{describe(o)} is borderline between floating and grounded")
            if abs(o["h"] - L) * S < 12:
                errs.append(f"{describe(o)}'s top is level with the water")
    # tank limits over the whole clip
    for k in range(nf):
        Lk, ys = fr[k]
        if Lk > th - 2.5 or Lk < 0.8:
            errs.append("water out of range")
            break
        if any(yy + o["h"] > th + 0.5 for o, yy in zip(objs, ys)):
            errs.append("object leaves the tank")
            break
    if scene["scenario"] == "drop":
        for i, o in enumerate(objs):
            if any(fr[k][1][i] <= fr[k][0] + 1.0 for k in range(o["release"] + 1)):
                errs.append(f"{describe(o)} is wet before its release")
    ev = events(scene)
    need = {"drop": "enter", "rise": "surface", "fill": "liftoff", "drain": "touchdown"}[scene["scenario"]]
    if not any(need in e and 12 <= e[need] <= nf - 13 for e in ev.values()):
        errs.append(f"no usable {need} event")
    return errs


# ---------------------------------------------------------------- build

def _dims(rng: random.Random, shape: str) -> tuple[float, float]:
    if shape == "square":
        s = rng.uniform(5, 11)
        return s, s
    if shape == "wide":
        w = rng.uniform(9, 17)
        return w, max(4.0, w * rng.uniform(0.35, 0.6))
    if shape == "tall":
        h = rng.uniform(9, 16)
        return max(4.0, h * rng.uniform(0.4, 0.65)), h
    if shape == "ball":
        d = rng.uniform(5.5, 12)
        return d, d
    if shape == "hexagon":
        w = rng.uniform(6.5, 13)
        return w, w * 0.866
    w = rng.uniform(8, 15)
    return w, w * rng.uniform(0.7, 1.0)


def _style(rng: random.Random, tw: float, th: float, flow: bool) -> dict:
    right = 150 if flow else 90
    S = min((1220 - 60 - right) / tw, (690 - 115) / th) * rng.uniform(0.8, 1.0)
    floor = rng.randint(640, 690)
    width = tw * S
    x0 = rng.uniform(60, 1220 - right - width)
    grids = [u for u in (5, 6, 8, 10) if u * S >= 55]
    return {"scale": round(S, 3), "x0": round(x0, 1), "floor": floor,
            "bg": rng.choice(BACKGROUNDS), "wall": rng.choice(WALLS), "back": rng.choice(BACKWALLS),
            "water": list(rng.choice(WATERS)), "alpha": rng.randint(48, 66),
            "grid": rng.choice(grids), "dashed": rng.random() < 0.5, "wall_px": rng.choice([5, 6, 7])}


NONE = {}                                       # stats only: rejected attempts by reason


def _none(i):
    NONE[i] = NONE.get(i, 0) + 1
    return None


LAST_ATTEMPTS = [0]                             # stats only; never feeds back into a scene


def build(rng: random.Random) -> dict:
    import questions                            # lazy: questions imports sim
    sc = rng.choice(SCENARIOS)
    tries = 0
    while True:
        tries += 1
        scene = _attempt(rng, sc)
        if scene is None or check(scene):
            continue
        if not questions.feasible(scene):
            continue
        LAST_ATTEMPTS[0] = tries
        return scene


def _r(x, k=2):
    return round(x, k)


def _attempt(rng: random.Random, sc: str) -> dict | None:
    n = rng.randint(3, 6)
    tw = _r(rng.uniform(70, 110), 1)
    th = _r(rng.uniform(46, 62), 1)
    flow = sc in ("fill", "drain")
    pfloat = rng.uniform(0.45, 0.8)
    objs = []
    for i, c in enumerate(rng.sample(list(COLOURS), n)):
        shape = rng.choice(SHAPES)
        w, h = _dims(rng, shape)
        if rng.random() < pfloat:
            rho = rng.choice([0.25, 0.5, 0.75]) + rng.uniform(-0.02, 0.02) if rng.random() < 0.4 \
                else rng.uniform(0.2, 0.8)
        else:
            rho = rng.uniform(1.5, 3.5)
        objs.append({"shape": shape, "w": _r(w), "h": _r(h), "rho": _r(rho, 3), "colour": c,
                     "pattern": rng.choice(PATTERNS)})
    if sum(o["rho"] < 1 for o in objs) < 2:
        return _none(4)
    # left-to-right layout (random order, random gaps)
    room = tw - (8 if flow else 2) - 2
    slack = room - sum(o["w"] for o in objs) - 3.5 * (n - 1)
    if slack < 2:
        return _none(5)
    order = list(range(n))
    rng.shuffle(order)
    wts = [rng.random() + 0.05 for _ in range(n + 1)]
    x = 2.0
    for k, i in enumerate(order):
        x += (3.5 if k else 0.0) + slack * wts[k] / sum(wts)
        objs[i]["x"] = _r(x)
        x += objs[i]["w"]
    scene = {"scenario": sc, "tank": {"w": tw, "h": th}, "flow": None, "objects": objs}
    hmax = max(o["h"] for o in objs)
    if sc in ("drop", "rise"):
        rel_order = list(range(n))
        rng.shuffle(rel_order)
        f = rng.randint(14, 28)
        for i in rel_order:
            objs[i]["release"] = f
            f += rng.randint(12, max(12, min(34, (150 - f) // max(1, n - 1))))
        if sc == "drop":
            L0 = rng.uniform(10, 26)
            scene["water0"] = _r(tw * L0)
            Lf, _ = final_state(scene)
            for o in objs:
                lo, hi = Lf + 3.0, th - o["h"] - 1.5
                if hi < lo:
                    return _none(6)
                o["y0"] = _r(rng.uniform(lo, min(hi, lo + 14)))
        else:
            lo = hmax + 4
            if lo > th - 12:
                return _none(7)
            L0 = rng.uniform(lo, min(th - 10, lo + 14))
            for o in objs:
                o["y0"] = 0.0
            scene["water0"] = _r(tw * L0 - sum(area(o) for o in objs))
    else:
        dur = rng.randint(66, 140)                 # frames the tap / drain runs
        on = rng.randint(12, 26)
        drafts = [d for d in (draft(o) for o in objs) if d is not None]

        def clear(L):                              # no floater or top borderline at level L
            return all(not (d - 0.8 < L < d + 2.4) for d in drafts) and \
                all(abs(o["h"] - L) >= 1.7 for o in objs)

        if sc == "fill":
            L0 = rng.uniform(1.5, 5.0)
            ends = [rng.uniform(max(d + 2.4, 8.0), max(d + 2.4, 8.0) + 12) for d in drafts]
        else:
            L0 = rng.uniform(max([15.0] + [d + 2.4 for d in drafts]), th - 8)
            ends = [rng.uniform(max(2.5, d - 9), d - 0.8) for d in drafts if d - 0.8 > 2.5]
        ends = [e for e in ends if clear(e) and e < th - 8]
        if not ends or not clear(L0):
            return _none(11)
        Lend = rng.choice(ends)
        speed = (Lend - L0) / (dur / FPS)          # cm/s level change (roughly)
        if not 1.5 <= abs(speed) <= 7.0:
            return _none(8)

        def water_for(L):
            disp = 0.0
            for o in objs:
                d = draft(o)
                disp += o["rho"] * area(o) if (d is not None and L >= d) else sub_area(o, L)
            return tw * L - disp

        w0 = _r(water_for(L0))
        rate = _r((water_for(Lend) - w0) / (dur / FPS))
        for o in objs:
            o["release"] = None
        scene["flow"] = {"on": on, "off": on + dur, "rate": rate}
        scene["water0"] = w0
        if scene["water0"] <= 0:
            return _none(9)
    for o in objs:
        if o.get("release") is None:
            o.pop("release", None)
    scene["style"] = _style(rng, tw, th, flow)
    sf = settle_frame(scene)
    sched = [o["release"] for o in objs if o.get("release") is not None]
    if flow:
        sched += [scene["flow"]["on"], scene["flow"]["off"]]
    scene["n_frames"] = max(sf, max(sched)) + HOLD_FRAMES + 1
    if scene["n_frames"] > MAX_FRAMES:
        return _none(10)
    return scene
