"""Conveyor belt sort: objects come out of a tunnel onto a moving belt, pass
a sorting gate and slide down a chute into a tray ("bin"), where each one
glides to the farthest free slot and stops. The machine follows a hidden
rule that the viewer has to work out from the objects it has already
sorted; the clip ends while the last one or two objects are still on the
belt, before they reach the gate.

Scenarios (`scene["scenario"]`):
  size       two bins (upper / lower); small objects go to one, large to the other
  colour     two bins; each colour (2-3 colours) always goes to the same bin
  shape      two bins; each shape (2-3 shapes) always goes to the same bin
  three_way  three bins (upper / middle / lower); one property with three values
             (colour, shape or size) picks the bin
  alternate  two bins; the gate sends objects to the two bins in turn, whatever
             they look like

The other properties (colour, shape, stripes / dots, size) are distractors:
in every clip each of them is contradicted by the objects already sorted
(two objects that share it end up in different bins), and so is taking turns
(except in `alternate`), so exactly one simple rule fits what is shown.

Motion is piecewise linear between whole frames (`keys`): every object
moves at the belt speed v (px / frame) through the tunnel mouth, along the
belt (D_b frames) to the gate, down its chute (D_c frames) and along the
tray to its slot (slot k of a tray: D_in + (cap - 1 - k) * D_slot frames).
Lengths are v * frames, so every event (out of the tunnel, at the gate, at
rest) falls on a whole frame. Slots fill from the far end, so a later object
never passes an earlier one, and objects at least D_slot frames apart never
touch (the slot spacing is wider than any two objects).

Text on screen: only the clock. Objects are named by colour, pattern and
shape (unique in each clip), bins by position.
"""
from __future__ import annotations
import math
import random
from fractions import Fraction as Fr

W, H, FPS = 1280, 720, 24
MAX_FRAMES = 240
SCENARIOS = ("size", "colour", "shape", "three_way", "alternate")
TIMER_BOX = (14, 14, 196, 70)
TIMER_FONT = 36
EDGE = 12                 # frames: every event stays this far from the clip's start and end
CLEAR = 12                # px: geometry stays this far from the clock panel and the frame edges
PEND_GAP = 40             # px: a pending object is at least this far before the gate at the end

COLOURS = {"red": (226, 54, 48), "blue": (34, 118, 230), "green": (46, 170, 74), "yellow": (250, 210, 40),
           "purple": (150, 70, 200), "orange": (250, 130, 20)}
SHAPES = ("circle", "square", "triangle", "star")
PATTERNS = ("plain", "striped", "dotted")
SIZE_SCALE = {"small": Fr(1), "medium": Fr(5, 4), "large": Fr(3, 2)}
EXT = 1.22                # every shape fits in a circle of EXT * r
ATTRS = ("size", "colour", "shape", "pattern")
TWO = ("upper", "lower")
THREE = ("upper", "middle", "lower")

BACKGROUNDS = [  # (top, bottom, light?)
    ((236, 232, 222), (218, 212, 198), True), ((226, 232, 238), (202, 212, 224), True),
    ((230, 236, 226), (210, 218, 204), True), ((28, 32, 44), (44, 50, 66), False),
    ((34, 36, 36), (22, 24, 26), False), ((38, 30, 46), (24, 20, 32), False), ((22, 40, 42), (30, 56, 58), False),
]
BELTS = {True: [(70, 74, 82), (92, 80, 70), (64, 78, 72)], False: [(104, 108, 118), (120, 108, 96), (96, 116, 110)]}
METALS = {True: [(96, 102, 112), (120, 110, 100), (84, 96, 110)], False: [(170, 176, 186), (186, 176, 160), (150, 170, 176)]}


# ---------------------------------------------------------------- clock helpers

def tenths(f: int) -> int:
    return (10 * f) // FPS


def clock_text(k: int) -> str:
    return f"{k // 10}.{k % 10} s"


def frames_showing(k: int) -> tuple[int, int]:
    lo = -(-k * FPS // 10)
    hi = -(-(k + 1) * FPS // 10) - 1
    return lo, hi


# ---------------------------------------------------------------- objects and rules

def ext(o) -> float:
    return EXT * o["r"]


def name(o) -> str:
    return f"the {o['pattern']} {o['colour']} {o['shape']}"


def value(o, attr):
    return o[attr]


def bin_names(scene) -> tuple:
    return THREE if scene["scenario"] == "three_way" else TWO


def rule_bins(rule, objs, order=None) -> list:
    """Bin of each object (in `order` of reaching the gate) under `rule`."""
    order = list(range(len(objs))) if order is None else order
    out = [None] * len(objs)
    if rule["kind"] == "alternate":
        other = {"upper": "lower", "lower": "upper"}
        b = rule["start"]
        for i in order:
            out[i] = b
            b = other[b]
    else:
        for i in order:
            out[i] = rule["map"][str(objs[i][rule["attr"]])]
    return out


def hypotheses(objs, bins, nb) -> list:
    """Simple rules that fit the sorted objects (in gate order): one property
    -> bin, or taking turns through the nb bins."""
    out = []
    for a in ATTRS:
        m = {}
        if all(m.setdefault(o[a], b) == b for o, b in zip(objs, bins)):
            out.append(("attr", a))
    if len(bins) >= nb and len(set(bins[:nb])) == nb and all(bins[i] == bins[i - nb] for i in range(nb, len(bins))):
        out.append(("turns", nb))
    return out


# ---------------------------------------------------------------- kinematics

def lane_frames(scene, b, k, speed=Fr(1)) -> Fr:
    """Frames from the gate to rest in slot k of bin b."""
    fr = scene["frames"]
    return Fr(fr["chute"][b] + fr["in"] + (scene["bins"][b]["cap"] - 1 - k) * fr["slot"]) / speed


def simulate(scene, entries=None, rule=None, speed=Fr(1), drop=()) -> list:
    """[(gate, land, bin, slot)] per object (None if dropped); exact Fractions of
    a frame. Objects reach the gate in entry order; each bin fills from slot 0.
    A bin overflowing its tray gives land None."""
    objs = scene["objects"]
    entries = [o["e"] for o in objs] if entries is None else entries
    rule = scene["rule"] if rule is None else rule
    keep = [i for i in range(len(objs)) if i not in drop]
    order = sorted(keep, key=lambda i: (entries[i], i))
    bins = rule_bins(rule, objs, order)
    fill = {b: 0 for b in scene["bins"]}
    out = [None] * len(objs)
    for i in order:
        g = Fr(entries[i]) + Fr(scene["frames"]["belt"]) / speed
        b = bins[i]
        k = fill[b]
        fill[b] += 1
        land = g + lane_frames(scene, b, k, speed) if k < scene["bins"][b]["cap"] else None
        out[i] = (g, land, b, k)
    return out


def keyframes(scene, o, b, k) -> list:
    v = Fr(scene["v2"], 2)
    d = scene["dir"]
    fr = scene["frames"]
    pre = math.ceil((ext(o) + 6) / v)
    e = o["e"]
    a = e + fr["belt"]
    bn = scene["bins"][b]
    t1 = a + fr["chute"][b]
    land = t1 + fr["in"] + (bn["cap"] - 1 - k) * fr["slot"]
    y = scene["belt_y"]
    return [[e - pre, float(scene["mouth"] - d * v * pre), y], [e, scene["mouth"], y],
            [a, scene["gate"][0], y], [t1] + bn["start"], [land] + bn["slots"][k]]


def pos(keys, f):
    if f <= keys[0][0]:
        return keys[0][1], keys[0][2]
    for (f0, x0, y0), (f1, x1, y1) in zip(keys, keys[1:]):
        if f <= f1:
            if f == f1:
                return x1, y1
            u = (f - f0) / (f1 - f0)
            return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    return keys[-1][1], keys[-1][2]


# ---------------------------------------------------------------- generation

def _style(rng):
    top, bot, light = rng.choice(BACKGROUNDS)
    if rng.random() < 0.4:
        top, bot = bot, top
    return {"bg": [list(top), list(bot)], "light": light, "bg_style": rng.choice(["flat", "gradient", "vignette"]),
            "belt": list(rng.choice(BELTS[light])), "metal": list(rng.choice(METALS[light])),
            "cleat": rng.choice([28, 34, 40, 48]), "outline": rng.choice([3, 4]),
            "tray_tint": rng.choice([0.10, 0.16, 0.22]), "rail": rng.choice([3, 4, 5])}


def _objects(rng, scenario, n_s, n_p):
    """Attributes + rule; (objs, rule) with a unique fitting rule, or None."""
    nb = 3 if scenario == "three_way" else 2
    names = THREE if nb == 3 else TWO
    if scenario == "three_way":
        rattr = rng.choice(["colour", "shape", "size"])
    else:
        rattr = {"size": "size", "colour": "colour", "shape": "shape", "alternate": None}[scenario]
    if rattr == "colour":
        nc = 3 if nb == 3 else rng.randint(2, 3)
    else:
        nc = rng.randint(2, 4)
    if rattr == "shape":
        ns = 3 if nb == 3 else rng.randint(2, 3)
    else:
        ns = rng.randint(1, 3)
    pools = {"colour": rng.sample(sorted(COLOURS), nc), "shape": rng.sample(SHAPES, ns),
             "pattern": rng.sample(PATTERNS, rng.randint(1, 3))}
    if rattr == "size":
        pools["size"] = ["small", "medium", "large"] if nb == 3 else ["small", "large"]
    else:
        pools["size"] = rng.choice([["medium"], ["small", "large"]])
    if rattr is None:
        rule = {"kind": "alternate", "start": rng.choice(TWO)}
    else:
        vals = pools[rattr]
        while True:
            m = {str(v): rng.choice(names) for v in vals} if nb == 2 else dict(zip(vals, rng.sample(names, 3)))
            if len(set(m.values())) == nb:
                break
        rule = {"kind": "attr", "attr": rattr, "map": m}
    n = n_s + n_p
    if len(pools["colour"]) * len(pools["shape"]) * len(pools["pattern"]) < n:
        return None
    for _ in range(60):
        objs, seen = [], set()
        while len(objs) < n:
            o = {a: rng.choice(pools[a]) for a in ATTRS}
            key = (o["colour"], o["pattern"], o["shape"])
            if key not in seen:
                seen.add(key)
                objs.append(o)
        bins = rule_bins(rule, objs)
        sb = bins[:n_s]
        if len(set(sb)) != nb:
            continue
        hyp = hypotheses(objs[:n_s], sb, nb)
        want = [("turns", 2)] if rattr is None else [("attr", rattr)]
        if hyp != want:
            continue
        if rattr and any(o[rattr] not in {s[rattr] for s in objs[:n_s]} for o in objs[n_s:]):
            continue
        if len({o["size"] for o in objs}) != len(pools["size"]):
            continue
        return objs, rule
    return None


def _attempt(rng, scenario, n_s, n_p):
    got = _objects(rng, scenario, n_s, n_p)
    if got is None:
        return None
    objs, rule = got
    r0 = rng.randint(22, 25)
    for o in objs:
        o["r"] = float(round(r0 * SIZE_SCALE[o["size"]] * 2) / 2)
    emax = max(ext(o) for o in objs)
    v2 = rng.randint(12, 20)                       # v = v2 / 2 px per frame (144-240 px/s)
    v = Fr(v2, 2)
    nb = 3 if scenario == "three_way" else 2
    names = THREE if nb == 3 else TWO
    slot = math.ceil((2 * emax + 10) / v) + rng.randint(0, 2)
    din = math.ceil((emax + 8) / v) + rng.randint(0, 2)
    belt_f = rng.randint(max(24, math.ceil(220 / v)), min(60, math.floor(560 / v)))
    bins = rule_bins(rule, objs)
    count = {b: bins.count(b) for b in names}
    cap = {b: count[b] + rng.randint(0, 1) for b in names}
    if max(cap.values()) > 6:
        return None
    # layout in canonical coordinates u (along the belt, left to right) and y
    d = rng.choice([1, -1])
    half = emax + 8                                # half-height of the belt band and of a tray
    mouth_u = rng.randint(max(70, math.ceil(2 * emax + 16)), 130)      # the tunnel runs from the frame edge to here
    gate_u = mouth_u + float(v * belt_f)
    belt_y = rng.randint(300, 420)
    dy = rng.randint(int(2 * half + 24), 250)
    dx = rng.randint(60, 190)
    chute = {}
    start = {}
    ys = {"upper": belt_y - dy, "lower": belt_y + dy, "middle": belt_y}
    for b in names:
        if b == "middle":
            fr = max(8, round(dx / v))
            chute[b] = fr
            start[b] = (gate_u + float(v * fr), belt_y)
        else:
            fr = max(8, round(math.hypot(dx, dy) / v))
            sc = float(v * fr) / math.hypot(dx, dy)
            chute[b] = fr
            start[b] = (gate_u + dx * sc, belt_y + (ys[b] - belt_y) * sc)
    # consecutive objects must stay clear of each other where one has turned into a
    # chute and the next is still on the belt: their distance there is at least
    # (gap * v) * cos(theta / 2), theta the chute's turn away from the belt
    turn = max(math.atan2(abs(start[b][1] - belt_y), start[b][0] - gate_u) for b in names)
    def gap_need(a, b):
        return max(slot, math.ceil((ext(a) + ext(b) + 8) / (float(v) * math.cos(turn / 2))))
    # trays: horizontal, from the chute's end away from the belt
    geo = {}
    for b in names:
        su, sy = start[b]
        slots_u = [su + float(v * (din + (cap[b] - 1 - k) * slot)) for k in range(cap[b])]
        far = slots_u[0] + float(v * slot) / 2 + 6
        near = su - 6
        geo[b] = (su, sy, slots_u, near, far)
        if far > W - CLEAR or sy - half < CLEAR or sy + half > H - CLEAR:
            return None
    if nb == 3 and abs(geo["upper"][1] - geo["middle"][1]) < 2 * half + 20:
        return None
    X = (lambda u: u) if d == 1 else (lambda u: W - u)
    scene = {
        "scenario": scenario, "rule": rule, "dir": d, "v2": v2,
        "frames": {"belt": belt_f, "chute": chute, "in": din, "slot": slot},
        "mouth": float(X(mouth_u)), "tunnel": [X(0), belt_y - half - 10, X(mouth_u), belt_y + half + 10],
        "belt_y": float(belt_y), "half": half, "gate": [float(X(gate_u)), float(belt_y)],
        "bins": {}, "objects": [], "labels": [],
    }
    t = scene["tunnel"]
    scene["tunnel"] = [float(min(t[0], t[2])), float(t[1]), float(max(t[0], t[2])), float(t[3])]
    for b in names:
        su, sy, slots_u, near, far = geo[b]
        x0, x1 = sorted((X(near), X(far)))
        scene["bins"][b] = {"cap": cap[b], "start": [float(X(su)), float(sy)],
                            "slots": [[float(X(u)), float(sy)] for u in slots_u],
                            "rect": [float(x0), float(sy - half), float(x1), float(sy + half)]}
    if _clock_overlap(scene):
        return None
    # timing: sorted objects at least a slot apart; the pending ones late enough to be
    # on the belt, before the gate, when the clip ends
    for i, o in enumerate(objs):
        o["id"] = i
    e = EDGE + math.ceil(emax / v) + rng.randint(0, 10)
    for o in objs[:n_s]:
        o["e"] = e
        if o["id"] < n_s - 1:
            e += gap_need(o, objs[o["id"] + 1]) + rng.randint(0, slot + slot // 2)
    scene["objects"] = objs
    for o in objs[n_s:]:
        o["e"] = 0
    res = simulate(scene, drop=range(n_s, n_s + n_p))
    last_land = max(res[i][1] for i in range(n_s))
    e = max(objs[n_s - 1]["e"] + gap_need(objs[n_s - 1], objs[n_s]), int(last_land) + 1 + EDGE + math.ceil(PEND_GAP / v) - belt_f)
    e += rng.randint(0, 10)
    for o in objs[n_s:]:
        o["e"] = e
        if o["id"] < len(objs) - 1:
            e += gap_need(o, objs[o["id"] + 1]) + rng.randint(0, slot)
    lo = max(int(last_land) + EDGE + 1, max(o["e"] + math.ceil(ext(o) / v) for o in objs[n_s:]) + EDGE + 1)
    hi = min(MAX_FRAMES, objs[n_s]["e"] + belt_f + 1 - math.ceil(PEND_GAP / v))
    if lo > hi:
        return None
    scene["n_frames"] = rng.randint(lo, min(hi, lo + 24))
    res = simulate(scene)
    for o, (g, land, b, k) in zip(objs, res):
        if land is None:
            return None
        o["bin"], o["slot"] = b, k
        o["pending"] = g > scene["n_frames"] - 1
        o["keys"] = keyframes(scene, o, b, k)
    if [o["pending"] for o in objs] != [False] * n_s + [True] * n_p:
        return None
    return scene


def _clock_overlap(scene) -> bool:
    tb = TIMER_BOX
    boxes = [scene["tunnel"]] + [b["rect"] for b in scene["bins"].values()]
    y, h = scene["belt_y"], scene["half"]
    gx = scene["gate"][0]
    boxes.append([min(scene["mouth"], gx), y - h - 30, max(scene["mouth"], gx), y + h + 30])
    for x0, y0, x1, y1 in boxes:
        if x0 < tb[2] + CLEAR and y0 < tb[3] + CLEAR and x1 > tb[0] - CLEAR and y1 > tb[1] - CLEAR:
            return True
        if x0 < 0 or x1 > W or y0 < CLEAR or y1 > H - CLEAR:
            return True
    return False


N_SORTED = {"size": (4, 6), "colour": (4, 6), "shape": (4, 6), "three_way": (5, 7), "alternate": (4, 7)}


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    n_s = rng.randint(*N_SORTED[scenario])
    n_p = rng.choice([1, 1, 2])
    style = _style(rng)
    for attempt in range(1, 400):
        scene = _attempt(rng, scenario, n_s, n_p)
        if scene is not None and not check(scene):
            scene["style"] = style
            scene["attempts"] = attempt
            return scene
    raise RuntimeError(f"no valid {scenario} scene")


# ---------------------------------------------------------------- checks

def events(scene) -> dict:
    """Per object: entry, gate and rest frames inside the clip (None if not)."""
    n = scene["n_frames"]
    out = []
    for o in scene["objects"]:
        k = o["keys"]
        out.append({"e": k[1][0], "gate": k[2][0] if k[2][0] <= n - 1 else None,
                    "land": k[4][0] if k[4][0] <= n - 1 else None})
    return out


def clearance(scene) -> float:
    """Smallest gap (px) between the outlines' bounding circles of any two
    objects on any frame, objects wholly inside the tunnel left out."""
    objs = scene["objects"]
    d = scene["dir"]
    best = 1e9
    for f in range(scene["n_frames"]):
        ps = []
        for o in objs:
            x, y = pos(o["keys"], f)
            if (x - scene["mouth"]) * d > -ext(o):
                ps.append((x, y, ext(o)))
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                best = min(best, math.hypot(ps[i][0] - ps[j][0], ps[i][1] - ps[j][1]) - ps[i][2] - ps[j][2])
    return best


def check(scene) -> list[str]:
    errs = []
    objs = scene["objects"]
    n = scene["n_frames"]
    v = Fr(scene["v2"], 2)
    if not 1 <= n <= MAX_FRAMES:
        errs.append("clip length")
    keys = {(o["colour"], o["pattern"], o["shape"]) for o in objs}
    if len(keys) != len(objs):
        errs.append("two objects look the same")
    if any(o["r"] < 10 for o in objs):
        errs.append("object too small")
    res = simulate(scene)
    for o, (g, land, b, k) in zip(objs, res):
        if land is None or o["bin"] != b or o["slot"] != k or o["keys"] != keyframes(scene, o, b, k):
            errs.append(f"object {o['id']} does not re-simulate")
    srt = sorted(range(len(objs)), key=lambda i: objs[i]["e"])
    if srt != list(range(len(objs))):
        errs.append("objects not in entry order")
    for a, b in zip(objs, objs[1:]):
        if b["e"] - a["e"] < max(scene["frames"]["slot"], 10):
            errs.append("objects too close on the belt")
        if v * (b["e"] - a["e"]) < ext(a) + ext(b) + 10:
            errs.append("objects touch on the belt")
    sp = float(v * scene["frames"]["slot"])
    if sp < 2 * max(ext(o) for o in objs) + 10:
        errs.append("slots too narrow")
    if objs[0]["e"] - math.ceil(ext(objs[0]) / v) < EDGE:
        errs.append("first object comes out too early")
    ev = events(scene)
    pend = [o for o in objs if o["pending"]]
    done = [o for o in objs if not o["pending"]]
    if not pend or not done:
        errs.append("needs sorted and pending objects")
    for o, e in zip(objs, ev):
        if o["pending"]:
            if e["gate"] is not None or (o["keys"][2][0] - (n - 1)) * v < PEND_GAP:
                errs.append(f"pending object {o['id']} too near the gate")
            if e["e"] + math.ceil(ext(o) / v) > n - 1 - EDGE:
                errs.append(f"pending object {o['id']} comes out too late")
        elif e["land"] is None or e["land"] > n - 1 - EDGE:
            errs.append(f"object {o['id']} lands too late")
    nb = len(bin_names(scene))
    sb = [o["bin"] for o in done]
    hyp = hypotheses(done, sb, nb)
    want = [("turns", 2)] if scene["rule"]["kind"] == "alternate" else [("attr", scene["rule"]["attr"])]
    if hyp != want:
        errs.append(f"the sorted objects fit {hyp}, not just {want}")
    if len(set(sb)) != nb:
        errs.append("a bin gets nothing before the end")
    if scene["rule"]["kind"] == "attr":
        a = scene["rule"]["attr"]
        if any(o[a] not in {s[a] for s in done} for o in pend):
            errs.append("a pending object's value was never sorted")
    if _clock_overlap(scene):
        errs.append("geometry overlaps the clock or leaves the frame")
    if clearance(scene) < 6:
        errs.append("two objects come too close")
    if scene["labels"]:
        errs.append("unexpected labels")
    return errs
