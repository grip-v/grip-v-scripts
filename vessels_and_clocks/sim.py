"""Scene sampler and closed-form flow laws for vessels_and_clocks.

Fluid volume and rate laws (communicating vessels + hourglass vs water
clock). 3-5 glass vessels stand in a row on a bench. Each is a rectangular
column 10 marks tall with its own width; all are equally deep, so the water
in a vessel is level x width. A shared ruler (0-10, numbered on the even
marks) stands at one end of the row and unnumbered gridlines run across
the scene at every mark. Each vessel does one thing:

  tap     water pours in at a constant rate q: the level rises in a
          straight line (q / w marks per s) and, once it reaches the top,
          the vessel overflows (water spills down both sides) and stays full
  steady  a drain with a constant outflow q (the hourglass law): the level
          falls in a straight line until the vessel is empty
  leaky   a hole whose outflow is k * sqrt(level) (the water-clock law): the
          level falls fast at first and then slows; sqrt(level) falls in a
          straight line and the vessel empties at 2 w sqrt(h0) / k
  pair    two neighbouring vessels share a pipe at the bottom with a closed
          valve; when the valve opens, water flows from the higher level to
          the lower at K * (difference) and both settle at the shared level
          (w_a h_a + w_b h_b) / (w_a + w_b)

Scenarios (scene["scenario"]):
  taps_drains  3-5 vessels, each a tap, a steady drain or a leaky drain
  drain_race   3-5 drains, at least one of each law
  fill_race    3-5 taps with different widths, levels and flows
  joined_pair  one valve-joined pair plus 1-3 taps / drains
  two_pairs    two valve-joined pairs (valves open at different times),
               plus one tap / drain when there are 5 vessels

Vessels are named by colour (a coloured rim band and plinth, unique per
clip). Widths `w` are inner widths in units of 100 px, so w * 100 is the
exact pixel width. Frame f is shown at t = f / 24 and the clock shows
f * 10 // 24 tenths. Every in-clip event (overflow, run empty, valve
opens) is placed 0.2-0.4 or 0.6-0.8 of the way through a frame interval,
so the frame that first shows it is unambiguous (and so is the one for
twice the time, used by the what-ifs). Pure function of the rng passed in.
"""
from __future__ import annotations
import math
import random

FPS = 24
MIN_FRAMES, MAX_FRAMES = 168, 240        # 7-10 s
H = 10.0                                 # marks
EDGE_S = 0.8                             # in-clip events stay this far inside the clip
OUT_S = 0.3                              # events after the clip are at least this far past the last frame
GAP = 0.3                                # any two in-clip events at least this far apart
SETTLE = 0.01                            # joined vessels differ by at most this at the last frame
SCENARIOS = ("taps_drains", "drain_race", "fill_race", "joined_pair", "two_pairs")
CLOCK_BOX = (18, 16, 214, 84)
LABEL_PX = 36                            # ruler numbers (digits >= 24 px tall)
TAP_TOP = 96                             # nothing is drawn above this (the clock box sits above it)
SPOUT_L = 26                             # drain spout length (px)
SPILL_W = 10                             # overflow streams down the outer walls
RULER_GAP = 72                           # ruler line to the nearest outer wall
FRACS = ((0.2, 0.4), (0.6, 0.8))         # where in a frame interval events are placed

# vessel identity colours per theme (no blues: the water is blue)
PALETTE = {
    "light": {"red": (205, 40, 40), "orange": (235, 125, 15), "yellow": (222, 182, 0), "green": (40, 150, 60),
              "purple": (130, 60, 195), "pink": (228, 80, 170), "brown": (125, 80, 40), "black": (28, 28, 32)},
    "dark": {"red": (245, 70, 70), "orange": (255, 150, 40), "yellow": (245, 220, 60), "green": (80, 215, 100),
             "purple": (180, 125, 255), "pink": (255, 125, 205), "brown": (185, 125, 70), "white": (240, 240, 240)},
}
WATER = {"light": [(60, 140, 225), (40, 120, 210), (70, 155, 235), (45, 130, 200)],
         "dark": [(70, 160, 240), (55, 140, 230), (90, 175, 245), (60, 150, 220)]}
THEMES = {
    "light": {"bg": [(236, 234, 228), (228, 232, 236), (238, 236, 242), (230, 236, 230), (242, 240, 234)],
              "grid": [(178, 176, 170), (170, 176, 186), (182, 178, 190)], "ink": (22, 22, 26),
              "glass": [(120, 130, 145), (100, 110, 125), (135, 140, 150)],
              "bench": [(160, 125, 90), (140, 140, 145), (120, 100, 80), (175, 160, 130)]},
    "dark": {"bg": [(22, 24, 32), (30, 26, 34), (20, 30, 32), (32, 32, 34), (24, 28, 42)],
             "grid": [(78, 82, 96), (86, 80, 98), (72, 92, 92)], "ink": (236, 236, 240),
             "glass": [(170, 180, 195), (150, 160, 175), (190, 195, 205)],
             "bench": [(110, 85, 60), (85, 85, 92), (70, 60, 52), (120, 110, 90)]},
}
ROLE_WORD = {"tap": "overflows", "steady": "runs empty", "leaky": "runs empty"}


# ------------------------------------------------------------------ time
def n_frames(scene) -> int:
    return scene["frames"]


def duration(scene) -> float:
    return scene["frames"] / FPS


def t_last(scene) -> float:
    return (scene["frames"] - 1) / FPS


def clock_tenths(frame: int) -> int:
    return frame * 10 // FPS


def first_frame(t: float) -> int:
    """First frame that shows an event at time t (t is never on a frame instant)."""
    return math.ceil(t * FPS)


def frac_ok(t: float, lo: float = 0.15) -> bool:
    x = t * FPS - math.floor(t * FPS)
    return lo <= x <= 1 - lo


# ------------------------------------------------------------------ flow laws
def partner(scene, i):
    for a, b in scene["pairs"]:
        if i == a:
            return b
        if i == b:
            return a
    return None


def law_time(v) -> float | None:
    """Unbounded time of a tap's overflow or a drain's emptying (None for pair vessels)."""
    if v["role"] == "tap":
        return (H - v["h0"]) * v["w"] / v["q"]
    if v["role"] == "steady":
        return v["h0"] * v["w"] / v["q"]
    if v["role"] == "leaky":
        return 2 * v["w"] * math.sqrt(v["h0"]) / v["k"]
    return None


def solo_level(v, t) -> float:
    if v["role"] == "tap":
        return min(H, v["h0"] + v["q"] * t / v["w"])
    if v["role"] == "steady":
        return max(0.0, v["h0"] - v["q"] * t / v["w"])
    r = math.sqrt(v["h0"]) - v["k"] * t / (2 * v["w"])
    return r * r if r > 0 else 0.0


def pair_level(va, vb, t) -> float:
    """Level of va, joined to vb by a pipe whose valve opens at va['valve']."""
    tv = va["valve"]
    if t <= tv:
        return va["h0"]
    wa, wb = va["w"], vb["w"]
    mean = (wa * va["h0"] + wb * vb["h0"]) / (wa + wb)
    d = (va["h0"] - vb["h0"]) * math.exp(-va["K"] * (1 / wa + 1 / wb) * (t - tv))
    return mean + d * wb / (wa + wb)


def level(scene, i, t) -> float:
    v = scene["vessels"][i]
    if v["role"] == "pair":
        return pair_level(v, scene["vessels"][partner(scene, i)], t)
    return solo_level(v, t)


def outflow(scene, i, t) -> float:
    """Drain outflow (volume / s) at time t, 0 once empty."""
    v = scene["vessels"][i]
    h = level(scene, i, t)
    if h <= 0:
        return 0.0
    return v["q"] if v["role"] == "steady" else v["k"] * math.sqrt(h)


def shared(scene, pair) -> float:
    a, b = (scene["vessels"][k] for k in pair)
    return (a["w"] * a["h0"] + b["w"] * b["h0"]) / (a["w"] + b["w"])


def events(scene) -> list[dict]:
    """In-clip events sorted by time: {'t', 'kind': overflows|runs empty|valve, 'i', ('j' for valves)}."""
    out = []
    tl = t_last(scene)
    for i, v in enumerate(scene["vessels"]):
        te = law_time(v)
        if te is not None and te <= tl:
            out.append({"t": te, "kind": ROLE_WORD[v["role"]], "i": i})
    for a, b in scene["pairs"]:
        out.append({"t": scene["vessels"][a]["valve"], "kind": "valve", "i": a, "j": b})
    return sorted(out, key=lambda e: e["t"])


# ------------------------------------------------------------------ sampling
def _snap(rng, t):
    """Move t inside its frame interval to a spot 0.2-0.4 or 0.6-0.8 of the way through."""
    lo, hi = rng.choice(FRACS)
    return (math.floor(t * FPS) + rng.uniform(lo, hi)) / FPS


def _widths(rng, n):
    lo, hi = {3: (70, 220), 4: (64, 180), 5: (60, 140)}[n]
    for _ in range(200):
        ws = sorted(rng.randint(lo, hi) for _ in range(n))
        if all(b >= 1.12 * a for a, b in zip(ws, ws[1:])):
            rng.shuffle(ws)
            return ws
    return None


def _solo(rng, role, w, tl, want_event):
    """A tap / steady / leaky vessel; its event in the clip (EDGE_S..tl-EDGE_S) or well after it."""
    for _ in range(60):
        if role == "tap":
            h0 = round(rng.uniform(0.8, 8.5), 2)
        elif role == "steady":
            h0 = round(rng.uniform(2.0, 9.5), 2)
        else:
            h0 = round(rng.uniform(3.0, 9.5), 2)
        if want_event:
            te = _snap(rng, rng.uniform(EDGE_S + 0.05, tl - EDGE_S - 0.05))
        else:
            te = rng.uniform(tl + OUT_S + 0.05, tl * (2.6 if role != "leaky" else 1 / 0.36))
        v = {"role": role, "w": w, "h0": h0}
        if role == "tap":
            v["q"] = round((H - h0) * w / te, 5)
            rate = v["q"] / w
            ok = 0.35 <= rate <= 3.0
        elif role == "steady":
            v["q"] = round(h0 * w / te, 5)
            rate = v["q"] / w
            ok = 0.35 <= rate <= 3.0
        else:
            v["k"] = round(2 * w * math.sqrt(h0) / te, 5)
            rate = v["k"] * math.sqrt(h0) / w
            ok = 0.5 <= rate <= 4.0
        if ok:
            return v
    return None


def _pair(rng, va, vb, tl):
    hi, lo = round(rng.uniform(5.5, 9.5), 2), round(rng.uniform(0.5, 4.5), 2)
    if hi - lo < 2.5:
        return False
    if rng.random() < 0.5:
        hi, lo = lo, hi
    va["h0"], vb["h0"] = hi, lo
    tau = rng.uniform(0.3, 0.7)
    need = tau * math.log(abs(hi - lo) / (SETTLE / 2)) + 0.1
    if tl - EDGE_S - need < EDGE_S:
        return False
    tv = _snap(rng, rng.uniform(EDGE_S + 0.05, tl - need))
    K = round(1 / (tau * (1 / va["w"] + 1 / vb["w"])), 5)
    for v in (va, vb):
        v.update(valve=round(tv, 6), K=K)
    return True


def _roles(rng, scenario, n):
    if scenario == "taps_drains":
        roles = [rng.choice(("tap", "steady", "leaky")) for _ in range(n)]
        if "tap" not in roles or not ({"steady", "leaky"} & set(roles)):
            return None, None
        return roles, []
    if scenario == "drain_race":
        roles = [rng.choice(("steady", "leaky")) for _ in range(n)]
        if len(set(roles)) < 2:
            return None, None
        return roles, []
    if scenario == "fill_race":
        return ["tap"] * n, []
    if scenario == "joined_pair":
        p = rng.randrange(n - 1)
        roles = [rng.choice(("tap", "steady", "leaky")) for _ in range(n)]
        roles[p] = roles[p + 1] = "pair"
        return roles, [[p, p + 1]]
    # two_pairs
    if n == 4:
        pairs = [[0, 1], [2, 3]]
    else:
        solo = rng.choice((0, 2, 4))
        rest = [k for k in range(5) if k != solo]
        pairs = [rest[:2], rest[2:]]
    roles = ["pair"] * n
    if n == 5:
        roles[solo] = rng.choice(("tap", "steady", "leaky"))
    return roles, pairs


def _style(rng, theme):
    th = THEMES[theme]
    water = rng.choice(WATER[theme])
    return {
        "theme": theme, "bg": rng.choice(th["bg"]), "bg_pat": rng.choice(("plain", "gradient", "stripes", "dots")),
        "grid": rng.choice(th["grid"]), "grid_style": rng.choice(("solid", "dashed", "dotted")),
        "ink": th["ink"], "glass": rng.choice(th["glass"]), "water": water,
        "water_dark": tuple(max(0, c - 40) for c in water), "bench": rng.choice(th["bench"]),
        "bench_pat": rng.choice(("plain", "grain", "tiles")), "plinth_pat": rng.choice(("plain", "stripes", "dots")),
        "rim": rng.choice(("band", "lip")), "tick_side": rng.choice(("left", "right")),
        "metal": rng.choice(((150, 155, 165), (175, 145, 70), (90, 92, 100))),
        "wall": rng.randint(3, 6), "plinth": rng.randint(20, 28), "bench_h": rng.randint(22, 40),
        "tick_len": rng.randint(8, 11),
    }


def _layout(rng, scene):
    """Pixel layout of the row (scene['layout']); None if it doesn't fit."""
    st, vs = scene["style"], scene["vessels"]
    n = len(vs)
    M = rng.randint(34, 44)
    wall = st["wall"]
    bottom_max = 712 - st["bench_h"] - st["plinth"] - wall
    bottom_min = 178 + 10 * M
    if bottom_min > bottom_max:
        return None
    bottom = rng.randint(bottom_min, bottom_max)
    paired = {k for p in scene["pairs"] for k in p}
    spouts = [None] * n
    for i, v in enumerate(vs):
        if v["role"] in ("steady", "leaky"):
            spouts[i] = rng.choice(("left", "right"))

    def reach(i, side):                                   # how far features stick out of vessel i's wall on a side
        r = 0
        if spouts[i] == side:
            r = max(r, SPOUT_L + 20)
        if vs[i]["role"] == "tap":
            r = max(r, SPILL_W + 4)
        return r

    gaps = []
    for i in range(n - 1):
        g = rng.randint(76, 150)
        if i in paired and partner(scene, i) == i + 1:
            g = rng.randint(70, 130)
        gaps.append(max(g, reach(i, "right") + reach(i + 1, "left") + 30))
    widths = [round(v["w"] * 100) for v in vs]
    row = sum(widths) + 2 * wall * n + sum(gaps)
    ruler_side = rng.choice(("left", "right"))
    out_l = reach(0, "left") + 24
    out_r = reach(n - 1, "right") + 24
    if ruler_side == "left":
        out_l = max(out_l, RULER_GAP + 70)
    else:
        out_r = max(out_r, RULER_GAP + 70)
    slack = 1250 - 30 - out_l - out_r - row
    if slack < 0:
        return None
    x = 30 + out_l + rng.randint(0, slack) + wall
    xs = []
    for i in range(n):
        xs.append([x, x + widths[i]])
        x += widths[i] + 2 * wall + (gaps[i] if i < n - 1 else 0)
    left_wall, right_wall = xs[0][0] - wall, xs[-1][1] + wall
    ruler_x = left_wall - RULER_GAP if ruler_side == "left" else right_wall + RULER_GAP
    return {"M": M, "bottom": bottom, "top": bottom - 10 * M, "x": xs, "spout": spouts,
            "ruler_side": ruler_side, "ruler_x": ruler_x,
            "grid_x": [min(ruler_x, left_wall - 24), max(ruler_x, right_wall + 24)]}


def labels(scene) -> list[dict]:
    """The ruler numbers: 0, 2, ..., 10 beside the ruler, vertically centred on their marks."""
    lay = scene["layout"]
    left = lay["ruler_side"] == "left"
    x = lay["ruler_x"] - 14 if left else lay["ruler_x"] + 14
    return [{"text": str(m), "x": x, "y": lay["bottom"] - m * lay["M"], "anchor": "rm" if left else "lm",
             "size": LABEL_PX} for m in range(0, 11, 2)]


def _draw(rng, scenario):
    n = rng.randint(4, 5) if scenario == "two_pairs" else rng.randint(3, 5)
    frames = rng.randint(MIN_FRAMES, MAX_FRAMES)
    tl = (frames - 1) / FPS
    roles, pairs = _roles(rng, scenario, n)
    if roles is None:
        return None
    ws = _widths(rng, n)
    if ws is None:
        return None
    theme = "light" if rng.random() < 0.55 else "dark"
    names = rng.sample(sorted(PALETTE[theme]), n)
    p_event = {"taps_drains": 0.75, "drain_race": 0.8, "fill_race": 0.75}.get(scenario, 0.7)
    vessels = []
    for i, role in enumerate(roles):
        w = ws[i] / 100
        if role == "pair":
            v = {"role": "pair", "w": w}
        else:
            v = _solo(rng, role, w, tl, rng.random() < p_event)
            if v is None:
                return None
        v["colour"] = names[i]
        vessels.append(v)
    for a, b in pairs:
        if not _pair(rng, vessels[a], vessels[b], tl):
            return None
    vessels = [{k: v[k] for k in sorted(v)} for v in vessels]
    scene = {"scenario": scenario, "frames": frames, "vessels": vessels, "pairs": pairs,
             "style": _style(rng, theme)}
    lay = _layout(rng, scene)
    if lay is None:
        return None
    scene["layout"] = lay
    scene["labels"] = labels(scene)
    return scene


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    for attempt in range(1, 5000):
        scene = _draw(rng, scenario)
        if scene is None or check(scene) or not _ladder_ok(scene):
            continue
        scene["build_attempts"] = attempt
        return scene
    raise RuntimeError("no valid scene")


def _ladder_ok(scene):
    import questions                                   # late import: questions.py imports this module
    try:
        questions.build(scene, random.Random(0))
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------ geometry used by the renderer and checks
def vessel_box(scene, i):
    """Outer box of vessel i incl. walls, rim and plinth."""
    lay, st = scene["layout"], scene["style"]
    x0, x1 = lay["x"][i]
    w = st["wall"]
    return (x0 - w - 4, lay["top"] - 10, x1 + w + 4, lay["bottom"] + w + st["plinth"])


def feature_boxes(scene, i):
    """Boxes of things drawn outside vessel i's walls: spout + its stream, overflow streams, faucet."""
    lay, st = scene["layout"], scene["style"]
    x0, x1 = lay["x"][i]
    w, bot = st["wall"], lay["bottom"]
    bench_top = bot + w + st["plinth"]
    out = []
    side = lay["spout"][i]
    if side:
        if side == "right":
            out.append((x1 + w, bot - 16, x1 + w + SPOUT_L + 20, bench_top))
        else:
            out.append((x0 - w - SPOUT_L - 20, bot - 16, x0 - w, bench_top))
    v = scene["vessels"][i]
    if v["role"] == "tap":
        out.append((x0 - w - SPILL_W - 3, lay["top"] - 12, x0 - w, bench_top))
        out.append((x1 + w, lay["top"] - 12, x1 + w + SPILL_W + 3, bench_top))
        cx = (x0 + x1) / 2
        out.append((cx - 44, TAP_TOP, cx + 14, lay["top"] - 34))
    return out


def label_box(lab, font_mod=None):
    from common.video5 import font
    l, t, r, b = font(lab["size"]).getbbox(lab["text"], anchor=lab["anchor"])
    return (lab["x"] + l, lab["y"] + t, lab["x"] + r, lab["y"] + b)


def _overlap(a, b, pad=0):
    return a[0] - pad < b[2] and b[0] - pad < a[2] and a[1] - pad < b[3] and b[1] - pad < a[3]


# ------------------------------------------------------------------ consistency check
def check(scene: dict) -> list[str]:
    errs = []
    vs, F = scene["vessels"], scene["frames"]
    n = len(vs)
    tl = t_last(scene)
    if not MIN_FRAMES <= F <= MAX_FRAMES:
        errs.append("clip length out of range")
    if scene["scenario"] not in SCENARIOS:
        errs.append("unknown scenario")
    if len({v["colour"] for v in vs}) != n:
        errs.append("two vessels share a colour")
    ws = sorted(v["w"] for v in vs)
    if any(b < 1.12 * a - 1e-9 for a, b in zip(ws, ws[1:])):
        errs.append("two vessels too close in width")
    if any(abs(v["w"] * 100 - round(v["w"] * 100)) > 1e-9 or v["w"] < 0.6 for v in vs):
        errs.append("vessel width not a whole number of px or too narrow")
    # events: inside the clip with margins, or clearly after it
    for i, v in enumerate(vs):
        te = law_time(v)
        if te is None:
            continue
        if te < EDGE_S or tl - EDGE_S < te < tl + OUT_S:
            errs.append(f"vessel {i}: event too near the start or the end")
        if te <= tl and not frac_ok(te):
            errs.append(f"vessel {i}: event too near a frame instant")
        if not 0 < v["h0"] < H:
            errs.append(f"vessel {i}: start level out of range")
        if te > tl:                                       # no event: clearly short of full / empty at the end
            h_end = solo_level(v, tl)
            if (v["role"] == "tap" and h_end > H - 0.3) or (v["role"] != "tap" and h_end < 0.3):
                errs.append(f"vessel {i}: ends too close to full / empty")
        if v["role"] == "leaky":
            h_end = solo_level(v, min(te, tl))
            if h_end > 0.42 * v["h0"]:
                errs.append(f"vessel {i}: water-clock slowing not visible")
    evs = events(scene)
    for e in evs:
        if e["kind"] == "valve" and (e["t"] < EDGE_S or not frac_ok(e["t"])):
            errs.append("valve opens too early or near a frame instant")
    for a, b in zip(evs, evs[1:]):
        if b["t"] - a["t"] < GAP:
            errs.append("two events too close together")
    if any(e["t"] > tl - EDGE_S for e in evs):
        errs.append("an event too near the end")
    for a, b in scene["pairs"]:
        va, vb = vs[a], vs[b]
        if b != a + 1 or va["role"] != "pair" or vb["role"] != "pair":
            errs.append("joined vessels must be neighbours")
        if abs(va["h0"] - vb["h0"]) < 2.5:
            errs.append("joined vessels start too close in level")
        if abs(level(scene, a, tl) - level(scene, b, tl)) > SETTLE:
            errs.append("joined vessels haven't settled by the end")
    sc = scene["scenario"]
    roles = [v["role"] for v in vs]
    solo_ev = [e for e in evs if e["kind"] != "valve"]
    if sc in ("taps_drains", "drain_race", "fill_race") and len(solo_ev) < 2:
        errs.append("fewer than two overflow / empty events")
    if sc == "drain_race" and not {"steady", "leaky"} <= set(roles):
        errs.append("drain race needs both drain laws")
    if sc == "taps_drains" and ("tap" not in roles or not {"steady", "leaky"} & set(roles)):
        errs.append("taps_drains needs a tap and a drain")
    if sc == "joined_pair" and (len(scene["pairs"]) != 1 or not solo_ev):
        errs.append("joined_pair needs one pair and an overflow / empty event")
    if sc == "two_pairs" and len(scene["pairs"]) != 2:
        errs.append("two_pairs needs two pairs")
    errs += _layout_errors(scene)
    return errs


def _layout_errors(scene):
    errs = []
    lay, st = scene["layout"], scene["style"]
    n = len(scene["vessels"])
    if lay["top"] < 176 or lay["bottom"] + st["wall"] + st["plinth"] + st["bench_h"] > 716:
        errs.append("vessels leave the frame vertically")
    boxes = [vessel_box(scene, i) for i in range(n)]
    feats = [feature_boxes(scene, i) for i in range(n)]
    for i in range(n):
        x0, x1 = lay["x"][i]
        if x1 - x0 < 60:
            errs.append("vessel narrower than 60 px")
        for j in range(n):
            if j == i:
                continue
            if j > i and _overlap(boxes[i], boxes[j]):
                errs.append("vessels overlap")
            for f in feats[i]:
                if _overlap(f, boxes[j], 4):
                    errs.append("a spout / spill / faucet touches another vessel")
                for g in feats[j]:
                    if j > i and _overlap(f, g, 4):
                        errs.append("features of two vessels touch")
    things = boxes + [f for fs in feats for f in fs]
    labs = [label_box(lab) for lab in scene["labels"]]
    rx = lay["ruler_x"]
    ruler = (rx - 4, lay["top"] - 2, rx + 4, lay["bottom"] + 2)
    for k, lb in enumerate(labs):
        if lb[3] - lb[1] < 24:
            errs.append("ruler number shorter than 24 px")
        if lb[0] < 8 or lb[2] > 1272:
            errs.append("ruler number off screen")
        if any(_overlap(lb, t, 6) for t in things) or _overlap(lb, ruler, 4) or _overlap(lb, CLOCK_BOX, 6):
            errs.append("ruler number overlaps something")
        if any(_overlap(lb, o, 2) for o in labs[k + 1:]):
            errs.append("ruler numbers overlap")
    if any(_overlap(ruler, t, 10) for t in things):
        errs.append("ruler touches a vessel")
    if any(t[1] < TAP_TOP or _overlap(t, CLOCK_BOX, 6) for t in things):
        errs.append("something reaches the clock area")
    return errs
