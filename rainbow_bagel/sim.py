"""Disassembled Rainbow Bagel (after Jane Street, May 2019): a bagel of
concentric rings is cut into equal slices. The slices lift off a wave at a
time, all turn to the same upright arch shape, glide among each other, pass
completely behind an opaque screen and come to rest in the trays below. All
pieces of one ring look exactly alike, so a piece's original slice can only be
known by following it.

One piece turns gold MID-FLIGHT, at least GOLD_AWAY px from where it was cut
(never in the first frame), so its origin has to be traced back through the
look-alikes.

Every clip: 3-4 rings cut into quarters, two trays (the original setup).
Scenarios (scene["scenario"]):
    wide_hole    hole radius 2 units instead of 1 (3 rings)
    one_by_one   pieces lift off one at a time
    bunches      pieces lift off in bunches of 3-4
    long_hide    pieces stay completely behind the screen ~1 s instead of ~0.6 s
Across all scenarios the gold piece comes from the innermost or the outermost
ring (scene["gold_ring"]).

Ring i (1 = innermost) runs from radius r0+i-1 to r0+i units, where r0 is the
hole radius; one unit is one tick on the (unnumbered) ruler. A piece of ring i
cut into n slices has area pi * ((r0+i)^2 - (r0+i-1)^2) / n.

Everything is planned in whole frames (24 fps); `state(scene, i, f)` gives a
piece's centroid and rotation on frame f, and `poly` its drawn outline."""
from __future__ import annotations
import math
import random

import numpy as np

W, H, FPS = 1280, 720, 24
MAX_FRAMES = 240                       # 10 s
SCR = (20, 300, 1260, 452)             # opaque screen (x0, y0, x1, y1)
TRAY_Y = (470, 712)
TRAY_X = {2: {"left": (36, 624), "right": (656, 1244)}}
TIMER_BOX = (14, 12, 168, 60)
CUT0, CUT1 = 14, 24                    # whole bagel until CUT0, slices part until CUT1
SPREAD = 5.0                           # px of radial gap per ring index once cut
LIFT_F, LIFT_OUT = 8, 14.0
EDGE = 12                              # events stay >= 0.5 s from both ends
MIN_HIDE = 13                          # frames each piece is completely behind the screen
SEP_F, SEP_X = 8, 90                   # look-alikes behind the screen at once: entries/exits apart
GAP = 14                               # px between pieces resting in a tray
GOLD_AWAY = 150.0                      # px from its cut position when a piece turns gold
MAX_TRACK_FPS = 8                      # following a piece among look-alikes must work at <= 8 fps
N_ARC = 24

SLICES = {4: [(45, "top-right"), (135, "top-left"), (225, "bottom-left"), (315, "bottom-right")]}
UNIT = {4: "quarter"}
SCENARIOS = ("wide_hole", "one_by_one", "bunches", "long_hide")
# per scenario: hole radius, ring counts, lift-off bunch sizes, frames between bunches, frames fully hidden
PLAN = {
    "wide_hole": (2, (3,), (1, 2, 2, 3, 3, 4), (9, 16), (14, 18)),
    "one_by_one": (1, (3, 4), (1,), (6, 8), (14, 18)),
    "bunches": (1, (3, 4), (3, 4), (12, 20), (14, 18)),
    "long_hide": (1, (3, 4), (2, 3, 3, 4), (9, 12), (22, 28)),
}

GOLD = (232, 178, 30)
GOLD_EDGE = (140, 98, 8)
THEMES = [  # bg, ink (ruler, tray edges), tray fill, clock panel, clock text
    {"name": "cream", "bg": (246, 243, 236), "ink": (44, 44, 50), "tray": (230, 222, 206), "panel": (255, 255, 255), "clock": (20, 20, 24)},
    {"name": "mist", "bg": (236, 242, 248), "ink": (38, 48, 62), "tray": (216, 226, 238), "panel": (255, 255, 255), "clock": (20, 20, 24)},
    {"name": "sage", "bg": (238, 245, 236), "ink": (40, 54, 42), "tray": (218, 232, 214), "panel": (255, 255, 255), "clock": (20, 20, 24)},
    {"name": "lilac", "bg": (243, 238, 247), "ink": (52, 42, 62), "tray": (228, 218, 238), "panel": (255, 255, 255), "clock": (20, 20, 24)},
    {"name": "stone", "bg": (232, 232, 230), "ink": (40, 40, 40), "tray": (212, 212, 206), "panel": (255, 255, 255), "clock": (20, 20, 24)},
    {"name": "night", "bg": (34, 38, 46), "ink": (226, 230, 236), "tray": (58, 64, 78), "panel": (12, 14, 18), "clock": (245, 245, 245)},
    {"name": "cocoa", "bg": (48, 40, 36), "ink": (238, 228, 214), "tray": (76, 64, 56), "panel": (14, 10, 8), "clock": (245, 245, 245)},
]
MONO = {"teal": (52, 146, 140), "blue": (58, 108, 204), "purple": (128, 78, 172), "green": (64, 150, 76),
        "crimson": (192, 56, 72), "slate": (96, 112, 136), "pink": (214, 92, 150)}
RAINBOW = [(212, 58, 58), (66, 160, 72), (38, 160, 196), (60, 104, 212), (140, 78, 182), (204, 66, 160)]
SCREENS = {"walnut": (120, 88, 62), "slate": (70, 80, 96), "pine": (52, 88, 66), "plum": (92, 58, 90),
           "navy": (38, 56, 100)}
SCREEN_PATTERNS = ("hstripe", "vstripe", "plain", "diag", "brick")
PATTERNS = ("plain", "sesame", "speckle", "band")


def clock_tenths(f: int) -> int:
    """Clock reading (tenths of a second) shown on frame f."""
    return (f * 10) // FPS


def clock_text(f: int) -> str:
    t = clock_tenths(f)
    return f"{t // 10}.{t % 10} s"


# ---------------------------------------------------------------- geometry
def sector(r0: int, u: float, ring: int, n: int):
    """(r, R, half-angle in rad, centroid distance from the circle centre), rounded."""
    r, R = (r0 + ring - 1) * u, (r0 + ring) * u
    half = math.pi / n
    d = (2 / 3) * (R ** 3 - r ** 3) / (R ** 2 - r ** 2) * math.sin(half) / half
    return round(r, 4), round(R, 4), round(half, 8), round(d, 4)


def piece_area(r0: int, ring: int, n: int) -> float:
    return math.pi * ((r0 + ring) ** 2 - (r0 + ring - 1) ** 2) / n


def upright_box(p) -> tuple[float, float, float, float]:
    """Bounding box (left, top, right, bottom) of an upright piece relative to its centroid."""
    s = math.sin(p["half"])
    return -p["R"] * s, p["d"] - p["R"], p["R"] * s, p["d"] - p["r"] * math.cos(p["half"])


def local_pts(p) -> np.ndarray:
    """Outline of the upright piece (mid-angle pointing up), relative to its centroid, image coords."""
    a = math.pi / 2 + np.linspace(-p["half"], p["half"], N_ARC + 1)
    outer = np.stack([p["R"] * np.cos(a), p["d"] - p["R"] * np.sin(a)], 1)
    inner = np.stack([p["r"] * np.cos(a[::-1]), p["d"] - p["r"] * np.sin(a[::-1])], 1)
    return np.concatenate([outer, inner])


def place(pts: np.ndarray, x: float, y: float, rot_deg: float) -> np.ndarray:
    """Rotate local points CCW (as seen on screen) by rot_deg about the centroid, then move to (x, y)."""
    th = math.radians(rot_deg)
    c, s = math.cos(th), math.sin(th)
    return np.stack([pts[:, 0] * c + pts[:, 1] * s + x, -pts[:, 0] * s + pts[:, 1] * c + y], 1)


def poly(scene, i: int, f: int) -> np.ndarray:
    x, y, rot = state(scene, i, f)
    return place(local_pts(scene["pieces"][i]), x, y, rot)


# ---------------------------------------------------------------- motion
def _smooth(x: float) -> float:
    return x * x * (3 - 2 * x)


def _norm180(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def rest_xy(scene, p, spread: float):
    b = scene["bagel"]
    m = math.radians(p["mid"])
    dist = p["d"] + spread
    return b["cx"] + dist * math.cos(m), b["cy"] - dist * math.sin(m)


def phases(p) -> tuple[int, int, int, int]:
    """Frames at which the glide, the screen pass, the settle and the rest start."""
    g = p["lift"] + LIFT_F
    h = g + p["gf"]
    s = h + p["hf"]
    return g, h, s, s + p["sf"]


def state(scene, i: int, f: int) -> tuple[float, float, float]:
    """(centroid x, centroid y, rotation in degrees) of piece i on frame f."""
    p = scene["pieces"][i]
    base = _norm180(p["mid"] - 90.0)
    if f < p["lift"]:
        k = min(1.0, max(0.0, (f - CUT0) / (CUT1 - CUT0)))
        x, y = rest_xy(scene, p, SPREAD * p["ring"] * k)
        return x, y, base
    g, h, s, land = phases(p)
    if f < g:
        e = _smooth((f - p["lift"]) / LIFT_F)
        x, y = rest_xy(scene, p, SPREAD * p["ring"] + LIFT_OUT * e)
        return x, y, base * (1 - e)
    if f < h:
        tau = (f - g) / p["gf"]
        x0, y0 = rest_xy(scene, p, SPREAD * p["ring"] + LIFT_OUT)
        cx, cy = p["ctrl"]
        a, b, c = (1 - tau) ** 2, 2 * (1 - tau) * tau, tau * tau
        return a * x0 + b * cx + c * p["x_in"], a * y0 + b * cy + c * p["y_in"], 0.0
    if f < s:
        tau = (f - h) / p["hf"]
        return p["x_in"] + (p["x_out"] - p["x_in"]) * tau, p["y_in"] + (p["y_out"] - p["y_in"]) * tau, 0.0
    if f < land:
        tau = (f - s) / p["sf"]
        e = 1 - (1 - tau) ** 2
        cx, cy = p["ctrl2"]
        a, b, c = (1 - e) ** 2, 2 * (1 - e) * e, e * e
        return a * p["x_out"] + b * cx + c * p["slot"][0], a * p["y_out"] + b * cy + c * p["slot"][1], 0.0
    return p["slot"][0], p["slot"][1], 0.0


def is_gold(scene, i: int, f: int) -> bool:
    return i == scene["marked"] and f >= scene["gold_frame"]


def tracks(scene) -> np.ndarray:
    """(n_frames, n_pieces, 2) centroid positions."""
    nf, n = scene["n_frames"], len(scene["pieces"])
    out = np.zeros((nf, n, 2))
    for i in range(n):
        for f in range(nf):
            x, y, _ = state(scene, i, f)
            out[f, i] = (x, y)
    return out


def hidden_mask(scene, P: np.ndarray | None = None) -> np.ndarray:
    """(n_frames, n_pieces) True where a piece is completely behind the screen.
    Pieces are upright whenever they are near the screen, so the box test is exact."""
    P = tracks(scene) if P is None else P
    out = np.zeros(P.shape[:2], bool)
    fr = np.arange(P.shape[0])
    for i, p in enumerate(scene["pieces"]):
        l, t, r, b = upright_box(p)
        g, h, s, land = phases(p)
        x, y = P[:, i, 0], P[:, i, 1]
        out[:, i] = ((fr >= h) & (fr < s) & (y + t >= SCR[1]) & (y + b <= SCR[3])
                     & (x + l >= SCR[0]) & (x + r <= SCR[2]))
    return out


# ---------------------------------------------------------------- tray packing
def pack(scene) -> bool:
    """Rows from the tray floor up, pieces placed left to right in landing order.
    Writes p["slot"] (resting centroid) and p["row"]. False if a tray overflows."""
    pcs = scene["pieces"]
    row_h = max(upright_box(p)[3] - upright_box(p)[1] for p in pcs) + 12
    for tray, (x0, x1) in TRAY_X[scene["n_trays"]].items():
        mine = sorted((p for p in pcs if p["tray"] == tray), key=lambda p: (p["land"], p["ring"], p["slice"]))
        x, row = x0 + 12.0, 0
        for p in mine:
            l, t, r, b = upright_box(p)
            if x + (r - l) > x1 - 12:
                x, row = x0 + 12.0, row + 1
            feet = TRAY_Y[1] - 10 - row * row_h
            if feet - row_h + 12 < TRAY_Y[0] + 6:
                return False
            p["slot"] = [round(x - l, 3), round(feet - b, 3)]
            p["row"] = row
            x += (r - l) + GAP
    return True


# ---------------------------------------------------------------- build


def _style(rng, k):
    th = rng.choice(THEMES)
    seen = lambda c: _far(c, th["bg"], 150) and _far(c, th["tray"], 150)   # noqa: E731
    if rng.random() < 0.4:
        runs = [[RAINBOW[(start + step * j) % len(RAINBOW)] for j in range(k)]
                for start in range(len(RAINBOW)) for step in (1, -1)]
        cols = rng.choice([r for r in runs if all(seen(c) for c in r)])
        mode, name = "rainbow", "rainbow"
    else:
        name = rng.choice([m for m in sorted(MONO) if seen(MONO[m])])
        cols, mode = [MONO[name]] * k, "mono"
    seeds = []
    for ring in range(k):
        seeds.append([[round(rng.uniform(0.25, 0.75), 3), round(rng.uniform(0.12, 0.88), 3),
                       round(rng.uniform(0, 180), 1)] for _ in range(3 + 2 * ring)])
    return {"theme": th["name"], "palette": mode, "colour": name, "ring_colours": [list(c) for c in cols],
            "pattern": rng.choice(PATTERNS), "seeds": seeds,
            "screen": rng.choice([k for k in sorted(SCREENS) if _far(SCREENS[k], th["bg"]) and _far(SCREENS[k], GOLD)]),
            "screen_pattern": rng.choice(SCREEN_PATTERNS), "outline": 3}


def _far(c1, c2, d: int = 120) -> bool:
    return sum(abs(a - b) for a, b in zip(c1, c2)) >= d


def theme(scene) -> dict:
    return next(t for t in THEMES if t["name"] == scene["style"]["theme"])


def _ruler(rng, cx, R_out, u, r0, k):
    n = r0 + k + rng.choice([1, 2])
    length = n * u
    y = round(rng.uniform(112, 206), 1)
    rooms = {"left": (30, cx - R_out - 50 - length), "right": (cx + R_out + 50, W - 30 - length)}
    side = rng.choice([s for s, (a, b) in rooms.items() if b >= a])
    a, b = rooms[side]
    return {"x0": round(rng.uniform(a, b), 1), "y": y, "step": u, "n": n, "side": side}


def ruler_box(ru) -> tuple[float, float, float, float]:
    return ru["x0"] - 10, ru["y"] - 26, ru["x0"] + ru["n"] * ru["step"] + 10, ru["y"] + 16


def _waves(rng, order, sizes):
    waves, j = [], 0
    while j < len(order):
        m = min(len(order) - j, rng.choice(sizes))
        waves.append(order[j:j + m])
        j += m
    return waves


def _sep_ok(p, others) -> bool:
    """Look-alikes behind the screen at the same time must enter and exit apart (in time or place)."""
    a0, a1 = phases(p)[1], phases(p)[2]
    for q in others:
        if q["ring"] != p["ring"]:
            continue
        b0, b1 = phases(q)[1], phases(q)[2]
        if a1 < b0 or b1 < a0:
            continue
        if abs(a0 - b0) < SEP_F and abs(p["x_in"] - q["x_in"]) < SEP_X:
            return False
        if abs(a1 - b1) < SEP_F and abs(p["x_out"] - q["x_out"]) < SEP_X:
            return False
    return True


def _glide_ok(scene, p) -> bool:
    """Quick test of the glide path (pieces are upright while gliding) against the frame, clock and ruler."""
    l, t, r, b = upright_box(p)
    x0, y0 = rest_xy(scene, p, SPREAD * p["ring"] + LIFT_OUT)
    vx = (p["x_out"] - p["x_in"]) / p["hf"]
    vy = (p["y_out"] - p["y_in"]) / p["hf"]
    cx, cy = p["x_in"] - vx * p["gf"] / 2, p["y_in"] - vy * p["gf"] / 2
    boxes = [TIMER_BOX, ruler_box(scene["ruler"])]
    for j in range(p["gf"] + 1):
        tau = j / p["gf"]
        a, bb, c = (1 - tau) ** 2, 2 * (1 - tau) * tau, tau * tau
        x, y = a * x0 + bb * cx + c * p["x_in"], a * y0 + bb * cy + c * p["y_in"]
        if x + l < 3 or x + r > W - 3 or y + t < 3:
            return False
        for bx in boxes:
            if x + r > bx[0] - 4 and x + l < bx[2] + 4 and y + b > bx[1] - 4 and y + t < bx[3] + 4:
                return False
    return True


def _try_build(rng, scenario, gold_ring):
    r0, ks, sizes, gaps, hides = PLAN[scenario]
    k, n, n_trays = rng.choice(ks), 4, 2
    R_out = round(rng.uniform(max(96.0, 20.6 * (r0 + k)), 108), 2)
    u = round(R_out / (r0 + k), 4)
    cx = round(rng.uniform(470, 810), 1)
    scene = {"scenario": scenario, "slices": n, "rings": k, "hole": r0, "n_trays": n_trays,
             "trays": list(TRAY_X[n_trays]), "bagel": {"cx": cx, "cy": 158.0, "R": R_out, "u": u},
             "gold_ring": gold_ring, "style": _style(rng, k)}
    scene["ruler"] = _ruler(rng, cx, R_out, u, r0, k)
    pcs = []
    for ring in range(1, k + 1):
        for s, (mid, _) in enumerate(SLICES[n]):
            r, R, half, d = sector(r0, u, ring, n)
            pcs.append({"ring": ring, "slice": s, "mid": mid, "r": r, "R": R, "half": half, "d": d})
    order = list(range(len(pcs)))
    rng.shuffle(order)
    trays = scene["trays"]
    tr = [trays[j % n_trays] for j in range(2 * n_trays)] + [rng.choice(trays) for _ in range(len(pcs) - 2 * n_trays)]
    rng.shuffle(tr)
    for j, i in enumerate(order):
        pcs[i]["tray"] = tr[j]
    f = CUT1 + rng.randint(6, 12)
    for w in _waves(rng, order, sizes):
        for i in w:
            p = pcs[i]
            l, t, r_, b = upright_box(p)
            h = b - t
            hide = rng.randint(*hides)
            p["lift"] = f
            p["gf"] = rng.randint(18, 30)
            p["hf"] = math.ceil(hide * ((SCR[3] - SCR[1]) + 4 + h) / ((SCR[3] - SCR[1]) - h)) + 1
            p["sf"] = rng.randint(12, 18)
            p["land"] = phases(p)[3]
        f += rng.randint(*gaps)
    if max(p["land"] for p in pcs) + 24 > MAX_FRAMES:
        return None, "too long"
    scene["pieces"] = pcs
    if not pack(scene):
        return None, "tray overflow"
    placed = []
    for i in order:   # screen pass: exit inside the destination tray, entry shifted by a drift
        p = pcs[i]
        l, t, r_, b = upright_box(p)
        x0, x1 = TRAY_X[n_trays][p["tray"]]
        p["y_in"] = round(SCR[1] - 2 - b, 3)
        p["y_out"] = round(SCR[3] + 2 - t, 3)
        for _ in range(40):
            x_out = rng.uniform(x0 + 10 - l, x1 - 10 - r_)
            x_in = x_out - rng.choice([-1, 1]) * rng.uniform(40, 240)
            if not (SCR[0] + 8 - l <= x_in <= SCR[2] - 8 - r_):
                continue
            p["x_in"], p["x_out"] = round(x_in, 2), round(x_out, 2)
            if _sep_ok(p, [pcs[j] for j in placed]) and _glide_ok(scene, p):
                break
        else:
            return None, "screen pass"
        placed.append(i)
    for p in pcs:   # tangent-continuous control points into and out of the screen pass
        vx = (p["x_out"] - p["x_in"]) / p["hf"]
        vy = (p["y_out"] - p["y_in"]) / p["hf"]
        p["ctrl"] = [round(p["x_in"] - vx * p["gf"] / 2, 3), round(p["y_in"] - vy * p["gf"] / 2, 3)]
        p["ctrl2"] = [round(p["x_out"] + vx * p["sf"] / 4, 3), round(p["y_out"] + vy * p["sf"] / 4, 3)]
    scene["n_frames"] = max(p["land"] for p in pcs) + rng.randint(24, 30)
    if scene["n_frames"] > MAX_FRAMES:
        return None, "too long"
    # the gold piece: preferably one that shares the air with a look-alike; turns gold mid-glide
    P = tracks(scene)
    cands = []
    gold_ring = 1 if scene["gold_ring"] == "inner" else k
    for i, p in enumerate(pcs):
        if p["ring"] != gold_ring:
            continue
        g, h, _, _ = phases(p)
        rest = np.array(rest_xy(scene, p, SPREAD * p["ring"]))
        frames = [fr for fr in range(g + 12, h - 5) if math.hypot(*(P[fr, i] - rest)) >= GOLD_AWAY]
        if not frames:
            continue
        mates = [q for q in pcs if q is not p and q["ring"] == p["ring"] and q["lift"] <= frames[-1]
                 and q["land"] > p["lift"]]
        cands.append((i, frames, bool(mates)))
    if not cands:
        return None, "no gold candidate"
    rng.shuffle(cands)
    cands.sort(key=lambda c: not c[2])          # pieces sharing the air with a look-alike first
    hid = hidden_mask(scene, P)
    for i, frames, _ in cands:
        gf = rng.choice(frames)
        tf = track_fps(scene, i, pcs[i]["lift"], gf, P, hid)
        if tf is not None and tf <= MAX_TRACK_FPS:
            break
    else:
        return None, "gold not trackable"
    scene["marked"] = i
    scene["gold_frame"] = gf
    scene["text"] = {"timer": {"box": list(TIMER_BOX), "font_px": 34, "format": "<seconds, 1 decimal> s"},
                     "labels": []}
    return scene, None


def build(rng: random.Random) -> dict:
    """Draws the scenario and the gold ring once, then retries the rest until the scene passes
    check(); the number of tries is kept in the scene."""
    scenario, gold_ring = rng.choice(SCENARIOS), rng.choice(["inner", "outer"])
    tries, why = 0, {}
    while True:
        tries += 1
        scene, reason = _try_build(rng, scenario, gold_ring)
        if scene is not None:
            errs = check(scene)
            if not errs:
                scene["build_tries"] = tries
                scene["retry_reasons"] = dict(sorted(why.items()))
                return scene
            reason = "check: " + errs[0].split(":")[0]
        why[reason] = why.get(reason, 0) + 1


# ---------------------------------------------------------------- tracking
FPS_LADDER = (1, 2, 3, 4, 6, 8, 12, 24)


def track_fps(scene, i: int, a: int, b: int, P=None, hid=None):
    """Smallest sampling rate at which nearest-neighbour matching follows piece i from
    frame a to frame b without mixing it up with a look-alike (same ring), for every
    sampling phase: between two samples the piece must move less than 0.6x the distance
    from its old place to any visible look-alike. Sample pairs that straddle a pass behind
    the screen are covered by the entry/exit separation rule instead. None if 24 fps fails."""
    P = tracks(scene) if P is None else P
    hid = hidden_mask(scene, P) if hid is None else hid
    mates = [j for j, q in enumerate(scene["pieces"]) if j != i and q["ring"] == scene["pieces"][i]["ring"]]
    for fps in FPS_LADDER:
        step = FPS // fps
        ok = True
        for off in range(step):
            fr = sorted(set([a, b] + list(range(a + (off - a) % step, b + 1, step))))
            for f1, f2 in zip(fr, fr[1:]):
                if hid[f1:f2 + 1, i].any():
                    continue
                own = math.hypot(*(P[f2, i] - P[f1, i]))
                if any(not hid[f2, j] and own >= 0.6 * math.hypot(*(P[f2, j] - P[f1, i])) for j in mates):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return fps
    return None


# ---------------------------------------------------------------- check
def check(scene: dict) -> list[str]:
    errs = []
    pcs, n, k = scene["pieces"], scene["slices"], scene["rings"]
    r0, ks, sizes, gaps, hides = PLAN[scene["scenario"]]
    if n != 4 or scene["n_trays"] != 2 or k not in ks or scene["hole"] != r0:
        errs.append("setup: not 3-4 rings in quarters with two trays as the scenario says")
    if pcs[scene["marked"]]["ring"] != (1 if scene["gold_ring"] == "inner" else k):
        errs.append("gold: piece is not from the ring the scene says")
    if len({p["lift"] for p in pcs}) != len(pcs) and scene["scenario"] == "one_by_one":
        errs.append("setup: two pieces lift off together in a one-by-one clip")
    if sorted((p["ring"], p["slice"]) for p in pcs) != [(i, s) for i in range(1, k + 1) for s in range(n)]:
        errs.append("pieces are not exactly the slices of every ring")
    if scene["bagel"]["u"] < 20:
        errs.append("rings narrower than 20 px")
    if any(p["land"] != phases(p)[3] for p in pcs):
        errs.append("a landing frame doesn't match the flight plan")
    copy = {"n_trays": scene["n_trays"], "pieces": [dict(p) for p in pcs]}
    if not pack(copy) or [(p["slot"], p["row"]) for p in copy["pieces"]] != [(p["slot"], p["row"]) for p in pcs]:
        errs.append("tray layout doesn't match a re-pack")
    nf = scene["n_frames"]
    if nf > MAX_FRAMES:
        errs.append("clip longer than 10 s")
    if min(p["lift"] for p in pcs) < EDGE or max(p["land"] for p in pcs) > nf - 1 - EDGE:
        errs.append("an event is within 0.5 s of the clip's start or end")
    P = tracks(scene)
    hid = hidden_mask(scene, P)
    for i in range(len(pcs)):
        idx = np.flatnonzero(hid[:, i])
        if len(idx) < MIN_HIDE:
            errs.append(f"hide: piece {i} completely hidden for only {len(idx)} frames")
        elif idx[-1] - idx[0] + 1 != len(idx):
            errs.append(f"hide: piece {i} hides more than once")
    full = hid.sum(0)
    if scene["scenario"] == "long_hide" and full.min() < 21:
        errs.append(f"hide: long_hide clip with a piece hidden only {full.min()} frames")
    if scene["scenario"] != "long_hide" and full.max() > 20:
        errs.append(f"hide: piece hidden {full.max()} frames outside long_hide")
    boxes = [TIMER_BOX, ruler_box(scene["ruler"])]
    for i, p in enumerate(pcs):   # exact outlines on every frame: inside the frame, off the clock and ruler
        loc = local_pts(p)
        for f in range(nf):
            x, y, rot = state(scene, i, f)
            pts = place(loc, x, y, rot)
            lo, hi = pts.min(0), pts.max(0)
            if lo[0] < 1 or lo[1] < 1 or hi[0] > W - 1 or hi[1] > H - 1:
                errs.append(f"frame: piece {i} leaves the frame at {f}")
                break
            if any(hi[0] > bx[0] - 2 and lo[0] < bx[2] + 2 and hi[1] > bx[1] - 2 and lo[1] < bx[3] + 2
                   for bx in boxes):
                errs.append(f"clock/ruler: piece {i} covers the clock or the ruler at {f}")
                break
    for i, p in enumerate(pcs):
        if not _sep_ok(p, pcs[:i]):
            errs.append(f"lookalikes: piece {i} enters/exits the screen too close to a look-alike")
    for tray, (x0, x1) in TRAY_X[scene["n_trays"]].items():
        bxs = []
        for p in pcs:
            if p["tray"] != tray:
                continue
            l, t, r, b = upright_box(p)
            sx, sy = p["slot"]
            bx = (sx + l, sy + t, sx + r, sy + b)
            if bx[0] < x0 or bx[2] > x1 or bx[1] < TRAY_Y[0] or bx[3] > TRAY_Y[1]:
                errs.append(f"tray: a piece rests outside the {tray} tray")
            for o in bxs:
                if bx[0] < o[2] + GAP - 0.5 and o[0] < bx[2] + GAP - 0.5 and bx[1] < o[3] + 6 and o[1] < bx[3] + 6:
                    errs.append(f"tray: two pieces touch in the {tray} tray")
            bxs.append(bx)
        if len(bxs) < 2:
            errs.append(f"tray: the {tray} tray holds fewer than two pieces")
    m = pcs[scene["marked"]]
    g, h, _, _ = phases(m)
    gf = scene["gold_frame"]
    if not (g + 12 <= gf < h - 5):
        errs.append("gold: doesn't turn gold mid-glide")
    rest = np.array(rest_xy(scene, m, SPREAD * m["ring"]))
    if math.hypot(*(P[gf, scene["marked"]] - rest)) < GOLD_AWAY:
        errs.append("gold: turns gold too close to where it was cut")
    tf = track_fps(scene, scene["marked"], m["lift"], gf, P, hid)
    if tf is None or tf > MAX_TRACK_FPS:
        errs.append(f"gold: can't be followed back to the bagel at {MAX_TRACK_FPS} fps")
    th = theme(scene)
    for c in scene["style"]["ring_colours"]:
        if not (_far(c, GOLD, 150) and _far(c, th["bg"], 150) and _far(c, th["tray"], 150)):
            errs.append(f"colour: piece colour {c} too close to gold, the background or the trays")
    scr = SCREENS[scene["style"]["screen"]]
    if not (_far(scr, GOLD) and _far(scr, th["bg"])):
        errs.append(f"colour: screen {scr} too close to gold or the background")
    return errs
