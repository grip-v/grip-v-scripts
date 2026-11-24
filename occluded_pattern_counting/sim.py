"""Scene simulation for occluded_pattern_counting v3.

A regular pattern of identical shapes (grid, ring, triangle or honeycomb)
sits still on the floor. One or two opaque occluders hold at a position,
slide to the next, hold again, and so on (2-4 stops). At every hold each
object is either fully visible or fully hidden (never cut by an occluder
edge); objects are only partly covered while an occluder slides. Every slot
of the pattern is uncovered at some hold, so every object (and every empty
slot in the gaps scenario) is seen at least once.

Time is in frames (24 fps); the clock shows floor(f * 10 / 24) tenths.
Everything is a pure function of the seeded rng passed to build().
"""
from __future__ import annotations
import functools
import json
import math

FPS = 24
W, H = 1280, 720
ARENA = (30, 74, 1250, 700)
CLOCK_BOX = (1098, 8, 1268, 58)
SCENARIOS = ("slide", "two_occluders", "gaps", "odd_one")
FAMILIES = ("grid", "ring", "triangle", "honeycomb")
COLORS = {
    "red": (230, 70, 60), "blue": (60, 130, 240), "green": (60, 185, 90), "yellow": (240, 205, 45),
    "purple": (160, 90, 230), "orange": (245, 135, 35), "pink": (240, 95, 175), "cyan": (45, 200, 215),
}
SHAPES = ("circle", "square", "triangle", "star", "diamond", "hexagon")
BACKGROUNDS = {"paper": (238, 235, 226), "mist": (224, 230, 236), "sage": (222, 232, 218), "charcoal": (30, 31, 36),
               "navy": (20, 26, 46)}
OCC_COLORS = {"slate": (84, 92, 104), "brown": (122, 90, 58), "olive": (96, 106, 58), "plum": (104, 70, 104),
              "teal": (40, 104, 112)}
TEXTURES = ("solid", "hatched", "dotted", "checker")
EDGE_F = 12          # every slide starts/ends >= 0.5 s from the clip ends
HOLD_F = (24, 44)    # holds last 1.0-1.8 s
SLIDE_F = (14, 26)   # slides last 0.6-1.1 s
ODD_DIST = 120       # odd colour must stand out from the main colour after downscaling
PAD = 3              # an object counts as clear/hidden only with >= 3 px to spare


# ------------------------------------------------------------------ geometry
def cover(rect, x, y, r):
    """0 = clear of rect, 1 = cut by its edge, 2 = completely hidden (both with PAD px to spare)."""
    x0, y0, x1, y1 = rect
    dx = max(x0 - x, 0, x - x1)
    dy = max(y0 - y, 0, y - y1)
    if math.hypot(dx, dy) >= r + PAD:
        return 0
    if x0 + r + PAD <= x <= x1 - r - PAD and y0 + r + PAD <= y <= y1 - r - PAD:
        return 2
    return 1


def _slots(rng, fam):
    """Unit-free slot positions (centred at 0,0) and spacing 1."""
    if fam == "grid":
        rows, cols = rng.randint(3, 6), rng.randint(3, 7)
        return [((c - (cols - 1) / 2), (r - (rows - 1) / 2)) for r in range(rows) for c in range(cols)]
    if fam == "honeycomb":
        rows, cols = rng.randint(3, 5), rng.randint(3, 6)
        h = math.sqrt(3) / 2
        return [((c - (cols - 1) / 2) + (0.5 if r % 2 else 0) - 0.25, (r - (rows - 1) / 2) * h)
                for r in range(rows) for c in range(cols)]
    if fam == "triangle":
        rows = rng.randint(3, 6)
        h = math.sqrt(3) / 2
        return [((c - r / 2), (r - (rows - 1) / 2) * h) for r in range(rows) for c in range(r + 1)]
    n = rng.randint(8, 16)
    R = n / (2 * math.pi) * 1.15
    return [(R * math.cos(2 * math.pi * k / n - math.pi / 2), R * math.sin(2 * math.pi * k / n - math.pi / 2))
            for k in range(n)]


# ------------------------------------------------------------------ track
def _rect_at(occ, f):
    """Occluder rectangle at frame f (linear slide between stops)."""
    w, h = occ["w"], occ["h"]
    for (fa, xa, ya), (fb, xb, yb) in zip(occ["kf"], occ["kf"][1:]):
        if f <= fa:
            break
        if f <= fb:
            u = (f - fa) / (fb - fa)
            x, y = xa + (xb - xa) * u, ya + (yb - ya) * u
            return (round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2))
    fa, xa, ya = occ["kf"][0] if f <= occ["kf"][0][0] else occ["kf"][-1]
    return (round(xa, 2), round(ya, 2), round(xa + w, 2), round(ya + h, 2))


def _apply_mods(scene, mods):
    occs = [dict(o) for o in scene["occluders"]]
    for m in mods:
        if m[0] == "stay":                       # never leaves its first position
            for o in occs:
                o["kf"] = [o["kf"][0]]
        elif m[0] == "stop_early":               # skips the last slide
            for o in occs:
                o["kf"] = o["kf"][:-2] if len(o["kf"]) >= 4 else o["kf"][:1]
        elif m[0] == "remove":
            occs[m[1]] = None
    return [o for o in occs if o is not None]


@functools.lru_cache(maxsize=256)
def _track(key: str, mods: tuple):
    scene = json.loads(key)
    occs = _apply_mods(scene, mods)
    return tuple(tuple(_rect_at(o, f) for o in occs) for f in range(scene["frames"]))


def track(scene, mods=()):
    """Per frame: tuple of occluder rects (x0, y0, x1, y1) as drawn."""
    return _track(json.dumps(scene, sort_keys=True), tuple(mods))


def holds(scene):
    """(first, last) frame of each hold, shared by all occluders."""
    return [tuple(h) for h in scene["holds"]]


def states(scene, mods=()):
    """Per frame, per object: max cover over occluders (0 clear, 1 cut, 2 hidden)."""
    objs = scene["objects"]
    out = []
    for rects in track(scene, mods):
        out.append([max([cover(rc, o["x"], o["y"], o["r"]) for rc in rects] or [0]) for o in objs])
    return out


def clock_tenths(f):
    return f * 10 // FPS


# ------------------------------------------------------------------ checks
def run_errors(scene, mods=()):
    n = scene["frames"]
    st = states(scene, mods)
    tr = track(scene, mods)
    errs = []
    for a, b in holds(scene):
        if a < 0 or b > n - 1:
            errs.append("hold outside clip")
        if any(s == 1 for f in (a, (a + b) // 2, b) for s in st[f]):
            errs.append("object cut by an occluder at a hold")
    L, T, R, B = ARENA
    for f in range(0, n, 2):
        rs = tr[f]
        for rc in rs:
            if rc[0] < L or rc[1] < T or rc[2] > R or rc[3] > B:
                errs.append("occluder leaves the arena")
                break
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                p, q = rs[i], rs[j]
                if p[0] < q[2] + 12 and q[0] < p[2] + 12 and p[1] < q[3] + 12 and q[1] < p[3] + 12:
                    errs.append("occluders touch")
                    break
    return sorted(set(errs))


def check(scene):
    errs = run_errors(scene)
    st = states(scene)
    hs = holds(scene)
    n = scene["frames"]
    objs = scene["objects"]
    if hs[0][0] != 0 or hs[-1][1] != n - 1:
        errs.append("holds must cover clip start and end")
    for (a, b), (c, d) in zip(hs, hs[1:]):
        if c - b < SLIDE_F[0] or b < EDGE_F or c > n - 1 - EDGE_F:
            errs.append("slide too short or too close to clip edge")
    per_hold = [[st[(a + b) // 2][i] for i in range(len(objs))] for a, b in hs]
    for row in per_hold:
        k = sum(s == 2 for s in row)
        if k < 2 or k > 0.6 * len(objs):
            errs.append("hold hides too few or too many objects")
    for x, y in zip(per_hold, per_hold[1:]):
        if x == y:
            errs.append("a slide changes nothing")
    for i in range(len(objs)):
        if all(row[i] == 2 for row in per_hold):
            errs.append("object never seen")
    rects_at_hold = [track(scene)[(a + b) // 2] for a, b in hs]
    for g in scene.get("gaps", []):
        if all(any(cover(rc, g["x"], g["y"], g["r"]) for rc in rects) for rects in rects_at_hold):
            errs.append("gap never seen")
        if any(cover(rc, g["x"], g["y"], g["r"]) == 1 for rects in rects_at_hold for rc in rects):
            errs.append("gap cut by an occluder at a hold")
    return sorted(set(errs))


# ------------------------------------------------------------------ building
def _try(rng, sc, fam):
    slots = _slots(rng, fam)
    xs, ys = [p[0] for p in slots], [p[1] for p in slots]
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    L, T, R, B = ARENA
    sp_max = min((R - L - 260) / max(span_x, 1), (B - T - 120) / max(span_y, 1), 110)
    if sp_max < 52:
        return None
    sp = rng.uniform(52, sp_max)
    offset = fam in ("triangle", "honeycomb")      # offset rows need wider gaps for clean occluder edges
    r = int(min(sp * (rng.uniform(0.17, 0.22) if offset else rng.uniform(0.26, 0.36)), 28))
    if r < 12:
        return None
    cx = rng.uniform(L + 130 + sp * (-min(xs)), R - 130 - sp * max(xs))
    cy = rng.uniform(T + 60 + sp * (-min(ys)), B - 60 - sp * max(ys))
    pts = [(round(cx + sp * x, 1), round(cy + sp * y, 1)) for x, y in slots]
    color = rng.choice(sorted(COLORS))
    shape = rng.choice(SHAPES)
    objs = [{"x": x, "y": y, "r": r, "color": color} for x, y in pts]
    gaps = []
    if sc == "gaps":
        for k in sorted(rng.sample(range(len(objs)), rng.randint(1, min(3, len(objs) // 5)))):
            gaps.append(dict(objs[k]))
        objs = [o for o in objs if o not in gaps]
    if sc == "odd_one":
        odd = rng.choice([c for c in COLORS if math.dist(COLORS[c], COLORS[color]) >= ODD_DIST])
        for k in rng.sample(range(len(objs)), rng.randint(1, 3)):
            objs[k]["color"] = odd
    n_occ = 2 if sc == "two_occluders" else 1
    n_stops = rng.randint(2, 4) if n_occ == 1 else rng.randint(2, 3)
    holds_ = []
    f = 0
    for k in range(n_stops):
        h = rng.randint(*HOLD_F)
        holds_.append([f, f + h])
        f += h + rng.randint(*SLIDE_F)
    n = holds_[-1][1] + rng.randint(24, 40)
    if n > 10 * FPS:
        return None
    holds_[-1][1] = n - 1
    bx0, by0 = min(p[0] for p in pts) - r, min(p[1] for p in pts) - r
    bx1, by1 = max(p[0] for p in pts) + r, max(p[1] for p in pts) + r
    midx = (bx0 + bx1) / 2
    occs = []
    ocols = rng.sample(sorted(OCC_COLORS), n_occ)
    for j in range(n_occ):
        lo = 0.9 if offset else 1.2
        w = rng.uniform(lo * sp, max((lo + 0.1) * sp, 0.55 * (bx1 - bx0) / n_occ))
        h = rng.uniform(lo * sp, max((lo + 0.1) * sp, 0.6 * (by1 - by0)))
        kf = []
        prev = None
        always = None                            # hidden at every stop so far
        for k, (a, b) in enumerate(holds_):
            for _ in range(300 if offset else 150):
                x = rng.uniform(max(L, bx0 - w / 2), min(R - w, bx1 - w / 2))
                y = rng.uniform(max(T, by0 - h / 2), min(B - h, by1 - h / 2))
                rc = (x, y, x + w, y + h)
                if n_occ == 2 and not (x + w <= midx - 8 if j == 0 else x >= midx + 8):
                    continue
                if any(cover(rc, o["x"], o["y"], o["r"]) == 1 for o in objs + gaps):
                    continue
                hid = frozenset(i for i, o in enumerate(objs) if cover(rc, o["x"], o["y"], o["r"]) == 2)
                lo, hi = (2, 0.6 * len(objs)) if n_occ == 1 else (1, 0.3 * len(objs))
                if lo <= len(hid) <= hi and hid != prev and not (k == len(holds_) - 1 and always & hid):
                    prev = hid
                    always = hid if always is None else always & hid
                    break
            else:
                return None
            kf.append([a, round(x, 1), round(y, 1)] if k == 0 else [a, round(x, 1), round(y, 1)])
        full = [kf[0]]
        for k in range(1, len(kf)):
            full.append([holds_[k - 1][1], kf[k - 1][1], kf[k - 1][2]])
            full.append(kf[k])
        full = [full[0]] + full[1:]
        occs.append({"w": round(w, 1), "h": round(h, 1), "kf": [kf[0]] + full[1:], "color": ocols[j],
                     "texture": rng.choice(TEXTURES)})
    for o in occs:                               # first kf is the clip start at stop 0
        o["kf"][0] = [0, o["kf"][0][1], o["kf"][0][2]]
    return {"scenario": sc, "family": fam, "frames": n, "background": rng.choice(sorted(BACKGROUNDS)),
            "shape": shape, "spacing": round(sp, 1), "objects": objs, "gaps": gaps, "occluders": occs,
            "holds": holds_}


def build(rng):
    import questions  # late import: questions imports sim
    sc = rng.choice(SCENARIOS)
    fam = rng.choice(FAMILIES)
    for attempt in range(1, 3000):
        scene = _try(rng, sc, fam)
        if scene is not None and not check(scene) and questions.feasible(scene):
            scene["build_attempts"] = attempt
            return scene
    raise RuntimeError("no clean scene")
