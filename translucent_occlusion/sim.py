"""Scene simulation for translucent_occlusion v3.

Coloured shapes move on a dark floor past fixed, nearly-opaque translucent
panels. A shape is "completely hidden" while its whole disc (radius r) is
inside a panel. Every shape moves along straight keyframed segments at a
constant speed; it can pass through a panel, go in and come back out the same
side, go in and stay, or start hidden and come out. In the hidden_change
scenario a shape changes colour, shape or size while it is completely hidden.

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
SCENARIOS = ("crossing", "hidden_change", "hide_and_stay", "two_panels")
COLORS = {
    "red": (235, 72, 60), "blue": (66, 135, 245), "green": (70, 200, 100), "yellow": (240, 210, 50),
    "purple": (170, 95, 235), "orange": (245, 140, 40), "pink": (240, 100, 180), "cyan": (60, 215, 225),
    "white": (240, 240, 240),
}
SHAPES = ("circle", "square", "triangle", "pentagon", "hexagon", "star", "diamond", "cross")
BACKGROUNDS = {"charcoal": (22, 22, 26), "navy": (16, 20, 38), "forest": (16, 30, 22), "plum": (32, 18, 32),
               "slate": (30, 34, 40)}
FLOORS = ("plain", "grid", "dots")
TINTS = {"grey": (185, 185, 196), "frost": (180, 200, 220), "sand": (210, 196, 170), "smoke": (150, 150, 158)}
EDGE_F = 12          # events >= 0.5 s from both ends of the clip
GAP_F = 6            # any two hide/reveal events >= 6 frames (0.25 s) apart
ORDER_F = 8          # comparisons need >= 8 frames (0.33 s)
INSET = 8            # paused-inside points sit >= 8 px inside the panel edge
MIN_STEP = 4.0       # px per frame while partly behind a panel (no grazing)
SIZE_MARGIN = 1.2


# ------------------------------------------------------------------ geometry
def cover_state(p, x, y, r):
    """0 = clear of panel p, 1 = partly behind it, 2 = completely hidden."""
    if p["kind"] == "rect":
        dx = max(p["x0"] - x, 0, x - p["x1"])
        dy = max(p["y0"] - y, 0, y - p["y1"])
        if math.hypot(dx, dy) >= r:
            return 0
        inside = p["x0"] + r <= x <= p["x1"] - r and p["y0"] + r <= y <= p["y1"] - r
        return 2 if inside else 1
    d = math.hypot(x - p["cx"], y - p["cy"])
    if d >= p["R"] + r:
        return 0
    return 2 if d <= p["R"] - r else 1


def label(color, shape):
    return f"the {color} {shape}"


# ------------------------------------------------------------------ track
def _pos(kf, f):
    if f <= kf[0][0]:
        return kf[0][1], kf[0][2]
    for (fa, xa, ya), (fb, xb, yb) in zip(kf, kf[1:]):
        if f <= fb:
            u = (f - fa) / (fb - fa) if fb > fa else 1.0
            return xa + (xb - xa) * u, ya + (yb - ya) * u
    return kf[-1][1], kf[-1][2]


def _apply_mods(scene, mods):
    actors = [dict(a) for a in scene["actors"]]
    panels = list(scene["panels"])
    for m in mods:
        if m[0] == "delay":
            a = actors[m[1]]
            kf = a["kf"]
            a["kf"] = [kf[0]] + [[f + FPS, x, y] for f, x, y in kf[1:]]
            a["changes"] = [[f + FPS, k, v] for f, k, v in a.get("changes", [])]
        elif m[0] == "nopanel":
            panels = []
        elif m[0] == "remove":
            actors[m[1]] = dict(actors[m[1]], removed=True)
    return actors, panels


@functools.lru_cache(maxsize=256)
def _track(key: str, mods: tuple):
    scene = json.loads(key)
    actors, _ = _apply_mods(scene, mods)
    out = []
    for f in range(scene["frames"]):
        row = []
        for a in actors:
            if a.get("removed"):
                row.append(None)
                continue
            x, y = _pos(a["kf"], f)
            st = {"color": a["color"], "shape": a["shape"], "r": a["r"]}
            for cf, k, v in a.get("changes", []):
                if f >= cf:
                    st[k] = v
            row.append((round(x, 2), round(y, 2), st["r"], st["color"], st["shape"]))
        out.append(tuple(row))
    return tuple(out)


def track(scene, mods=()):
    """Per frame, per actor: (x, y, r, colour, shape) as drawn, or None if removed."""
    return _track(json.dumps(scene, sort_keys=True), tuple(mods))


def panels_for(scene, mods=()):
    return _apply_mods(scene, mods)[1]


def states(scene, mods=()):
    """Per frame, per actor: max cover state over panels (None if removed)."""
    tr, ps = track(scene, mods), panels_for(scene, mods)
    return [[None if a is None else max([cover_state(p, a[0], a[1], a[2]) for p in ps] or [0]) for a in row]
            for row in tr]


def events(scene, mods=()):
    """Sorted (frame, actor, 'hide'|'reveal'): first frame completely hidden / first frame not."""
    st = states(scene, mods)
    ev = []
    for f in range(1, len(st)):
        for i, (a, b) in enumerate(zip(st[f - 1], st[f])):
            if a is None:
                continue
            if b == 2 and a != 2:
                ev.append((f, i, "hide"))
            elif a == 2 and b != 2:
                ev.append((f, i, "reveal"))
    return ev


def clock_tenths(f):
    return f * 10 // FPS


# ------------------------------------------------------------------ checks
def run_errors(scene, mods=(), strict=True):
    """Reasons this (possibly modified) clip is ambiguous; [] if clean."""
    n = scene["frames"]
    tr, ps, st = track(scene, mods), panels_for(scene, mods), states(scene, mods)
    ev = events(scene, mods)
    errs = []
    fs = [e[0] for e in ev]
    if any(f < EDGE_F or f > n - 1 - EDGE_F for f in fs):
        errs.append("event too close to clip edge")
    if any(b - a < GAP_F for a, b in zip(fs, fs[1:])):
        errs.append("events too close together")
    for i in range(len(tr[0])):
        if tr[0][i] is None:
            continue
        if st[0][i] == 1 or st[-1][i] == 1:
            errs.append(f"actor {i} partly hidden at clip edge")
        for f in range(1, n):
            if st[f][i] == 1:
                a, b = tr[f - 1][i], tr[f][i]
                if math.hypot(b[0] - a[0], b[1] - a[1]) < MIN_STEP:
                    errs.append(f"actor {i} lingers at a panel edge")
                    break
        runs, f = [], 0
        while f < n:
            if st[f][i] == 1:
                g = f
                while g < n and st[g][i] == 1:
                    g += 1
                before = st[f - 1][i] if f > 0 else None
                after = st[g][i] if g < n else None
                if {before, after} != {0, 2}:
                    runs.append((f, g))
                f = g
            else:
                f += 1
        if runs:
            errs.append(f"actor {i} grazes a panel")
    if strict:
        L, T, R, B = ARENA
        for f in range(0, n, 2):
            row = tr[f]
            for i, a in enumerate(row):
                if a is None:
                    continue
                if not (L + a[2] <= a[0] <= R - a[2] and T + a[2] <= a[1] <= B - a[2]):
                    errs.append(f"actor {i} leaves the arena")
                for j in range(i + 1, len(row)):
                    b = row[j]
                    if b is None or (st[f][i] == 2 and st[f][j] == 2):
                        continue
                    if math.hypot(a[0] - b[0], a[1] - b[1]) < a[2] + b[2] + 8:
                        errs.append(f"actors {i},{j} touch")
                        break
        for i, a in enumerate(scene["actors"]):
            for cf, _, _ in a.get("changes", []):
                if any(st[g][i] != 2 for g in range(cf - 6, cf + 7)):
                    errs.append(f"actor {i} changes while visible")
        looks = [(a["color"], a["shape"]) for a in scene["actors"]]
        looks += [(tr[-1][i][3], tr[-1][i][4]) for i, a in enumerate(scene["actors"])
                  if a.get("changes") and a["changes"][0][1] != "r"]
        if len(set(looks)) != len(looks):
            errs.append("two actors share a look")
    return sorted(set(errs))


def check(scene):
    return run_errors(scene)


# ------------------------------------------------------------------ building
def _panel(rng, box, rmax):
    L, T, R, B = box
    kind = rng.choice(("rect", "rect", "circle"))
    if kind == "rect":
        w = rng.uniform(max(2 * rmax + 60, 170), min(380, R - L - 40))
        h = rng.uniform(max(2 * rmax + 60, 170), min(460, B - T - 40))
        x0 = rng.uniform(L + 20, R - 20 - w)
        y0 = rng.uniform(T + 20, B - 20 - h)
        return {"kind": "rect", "x0": round(x0), "y0": round(y0), "x1": round(x0 + w), "y1": round(y0 + h)}
    Rr = rng.uniform(max(rmax + 70, 110), min(210, (B - T) / 2 - 30, (R - L) / 2 - 30))
    return {"kind": "circle", "cx": round(rng.uniform(L + Rr + 20, R - Rr - 20)),
            "cy": round(rng.uniform(T + Rr + 20, B - Rr - 20)), "R": round(Rr)}


def _lane(rng, p, r):
    """(outside point A, hidden point In, outside point B, axis direction) for a path through panel p."""
    if p["kind"] == "rect":
        horiz = rng.random() < 0.6
        if horiz:
            y = rng.uniform(p["y0"] + r + INSET, p["y1"] - r - INSET)
            xi = rng.uniform(p["x0"] + r + INSET, p["x1"] - r - INSET)
            a, b = (p["x0"] - r - rng.uniform(40, 300), y), (p["x1"] + r + rng.uniform(40, 300), y)
            pin = (xi, y)
        else:
            x = rng.uniform(p["x0"] + r + INSET, p["x1"] - r - INSET)
            yi = rng.uniform(p["y0"] + r + INSET, p["y1"] - r - INSET)
            a, b = (x, p["y0"] - r - rng.uniform(40, 220)), (x, p["y1"] + r + rng.uniform(40, 220))
            pin = (x, yi)
        if rng.random() < 0.5:
            a, b = b, a
        return a, pin, b
    th = rng.choice((0, 90, 180, 270)) + rng.uniform(-20, 20)
    ux, uy = math.cos(math.radians(th)), math.sin(math.radians(th))
    lim = max(0.0, (p["R"] - r - INSET) * 0.5)
    o = rng.uniform(-lim, lim)
    cx, cy = p["cx"] - uy * o, p["cy"] + ux * o
    da, db = p["R"] + r + rng.uniform(40, 260), p["R"] + r + rng.uniform(40, 260)
    return (cx - ux * da, cy - uy * da), (cx, cy), (cx + ux * db, cy + uy * db)


def _seg(f, pa, pb, v):
    return f + max(2, math.ceil(math.dist(pa, pb) / v * FPS))


def _path(rng, kind, p, r, n, v):
    a, pin, b = _lane(rng, p, r)
    t0 = rng.randint(0, n // 2)
    if kind == "through":
        if rng.random() < 0.5:
            f1 = _seg(t0, a, pin, v)
            f2 = f1 + rng.randint(6, 40)
            return [[t0, *a], [f1, *pin], [f2, *pin], [_seg(f2, pin, b, v), *b]]
        return [[t0, *a], [_seg(t0, a, b, v), *b]]
    if kind == "back":
        f1 = _seg(t0, a, pin, v)
        f2 = f1 + rng.randint(10, 50)
        return [[t0, *a], [f1, *pin], [f2, *pin], [_seg(f2, pin, a, v), *a]]
    if kind == "stay":
        t0 = rng.randint(n // 4, n // 2 + 20)
        return [[t0, *a], [_seg(t0, a, pin, v), *pin]]
    # emerge: hidden at the start, comes out
    f1 = rng.randint(EDGE_F + 6, n // 2)
    return [[0, *pin], [f1, *pin], [_seg(f1, pin, b, v), *b]]


def _bystander(rng, box, r, n, v):
    L, T, R, B = box
    a = (rng.uniform(L + r, R - r), rng.uniform(T + r, B - r))
    if rng.random() < 0.4:
        return [[0, *a]]
    b = (rng.uniform(L + r, R - r), rng.uniform(T + r, B - r))
    t0 = rng.randint(0, n // 3)
    return [[t0, *a], [_seg(t0, a, b, v * 0.7), *b]]


def _fit(p, pin, a, r):
    """Pull outside point a toward pin until the disc is inside the arena; None if it can't stay clear of p."""
    L, T, R, B = ARENA
    ax, ay = a
    s = 1.0
    for _ in range(40):
        x, y = pin[0] + (ax - pin[0]) * s, pin[1] + (ay - pin[1]) * s
        if L + r <= x <= R - r and T + r <= y <= B - r:
            break
        s *= 0.95
    else:
        return None
    if cover_state(p, x, y, r + 24) != 0:
        return None
    return (x, y)


def _positions(kf, n):
    return [_pos(kf, f) for f in range(n)]


def _ok_with(placed, pos, r, n):
    L, T, R, B = ARENA
    if any(not (L + r <= x <= R - r and T + r <= y <= B - r) for x, y in pos[::2]):
        return False
    for q, rq in placed:
        if any(math.hypot(pos[f][0] - q[f][0], pos[f][1] - q[f][1]) < r + rq + 10 for f in range(0, n, 2)):
            return False
    return True


def _round_kf(kf):
    return [[int(f), round(x, 1), round(y, 1)] for f, x, y in kf]


def _try(rng, sc):
    n = rng.randint(7 * FPS, 10 * FPS)
    scene = {"scenario": sc, "frames": n, "background": rng.choice(sorted(BACKGROUNDS)), "floor": rng.choice(FLOORS),
             "tint": rng.choice(sorted(TINTS)), "alpha": round(rng.uniform(0.86, 0.93), 3)}
    k = rng.randint(*{"crossing": (3, 6), "hidden_change": (3, 5), "hide_and_stay": (3, 6), "two_panels": (3, 6)}[sc])
    rs = [rng.randint(22, 42) for _ in range(k)]
    box = ARENA
    if sc == "two_panels":
        mid = (box[0] + box[2]) // 2
        panels = [_panel(rng, (box[0], box[1], mid - 40, box[3]), max(rs)),
                  _panel(rng, (mid + 40, box[1], box[2], box[3]), max(rs))]
    else:
        panels = [_panel(rng, box, max(rs))]
    looks = rng.sample([(c, s) for c in COLORS for s in SHAPES], k)
    n_hidden = rng.randint(max(2, k - 2), k)
    kinds = {"crossing": ("through", "through", "back"), "hidden_change": ("through", "back"),
             "hide_and_stay": ("through", "back", "stay", "emerge"), "two_panels": ("through", "back")}[sc]
    actors, placed = [], []
    for i in range(k):
        r = rs[i]
        for _ in range(40):
            v = rng.uniform(150, 330)
            if i < n_hidden:
                kind = ("stay", "emerge")[i] if sc == "hide_and_stay" and i < 2 else rng.choice(kinds)
                p = panels[i % len(panels)] if sc == "two_panels" and i < 2 else rng.choice(panels)
                kf = _path(rng, kind, p, r, n, v)
                pin = kf[1][1:] if kind != "emerge" else kf[0][1:]
                fixed = []
                for fr in kf:
                    pt = tuple(fr[1:])
                    if cover_state(p, pt[0], pt[1], r) == 2:
                        fixed.append(fr)
                        continue
                    q = _fit(p, pin, pt, r)
                    if q is None:
                        break
                    fixed.append([fr[0], *q])
                if len(fixed) != len(kf):
                    continue
                kf = fixed
            else:
                kind = "bystander"
                kf = _bystander(rng, box, r, n, v)
                pos = _positions(kf, n)
                if any(cover_state(pp, x, y, r + 24) for pp in panels for x, y in pos[::2]):
                    continue
            kf = _round_kf(kf)
            pos = _positions(kf, n)
            if kf[-1][0] <= n - EDGE_F - 2 and _ok_with(placed, pos, r, n):
                break
        else:
            return None
        placed.append((pos, r))
        actors.append({"color": looks[i][0], "shape": looks[i][1], "r": r, "kind": kind, "kf": kf, "changes": []})
    order = list(range(k))
    rng.shuffle(order)
    scene["panels"] = panels
    scene["actors"] = [actors[i] for i in order]
    actors = scene["actors"]
    if sc == "hidden_change":
        st = states(scene)
        cands = []
        for i in range(k):
            runs = [f for f in range(n) if st[f][i] == 2]
            if len(runs) >= 18 and runs[-1] - runs[0] == len(runs) - 1 and runs[-1] < n - 1:
                cands.append((i, (runs[0] + runs[-1]) // 2))
        if not cands:
            return None
        used = {(a["color"], a["shape"]) for a in actors}
        for i, cf in rng.sample(cands, min(len(cands), rng.randint(1, 2))):
            a = actors[i]
            what = rng.choice(("color", "shape", "size"))
            if what == "size":
                nr = a["r"] + rng.choice((-1, 1)) * rng.randint(9, 14)
                if not 20 <= nr <= 46 or max(nr, a["r"]) < SIZE_MARGIN * min(nr, a["r"]):
                    return None
                a["changes"] = [[cf, "r", nr]]
                continue
            pool = [c for c in COLORS if (c, a["shape"]) not in used] if what == "color" else \
                [s for s in SHAPES if (a["color"], s) not in used]
            val = rng.choice(pool)
            used.add((val, a["shape"]) if what == "color" else (a["color"], val))
            a["changes"] = [[cf, what, val]]
    if not events(scene):
        return None
    return scene


def build(rng):
    import questions  # late import: questions imports sim
    sc = rng.choice(SCENARIOS)
    for attempt in range(1, 2000):
        scene = _try(rng, sc)
        if scene is not None and not check(scene) and questions.feasible(scene):
            scene["build_attempts"] = attempt
            return scene
    raise RuntimeError("no clean scene")
