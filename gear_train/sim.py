"""Gear trains: 3-8 meshed spur gears driven from ONE driver gear (black hub),
plus sometimes one loose gear that meshes with nothing and never turns.

Physics (kept from the Manim version, commit e2e24ee, now in pixel space):

- One module m (tooth size, px) per scene, so every pair can mesh. A gear's
  pitch radius is m * z / 2 for z teeth; meshing gears sit exactly
  r_a + r_b apart and their teeth are phase-aligned (a tooth of one sits in
  a gap of the other) by walking the mesh tree out from the driver.
- Meshing gears turn in opposite directions and |omega| * z = K is the same
  for every gear in the train (they share one tooth pitch), so a gear with
  half the teeth turns twice as fast, whatever sits in between (idlers).
- Gears that do not mesh always keep a clearly visible gap between their tooth
  tips (>= 22 px), so nothing looks meshed when it is not.

Five scenarios (scene["scenario"], uniform per seed):

  steady        the driver turns at a constant speed the whole clip
  reversal      the driver reverses once or twice (0.6 s ramp through zero);
                the whole train reverses with it
  disengage     one non-driver gear slides straight out of mesh (a visible gap
                opens), it and every gear it drives spin down and stand still
                for 1.85-3.15 s, spin up and slide back in. The stop is a whole
                number of tooth periods so the teeth re-mesh exactly
  brake         a grey brake block presses on one gear's teeth; the whole
                connected train (driver included) slows to a stop, stands
                still, and starts again when the brake lifts
  speed_change  the driver's motor changes speed (x2, x3, x1/2 or x1/3 over
                0.4 s); every gear in the train changes by the same factor

Every gear angle is an exact closed-form function of t (`theta`); answers.py
re-derives everything with a separate time-stepped integration over the
drawn geometry. Angles follow the maths convention on screen: positive =
counter-clockwise. Screen y points down.
"""
from __future__ import annotations
import itertools
import json
import math
import random

W, H = 1280, 720
FPS = 24
SCENARIOS = ("steady", "reversal", "disengage", "brake", "speed_change")
TOPOLOGIES = ("chain", "branching", "tree")
TIMER_BOX = (1078, 12, 1266, 78)
TIMER_KEEP = 22                  # px: gears stay this far from the clock box
AREA = (30, 30, 1250, 690)       # tooth tips stay inside
MAX_DUR = 10.0
M_RANGE = (8.0, 14.0)            # module, px
Z_RANGE = (10, 36)
ROOT_MIN = 36.0                  # px: smallest root radius (gear body >= 72 px across)
TIP_GAP_MOD = 2.8                # modules: tip-to-tip gap between gears that do not mesh (>= 22 px)
W_MAX = 2.0                      # rad/s: fastest gear before a speed change
W_MAX_ANY = 2.45                 # rad/s: fastest gear at any time
W_MIN = 0.3                      # rad/s: slowest train gear at the start
W_MIN_ANY = 0.2                  # rad/s: slowest train gear whenever the train runs freely
PAD = 0.5                        # s: question events stay this far from start / end
STEADY_HEAD = 1.5                # s: steady motion before the first event
STEADY_TAIL = 1.2                # s: steady motion after the last event
CRISP = 0.15                     # s: event times / stop lengths lie within this of a whole second
REV_RAMP = 0.6
SPEED_RAMP = 0.4
SLIDE_S = 0.5
SPIN_RAMP = 0.4
BRAKE_APPROACH = 0.3
BRAKE_RAMP = 0.35
SPEED_FACTORS = (2.0, 3.0, 0.5, 1.0 / 3.0)

COLOURS = {
    "red": (222, 52, 52), "orange": (245, 140, 30), "yellow": (240, 204, 30),
    "green": (50, 168, 72), "cyan": (34, 196, 212), "blue": (52, 102, 232),
    "purple": (150, 82, 212), "pink": (240, 116, 186), "brown": (146, 92, 46),
}
PATTERNS = ("plain", "holes", "spokes", "ring")
THEMES = {   # name: (background, dark?)
    "navy": ((18, 24, 42), True), "charcoal": ((30, 30, 34), True), "forest": ((18, 36, 30), True),
    "plum": ((36, 22, 44), True), "slate": ((34, 42, 54), True), "cocoa": ((40, 30, 24), True),
    "cream": ((238, 232, 216), False), "fog": ((222, 228, 236), False), "sky": ((214, 230, 242), False),
    "sand": ((232, 220, 196), False), "mint": ((218, 236, 226), False),
}
BG_STYLES = ("plain", "gradient", "grid", "dots", "pegboard", "hatch", "vignette")
BRAKE_RGB = (88, 92, 100)


# ------------------------------------------------------------------ helpers
def ss(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ss_int(x: float) -> float:
    """Integral of smoothstep from 0 to x (x clipped to [0, 1]); ss_int(1) = 0.5."""
    x = min(1.0, max(0.0, x))
    return x ** 3 - x ** 4 / 2


def crisp_time(rng, lo_int: int, hi_int: int) -> float:
    return rng.randint(lo_int, hi_int) + rng.uniform(-CRISP, CRISP)


def end_time(scene) -> float:
    """Time of the last rendered frame (the clip shows frames 0 .. n-1 at i / FPS)."""
    return (int(round(scene["duration"] * FPS)) - 1) / FPS


def gmap(scene) -> dict:
    return {g["id"]: g for g in scene["gears"]}


def train_ids(scene) -> list:
    return [g["id"] for g in scene["gears"] if not g["loose"]]


def loose_id(scene):
    return next((g["id"] for g in scene["gears"] if g["loose"]), None)


def children(scene) -> dict:
    out = {g["id"]: [] for g in scene["gears"]}
    for g in scene["gears"]:
        if g["parent"]:
            out[g["parent"]].append(g["id"])
    return out


def subtree(scene, gid) -> list:
    ch = children(scene)
    out, stack = [], [gid]
    while stack:
        u = stack.pop()
        out.append(u)
        stack.extend(ch[u])
    return sorted(out, key=lambda k: int(k[1:]))


def radius(scene, g) -> float:
    return scene["m"] * g["z"] / 2.0


def tip_r(scene, g) -> float:
    return radius(scene, g) + scene["m"]


def root_r(scene, g) -> float:
    return radius(scene, g) - 1.25 * scene["m"]


def phi(a, b) -> float:
    """Screen direction from point a to point b, maths convention (CCW +)."""
    return math.atan2(-(b[1] - a[1]), b[0] - a[0])


# ----------------------------------------------------------- closed-form motion
def _stop_def(t, a, r, b):
    """Integral over [0, t] of (1 - S) where S ramps 1->0 on [a, a+r], is 0 on
    [a+r, b] and ramps back to 1 on [b, b+r] (smoothstep ramps)."""
    return (r * ss_int((t - a) / r) + min(max(t - (a + r), 0.0), b - (a + r))
            + r * (min(max((t - b) / r, 0.0), 1.0) - ss_int((t - b) / r)))


def _stop_mult(t, a, r, b):
    return 1.0 - ss((t - a) / r) + ss((t - b) / r)


def motor_int(scene, t) -> float:
    """Psi for the gears fed by the driver: integral of the train's speed multiplier."""
    p = scene["program"]
    sc = scene["scenario"]
    if sc == "reversal":
        out, sgn, last = 0.0, 1.0, 0.0
        for rv in p["reversals"]:
            a, r = rv["mid"] - rv["ramp"] / 2, rv["ramp"]
            if t <= a:
                break
            out += sgn * (a - last)
            u = min(1.0, (t - a) / r)
            out += sgn * r / math.pi * math.sin(math.pi * u)
            sgn = -sgn
            last = a + r
            if t <= a + r:
                return out
        return out + sgn * max(0.0, t - last)
    if sc == "speed_change":
        sp = p["speed"]
        a, r, f = sp["mid"] - sp["ramp"] / 2, sp["ramp"], sp["factor"]
        return t + (f - 1) * (r * ss_int((t - a) / r) + max(0.0, t - (a + r)))
    if sc == "brake":
        b = p["brake"]
        return t - _stop_def(t, b["t_touch"], b["ramp"], b["release"])
    return t


def motor_mult(scene, t) -> float:
    p = scene["program"]
    sc = scene["scenario"]
    if sc == "reversal":
        m = 1.0
        for rv in p["reversals"]:
            a, r = rv["mid"] - rv["ramp"] / 2, rv["ramp"]
            m *= math.cos(math.pi * min(1.0, max(0.0, (t - a) / r)))
        return m
    if sc == "speed_change":
        sp = p["speed"]
        return 1 + (sp["factor"] - 1) * ss((t - (sp["mid"] - sp["ramp"] / 2)) / sp["ramp"])
    if sc == "brake":
        b = p["brake"]
        return _stop_mult(t, b["t_touch"], b["ramp"], b["release"])
    return 1.0


def _in_disengaged(scene, gid) -> bool:
    d = scene["program"].get("disengage")
    return bool(d) and gid in d["moving"]


def psi(scene, gid, t) -> float:
    v = motor_int(scene, t)
    if _in_disengaged(scene, gid):
        d = scene["program"]["disengage"]
        v -= _stop_def(t, d["still0"] - d["ramp"], d["ramp"], d["still1"])
    return v


def psi_rate(scene, gid, t) -> float:
    v = motor_mult(scene, t)
    if _in_disengaged(scene, gid):
        d = scene["program"]["disengage"]
        v *= _stop_mult(t, d["still0"] - d["ramp"], d["ramp"], d["still1"])
    return v


def theta(scene, gid, t) -> float:
    g = gmap(scene)[gid]
    if g["loose"]:
        return g["phase"]
    return g["phase"] + g["dir"] * scene["K"] / g["z"] * psi(scene, gid, t)


def omega(scene, gid, t) -> float:
    g = gmap(scene)[gid]
    if g["loose"]:
        return 0.0
    return g["dir"] * scene["K"] / g["z"] * psi_rate(scene, gid, t)


def offset(scene, gid, t) -> tuple:
    d = scene["program"].get("disengage")
    if not d or gid not in d["moving"]:
        return (0.0, 0.0)
    s = d["slide"]
    k = ss((t - d["t_out"]) / s) - ss((t - (d["still1"] + d["ramp"])) / s)
    return (k * d["shift"][0], k * d["shift"][1])


def centre(scene, gid, t) -> tuple:
    g = gmap(scene)[gid]
    dx, dy = offset(scene, gid, t)
    return (g["x"] + dx, g["y"] + dy)


def pad_frac(scene, t) -> float:
    b = scene["program"].get("brake")
    if not b:
        return 0.0
    a = b["approach"]
    return ss((t - (b["t_touch"] - a)) / a) - ss((t - b["release"]) / a)


def pad_inner(scene, t) -> float:
    """Distance of the brake block's inner face from the braked gear's centre."""
    b = scene["program"]["brake"]
    g = gmap(scene)[b["gear"]]
    return tip_r(scene, g) + b["travel"] * (1.0 - pad_frac(scene, t))


def pad_polys(scene, inner: float) -> list:
    """Brake block + arm as two quads in screen px, for a given inner-face distance."""
    b = scene["program"]["brake"]
    g = gmap(scene)[b["gear"]]
    m = scene["m"]
    ux, uy = math.cos(b["angle"]), -math.sin(b["angle"])
    vx, vy = -uy, ux
    hw = max(24.0, 2.4 * m)
    depth = max(22.0, 2.0 * m)

    def quad(r0, r1, w):
        return [(g["x"] + ux * r + vx * s * w, g["y"] + uy * r + vy * s * w)
                for r, s in ((r0, -1), (r1, -1), (r1, 1), (r0, 1))]
    return [quad(inner, inner + depth, hw), quad(inner + depth - 2, inner + depth + 56, 8.0)]


# ----------------------------------------------------------------- building
def _tree(rng, n, topo):
    par = [None]
    if topo == "chain":
        par += list(range(n - 1))
    elif topo == "branching":
        hub = 1 if n >= 4 else 0
        par += [0] if hub == 1 else []
        rest = list(range(len(par), n))
        nb = 2 if len(rest) <= 3 else rng.choice([2, 3])
        tails = [hub] * nb
        for k, g in enumerate(rest):
            par.append(tails[k % nb])
            tails[k % nb] = g
    else:
        deg = {0: 0}
        for g in range(1, n):
            cands = [p for p in range(g) if deg[p] < (2 if p else 3)]
            p = rng.choice(cands)
            par.append(p)
            deg[p] += 1
            deg[g] = 0
    return par


def _teeth(rng, n, style):
    if style == "family":
        base = rng.choice([4, 5, 6])
        ks = [k for k in range(1, 20) if Z_RANGE[0] <= base * k <= Z_RANGE[1]]
        return [base * rng.choice(ks) for _ in range(n)]
    return [rng.randint(*Z_RANGE) for _ in range(n)]


def _tip_gap(p, q, zp, zq):
    """Tip-to-tip gap in modules between two gears (positions in modules)."""
    return math.dist(p, q) - (zp + zq) / 2.0 - 2.0


def _layout(rng, par, z, bend):
    n = len(z)
    pos = {0: (0.0, 0.0)}
    head = {0: rng.uniform(-math.pi, math.pi)}
    kids = {i: [g for g in range(n) if par[g] == i] for i in range(n)}
    for u in range(n):
        ks = kids[u]
        if not ks:
            continue
        if len(ks) == 1:
            angs = [head[u] + math.radians(rng.uniform(-bend, bend))]
        else:
            spread = math.radians(rng.uniform(100, 200))
            c = head[u] + math.radians(rng.uniform(-30, 30))
            angs = [c - spread / 2 + spread * k / (len(ks) - 1) for k in range(len(ks))]
        for v, a in zip(ks, angs):
            d = (z[u] + z[v]) / 2.0
            pos[v] = (pos[u][0] + math.cos(a) * d, pos[u][1] - math.sin(a) * d)
            head[v] = a
    for a, b in itertools.combinations(range(n), 2):
        if par[a] == b or par[b] == a:
            continue
        if _tip_gap(pos[a], pos[b], z[a], z[b]) < TIP_GAP_MOD:
            return None
    return pos


def _theme(rng):
    name = rng.choice(sorted(THEMES))
    bg, dark = THEMES[name]
    return {"name": name, "bg": list(bg), "dark": dark, "style": rng.choice(BG_STYLES),
            "texture_seed": rng.randrange(10 ** 6), "outline_w": rng.choice([2, 3])}


def _box_gap(x, y, r, box):
    x0, y0, x1, y1 = box
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy) - r


def _gears_ok_in_frame(gs, m):
    for x, y, z in gs:
        rt = m * z / 2 + m
        if x - rt < AREA[0] or x + rt > AREA[2] or y - rt < AREA[1] or y + rt > AREA[3]:
            return False
        if _box_gap(x, y, rt, TIMER_BOX) < TIMER_KEEP:
            return False
    return True


def _try(rng, scenario):
    n = rng.randint(4 if scenario in ("disengage", "brake") else 3, 8)
    topo = rng.choice(TOPOLOGIES)
    par = _tree(rng, n, topo)
    tstyle = rng.choice(["free", "family"])
    bend = rng.uniform(25, 75)
    pos = z = None
    for _ in range(60):
        z = _teeth(rng, n, tstyle)
        pos = _layout(rng, par, z, bend)
        if pos is not None:
            break
    if pos is None:
        return None
    want_loose = n < 9 and rng.random() < 0.35
    z_loose = None
    if want_loose:
        z_loose = rng.randint(10, 30)
        placed = False
        for _ in range(60):
            a = rng.randrange(n)
            ang = rng.uniform(-math.pi, math.pi)
            d = (z[a] + z_loose) / 2.0 + 2.0 + rng.uniform(3.2, 6.0)
            p = (pos[a][0] + math.cos(ang) * d, pos[a][1] - math.sin(ang) * d)
            if all(_tip_gap(p, pos[i], z_loose, z[i]) >= 3.2 for i in range(n)):
                pos[n] = p
                z = z + [z_loose]
                placed = True
                break
        if not placed:
            return None
    N = len(z)
    # scale to the frame (framing knob: fill fraction + offset)
    xs0 = min(pos[i][0] - z[i] / 2 - 1 for i in range(N))
    xs1 = max(pos[i][0] + z[i] / 2 + 1 for i in range(N))
    ys0 = min(pos[i][1] - z[i] / 2 - 1 for i in range(N))
    ys1 = max(pos[i][1] + z[i] / 2 + 1 for i in range(N))
    aw, ah = AREA[2] - AREA[0], AREA[3] - AREA[1]
    fill = rng.uniform(0.62, 0.98)
    m = min(fill * aw / (xs1 - xs0), fill * ah / (ys1 - ys0), M_RANGE[1] * rng.uniform(0.85, 1.0))
    if m < M_RANGE[0]:
        return None
    m = round(m, 3)
    if min(m * zz / 2 - 1.25 * m for zz in z) < ROOT_MIN:
        return None
    bw, bh = (xs1 - xs0) * m, (ys1 - ys0) * m
    px = None
    for _ in range(20):
        ox = AREA[0] + rng.uniform(0, aw - bw) - xs0 * m
        oy = AREA[1] + rng.uniform(0, ah - bh) - ys0 * m
        cand = {i: (round(pos[i][0] * m + ox, 2), round(pos[i][1] * m + oy, 2)) for i in range(N)}
        if _gears_ok_in_frame([(cand[i][0], cand[i][1], z[i]) for i in range(N)], m):
            px = cand
            break
    if px is None:
        return None
    # speeds: |omega| * z = K for every train gear
    zt = z[:n]
    ratio = max(zt) / min(zt)
    lo, hi = max(0.7, W_MIN * ratio), W_MAX
    factor = None
    if scenario == "speed_change":
        factor = rng.choice(SPEED_FACTORS)
        if factor > 1:
            hi = min(hi, W_MAX_ANY / factor * 0.98)
        else:
            lo = max(lo, W_MIN_ANY * ratio / factor * 1.02)
    if lo > hi:
        return None
    w_fast = rng.uniform(lo, hi)
    K = round(w_fast * min(zt), 4)
    sign = rng.choice([-1, 1])
    names = rng.sample(sorted(COLOURS), N)
    theme = _theme(rng)
    gears = []
    depth = {0: 0}
    for i in range(N):
        if i < n and i > 0:
            depth[i] = depth[par[i]] + 1
        base = COLOURS[names[i]]
        jit = rng.randint(-14, 14)
        rgb = [min(255, max(0, c + jit)) for c in base]
        pat = rng.choice(PATTERNS)
        gears.append({
            "id": f"g{i}", "colour": names[i], "rgb": rgb, "z": z[i], "x": px[i][0], "y": px[i][1],
            "pattern": pat, "pattern_n": rng.randint(3, 6), "loose": i >= n,
            "parent": f"g{par[i]}" if 0 < i < n else None,
            "dir": (sign * (-1) ** depth[i]) if i < n else 0, "phase": 0.0,
        })
    # phase-align teeth along the mesh tree (tooth k of a gear is centred at phase + 2*pi*k/z)
    gears[0]["phase"] = round(rng.uniform(0, 2 * math.pi), 6)
    for i in range(1, n):
        p, c = gears[par[i]], gears[i]
        f = phi((p["x"], p["y"]), (c["x"], c["y"]))
        up = (f - p["phase"]) * p["z"] / (2 * math.pi)
        c["phase"] = round((f + math.pi - 2 * math.pi / c["z"] * (0.5 - up)) % (2 * math.pi), 9)
    if N > n:
        gears[n]["phase"] = round(rng.uniform(0, 2 * math.pi), 6)
    scene = {"scenario": scenario, "topology": topo, "teeth_style": tstyle, "m": m, "K": K,
             "sign": sign, "driver": "g0", "w_driver": round(K / z[0], 6), "gears": gears, "theme": theme,
             "fill": round(fill, 3), "program": {}}
    if not _program(rng, scene, factor):
        return None
    scene["log"] = make_log(scene)
    return scene


def _quant(d):
    return round(round(d * FPS) / FPS, 6)


def _program(rng, scene, factor) -> bool:
    sc = scene["scenario"]
    p = scene["program"]
    if sc == "steady":
        scene["duration"] = _quant(rng.uniform(5.0, MAX_DUR))
        return True
    if sc == "reversal":
        nrev = rng.choice([1, 2])
        m1 = crisp_time(rng, 2, 5 if nrev == 1 else 4)
        mids = [m1]
        if nrev == 2:
            mids.append(crisp_time(rng, math.ceil(m1 + 2.5), 7))
        p["reversals"] = [{"mid": round(t, 4), "ramp": REV_RAMP} for t in mids]
        need = mids[-1] + REV_RAMP / 2 + STEADY_TAIL + 0.3
    elif sc == "speed_change":
        mid = crisp_time(rng, 3, 6)
        p["speed"] = {"mid": round(mid, 4), "ramp": SPEED_RAMP, "factor": factor}
        need = mid + SPEED_RAMP / 2 + 2.5
    elif sc == "disengage":
        if not _pick_disengage(rng, scene):
            return False
        d = p["disengage"]
        t_out = crisp_time(rng, 2, 3)
        period = 2 * math.pi / scene["K"]
        target = rng.choice([2, 3])
        ks = [k for k in range(1, 400) if abs(k * period - SPIN_RAMP - target) <= CRISP]
        if not ks:
            return False
        k = rng.choice(ks)
        still = k * period - SPIN_RAMP
        d.update({"t_out": round(t_out, 4), "slide": SLIDE_S, "ramp": SPIN_RAMP,
                  "still0": round(t_out + SLIDE_S + SPIN_RAMP, 6), "teeth_lag": k})
        d["still1"] = round(d["still0"] + still, 9)
        need = d["still1"] + SPIN_RAMP + SLIDE_S + STEADY_TAIL
    else:   # brake
        if not _pick_brake(rng, scene):
            return False
        b = p["brake"]
        t_touch = crisp_time(rng, 2, 3)
        still = crisp_time(rng, 2, 3)
        b.update({"t_touch": round(t_touch, 4), "approach": BRAKE_APPROACH, "ramp": BRAKE_RAMP,
                  "still0": round(t_touch + BRAKE_RAMP, 6)})
        b["release"] = round(b["still0"] + still, 6)
        need = b["release"] + BRAKE_RAMP + STEADY_TAIL
    if need > MAX_DUR:
        return False
    lo = max(need, 5.0)
    scene["duration"] = _quant(rng.uniform(lo, min(MAX_DUR, lo + 2.5)))
    return scene["duration"] >= need - 1e-9


def _clear_of(scene, x, y, rt, skip, gap):
    for g in scene["gears"]:
        if g["id"] in skip:
            continue
        if math.dist((x, y), (g["x"], g["y"])) - rt - tip_r(scene, g) < gap:
            return False
    return True


def _pick_disengage(rng, scene) -> bool:
    m = scene["m"]
    gm = gmap(scene)
    cands = []
    for g in scene["gears"]:
        if g["loose"] or g["parent"] is None:
            continue
        p = gm[g["parent"]]
        L = math.dist((p["x"], p["y"]), (g["x"], g["y"]))
        ux, uy = (g["x"] - p["x"]) / L, (g["y"] - p["y"]) / L
        dist = 2 * m + max(26.0, 2.6 * m)
        moving = subtree(scene, g["id"])
        ok = True
        for frac in (0.35, 0.7, 1.0):
            sx, sy = ux * dist * frac, uy * dist * frac
            pts = [(gm[k]["x"] + sx, gm[k]["y"] + sy, gm[k]["z"]) for k in moving]
            if not _gears_ok_in_frame(pts, m):
                ok = False
                break
            for k in moving:
                if not _clear_of(scene, gm[k]["x"] + sx, gm[k]["y"] + sy, tip_r(scene, gm[k]),
                                 set(moving) | {g["parent"]}, TIP_GAP_MOD * m):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            cands.append((g["id"], [round(ux * dist, 3), round(uy * dist, 3)], moving))
    if not cands:
        return False
    gid, shift, moving = rng.choice(cands)
    scene["program"]["disengage"] = {"gear": gid, "parent": gm[gid]["parent"], "shift": shift,
                                     "moving": moving}
    return True


def _poly_clear(scene, polys, skip, gap) -> bool:
    for poly in polys:
        for (x, y) in poly:
            if not (AREA[0] <= x <= AREA[2] and AREA[1] <= y <= AREA[3]):
                return False
            if _box_gap(x, y, 0, TIMER_BOX) < TIMER_KEEP:
                return False
        cx = sum(p[0] for p in poly) / 4
        cy = sum(p[1] for p in poly) / 4
        pts = poly + [(cx, cy)] + [((poly[i][0] + poly[(i + 1) % 4][0]) / 2,
                                    (poly[i][1] + poly[(i + 1) % 4][1]) / 2) for i in range(4)]
        for g in scene["gears"]:
            if g["id"] in skip:
                continue
            for (x, y) in pts:
                if math.dist((x, y), (g["x"], g["y"])) - tip_r(scene, g) < gap:
                    return False
    return True


def _pick_brake(rng, scene) -> bool:
    cands = []
    for g in scene["gears"]:
        if g["loose"]:
            continue
        for k in range(24):
            ang = 2 * math.pi * k / 24
            scene["program"]["brake"] = {"gear": g["id"], "angle": ang, "travel": 34.0}
            inner_out = tip_r(scene, g) + 34.0
            if (_poly_clear(scene, pad_polys(scene, inner_out), {g["id"]}, 16.0)
                    and _poly_clear(scene, pad_polys(scene, tip_r(scene, g)), {g["id"]}, 16.0)):
                cands.append((g["id"], round(ang, 6)))
    scene["program"].pop("brake", None)
    if not cands:
        return False
    gid, ang = rng.choice(cands)
    scene["program"]["brake"] = {"gear": gid, "angle": ang, "travel": 34.0}
    return True


# ----------------------------------------------------------------- the log
def event_windows(scene) -> dict:
    """start: steady window before the first event; end: steady window after the last."""
    sc, p, D = scene["scenario"], scene["program"], end_time(scene)
    if sc == "steady":
        return {"start": [0.0, D], "end": [0.0, D], "events": []}
    if sc == "reversal":
        evs = [[r["mid"] - r["ramp"] / 2, r["mid"] + r["ramp"] / 2] for r in p["reversals"]]
    elif sc == "speed_change":
        s = p["speed"]
        evs = [[s["mid"] - s["ramp"] / 2, s["mid"] + s["ramp"] / 2]]
    elif sc == "disengage":
        d = p["disengage"]
        evs = [[d["t_out"], d["still1"] + d["ramp"] + d["slide"]]]
    else:
        b = p["brake"]
        evs = [[b["t_touch"] - b["approach"], b["release"] + max(b["ramp"], b["approach"])]]
    return {"start": [0.0, evs[0][0]], "end": [evs[-1][1], D], "events": evs}


def make_log(scene) -> dict:
    """Generator-side facts (questions.py reads these; answers.py never does)."""
    sc, p = scene["scenario"], scene["program"]
    win = event_windows(scene)
    log = {"windows": win, "stopped": [], "still": None, "event_t": [], "cw_start": None}
    tr = train_ids(scene)
    if sc == "disengage":
        d = p["disengage"]
        log["stopped"] = list(d["moving"])
        log["still"] = [d["still0"], d["still1"]]
        log["event_t"] = [d["t_out"]]
    elif sc == "brake":
        b = p["brake"]
        log["stopped"] = list(tr)
        log["still"] = [b["still0"], b["release"]]
        log["event_t"] = [b["t_touch"]]
    elif sc == "reversal":
        log["event_t"] = [r["mid"] for r in p["reversals"]]
    elif sc == "speed_change":
        log["event_t"] = [p["speed"]["mid"]]
    gm = gmap(scene)
    log["cw_start"] = sorted((k for k in tr if gm[k]["dir"] < 0), key=lambda k: int(k[1:]))
    # total angle turned through (path length, so a reversal adds rather than cancels), up to the last
    # frame: the motion is monotone between the reversal turning points
    cuts = [0.0] + [r["mid"] for r in p.get("reversals", [])] + [end_time(scene)]
    log["turns_total"] = {k: round(sum(abs(theta(scene, k, b) - theta(scene, k, a)) for a, b in zip(cuts, cuts[1:]))
                                   / (2 * math.pi), 6) for k in tr}
    return log


def build(rng: random.Random) -> dict:
    import questions                       # lazy: questions imports sim
    scenario = rng.choice(SCENARIOS)       # fixed per seed, so the mix stays uniform
    tries = 0
    while True:
        tries += 1
        scene = _try(rng, scenario)
        if scene is None or check(scene) or not questions.pool_ok(scene):
            continue
        scene["tries"] = tries
        return scene


# ------------------------------------------------------------------- check
def check(scene: dict) -> list:
    errs = []
    if scene["scenario"] not in SCENARIOS:
        errs.append("unknown scenario")
    gs = scene["gears"]
    gm = gmap(scene)
    m = scene["m"]
    D = scene["duration"]
    if D > MAX_DUR + 1e-9:
        errs.append("clip longer than 10 s")
    if len({g["colour"] for g in gs}) != len(gs):
        errs.append("two gears share a colour")
    if not 3 <= len(train_ids(scene)) <= 8 or len(gs) > 9:
        errs.append("gear count out of range")
    if m < M_RANGE[0] - 1e-9:
        errs.append("module too small")
    for g in gs:
        if root_r(scene, g) < ROOT_MIN - 1e-6:
            errs.append(f"{g['id']} too small")
    if not _gears_ok_in_frame([(g["x"], g["y"], g["z"]) for g in gs], m):
        errs.append("a gear leaves the frame or comes near the clock")
    # meshing pairs are exactly tangent; every other pair keeps a visible gap
    for a, b in itertools.combinations(gs, 2):
        d = math.dist((a["x"], a["y"]), (b["x"], b["y"]))
        meshed = a["parent"] == b["id"] or b["parent"] == a["id"]
        if meshed:
            if abs(d - radius(scene, a) - radius(scene, b)) > 0.05:
                errs.append(f"{a['id']}-{b['id']} not at mesh distance")
        elif d - tip_r(scene, a) - tip_r(scene, b) < TIP_GAP_MOD * m - 0.05:
            errs.append(f"{a['id']} and {b['id']} look meshed but are not")
    # directions alternate, the train is one tree rooted at the driver
    for g in gs:
        if g["loose"]:
            continue
        if g["parent"] is None:
            if g["id"] != scene["driver"] or g["dir"] != scene["sign"]:
                errs.append("bad driver")
        elif g["dir"] != -gm[g["parent"]]["dir"]:
            errs.append(f"{g['id']} turns the same way as the gear it meshes with")
    # frames: teeth stay phase-locked while engaged; speeds stay readable
    nfr = int(round(D * FPS))
    tr = train_ids(scene)
    for f in range(0, nfr + 1, 3):
        t = f / FPS
        for g in gs:
            if g["loose"] or g["parent"] is None:
                continue
            p = gm[g["parent"]]
            cp, cc = centre(scene, p["id"], t), centre(scene, g["id"], t)
            d = math.dist(cp, cc)
            if d < radius(scene, p) + radius(scene, g) + 2 * m + 0.5:
                ang = phi(cp, cc)
                up = (ang - theta(scene, p["id"], t)) * p["z"] / (2 * math.pi)
                uc = (ang + math.pi - theta(scene, g["id"], t)) * g["z"] / (2 * math.pi)
                x = (up + uc - 0.5) % 1.0
                if min(x, 1 - x) > 1e-5:
                    errs.append(f"{g['id']} teeth out of phase with {p['id']} at frame {f}")
                    break
        ws = [abs(omega(scene, k, t)) for k in tr]
        if max(ws) > W_MAX_ANY + 1e-6:
            errs.append(f"a gear turns too fast at frame {f}")
            break
    win = event_windows(scene)
    ws0 = [abs(omega(scene, k, 0.0)) for k in tr]
    if min(ws0) < W_MIN - 1e-6 or max(ws0) > W_MAX + 1e-6:
        errs.append("start speeds out of range")
    if scene["scenario"] == "speed_change":
        wsD = [abs(omega(scene, k, D)) for k in tr]
        if min(wsD) < W_MIN_ANY - 1e-6:
            errs.append("a gear turns too slowly after the change")
    if win["events"]:
        if win["start"][1] < STEADY_HEAD - 1e-6:
            errs.append("first event too early")
        if D - win["end"][0] < STEADY_TAIL - 1e-6:
            errs.append("last event too close to the end")
        for (a0, a1), (b0, b1) in zip(win["events"], win["events"][1:]):
            if b0 - a1 < 1.0:
                errs.append("events too close together")
    for t in scene["log"]["event_t"] if "log" in scene else []:
        if abs(t - round(t)) > CRISP + 1e-6:
            errs.append("event time too close to a rounding boundary")
    if scene["log"].get("still"):
        s0, s1 = scene["log"]["still"]
        if abs((s1 - s0) - round(s1 - s0)) > CRISP + 1e-6 or s1 - s0 < 1.5:
            errs.append("stop length too close to a rounding boundary")
    d = scene["program"].get("disengage")
    if d:
        # whole number of teeth lost -> the teeth re-mesh
        lag = scene["K"] * (d["ramp"] + d["still1"] - d["still0"]) / (2 * math.pi)
        if abs(lag - round(lag)) > 1e-6:
            errs.append("disengaged gear does not re-mesh on a whole tooth")
        if sorted(d["moving"], key=lambda k: int(k[1:])) != subtree(scene, d["gear"]):
            errs.append("disengage moves the wrong gears")
        g, p = gm[d["gear"]], gm[d["parent"]]
        cg = centre(scene, g["id"], (d["still0"] + d["still1"]) / 2)
        gap = math.dist(cg, (p["x"], p["y"])) - tip_r(scene, g) - tip_r(scene, p)
        if gap < 24.0:
            errs.append("disengage gap too small to see")
    b = scene["program"].get("brake")
    if b:
        g = gm[b["gear"]]
        for inner in (tip_r(scene, g), tip_r(scene, g) + b["travel"]):
            if not _poly_clear(scene, pad_polys(scene, inner), {g["id"]}, 15.0):
                errs.append("brake block too close to another gear")
                break
    if "log" in scene and json.dumps(scene["log"], sort_keys=True) != json.dumps(make_log(scene), sort_keys=True):
        errs.append("stored log doesn't match a replay")
    return errs


def signature(scene: dict) -> str:
    """Hash input for the diversity check: rounded scene parameters."""
    key = {"scenario": scene["scenario"], "theme": [scene["theme"]["name"], scene["theme"]["style"]],
           "m": round(scene["m"], 1), "K": round(scene["K"], 2), "sign": scene["sign"],
           "gears": [[g["colour"], g["z"], round(g["x"]), round(g["y"]), g["pattern"], g["loose"], g["parent"]]
                     for g in scene["gears"]],
           "program": json.loads(json.dumps(scene["program"], sort_keys=True), parse_float=lambda s: round(float(s), 2)),
           "duration": round(scene["duration"], 2)}
    return json.dumps(key, sort_keys=True)
