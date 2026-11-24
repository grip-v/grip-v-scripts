"""simple_harmonic_motion scene sampler.

3-6 oscillators move side by side. Each one is a coloured mass (unique
colour per clip; shape, pattern and size vary) with its own unnumbered tick
ruler whose taller tick is the rest mark. Scenarios (scene["scenario"]):

  horizontal  blocks on horizontal springs fixed to a wall, undamped, any start phase
  vertical    masses hanging from a ceiling on springs, undamped, any start phase
  pendulum    small-swing pendulums of different string lengths (simple harmonic)
  damped      springs (horizontal or vertical) with a dashpot beside each spring;
              every mass starts at rest, pulled out, at t = 0 and the swing dies away
  staggered   springs (horizontal or vertical); each mass is held out by a peg and
              released at its own time, then swings undamped

Motion is closed form (u = displacement along the ruler in px; for a
pendulum u is the bob's sideways offset L sin(theta)). Times are seconds;
frame f is shown at t = f / 24 and the clock shows f * 10 // 24 tenths.
Pure function of the rng passed in; JSON-only output.
"""
from __future__ import annotations
import math
import random

FPS = 24
MIN_FRAMES, MAX_FRAMES = 144, 240          # 6-10 s
EDGE = 0.5                                 # question events >= 0.5 s from start and end
AREA = (250, 24, 1262, 704)                # all action lives here; the clock is left of x = 214
CLOCK_BOX = (18, 16, 214, 84)
VMAX = 1000.0                              # px/s, so nothing moves more than ~42 px per frame
PERIOD_GAP = 0.10                          # periods differ pairwise by >= 10 %
TICK_EPS = 0.15                            # amplitudes sit within 0.15 tick of a whole tick count
SCENARIOS = ("horizontal", "vertical", "pendulum", "damped", "staggered")

COLOURS = {
    "red": (222, 40, 40), "orange": (247, 130, 20), "yellow": (240, 200, 20), "green": (35, 165, 60),
    "teal": (0, 170, 170), "blue": (40, 90, 230), "purple": (140, 60, 210), "pink": (240, 100, 190),
    "brown": (135, 80, 35),
}
SHAPES = ("circle", "square", "hexagon", "diamond", "triangle")
PATTERNS = ("solid", "striped", "dotted", "ringed")
SPRING_STYLES = ("zigzag", "coil", "wave")
THEMES = {
    "light": {"bg": [(236, 233, 226), (222, 228, 234), (232, 230, 238), (226, 232, 224), (240, 240, 240)],
              "ink": [(70, 70, 76), (60, 66, 80), (84, 72, 64)],
              "spring": [(110, 112, 120), (90, 96, 110), (120, 108, 96)],
              "fixture": [(150, 150, 156), (136, 140, 150), (158, 150, 140)],
              "rest": [(30, 30, 34), (20, 40, 70)], "decor": 18},
    "dark": {"bg": [(22, 24, 30), (30, 26, 34), (20, 30, 30), (34, 34, 36), (16, 22, 36)],
             "ink": [(200, 200, 206), (190, 196, 210), (214, 204, 192)],
             "spring": [(170, 172, 180), (150, 156, 170), (180, 170, 160)],
             "fixture": [(110, 112, 120), (96, 102, 116), (120, 112, 104)],
             "rest": [(250, 250, 250), (230, 240, 255)], "decor": 14},
}


# ------------------------------------------------------------------ closed-form motion
def omega_d(o):
    return o["omega"] * math.sqrt(1 - o["zeta"] ** 2)


def theta0(o):
    return math.asin(o["A"] / o["L"])


def _phase_u(o, t):
    """Displacement (px along the ruler) at time t."""
    m, A, w = o["mode"], o["A"], o["omega"]
    if m == "free":
        c = math.cos(w * t + o["phi"])
    elif m == "held":
        c = o["s"] if t <= o["t_rel"] else o["s"] * math.cos(w * (t - o["t_rel"]))
    else:  # damped, released from rest at t = 0
        wd = omega_d(o)
        c = o["s"] * math.exp(-o["zeta"] * w * t) * (math.cos(wd * t) + o["zeta"] * w / wd * math.sin(wd * t))
    if "L" in o:                       # pendulum: theta = theta0 * c, sideways offset L sin(theta)
        return o["L"] * math.sin(theta0(o) * c)
    return A * c


def u_at(o, t):
    return _phase_u(o, t)


def v_sign(o, t):
    """Sign of du/dt (0 while held)."""
    m, w = o["mode"], o["omega"]
    if m == "free":
        x = -math.sin(w * t + o["phi"])
    elif m == "held":
        x = 0.0 if t <= o["t_rel"] else -o["s"] * math.sin(w * (t - o["t_rel"]))
    else:
        x = -o["s"] * math.sin(omega_d(o) * t)
    return (x > 0) - (x < 0)


def crossings(o, t0, t1):
    """Times in [t0, t1] at which the mass passes its rest mark (u = 0)."""
    m, w = o["mode"], o["omega"]
    out = []
    if m == "free":
        n = math.ceil((w * t0 + o["phi"] - math.pi / 2) / math.pi)
        while True:
            t = (math.pi / 2 + n * math.pi - o["phi"]) / w
            if t > t1:
                break
            if t >= t0:
                out.append(t)
            n += 1
    elif m == "held":
        n = 0
        while True:
            t = o["t_rel"] + (math.pi / 2 + n * math.pi) / w
            if t > t1:
                break
            if t >= t0:
                out.append(t)
            n += 1
    else:
        wd = omega_d(o)
        d = math.atan2(o["zeta"] * w / wd, 1.0)
        n = 0
        while True:
            t = (d + math.pi / 2 + n * math.pi) / wd
            if t > t1:
                break
            if t >= t0:
                out.append(t)
            n += 1
    return out


def turns(o, t0, t1):
    """[(t, side)] turning points (the mass stops and reverses) in [t0, t1]; side = sign of u there.
    A held mass's release is not a turning point."""
    m, w = o["mode"], o["omega"]
    out = []
    if m == "free":
        n = math.ceil((w * t0 + o["phi"]) / math.pi)
        while True:
            t = (n * math.pi - o["phi"]) / w
            if t > t1:
                break
            if t >= t0:
                out.append((t, 1 if n % 2 == 0 else -1))
            n += 1
    else:
        base, step = (o["t_rel"], math.pi / w) if m == "held" else (0.0, math.pi / omega_d(o))
        n = 1
        while True:
            t = base + n * step
            if t > t1:
                break
            if t >= t0:
                out.append((t, o["s"] * (1 if n % 2 == 0 else -1)))
            n += 1
    return out


def period(o):
    return 2 * math.pi / (omega_d(o) if o["mode"] == "damped" else o["omega"])


def n_frames(scene):
    return scene["frames"]


def duration(scene):
    return scene["frames"] / FPS


def clock_tenths(frame: int) -> int:
    return frame * 10 // FPS


# ------------------------------------------------------------------ layout helpers (px)
def footprint(scene, o):
    """Axis-aligned box [x0, y0, x1, y1] covering everything this oscillator ever draws
    (mass over its whole swing, ruler, spring/string, fixture)."""
    h = o["size"] / 2
    rx, ry = o["rest"]
    ax, ay = o["anchor"]
    reach = o["A"] + h
    nt = o["n_ticks"] * scene["tick"]
    if scene["orient"] == "horizontal":
        return [ax - 6, ry - max(h, 30), rx + max(reach, nt) + 4, o["ruler"] + 26]
    if scene["orient"] == "vertical":
        return [rx - max(h, 30), ay - 4, o["ruler"] + 26, ry + max(reach, nt) + 4]
    return [rx - max(reach, nt) - 4, ay - 4, rx + max(reach, nt) + 4, o["ruler"] + 26]


def _sample_periods(rng, n, lo, hi):
    for _ in range(200):
        ts = sorted(rng.uniform(lo, hi) for _ in range(n))
        if all(b / a >= 1 + PERIOD_GAP for a, b in zip(ts, ts[1:])):
            rng.shuffle(ts)
            return ts
    return None


def _amp_ticks(rng, tick, max_px, kmin=1):
    kmax = int((max_px / tick) + TICK_EPS)
    if kmax < kmin:
        return None
    for _ in range(20):
        k = rng.randint(kmin, kmax)
        a = (k + rng.uniform(-TICK_EPS, TICK_EPS)) * tick
        if 30 <= a <= max_px:
            return a
    return None


def _style(rng, n):
    theme = rng.choice(("light", "light", "dark"))
    T = THEMES[theme]
    cols = rng.sample(sorted(COLOURS), n)
    return theme, {
        "theme": theme, "bg": list(rng.choice(T["bg"])), "bg_pattern": rng.choice(("plain", "gradient", "grid", "dots")),
        "bg_step": rng.randint(40, 90), "ink": list(rng.choice(T["ink"])), "rest_col": list(rng.choice(T["rest"])),
        "spring": {"col": list(rng.choice(T["spring"])), "style": rng.choice(SPRING_STYLES),
                   "coils": rng.randint(6, 14), "width": rng.randint(12, 24), "lw": rng.randint(3, 5)},
        "fixture": list(rng.choice(T["fixture"])), "fixture_style": rng.choice(("plain", "hatched")),
        "tick_len": rng.randint(9, 13), "decor": T["decor"],
    }, cols


def _objects(rng, n, cols):
    out = []
    for i in range(n):
        out.append({"id": i, "colour": cols[i], "shape": rng.choice(SHAPES), "pattern": rng.choice(PATTERNS)})
    return out


# ------------------------------------------------------------------ per-orientation layouts
def _layout_springs(rng, scene, objs, periods, damped, held):
    orient, tick, dur = scene["orient"], scene["tick"], duration(scene)
    n = len(objs)
    if orient == "horizontal":
        top, bot = rng.uniform(70, 120), rng.uniform(650, 690)
        pitch = (bot - top) / n
        if pitch < 96:
            return False
        wall = rng.uniform(262, 330)
        scene["fixture"] = {"kind": "wall", "x": round(wall, 2), "y0": round(top, 2), "y1": round(bot, 2)}
        for i, o in enumerate(objs):
            cy = top + pitch * (i + 0.5) - 14
            o["size"] = rng.randint(36, int(min(64, pitch - 44)))
            if damped:
                o["size"] = max(o["size"], 44)
            w = 2 * math.pi / periods[i]
            room = (1250 - (wall + 200)) / 2 - o["size"] / 2 - 8
            a = _amp_ticks(rng, tick, min(room, VMAX / w, 6.15 * tick), 2 if damped else 1)
            if a is None:
                return False
            lo, hi = wall + 190 + a + o["size"] / 2, 1250 - a - o["size"] / 2 - tick
            if hi < lo:
                return False
            rx = rng.uniform(lo, hi)
            o.update(A=a, omega=w, rest=[round(rx, 2), round(cy, 2)], anchor=[round(wall, 2), round(cy, 2)],
                     ruler=round(cy + o["size"] / 2 + 12, 2))
    else:
        ceil_y = rng.uniform(40, 80)
        left, right = rng.uniform(270, 330), rng.uniform(1180, 1250)
        pitch = (right - left) / n
        if pitch < 120:
            return False
        scene["fixture"] = {"kind": "ceiling", "y": round(ceil_y, 2), "x0": round(left, 2), "x1": round(right, 2)}
        for i, o in enumerate(objs):
            cx = left + pitch * (i + 0.5) - 16
            o["size"] = rng.randint(36, int(min(64, pitch - 56)))
            if damped:
                o["size"] = max(o["size"], 44)
            w = 2 * math.pi / periods[i]
            room = (690 - (ceil_y + 150)) / 2 - o["size"] / 2 - 6
            a = _amp_ticks(rng, tick, min(room, VMAX / w, 6.15 * tick), 2 if damped else 1)
            if a is None:
                return False
            lo, hi = ceil_y + 140 + a + o["size"] / 2, 696 - a - o["size"] / 2 - tick
            if hi < lo:
                return False
            ry = rng.uniform(lo, hi)
            o.update(A=a, omega=w, rest=[round(cx, 2), round(ry, 2)], anchor=[round(cx, 2), round(ceil_y, 2)],
                     ruler=round(cx + o["size"] / 2 + 14, 2))
    for o in objs:
        o["n_ticks"] = math.ceil(o["A"] / tick) + 1
        o["m"] = round(rng.uniform(0.4, 3.0), 4)
        o["k"] = o["m"] * o["omega"] ** 2
        o["zeta"] = rng.uniform(0.10, 0.32) / o["omega"] if damped else 0.0    # decay rate 0.10-0.32 per s
        o["b"] = 2 * o["zeta"] * math.sqrt(o["k"] * o["m"])
        if damped:
            o["mode"], o["s"], o["phi"], o["t_rel"] = "damped", rng.choice((1, -1)), 0.0, 0.0
        elif held:
            o["mode"], o["s"], o["phi"] = "held", rng.choice((1, -1)), 0.0
        else:
            o["mode"], o["s"], o["phi"], o["t_rel"] = "free", 1, rng.uniform(0, 2 * math.pi), 0.0
    if held:
        lo, hi = 0.6, min(4.6, dur - 2.6)
        for _ in range(100):
            rel = sorted(rng.randint(round(lo * FPS), round(hi * FPS)) for _ in objs)
            rel = [f for f in rel if (f * 10 // FPS) == ((f + 1) * 10 // FPS)]
            if len(rel) == len(objs) and all(b - a >= 10 for a, b in zip(rel, rel[1:])):
                break
        else:
            return False
        rng.shuffle(rel)
        for o, f in zip(objs, rel):
            o["t_rel"] = f / FPS
    return True


def _layout_pendulums(rng, scene, objs):
    tick, n = scene["tick"], len(objs)
    piv_y = rng.uniform(40, 90)
    g = 39.478 * 380 / rng.uniform(1.2, 2.0) ** 2           # px/s^2: a 380 px string has a 1.2-2.0 s period
    scene["g"] = g
    for _ in range(200):
        Ls = sorted(rng.uniform(200, 560) for _ in range(n))
        if all(b / a >= (1 + PERIOD_GAP) ** 2 for a, b in zip(Ls, Ls[1:])):
            break
    else:
        return False
    rng.shuffle(Ls)
    widths = []
    for o, L in zip(objs, Ls):
        o["size"] = rng.randint(32, 52)
        o["L"] = L
        o["omega"] = math.sqrt(g / L)
        ruler_y = piv_y + L + o["size"] / 2 + 12
        if ruler_y + 26 > 700:
            o["L"] = L = 700 - 26 - 12 - o["size"] / 2 - piv_y
            o["omega"] = math.sqrt(g / L)
        a = _amp_ticks(rng, tick, min(0.26 * L, VMAX / o["omega"], 4.15 * tick))
        if a is None:
            return False
        o["A"] = a
        o["n_ticks"] = math.ceil(a / tick) + 1
        widths.append(max(a + o["size"] / 2, o["n_ticks"] * tick) + 4)
    ts = sorted(2 * math.pi / o["omega"] for o in objs)
    if not all(b / a >= 1 + PERIOD_GAP for a, b in zip(ts, ts[1:])):
        return False
    free = (AREA[2] - 8) - (AREA[0] + 10) - 2 * sum(widths)
    if free < 24 * (n + 1):
        return False
    gaps = [rng.random() + 0.2 for _ in range(n + 1)]
    x = AREA[0] + 10
    for i, (o, wdt) in enumerate(zip(objs, widths)):
        x += free * gaps[i] / sum(gaps) + wdt
        o.update(anchor=[round(x, 2), round(piv_y, 2)], rest=[round(x, 2), round(piv_y + o["L"], 2)],
                 ruler=round(piv_y + o["L"] + o["size"] / 2 + 12, 2), mode="free", s=1,
                 phi=rng.uniform(0, 2 * math.pi), t_rel=0.0, zeta=0.0, m=round(rng.uniform(0.1, 2.0), 4))
        x += wdt
    scene["fixture"] = {"kind": "bar", "y": round(piv_y, 2), "x0": round(objs[0]["anchor"][0] - widths[0], 2),
                        "x1": round(objs[-1]["anchor"][0] + widths[-1], 2)}
    return True


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    for attempt in range(1, 500):
        if scenario == "pendulum":
            orient, n = "pendulum", rng.randint(3, 5)
        else:
            orient, n = rng.choice(("horizontal", "vertical")), rng.randint(3, 6)
            if scenario in ("horizontal", "vertical"):
                orient = scenario
        theme, style, cols = _style(rng, n)
        scene = {"scenario": scenario, "orient": orient, "frames": rng.randint(MIN_FRAMES, MAX_FRAMES),
                 "tick": rng.randint(22, 34) if orient == "pendulum" else rng.randint(24, 44), "style": style}
        objs = _objects(rng, n, cols)
        if orient == "pendulum":
            ok = _layout_pendulums(rng, scene, objs)
        else:
            periods = _sample_periods(rng, n, 0.8, 3.0)
            ok = periods is not None and _layout_springs(rng, scene, objs, periods, scenario == "damped",
                                                         scenario == "staggered")
        if not ok:
            continue
        for o in objs:                  # round to stored precision first, so everything downstream uses the same numbers
            for k in ("A", "omega", "phi", "zeta", "k", "b", "L"):     # t_rel stays an exact frame / 24
                if k in o:
                    o[k] = round(o[k], 6)
        scene["objects"] = objs
        if "g" in scene:
            scene["g"] = round(scene["g"], 4)
        scene["build_attempts"] = attempt
        if check(scene):
            continue
        return scene
    raise RuntimeError("no valid scene")


# ------------------------------------------------------------------ consistency check
def _overlap(a, b, gap=0):
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def check(scene: dict) -> list[str]:
    errs = []
    objs, tick = scene["objects"], scene["tick"]
    if not MIN_FRAMES <= scene["frames"] <= MAX_FRAMES:
        errs.append("clip length out of range")
    if scene["scenario"] not in SCENARIOS:
        errs.append("unknown scenario")
    looks = [o["colour"] for o in objs]
    if len(set(looks)) != len(looks):
        errs.append("two objects share a colour")
    ts = sorted(period(o) for o in objs)
    if not all(b / a >= 1 + PERIOD_GAP - 1e-9 for a, b in zip(ts, ts[1:])):
        errs.append("two periods within 10 %")
    for o in objs:
        r = o["A"] / tick
        if abs(r - round(r)) > TICK_EPS + 1e-9 or round(r) < 1:
            errs.append(f"{o['colour']}: amplitude not near a whole tick count")
        if o["A"] * o["omega"] > VMAX + 1e-6:
            errs.append(f"{o['colour']}: moves too fast")
        if o["size"] < 20 or o["A"] < 30:
            errs.append(f"{o['colour']}: too small")
        if o["mode"] == "damped" and not (0 < o["zeta"] < 0.2 and 0.1 - 1e-6 <= o["zeta"] * o["omega"] <= 0.32 + 1e-6):
            errs.append("bad damping")
        if "L" in o and o["A"] > 0.27 * o["L"]:
            errs.append("pendulum swing too wide for simple harmonic motion")
    if scene["scenario"] == "staggered":
        rel = sorted(o["t_rel"] for o in objs)
        if rel[0] < EDGE or rel[-1] > duration(scene) - 2.0 or any(b - a < 0.4 - 1e-9 for a, b in zip(rel, rel[1:])):
            errs.append("release times too close to each other or to the clip ends")
    boxes = [footprint(scene, o) for o in objs]
    for i in range(len(boxes)):
        b = boxes[i]
        if b[0] < AREA[0] - 2 or b[1] < 8 or b[2] > 1276 or b[3] > 716:
            errs.append(f"{objs[i]['colour']}: leaves the frame")
        if _overlap(b, [CLOCK_BOX[0] - 10, CLOCK_BOX[1] - 10, CLOCK_BOX[2] + 10, CLOCK_BOX[3] + 10]):
            errs.append("an object can reach the clock")
        for j in range(i + 1, len(boxes)):
            if scene["orient"] == "pendulum" and _overlap(b, boxes[j], 15):
                errs.append("two pendulums' swings overlap")
            if scene["orient"] != "pendulum" and _overlap(_mass_box(objs[i], scene), _mass_box(objs[j], scene), 15):
                errs.append("two masses' paths come within 15 px")
    return errs


def _mass_box(o, scene):
    h = o["size"] / 2
    rx, ry = o["rest"]
    if scene["orient"] == "horizontal":
        return [rx - o["A"] - h, ry - h, rx + o["A"] + h, o["ruler"] + 14]
    return [rx - h, ry - o["A"] - h, o["ruler"] + 14, ry + o["A"] + h]
