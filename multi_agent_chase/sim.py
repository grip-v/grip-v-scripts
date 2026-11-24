"""Scene generation and simulation for multi_agent_chase (v3).

4-9 patterned balls move inside a walled arena. Every ball has a unique
colour AND a unique pattern, so "the blue striped ball" names exactly one
ball. Catchers look like any other ball: who hunts whom is read from the
motion. A caught ball vanishes at once and its catcher grows smoothly to the
radius whose area is the two areas added together (and slows down a little).

Scenarios (scene["scenario"]):
- open       1-3 hunters chase prey in an empty arena.
- pillars    1-3 round grey pillars stand in the arena; every ball steers around them.
- safe_zone  a shaded circle is a refuge: prey inside it cannot be caught, hunters keep
             out of it, threatened prey run toward it.
- bigger     no fixed roles: any ball eats a strictly smaller ball it touches, chases
             smaller balls and flees bigger ones (sizes start >= 15 % apart).

Motion is acceleration-limited steering at DT = 1/48 s (two steps per video
frame): each step the velocity moves toward the desired velocity by at most
A_MAX * DT, so headings and speeds change smoothly. Walls, pillars and (for
hunters) the refuge push balls away smoothly from CLEAR_M px; `check`
rejects any clip where a ball ever accelerates harder than the limit.

The clip is `frames` frames at 24 fps (7-10 s). Frame f shows the state at
t = f / 24 s and the clock reads floor(f * 10 / 24) tenths. A catch that
happens between frames f-1 and f is "at frame f": the victim is drawn on
f-1 and gone from f on.

Everything is a pure function of the scene dict, which is a pure function of
the seeded rng passed to `build`. `run(scene, mods)` re-simulates with a
changed rule (counterfactuals): ("remove", i), ("immune", i), ("nogrow",),
("nozone",), ("nopillars",), ("slow", i).
"""
from __future__ import annotations
import functools
import json
import math
import random

FPS = 24
SUB = 2                                   # simulation steps per frame
DT = 1 / (FPS * SUB)
SIZE = (1280, 720)
COLORS = {"red": (214, 40, 40), "orange": (245, 130, 15), "yellow": (238, 206, 25), "lime": (140, 205, 40),
          "green": (25, 135, 55), "teal": (15, 150, 150), "sky": (70, 175, 240), "blue": (35, 70, 205),
          "purple": (125, 55, 190), "pink": (238, 90, 170), "brown": (130, 80, 35), "black": (35, 35, 40)}
PATTERNS = ["solid", "striped", "dotted", "checkered", "ringed", "two-tone", "quartered", "grid", "starred"]
SCENARIOS = ("open", "pillars", "safe_zone", "bigger")
STYLES = [  # (backdrop, floor, wall)
    ((232, 230, 224), (248, 247, 242), (95, 98, 108)),
    ((214, 224, 232), (240, 245, 249), (70, 90, 115)),
    ((222, 232, 216), (244, 249, 240), (80, 105, 75)),
    ((236, 226, 214), (250, 245, 238), (120, 90, 65)),
    ((60, 64, 72), (236, 236, 236), (30, 32, 38)),
    ((228, 222, 236), (246, 243, 250), (95, 80, 120)),
]
FLOOR_TEX = ("plain", "grid", "dots")
WALL_STYLE = ("solid", "double", "thick")
LAYOUTS = ("scatter", "split", "ring")
CLOCK_BOX = (1098, 8, 1268, 58)            # fixed; the arena always starts below it
FRAMES = (168, 240)                        # 7-10 s
A_MAX = 520.0                              # px/s^2
CLEAR_M = 95.0                             # walls / pillars push from this far
SEP = 22.0                                 # balls that can't eat each other keep this gap
FLEE_R = 220.0
SIGHT = 460.0
REST_S = 0.9                               # a catcher wanders this long after a catch
GROW_RATE = 45.0                           # px/s of radius growth after a catch
FIRST_F = 20                               # first catch >= 0.83 s
END_PAD_F = 24                             # last catch >= 1 s before the end (growth has finished)
GAP_F = 10                                 # catches >= 10 frames (0.42 s) apart
CLEAR = 20.0                               # px between a victim and any ball other than its catcher
EAT_RATIO = 1.12                           # bigger: catcher radius >= 1.12 x victim radius
SIZE_MARGIN = 1.12                         # radius ratio for "bigger" questions
TAU = 2 * math.pi


def label(b: dict) -> str:
    return f"the {b['color']} {b['pattern']} ball"


def duration(scene) -> float:
    return scene["frames"] / FPS


def clock_tenths(f: int) -> int:
    return f * 10 // FPS


def _wander(b, t):
    return b["h0"] + b["a1"] * math.sin(b["w1"] * t + b["p1"]) + b["a2"] * math.sin(b["w2"] * t + b["p2"])


def _key(scene):
    return json.dumps(scene, sort_keys=True)


def _push(clear, w=3.5):
    return w * (1 - max(0.0, clear) / CLEAR_M) ** 2 if clear < CLEAR_M else 0.0


@functools.lru_cache(maxsize=256)
def _run(key: str, mods: tuple = ()):
    """track[f] = tuple of (x, y, r, alive) per ball for f = 0..frames-1;
    catches = [(frame, catcher, victim, catcher_r, victim_r)]."""
    scene = json.loads(key)
    bs = scene["balls"]
    n = len(bs)
    removed = {m[1] for m in mods if m[0] == "remove"}
    immune = {m[1] for m in mods if m[0] == "immune"}
    slow = {m[1] for m in mods if m[0] == "slow"}
    grow = ("nogrow",) not in mods
    zone = None if ("nozone",) in mods else scene.get("zone")
    pillars = [] if ("nopillars",) in mods else scene.get("pillars", [])
    bigger = scene["scenario"] == "bigger"
    x0, y0, x1, y1 = scene["arena"]
    pos = [[b["x"], b["y"]] for b in bs]
    vel = [[b["v"] * math.cos(b["h0"]), b["v"] * math.sin(b["h0"])] for b in bs]
    rad = [float(b["r"]) for b in bs]
    goal = rad[:]
    alive = [i not in removed for i in range(n)]
    hunter = [b["role"] == "hunter" for b in bs]
    rest = [-1.0] * n
    vmul = [0.5 if i in slow else 1.0 for i in range(n)]

    def in_zone(i):
        return zone is not None and math.dist(pos[i], (zone["x"], zone["y"])) <= zone["r"] - rad[i]

    def can_eat(i, j):
        if not (alive[i] and alive[j]) or i == j or j in immune:
            return False
        if bigger:
            return rad[i] > rad[j]
        return hunter[i] and not hunter[j] and not in_zone(j)

    track, catches = [], []
    frames = scene["frames"]
    for f in range(frames):
        track.append(tuple((round(pos[i][0], 2), round(pos[i][1], 2), round(rad[i], 2), alive[i]) for i in range(n)))
        if f == frames - 1:
            break
        for s in range(SUB):
            t = (f * SUB + s) * DT
            live = [i for i in range(n) if alive[i]]
            for i in live:
                b = bs[i]
                x, y = pos[i]
                food = [j for j in live if can_eat(i, j) and math.dist(pos[i], pos[j]) < SIGHT]
                if food and t >= rest[i]:
                    j = min(food, key=lambda j: math.dist(pos[i], pos[j]))
                    dx, dy = pos[j][0] - x, pos[j][1] - y
                    d = math.hypot(dx, dy) or 1.0
                    dx, dy = dx / d, dy / d
                else:
                    h = _wander(b, t)
                    dx, dy = math.cos(h), math.sin(h)
                threat = False
                for k in live:
                    if k == i:
                        continue
                    ex, ey = x - pos[k][0], y - pos[k][1]
                    d = math.hypot(ex, ey) or 1.0
                    gap = d - rad[i] - rad[k]
                    if (hunter[k] and not hunter[i] and not bigger) or (bigger and rad[k] > rad[i]):
                        if d < FLEE_R:
                            w = 3.0 * (1 - d / FLEE_R) ** 2
                            dx += w * ex / d
                            dy += w * ey / d
                            threat = True
                    elif not can_eat(i, k) and gap < SEP:
                        w = 2.0 * (1 - max(0.0, gap) / SEP) ** 2
                        dx += w * ex / d
                        dy += w * ey / d
                if zone is not None:
                    zx, zy = zone["x"] - x, zone["y"] - y
                    d = math.hypot(zx, zy) or 1.0
                    if hunter[i]:
                        dx -= _push(d - zone["r"] - rad[i], 3.0) * zx / d
                        dy -= _push(d - zone["r"] - rad[i], 3.0) * zy / d
                    elif threat:
                        dx += 1.2 * zx / d
                        dy += 1.2 * zy / d
                for p in pillars:
                    px, py = x - p["x"], y - p["y"]
                    d = math.hypot(px, py) or 1.0
                    w = _push(d - p["r"] - rad[i])
                    dx += w * px / d
                    dy += w * py / d
                dx += _push(x - x0 - rad[i]) - _push(x1 - x - rad[i])
                dy += _push(y - y0 - rad[i]) - _push(y1 - y - rad[i])
                m = math.hypot(dx, dy) or 1.0
                v = b["v"] * vmul[i] * math.sqrt(b["r"] / rad[i])        # bigger balls are slower
                ax, ay = v * dx / m - vel[i][0], v * dy / m - vel[i][1]
                a = math.hypot(ax, ay)
                lim = A_MAX * DT
                if a > lim:
                    ax, ay = ax * lim / a, ay * lim / a
                vel[i][0] += ax
                vel[i][1] += ay
            for i in live:
                pos[i][0] = min(max(pos[i][0] + vel[i][0] * DT, x0 + rad[i]), x1 - rad[i])
                pos[i][1] = min(max(pos[i][1] + vel[i][1] * DT, y0 + rad[i]), y1 - rad[i])
                for p in pillars:              # hard pillar surface: never needed in an accepted clip
                    px, py = pos[i][0] - p["x"], pos[i][1] - p["y"]
                    d = math.hypot(px, py) or 1.0
                    if d < p["r"] + rad[i]:
                        pos[i][0] = p["x"] + px / d * (p["r"] + rad[i])
                        pos[i][1] = p["y"] + py / d * (p["r"] + rad[i])
                if rad[i] < goal[i]:
                    rad[i] = min(goal[i], rad[i] + GROW_RATE * DT)
            for i in range(n):
                for j in range(n):
                    if can_eat(i, j) and math.dist(pos[i], pos[j]) <= rad[i] + rad[j]:
                        catches.append((f + 1, i, j, round(rad[i], 2), round(rad[j], 2)))
                        alive[j] = False
                        rest[i] = t + DT + REST_S
                        if grow:
                            goal[i] = math.sqrt(goal[i] ** 2 + rad[j] ** 2)
    return tuple(track), tuple(catches)


def run(scene, mods: tuple = ()):
    return _run(_key(scene), tuple(mods))


def track(scene, mods: tuple = ()):
    return run(scene, mods)[0]


def catches(scene, mods: tuple = ()) -> list:
    return list(run(scene, mods)[1])


def state(scene, f: int):
    tr = track(scene)
    return tr[min(len(tr) - 1, max(0, f))]


def run_errors(scene, mods: tuple = (), need=1) -> list:
    """Ambiguity checks shared by the factual clip and every re-simulated what-if."""
    errs = []
    tr, cs = run(scene, mods)
    fr = [c[0] for c in cs]
    if len(cs) < need:
        errs.append(f"only {len(cs)} catches")
    if fr and (fr[0] < FIRST_F or fr[-1] > scene["frames"] - END_PAD_F):
        errs.append("catch too near the start or the end")
    if any(b - a < GAP_F for a, b in zip(fr, fr[1:])):
        errs.append("two catches too close together")
    for f, h, p, rh, rp in cs:
        row = tr[f - 1]
        x, y, r, _ = row[p]
        if scene["scenario"] == "bigger" and rh < EAT_RATIO * rp:
            errs.append("catcher and victim nearly the same size")
        for i, (xi, yi, ri, ai) in enumerate(row):
            if i not in (h, p) and ai and math.dist((x, y), (xi, yi)) - ri - r < CLEAR:
                errs.append(f"another ball is near {label(scene['balls'][p])} when it is caught")
                break
    return errs


def check(scene) -> list:
    errs = []
    bs = scene["balls"]
    n = len(bs)
    if not 4 <= n <= 9:
        errs.append(f"{n} balls")
    if len({b["color"] for b in bs}) != n or len({b["pattern"] for b in bs}) != n:
        errs.append("colours or patterns repeat")
    if scene["scenario"] not in SCENARIOS:
        errs.append("unknown scenario")
    if not FRAMES[0] <= scene["frames"] <= FRAMES[1]:
        errs.append("clip length out of range")
    if scene["arena"][1] < CLOCK_BOX[3] + 8:
        errs.append("arena under the clock")
    errs += run_errors(scene, need=2)
    tr = track(scene)
    lim = A_MAX * 1.1
    for i in range(n):
        if not tr[0][i][3]:
            errs.append("a ball is missing at the start")
        gone = False
        for k in range(len(tr)):
            if not tr[k][i][3]:
                gone = True
            elif gone:
                errs.append("a ball reappears")
                break
        for k in range(2, len(tr)):
            if not tr[k][i][3]:
                break
            (xa, ya, *_), (xb, yb, *_), (xc, yc, *_) = tr[k - 2][i], tr[k - 1][i], tr[k][i]
            if math.hypot(xc - 2 * xb + xa, yc - 2 * yb + ya) * FPS * FPS > lim:
                errs.append(f"{label(bs[i])} jerks at frame {k}")
                break
    return errs


# ------------------------------------------------------------------ scene building
def _place(rng, scene, r, role, out, layout):
    x0, y0, x1, y1 = scene["arena"]
    m = CLEAR_M * 0.8 + r
    for _ in range(300):
        if layout == "split":
            half = (x0 + x1) / 2
            left = (role == "hunter") == scene["hunter_left"]
            xa, xb = (x0 + m, half - 30) if left else (half + 30, x1 - m)
            x, y = rng.uniform(xa, xb), rng.uniform(y0 + m, y1 - m)
        elif layout == "ring":
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rr = 0.0 if role == "hunter" and not out else rng.uniform(0.55, 0.95)
            a = rng.uniform(0, TAU)
            x, y = cx + rr * ((x1 - x0) / 2 - m) * math.cos(a), cy + rr * ((y1 - y0) / 2 - m) * math.sin(a)
        else:
            x, y = rng.uniform(x0 + m, x1 - m), rng.uniform(y0 + m, y1 - m)
        far = 100 if scene["scenario"] == "bigger" else 300
        if all(math.hypot(x - o["x"], y - o["y"]) >= r + o["r"] +
               (far if (role == "hunter") != (o["role"] == "hunter") or scene["scenario"] == "bigger" else 45)
               for o in out) and \
                all(math.hypot(x - p["x"], y - p["y"]) >= r + p["r"] + 40 for p in scene.get("pillars", [])) and \
                (scene.get("zone") is None or
                 math.hypot(x - scene["zone"]["x"], y - scene["zone"]["y"]) >= r + scene["zone"]["r"] + 40):
            return round(x, 1), round(y, 1)
    return None


def _attempt(rng: random.Random, scenario: str) -> dict | None:
    w = rng.randint(980, 1220)
    h = rng.randint(500, 640 - 0)
    x0 = rng.randint(30, SIZE[0] - 30 - w)
    y0 = rng.randint(CLOCK_BOX[3] + 12, SIZE[1] - 20 - h) if SIZE[1] - 20 - h > CLOCK_BOX[3] + 12 else CLOCK_BOX[3] + 12
    h = min(h, SIZE[1] - 20 - y0)
    arena = [x0, y0, x0 + w, y0 + h]
    scene = {"scenario": scenario, "frames": rng.randint(*FRAMES), "arena": arena,
             "style": {"palette": rng.randrange(len(STYLES)), "floor": rng.choice(FLOOR_TEX),
                       "wall": rng.choice(WALL_STYLE)},
             "layout": rng.choice(LAYOUTS), "hunter_left": rng.random() < 0.5}
    if scenario == "pillars":
        ps = []
        for _ in range(rng.randint(1, 3)):
            pr = rng.randint(32, 60)
            for _ in range(100):
                px, py = rng.uniform(x0 + 160 + pr, x0 + w - 160 - pr), rng.uniform(y0 + 120 + pr, y0 + h - 120 - pr)
                if all(math.hypot(px - q["x"], py - q["y"]) >= pr + q["r"] + 160 for q in ps):
                    ps.append({"x": round(px, 1), "y": round(py, 1), "r": pr})
                    break
        scene["pillars"] = ps
    if scenario == "safe_zone":
        zr = rng.randint(85, 120)
        scene["zone"] = {"x": round(rng.uniform(x0 + zr + 110, x0 + w - zr - 110), 1),
                         "y": round(rng.uniform(y0 + zr + 90, y0 + h - zr - 90), 1), "r": zr}
    n = rng.randint(5, 9) if scenario in ("safe_zone", "bigger") else rng.randint(4, 9)
    if scenario == "bigger":
        n = min(n, 8)
        roles = ["prey"] * n
        radii = [rng.uniform(13, 17)]
        for _ in range(n - 1):
            radii.append(radii[-1] * rng.uniform(1.16, 1.24))
        rng.shuffle(radii)
        base = rng.uniform(165, 215)
        speeds = [base * rng.uniform(0.9, 1.1) for _ in range(n)]
    else:
        n_h = rng.choice([1, 1, 2]) if n <= 5 else rng.choice([1, 2, 2, 3])
        if scenario != "open":
            n_h = min(n_h, 2)
        roles = ["hunter"] * n_h + ["prey"] * (n - n_h)
        rng.shuffle(roles)
        hv = rng.uniform(200, 260)
        radii = [rng.uniform(24, 32) if ro == "hunter" else rng.uniform(15, 26) for ro in roles]
        speeds = [hv * rng.uniform(0.95, 1.08) if ro == "hunter" else hv * rng.uniform(0.7, 0.9) for ro in roles]
    cols = rng.sample(list(COLORS), n)
    pats = rng.sample(PATTERNS, n)
    balls = []
    for k in range(n):
        r = round(radii[k], 1)
        xy = _place(rng, scene, r, roles[k], balls, scene["layout"])
        if xy is None:
            return None
        balls.append({"role": roles[k], "color": cols[k], "pattern": pats[k], "r": r, "v": round(speeds[k], 2),
                      "x": xy[0], "y": xy[1], "h0": round(rng.uniform(0, TAU), 4),
                      "a1": round(rng.uniform(0.6, 1.4), 3), "w1": round(rng.uniform(0.3, 0.9), 3),
                      "p1": round(rng.uniform(0, TAU), 3), "a2": round(rng.uniform(0.2, 0.6), 3),
                      "w2": round(rng.uniform(0.9, 1.8), 3), "p2": round(rng.uniform(0, TAU), 3)})
    scene["balls"] = balls
    return scene


def _has_whatif(scene) -> bool:
    """At least one re-simulated what-if is unambiguous (the ladder needs one L5 question)."""
    cs = catches(scene)
    mods = [(("nogrow",),), (("remove", cs[0][1]),), (("slow", cs[0][1]),), (("immune", cs[0][2]),)]
    return any(not run_errors(scene, m, need=0) for m in mods)


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    for attempt in range(1, 2000):
        scene = _attempt(rng, scenario)
        if scene is not None and not check(scene) and _has_whatif(scene):
            scene["build_attempts"] = attempt
            return scene
    raise RuntimeError("no valid scene")
