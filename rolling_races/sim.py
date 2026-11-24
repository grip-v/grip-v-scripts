"""rolling_races scene sampler and physics.

2-10 frictionless balls start together from one start point with the same
push and run down their own colour-coded tracks to one finish hole (W m
across, D m lower). Each ball rides only its own track.

A track may have GAPS: a missing stretch [x0, x1] with a take-off lip at its
near edge that launches the ball `lip` degrees above the track's direction.
The ball flies as a projectile. If its path comes back down to the track line
beyond the far edge x1 it LANDS: the velocity component along the track is
kept, the part into the track is lost, and it rolls on. If the path drops
through the track line inside the gap it FALLS and comes to rest on the pit
floor at the bottom of the screen, out of the race. A ball that reaches the
finish drops into the hole and fills its own slot in the finish rack beside
it (the rack's slot order is fixed and random, so it never shows the order).

Scenarios (scene["scenario"]):
  plain     no gaps at all: a pure race of shapes (about a third of clips)
  jumps     1 to about half of the tracks have one gap
  double    1-3 tracks have two gaps one after another; others may have one
  all_gaps  every track has a gap
  steep     a tall, narrow arena (W 5.5-7 m, D 4-5 m) with 1 to about half
            of the tracks gapped

Physics: on the track energy is conserved, v^2 = c + 2 g y (y = depth), with
c fixed per stretch; the time along a stretch is the integral of ds / v
(Simpson's rule on each piece of a dense polyline). The flight is the closed-form parabola
(bisection for where it meets the track line). The race plays `slow` times
slower than real time; the clock shows clip time (frame * 10 // 24 tenths),
the balls are released at clock time t0 (a frame instant).

World: x from the start (0) to the finish (W), depth y downward from the
start. Pure function of the rng passed in.
"""
from __future__ import annotations
import bisect
import functools
import math
import random

FPS = 24
MAX_FRAMES = 240
SIZE = (1280, 720)
G = 9.8
N = 2000                      # polyline segments per track
EDGE_S = 0.5                  # question events stay this far inside the clip
FAMILIES = ("straight", "cycloid", "sag", "slow_start", "steep_start", "wavy", "bend")
SCENARIOS = ("plain", "jumps", "double", "all_gaps", "steep")
SCEN_W = (0.34, 0.18, 0.16, 0.16, 0.16)      # plain ~ a third: about a third of races have no gaps
MARGIN_X = 0.15               # a jump clears or misses its far edge by at least this much (m)
SEP_M = 0.17                  # two tracks never run closer than this for > 20 % of the middle stretch
PUSH_CF = 3.0                 # the what-if push is three times the real one
REST_SEP = 0.5                # fallen balls rest at least this far apart on the pit floor (m)
CLOCK_W, CLOCK_H = 196, 68
PATTERNS = ("solid", "dot", "ring", "cross")
TRACK_STYLES = ("solid", "sleepers", "cased")
BG_PATS = ("plain", "grid", "dots")

# per theme: colour name -> RGB (the names are what the questions use); every pair >= 135 apart (summed |dRGB|)
# and >= 135 from the theme's ink, so no ball looks like the ledge, the gate or the floor line
BALLS = {
    "light": {"red": (215, 38, 38), "blue": (28, 96, 222), "green": (24, 150, 58), "orange": (245, 128, 0),
              "purple": (128, 58, 204), "teal": (0, 168, 170), "pink": (236, 92, 178), "brown": (110, 66, 10),
              "gray": (118, 118, 128), "lime": (130, 190, 20)},
    "dark": {"red": (255, 64, 64), "blue": (70, 140, 255), "green": (60, 210, 90), "orange": (255, 160, 0),
             "purple": (170, 90, 240), "teal": (0, 210, 210), "pink": (255, 110, 200), "yellow": (240, 230, 60),
             "mint": (120, 255, 190), "brown": (176, 112, 56)},
}
THEMES = {
    "light": {"bg": [(242, 239, 232), (232, 236, 242), (238, 242, 232), (246, 240, 234), (236, 234, 240)],
              "ink": (34, 36, 42), "pit": (200, 194, 182), "hole": (24, 24, 28), "rack": (214, 210, 200)},
    "dark": {"bg": [(20, 22, 30), (28, 22, 32), (16, 28, 30), (30, 30, 32), (22, 26, 40)],
             "ink": (232, 232, 238), "pit": (56, 56, 64), "hole": (0, 0, 0), "rack": (46, 48, 58)},
}


# ------------------------------------------------------------------ track shapes
@functools.lru_cache(maxsize=None)
def _cycloid_theta(ratio: float) -> float:
    lo, hi = 0.1, 2 * math.pi - 1e-6
    for _ in range(200):
        mid = (lo + hi) / 2
        if (mid - math.sin(mid)) / (1 - math.cos(mid)) < ratio:
            lo = mid
        else:
            hi = mid
    return lo


def shape(scene: dict, b: dict) -> tuple:
    return (b["family"], b.get("a", 0.0), b.get("k", 0), scene["W"], scene["D"])


def gaps(b: dict) -> tuple:
    return tuple((g["x0"], g["x1"], g["lip"]) for g in b.get("gaps", ()))


@functools.lru_cache(maxsize=512)
def polyline(sh: tuple) -> tuple:
    """(x, depth) points from (0, 0) to (W, D), x strictly increasing."""
    fam, a, k, W, D = sh
    if fam == "cycloid":
        th = _cycloid_theta(W / D)
        r = D / (1 - math.cos(th))
        return tuple((r * (u - math.sin(u)), r * (1 - math.cos(u))) for u in (th * i / N for i in range(N + 1)))
    f = {
        "straight": lambda u: u,
        "sag": lambda u: u + a * math.sin(math.pi * u),
        "slow_start": lambda u: u ** a,
        "steep_start": lambda u: u ** a,
        "wavy": lambda u: u + a * math.sin(k * math.pi * u),
        "bend": lambda u: u + a * math.sin(math.pi * u ** (k / 10)),
    }[fam]
    p = 2 if fam == "steep_start" or (fam == "bend" and k < 10) else 1    # finer near a vertical drop at u = 0
    return tuple((W * u, D * f(u)) for u in ((i / N) ** p for i in range(N + 1)))


@functools.lru_cache(maxsize=512)
def _xs(sh):
    return tuple(p[0] for p in polyline(sh))


def seg_at(sh, x: float) -> int:
    return min(N - 1, max(0, bisect.bisect_right(_xs(sh), x) - 1))


def depth_at(sh, x: float) -> float:
    pts = polyline(sh)
    i = seg_at(sh, x)
    (xa, ya), (xb, yb) = pts[i], pts[i + 1]
    return ya + (x - xa) / (xb - xa) * (yb - ya)


def _ride(sh, xa: float, xb: float, c: float, t0: float):
    """Ride the track from x = xa to x = xb with v^2 = c + 2 g y. Returns [(t, x, y)] at xa, every vertex in
    between and xb, or None if the ball would stall."""
    pts = polyline(sh)
    ia, ib = seg_at(sh, xa), seg_at(sh, xb)
    path = [(xa, depth_at(sh, xa))] + [pts[j] for j in range(ia + 1, ib + 1) if xa < pts[j][0] < xb] + \
        [(xb, depth_at(sh, xb))]
    out = [(t0, path[0][0], path[0][1])]
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        v2a, v2m, v2b = c + 2 * G * y0, c + G * (y0 + y1), c + 2 * G * y1
        if v2a < 0.01 or v2b < 0.01:
            return None
        ds = math.dist((x0, y0), (x1, y1))                 # Simpson's rule on ds / v
        out.append((out[-1][0] + ds * (1 / math.sqrt(v2a) + 4 / math.sqrt(v2m) + 1 / math.sqrt(v2b)) / 6, x1, y1))
    return out


@functools.lru_cache(maxsize=512)
def run(sh: tuple, gp: tuple, v0: float, floor: float):
    """The ball's whole race in physics time (s). Returns a dict with
    phases: ("track", [(t, x, y)], times) | ("fly", x0, y0, vx, vy, t0, t1) | ("rest", x, y, t0),
    finish: time it reaches the finish hole or None, fell: index of the gap it falls through or None,
    leave: [time it reaches the near edge of each gap it gets to], land: [time it lands past each gap it clears],
    floor_t: time a fallen ball reaches the pit floor, rest_x: where it rests,
    margins: [how far (m) each jump cleared or missed its far edge], ok: False if degenerate (stalls,
    overshoots the finish, lands too close to the next gap or the end, or hits its own track from below)."""
    W = sh[3]
    pts = polyline(sh)
    out = {"phases": [], "finish": None, "fell": None, "leave": [], "land": [], "floor_t": None, "rest_x": None,
           "margins": [], "ok": True, "end_t": None, "v_end": None}
    c, x, t = v0 * v0, 0.0, 0.0
    for gi, (gx0, gx1, lip) in enumerate(gp):
        if gx0 <= x + 0.3:
            out["ok"] = False
            return out
        seg = _ride(sh, x, gx0, c, t)
        if seg is None:
            out["ok"] = False
            return out
        out["phases"].append(("track", seg, tuple(p[0] for p in seg)))
        t = seg[-1][0]
        out["leave"].append(t)
        y0 = depth_at(sh, gx0)
        i = seg_at(sh, gx0 - 1e-9)
        phi = math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0]) - math.radians(lip)
        v = math.sqrt(c + 2 * G * y0)
        vx, vy = v * math.cos(phi), v * math.sin(phi)

        def fy(tau):
            return y0 + vy * tau + G * tau * tau / 2
        tau, step = 1e-4, 2e-3
        while True:
            xn = gx0 + vx * (tau + step)
            if xn > W:
                out["ok"] = False
                return out
            if fy(tau + step) >= depth_at(sh, xn):
                break
            tau += step
        lo, hi = tau, tau + step
        for _ in range(60):
            mid = (lo + hi) / 2
            if fy(mid) >= depth_at(sh, gx0 + vx * mid):
                hi = mid
            else:
                lo = mid
        tau = hi
        xc = gx0 + vx * tau
        if xc < gx1:                                       # drops through the gap
            out["fell"] = gi
            out["margins"].append(gx1 - xc)
            t_end = (-vy + math.sqrt(vy * vy + 2 * G * (floor - y0))) / G
            k = tau + 0.005
            while k < t_end:                               # it must stay under its own track
                xk = gx0 + vx * k
                if gx1 <= xk <= W and fy(k) < depth_at(sh, xk) + 0.03:
                    out["ok"] = False
                k += 0.005
            out["phases"].append(("fly", gx0, y0, vx, vy, t, t + t_end))
            out["floor_t"] = t + t_end
            out["rest_x"] = gx0 + vx * t_end
            out["phases"].append(("rest", out["rest_x"], floor, t + t_end))
            out["end_t"] = t + t_end
            return out
        out["margins"].append(xc - gx1)
        nxt = gp[gi + 1][0] if gi + 1 < len(gp) else W
        if xc > nxt - 0.4:
            out["ok"] = False
            return out
        j = seg_at(sh, xc)
        tx, ty = pts[j + 1][0] - pts[j][0], pts[j + 1][1] - pts[j][1]
        along = (vx * tx + (vy + G * tau) * ty) / math.hypot(tx, ty)
        yc = depth_at(sh, xc)
        if along < 0.3:
            out["ok"] = False
            return out
        out["phases"].append(("fly", gx0, y0, vx, vy, t, t + tau))
        t += tau
        out["land"].append(t)
        c, x = along * along - 2 * G * yc, xc
    seg = _ride(sh, x, W, c, t)
    if seg is None:
        out["ok"] = False
        return out
    out["phases"].append(("track", seg, tuple(p[0] for p in seg)))
    out["finish"] = out["end_t"] = seg[-1][0]
    out["v_end"] = math.sqrt(c + 2 * G * sh[4])
    return out


def ball_run(scene, b, v0=None, no_gap=False, straight=False):
    v0 = scene["v0"] if v0 is None else v0
    sh = shape(scene, b)
    if straight:
        sh = ("straight", 0.0, 0, scene["W"], scene["D"])
    return run(sh, () if (no_gap or straight) else gaps(b), v0, scene["floor"])


def world_at(r: dict, tau: float):
    """(x, depth, airborne) at physics time tau, or None once in the finish hole."""
    if tau <= 0:
        return 0.0, 0.0, False
    for ph in r["phases"]:
        if ph[0] == "track":
            s = ph[1]
            if tau <= s[-1][0]:
                k = max(0, min(len(s) - 2, bisect.bisect_right(ph[2], tau) - 1))
                (ta, xa, ya), (tb, xb, yb) = s[k], s[k + 1]
                f = 0.0 if tb <= ta else min(1.0, max(0.0, (tau - ta) / (tb - ta)))
                return xa + f * (xb - xa), ya + f * (yb - ya), False
        elif ph[0] == "fly":
            _, x0, y0, vx, vy, t0, t1 = ph
            if tau <= t1:
                u = max(0.0, tau - t0)
                return x0 + vx * u, y0 + vy * u + G * u * u / 2, True
        else:
            return ph[1], ph[2], False
    return None


# ------------------------------------------------------------------ clock / timing
def clock_t(scene, tau: float) -> float:
    """Physics time -> clip (clock) time."""
    return scene["t0"] + scene["slow"] * tau


def n_frames(scene):
    return scene["frames"]


def duration(scene):
    return scene["frames"] / FPS


def release_frame(scene) -> int:
    return round(scene["t0"] * FPS)


def clock_tenths(frame: int) -> int:
    return frame * 10 // FPS


def first_frame(t: float) -> int:
    """First frame drawn at or after clip time t."""
    return math.ceil(t * FPS - 1e-9)


# ------------------------------------------------------------------ screen geometry
def px(scene, x, y):
    """World (x, depth) -> screen pixel of the track point (the ball's centre is R px above it)."""
    st = scene["style"]
    X = st["ox"] + st["S"] * x
    if st["mirror"]:
        X = SIZE[0] - X
    return X, st["oy"] + st["S"] * y


def clock_box(scene):
    """The clock sits in the top corner opposite the start."""
    return [18, 16, 18 + CLOCK_W, 16 + CLOCK_H] if scene["style"]["mirror"] else \
        [SIZE[0] - 18 - CLOCK_W, 16, SIZE[0] - 18, 16 + CLOCK_H]


def hole_px(scene):
    return px(scene, scene["W"], scene["D"])


def slot_px(scene, colour):
    """Centre of a ball's slot in the finish rack (a column beside the hole, away from the tracks)."""
    st = scene["style"]
    i = scene["rack"].index(colour)
    hx, _ = hole_px(scene)
    dx = st["rack_dx"] * (-1 if st["mirror"] else 1)
    return hx + dx, st["rack_y0"] + i * st["rack_step"]


def floor_px(scene):
    return scene["style"]["oy"] + scene["style"]["S"] * scene["floor"]


# ------------------------------------------------------------------ generation
def _track(rng, fam):
    b = {"family": fam}
    if fam == "sag":
        b["a"] = round(rng.uniform(0.15, 0.75), 3)
    elif fam == "slow_start":
        b["a"] = round(rng.uniform(1.3, 2.3), 3)
    elif fam == "steep_start":
        b["a"] = round(rng.uniform(0.4, 0.8), 3)
    elif fam == "wavy":
        b["a"], b["k"] = round(rng.uniform(0.08, 0.22), 3), rng.choice([2, 3])
    elif fam == "bend":
        b["k"] = rng.choice([5, 7, 10, 14, 20])
        b["a"] = round(rng.uniform(-0.4, 0.9), 3)
    return b


def _gap_list(rng, W, two):
    if two:
        a0 = rng.uniform(0.18, 0.36) * W
        a1 = a0 + rng.uniform(0.09, 0.16) * W
        b0 = a1 + rng.uniform(0.12, 0.22) * W
        b1 = b0 + rng.uniform(0.08, 0.16) * W
        raw = [(a0, a1), (b0, b1)]
    else:
        x0 = rng.uniform(0.24, 0.6) * W
        raw = [(x0, x0 + rng.uniform(0.11, 0.3) * W)]
    return [{"x0": round(x0, 3), "x1": round(x1, 3), "lip": rng.randint(8, 24)} for x0, x1 in raw]


@functools.lru_cache(maxsize=512)
def _profile(sh) -> tuple:
    W = sh[3]
    return tuple(depth_at(sh, 0.1 * W + 0.8 * W * i / 80) for i in range(81))


def too_close(sh1, sh2) -> bool:
    a, b = _profile(sh1), _profile(sh2)
    return sum(abs(x - y) < SEP_M for x, y in zip(a, b)) > 0.2 * len(a)


def robust(scene, b) -> bool:
    """Every jump clears or misses by MARGIN_X at the real push and at the what-if push, and nothing is
    degenerate (also without the gaps and as a straight ramp)."""
    for v0 in (scene["v0"], PUSH_CF * scene["v0"]):
        r = ball_run(scene, b, v0)
        if not r["ok"] or any(m < MARGIN_X for m in r["margins"]):
            return False
        if v0 == scene["v0"] and r["fell"] is not None and not 0.3 < r["rest_x"] < scene["W"] - 0.5:
            return False
        for kw in ({"no_gap": True}, {"straight": True}):
            if not ball_run(scene, b, v0, **kw)["ok"]:
                return False
    return True


def _arena(rng, scenario):
    if scenario == "steep":
        return round(rng.uniform(5.5, 7.0), 3), round(rng.uniform(4.0, 5.0), 3)
    return round(rng.uniform(8.5, 11.0), 3), round(rng.uniform(2.6, 4.0), 3)


def _plan_gaps(rng, scenario, n):
    """Per ball: 0, 1 or 2 gaps."""
    if scenario == "plain":
        return [0] * n
    if scenario == "all_gaps":
        return [1] * n
    k = rng.randint(1, max(1, (n + 1) // 2))
    plan = [1] * k + [0] * (n - k)
    if scenario == "double":
        for i in range(min(k, rng.randint(1, 3))):
            plan[i] = 2
    rng.shuffle(plan)
    return plan


def _style(rng, scene):
    theme = rng.choice(("light", "light", "dark"))
    T = THEMES[theme]
    W, floor = scene["W"], scene["floor"]
    n = len(scene["balls"])
    rack_dx = rng.randint(64, 96)
    top = 120
    s_w = (SIZE[0] - 40 - 60 - rack_dx - 16 - 30) / W
    s_h = (SIZE[1] - 26 - top - 32) / floor
    smax = min(118.0, s_w, s_h)
    S = round(rng.uniform(0.84, 1.0) * smax, 2)
    R = rng.randint(11, max(11, min(16, int((REST_SEP * S - 8) / 2))))      # fallen balls never overlap
    rack_step = 2 * R + rng.randint(10, 16)
    free_x = SIZE[0] - 40 - 60 - rack_dx - 16 - 30 - S * W
    ox = round(100 + rng.random() * max(0.0, free_x), 1)
    free_y = SIZE[1] - 26 - 32 - top - S * floor
    oy = round(top + rng.random() * max(0.0, free_y), 1)
    fy = oy + S * floor
    span = (n - 1) * rack_step
    lo, hi = top, fy - R - 8 - span
    rack_y0 = round(lo + rng.random() * max(0.0, hi - lo), 1)
    return {"theme": theme, "bg": list(rng.choice(T["bg"])), "bg_pat": rng.choice(BG_PATS),
            "bg_step": rng.randint(40, 80), "ink": list(T["ink"]), "pit": list(T["pit"]), "hole": list(T["hole"]),
            "rack": list(T["rack"]), "R": R, "track_w": rng.randint(3, 6), "track_style": rng.choice(TRACK_STYLES),
            "sleeper_step": rng.randint(18, 30), "S": S, "ox": ox, "oy": oy, "mirror": rng.random() < 0.5,
            "rack_dx": rack_dx, "rack_y0": rack_y0, "rack_step": rack_step}


def build(rng: random.Random) -> dict:
    scenario = rng.choices(SCENARIOS, SCEN_W)[0]
    for attempt in range(1, 4000):
        W, D = _arena(rng, scenario)
        n = rng.randint(2, 10)
        v0 = round(rng.uniform(0.6, 1.5), 3)
        plan = _plan_gaps(rng, scenario, n)
        base = {"W": W, "D": D, "v0": v0, "floor": round(D * rng.uniform(1.25, 1.45) + 0.5, 3)}
        picked, rests = [], []
        for want in plan:
            for _ in range(80):
                b = _track(rng, rng.choice(FAMILIES))
                if want:
                    b["gaps"] = _gap_list(rng, W, want == 2)
                sh = shape(base, b)
                ys = [y for _, y in polyline(sh)]
                if min(ys) < -1e-9 or max(ys) > base["floor"] - 0.4:
                    continue
                if any(too_close(sh, shape(base, o)) for o in picked) or not robust(base, b):
                    continue
                r = ball_run(base, b)
                if want == 2 and not r["land"]:
                    continue                           # a two-gap track whose ball never reaches the second gap
                if r["fell"] is not None and any(abs(r["rest_x"] - x) < REST_SEP for x in rests):
                    continue
                if r["fell"] is not None:
                    rests.append(r["rest_x"])
                picked.append(b)
                break
            else:
                break
        if len(picked) < n:
            continue
        rs = [ball_run(base, b) for b in picked]
        if all(r["finish"] is None for r in rs):
            continue
        tau_end = max(r["end_t"] for r in rs)
        t0 = round(rng.uniform(0.6, 1.2) * FPS) / FPS
        tail = rng.uniform(0.8, 1.4)
        slow_max = min(2.6, (MAX_FRAMES / FPS - 0.05 - t0 - tail) / tau_end)
        if slow_max < 1.4:
            continue
        slow = round(rng.uniform(1.4, slow_max), 3)
        frames = int((t0 + slow * tau_end + tail) * FPS)
        if frames > MAX_FRAMES:
            continue
        scene = {"scenario": scenario, **base, "slow": slow, "t0": t0, "frames": frames, "balls": []}
        scene["style"] = _style(rng, {**scene, "balls": picked})
        cols = rng.sample(sorted(BALLS[scene["style"]["theme"]]), n)
        scene["balls"] = [{"colour": cols[i], "pattern": rng.choice(PATTERNS), **b} for i, b in enumerate(picked)]
        rack = list(cols)
        rng.shuffle(rack)
        scene["rack"] = rack
        scene["build_attempts"] = attempt
        if check(scene) or not _ladder_ok(scene):
            continue
        return scene
    raise RuntimeError("no valid race")


def _ladder_ok(scene):
    """Can this race fill the 10-question ladder? (A trial build with its own fixed rng.)"""
    import questions                                   # late import: questions.py imports this module
    try:
        questions.build(scene, random.Random(0))
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------ consistency check
def events(scene) -> list[tuple]:
    """Every event in clip time: (t, kind, colour) for release, leave, land, floor and finish."""
    out = [(scene["t0"], "release", None)]
    for b in scene["balls"]:
        r = ball_run(scene, b)
        out += [(clock_t(scene, t), "leave", b["colour"]) for t in r["leave"]]
        out += [(clock_t(scene, t), "land", b["colour"]) for t in r["land"]]
        if r["floor_t"] is not None:
            out.append((clock_t(scene, r["floor_t"]), "floor", b["colour"]))
        if r["finish"] is not None:
            out.append((clock_t(scene, r["finish"]), "finish", b["colour"]))
    return sorted(out, key=lambda e: (e[0], e[1], e[2] or ""))


def ball_centre(scene, b, f: int):
    """Screen centre of a ball at frame f, (x, y, airborne), or None while it is in the finish hole."""
    tau = (f / FPS - scene["t0"]) / scene["slow"]
    w = world_at(ball_run(scene, b), tau)
    if w is None:
        return None
    X, Y = px(scene, w[0], w[1])
    return X, Y - scene["style"]["R"], w[2]


def check(scene: dict) -> list[str]:
    errs = []
    bs = scene["balls"]
    st = scene["style"]
    n = len(bs)
    if not 2 <= n <= 10:
        errs.append("ball count out of range")
    if len({b["colour"] for b in bs}) != n or sorted(scene["rack"]) != sorted(b["colour"] for b in bs):
        errs.append("ball colours not distinct or rack doesn't match")
    pal = BALLS[st["theme"]]
    for i, b in enumerate(bs):
        if b["colour"] not in pal:
            errs.append(f"{b['colour']} not in the {st['theme']} palette")
    if any(too_close(shape(scene, x), shape(scene, y)) for i, x in enumerate(bs) for y in bs[i + 1:]):
        errs.append("two tracks run too close together")
    gp = [len(b.get("gaps", ())) for b in bs]
    sc = scene["scenario"]
    if sc == "plain" and any(gp):
        errs.append("plain race with a gap")
    if sc in ("jumps", "double", "steep") and not any(gp):
        errs.append("gapped scenario without a gap")
    if sc == "all_gaps" and not all(gp):
        errs.append("all_gaps race with an ungapped track")
    if sc == "double" and 2 not in gp:
        errs.append("double scenario without a two-gap track")
    if sc != "double" and 2 in gp:
        errs.append("two-gap track outside the double scenario")
    if (sc == "steep") != (scene["W"] < 8.0):
        errs.append("arena size doesn't match the scenario")
    for b in bs:
        pts = polyline(shape(scene, b))
        if pts[0] != (0.0, 0.0) or math.dist(pts[-1], (scene["W"], scene["D"])) > 1e-6:
            errs.append(f"{b['colour']} track doesn't run from start to finish")
        if any(y < -1e-9 for _, y in pts) or max(y for _, y in pts) > scene["floor"] - 0.4:
            errs.append(f"{b['colour']} track rises above the start or comes too near the pit floor")
        if not robust(scene, b):
            errs.append(f"{b['colour']} jump too close to call or degenerate")
    rs = {b["colour"]: ball_run(scene, b) for b in bs}
    if not any(r["finish"] is not None for r in rs.values()):
        errs.append("no ball finishes")
    if sc == "double" and not any(len(r["land"]) >= 1 and len(b.get("gaps", ())) == 2
                                  for b, r in zip(bs, rs.values())):
        errs.append("no ball clears the first of two gaps")
    ev = events(scene)
    dur = duration(scene)
    if not 72 <= scene["frames"] <= MAX_FRAMES:
        errs.append("clip length out of range")
    if abs(scene["t0"] * FPS - round(scene["t0"] * FPS)) > 1e-6:
        errs.append("release is not on a frame instant")
    for t, kind, col in ev:
        if t < EDGE_S + 0.05 or t > dur - EDGE_S - 0.25:
            errs.append(f"{kind} event of {col} at {t:.2f} s is too close to the clip edge")
    # fallen balls rest apart from each other, inside the pit
    fx = sorted(px(scene, r["rest_x"], 0)[0] for r in rs.values() if r["fell"] is not None)
    if any(b - a < 2 * st["R"] + 8 for a, b in zip(fx, fx[1:])):
        errs.append("two fallen balls rest on top of each other")
    if any(r["fell"] is not None and not (0.3 < r["rest_x"] < scene["W"] - 0.4) for r in rs.values()):
        errs.append("a fallen ball rests outside the pit")
    # nothing ever covers the clock; balls stay on screen; sizes
    cb = clock_box(scene)
    R = st["R"]
    for f in range(0, scene["frames"], 2):
        for b in bs:
            p = ball_centre(scene, b, f)
            if p is None:
                continue
            x, y, _ = p
            if x - R < 4 or x + R > SIZE[0] - 4 or y - R < 4 or y + R > SIZE[1] - 4:
                errs.append(f"{b['colour']} ball leaves the screen at frame {f}")
                break
            if x + R > cb[0] - 6 and x - R < cb[2] + 6 and y - R < cb[3] + 6:
                errs.append(f"{b['colour']} ball covers the clock at frame {f}")
                break
    hx, hy = hole_px(scene)
    sx = [slot_px(scene, c) for c in scene["rack"]]
    if any(y - R - 4 < cb[3] + 8 or y + R + 4 > floor_px(scene) - 4 for _, y in sx) or \
            any(x - R < 4 or x + R > SIZE[0] - 4 for x, _ in sx):
        errs.append("finish rack leaves its area")
    if 2 * R < 20 or st["track_w"] < 3 or floor_px(scene) > SIZE[1] - 20:
        errs.append("objects too small or pit floor off screen")
    return errs
