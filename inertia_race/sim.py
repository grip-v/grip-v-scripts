"""Scene generator for inertia_race (1280x720, 24 fps, <= 9.5 s).

Bodies of different shape are released on a common inclined ramp and roll to a
finish line. Rolling without slipping gives

    a = g sin(theta) / (1 + k)        k = I / (m R^2)

so acceleration depends on the shape's inertia coefficient k and on nothing
else: mass and radius cancel exactly. Time from rest over a ramp fraction d is

    t = tau * T0 * sqrt(d)       tau = sqrt(1 + k),  T0 = sqrt(2 L / g sin(theta))

`build(rng)` draws a scene's composition once (scenario, racer count, shapes,
colours, radii, whether a violation is planted and which), then retries only the
continuous and frame-valued parameters until `check()` returns no reasons. The
composition is held fixed across retries so rejection cannot skew its mix.

Nothing here knows about questions, options or rendering. render.py turns the
scene into pixels; answers.py re-reads it only through render.layout().
"""
from __future__ import annotations

import itertools
import math
import random

FPS = 24
MAX_FRAMES = 228                 # 9.5 s hard cap
EDGE_F = 12                      # question events stay >= 0.5 s from both ends
ORDER_GAP_F = 8                  # events compared in time are >= 8 frames (0.33 s) apart
LEAD_MIN_F = 53                  # first finish no earlier than ~2.2 s
G = 9.81

# ---------------------------------------------------------------- bodies

K = {"slider": 0.0, "solid_sphere": 0.4, "solid_disc": 0.5,
     "hollow_sphere": 2.0 / 3.0, "hoop": 1.0}
TAU = {s: math.sqrt(1.0 + k) for s, k in K.items()}
SHAPES = tuple(K)
ROLLS = tuple(s for s in SHAPES if s != "slider")   # a slider translates, never spins

COLOURS = ("orange", "yellow", "green", "teal", "blue", "purple", "pink")
PATTERNS = ("plain", "striped", "two-tone", "ringed")

# Drawn radius in output pixels. The ceiling at each racer count is what still
# leaves MIN_RUN of ramp at the steepest angle; the spacing keeps a `twin` pair
# distinguishable by eye.
R_PX = {2: (40, 52, 64), 3: (32, 41, 50), 4: (24, 30, 36), 5: (20, 24, 28)}
R_TWIN_GAP = 6                   # least visible radius difference for a twin pair

SCENARIOS = ("fair_start", "handicap", "stagger", "pursuit", "twin")
SEQUENTIAL = ("stagger", "pursuit")      # release / overtake frames must read on the timer
TIED = ("twin",)                         # the only scenario where a dead heat is intended
N_RANGE = {"fair_start": (2, 3), "handicap": (2, 5), "stagger": (2, 5),
           "pursuit": (2, 4), "twin": (3, 4)}
DNF_OK = ("handicap", "stagger", "pursuit")   # a racer may still be rolling at the cap

# Shallower than 15 deg leaves the ramp a thin diagonal with dead corners;
# steeper than 27 deg leaves no room for five lanes above MIN_RUN.
ANGLES = (15.0, 18.0, 21.0, 24.0, 27.0)

VIOLATIONS = ("wrong_inertia", "slip")
# Tuned so the share of clips that *survive* `check` carrying a violation lands on
# the ~49/51 split the ladder calls for: some flawed draws are rejected.
VIOLATION_RATE = 0.56
SLIP_GRIP = 0.55                 # a slipping body turns this fraction of its rolling angle

# ---------------------------------------------------------------- canvas

W, H = 1280, 720
TIMER_STRIP = 84                 # full-width band at the top, reserved for the timer
PAD_X, PAD_B = 32, 24
AVAIL_W = W - 2 * PAD_X
AVAIL_H = H - TIMER_STRIP - PAD_B
MIN_RUN = 520                    # below this the ramp is too short to read
LANE_T = 14                      # drawn lane slab thickness, below the surface line
# Extra perpendicular room between lanes, so a clip with few racers fills the
# frame instead of leaving most of it empty. The pad is the largest of these
# that still leaves MIN_RUN of ramp at this angle, so it never starves the run.
PITCH_PADS = (80.0, 60.0, 45.0, 30.0, 20.0, 10.0, 0.0)
RULER_DROP = 26                  # ruler line, below the lowest lane surface
LABEL_DROP = 28                  # label centres, below the ruler line
LABEL_H = 13                     # half-height of a label glyph box
RULER_TICKS = 20                 # gridlines every 0.05 of the ramp
LABEL_AT = (0.00, 0.25, 0.50, 0.75, 1.00)

# ---------------------------------------------------------------- timer


def aligned(f: int) -> bool:
    """Frame whose exact time is within 0.045 s above a tenth."""
    return (f * 10) % FPS <= 10


def timer_tenths(f: int) -> int:
    return (f * 10) // FPS


def timer_text(f: int) -> str:
    t = timer_tenths(f)
    return f"{t // 10}.{t % 10} s"


ALIGNED_F = tuple(f for f in range(MAX_FRAMES) if aligned(f))


# ---------------------------------------------------------------- feasible shape sets


def _window(subset, lead_s: float, last_s: float, gap_s: float):
    """T0 range over which `subset`, all released together from the top, separates."""
    t = sorted(TAU[s] for s in subset)
    dmin = min(b - a for a, b in zip(t, t[1:]))
    lo = max(lead_s / t[0], gap_s / dmin)
    hi = last_s / t[-1]
    return (lo, hi) if lo <= hi else None


def shape_key(shapes) -> tuple:
    """Canonical ordering for a shape set: by inertia coefficient, ascending."""
    return tuple(sorted(set(shapes), key=SHAPES.index))


def _feasible_sets():
    """Shape sets usable where finish order is shape order alone (fair_start, twin)."""
    lead_s = LEAD_MIN_F / FPS
    last_s = (MAX_FRAMES - 1 - EDGE_F) / FPS
    gap_s = ORDER_GAP_F / FPS * 2.0          # keep a clear 0.67 s margin
    out = {}
    for n in (2, 3):
        for sub in itertools.combinations(SHAPES, n):
            w = _window(sub, lead_s, last_s, gap_s)
            if w:
                out[shape_key(sub)] = w
    return out


FEASIBLE_SETS = _feasible_sets()

# ---------------------------------------------------------------- kinematics


def accel(shape: str, angle_deg: float, length_m: float) -> float:
    """Along-ramp acceleration in m/s^2 for a body of this shape."""
    return G * math.sin(math.radians(angle_deg)) / (1.0 + K[shape])


def _a_frac(r: dict, t0: float) -> float:
    """Acceleration in ramp fractions / s^2, from the racer's *drawn* inertia."""
    return 2.0 / (TAU[r["k_shape"]] * t0) ** 2


def finish_frame(r: dict, t0: float) -> int | None:
    """First frame at which `r` has reached the finish line, or None if it never does."""
    d = 1.0 - r["start"]
    if d <= 0.0:
        return r["release_f"]
    t = TAU[r["k_shape"]] * t0 * math.sqrt(d)
    f = r["release_f"] + math.ceil(t * FPS - 1e-9)
    return f if f <= MAX_FRAMES - 1 else None


def pos_at(r: dict, t0: float, f: int) -> float:
    """Ramp fraction covered by `r` at frame `f`, clamped at the finish line."""
    if f <= r["release_f"]:
        return r["start"]
    t = (f - r["release_f"]) / FPS
    return min(1.0, r["start"] + 0.5 * _a_frac(r, t0) * t * t)


def spin_turns(r: dict, t0: float, f: int) -> float:
    """Revolutions the body has made by frame `f` (0 for a slider)."""
    if r["shape"] == "slider":
        return 0.0
    grip = SLIP_GRIP if r["violates"] == "slip" else 1.0
    return grip * (pos_at(r, t0, f) - r["start"]) / (2.0 * math.pi * r["radius"])


# ---------------------------------------------------------------- drawing geometry
#
# One ramp, drawn in side view, carrying n lanes stacked along its surface normal.
# A point is addressed by (s, off): `s` is the fraction of the ramp travelled from
# the top, `off` is a displacement along the normal, positive meaning up off the
# surface. Lane i has its surface at off = -pitch * i, so lane 0 is the highest;
# a body of radius rp resting there has its centre at off = -pitch * i + rp.
# The ruler and its labels hang below the lowest lane at negative off.


def _pitch(n: int, rp_max: float, pad: float) -> float:
    return 2.0 * rp_max + 14.0 + pad


def _fit(angle_deg: float, n: int, rp_max: float):
    """(pad, run) using the most lane spacing this angle can afford, or None."""
    a = math.radians(angle_deg)
    sin, cos = math.sin(a), math.cos(a)
    for pad in PITCH_PADS:
        lo, hi = _normal_span(n, _pitch(n, rp_max, pad), rp_max)
        spread = hi - lo
        run = min((AVAIL_H - cos * spread) / sin, (AVAIL_W - sin * spread) / cos)
        if run >= MIN_RUN:
            return pad, run
    return None


def _normal_span(n: int, pitch: float, rp_max: float) -> tuple[float, float]:
    """(lowest, highest) normal offset any drawn content reaches."""
    hi = 2.0 * rp_max                                      # top of the tallest body
    lo = -(pitch * (n - 1)) - LANE_T - RULER_DROP - LABEL_DROP - LABEL_H
    return lo, hi


def _view(angle_deg: float, n: int, rp_max: float, pad: float, run: float) -> dict:
    """Ramp placement, lane offsets and ruler, all in output pixels."""
    a = math.radians(angle_deg)
    sin, cos = math.sin(a), math.cos(a)
    pitch = _pitch(n, rp_max, pad)
    lo, hi = _normal_span(n, pitch, rp_max)
    spread = hi - lo
    width = cos * run + sin * spread
    height = sin * run + cos * spread
    # centre the content box inside the free area, then back out the ramp's origin
    x0 = PAD_X + (AVAIL_W - width) / 2.0 - sin * lo
    y0 = TIMER_STRIP + (AVAIL_H - height) / 2.0 + cos * hi
    return {
        "angle_deg": angle_deg,
        "x0": round(x0, 3), "y0": round(y0, 3),
        "ux": round(cos, 9), "uy": round(sin, 9),       # unit vector down the ramp
        "nx": round(sin, 9), "ny": round(-cos, 9),      # unit normal, up off the surface
        "run": round(run, 3), "pitch": round(pitch, 3),
        "lane_off": [round(-pitch * i, 3) for i in range(n)],
        "lane_t": LANE_T,
        "ruler_off": round(-(pitch * (n - 1)) - LANE_T - RULER_DROP, 3),
        "label_off": round(-(pitch * (n - 1)) - LANE_T - RULER_DROP - LABEL_DROP, 3),
        "off_lo": round(lo, 3), "off_hi": round(hi, 3),
        "ticks": RULER_TICKS, "tick_major": 5,
    }


def ramp_point(view: dict, s: float, off: float) -> tuple[float, float]:
    """Pixel point at ramp fraction `s`, displaced `off` px along the surface normal."""
    d = s * view["run"]
    return (view["x0"] + view["ux"] * d + view["nx"] * off,
            view["y0"] + view["uy"] * d + view["ny"] * off)


def lane_off(view: dict, racer: dict) -> float:
    """Normal offset of the centre of `racer`, resting on its lane surface."""
    return view["lane_off"][racer["lane"]] + racer["r_px"]


def content_bbox(scene: dict) -> tuple[float, float, float, float]:
    """(x_min, y_min, x_max, y_max) of everything the renderer will draw."""
    v = scene["view"]
    lo, hi = v["off_lo"], v["off_hi"]
    xs, ys = [], []
    for s in (0.0, 1.0):
        for off in (lo, hi):
            x, y = ramp_point(v, s, off)
            xs.append(x)
            ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def _labels(view: dict) -> list[dict]:
    """The only text besides the timer: ruler numbers at 0.00 / 0.25 / ... / 1.00."""
    out = []
    for s in LABEL_AT:
        x, y = ramp_point(view, s, view["label_off"])
        out.append({"text": f"{s:.2f}", "x": round(x, 3), "y": round(y, 3), "size": 20})
    return out


THEMES = (
    {"bg": (244, 243, 238), "ramp": (221, 218, 209), "lane": (233, 231, 223),
     "line": (30, 30, 34), "grid_alpha": 0.17, "finish": (226, 30, 36), "mark": "bar"},
    {"bg": (236, 240, 245), "ramp": (212, 220, 231), "lane": (226, 232, 240),
     "line": (28, 32, 40), "grid_alpha": 0.20, "finish": (214, 40, 70), "mark": "dot"},
    {"bg": (242, 240, 232), "ramp": (225, 215, 197), "lane": (236, 229, 215),
     "line": (34, 30, 26), "grid_alpha": 0.15, "finish": (200, 54, 40), "mark": "bar"},
)

# ---------------------------------------------------------------- composition


def _compose(rng: random.Random) -> dict | None:
    """Everything categorical, drawn once. None if the draw is self-inconsistent."""
    scen = rng.choice(SCENARIOS)
    n = rng.randint(*N_RANGE[scen])
    angle = rng.choice(ANGLES)
    r_px = sorted(rng.sample(R_PX[n], min(n, len(R_PX[n]))) +
                  [rng.choice(R_PX[n]) for _ in range(max(0, n - len(R_PX[n])))])
    fit = _fit(angle, n, max(r_px))
    if fit is None:
        return None
    pad, run = fit

    if scen == "fair_start":
        sets = [s for s in FEASIBLE_SETS if len(s) == n]
        shapes = list(rng.choice(sets))
        twin_of = None
    elif scen == "twin":
        base = rng.choice(ROLLS)
        sets = [s for s in FEASIBLE_SETS if len(s) == n - 1 and base in s]
        if not sets:
            return None
        chosen = rng.choice(sets)
        shapes = [base, base] + [s for s in chosen if s != base]
        if len(shapes) != n:
            return None
        twin_of = (0, 1)
        # the twin pair must be told apart by radius alone, and r_px is fixed for
        # the life of this composition, so reject here rather than retry into a wall
        if abs(r_px[0] - r_px[1]) < R_TWIN_GAP:
            return None
    elif scen == "pursuit":
        fast = rng.choice(("slider", "solid_sphere"))
        slow = rng.choice(("hollow_sphere", "hoop"))
        rest = [s for s in SHAPES if s not in (fast, slow)]
        shapes = [fast, slow] + rng.sample(rest, n - 2)
        twin_of = None
    else:
        shapes = rng.sample(SHAPES, n)
        twin_of = None

    # Decide who could carry a violation before deciding whether one is planted, so
    # a composition with no eligible body does not dilute the realised rate.
    pool = [i for i, s in enumerate(shapes) if s != "slider"]
    flawed = bool(pool) and rng.random() < VIOLATION_RATE
    kind = target = as_if = None
    if flawed:
        target = rng.choice(pool)
        kind = rng.choice(VIOLATIONS)
        if scen == "twin" and target in twin_of:
            # A slipping twin still dead-heats, since slip changes only the drawn
            # rotation; wrong_inertia would break the tie the scenario exists to show.
            kind = "slip"
        if kind == "wrong_inertia":
            opts = _stand_ins(scen, shapes, target)
            if opts:
                as_if = rng.choice(opts)
            else:
                # No stand-in inertia keeps this scene separable, so plant the other
                # kind rather than discard the composition: discarding here would pull
                # the realised violation rate below VIOLATION_RATE.
                kind = "slip"

    rng.shuffle(r_px)
    if scen == "twin" and abs(r_px[0] - r_px[1]) < R_TWIN_GAP:
        return None
    return {
        "scenario": scen, "n": n, "angle": angle, "run": run, "pad": pad,
        "shapes": shapes, "twin_of": twin_of,
        "eff_shapes": [as_if if (kind == "wrong_inertia" and i == target) else s
                       for i, s in enumerate(shapes)],
        "r_px": r_px,
        "colours": rng.sample(COLOURS, n),
        "patterns": [rng.choice(PATTERNS) for _ in range(n)],
        "lanes": rng.sample(range(n), n),
        "theme": rng.choice(THEMES),
        "flawed": flawed, "kind": kind, "target": target, "as_if": as_if,
    }


def _stand_ins(scen: str, shapes: list[str], target: int) -> list[str]:
    """Shapes a `wrong_inertia` body may move as, without breaking the scene.

    In `fair_start` and `twin` the release time window is derived from the spread of
    tau across the field, so the stand-in must leave that spread feasible; elsewhere
    start offsets absorb it and any other shape will do.
    """
    rest = [s for i, s in enumerate(shapes) if i != target]
    if scen not in ("fair_start", "twin"):
        return [s for s in SHAPES if s != shapes[target]]
    out = []
    for s in SHAPES:
        if s in rest or s == shapes[target]:
            continue
        if shape_key(rest + [s]) in FEASIBLE_SETS:
            out.append(s)
    return out


def _attempt(rng: random.Random, c: dict) -> dict:
    """One draw of the continuous and frame-valued parameters for composition `c`."""
    scen, n = c["scenario"], c["n"]
    run, angle = c["run"], c["angle"]

    if scen in ("fair_start", "twin"):
        lo, hi = FEASIBLE_SETS[shape_key(c["eff_shapes"])]
        t0 = rng.uniform(lo, hi)
        starts, rel = [0.0] * n, [0] * n
    else:
        t0 = rng.uniform(3.0, 7.6)
        if scen == "handicap":
            starts = [round(rng.uniform(0.0, 0.36), 3) for _ in range(n)]
            rel = [0] * n
        elif scen == "stagger":
            starts = [0.0] * n
            rel = sorted(rng.sample([f for f in ALIGNED_F if f <= 46], n))
        else:                                   # pursuit: the slow body leads from the front
            order = sorted(range(n), key=lambda i: TAU[c["shapes"][i]])
            starts = [0.0] * n
            for rank, i in enumerate(order):
                starts[i] = round(0.095 * rank + rng.uniform(0.0, 0.040), 3)
            rel = [0] * n

    length = t0 * t0 * G * math.sin(math.radians(angle)) / 2.0
    view = _view(angle, n, max(c["r_px"]), c["pad"], run)

    racers = []
    for i, sh in enumerate(c["shapes"]):
        viol = "none"
        k_shape = sh
        if c["flawed"] and i == c["target"]:
            viol = c["kind"]
            if c["kind"] == "wrong_inertia":
                k_shape = c["as_if"]
        racers.append({
            "id": f"{c['colours'][i]}_{sh}", "lane": c["lanes"][i],
            "colour": c["colours"][i], "pattern": c["patterns"][i],
            "shape": sh, "k_shape": k_shape,
            "r_px": c["r_px"][i], "radius": round(c["r_px"][i] / run, 6),
            "start": starts[i], "release_f": rel[i],
            "violates": viol, "finish_f": None,
        })
    for r in racers:
        r["finish_f"] = finish_frame(r, t0)

    return {
        "category": "inertia_race", "scenario": scen,
        "n_frames": MAX_FRAMES, "t0": round(t0, 9),
        "ramp": {"angle_deg": angle, "length_m": round(length, 4), "g": G},
        "racers": racers, "twin_of": list(c["twin_of"]) if c["twin_of"] else None,
        "view": view, "labels": _labels(view), "theme": c["theme"],
        "timer": {"box": [28, 24, 150, 72], "size": 26},
        "flawed": c["flawed"],
    }


L6_OPTIONS = ("every body behaves consistently",
              "at least one body breaks the rules")
L6_TEXT = ("Watch the whole clip. Does every body move the way a real object of its "
           "shape would when rolling down a ramp, or is at least one of them breaking "
           "the rules of rolling motion?")


def _l6(scene: dict) -> dict:
    """The universal consistency item. It sits outside the ten L1-L5 questions,
    because deciding it needs shape, motion, spin and finish order together, and
    because common/ladder5.py requires the graded ladder to be exactly ten."""
    bad = scene["flawed"]
    text = L6_OPTIONS[1] if bad else L6_OPTIONS[0]
    return {
        "qid": "l6", "difficulty_level": 6, "rung": "consistency",
        "task": "Physical Consistency", "question_type": "multiple_choice",
        "derivation_key": "l6_consistency", "question": L6_TEXT,
        "options": list(L6_OPTIONS), "answer": "B" if bad else "A",
        "answer_text": text, "groundtruth": text,
        "params": {"min_fps": 4.0}, "requires_motion": True,
    }


def build(rng: random.Random) -> dict:
    """Draw a composition, then retry its free parameters until `check()` accepts."""
    tries = 0
    for _ in range(200):
        c = _compose(rng)
        if c is None:
            continue
        for _ in range(80):
            tries += 1
            scene = _attempt(rng, c)
            if not check(scene):
                scene["tries"] = tries
                scene["l6"] = _l6(scene)
                return scene
    raise RuntimeError("no acceptable scene")


# ---------------------------------------------------------------- acceptance


def check(scene: dict) -> list[str]:
    """Reasons to reject. Empty list means accept."""
    bad: list[str] = []
    rs, t0 = scene["racers"], scene["t0"]
    n, scen = len(rs), scene["scenario"]
    end = scene["n_frames"] - 1

    if scene["n_frames"] > MAX_FRAMES:
        bad.append("clip longer than the cap")
    if len({r["colour"] for r in rs}) != n:
        bad.append("repeated colour")
    if sorted(r["lane"] for r in rs) != list(range(n)):
        bad.append("lanes are not a permutation")

    # duplicate shapes are the point of `twin` and a defect anywhere else
    dup = n - len({r["shape"] for r in rs})
    if scen in TIED:
        if dup != 1:
            bad.append("twin needs exactly one duplicated shape")
        else:
            a, b = (rs[i] for i in scene["twin_of"])
            if a["shape"] != b["shape"]:
                bad.append("twin pair is not the same shape")
            if abs(a["r_px"] - b["r_px"]) < R_TWIN_GAP:
                bad.append("twin radii are too close to tell apart")
            if a["finish_f"] is None or a["finish_f"] != b["finish_f"]:
                bad.append("twin pair does not dead-heat")
    elif dup:
        bad.append("repeated shape")

    # a body that ends a hair short of the line is indistinguishable from one that
    # crossed it, so never leave that call to a threshold
    for r in rs:
        if r["finish_f"] is None and pos_at(r, t0, end) > 0.97:
            bad.append(f"{r['id']} ends too close to the line to call it unfinished")

    fin = [r for r in rs if r["finish_f"] is not None]
    if len(fin) < 2:
        bad.append("fewer than two finishers")
        return bad
    if n - len(fin) > (1 if scen in DNF_OK else 0):
        bad.append("too many racers never finish")

    ff = sorted(r["finish_f"] for r in fin)
    if ff[0] < max(EDGE_F, LEAD_MIN_F):
        bad.append("first finish too early")
    if ff[-1] > end - EDGE_F:
        bad.append("last finish too close to the end")
    if not aligned(ff[0]):
        bad.append("first finish is not timer-aligned")

    # at least one pair must be far enough apart in time to be asked about
    if not any(b - a >= ORDER_GAP_F for a, b in itertools.combinations(ff, 2)):
        bad.append("no finish pair clears the order gap")

    # the winner is unambiguous everywhere except the intended dead heat
    lead = [r for r in fin if r["finish_f"] == ff[0]]
    if scen in TIED:
        if len(lead) not in (1, 2):
            bad.append("more than two bodies share the lead")
        runner = sorted(f for f in ff if f > ff[0])
        if runner and runner[0] - ff[0] < ORDER_GAP_F:
            bad.append("dead heat not clearly ahead of the field")
    else:
        if len(lead) != 1:
            bad.append("winner is a tie")
        elif ff[1] - ff[0] < ORDER_GAP_F:
            bad.append("winner not clearly ahead")

    for r in rs:
        if not (0.0 <= r["start"] < 0.92):
            bad.append(f"{r['id']} starts off the ramp")
        if r["release_f"] > end - EDGE_F - LEAD_MIN_F:
            bad.append(f"{r['id']} released too late to matter")
        if not (0.014 <= r["radius"] <= 0.115):
            bad.append(f"{r['id']} drawn radius is out of range")
        turns = spin_turns(r, t0, r["finish_f"] or end)
        if r["shape"] != "slider" and turns < 1.2:
            bad.append(f"{r['id']} makes under one and a bit turns")

    if scen in SEQUENTIAL:
        for r in rs:
            if not aligned(r["release_f"]):
                bad.append(f"{r['id']} release is not timer-aligned")

    if scen == "stagger" and len({r["release_f"] for r in rs}) != n:
        bad.append("stagger needs distinct release frames")
    if scen == "handicap" and len({round(r["start"], 2) for r in rs}) != n:
        bad.append("handicap needs distinct start marks")

    if scen == "pursuit":
        cross = crossings(scene)
        if not cross:
            bad.append("pursuit produced no overtake")
        elif not any(EDGE_F <= f <= end - EDGE_F and aligned(f) for _, _, f in cross):
            bad.append("pursuit overtake is not inside the readable window")

    if max(r["r_px"] for r in rs) * 2.0 + 14.0 > scene["view"]["pitch"] + 1e-6:
        bad.append("lane pitch too tight for the drawn radii")

    # nothing may leave the canvas or stray into the timer strip
    x0, y0, x1, y1 = content_bbox(scene)
    if x0 < PAD_X - 1 or x1 > W - PAD_X + 1:
        bad.append("content leaves the canvas horizontally")
    if y0 < TIMER_STRIP - 1 or y1 > H - PAD_B + 1:
        bad.append("content leaves the canvas vertically")

    return bad


# ---------------------------------------------------------------- derived facts


# A lead change only counts once the new leader is clearly ahead: without a
# deadband, two bodies running abreast register dozens of phantom overtakes from
# float noise, and the renderer's measurement of the same moment disagrees.
LEAD_DEADBAND = 0.02             # 0.4 of a gridline, tens of pixels on screen
LEAD_HOLD = 3                    # frames a new leader must hold before it counts


def lead_changes(seq) -> list[tuple[int, int]]:
    """(old sign, frame) for each confirmed change of leader in a sign sequence.

    `seq` is +1 / -1 / 0 per frame: who is clearly ahead, or nobody. A change is
    confirmed only after LEAD_HOLD frames, so a single frame sitting exactly on the
    deadband cannot decide the answer -- which matters because answers.py applies
    this same rule to positions measured off the drawing, a few parts in a million
    away from these.
    """
    out, cur, run, sign = [], None, 0, None
    for f, s in enumerate(seq):
        if s == 0:
            run, sign = 0, None
            continue
        if s == sign:
            run += 1
        else:
            sign, run = s, 1
        if run >= LEAD_HOLD and s != cur:
            if cur is not None:
                out.append((cur, f))
            cur = s
    return out


def crossings(scene: dict) -> list[tuple[str, str, int]]:
    """(was ahead, now ahead, frame) for every confirmed lead change between a pair."""
    t0 = scene["t0"]
    out = []
    for a, b in itertools.combinations(scene["racers"], 2):
        seq = []
        for f in range(scene["n_frames"]):
            sa, sb = pos_at(a, t0, f), pos_at(b, t0, f)
            d = sa - sb
            seq.append(0 if min(sa, sb) >= 1.0 or abs(d) < LEAD_DEADBAND
                       else (1 if d > 0 else -1))
        for old, f in lead_changes(seq):
            out.append((a["id"] if old > 0 else b["id"],
                        b["id"] if old > 0 else a["id"], f))
    return out


def clear_finishers(scene: dict) -> list[str] | None:
    """Finishers in order, when every one of them is clearly separated in time."""
    done = sorted((r["finish_f"], r["id"]) for r in scene["racers"]
                  if r["finish_f"] is not None)
    if len(done) < 2:
        return None
    if any(b[0] - a[0] < ORDER_GAP_F for a, b in zip(done, done[1:])):
        return None
    return [i for _, i in done]


def order_of(scene: dict) -> list[str]:
    """Finish order, then any racer still rolling, by how far it got."""
    t0, end = scene["t0"], scene["n_frames"] - 1
    return [r["id"] for r in sorted(
        scene["racers"],
        key=lambda r: (r["finish_f"] if r["finish_f"] is not None
                       else MAX_FRAMES + (1.0 - pos_at(r, t0, end)), r["lane"]))]