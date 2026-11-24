"""Scene generator for glasses_fill: jugs pour liquid into a row of glasses.

A row of numbered 100 ml glasses (scale ticks every 10 ml, ruler numbers every
20 ml) stands on a table; 3-4 jugs wait on a shelf. One jug at a time moves
over a glass, tilts, pours and goes back. Each jug has its own body pattern
and its own liquid colour. A pour delivers min(asked, what is left in the
jug), so a jug can run dry in the middle of a pour (the stream stops while it
is still tilted) or pour nothing at all. Liquids never mix: each pour sits on
top as a new layer. A glass can crack after its last pour: a hole appears on
its wall and the liquid above the hole drains out until the level reaches
the hole. Every amount is a multiple of 10 ml, so every level sits exactly on
a scale tick.

Scenarios (scene["scenario"]):
  row_pours      5-10 glasses, every pour goes into a different glass
  layer_stack    4-6 glasses, later pours land on top of earlier ones (layers)
  dry_jug        4-9 glasses; one jug is empty (tilts, nothing comes out) and one jug runs
                 dry part-way through a pour
  cracked_glass  4-9 glasses; 1-2 crack after their last pour and drain to the crack

The scene dict is a pure function of the seeded rng. It holds the inputs
(jugs, what each pour asks for) and the drawn track (`got_ml`, every phase
time, the leaks), plus the on-screen labels; render.py draws only from it and
answers.py re-derives every answer from it with its own code.
"""
from __future__ import annotations
import math
import random

FPS = 24
MAX_FRAMES = 240                 # 10 s
EDGE_S = 0.5                     # nothing a question needs happens closer than this to either end
CAP = 100                        # every glass holds 100 ml
STEP = 10                        # every amount is a multiple of this
JUG_CAP = 150
SCENARIOS = ("row_pours", "layer_stack", "dry_jug", "cracked_glass")
MAX_ATTEMPTS = 60
MIN_FLOW_S = 0.5                 # a full pour's stream lasts at least this long

LIQUIDS = {"red": "#D64545", "blue": "#3B7DD8", "green": "#3FA34D", "orange": "#F08A24",
           "purple": "#8E5CC4", "yellow": "#E3BE1C", "teal": "#17A2A2", "brown": "#8A5A2E"}
PATTERNS = ("striped", "dotted", "checkered", "zigzag", "diagonal", "banded", "crosshatch", "wavy")
PALETTES = ({"bg": "#F7F5EF", "guide": "#E4DDCF"}, {"bg": "#EEF3F7", "guide": "#D9E2EA"},
            {"bg": "#F4EFF6", "guide": "#E2D9E7"}, {"bg": "#F1F4EC", "guide": "#DDE4D4"})
TABLES = ("#C9B79C", "#B8A48A", "#A9B3BC", "#C2A987")
JUG_BODIES = ("#FFFFFF", "#FBF6E9", "#EDF1F5")

# ---- layout (1280 x 720); shared with render.py
W, H = 1280, 720
RULER_X = 112
GLASS_X0, GLASS_X1 = 250, 1250
Y_BOT = 600                      # inside bottom of every glass (0 ml)
PX_PER_ML = 2.0
Y_TOP = Y_BOT - CAP * PX_PER_ML  # 100 ml line = glass rim
BASE_H = 16
TABLE_Y = Y_BOT + BASE_H
LABEL_Y = TABLE_Y + 38
SHELF_Y = 172
LABEL_PX = 34                    # font size: digits are ~25 px tall
SCALE_LABEL_EVERY = 20           # ruler numbers every 20 ml (ticks every 10 ml)
CLOCK_XY = (1262, 14)            # top-right corner, right-aligned


def ml_to_y(ml: float) -> float:
    return Y_BOT - ml * PX_PER_ML


def frame_of(t: float) -> int:
    """First frame index whose time f / FPS is >= t."""
    return math.ceil(t * FPS - 1e-9)


def clock_tenths(frame: int) -> int:
    """What the on-screen clock shows on this frame, in tenths of a second (floored)."""
    return frame * 10 // FPS


def _r(x):
    return round(x, 4)


# ------------------------------------------------------------------ build
def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    for attempt in range(MAX_ATTEMPTS):
        scene = _attempt(rng, scenario)
        if scene is not None and not check(scene):
            scene["attempts"] = attempt + 1
            return scene
    raise RuntimeError(f"no valid {scenario} scene in {MAX_ATTEMPTS} attempts")


def _ask(rng, space):
    """What a pour asks for: half a glass, a top-up, or a random amount."""
    u = rng.random()
    if u < 0.3 and space >= 50:
        return 50
    if u < 0.45:
        return space
    return rng.randrange(STEP, space + 1, STEP)


def _attempt(rng, scenario):
    n_glasses = {"row_pours": (5, 10), "layer_stack": (4, 6), "dry_jug": (4, 9), "cracked_glass": (4, 9)}[scenario]
    n_glasses = rng.randint(*n_glasses)
    n_jugs = rng.randint(3, 4)
    n_pours = rng.randint(3, 4) if scenario == "dry_jug" else rng.randint(2, 4)
    pats = rng.sample(PATTERNS, n_jugs)
    cols = rng.sample(sorted(LIQUIDS), n_jugs)
    move_s, tilt_s = rng.uniform(0.34, 0.46), rng.uniform(0.18, 0.26)
    rate = rng.uniform(55.0, 90.0)
    jugs = [{"pattern": pats[j], "colour": cols[j], "start_ml": rng.randrange(40, JUG_CAP + 1, STEP)}
            for j in range(n_jugs)]

    # who pours, into which glass, asking for how much
    filled = [0] * n_glasses
    top = [None] * n_glasses
    left = [jg["start_ml"] for jg in jugs]
    plan = []                                     # (jug, glass, ask, got)
    special = {}
    if scenario == "dry_jug":
        dry, low = rng.sample(range(n_jugs), 2)
        jugs[dry]["start_ml"] = left[dry] = 0
        jugs[low]["start_ml"] = left[low] = rng.choice((20, 30, 40))
        slots = rng.sample(range(n_pours), 2)
        special = {slots[0]: dry, slots[1]: low}
    for k in range(n_pours):
        if k in special:
            j = special[k]
        else:
            wet = [j for j in range(n_jugs) if left[j] >= STEP and j not in special.values()]
            if not wet:
                return None
            j = rng.choice(wet)
        if scenario == "row_pours":
            cand = [g for g in range(n_glasses) if filled[g] == 0]
        elif scenario == "layer_stack" and k > 0 and rng.random() < 0.8:
            cand = [g for g in range(n_glasses) if 0 < filled[g] <= CAP - STEP and top[g] != j]
        else:
            cand = [g for g in range(n_glasses) if filled[g] <= CAP - STEP and top[g] != j]
        if special.get(k) is not None and left[j] > 0:          # the jug that runs dry: room for more than it has
            cand = [g for g in cand if CAP - filled[g] >= left[j] + 2 * STEP]
        if not cand:
            return None
        g = rng.choice(cand)
        space = CAP - filled[g]
        if special.get(k) is not None and left[j] > 0:
            ask = rng.randrange(left[j] + 2 * STEP, space + 1, STEP)
        elif special.get(k) is not None:
            ask = _ask(rng, space)
        else:
            ask = min(_ask(rng, space), left[j] // STEP * STEP)
        got = min(ask, left[j])
        left[j] -= got
        filled[g] += got
        if got:
            top[g] = j
        plan.append((j, g, ask, got))

    # timeline
    t = rng.uniform(0.6, 0.85)
    pours = []
    for j, g, ask, got in plan:
        r = min(rate, got / MIN_FLOW_S) if got else rate        # every stream lasts >= MIN_FLOW_S
        flow = got / r
        if got == 0:
            hold = rng.uniform(0.7, 0.9)                         # tilts, nothing comes out
        elif got < ask:
            hold = flow + rng.uniform(0.45, 0.6)                 # runs dry: stream stops, jug stays tilted
        else:
            hold = flow                                          # turns back up as the stream ends
        p = {"jug": j, "glass": g, "ask_ml": ask, "got_ml": got, "rate_ml_s": _r(r)}
        p["t_move"] = _r(t)
        p["t_tilt"] = _r(t + move_s)
        p["t_pour"] = _r(t + move_s + tilt_s)
        p["t_flow_end"] = _r(p["t_pour"] + flow)
        p["t_untilt"] = _r(p["t_pour"] + hold)
        p["t_back"] = _r(p["t_untilt"] + tilt_s)
        p["t_end"] = _r(p["t_back"] + move_s)
        pours.append(p)
        t = p["t_end"] + rng.uniform(0.1, 0.18)
    last = pours[-1]["t_end"]

    leaks = []
    if scenario == "cracked_glass":
        cands = [g for g in range(n_glasses) if filled[g] >= 3 * STEP]
        if not cands:
            return None
        for g in sorted(rng.sample(cands, min(len(cands), rng.randint(1, 2)))):
            hole = rng.randrange(0, filled[g] - 2 * STEP + 1, STEP)
            last_in = max(p["t_flow_end"] for p in pours if p["glass"] == g and p["got_ml"])
            lrate = rng.uniform(25.0, 40.0)
            t_crack = _r(last_in + rng.uniform(0.35, 0.8))
            leaks.append({"glass": g, "hole_ml": hole, "rate_ml_s": _r(lrate), "t_crack": t_crack,
                          "t_drain_end": _r(t_crack + (filled[g] - hole) / lrate)})
            last = max(last, leaks[-1]["t_drain_end"])

    n_frames = frame_of(last + rng.uniform(0.7, 0.95))
    if n_frames > MAX_FRAMES:
        return None

    slot = (GLASS_X1 - GLASS_X0) / n_glasses
    gw = int(round(max(48.0, min(118.0, rng.uniform(0.5, 0.68) * slot))))
    cx = [int(round(GLASS_X0 + slot * (g + 0.5))) for g in range(n_glasses)]
    labels = [{"kind": "scale", "text": str(ml), "x": RULER_X - 16, "y": ml_to_y(ml), "anchor": "rm"}
              for ml in range(0, CAP + 1, SCALE_LABEL_EVERY)]
    labels += [{"kind": "glass", "text": str(g + 1), "x": cx[g], "y": LABEL_Y, "anchor": "mm"}
               for g in range(n_glasses)]
    return {
        "scenario": scenario, "n_frames": n_frames, "duration": round(n_frames / FPS, 6),
        "palette": dict(rng.choice(PALETTES), table=rng.choice(TABLES), jug_body=rng.choice(JUG_BODIES)),
        "glasses": {"n": n_glasses, "cx": cx, "w": gw}, "jugs": jugs, "pours": pours, "leaks": leaks,
        "labels": labels, "label_px": LABEL_PX, "clock_xy": list(CLOCK_XY),
    }


# ------------------------------------------------------------------ state at time t (renderer)
def poured_at(p, t):
    if p["got_ml"] == 0 or t <= p["t_pour"]:
        return 0.0
    return min(float(p["got_ml"]), (t - p["t_pour"]) * p["rate_ml_s"])


def drained_at(lk, peak, t):
    if t <= lk["t_crack"]:
        return 0.0
    return min(float(peak - lk["hole_ml"]), (t - lk["t_crack"]) * lk["rate_ml_s"])


def peak_ml(scene, g):
    return sum(p["got_ml"] for p in scene["pours"] if p["glass"] == g)


def _remove_band(layers, lo, amount):
    """Remove `amount` ml starting at height `lo` upward; the layers above sink."""
    out, h, hi = [], 0.0, lo + amount
    for col, ml in layers:
        cut = max(0.0, min(h + ml, hi) - max(h, lo))
        if ml - cut > 1e-9:
            out.append([col, ml - cut])
        h += ml
    return out


def glass_layers(scene, g, t):
    """Bottom-up [[colour, ml], ...] in glass g at time t."""
    layers = []
    for p in scene["pours"]:
        if p["glass"] != g:
            continue
        ml = poured_at(p, t)
        if ml <= 1e-9:
            continue
        col = scene["jugs"][p["jug"]]["colour"]
        if layers and layers[-1][0] == col:
            layers[-1][1] += ml
        else:
            layers.append([col, ml])
    for lk in scene["leaks"]:
        if lk["glass"] == g:
            d = drained_at(lk, peak_ml(scene, g), t)
            if d > 0:
                layers = _remove_band(layers, lk["hole_ml"], d)
    return layers


def jug_ml(scene, j, t):
    return scene["jugs"][j]["start_ml"] - sum(poured_at(p, t) for p in scene["pours"] if p["jug"] == j)


def leak_flowing(scene, g, t):
    """Colour of the liquid leaving glass g's hole at time t, or None."""
    for lk in scene["leaks"]:
        if lk["glass"] == g and lk["t_crack"] < t < lk["t_drain_end"]:
            h = 0.0
            for col, ml in glass_layers(scene, g, t):
                if h + ml > lk["hole_ml"] + 1e-6:
                    return col
                h += ml
    return None


# ------------------------------------------------------------------ checks
def check(scene) -> list[str]:
    errs = []
    jugs, pours, leaks = scene["jugs"], scene["pours"], scene["leaks"]
    n = scene["glasses"]["n"]
    dur = (scene["n_frames"] - 1) / FPS
    if scene["n_frames"] > MAX_FRAMES:
        errs.append("clip longer than 10 s")
    if len({j["pattern"] for j in jugs}) != len(jugs) or len({j["colour"] for j in jugs}) != len(jugs):
        errs.append("two jugs share a pattern or a colour")
    if pours[0]["t_move"] < EDGE_S:
        errs.append("first pour starts too early")
    ends = [p["t_end"] for p in pours] + [lk["t_drain_end"] for lk in leaks]
    if max(ends) > dur - EDGE_S:
        errs.append("an event ends too close to the end of the clip")
    left = [j["start_ml"] for j in jugs]
    filled = [0] * n
    for i, p in enumerate(pours):
        if p["ask_ml"] % STEP or p["got_ml"] % STEP or p["ask_ml"] <= 0:
            errs.append(f"pour {i}: amounts not multiples of {STEP} ml")
        if p["got_ml"] != min(p["ask_ml"], left[p["jug"]]):
            errs.append(f"pour {i}: got != min(ask, jug left)")
        left[p["jug"]] -= p["got_ml"]
        filled[p["glass"]] += p["got_ml"]
        if filled[p["glass"]] > CAP:
            errs.append(f"pour {i}: glass {p['glass'] + 1} overflows")
        if i and (p["t_move"] - pours[i - 1]["t_end"]) * FPS < 2:
            errs.append(f"pour {i}: starts < 2 frames after the previous jug is back")
        if 0 < p["got_ml"] < p["ask_ml"] and p["t_untilt"] - p["t_flow_end"] < 0.4:
            errs.append(f"pour {i}: dry-out not visible long enough")
        if p["got_ml"] == p["ask_ml"] and abs(p["t_untilt"] - p["t_flow_end"]) > 1e-3:
            errs.append(f"pour {i}: jug stays tilted after a full pour")
        if p["got_ml"] and p["t_flow_end"] - p["t_pour"] < MIN_FLOW_S - 1e-3:
            errs.append(f"pour {i}: stream shorter than {MIN_FLOW_S} s")
    for lk in leaks:
        g = lk["glass"]
        ins = [p for p in pours if p["glass"] == g and p["got_ml"]]
        if not ins or lk["t_crack"] < max(p["t_flow_end"] for p in ins) + 0.3:
            errs.append(f"leak in glass {g + 1}: cracks < 0.3 s after its last pour")
        if any(p["glass"] == g and p["t_move"] >= lk["t_crack"] for p in pours):
            errs.append(f"leak in glass {g + 1}: poured into after cracking")
        if filled[g] - lk["hole_ml"] < 2 * STEP:
            errs.append(f"leak in glass {g + 1}: loses < 20 ml")
    return errs
