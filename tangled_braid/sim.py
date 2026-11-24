"""Tangled braid (after Jane Street "Tangled!", Sep 2020).

3-8 strands come out of a loom bar side by side, each with its own colour
(unique within a clip), line style and thickness. The braid is woven at the
bar and scrolls away from it, so older crossings leave the frame: which
strand went over at an early crossing, and how the strands got reordered,
is only recoverable by watching the whole clip.

A crossing swaps the strands at two neighbouring places i and i+1; the
strand coming from the lower place ("low": left, or top when the braid runs
sideways) or the higher place ("high") passes over. A step is one moment of
weaving: one crossing, or (scenario `parallel`) 2-3 crossings at the same
time on places that don't touch.

Scenarios (`scene["scenario"]`):
  classic   vertical braid, one crossing per step, steady weaving
  parallel  vertical braid, some steps cross 2-3 pairs at the same moment
  sideways  the braid runs left-right; places are counted from the top
  pauses    the loom stops for 0.6-1.25 s between some crossings
  pure      the crossings bring every strand back to its starting place

Time: the braid's length coordinate s (px) at the bar is a piecewise-linear
function of the frame; step k occupies frames a..a+c (c even) and braid
length S..S+L, and its strands meet at frame m = a + c/2 (s = S + L/2).
All randomness comes from the rng passed to build().
"""
from __future__ import annotations
import hashlib
import json
import random

FPS = 24
W, H = 1280, 720
MAX_FRAMES = 240
EDGE = 12                      # frames (0.5 s) between any event and the clip's ends
SCENARIOS = ("classic", "parallel", "sideways", "pauses", "pure")

COLORS = {"red": "#D7372F", "blue": "#2E62D9", "green": "#2C9A4B", "orange": "#F07C1A",
          "purple": "#8446C8", "yellow": "#D9A800", "pink": "#E8509E", "brown": "#8A5426",
          "teal": "#14A3A3", "gray": "#6E7480"}
STYLES = ("solid", "dashed", "dotted", "double", "outlined", "beaded", "banded")
WIDTHS = {"thin": 7, "medium": 10, "thick": 13}
CURVES = ("smooth", "cosine", "smoother")
BGS = {"cream": "#F7F4EC", "ice": "#EDF3F8", "sand": "#F2ECDF", "mint": "#ECF5EC",
       "blush": "#F7EDF0", "paper": "#F3F3F1", "ivory": "#FBF8EF", "slate": "#E9EDF3"}
PATTERNS = ("plain", "grid", "dots", "gradient", "hatch")
LOOMS = {"walnut": "#5B4636", "steel": "#3C4654", "forest": "#2F4632", "plum": "#4A3B58"}
TIMER_BOX = (16, 14, 196, 70)
TIMER_FONT = 36
FADE = 40                      # px over which the braid fades out at the far edge
N_RANGE = {"classic": (4, 8), "parallel": (4, 8), "sideways": (4, 6), "pauses": (4, 7), "pure": (4, 6)}
STATS = {"builds": 0, "tries": 0}


# ---------------------------------------------------------------- clock

def timer_tenths(f: int) -> int:
    """What the on-screen clock shows at frame f, in tenths of a second (truncated)."""
    return (10 * f) // FPS


def clock_text(tenths: int) -> str:
    return f"{tenths // 10}.{tenths % 10} s"


def frames_showing(tenths: int) -> tuple[int, int]:
    """(first, last) frame whose clock reads `tenths`."""
    return -(-tenths * FPS // 10), -(-(tenths + 1) * FPS // 10) - 1


# ---------------------------------------------------------------- replay (generator side)

def replay(start: list, steps: list) -> tuple[list, list]:
    """Orders before every step (+ the final order) and one event per crossing:
    dict(step, i, low, high, over, under, m)."""
    order = list(start)
    orders, events = [list(order)], []
    for k, st in enumerate(steps):
        for mv in st["moves"]:
            i = mv["i"]
            lo, hi = order[i], order[i + 1]
            over = lo if mv["over"] == "low" else hi
            events.append({"step": k, "i": i, "low": lo, "high": hi, "over": over,
                           "under": hi if over == lo else lo, "m": st.get("m")})
        for mv in st["moves"]:
            i = mv["i"]
            order[i], order[i + 1] = order[i + 1], order[i]
        orders.append(list(order))
    return orders, events


def final_order(scene) -> list:
    return replay(scene["start"], scene["steps"])[0][-1]


def strand(scene, name) -> dict:
    return next(s for s in scene["strands"] if s["name"] == name)


# ---------------------------------------------------------------- timing helpers

def head_s(scene, f: float) -> float:
    """Braid length (px) at the loom bar at frame f."""
    tm, steps = scene["timing"], scene["steps"]
    v, L = tm["speed"], tm["L"]
    if f <= steps[0]["a"]:
        return f * v
    for k, st in enumerate(steps):
        a, c, S = st["a"], st["c"], st["S"]
        if f <= a + c:
            return S + L * (f - a) / c
        e = a + c
        nxt = steps[k + 1] if k + 1 < len(steps) else None
        if nxt is None or f <= nxt["a"]:
            sp = 0.0 if (nxt is not None and nxt["pause"]) else v
            return S + L + (f - e) * sp
    raise AssertionError("unreachable")


def window(scene) -> float:
    lay = scene["layout"]
    return abs(lay["loom"] - lay["far"])


def visible_until(scene, k: int) -> int:
    """Last frame on which step k's meeting point is still inside the window."""
    sm = scene["steps"][k]["S"] + scene["timing"]["L"] / 2
    Wn = window(scene)
    last = scene["steps"][k]["m"]
    for f in range(scene["steps"][k]["m"], scene["n_frames"]):
        if head_s(scene, f) - sm <= Wn - 6:
            last = f
        else:
            break
    return last


def vis_frames(scene, k: int) -> int:
    """How many frames step k's crossing(s) can be seen after the strands meet."""
    return visible_until(scene, k) - scene["steps"][k]["m"] + 1


def quiet_spans(scene) -> list:
    """(first, last) frames when no strand is moving at the loom bar."""
    st = scene["steps"]
    out = [(0, st[0]["a"])]
    for k in range(len(st) - 1):
        out.append((st[k]["a"] + st[k]["c"], st[k + 1]["a"]))
    out.append((st[-1]["a"] + st[-1]["c"], scene["n_frames"] - 1))
    return out


def order_at_frame(scene, f: int):
    """Order at the bar during a quiet span (None while strands move)."""
    orders, _ = replay(scene["start"], scene["steps"])
    for k, (lo, hi) in enumerate(quiet_spans(scene)):
        if lo <= f <= hi:
            return orders[k]
    return None


# ---------------------------------------------------------------- generation

def _strands(rng, n):
    cols = list(COLORS)
    rng.shuffle(cols)
    out = []
    for c in cols[:n]:
        st = rng.choice(STYLES)
        out.append({"name": f"{c} {st}", "color_name": c, "color": COLORS[c], "style": st,
                    "thickness": rng.choice(list(WIDTHS))})
    return out


def _single_moves(rng, n, k_steps, forbid_undo=True):
    steps = []
    while len(steps) < k_steps:
        mv = {"i": rng.randrange(n - 1), "over": rng.choice(["low", "high"])}
        if forbid_undo and steps and steps[-1]["moves"][0]["i"] == mv["i"] \
                and steps[-1]["moves"][0]["over"] != mv["over"]:
            continue
        steps.append({"moves": [mv]})
    return steps


def _parallel_moves(rng, n, k_steps):
    steps = []
    for _ in range(k_steps):
        g = rng.choices([1, 2, 3], weights=[3, 5, 2])[0]
        free = list(range(n - 1))
        picked = []
        while free and len(picked) < g:
            i = rng.choice(free)
            picked.append(i)
            free = [j for j in free if abs(j - i) >= 2]
        steps.append({"moves": [{"i": i, "over": rng.choice(["low", "high"])} for i in sorted(picked)]})
    if sum(len(s["moves"]) >= 2 for s in steps) < 2:
        return None
    return steps


def _pure_moves(rng, n, k_steps):
    r = rng.randint(2, max(2, k_steps // 2))
    steps = _single_moves(rng, n, r, forbid_undo=True)
    start = list(range(n))
    order, _ = replay(start, steps)
    cur = list(order[-1])
    if cur == start:
        return None
    undo = []
    while cur != start:
        cand = [p for p in range(n - 1) if cur[p] > cur[p + 1]]
        p = rng.choice(cand)
        undo.append({"moves": [{"i": p, "over": rng.choice(["low", "high"])}]})
        cur[p], cur[p + 1] = cur[p + 1], cur[p]
    steps += undo
    if not 4 <= len(steps) <= k_steps + 3:
        return None
    return steps


def _timing(rng, sc, steps):
    c_base = rng.choice([10, 12, 14, 16, 18])
    L = rng.randint(70, 150)
    v = round(L / c_base, 4)
    intro = rng.randint(14, 26)
    outro = rng.randint(16, 30)
    f = intro
    S = intro * v
    n_pause = 0
    for k, st in enumerate(steps):
        if k:
            pause = sc == "pauses" and rng.random() < 0.5
            gap = rng.randint(14, 30) if pause else rng.randint(4, 12)
            n_pause += pause
            st["pause"] = pause
            S += 0 if pause else gap * v
            f += gap
        else:
            st["pause"] = False
        st["c"] = max(10, c_base + rng.choice([-2, 0, 0, 2]))
        st["a"] = f
        st["m"] = f + st["c"] // 2
        st["S"] = round(S, 4)
        f += st["c"]
        S += L
    if sc == "pauses" and n_pause < 2:
        return None
    nf = f + outro
    if nf > MAX_FRAMES:
        return None
    return {"c_base": c_base, "L": L, "speed": v, "intro": intro, "outro": outro}, nf


def _layout(rng, sc, n):
    if sc == "sideways":
        grow = rng.choice([1, -1])                    # +1: woven at the right, braid scrolls left
        gap = rng.randint(72, min(115, 520 // (n - 1)))
        span = gap * (n - 1)
        lo_c, hi_c = 150 + span / 2, 670 - span / 2
        cu = rng.randint(int(lo_c), int(hi_c)) if hi_c > lo_c else int((lo_c + hi_c) / 2)
        loom = rng.randint(1130, 1185) if grow > 0 else rng.randint(95, 150)
        far = rng.randint(55, 110) if grow > 0 else rng.randint(1170, 1225)
        return {"orient": "h", "grow": grow, "gap": gap, "u0": cu - span / 2, "loom": loom, "far": far}
    grow = rng.choice([1, 1, -1])                     # +1: woven at the bottom, braid scrolls up
    gap = rng.randint(70, min(150, 960 // (n - 1)))
    span = gap * (n - 1)
    lo_c, hi_c = 150 + span / 2, 1130 - span / 2
    cu = rng.randint(int(lo_c), int(hi_c)) if hi_c > lo_c else int((lo_c + hi_c) / 2)
    loom = rng.randint(585, 640) if grow > 0 else rng.randint(118, 160)
    far = rng.randint(100, 135) if grow > 0 else rng.randint(655, 700)
    return {"orient": "v", "grow": grow, "gap": gap, "u0": cu - span / 2, "loom": loom, "far": far}


def _style(rng):
    bg = rng.choice(list(BGS))
    return {"bg": bg, "bg_hex": BGS[bg], "pattern": rng.choice(PATTERNS), "loom": rng.choice(list(LOOMS)),
            "curve": rng.choice(CURVES), "halo": rng.randint(5, 8), "pattern_step": rng.randint(36, 64)}


def _try(rng, sc):
    STATS["tries"] += 1
    lo, hi = N_RANGE[sc]
    n = rng.randint(lo, hi)
    k_steps = rng.randint(5, 10) if sc != "pauses" else rng.randint(4, 7)
    if sc == "parallel":
        steps = _parallel_moves(rng, n, rng.randint(4, 7))
    elif sc == "pure":
        steps = _pure_moves(rng, n, k_steps)
    else:
        steps = _single_moves(rng, n, k_steps)
    if steps is None:
        return None
    t = _timing(rng, sc, steps)
    if t is None:
        return None
    tm, nf = t
    strands = _strands(rng, n)
    start = [s["name"] for s in strands]
    rng.shuffle(start)
    scene = {"scenario": sc, "strands": strands, "start": start, "steps": steps, "timing": tm,
             "n_frames": nf, "duration": round(nf / FPS, 6), "layout": _layout(rng, sc, n),
             "style": _style(rng), "timer": {"box": list(TIMER_BOX), "size": TIMER_FONT},
             "labels": []}
    return scene


def build(rng: random.Random) -> dict:
    STATS["builds"] += 1
    sc = rng.choice(SCENARIOS)
    while True:
        s = _try(rng, sc)
        if s is not None and not check(s):
            return s


def signature(scene) -> str:
    """Hash of the rounded scene parameters (diversity statistics)."""
    def rnd(x):
        if isinstance(x, float):
            return round(x, 2)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rnd(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(rnd(scene), sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- checks

def columns(scene) -> list:
    lay = scene["layout"]
    return [lay["u0"] + lay["gap"] * p for p in range(len(scene["start"]))]


def layout_errors(scene) -> list:
    errs = []
    lay = scene["layout"]
    us = columns(scene)
    wmax = max(WIDTHS[s["thickness"]] for s in scene["strands"]) + 8
    tb = TIMER_BOX
    if lay["orient"] == "v":
        if us[0] - wmax < 20 or us[-1] + wmax > W - 20:
            errs.append("strands leave the frame")
        lo_v = min(lay["loom"], lay["far"]) - 30
        if lo_v < tb[3] + 10:
            errs.append("braid or bar reaches the timer")
        if max(lay["loom"], lay["far"]) + 30 > H:
            errs.append("bar leaves the frame")
    else:
        if us[0] - wmax < tb[3] + 30 or us[-1] + wmax > H - 20:
            errs.append("strands reach the timer or leave the frame")
        if min(lay["loom"], lay["far"]) - 30 < 0 or max(lay["loom"], lay["far"]) + 30 > W:
            errs.append("bar leaves the frame")
    if lay["gap"] < 70:
        errs.append("strands closer than 70 px")
    if window(scene) < 420:
        errs.append("viewing window too short")
    return errs


def scenario_errors(scene) -> list:
    sc, steps = scene["scenario"], scene["steps"]
    errs = []
    n = len(scene["start"])
    lo, hi = N_RANGE[sc]
    if not lo <= n <= hi:
        errs.append("strand count out of range for the scenario")
    multi = sum(len(s["moves"]) > 1 for s in steps)
    if sc == "parallel" and multi < 2:
        errs.append("parallel: fewer than 2 steps with simultaneous crossings")
    if sc != "parallel" and multi:
        errs.append("simultaneous crossings outside the parallel scenario")
    if sc == "pure" and final_order(scene) != scene["start"]:
        errs.append("pure: strands don't return to their places")
    if sc == "pauses" and sum(s["pause"] for s in steps) < 2:
        errs.append("pauses: fewer than 2 pauses")
    if sc != "pauses" and any(s["pause"] for s in steps):
        errs.append("pause outside the pauses scenario")
    if (sc == "sideways") != (scene["layout"]["orient"] == "h"):
        errs.append("orientation doesn't match the scenario")
    return errs


def timing_errors(scene) -> list:
    errs = []
    st = scene["steps"]
    nf = scene["n_frames"]
    if nf > MAX_FRAMES:
        errs.append("clip longer than 10 s")
    if st[0]["a"] < EDGE:
        errs.append("first crossing starts within 0.5 s of the start")
    end = st[-1]["a"] + st[-1]["c"]
    if nf - 1 - end < EDGE:
        errs.append("last crossing ends within 0.5 s of the end")
    for k, s in enumerate(st):
        if s["c"] % 2 or s["c"] < 10 or s["m"] != s["a"] + s["c"] // 2:
            errs.append(f"step {k}: bad crossing duration")
        if k and s["a"] - (st[k - 1]["a"] + st[k - 1]["c"]) < 4:
            errs.append(f"step {k}: less than 4 still frames before it")
        if k and s["m"] - st[k - 1]["m"] < 12:
            errs.append(f"step {k}: meets within 0.5 s of the previous step")
        if vis_frames(scene, k) < 14:
            errs.append(f"step {k}: crossing visible for under 14 frames")
        for mv in s["moves"]:
            if not 0 <= mv["i"] < len(scene["start"]) - 1 or mv["over"] not in ("low", "high"):
                errs.append(f"step {k}: bad move")
        idx = sorted(mv["i"] for mv in s["moves"])
        if any(b - a < 2 for a, b in zip(idx, idx[1:])):
            errs.append(f"step {k}: simultaneous crossings touch")
    if abs(scene["duration"] - nf / FPS) > 1e-6:
        errs.append("duration doesn't match n_frames")
    return errs


def check(scene) -> list:
    errs = []
    names = [s["name"] for s in scene["strands"]]
    cols = [s["color_name"] for s in scene["strands"]]
    if len(set(names)) != len(names) or len(set(cols)) != len(cols):
        errs.append("strand names or colours repeat")
    if any(s["style"] not in STYLES or s["thickness"] not in WIDTHS for s in scene["strands"]):
        errs.append("unknown strand style or thickness")
    if sorted(scene["start"]) != sorted(names):
        errs.append("start order doesn't list every strand once")
    if len(replay(scene["start"], scene["steps"])[1]) < 4:
        errs.append("fewer than 4 crossings")
    errs += scenario_errors(scene) + timing_errors(scene) + layout_errors(scene)
    return errs
