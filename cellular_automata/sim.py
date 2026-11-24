"""Scene generation for cellular_automata (New Problem Types B11).

A grid of cells evolves under one hidden birth/survival rule. A dead
(empty) cell is BORN when its number of filled neighbours is in the rule's
B set; a filled cell SURVIVES when its count is in the S set. The rule is
never shown; it has to be inferred from how the grid changes.

Five scenarios (scene["scenario"]):

    life_like     Moore neighbourhood (8 neighbours), cells beyond the edge
                  count as empty, 2 states.
    torus         Moore neighbourhood on a wrap-around grid (left edge
                  touches right, top touches bottom; drawn with a dashed
                  border). The pattern always reaches across an edge, so the
                  wrap changes what happens.
    von_neumann   only the 4 side neighbours count, edges empty, 2 states.
    generations   Moore, 3 states: a filled cell that fails to survive fades
                  (second colour) for one step, then empties; fading cells
                  are not counted as neighbours and cannot be re-born.
    elementary    a 1-D row; each generation is drawn as a new row under the
                  previous one (a space-time diagram). A cell's neighbours are
                  the two cells beside it in the row above.

Everything (rule, grid size, start layout, timing, colours, cell shape,
framing, marked cells) comes from the seeded rng. `check` rejects scenes
whose rule is not identifiable from the generations shown, that barely
change, that grow too crowded to count, or whose timing breaks the
frame-sampling rules.

Timing is in whole frames (24 fps): generation 0 holds for start_f frames;
the change to generation k starts at frame s_k = start_f + (k-1)*gen_f and
fades over fade_f frames; generation G then holds for end_f frames.
"""
from __future__ import annotations
import random

import numpy as np

FPS = 24
MAX_FRAMES = 240                 # 10 s
SCENARIOS = ("life_like", "torus", "von_neumann", "generations", "elementary")
MAX_POP = 60                     # cap on filled (+fading) cells in any one generation
MAX_TRIES = 400

POOLS = {
    "moore": ["B3/S23", "B36/S23", "B2/S", "B3678/S34678", "B3/S12345", "B36/S125", "B3/S012345678",
              "B35678/S5678", "B368/S245", "B37/S23", "B38/S23", "B3/S45678", "B357/S1358", "B34/S34",
              "B345/S5"],
    "vn": ["B1/S", "B1/S1", "B1/S01", "B1/S12", "B1/S013", "B2/S", "B2/S1", "B2/S12", "B12/S", "B13/S",
           "B13/S1", "B2/S23", "B3/S123", "B23/S01"],
    "gen3": ["B2/S", "B2/S345", "B2/S34", "B23/S", "B3/S23", "B34/S23", "B3/S234", "B24/S3"],
    "elem": [f"B{b}/S{s}" for b in ("1", "2", "12") for s in ("", "0", "1", "2", "01", "02", "12", "012")],
}
SCENARIO_SETUP = {   # scenario -> (pool, neighbourhood, wrap, states)
    "life_like": ("moore", "moore", False, 2),
    "torus": ("moore", "moore", True, 2),
    "von_neumann": ("vn", "vn", False, 2),
    "generations": ("gen3", "moore", False, 3),
    "elementary": ("elem", "elem", False, 2),
}
OFFSETS = {
    "moore": [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc],
    "vn": [(-1, 0), (1, 0), (0, -1), (0, 1)],
    "elem": [(0, -1), (0, 1)],
}
N_NEIGH = {"moore": 8, "vn": 4, "elem": 2}

# ---- styles -----------------------------------------------------------------
THEMES = {
    "light": {"bg": [(222, 219, 211), (206, 214, 222), (230, 226, 236), (214, 224, 210)],
              "dead": [(247, 246, 242), (240, 244, 248), (250, 247, 236)],
              "ink": (40, 42, 50)},
    "dark": {"bg": [(20, 22, 28), (28, 24, 34), (18, 30, 30)],
             "dead": [(52, 56, 68), (60, 58, 70), (48, 62, 62)],
             "ink": (236, 236, 236)},
}
LIVE = {
    "light": [("navy blue", (30, 55, 140)), ("black", (25, 25, 30)), ("dark green", (20, 105, 55)),
              ("purple", (105, 45, 150)), ("brown", (115, 65, 25)), ("teal", (0, 115, 125))],
    "dark": [("yellow", (250, 215, 50)), ("white", (245, 245, 245)), ("cyan", (60, 210, 235)),
             ("lime green", (150, 235, 60)), ("orange", (250, 145, 35))],
}
DYING = {
    "light": [("light blue", (125, 170, 235)), ("pink", (245, 150, 195)), ("tan", (210, 175, 115)),
              ("lavender", (190, 165, 230)), ("grey", (155, 155, 155))],
    "dark": [("maroon", (140, 35, 65)), ("olive", (120, 120, 30)), ("slate blue", (85, 85, 175)),
             ("dark teal", (25, 115, 115))],
}
MARKER_COLS = {
    "light": [("red", (225, 30, 30)), ("orange", (245, 135, 0)), ("magenta", (225, 0, 205)),
              ("bright green", (0, 200, 70)), ("sky blue", (40, 170, 255))],
    "dark": [("red", (255, 55, 55)), ("magenta", (255, 70, 230)), ("sky blue", (110, 200, 255)),
             ("white", (250, 250, 250)), ("orange", (255, 150, 30))],
}
SHAPES_2D = ("ring", "square frame", "diamond frame")
SHAPES_1D = ("triangle", "dot", "square")
CELL_SHAPES = ("square", "rounded", "circle")
AREA = (250, 30, 1250, 700)      # the grid always sits inside this box; the timer lives top-left


def parse(rule: str) -> tuple[list[int], list[int]]:
    b, s = rule.split("/")
    return [int(c) for c in b[1:]], [int(c) for c in s[1:]]


def describe(rule: str, scenario: str) -> str:
    """Plain-English rule text (neighbour counts only; no rule names)."""
    b, s = parse(rule)
    nn = N_NEIGH[SCENARIO_SETUP[scenario][1]]

    def nums(d):
        d = [str(x) for x in d]
        return d[0] if len(d) == 1 else ", ".join(d[:-1]) + " or " + d[-1]
    if scenario == "elementary":
        born = f"a new cell under an empty cell is filled when {nums(b)} of the 2 cells diagonally above it are filled"
        if not s:
            keep = "a new cell under a filled cell is always empty"
        elif len(s) == 3:
            keep = "a new cell under a filled cell is always filled"
        else:
            keep = f"a new cell under a filled cell is filled when {nums(s)} of those 2 cells are filled"
        return f"{born}; {keep}"
    born = f"an empty cell fills with {nums(b)} filled neighbours (of its {nn})"
    if not s:
        keep = "a filled cell never stays filled"
    elif len(s) == nn + 1:
        keep = "a filled cell always stays filled"
    else:
        keep = f"a filled cell stays filled with {nums(s)}"
    return f"{born}; {keep}"


# ---- dynamics (dense numpy; answers.py uses an independent sparse method) ----
def step(state: np.ndarray, rule: str, nbhd: str, wrap: bool, states: int) -> np.ndarray:
    b, s = parse(rule)
    live = (state == 1).astype(np.int16)
    n = np.zeros_like(live)
    if wrap:
        for dr, dc in OFFSETS[nbhd]:
            n += np.roll(live, (-dr, -dc), axis=(0, 1))
    else:
        p = np.pad(live, 1)
        R, C = live.shape
        for dr, dc in OFFSETS[nbhd]:
            n += p[1 + dr:1 + dr + R, 1 + dc:1 + dc + C]
    born = np.isin(n, b) & (state == 0)
    keep = np.isin(n, s) & (state == 1)
    out = np.zeros_like(state)
    out[born | keep] = 1
    if states == 3:
        out[(state == 1) & ~keep] = 2
    return out


def history(scene: dict, rule: str | None = None, start=None, gens: int | None = None, wrap=None) -> list:
    rule = rule or scene["rule"]
    g = np.array(scene["start"] if start is None else start, dtype=np.int16)
    w = scene["wrap"] if wrap is None else wrap
    out = [g]
    for _ in range(scene["gens"] if gens is None else gens):
        out.append(step(out[-1], rule, scene["nbhd"], w, scene["states"]))
    return out


def as_lists(h) -> list:
    return [g.astype(int).tolist() for g in h]


def live_count(g) -> int:
    return int((np.asarray(g) == 1).sum())


# ---- timeline -----------------------------------------------------------------
def n_frames(scene) -> int:
    t = scene["timing"]
    return t["start_f"] + (scene["gens"] - 1) * t["gen_f"] + t["end_f"]


def change_frame(scene, k: int) -> int:
    """First frame of the change from generation k-1 to k (k >= 1)."""
    t = scene["timing"]
    return t["start_f"] + (k - 1) * t["gen_f"]


def stable_window(scene, k: int) -> tuple[int, int]:
    """Inclusive frame range where generation k is fully drawn and still."""
    t = scene["timing"]
    if k == 0:
        return 0, t["start_f"] - 1
    lo = change_frame(scene, k) + t["fade_f"] - 1
    hi = change_frame(scene, k + 1) - 1 if k < scene["gens"] else n_frames(scene) - 1
    return lo, hi


def tenths(frame: int) -> int:
    """The on-screen timer reading at `frame`, in tenths of a second."""
    return frame * 10 // FPS


def fmt_t(tn: int) -> str:
    return f"{tn // 10}.{tn % 10}"


def time_ref(scene, k: int) -> str | None:
    """A timer reading (e.g. "3.4") whose every frame shows generation k still,
    >= 2 frames from any change and >= 0.5 s from both clip ends."""
    lo, hi = stable_window(scene, k)
    lo, hi = lo + 2, hi - 2
    lo, hi = max(lo, FPS // 2), min(hi, n_frames(scene) - 1 - FPS // 2)
    cands = []
    for tn in range(tenths(lo), tenths(hi) + 1):
        fr = [f for f in range(lo - 3, hi + 4) if tenths(f) == tn]
        if fr and fr[0] >= lo and fr[-1] <= hi:
            cands.append(tn)
    if not cands:
        return None
    return fmt_t(cands[len(cands) // 2])


def min_fps(scene, gens) -> float:
    """Sampling rate that guarantees one sample inside every relied-on
    generation's still window (1 decimal, rounded up)."""
    w = min(stable_window(scene, k)[1] - stable_window(scene, k)[0] + 1 for k in gens)
    return float(np.ceil(FPS / w * 10) / 10)


# ---- checks -----------------------------------------------------------------
def check(scene: dict) -> list:
    errs = []
    sc = scene["scenario"]
    pool, nbhd, wrap, states = SCENARIO_SETUP[sc]
    if (scene["nbhd"], scene["wrap"], scene["states"]) != (nbhd, wrap, states) or scene["rule"] not in POOLS[pool]:
        errs.append("scenario setup mismatch")
        return errs
    h = history(scene)
    if as_lists(h) != scene["history"]:
        errs.append("stored history doesn't match a replay")
    G = scene["gens"]
    pops = [int((g > 0).sum()) for g in h]
    if live_count(h[0]) < 3 or live_count(h[1]) == 0:
        errs.append("pattern too small or dies at once")
    if max(pops) > MAX_POP:
        errs.append("too crowded to count")
    if sc != "elementary" and any(np.array_equal(a, b) for a, b in zip(h, h[1:])):
        errs.append("a step leaves the grid unchanged (that generation would be invisible)")
    if sc == "elementary" and sum(1 for a, b in zip(h, h[1:]) if not np.array_equal(a, b)) < 0.5 * G:
        errs.append("rows barely change")
    for other in POOLS[pool]:
        if other != scene["rule"]:
            if all(np.array_equal(step(a, other, nbhd, wrap, states), b) for a, b in zip(h, h[1:])):
                errs.append(f"rule {other} is indistinguishable")
    if wrap and as_lists(history(scene, wrap=False)) == scene["history"]:
        errs.append("torus scene never uses the wrap")
    # timing
    t = scene["timing"]
    if n_frames(scene) > MAX_FRAMES:
        errs.append("clip longer than 10 s")
    if t["start_f"] < FPS * 0.8 or t["end_f"] < FPS or t["gen_f"] < 14 or t["fade_f"] < 2 or \
            t["gen_f"] - t["fade_f"] < 9:
        errs.append("timing too tight")
    for k in range(G + 1):
        if time_ref(scene, k) is None:
            errs.append(f"no clean time reference for generation {k}")
    # markers: distinct, in range, apart, each changes at least once
    ms = scene["markers"]
    if len({(m["colour"]) for m in ms}) != len(ms) or len({m["shape"] for m in ms}) != len(ms):
        errs.append("markers not uniquely identifiable")
    cells = [tuple(m["cell"]) for m in ms]
    for i, (r, c) in enumerate(cells):
        if not (0 <= r < scene["rows"] and 0 <= c < scene["cols"]):
            errs.append("marker outside grid")
            continue
        if len({int(g[r, c]) for g in h}) < 2:
            errs.append("marked cell never changes")
        for r2, c2 in cells[:i]:
            if max(abs(r - r2), abs(c - c2)) < 2:
                errs.append("markers too close")
    if not errs:
        import questions   # local import: questions imports sim
        if min(questions.supply(scene)) < 3:
            errs.append("too few distinct questions on some level")
    st = scene["style"]
    if st["cell_px"] < 24:
        errs.append("cells too small")
    return errs


# ---- build ---------------------------------------------------------------------
def _start(rng: random.Random, sc: str, R: int, C: int) -> tuple[str, list]:
    g = [[0] * C for _ in range(R)]
    if sc == "elementary":
        layout = rng.choice(["single", "few", "random", "block"])
        mid = C // 2
        if layout == "single":
            g[0][mid + rng.randint(-3, 3)] = 1
        elif layout == "few":
            for c in rng.sample(range(2, C - 2), rng.randint(2, 4)):
                g[0][c] = 1
        elif layout == "random":
            d = rng.uniform(0.2, 0.5)
            lo = rng.randint(1, C // 4)
            for c in range(lo, C - lo):
                g[0][c] = 1 if rng.random() < d else 0
        else:
            w = rng.randint(2, 5)
            c0 = rng.randint(2, C - 2 - w)
            for c in range(c0, c0 + w):
                g[0][c] = 1
        return layout, g
    layout = rng.choice(["patch", "two_patches", "mirror", "scatter"])

    def patch(h, w, dens, r0=None, c0=None, mirror=False):
        r0 = rng.randint(0, R - h) if r0 is None else r0
        c0 = rng.randint(0, C - w) if c0 is None else c0
        for r in range(h):
            for c in range(w):
                if rng.random() < dens:
                    g[r0 + r][c0 + c] = 1
                    if mirror:
                        g[r0 + r][c0 + w - 1 - c] = 1
    if sc == "torus":   # sit on an edge so the wrap matters
        h, w = rng.randint(3, 6), rng.randint(3, 6)
        if rng.random() < 0.5:
            patch(h, w, rng.uniform(0.4, 0.65), r0=rng.choice([0, R - h]))
        else:
            patch(h, w, rng.uniform(0.4, 0.65), c0=rng.choice([0, C - w]))
        if layout in ("two_patches", "scatter"):
            patch(rng.randint(2, 4), rng.randint(2, 4), rng.uniform(0.4, 0.7))
        return layout, g
    if layout == "patch":
        patch(rng.randint(3, 7), rng.randint(3, 8), rng.uniform(0.35, 0.6))
    elif layout == "two_patches":
        patch(rng.randint(2, 4), rng.randint(2, 5), rng.uniform(0.4, 0.7))
        patch(rng.randint(2, 4), rng.randint(2, 5), rng.uniform(0.4, 0.7))
    elif layout == "mirror":
        patch(rng.randint(3, 6), rng.randint(4, 8), rng.uniform(0.3, 0.55), mirror=True)
    else:
        d = rng.uniform(0.07, 0.16)
        r0, c0 = rng.randint(0, R // 4), rng.randint(0, C // 4)
        for r in range(r0, R - r0):
            for c in range(c0, C - c0):
                if rng.random() < d:
                    g[r][c] = 1
    return layout, g


def _far(c1, c2) -> float:
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def _style(rng: random.Random, sc: str, R_disp: int, C: int) -> dict:
    theme = rng.choice(["light", "light", "dark"])
    T = THEMES[theme]
    bg, dead = rng.choice(T["bg"]), rng.choice(T["dead"])
    live_name, live = rng.choice(LIVE[theme])
    dying_name, dying = None, None
    if sc == "generations":
        dying_name, dying = rng.choice([d for d in DYING[theme] if _far(d[1], live) > 110 and _far(d[1], dead) > 90])
    cell_shape = rng.choice(CELL_SHAPES)
    x0a, y0a, x1a, y1a = AREA
    top_pad = 40 if sc == "elementary" else 0     # room for column markers
    fit = min((x1a - x0a) // C, (y1a - y0a - top_pad) // R_disp, 52)
    cell_px = max(24, int(fit * rng.uniform(0.78, 1.0)))
    gap = rng.randint(3, max(3, min(6, cell_px // 7)))
    W, H = cell_px * C, cell_px * R_disp + top_pad
    x0 = rng.randint(x0a, x1a - W)
    y0 = rng.randint(y0a, y1a - H) + top_pad
    return {"theme": theme, "bg": list(bg), "dead": list(dead), "ink": list(T["ink"]), "live": list(live),
            "live_name": live_name, "dying": list(dying) if dying else None, "dying_name": dying_name,
            "cell_shape": cell_shape, "cell_px": cell_px, "gap": gap, "x0": x0, "y0": y0}


def _markers(rng: random.Random, sc: str, h: list, style: dict, n: int) -> list | None:
    theme = style["theme"]
    cols = [c for c in MARKER_COLS[theme]
            if all(_far(c[1], style[k]) > 110 for k in ("live", "dead", "bg") if style[k])
            and (not style["dying"] or _far(c[1], style["dying"]) > 110)
            and c[0] not in (style["live_name"], style["dying_name"])]
    R, C = h[0].shape
    cand = [(r, c) for r in range(R) for c in range(C) if len({int(g[r, c]) for g in h}) > 1]
    rng.shuffle(cand)
    picked = []
    for rc in cand:
        if all(max(abs(rc[0] - p[0]), abs(rc[1] - p[1])) >= 2 for p in picked):
            picked.append(rc)
        if len(picked) == n:
            break
    if len(picked) < n or len(cols) < n:
        return None
    colours = rng.sample(cols, n)
    shapes = rng.sample(SHAPES_1D if sc == "elementary" else SHAPES_2D, n)
    return [{"cell": list(rc), "colour": cn, "rgb": list(cr), "shape": sh}
            for rc, (cn, cr), sh in zip(picked, colours, shapes)]


def build(rng: random.Random) -> dict:
    sc = rng.choice(SCENARIOS)
    pool, nbhd, wrap, states = SCENARIO_SETUP[sc]
    for tries in range(1, MAX_TRIES + 1):
        rule = rng.choice(POOLS[pool])
        timing = {"start_f": rng.randint(20, 34), "gen_f": rng.randint(14, 26), "fade_f": rng.randint(2, 5),
                  "end_f": rng.randint(26, 40)}
        gmax = (MAX_FRAMES - timing["start_f"] - timing["end_f"]) // timing["gen_f"] + 1
        gens = rng.randint(min(6, gmax), min(12, gmax))
        if sc == "elementary":
            R, C = 1, rng.randrange(21, 42, 2)
            R_disp = gens + 1
        else:
            R, C = rng.randint(9, 15), rng.randint(11, 22)
            R_disp = R
        layout, start = _start(rng, sc, R, C)
        scene = {"scenario": sc, "rule": rule, "nbhd": nbhd, "wrap": wrap, "states": states,
                 "rows": R, "cols": C, "gens": gens, "start_layout": layout, "start": start,
                 "timing": timing}
        h = history(scene)
        scene["history"] = as_lists(h)
        style = _style(rng, sc, R_disp, C)
        markers = _markers(rng, sc, h, style, rng.choice([2, 2, 3]))
        if markers is None:
            continue
        scene["style"], scene["markers"], scene["tries"] = style, markers, tries
        if not check(scene):
            return scene
    raise RuntimeError("no valid scene found")
