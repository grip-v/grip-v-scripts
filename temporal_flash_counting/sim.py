"""Temporal flash counting (v3): shapes that light up ("flash") over time.

A clip (1280x720, 24 fps, 6-10 s) shows 4-30 dim shapes. A flash is the
shape lighting up to its full colour (or to near-white on a "white" flash)
for a whole number of frames, then going dim again. A decoy "pulse" makes a
shape grow and shrink without lighting up; it is not a flash. All timing is
in integer frames: flash k of shape i is lit on frames [f0, f1).

Scenarios (scene["scenario"]):
  grid     a rows x columns grid of shapes (2x2 .. 5x6) on optional plates
  scatter  4-9 shapes of different sizes / rotations at random spots
  relay    a hidden cause: one shape's flashes are followed about one second
           later by a flash of a "responder" (sometimes a chain of three)
  signal   4-6 round lamps on a panel blinking short / long patterns
  drift    4-6 shapes gliding left-right in their own lanes while flashing

Each shape flashes at a steady rhythm (fixed interval, +-1 frame jitter) or
at irregular times. One "designated pair" of shapes has its closest flashes
set to start together, overlap, or follow one another with a clear gap.
`check` rejects any scene whose question-relevant timing lacks a margin;
`build` retries (same rng, deterministic) until a scene passes.
"""
from __future__ import annotations
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.shapes import TYPES  # noqa: E402

FPS = 24
W, H = 1280, 720
AREA = (32, 100, 1248, 700)          # every shape (incl. glow / pulse) stays inside; the timer sits above
SCENARIOS = ("grid", "scatter", "relay", "signal", "drift")
EDGE_F = 12                           # no flash before 0.5 s or after T - 0.5 s
# Flash lengths and dark gaps in frames. "gap" is also the clip's timing margin (orders, windows,
# relations); "cut" is the short / long boundary the questions name (s).
TEMPO = {"fast": {"short": (6, 9), "long": (20, 30), "gap": 8, "cut": 0.6},
         "slow": {"short": (12, 14), "long": (27, 36), "gap": 12, "cut": 0.8}}
PULSE_SCALE = 0.3                     # a pulse grows the shape by up to 30 %
GLOW = 1.22                           # halo radius factor while lit
REACH = 1.3                           # max drawn radius / base radius (pulse)
LAG_WIN = (0.6, 1.4)                  # read as a response: starts 0.6-1.4 s after a trigger flash
CLEAR_WIN = (0.35, 1.65)               # a spontaneous response flash is outside this window of every trigger flash
LAG_WIN_F = (math.ceil(LAG_WIN[0] * FPS), math.floor(LAG_WIN[1] * FPS))
CLEAR_WIN_F = (round(CLEAR_WIN[0] * FPS), round(CLEAR_WIN[1] * FPS))

COLORS = {"red": (232, 66, 63), "orange": (242, 140, 40), "yellow": (245, 211, 59), "lime": (166, 224, 58),
          "green": (47, 179, 90), "teal": (34, 195, 181), "light blue": (102, 199, 244), "blue": (59, 111, 232),
          "purple": (154, 92, 230), "pink": (240, 106, 184), "brown": (168, 116, 63)}
BGS = {"navy": (16, 20, 38), "charcoal": (24, 24, 28), "forest": (14, 30, 22), "plum": (32, 18, 36),
       "slate": (22, 30, 40), "cocoa": (34, 25, 20), "black": (8, 8, 10)}
TEXTURES = ("plain", "vignette", "dots", "stripes", "grid", "noise")
SAME_TYPE_DIST = 110                  # two shapes of one type need colours this far apart (RGB)


# ---- small helpers ---------------------------------------------------------------
def tenths(f: int) -> int:
    """The on-screen timer reading at frame f, in tenths of a second."""
    return f * 10 // FPS


def timer_text(f: int) -> str:
    t = tenths(f)
    return f"{t // 10}.{t % 10} s"


def frame_of_reading(tn: int) -> int:
    """First frame whose timer shows `tn` tenths."""
    return -(-tn * FPS // 10)


def fmt_reading(tn: int) -> str:
    return f"{tn // 10}.{tn % 10} s"


def cdist(a, b) -> float:
    return math.dist(a, b)


def label(scene, i) -> str:
    s = scene["shapes"][i]
    return f"{s['color']} lamp" if scene["scenario"] == "signal" else f"{s['color']} {s['type']}"


def flashes_of(scene, i) -> list[dict]:
    return sorted((f for f in scene["flashes"] if f["i"] == i), key=lambda f: f["f0"])


def margin(scene) -> int:
    """Timing margin in frames (0.33 s fast, 0.5 s slow)."""
    return TEMPO[scene["tempo"]]["gap"]


def cut_s(scene) -> float:
    return TEMPO[scene["tempo"]]["cut"]


def is_long(f, cut: float) -> bool:
    return f["f1"] - f["f0"] > cut * FPS


def onset_cv(onsets) -> float | None:
    g = [b - a for a, b in zip(onsets, onsets[1:])]
    return statistics.pstdev(g) / statistics.mean(g) if len(g) >= 2 else None


def position(scene, i, f) -> tuple[float, float]:
    """Centre of shape i at frame f (drift shapes bounce left-right in their lane)."""
    s = scene["shapes"][i]
    m = s.get("motion")
    if not m:
        return s["x"], s["y"]
    span = m["xb"] - m["xa"]
    u = (m["x0"] - m["xa"] + m["v"] * f) % (2 * span)
    x = m["xa"] + (u if u <= span else 2 * span - u)
    y = s["y"] + m["bob"] * math.sin(2 * math.pi * f / m["bob_p"] + m["bob_ph"])
    return x, y


def links(scene) -> list[tuple[int, int]]:
    """(trigger, responder) shape pairs of the relay chain."""
    ch = (scene.get("causal") or {}).get("chain") or []
    return list(zip(ch, ch[1:]))


def resimulate(scene, removed: set[int]) -> list[dict]:
    """Flashes that would still happen if the shapes in `removed` never
    flashed: every flash caused (directly or through a chain) by a removed
    flash disappears too; spontaneous flashes stay."""
    gone = {f["id"] for f in scene["flashes"] if f["i"] in removed}
    changed = True
    while changed:
        changed = False
        for f in scene["flashes"]:
            if f["id"] not in gone and f["cause"] is not None and f["cause"] in gone:
                gone.add(f["id"])
                changed = True
    return [f for f in scene["flashes"] if f["id"] not in gone]


def lag_margins(scene, trig, resp) -> list[int]:
    """For every responder onset near a trigger onset: how many frames the
    trigger -> response delay is from the edge of LAG_WIN (inside or outside).
    Sampling every k frames keeps every reading on its side if k <= margin."""
    out = []
    for f in flashes_of(scene, resp):
        for g in flashes_of(scene, trig):
            d = f["f0"] - g["f0"]
            if -FPS <= d <= 3 * FPS:
                lo, hi = LAG_WIN_F
                out.append(min(d - lo, hi - d) + 1 if lo <= d <= hi else (lo - d if d < lo else d - hi))
    return out


def followed(scene, trig, resp) -> int:
    """How many of trig's flashes are followed (LAG_WIN) by an onset of resp."""
    rs = [f["f0"] for f in flashes_of(scene, resp)]
    return sum(1 for g in flashes_of(scene, trig) if any(LAG_WIN_F[0] <= r - g["f0"] <= LAG_WIN_F[1] for r in rs))


def closest_pair(scene, a, b):
    """(gap in frames to the runner-up, flash of a, flash of b) for the two flashes
    of a and b whose onsets are closest."""
    ps = sorted(((abs(x["f0"] - y["f0"]), x["f0"], y["f0"], x, y)
                 for x in flashes_of(scene, a) for y in flashes_of(scene, b)), key=lambda p: p[:3])
    second = ps[1][0] - ps[0][0] if len(ps) > 1 else 10 ** 6
    return second, ps[0][3], ps[0][4]


RELATIONS = ("they start at the same moment", "they overlap but start at different moments",
             "one ends before the other starts", "one starts at the exact moment the other ends")


def relation(x, y, m: int) -> str | None:
    """Relation of two flashes with margins (None if borderline)."""
    d = abs(x["f0"] - y["f0"])
    if d == 0:
        return RELATIONS[0]
    ov = min(x["f1"], y["f1"]) - max(x["f0"], y["f0"])
    if d >= m and ov >= m:
        return RELATIONS[1]
    if -ov >= m:
        return RELATIONS[2]
    return None


# ---- scene building ------------------------------------------------------------------
def _style(rng, scenario):
    return {"bg_name": rng.choice(sorted(BGS)), "texture": rng.choice(TEXTURES),
            "tex_step": rng.randint(28, 64), "tex_amp": rng.randint(5, 10),
            "outline": rng.randint(3, 6), "rest_mix": round(rng.uniform(0.2, 0.3), 3),
            "white_mix": round(rng.uniform(0.8, 0.88), 3), "glow_mix": round(rng.uniform(0.45, 0.62), 3),
            "plate": rng.choice(["rounded", "circle", "none"]) if scenario in ("grid", "relay") else "none",
            "rails": scenario == "drift" and rng.random() < 0.6}


def _grid_layout(rng, max_cells):
    combos = [(r, c) for r in range(1, 6) for c in range(2, 7) if 4 <= r * c <= max_cells]
    rows, cols = rng.choice(combos)
    aw, ah = AREA[2] - AREA[0], AREA[3] - AREA[1]
    scale = rng.uniform(0.72, 1.0)
    cell = min(aw * scale / cols, ah * scale / rows, 200)
    gx = AREA[0] + rng.uniform(0, aw - cell * cols)
    gy = AREA[1] + rng.uniform(0, ah - cell * rows)
    rf = rng.uniform(0.28, 0.335)
    rot_mode = rng.random() < 0.3
    out = []
    for k in range(rows * cols):
        r, c = divmod(k, cols)
        out.append({"x": round(gx + (c + 0.5) * cell, 1), "y": round(gy + (r + 0.5) * cell, 1),
                    "r": round(cell * rf, 1), "rot": round(rng.uniform(0, 360), 1) if rot_mode else 0.0,
                    "row": r + 1, "col": c + 1})
    return {"rows": rows, "cols": cols, "cell": round(cell, 1)}, out


def _scatter_layout(rng, n):
    out = []
    for _ in range(n):
        for _ in range(300):
            r = rng.uniform(28, 66)
            x = rng.uniform(AREA[0] + REACH * r + 4, AREA[2] - REACH * r - 4)
            y = rng.uniform(AREA[1] + REACH * r + 4, AREA[3] - REACH * r - 4)
            if all(math.dist((x, y), (o["x"], o["y"])) >= REACH * (r + o["r"]) + 18 for o in out):
                out.append({"x": round(x, 1), "y": round(y, 1), "r": round(r, 1), "rot": round(rng.uniform(0, 360), 1)})
                break
        else:
            return None
    return out


def _signal_layout(rng, n):
    aw = AREA[2] - AREA[0]
    r = rng.uniform(40, min(72, aw * 0.94 / n / 2.75))
    sp = rng.uniform(2.75, 3.4) * r
    sp = min(sp, (aw - 2 * REACH * r - 30) / (n - 1))
    width = sp * (n - 1)
    x0 = rng.uniform(AREA[0] + REACH * r + 15, AREA[2] - REACH * r - 15 - width)
    y = rng.uniform(AREA[1] + REACH * r + 30, AREA[3] - REACH * r - 30)
    out = [{"x": round(x0 + k * sp, 1), "y": round(y, 1), "r": round(r, 1), "rot": 0.0} for k in range(n)]
    panel = [round(x0 - 1.75 * r, 1), round(y - 1.75 * r, 1), round(x0 + width + 1.75 * r, 1), round(y + 1.75 * r, 1)]
    return out, panel


def _drift_layout(rng, n, n_frames):
    ah = AREA[3] - AREA[1]
    lane = ah / n
    bob = rng.choice([0, 0, rng.randint(3, 8)])
    rmax = min(56, (lane - 18 - 2 * bob) / (2 * REACH))
    out = []
    for k in range(n):
        r = rng.uniform(24, max(24, rmax))
        y = AREA[1] + (k + 0.5) * lane
        xa = AREA[0] + REACH * r + 4 + rng.uniform(0, 120)
        xb = AREA[2] - REACH * r - 4 - rng.uniform(0, 120)
        v = rng.uniform(30, 150) / FPS * rng.choice([-1, 1])
        out.append({"x": round(rng.uniform(xa, xb), 1), "y": round(y, 1), "r": round(r, 1),
                    "rot": round(rng.uniform(0, 360), 1) if rng.random() < 0.5 else 0.0,
                    "motion": {"xa": round(xa, 1), "xb": round(xb, 1), "v": round(v, 3), "bob": bob,
                               "bob_p": rng.randint(40, 90), "bob_ph": round(rng.uniform(0, 6.283), 3)}})
        out[-1]["motion"]["x0"] = out[-1]["x"]
    rng.shuffle(out)
    return out


def _identities(rng, n, scenario):
    same_type = scenario != "signal" and n <= 7 and rng.random() < 0.15
    pool_t = ["circle"] if scenario == "signal" else ([rng.choice(TYPES)] if same_type else TYPES)
    ids = []
    for _ in range(n):
        for _ in range(100):
            t = rng.choice(pool_t)
            c = rng.choice(sorted(COLORS))
            if all(not (t == t2 and cdist(COLORS[c], COLORS[c2]) < SAME_TYPE_DIST) for t2, c2 in ids):
                ids.append((t, c))
                break
        else:
            return None
    return ids


def _dur(rng, tp, p_long):
    lo, hi = TEMPO[tp]["long" if rng.random() < p_long else "short"]
    return rng.randint(lo, hi)


def fits(fl, n_frames, gap) -> bool:
    fl = sorted(fl, key=lambda f: f["f0"])
    return all(f["f0"] >= EDGE_F and f["f1"] <= n_frames - EDGE_F for f in fl) and \
        all(b["f0"] - a["f1"] >= gap for a, b in zip(fl, fl[1:]))


def _mk(i, f0, d, white, cause=None):
    return {"i": i, "f0": f0, "f1": f0 + d, "white": white, "cause": cause}


def _schedule(rng, i, sh, nf, tp, kmax=5):
    gap = TEMPO[tp]["gap"]
    pl, pw = sh["p_long"], sh["p_white"]
    for _ in range(80):
        if sh["rhythm"] == "steady":
            iv = round(rng.choice([1.0, 1.5, 2.0, 2.5]) * FPS + rng.uniform(-2.4, 2.4))
            start = rng.randint(EDGE_F, EDGE_F + iv // 2)
            fl, k = [], 0
            while True:
                f0 = start + k * iv + rng.randint(-1, 1)
                d = _dur(rng, tp, pl)
                if f0 + d > nf - EDGE_F:
                    break
                fl.append(_mk(i, f0, d, rng.random() < pw))
                k += 1
            if len(fl) >= 3 and fits(fl, nf, gap) and (onset_cv([f["f0"] for f in fl]) or 1) <= 0.06:
                return fl
        else:
            k = rng.randint(2, kmax)
            fl = []
            for _ in range(k):
                d = _dur(rng, tp, pl)
                fl.append(_mk(i, rng.randint(EDGE_F, nf - EDGE_F - d), d, rng.random() < pw))
            fl.sort(key=lambda f: f["f0"])
            cv = onset_cv([f["f0"] for f in fl])
            if fits(fl, nf, gap) and (cv is None or cv >= 0.3):
                return fl
    return None


def _relay(rng, chain, shapes, nf, tp):
    """Flashes of the chain shapes: trigger spontaneous, responders mostly caused."""
    gap = TEMPO[tp]["gap"]
    lag = round(rng.uniform(0.92, 1.08) * FPS)          # 22-26 frames, +-1 jitter: >= 6 frames inside LAG_WIN
    depth = len(chain) - 1
    trig = chain[0]
    out = {}
    for _ in range(60):
        k = rng.randint(4, 5)
        hi = nf - EDGE_F - depth * (lag + 3) - 34
        f0, fl = rng.randint(EDGE_F, EDGE_F + 20), []
        while len(fl) < k and f0 <= hi:
            fl.append(_mk(trig, f0, _dur(rng, tp, shapes[trig]["p_long"]), rng.random() < shapes[trig]["p_white"]))
            f0 += rng.randint(30, 52)
        if len(fl) >= 4 and fits(fl, nf, gap):
            out[trig] = fl
            break
    else:
        return None
    for a, b in zip(chain, chain[1:]):
        for _ in range(60):
            new = []
            for pf in out[a]:
                if rng.random() < 0.9:
                    new.append(_mk(b, pf["f0"] + lag + rng.randint(-1, 1), _dur(rng, tp, shapes[b]["p_long"]),
                                   rng.random() < shapes[b]["p_white"], cause=pf))
            for _ in range(rng.randint(0, 2 if b == chain[1] else 1)):
                d = _dur(rng, tp, shapes[b]["p_long"])
                new.append(_mk(b, rng.randint(EDGE_F, nf - EDGE_F - d), d, rng.random() < shapes[b]["p_white"]))
            new.sort(key=lambda f: f["f0"])
            if fits(new, nf, gap) and sum(f["cause"] is not None for f in new) >= (4 if a == chain[0] else 3):
                out[b] = new
                break
        else:
            return None
    return out


def _attempt(rng, scenario):
    nf = rng.randint(204, 240) if scenario == "relay" else rng.randint(144, 240)
    tp = "slow" if scenario == "signal" else rng.choice(sorted(TEMPO))
    style = _style(rng, scenario)
    extra = {}
    if scenario == "grid":
        g, geo = _grid_layout(rng, 30)
        extra.update(g)
    elif scenario == "relay":
        if rng.random() < 0.5:
            g, geo = _grid_layout(rng, 12)
            extra.update(g)
        else:
            geo = _scatter_layout(rng, rng.randint(5, 9))
    elif scenario == "scatter":
        geo = _scatter_layout(rng, rng.randint(4, 9))
    elif scenario == "signal":
        geo, panel = _signal_layout(rng, rng.randint(4, 6))
        style["panel"] = panel
    else:
        geo = _drift_layout(rng, rng.randint(4, 6), nf)
    if geo is None:
        return None
    n = len(geo)
    ids = _identities(rng, n, scenario)
    if ids is None:
        return None
    p_white = rng.choice([0.0, 0.2, 0.35])
    shapes = []
    for g, (t, c) in zip(geo, ids):
        sh = dict(g)
        sh.update({"type": t, "color": c, "aspect": round(rng.uniform(1.45, 1.9), 2),
                   "rhythm": "steady" if rng.random() < 0.35 else "random",
                   "p_long": rng.choice([0.0, 0.25, 0.45]), "p_white": rng.choice([0.0, p_white, 0.45])})
        shapes.append(sh)
    causal = None
    chain = []
    if scenario == "relay":
        chain = rng.sample(range(n), 3 if n >= 6 and rng.random() < 0.45 else 2)
        for k in chain:
            shapes[k]["rhythm"] = "relay"              # neither steady nor a steady-question distractor
            shapes[k]["p_long"] = min(shapes[k]["p_long"], 0.25)
    flashes = []
    for i, sh in enumerate(shapes):
        if i in chain:
            continue
        fl = _schedule(rng, i, sh, nf, tp, 4 if chain else 5)
        if fl is None:
            return None
        flashes += fl
    if chain:
        rel = _relay(rng, chain, shapes, nf, tp)
        if rel is None:
            return None
        for k in chain:
            flashes += rel[k]
        causal = {"chain": chain}
    # the designated pair: two non-chain shapes; one flash of the irregular one (b) is moved
    pair, kind = None, None
    cands = [k for k in range(n) if k not in chain]
    gap = TEMPO[tp]["gap"]
    m = gap
    want = rng.choice(RELATIONS[:3])
    for _ in range(40 if len(cands) >= 2 else 0):
        a, b = rng.sample(cands, 2)
        if shapes[b]["rhythm"] == "steady":
            a, b = b, a
        if shapes[b]["rhythm"] != "random":
            continue
        fa = [f for f in flashes if f["i"] == a]
        fb = [f for f in flashes if f["i"] == b]
        kind = want
        pool = [f for f in fa if f["f1"] - f["f0"] >= 2 * m + 2] if kind == RELATIONS[1] else fa
        if not pool:
            kind = None
            continue
        anchor, mover = rng.choice(pool), rng.choice(fb)
        old = (mover["f0"], mover["f1"])
        d, da = mover["f1"] - mover["f0"], anchor["f1"] - anchor["f0"]
        if kind == RELATIONS[1] and d < 2 * m + 2:
            d = rng.randint(*TEMPO[tp]["long"])
        if kind == RELATIONS[0]:
            f0 = anchor["f0"]
        elif kind == RELATIONS[1]:
            f0 = anchor["f0"] + rng.choice([-1, 1]) * rng.randint(m + 1, min(d, da) - m - 1)
        else:
            f0 = anchor["f1"] + rng.randint(m + 2, m + 16) if rng.random() < 0.5 else \
                anchor["f0"] - d - rng.randint(m + 2, m + 16)
        mover["f0"], mover["f1"] = f0, f0 + d
        tmp = {"flashes": fa + fb}
        cv = onset_cv(sorted(f["f0"] for f in fb))
        ok = fits(fb, nf, gap) and (cv is None or cv >= 0.3)
        if ok:
            gap2, x, y = closest_pair(tmp, a, b)
            ok = gap2 >= 12 and relation(x, y, m) == kind
        if ok:
            pair = [a, b]
            break
        mover["f0"], mover["f1"] = old
        kind = None
    # decoy pulses (not on lamps)
    pulses = []
    if scenario != "signal":
        p_pulse = rng.choice([0.0, 0.25, 0.45])
        for i in range(n):
            if rng.random() < p_pulse:
                own = [f for f in flashes if f["i"] == i]
                for _ in range(rng.randint(1, 2)):
                    d = rng.randint(12, 20)
                    f0 = rng.randint(EDGE_F, nf - EDGE_F - d)
                    p = {"i": i, "f0": f0, "f1": f0 + d}
                    if all(p["f1"] + 6 <= o["f0"] or o["f1"] + 6 <= p["f0"] for o in own + [q for q in pulses if q["i"] == i]):
                        pulses.append(p)
    # ids and causes
    flashes.sort(key=lambda f: (f["f0"], f["i"]))
    for k, f in enumerate(flashes):
        f["id"] = k
    for f in flashes:
        if isinstance(f["cause"], dict):
            f["cause"] = f["cause"]["id"]
    pulses.sort(key=lambda p: (p["f0"], p["i"]))
    for sh in shapes:
        sh.pop("p_long")
        sh.pop("p_white")
    scene = {"scenario": scenario, "n_frames": nf, "tempo": tp, "style": style, "shapes": shapes,
             "flashes": flashes, "pulses": pulses, "causal": causal, "pair": pair, "pair_kind": kind}
    scene.update(extra)
    return scene


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    tries = 0
    while True:
        tries += 1
        scene = _attempt(rng, scenario)
        if scene is not None and not check(scene):
            scene["tries"] = tries
            return scene


# ---- checks -------------------------------------------------------------------------
def max_radius(scene, i) -> float:
    pulsed = any(p["i"] == i for p in scene["pulses"])
    return scene["shapes"][i]["r"] * (max(1 + PULSE_SCALE, GLOW) if pulsed else GLOW)


def check(scene: dict) -> list[str]:
    errs = []
    sc, nf, n = scene["scenario"], scene["n_frames"], len(scene["shapes"])
    tp = TEMPO[scene["tempo"]]
    if sc not in SCENARIOS:
        errs.append("unknown scenario")
    if not 6 * FPS <= nf <= 10 * FPS:
        errs.append("clip length out of range")
    if n < 4:
        errs.append("fewer than 4 shapes")
    labs = [label(scene, i) for i in range(n)]
    if len(set(labs)) != n:
        errs.append("two shapes share a colour + type")
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = scene["shapes"][i], scene["shapes"][j]
            if si["type"] == sj["type"] and cdist(COLORS[si["color"]], COLORS[sj["color"]]) < SAME_TYPE_DIST:
                errs.append("two shapes of one type have similar colours")
    # geometry: inside the area, never overlapping (checked on every frame for drifting shapes)
    frames = range(0, nf, 3) if sc == "drift" else [0]
    for f in frames:
        pos = [position(scene, i, f) for i in range(n)]
        for i in range(n):
            x, y = pos[i]
            R = max_radius(scene, i)
            if x - R < AREA[0] or x + R > AREA[2] or y - R < AREA[1] or y + R > AREA[3]:
                errs.append(f"{labs[i]} leaves the action area")
            for j in range(i + 1, n):
                if math.dist(pos[i], pos[j]) < R + max_radius(scene, j) + 10:
                    errs.append(f"{labs[i]} and {labs[j]} come too close")
        if errs:
            break
    if min(s["r"] for s in scene["shapes"]) < 24:
        errs.append("a shape is too small")
    for i in range(n):
        fl = flashes_of(scene, i)
        if len(fl) < 2 or not fits(fl, nf, tp["gap"]):
            errs.append(f"{labs[i]}: too few flashes, off the clip edges, or too close together")
        for f in fl:
            d = f["f1"] - f["f0"]
            if not (tp["short"][0] <= d <= tp["short"][1] or tp["long"][0] <= d <= tp["long"][1]):
                errs.append(f"{labs[i]}: a flash of borderline length")
        cv = onset_cv([f["f0"] for f in fl])
        if scene["shapes"][i]["rhythm"] == "steady" and (cv is None or cv > 0.06 or len(fl) < 3):
            errs.append(f"{labs[i]}: 'steady' rhythm isn't steady")
        if scene["shapes"][i]["rhythm"] == "random" and cv is not None and cv < 0.3:
            errs.append(f"{labs[i]}: irregular rhythm looks steady")
        for p in scene["pulses"]:
            if p["i"] == i and any(not (p["f1"] + 6 <= f["f0"] or f["f1"] + 6 <= p["f0"]) for f in fl):
                errs.append(f"{labs[i]}: a pulse touches a flash")
            if p["i"] == i and not (EDGE_F <= p["f0"] and p["f1"] <= nf - EDGE_F):
                errs.append(f"{labs[i]}: a pulse too close to the clip edges")
    if scene["pair"]:
        gap2, x, y = closest_pair(scene, *scene["pair"])
        if gap2 < 12 or relation(x, y, margin(scene)) != scene["pair_kind"]:
            errs.append("the designated pair's timing is ambiguous")
    for trig, resp in links(scene):
        tf = flashes_of(scene, trig)
        for f in flashes_of(scene, resp):
            par = [g for g in tf if LAG_WIN_F[0] <= f["f0"] - g["f0"] <= LAG_WIN_F[1]]
            near = [g for g in tf if CLEAR_WIN_F[0] <= f["f0"] - g["f0"] <= CLEAR_WIN_F[1]]
            if f["cause"] is not None and (len(par) != 1 or par[0]["id"] != f["cause"] or len(near) != 1):
                errs.append("a response flash's cause is ambiguous")
            if f["cause"] is None and near:
                errs.append("a spontaneous response flash sits near a trigger flash")
        # a trigger lag must also be clear of every other shape's flashes being read as causes
        score = followed(scene, trig, resp)
        weak = [k for k in range(n) if k not in (trig, resp) and followed(scene, k, resp) <= score - 2]
        if score < 3 or len(weak) < 3:
            errs.append("the trigger isn't clearly the trigger")
    # read from timing alone, the clear "B reacts to A" pairs must be exactly the relay links
    seen = set()
    for b in range(n):
        sc_ = {a: sum(1 for f in flashes_of(scene, b) if any(LAG_WIN_F[0] <= f["f0"] - g["f0"] <= LAG_WIN_F[1]
                                                           for g in flashes_of(scene, a))) for a in range(n) if a != b}
        best = max(sc_, key=lambda a: (sc_[a], -a))
        if sc_[best] >= 3 and all(sc_[a] <= sc_[best] - 2 for a in sc_ if a != best):
            seen.add((best, b))
    if seen != set(links(scene)):
        errs.append("timing suggests a reaction that isn't part of the relay (or hides one that is)")
    if not errs:
        import questions
        sup = questions.supply(scene)
        if any(s < 3 for s in sup):
            errs.append(f"too few candidate questions per level {sup}")
    return errs
