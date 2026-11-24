"""Spring-chain waves: pulses travel along a horizontal chain of beads, bounce
off its ends and split where the beads change thickness.

The chain runs from ruler mark 0 to mark L. Its ends are held by a paddle
(it flicks pulses in and, being held, sends arriving pulses back upside
down), a clamp (also upside down) or a ring sliding on a pole (back the
same way up). Thick beads are heavier, so pulses there move 2 or 3 times
slower; where the thickness changes a pulse splits by the impedance-step
rule (impedance ~ 1/speed at equal tension):

    reflected   = (c2 - c1) / (c2 + c1) * incoming
    transmitted = 2 c2 / (c1 + c2) * incoming

Parts weaker than CUTOFF of a big launch are not followed (and not drawn).

Every pulse is one fixed profile (round / peaked / smooth) of duration
TAU = tau_f frames, so its width is speed * TAU. The clip is an exact list
of straight pulse segments (`trace`, in Fractions), and the displacement at
(x, t) is their sum, each clipped to its own section (method of images: a
held end stays still, a ring swings to double height).

Scenarios (`scene["scenario"]`):
  reflect   uniform chain, one pulse from a paddle, far end clamped or a ring
  junction  thin and thick beads (a step, or a thick band in the middle),
            one pulse that splits at every thickness change
  collide   a paddle at each end; each sends a pulse (one may send two);
            the pulses pass through each other and bounce between paddles
  train     uniform chain, a paddle sends 2-4 bumps / dips of two sizes
"""
from __future__ import annotations
import math
import random
from fractions import Fraction as Fr

import numpy as np

W, H, FPS = 1280, 720, 24
MAX_FRAMES = 240
SCENARIOS = ("reflect", "junction", "collide", "train")
TIMER_BOX = (14, 14, 196, 70)
TIMER_FONT = 36
LABEL_FONT = 36
EDGE = 12                       # frames: events stay this far from the clip's start and end
GAP = 12                        # frames: arrivals / launches at one point at least this far apart
END_AFTER = Fr(2, 5)            # s: no event in (T_last - 0.5 s, T_last + END_AFTER)
CUTOFF = Fr(3, 10)              # weakest pulse followed, relative to a big launch
SMALL = Fr(3, 5)                # amplitude of a "small" launched pulse
T_STOP = Fr(11)                 # trace horizon (s), beyond any clip
ISO_MIN = 8                     # frames each pulse must be seen on its own
ISO_PAD = 0.08                  # ruler units of clearance for "on its own"
PROFILES = {"round": 2.3, "peaked": 1.8, "smooth": 1.6}   # support, in pulse widths
TOP_LIMIT = 96                  # px: nothing moving above this (the clock panel ends at 70)

BACKGROUNDS = [  # (top, bottom, light?)
    ((246, 244, 238), (232, 228, 218), True), ((236, 242, 248), (214, 224, 236), True),
    ((244, 240, 228), (250, 248, 242), True), ((24, 28, 38), (40, 46, 60), False),
    ((30, 34, 30), (18, 22, 20), False), ((36, 28, 44), (22, 18, 30), False),
]
CHAIN_COLORS = {  # (thin, thick) per background lightness
    True: [((40, 96, 190), (170, 60, 40)), ((20, 130, 110), (120, 50, 150)), ((60, 60, 70), (200, 110, 20)),
           ((150, 40, 120), (30, 110, 60))],
    False: [((110, 180, 255), (255, 140, 90)), ((120, 230, 170), (230, 120, 230)), ((240, 240, 240), (255, 200, 60)),
            ((255, 130, 190), (130, 220, 120))],
}
BEAD_SHAPES = ("circle", "square", "diamond")


def tenths(f: int) -> int:
    return (10 * f) // FPS


def clock_text(k: int) -> str:
    return f"{k // 10}.{k % 10} s"


def frames_showing(k: int) -> tuple[int, int]:
    lo = -(-k * FPS // 10)
    hi = -(-(k + 1) * FPS // 10) - 1
    return lo, hi


def frame_of(t: Fr) -> int:
    """Frame on which an event at time t is shown (round half up, exact)."""
    return math.floor(t * FPS + Fr(1, 2))


def t_last(scene) -> Fr:
    return Fr(scene["n_frames"] - 1, FPS)


# ---------------------------------------------------------------- chain geometry and tracing (exact)

def speed(scene, k=Fr(1)) -> Fr:
    return Fr(scene["c4"], 4) * k


def sections(scene, k=Fr(1), swap=False, uniform=False, length=None) -> list[tuple]:
    """[(lo, hi, speed, heavy)] as Fractions."""
    c = speed(scene, k)
    L = Fr(length if length is not None else scene["L"])
    if uniform or len(scene["sections"]) == 1:
        return [(Fr(0), L, c, False)]
    out = []
    for lo, hi, heavy in scene["sections"]:
        hv = (not heavy) if swap else heavy
        out.append((Fr(lo), min(Fr(hi), L), c / scene["ratio"] if hv else c, hv))
    return [s for s in out if s[0] < s[1]]


def end_x(scene, side, L=None) -> Fr:
    return Fr(0) if side == "left" else Fr(L if L is not None else scene["L"])


def launch_list(scene) -> list[dict]:
    return scene["launches"]


def trace(scene, *, secs=None, ends=None, launches=None, t_stop=T_STOP) -> list[dict]:
    """Every pulse segment, in order of start time. Each segment carries its
    launch (`origin`), whether it has bounced off the far end (`via_far`),
    its parent index and how it was made."""
    secs = secs or sections(scene)
    ends = ends or scene["ends"]
    launches = launches if launches is not None else scene["launches"]
    L = secs[-1][1]
    home = scene.get("home")
    todo = []
    for k, ln in enumerate(launches):
        x = Fr(0) if ln["end"] == "left" else L
        amp = Fr(ln["sign"]) * (1 if ln["big"] else SMALL)
        todo.append((Fr(ln["f"], FPS), k, x, 1 if ln["end"] == "left" else -1, amp, k, False, -1, "launch"))
    seq = len(todo)
    out = []
    while todo and len(out) < 400:
        todo.sort(key=lambda z: (z[0], z[1]))
        tb, _, xb, d, a, org, vf, par, kind = todo.pop(0)
        si = next(i for i, s in enumerate(secs) if (s[0] <= xb < s[1] if d > 0 else s[0] < xb <= s[1]))
        lo, hi, c, heavy = secs[si]
        xe = hi if d > 0 else lo
        te = tb + abs(xe - xb) / c
        me = len(out)
        out.append({"tb": tb, "te": te, "xb": xb, "xe": xe, "d": d, "amp": a, "c": c, "sec": si, "lo": lo, "hi": hi,
                    "origin": org, "via_far": vf, "parent": par, "kind": kind})
        if te > t_stop:
            continue
        kids = []
        if xe == 0 or xe == L:
            side = "left" if xe == 0 else "right"
            r = 1 if ends[side] == "free" else -1
            far = home is not None and side != home
            kids.append((xe, -d, a * r, vf or far, "bounce"))
        else:
            c2 = secs[si + 1][2] if d > 0 else secs[si - 1][2]
            kids.append((xe, -d, a * (c2 - c) / (c2 + c), vf, "split_back"))
            kids.append((xe, d, a * 2 * c2 / (c + c2), vf, "split_on"))
        for x, dd, aa, v, kd in kids:
            if abs(aa) >= CUTOFF:
                todo.append((te, seq, x, dd, aa, org, v, me, kd))
                seq += 1
    return out


def ghosts(scene, segs) -> list[dict]:
    """Parts of a split weaker than CUTOFF. They are not followed and never
    counted; the renderer draws each only as its image plus a fade over one
    pulse duration after it starts, so the chain stays continuous where the
    beads change (without them the arriving pulse would leave a step)."""
    secs = sections(scene)
    L = secs[-1][1]
    out = []
    for s in segs:
        xe = s["xe"]
        if xe == 0 or xe == L:
            continue
        si = s["sec"]
        c, c2 = s["c"], (secs[si + 1][2] if s["d"] > 0 else secs[si - 1][2])
        for dd, aa in ((-s["d"], s["amp"] * (c2 - c) / (c2 + c)), (s["d"], s["amp"] * 2 * c2 / (c + c2))):
            if 0 < abs(aa) < CUTOFF:
                ti = next(i for i, z in enumerate(secs) if (z[0] <= xe < z[1] if dd > 0 else z[0] < xe <= z[1]))
                lo, hi, cc, _ = secs[ti]
                out.append({"tb": s["te"], "xb": xe, "d": dd, "amp": aa, "c": cc, "lo": lo, "hi": hi})
    return out


def visible(scene, segs) -> list[dict]:
    """Segments that can show up in the clip (images reach back S*TAU before tb)."""
    tl = t_last(scene)
    tau = Fr(scene["tau_f"], FPS)
    S = Fr(PROFILES[scene["profile"]]).limit_denominator(10)
    return [s for s in segs if s["tb"] - S * tau <= tl]


def stored(segs) -> list[dict]:
    keys = ("tb", "te", "xb", "xe", "c", "lo", "hi", "amp")
    return [{**{k: float(s[k]) for k in keys if k in s}, "d": s["d"]} for s in segs]


def stored_ghosts(scene, segs) -> list[dict]:
    S, tau = _support(scene)
    tl = t_last(scene)
    return stored([g for g in ghosts(scene, segs) if g["tb"] - S * tau <= tl])


# ---------------------------------------------------------------- displacement (numpy, float)

def profile(kind: str, z: np.ndarray) -> np.ndarray:
    a = np.abs(z)
    if kind == "round":
        return np.where(a < PROFILES["round"], np.exp(-z * z), 0.0)
    if kind == "peaked":
        return np.clip(1.0 - a / PROFILES["peaked"], 0.0, None)
    return np.where(a < PROFILES["smooth"], np.cos(np.pi * z / (2 * PROFILES["smooth"])) ** 2, 0.0)


def heights(scene, xs: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Displacement (in amplitude units) at chain positions xs on the given
    frames, from the stored segments. A sample at a section boundary belongs
    to the section on its right; the far end (x = L) to the last section."""
    L = float(scene["L"])
    tau = scene["tau_f"] / FPS
    S = PROFILES[scene["profile"]]
    t = frames.astype(float) / FPS
    y = np.zeros((len(frames), len(xs)))
    for s in scene["segs"]:
        on_t = (t >= s["tb"] - S * tau) & (t <= s["te"] + S * tau)
        if not on_t.any():
            continue
        on_x = (xs >= s["lo"]) & ((xs < s["hi"]) | ((s["hi"] == L) & (xs <= L)))
        if not on_x.any():
            continue
        ti = np.nonzero(on_t)[0]
        xi = np.nonzero(on_x)[0]
        w = s["c"] * tau
        pos = s["xb"] + s["d"] * s["c"] * (t[ti] - s["tb"])
        z = (xs[xi][None, :] - pos[:, None]) / w
        y[np.ix_(ti, xi)] += s["amp"] * profile(scene["profile"], z)
    for g in scene.get("ghosts", []):
        on_t = (t >= g["tb"] - S * tau) & (t <= g["tb"] + S * tau)
        on_x = (xs >= g["lo"]) & ((xs < g["hi"]) | ((g["hi"] == L) & (xs <= L)))
        if not on_t.any() or not on_x.any():
            continue
        ti = np.nonzero(on_t)[0]
        xi = np.nonzero(on_x)[0]
        fade = np.clip(1.0 - (t[ti] - g["tb"]) / (S * tau), 0.0, 1.0)
        pos = g["xb"] + g["d"] * g["c"] * (t[ti] - g["tb"])
        z = (xs[xi][None, :] - pos[:, None]) / (g["c"] * tau)
        y[np.ix_(ti, xi)] += (g["amp"] * fade)[:, None] * profile(scene["profile"], z)
    return y


def sample_xs(scene) -> np.ndarray:
    """Sample positions along the chain: every 2 px, plus both ends."""
    n = int(round(scene["L"] * scene["scale"] / 2))
    return np.linspace(0.0, float(scene["L"]), n + 1)


# ---------------------------------------------------------------- events and isolation (exact)

def _support(scene) -> tuple[Fr, Fr]:
    tau = Fr(scene["tau_f"], FPS)
    return Fr(PROFILES[scene["profile"]]).limit_denominator(10), tau


def pos_at(s, t):
    return s["xb"] + s["d"] * s["c"] * (t - s["tb"])


def _fl(s):
    if "_fl" not in s:
        s["_fl"] = tuple(float(s[k]) for k in ("tb", "te", "xb", "c", "lo", "hi")) + (s["d"],)
    return s["_fl"]


def isolated(scene, segs, i, f) -> bool:
    """Segment i is on its own on frame f: its whole bump is inside its
    section (with clearance) and no other drawn segment reaches it."""
    S = PROFILES[scene["profile"]]
    tau = scene["tau_f"] / FPS
    tb, te, xb, c, lo, hi, d = _fl(segs[i])
    t = f / FPS
    if not tb - 1e-9 <= t <= te + 1e-9:
        return False
    w = c * tau
    p = xb + d * c * (t - tb)
    ext = S * w + ISO_PAD
    if p - lo < ext or hi - p < ext:
        return False
    for j, o in enumerate(segs):
        otb, ote, oxb, oc, olo, ohi, od = _fl(o)
        if j == i or olo != lo or not (otb - S * tau <= t <= ote + S * tau):
            continue
        if abs(oxb + od * oc * (t - otb) - p) < 2 * S * w + ISO_PAD:
            return False
    return True


def iso_frames(scene, segs, i) -> list[int]:
    s = segs[i]
    f0 = math.ceil(s["tb"] * FPS)
    f1 = min(math.floor(s["te"] * FPS), scene["n_frames"] - 1)
    return [f for f in range(f0, f1 + 1) if isolated(scene, segs, i, f)]


def calm(scene, segs, f) -> bool:
    """Frame f shows only whole, separate pulses (nothing mid-bounce or mid-launch)."""
    S = PROFILES[scene["profile"]]
    tau = scene["tau_f"] / FPS
    t = f / FPS
    for i, s in enumerate(segs):
        tb, te = _fl(s)[:2]
        if tb - S * tau <= t <= te + S * tau and not isolated(scene, segs, i, f):
            return False
    return True


def active(segs, f) -> list[int]:
    t = Fr(f, FPS)
    return [i for i, s in enumerate(segs) if s["tb"] <= t <= s["te"]]


def crossings(segs) -> list[tuple]:
    """(t, x, i, j): two pulses from different launches passing through each
    other inside one section."""
    out = []
    for i, a in enumerate(segs):
        for j in range(i + 1, len(segs)):
            b = segs[j]
            if a["sec"] != b["sec"] or a["d"] == b["d"] or a["origin"] == b["origin"]:
                continue
            # a: xb + d c (t - tb) ; b: same section speed
            c = a["c"]
            t = (b["xb"] - a["xb"] + a["d"] * c * a["tb"] - b["d"] * c * b["tb"]) / ((a["d"] - b["d"]) * c)
            if max(a["tb"], b["tb"]) < t < min(a["te"], b["te"]):
                out.append((t, pos_at(a, t), i, j))
    return sorted(out, key=lambda z: (z[0], z[2], z[3]))


def boundary_events(scene, segs) -> dict:
    """Times (s) at which something happens at each boundary point: arrivals and launches."""
    ev = {}
    for s in segs:
        ev.setdefault(s["xe"], []).append(s["te"])
        if s["kind"] == "launch":
            ev.setdefault(s["xb"], []).append(s["tb"])
    return {x: sorted(set(v)) for x, v in ev.items()}


def far_side(scene):
    return None if scene["home"] is None else ("right" if scene["home"] == "left" else "left")


# ---------------------------------------------------------------- generation

def _style(rng):
    top, bot, light = rng.choice(BACKGROUNDS)
    thin, thick = rng.choice(CHAIN_COLORS[light])
    ink = (40, 42, 50) if light else (225, 228, 236)
    return {
        "bg": [list(top), list(bot)], "bg_style": rng.choice(["flat", "gradient", "vignette"]), "light": light,
        "thin": list(thin), "thick": list(thick), "ink": list(ink),
        "bead": rng.choice(BEAD_SHAPES), "bead_r": rng.choice([5, 6, 7, 8]), "line": rng.choice([3, 4]),
        "beads_per_unit": rng.choice([2, 3, 4]),
        "paddle": list(rng.choice([(90, 92, 104), (150, 90, 40), (60, 110, 150), (120, 70, 120)])),
        "metal": list(rng.choice([(130, 126, 118), (110, 118, 130), (150, 140, 120)])),
        "ring_r": rng.choice([12, 14, 16]),
    }


def _layout(scene, rng) -> bool:
    """Pick amplitude (px), rest line and ruler: the moving chain and the end
    hardware stay below the clock panel, the ruler sits just under the chain's
    lowest reach and its numbers stay inside the frame."""
    xs = sample_xs(scene)
    hy = heights(scene, xs, np.arange(scene["n_frames"]))
    up, down = max(0.0, float(hy.max())), max(0.0, float(-hy.min()))
    ends = hy[:, [0, -1]]
    e_reach = [(max(0.0, float(ends[:, k].max())), max(0.0, float(-ends[:, k].min()))) for k in range(2)]
    st = scene["style"]
    r = st["bead_r"] * 1.6 + 2
    gap = rng.randint(26, 60)
    amp = rng.uniform(58, 92)
    while amp >= 52:
        above, below = amp * up + r, amp * down + r            # chain extent around the rest line
        for k, side in enumerate(("left", "right")):
            eu, ed = e_reach[k]
            if scene["ends"][side] == "paddle":
                above, below = max(above, amp * eu + 40), max(below, amp * ed + 40)
            elif scene["ends"][side] == "free":
                above = max(above, amp * eu + st["ring_r"] + 24)
                below = max(below, amp * ed + st["ring_r"] + 24)
            else:
                above, below = max(above, 62), max(below, 62)
        # rest line y0: TOP_LIMIT + above <= y0 ; ruler = y0 + below + gap ; labels end at ruler + 30 + LABEL_FONT
        lo = TOP_LIMIT + above
        hi = H - 10 - LABEL_FONT - 30 - gap - below
        if lo <= hi:
            y0 = round(rng.uniform(lo, hi), 1)
            scene["amp_px"] = round(amp, 1)
            scene["y0"] = y0
            scene["ruler_y"] = int(math.ceil(y0 + below + gap))
            scene["reach"] = [round(up, 4), round(down, 4)]
            scene["end_reach"] = [[round(e, 4) for e in er] for er in e_reach]
            return True
        amp -= 3
    return False


def _labels(scene):
    return [{"text": str(i), "x": round(scene["x0"] + i * scene["scale"], 1), "y": scene["ruler_y"] + 30,
             "size": LABEL_FONT} for i in range(scene["L"] + 1)]


def _launch_f(rng, scene, lo_extra=0):
    S, tau = _support(scene)
    f_min = EDGE + math.ceil(S * tau * FPS) + 1
    return rng.randint(f_min + lo_extra, f_min + lo_extra + 14), f_min


def _attempt(rng: random.Random, scenario: str, n_pulses: int = 1):
    L = rng.choice([8, 10, 12])
    width = rng.uniform(820, 1050)
    scale = round(width / L, 1)
    span = L * scale
    x0 = round(rng.uniform(112, W - 112 - span), 1)
    # reflect / train must bring the first pulse home within the clip: 2L/c + ~1.5 s <= ~9.5 s
    c4 = rng.randint(max(6, L) if scenario in ("reflect", "train") else 6, 16)
    scene = {"scenario": scenario, "L": L, "scale": scale, "x0": x0, "c4": c4, "ratio": 1,
             "sections": [[0, L, False]], "profile": rng.choice(sorted(PROFILES)), "style": _style(rng)}
    crowded = n_pulses >= 3            # several pulses crossing: keep them narrow so each is seen on its own
    if crowded:
        scene["profile"] = rng.choice(["peaked", "smooth"])
    c = speed(scene)
    if scenario == "junction":
        ratio = rng.choice([2, 3]) if scene["c4"] >= 9 else 2
        scene["ratio"] = ratio
        if rng.random() < 0.7:
            j = rng.randint(3, L - 3)
            hl = rng.random() < 0.5
            scene["sections"] = [[0, j, hl], [j, L, not hl]]
        else:
            a = rng.randint(2, L - 5)
            b = rng.randint(a + 2, min(a + 4, L - 2))
            scene["sections"] = [[0, a, False], [a, b, True], [b, L, False]]
    c_min = c / scene["ratio"]
    taus = [t for t in range(3, 5 if crowded else 9)
            if c * Fr(t, FPS) <= Fr(3, 4) and float(c_min * Fr(t, FPS)) * scale >= 12]
    if not taus:
        return None
    scene["tau_f"] = rng.choice(taus)
    if scenario in ("reflect", "junction", "train"):
        home = rng.choice(["left", "right"])
        far = "right" if home == "left" else "left"
        scene["home"] = home
        scene["ends"] = {home: "paddle", far: rng.choice(["clamped", "free"])}
        f0, _ = _launch_f(rng, scene)
        if scenario == "train":
            n = n_pulses
            sizes = [True] * n if rng.random() < 0.3 else [rng.random() < 0.55 for _ in range(n)]
            # consecutive pulses must not overlap: gap > both footprints (2 S TAU) plus a margin
            S = PROFILES[scene["profile"]]
            gaps = [g for g in (12, 18, 24, 30) if g >= 2 * S * scene["tau_f"] + 3]
            if not gaps:
                return None
            fs = [f0]
            for _ in range(n - 1):
                fs.append(fs[-1] + rng.choice(gaps))
            scene["launches"] = [{"f": f, "end": home, "sign": rng.choice([1, -1]), "big": b} for f, b in zip(fs, sizes)]
        else:
            scene["launches"] = [{"f": f0, "end": home, "sign": rng.choice([1, -1]), "big": True}]
    else:
        scene["home"] = None
        scene["ends"] = {"left": "paddle", "right": "paddle"}
        m = rng.randint(2, L - 2)
        fl, f_min = _launch_f(rng, scene)
        fr = fl + round(FPS * (2 * m - L) / c)
        if fr < f_min:
            fl, fr = fl + f_min - fr, f_min
        if fr > 96 or fl > 96:
            return None
        ls = [{"f": fl, "end": "left", "sign": rng.choice([1, -1]), "big": rng.random() < 0.6},
              {"f": fr, "end": "right", "sign": rng.choice([1, -1]), "big": rng.random() < 0.6}]
        if n_pulses == 3:
            side = rng.choice(["left", "right"])
            base = fl if side == "left" else fr
            ls.append({"f": base + rng.choice([12, 18, 24, 30]), "end": side, "sign": rng.choice([1, -1]),
                       "big": rng.random() < 0.6})
        ls.sort(key=lambda z: (z["f"], z["end"]))
        scene["launches"] = ls
    segs = trace(scene)
    # clip length
    S, tau = _support(scene)
    need = _need(scene, segs)
    if need is None:
        return None
    lo_n = max(96, math.ceil((need + Fr(1, 2)) * FPS) + 1)
    times = sorted({s["tb"] for s in segs} | {s["te"] for s in segs})
    ok_n = []
    for n in range(lo_n, MAX_FRAMES + 1):
        tl = Fr(n - 1, FPS)
        if not any(tl - Fr(1, 2) < t < tl + END_AFTER for t in times):
            ok_n.append(n)
    if not ok_n:
        return None
    # a clip length under which every pulse is seen on its own for a while
    for n in rng.sample(ok_n, min(len(ok_n), 16)):
        scene["n_frames"] = n
        vis = visible(scene, segs)
        if _all_seen(scene, vis):
            break
    else:
        return None
    scene["segs"] = stored(vis)
    scene["ghosts"] = stored_ghosts(scene, vis)
    if not _layout(scene, rng):
        return None
    scene["labels"] = _labels(scene)
    return scene


def _all_seen(scene, segs) -> bool:
    tl = t_last(scene)
    return all(len(iso_frames(scene, segs, i)) >= ISO_MIN for i, s in enumerate(segs) if s["tb"] <= tl - Fr(1, 2))


def _need(scene, segs):
    """Time by which the clip's main events have happened."""
    home = scene["home"]
    if scene["scenario"] == "reflect":
        back = [s["te"] for s in segs if s["via_far"] and s["xe"] == end_x(scene, home) and s["origin"] == 0]
        return min(back) if back else None
    if scene["scenario"] == "train":            # every pulse has bounced off the far end once
        fx = end_x(scene, far_side(scene))
        firsts = [min((s["te"] for s in segs if s["origin"] == k and s["xe"] == fx), default=None)
                  for k in range(len(scene["launches"]))]
        return None if None in firsts else max(firsts)
    if scene["scenario"] == "junction":
        far = [s["te"] for s in segs if s["xe"] == end_x(scene, far_side(scene))]
        return min(far) if far else None
    xs = crossings(segs)
    if not xs:
        return None
    t = xs[0][0]
    for k in range(2):
        ln = scene["launches"][k]
        tgt = end_x(scene, "right" if ln["end"] == "left" else "left")
        arr = [s["te"] for s in segs if s["origin"] == k and s["xe"] == tgt]
        if not arr:
            return None
        t = max(t, min(arr))
    return t


# ---------------------------------------------------------------- checks

def _label_box(lb):
    w = 0.62 * lb["size"] * len(lb["text"])
    return (lb["x"] - w / 2, lb["y"] - 4, lb["x"] + w / 2, lb["y"] + lb["size"])


def check(scene) -> list[str]:
    errs = []
    segs = visible(scene, trace(scene))
    if stored(segs) != scene["segs"]:
        errs.append("stored segments don't match a re-trace")
    if stored_ghosts(scene, segs) != scene["ghosts"]:
        errs.append("stored faded parts don't match a re-trace")
    if scene["n_frames"] > MAX_FRAMES:
        errs.append("clip longer than 10 s")
    if scene["labels"] != _labels(scene):
        errs.append("labels don't match the ruler")
    tl = t_last(scene)
    S, tau = _support(scene)
    # sizes on screen
    if any(abs(Fr(s["amp"]).limit_denominator(100)) < CUTOFF for s in scene["segs"]):
        errs.append("a drawn pulse is weaker than the cutoff")
    if float(CUTOFF) * scene["amp_px"] < 15:
        errs.append("smallest pulse under 15 px")
    if min(sec[2] for sec in sections(scene)) * tau * scene["scale"] < 12:
        errs.append("narrowest pulse under 12 px")
    # layout: chain between the clock and the ruler, labels apart and below everything
    st = scene["style"]
    up, down = scene["reach"]
    r = st["bead_r"] * 1.6 + 2
    if scene["y0"] - scene["amp_px"] * up - r < TOP_LIMIT:
        errs.append("chain reaches the clock")
    if scene["y0"] + scene["amp_px"] * down + r > scene["ruler_y"] - 18:
        errs.append("chain reaches the ruler")
    if scene["x0"] < 112 or scene["x0"] + scene["L"] * scene["scale"] > W - 112:
        errs.append("chain too close to the frame edge")
    boxes = [_label_box(lb) for lb in scene["labels"]]
    for a, b in zip(boxes, boxes[1:]):
        if a[2] + 6 > b[0]:
            errs.append("ruler labels overlap")
            break
    if boxes and max(b[3] for b in boxes) > H - 8:
        errs.append("ruler labels leave the frame")
    # timing
    first = min(s["tb"] for s in segs if s["kind"] == "launch")
    if (first - S * tau) * FPS < EDGE:
        errs.append("first launch too close to the start")
    times = sorted({s["tb"] for s in segs} | {s["te"] for s in segs})
    if any(tl - Fr(1, 2) < t < tl + END_AFTER for t in times):
        errs.append("an event too close to the end")
    for x, ts in boundary_events(scene, segs).items():
        ts = [t for t in ts if t <= tl + END_AFTER]
        if any((b - a) * FPS < GAP for a, b in zip(ts, ts[1:])):
            errs.append(f"two events at mark {float(x):g} too close together")
            break
    # every pulse is seen on its own for a while
    if not _all_seen(scene, segs):
        errs.append("a pulse is never seen on its own")
    if scene["scenario"] == "collide":
        xs = [z for z in crossings(segs) if z[0] <= tl]
        for t, x, i, j in xs:
            if x < 1 or x > scene["L"] - 1:
                errs.append("two pulses cross too near a paddle")
                break
        if any((b[0] - a[0]) * FPS < GAP for a, b in zip(xs, xs[1:])):
            errs.append("two crossings too close together")
    return errs


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    # pulse count drawn once, so retries don't drift toward the easier short trains
    n_pulses = {"train": rng.choice([2, 3, 4]), "collide": rng.choice([2, 2, 3])}.get(scenario, 1)
    for attempt in range(1, 400):
        scene = _attempt(rng, scenario, n_pulses)
        if scene is None or check(scene) or not _askable(scene):
            continue
        scene["attempts"] = attempt
        return scene
    raise RuntimeError(f"no {scenario} scene")


def _askable(scene) -> bool:
    import questions
    P = questions.pools(questions.Facts(scene), random.Random(7))
    return any(all(len(P[lv]) >= k for lv, k in zip(range(1, 6), c))
               for c in questions.L._count_combos() if sum(c) == 10)
