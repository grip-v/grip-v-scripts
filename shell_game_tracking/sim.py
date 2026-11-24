"""Shell game: identical cups stand on mats; at the start every cup is lifted
for a moment and the tokens under some of them are shown; then the cups are
lowered and moved around (the tokens travel hidden inside them). Only by
following every move can one tell where a token ends up.

Scenarios (`scene["scenario"]`):
  classic  3-6 cups in a row, one token, the cups swap in pairs
  multi    4-6 cups in a row, 2-3 different tokens, pair swaps
  rotate   4-6 cups in a row, one token; moves are pair swaps, three cups
           turning at once (each to the next of three positions), or two
           pair swaps at the same time
  grid     4 or 6 cups on two rows (2x2 / 2x3), one token, pair swaps in any
           direction (sideways, front-back, diagonal)
  peek     3-6 cups in a row, one token, pair swaps; between swaps a single
           cup is lifted now and then, showing what is (or isn't) under it

Cups are all drawn alike (one look per clip); questions name positions
("the second position from the left", "the top-right position") and tokens
by colour and shape. The only text on screen is the clock.

Motion (px, whole frames): a move occupies frames start .. start + dur - 1
with u = (f - start + 1) / dur, so a cup has already moved on the first
frame and arrives on the last. A cup going from position A to B first steps
out sideways (the left normal of B - A: a cup going right rises, one going
left drops towards the viewer) by h, slides across, and steps back in; h
clears every cup on the way, and two cups trading places pass each other
apart. Lifts (the start
reveal and peeks) raise one or all cups straight up.
"""
from __future__ import annotations
import functools
import json
import math
import random

import numpy as np

W, H, FPS = 1280, 720, 24
MAX_FRAMES = 240
EDGE = 12                 # frames between the clip's start / end and any question-relevant event
TIMER_BOX = (14, 14, 196, 70)
TIMER_FONT = 36
CLOCK_CLEAR = 14
X_MIN = TIMER_BOX[2] + CLOCK_CLEAR
CLEAR = 6                 # px between cup boxes at all times
MAX_STEP = 40             # px a cup may travel from one frame to the next
SCENARIOS = ("classic", "multi", "rotate", "grid", "peek")
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]

CUP_COLOURS = {"slate": (112, 128, 150), "brick": (170, 74, 54), "olive": (118, 124, 52), "navy": (40, 58, 120),
               "plum": (120, 60, 110), "sand": (196, 164, 110), "teal": (30, 120, 120), "charcoal": (70, 70, 76)}
TOKENS = {"red": (226, 40, 40), "yellow": (250, 210, 20), "green": (40, 190, 80), "blue": (40, 120, 255),
          "purple": (150, 60, 220), "orange": (255, 140, 20), "pink": (255, 110, 190), "cyan": (40, 220, 230)}
TOKEN_SHAPES = ("ball", "star", "cube", "diamond", "triangle")
CUP_SHAPES = ("cup", "dome", "box", "bell")
CUP_PATTERNS = ("plain", "bands", "dots", "stripe")
BACKGROUNDS = [((238, 234, 224), (220, 214, 200), True), ((228, 234, 240), (204, 214, 226), True),
               ((232, 238, 228), (212, 220, 206), True), ((26, 30, 42), (42, 48, 64), False),
               ((36, 36, 40), (58, 58, 64), False), ((22, 40, 46), (36, 60, 66), False),
               ((48, 30, 30), (70, 44, 40), False)]
TABLES = {True: [(196, 170, 130), (170, 180, 190), (180, 196, 170)], False: [(92, 70, 52), (70, 80, 92), (60, 84, 64)]}


def tenths(f: int) -> int:
    return (10 * f) // FPS


def clock_text(k: int) -> str:
    return f"{k // 10}.{k % 10} s"


def frames_showing(k: int) -> tuple[int, int]:
    return -(-k * FPS // 10), -(-(k + 1) * FPS // 10) - 1


def smooth(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------- naming

def slot_rows(sc):
    """Rows of slot indices, top (back) row first, each left to right."""
    if sc["scenario"] == "grid":
        c = sc["cols"]
        return [list(range(c)), list(range(c, 2 * c))]
    return [list(range(sc["n"]))]


def slot_name(sc, s) -> str:
    rows = slot_rows(sc)
    if len(rows) == 1:
        n = len(rows[0])
        if s == 0:
            return "the leftmost position"
        if s == n - 1:
            return "the rightmost position"
        if n == 3:
            return "the middle position"
        return f"the {ORD[s]} position from the left"
    c = sc["cols"]
    r, k = divmod(s, c)
    col = ["left", "right"] if c == 2 else ["left", "middle", "right"]
    return f"the {'top' if r == 0 else 'bottom'}-{col[k]} position"


def token_name(t) -> str:
    return f"the {t['colour']} {t['shape']}"


def move_word(sc) -> str:
    return "move" if sc["scenario"] == "rotate" else "swap"


# ---------------------------------------------------------------- the moves

def apply(perm_pairs, at):
    """at[slot] = cup id; returns the new list after the move (pairs = [[from, to], ...])."""
    new = list(at)
    for a, b in perm_pairs:
        new[b] = at[a]
    return new


def play(sc) -> dict:
    """cup id at every slot before / after each move; token slots; event frames."""
    n = len(sc["slots"])
    at = list(range(n))
    hist = [list(at)]
    for m in sc["moves"]:
        at = apply(m["perm"], at)
        hist.append(list(at))
    tok_cup = [t["slot"] for t in sc["tokens"]]                # cup id = starting slot
    tok_slots = [[h.index(c) for h in hist] for c in tok_cup]  # [token][k] slot after k moves
    return {"hist": hist, "tok_slots": tok_slots}


# ---------------------------------------------------------------- kinematics

def lift_at(sc, f):
    """{cup slot-at-time index: lift px} for the reveal and peeks (cups are at rest then)."""
    out = {}
    rv = sc["reveal"]
    if f < rv["hold"] + rv["down"]:
        u = 1.0 if f <= rv["hold"] else 1 - (f - rv["hold"]) / rv["down"]
        for s in range(len(sc["slots"])):
            out[s] = rv["lift"] * u
    for p in sc["peeks"]:
        a, up, hold, down = p["start"], p["up"], p["hold"], p["down"]
        if a <= f < a + up + hold + down:
            k = f - a
            u = (k + 1) / up if k < up else 1.0 if k < up + hold else 1 - (k - up - hold + 1) / down
            out[p["slot"]] = sc["peek_lift"] * max(0.0, min(1.0, u))
    return out


LIFT_PART = 0.28         # share of a move spent rising (and again settling) before / after the slide


def path(sc, a, b, u):
    """Position of a cup moving from slot a to slot b at time fraction u in (0, 1]: it first steps out
    sideways of the straight line (perp * h), slides across, then steps back in."""
    pa, pb = np.array(sc["slots"][a], float), np.array(sc["slots"][b], float)
    d = pb - pa
    L = float(np.hypot(*d))
    perp = np.array([d[1], -d[0]]) / L                                # left normal (screen y down)
    h = arc_h(sc, a, b)
    off = math.sin(math.pi / 2 * min(1.0, min(u, 1 - u) / LIFT_PART))   # visibly off the mat from the first frame
    x = smooth((u - LIFT_PART) / (1 - 2 * LIFT_PART))
    return pa + d * x + perp * h * off


def arc_h(sc, a, b):
    """Arc height for a cup going from slot a to slot b: clears every cup on the way."""
    pa, pb = np.array(sc["slots"][a], float), np.array(sc["slots"][b], float)
    d = pb - pa
    L = float(np.hypot(*d))
    cw, ch = sc["cup"]["w"], sc["cup"]["h"]
    need = 0.0
    for s, p in enumerate(sc["slots"]):
        if s in (a, b):
            continue
        q = np.array(p, float) - pa
        t = float(q @ d) / (L * L)
        if 0 < t < 1:
            off = abs(float(q[0] * d[1] - q[1] * d[0])) / L         # distance of that cup from the straight path
            span = max(cw, ch)
            if off < span + CLEAR:
                need = max(need, span + CLEAR + 2 - off)
    return max(need, sc["arc"] * (ch + CLEAR) / 2 + 4)


@functools.lru_cache(maxsize=16)
def _poses(key: str) -> np.ndarray:
    sc = json.loads(key)
    return _build_poses(sc)


def poses(sc) -> np.ndarray:
    """P[f, cup] = (x, y) of every cup's bottom centre (y includes any lift)."""
    return _poses(json.dumps(sc, sort_keys=True))


def _build_poses(sc):
    n = len(sc["slots"])
    nf = sc["n_frames"]
    S = np.array(sc["slots"], float)
    P = np.zeros((nf, n, 2))
    at = list(range(n))
    seq = sorted(sc["moves"], key=lambda m: m["start"])
    k = 0
    for f in range(nf):
        while k < len(seq) and f >= seq[k]["start"] + seq[k]["dur"]:
            at = apply(seq[k]["perm"], at)
            k += 1
        pos = {c: S[s].copy() for s, c in enumerate(at)}
        if k < len(seq) and f >= seq[k]["start"]:
            m = seq[k]
            u = (f - m["start"] + 1) / m["dur"]
            for a, b in m["perm"]:
                pos[at[a]] = path(sc, a, b, u)
        for s_, lf in lift_at(sc, f).items():
            pos[at[s_]] = pos[at[s_]] + np.array([0.0, -lf])
        for c in range(n):
            P[f, c] = pos[c]
    return P


def token_visible(sc, f):
    """[(token index, x, y_bottom)] drawn on frame f: a token shows while its cup is lifted high enough."""
    pl = play(sc)
    k = sum(1 for m in sc["moves"] if f >= m["start"] + m["dur"])
    out = []
    lifts = lift_at(sc, f)
    for i, t in enumerate(sc["tokens"]):
        s = pl["tok_slots"][i][k]
        if lifts.get(s, 0) >= sc["tok_size"] + 8:
            x, y = sc["slots"][s]
            out.append((i, x, y))
    return out


# ---------------------------------------------------------------- generation

REJECTS: dict = {}
ATTEMPTS: list = []


def _reject(why):
    REJECTS[why] = REJECTS.get(why, 0) + 1
    return None


def _layout(rng, sc):
    grid = sc["scenario"] == "grid"
    cw = rng.randint(70, 100) if grid else rng.randint(78, 132)
    ch = int(round(cw * rng.uniform(0.85, 1.2)))
    sc["cup"] = {"w": cw, "h": ch}
    if grid:
        cols = sc["cols"]
        pitch = int(cw * rng.uniform(2.2, 2.6))
        q = int(max((ch + CLEAR) * rng.uniform(1.6, 1.9), ch + cw + 56))      # a lifted front cup clears the back tokens
        h = sc["arc"] * (ch + CLEAR) / 2 + 4
        width = (cols - 1) * pitch
        x0 = rng.uniform(X_MIN + cw / 2 + 20, W - 30 - cw / 2 - width)
        lo, hi = ch + q + h + 16, H - h - 16
        if lo > hi:
            return False
        y1 = rng.uniform(lo, hi)
        sc["slots"] = [[round(x0 + c * pitch, 1), round(y1 - q, 1)] for c in range(cols)] + \
                      [[round(x0 + c * pitch, 1), round(y1, 1)] for c in range(cols)]
    else:
        n = sc["n"]
        pitch_max = (W - 30 - X_MIN - cw - 40) / max(1, n - 1)
        pitch_min = cw + 34
        if pitch_min > pitch_max:
            return False
        pitch = rng.uniform(pitch_min, min(pitch_max, cw * 2.2))
        width = (n - 1) * pitch
        x0 = rng.uniform(X_MIN + cw / 2 + 20, W - 30 - cw / 2 - width)
        y = rng.uniform(H * 0.5, H * 0.62)
        sc["slots"] = [[round(x0 + i * pitch, 1), round(y, 1)] for i in range(n)]
    return True


def _pick_move(rng, sc):
    n = len(sc["slots"])
    s = sc["scenario"]
    if s == "grid":
        a, b = rng.sample(range(n), 2)
        return [[a, b], [b, a]]
    kinds = ["swap"]
    if s == "rotate":
        kinds = ["swap", "turn", "turn", "double"] if n >= 4 else ["swap", "turn"]
    kind = rng.choice(kinds)
    if kind == "swap":
        a = rng.randrange(n)
        b = rng.choice([x for x in range(n) if x != a and abs(x - a) <= 3])
        return [[a, b], [b, a]]
    if kind == "turn":
        trip = sorted(rng.sample(range(n), 3))
        if trip[2] - trip[0] > 3:
            return None
        a, b, c = trip if rng.random() < 0.5 else trip[::-1]
        return [[a, b], [b, c], [c, a]]
    a, b, c, d = rng.sample(range(n), 4)
    if abs(a - b) > 3 or abs(c - d) > 3:
        return None
    return [[a, b], [b, a], [c, d], [d, c]]


def _attempt(rng, s):
    sc = {"scenario": s, "labels": []}
    if s == "grid":
        sc["cols"] = rng.choice([2, 3])
        sc["n"] = 2 * sc["cols"]
    else:
        sc["n"] = rng.randint(4, 6) if s in ("multi", "rotate") else rng.randint(3, 6)
    sc["arc"] = rng.choice([1.0, 1.15, 1.3])
    if not _layout(rng, sc):
        return _reject("layout")
    n = sc["n"]
    nt = rng.choice([2, 2, 3]) if s == "multi" else 1
    nt = min(nt, n - 1)
    cols = rng.sample(list(TOKENS), nt)
    shapes = [rng.choice(TOKEN_SHAPES) for _ in range(nt)]
    slots = rng.sample(range(n), nt)
    sc["tokens"] = [{"colour": c, "shape": sh, "slot": sl} for c, sh, sl in zip(cols, shapes, slots)]
    sc["tok_size"] = int(min(sc["cup"]["w"] * 0.5, rng.randint(32, 50)))
    sc["peek_lift"] = sc["tok_size"] + rng.randint(18, 34)
    hold = rng.randint(16, 26)
    sc["reveal"] = {"hold": hold, "down": rng.randint(5, 8), "lift": sc["tok_size"] + rng.randint(20, 40)}
    base = rng.randint(12, 17)
    jit = rng.randint(0, 3)
    gap_lo = rng.randint(6, 8)
    t = hold + sc["reveal"]["down"] + rng.randint(6, 12)
    n_moves = rng.randint(3, 8)
    n_peeks = rng.randint(1, 3) if s == "peek" else 0
    items = ["move"] * n_moves
    for _ in range(n_peeks):
        items.insert(rng.randint(1, len(items)), "peek")
    tail = rng.randint(EDGE + 2, EDGE + 14)
    sc["moves"], sc["peeks"] = [], []
    for it in items:
        if it == "peek":
            p = {"start": t, "up": 4, "hold": rng.randint(10, 14), "down": 4, "slot": None}
            dur = p["up"] + p["hold"] + p["down"]
            if t + dur + tail > MAX_FRAMES:
                break
            sc["peeks"].append(p)
            t += dur + rng.randint(gap_lo, gap_lo + 8)
            continue
        dur = base + rng.randint(0, jit) + (2 if s in ("grid", "rotate") else 0)
        if t + dur + tail > MAX_FRAMES:
            break
        for _ in range(30):
            perm = _pick_move(rng, sc)
            if perm and not _collides(sc, perm):
                break
        else:
            return _reject("no clear move")
        while _max_step(sc, perm, dur) > MAX_STEP:                  # longer moves take longer
            dur += 1
        if t + dur + tail > MAX_FRAMES:
            break
        sc["moves"].append({"start": t, "dur": dur, "perm": perm})
        t += dur + rng.randint(gap_lo, gap_lo + 8)
    if len(sc["moves"]) < 3 or (s == "peek" and not sc["peeks"]):
        return _reject("too long")
    # peeks: which cup is lifted (about half the time the token's cup)
    pl = play(sc)
    for p in sc["peeks"]:
        k = sum(1 for m in sc["moves"] if m["start"] + m["dur"] <= p["start"])
        ts = pl["tok_slots"][0][k]
        p["slot"] = ts if rng.random() < 0.5 else rng.choice([x for x in range(n) if x != ts])
    last = max([m["start"] + m["dur"] for m in sc["moves"]] + [p["start"] + p["up"] + p["hold"] + p["down"]
                                                                for p in sc["peeks"]])
    sc["n_frames"] = last - 1 + tail
    return sc


def _max_step(sc, perm, dur):
    best = 0.0
    for a, b in perm:
        prev = np.array(sc["slots"][a], float)
        for k in range(1, dur + 1):
            p = path(sc, a, b, k / dur)
            best = max(best, float(np.hypot(*(p - prev))))
            prev = p
    return best


def _collides(sc, perm):
    """True if moving the cups along this move would bring two cup boxes closer than CLEAR."""
    n = len(sc["slots"])
    S = np.array(sc["slots"], float)
    moving = {a: b for a, b in perm}
    cw, ch = sc["cup"]["w"], sc["cup"]["h"]
    for k in range(1, 40):
        pos = [path(sc, i, moving[i], k / 40) if i in moving else S[i] for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if abs(pos[i][0] - pos[j][0]) < cw + CLEAR and abs(pos[i][1] - pos[j][1]) < ch + CLEAR:
                    return True
        for p in pos:
            if not (X_MIN + cw / 2 <= p[0] <= W - 8 - cw / 2 and ch + 12 <= p[1] <= H - 10):
                return True
    return False


def lum(rgb) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def cdist(a, b) -> float:
    return float(sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5)


def _style(rng, sc):
    """Background, table and cup colours with clear contrast; tokens stand out from table and cups."""
    for _ in range(200):
        top, bot, light = rng.choice(BACKGROUNDS)
        table = rng.choice(TABLES[light])
        cup = rng.choice(list(CUP_COLOURS))
        crgb = CUP_COLOURS[cup]
        if abs(lum(crgb) - lum(table)) < 40 or abs(lum(crgb) - lum(top)) < 40:
            continue
        if any(min(cdist(TOKENS[t["colour"]], c) for c in (table, crgb, top, bot)) < 110 for t in sc["tokens"]):
            continue
        break
    else:
        return None
    style = rng.choice(["flat", "gradient", "vignette"])
    if style == "flat":
        bot = top
    return {"bg": [list(top), list(bot)], "bg_style": style, "light": light,
            "cup_colour": cup, "cup_shape": rng.choice(CUP_SHAPES),
            "cup_pattern": rng.choice(CUP_PATTERNS), "table": list(table),
            "mat": rng.choice(["ellipse", "square", "ring"]), "outline": rng.choice([3, 4])}


def build(rng: random.Random) -> dict:
    s = rng.choice(SCENARIOS)
    for k in range(2000):
        sc = _attempt(rng, s)
        if sc is None:
            continue
        sc["style"] = _style(rng, sc)
        if sc["style"] is None:
            _reject("colours")
            continue
        sc = json.loads(json.dumps(sc))
        errs = check(sc)
        if errs:
            _reject("check: " + errs[0].split(":")[0])
            continue
        ATTEMPTS.append(k + 1)
        return sc
    raise RuntimeError(f"no {s} scene")


# ---------------------------------------------------------------- checks

def events(sc) -> list:
    ev = [("reveal_end", sc["reveal"]["hold"] + sc["reveal"]["down"])]
    for m in sc["moves"]:
        ev += [("move_start", m["start"]), ("move_end", m["start"] + m["dur"] - 1)]
    for p in sc["peeks"]:
        ev += [("peek_start", p["start"]), ("peek_end", p["start"] + p["up"] + p["hold"] + p["down"] - 1)]
    return sorted(ev, key=lambda e: e[1])


def check(sc) -> list[str]:
    errs = []
    n = sc["n_frames"]
    if not 1 <= n <= MAX_FRAMES:
        errs.append(f"length: {n} frames")
    ev = events(sc)
    for kind, f in ev:
        if not EDGE <= f <= n - 1 - EDGE:
            errs.append(f"edge: {kind} at frame {f}")
    fs = [f for _, f in ev]
    for a, b in zip(fs, fs[1:]):
        if b - a < 2:
            errs.append(f"spacing: events at {a} and {b}")
    for m in sc["moves"]:
        if m["dur"] < 12:
            errs.append("speed: a move shorter than 12 frames")
    looks = [(t["colour"], t["shape"]) for t in sc["tokens"]]
    if len(set(t["colour"] for t in sc["tokens"])) != len(looks):
        errs.append("tokens: two tokens share a colour")
    if sc["cup"]["w"] < 20 or sc["tok_size"] < 20:
        errs.append("size: cup or token under 20 px")
    P = poses(sc)
    cw, ch = sc["cup"]["w"], sc["cup"]["h"]
    ts = sc["tok_size"]
    for f in range(n):
        for i, x, y in token_visible(sc, f):
            for cx, cy in P[f]:
                if abs(cx - x) < (cw + ts) / 2 + 2 and y - 4 - ts - 2 < cy and cy - ch < y + 2:
                    errs.append(f"token: token {i} covered by a cup on frame {f}")
                    break
    step = np.hypot(*(P[1:] - P[:-1]).transpose(2, 0, 1)).max() if n > 1 else 0
    if step > MAX_STEP + 1e-6:
        errs.append(f"speed: a cup moves {step:.0f} px in one frame")
    for f in range(n):
        X, Y = P[f, :, 0], P[f, :, 1]
        dx = np.abs(X[:, None] - X[None, :])
        dy = np.abs(Y[:, None] - Y[None, :])
        bad = (dx < cw + CLEAR) & (dy < ch + CLEAR)
        np.fill_diagonal(bad, False)
        if bad.any():
            errs.append(f"overlap: cups too close on frame {f}")
            break
        if (X - cw / 2 < X_MIN).any() or (X + cw / 2 > W - 4).any() or (Y - ch < 4).any() or (Y > H - 4).any():
            errs.append(f"frame: a cup leaves the frame or nears the clock on frame {f}")
            break
    return errs
