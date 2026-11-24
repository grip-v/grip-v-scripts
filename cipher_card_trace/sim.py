"""Scene generation for cipher_card_trace (v3, 10-question ladder).

A cipher wheel: a ring of the 26 letters and, inside it, a ring of the 26
card symbols (the 13 red ranks and the 13 black ranks). Playing cards are
dealt one at a time onto a pile (two piles in `two_piles`); every card
lands exactly on the one before, so only the top card of a pile is ever
visible. A card stands for the letter its colour + rank sits under at the
moment the card lands. Some events turn one ring by 1-5 positions, which
changes the key for every later card; nothing on screen shows a decoded
letter. Together the cards spell a random code of 4-6 distinct letters
(random letters, not words, so the answer cannot be guessed from English).

Scenarios (`scene["scenario"]`):
  clockwise    1-3 jokers; each turns the inner ring clockwise
  both_ways    2-3 jokers; each turns the inner ring clockwise or
               counter-clockwise (both directions occur)
  letter_ring  1-3 jokers; the OUTER letter ring turns instead, either way
  trigger      no jokers; 1-3 ordinary cards turn the inner ring clockwise
               a moment after they land (the card itself still uses the
               key in force when it landed)
  two_piles    1-2 jokers, inner ring clockwise; cards and jokers land on
               a left or a right pile; the code is read in dealing order

Offsets: with the inner ring at offset oi and the letter ring at oo, the
symbol at ring index j sits in slot (j + oi) mod 26 and letter i in slot
(i + oo) mod 26 (slot 0 at the top, clockwise). A card with symbol j
decodes to letter (j + oi - oo) mod 26. A clockwise turn by k adds k to the
offset of the ring that turns.

All randomness comes from the random.Random passed to build(); every time
is an integer frame at 24 fps.
"""
from __future__ import annotations
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FPS = 24
W, H = 1280, 720
MAX_FRAMES = 240
MIN_EDGE_F = 13            # >= 0.54 s between the clip edges and any event
ABC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
RANK_NAME = {"A": "ace", "J": "jack", "Q": "queen", "K": "king"}
SUITS = {"red": ["hearts", "diamonds"], "black": ["spades", "clubs"]}
SCENARIOS = ("clockwise", "both_ways", "letter_ring", "trigger", "two_piles")
MAX_TURN = 5
STEP_DEG = 360 / 26
TIMER_BOX = (16, 14, 196, 70)
TIMER_FONT = 36
LABEL_FONT = 34
RING_STYLES = ("blocks", "interleaved", "reversed", "shuffled")
ENTRIES_ONE = ("far", "top", "bottom", "far_top", "far_bottom")
ENTRIES_TWO = ("top", "bottom")

# felt backgrounds (all dark, so white cards and the light wheel stand out)
BACKS = [(30, 70, 52), (28, 44, 78), (78, 30, 40), (48, 54, 62), (70, 52, 36), (20, 70, 74), (58, 40, 70),
         (36, 36, 40)]
BG_PATTERNS = ("plain", "gradient", "dots", "weave")
# (outer a, outer b, inner a, inner b, hub)
WHEELS = [((246, 240, 222), (232, 222, 196), (255, 255, 255), (236, 236, 236), (60, 110, 86)),
          ((228, 238, 250), (206, 222, 242), (255, 255, 255), (238, 242, 248), (52, 74, 120)),
          ((236, 248, 230), (214, 236, 204), (255, 255, 250), (240, 246, 236), (70, 104, 60)),
          ((252, 234, 236), (240, 212, 216), (255, 255, 255), (246, 238, 238), (120, 60, 70)),
          ((252, 246, 214), (240, 228, 170), (255, 255, 252), (246, 244, 232), (130, 100, 40)),
          ((238, 232, 250), (220, 210, 242), (255, 255, 255), (242, 238, 250), (84, 64, 130))]
JOKERS = [("purple", (120, 40, 175)), ("green", (20, 140, 70)), ("orange", (230, 120, 0)),
          ("blue", (25, 95, 210)), ("teal", (0, 140, 150))]
STATS = {"tries": 0, "builds": 0}


# ---------------------------------------------------------------- names / clock

def card_name(c: dict) -> str:
    return f"the {RANK_NAME.get(c['rank'], c['rank'])} of {c['suit']}"


def symbol(c: dict) -> str:
    return f"{c['color']}:{c['rank']}"


def sym_name(s: str) -> str:
    color, rank = s.split(":")
    return f"the {color} {RANK_NAME.get(rank, rank)}"


def timer_tenths(i: int) -> int:
    """What the on-screen clock shows at frame i, in tenths of a second."""
    return (10 * i) // FPS


def clock_text(tenths: int) -> str:
    return f"{tenths // 10}.{tenths % 10} s"


def frames_showing(tenths: int):
    """(first, last) frame whose clock reads `tenths`."""
    lo = -(-tenths * FPS // 10)
    hi = -(-(tenths + 1) * FPS // 10) - 1
    return lo, hi


# ---------------------------------------------------------------- replay helpers (integer offsets)

def cards(scene) -> list:
    return [i for i, d in enumerate(scene["deal"]) if d["kind"] == "card"]


def jokers(scene) -> list:
    return [i for i, d in enumerate(scene["deal"]) if d["kind"] == "joker"]


def turners(scene) -> list:
    """Deal indices of the items that turn a ring, in order."""
    return [i for i, d in enumerate(scene["deal"]) if d["turn"]]


def rel_shift(scene, turn: int) -> int:
    """Change of (oi - oo) caused by a signed turn of the moving ring."""
    return turn if scene["moving"] == "inner" else -turn


def rels(scene, turns=None) -> list:
    """(oi - oo) mod 26 at the moment each dealt item lands. `turns`
    overrides the signed turn of each turner (in order)."""
    T = turners(scene)
    turns = [scene["deal"][i]["turn"] for i in T] if turns is None else list(turns)
    r = (scene["start"]["inner"] - scene["start"]["outer"]) % 26
    out = []
    for i, d in enumerate(scene["deal"]):
        out.append(r)
        if i in T:
            r = (r + rel_shift(scene, turns[T.index(i)])) % 26
    return out


def final_rel(scene, turns=None) -> int:
    T = turners(scene)
    turns = [scene["deal"][i]["turn"] for i in T] if turns is None else list(turns)
    r = (scene["start"]["inner"] - scene["start"]["outer"]) % 26
    return (r + sum(rel_shift(scene, t) for t in turns)) % 26


def letter_of(scene, i: int, rel: int) -> str:
    return ABC[(scene["ring"].index(symbol(scene["deal"][i])) + rel) % 26]


def decode(scene, turns=None) -> str:
    R = rels(scene, turns)
    return "".join(letter_of(scene, i, R[i]) for i in cards(scene))


def decode_const(scene, rel: int) -> str:
    return "".join(letter_of(scene, i, rel) for i in cards(scene))


def end_frame(d: dict) -> int:
    """Last event frame of a dealt item (its landing, or the end of its turn)."""
    return d["te"] if d["turn"] else d["land"]


def ring_rests(scene) -> list:
    """(first, last, rel) frame spans where neither ring moves."""
    out, a = [], 0
    R = rels(scene)
    for i in turners(scene):
        d = scene["deal"][i]
        out.append((a, d["ts"], R[i]))
        a = d["te"]
    out.append((a, scene["n_frames"] - 1, final_rel(scene)))
    return out


def next_same_pile(scene, i: int):
    """Deal index of the next item dealt onto the same pile (None if last)."""
    p = scene["deal"][i]["pile"]
    return next((j for j in range(i + 1, len(scene["deal"])) if scene["deal"][j]["pile"] == p), None)


def top_span(scene, i: int):
    """(land, last frame) during which item i lies on top of its pile, fully
    uncovered: until the next item for that pile starts sliding in."""
    d = scene["deal"][i]
    j = next_same_pile(scene, i)
    return d["land"], (scene["deal"][j]["s"] - 1 if j is not None else scene["n_frames"] - 1)


# ---------------------------------------------------------------- generation

def _ring(rng):
    red = [f"red:{r}" for r in RANKS]
    black = [f"black:{r}" for r in RANKS]
    style = rng.choice(RING_STYLES)
    if style == "blocks":
        ring = red + black if rng.random() < 0.5 else black + red
    elif style == "interleaved":
        ring = [s for pair in (zip(red, black) if rng.random() < 0.5 else zip(black, red)) for s in pair]
    elif style == "reversed":
        ring = red + black[::-1] if rng.random() < 0.5 else black + red[::-1]
    else:
        ring = red + black
        rng.shuffle(ring)
    return style, ring


def _structure(rng, sc):
    """Code, ring, start offsets and the dealt items (no timing yet)."""
    L = rng.randint(5 if sc == "trigger" else 4, 6)
    code = "".join(rng.sample(ABC, L))
    ring_style, ring = _ring(rng)
    start = {"inner": rng.randrange(26), "outer": rng.randrange(26)}
    moving = "outer" if sc == "letter_ring" else "inner"
    n_t = {"clockwise": rng.randint(1, 3), "both_ways": rng.randint(2, 3), "letter_ring": rng.randint(1, 3),
           "trigger": rng.randint(1, 3), "two_piles": rng.randint(1, 2)}[sc]
    n_t = min(n_t, L - 1)
    turns = [rng.randint(1, MAX_TURN) for _ in range(n_t)]
    if sc in ("both_ways", "letter_ring"):
        signs = [rng.choice((1, -1)) for _ in range(n_t)]
        if sc == "both_ways" and len(set(signs)) < 2:
            signs[rng.randrange(n_t)] *= -1
        turns = [t * s for t, s in zip(turns, signs)]
    if sc == "trigger":
        trig = sorted(rng.sample(range(L - 1), n_t))
        gaps = []
    else:
        trig = []
        gaps = sorted(rng.sample(range(1, L), n_t))      # a joker goes right before card gaps[k]
    deal, rel, used, k = [], (start["inner"] - start["outer"]) % 26, set(), 0
    shift = (lambda t: t) if moving == "inner" else (lambda t: -t)
    for ci, ch in enumerate(code):
        if k < len(gaps) and gaps[k] == ci:
            deal.append({"kind": "joker", "turn": turns[k]})
            rel = (rel + shift(turns[k])) % 26
            k += 1
        color, rank = ring[(ABC.index(ch) - rel) % 26].split(":")
        suits = [s for s in SUITS[color] if (rank, s) not in used]
        if not suits:
            return None
        suit = rng.choice(suits)
        used.add((rank, suit))
        t = 0
        if ci in trig:
            t = turns[trig.index(ci)]
            rel = (rel + shift(t)) % 26
        deal.append({"kind": "card", "color": color, "rank": rank, "suit": suit, "turn": t})
    for d in deal:
        d["pile"] = 0
    if sc == "two_piles":
        n = len(deal)
        for _ in range(20):
            piles = [rng.randint(0, 1) for _ in range(n)]
            if min(piles.count(0), piles.count(1)) >= 2:
                break
        else:
            return None
        for d, p in zip(deal, piles):
            d["pile"] = p
    return {"code": code, "ring_style": ring_style, "ring": ring, "start": start, "moving": moving, "deal": deal}


def _timing(rng, deal):
    """Frame schedule. Returns (timing dict, n_frames) or None if > 10 s."""
    tm = {"intro_f": rng.randint(MIN_EDGE_F, 22), "hold_f": rng.randint(MIN_EDGE_F, 22),
          "slide_f": rng.randint(7, 11), "step_f": rng.randint(3, 6)}
    rest = rng.randint(10, 17)
    pre, post = rng.randint(5, 8), rng.randint(5, 8)
    cur = tm["intro_f"]
    sched = []
    for d in deal:
        s = cur
        land = s + tm["slide_f"]
        if d["turn"]:
            ts = land + pre + rng.randint(0, 2)
            te = ts + abs(d["turn"]) * tm["step_f"]
            cur = te + post + rng.randint(0, 2)
        else:
            ts = te = None
            cur = land + max(9, rest + rng.randint(-2, 2))
        sched.append((s, land, ts, te))
    nf = end_frame({"turn": deal[-1]["turn"], "land": sched[-1][1], "te": sched[-1][3]}) + tm["hold_f"] + 1
    if nf > MAX_FRAMES:
        return None
    return tm, sched, nf


def _geometry(rng, sc, n_piles):
    for _ in range(200):
        side = rng.choice(("left", "right"))
        r3 = rng.randint(298, 336)
        r2 = r3 - rng.randint(62, 70)
        r1 = r2 - rng.randint(58, 66)
        mx = rng.randint(22, 80)
        cx = r3 + mx if side == "left" else W - r3 - mx
        cy = rng.randint(r3 + 8, H - r3 - 8)
        cw = rng.randint(148, 172 if n_piles == 2 else 192)
        ch = int(round(cw * 1.4))
        lo = (cx + r3 + 40) if side == "left" else 20
        hi = (W - 20) if side == "left" else (cx - r3 - 40)
        need = n_piles * cw + (n_piles - 1) * 40
        if hi - lo < need:
            continue
        piles = []
        if n_piles == 1:
            px = rng.randint(lo + cw // 2, hi - cw // 2)
            piles.append([px, rng.randint(ch // 2 + 40, H - ch // 2 - 40)])
        else:
            gap = rng.randint(40, max(40, min(160, hi - lo - 2 * cw)))
            x0 = rng.randint(lo + cw // 2, hi - cw - gap - cw // 2)
            y = rng.randint(ch // 2 + 60, H - ch // 2 - 60)
            dy = rng.randint(-40, 40)
            piles = [[x0, y], [x0 + cw + gap, max(ch // 2 + 40, min(H - ch // 2 - 40, y + dy))]]
        entry = rng.choice(ENTRIES_ONE if n_piles == 1 else ENTRIES_TWO)
        g = {"side": side, "cx": cx, "cy": cy, "r3": r3, "r2": r2, "r1": r1, "cw": cw, "ch": ch,
             "piles": piles, "entry": entry}
        if not geometry_errors(g):
            return g
    return None


def entry_point(g: dict, pile) -> tuple:
    """Off-screen point a card for this pile slides in from."""
    px, py = pile
    far_x = W + g["cw"] // 2 + 12 if g["side"] == "left" else -g["cw"] // 2 - 12
    top_y, bot_y = -g["ch"] // 2 - 12, H + g["ch"] // 2 + 12
    e = g["entry"]
    if e == "far":
        return far_x, py
    if e == "top":
        return px, top_y
    if e == "bottom":
        return px, bot_y
    if e == "far_top":
        return far_x, top_y
    return far_x, bot_y


def _style(rng):
    bg = rng.choice(BACKS)
    w = rng.choice(WHEELS)
    jn, jrgb = rng.choice(JOKERS)
    return {"bg": list(bg), "bg_pattern": rng.choice(BG_PATTERNS), "outer_a": list(w[0]), "outer_b": list(w[1]),
            "inner_a": list(w[2]), "inner_b": list(w[3]), "hub": list(w[4]), "joker": jn, "joker_rgb": list(jrgb),
            "shadow": rng.random() < 0.6, "pip_scale": round(rng.uniform(0.24, 0.32), 3),
            "card_edge": rng.choice([[150, 150, 150], [90, 90, 96], [170, 140, 90]]),
            "slider_tilt": 0}


def labels_for(scene_ring, g) -> list:
    """Ring letters and ranks. Each label sits at radius `r` on its ring and
    angle -90 + (index + offset of that ring) * 360/26 degrees; upright."""
    out = []
    ro, ri = (g["r2"] + g["r3"]) / 2, (g["r1"] + g["r2"]) / 2
    for i, ch in enumerate(ABC):
        out.append({"ring": "outer", "index": i, "text": ch, "ink": "dark", "r": ro, "size": LABEL_FONT})
    for j, s in enumerate(scene_ring):
        color, rank = s.split(":")
        out.append({"ring": "inner", "index": j, "text": rank, "ink": color, "r": ri, "size": LABEL_FONT})
    return out


def _try(rng, sc):
    STATS["tries"] += 1
    st = _structure(rng, sc)
    if st is None:
        return None
    for _ in range(30):
        t = _timing(rng, st["deal"])
        if t is not None:
            break
    else:
        return None
    tm, sched, nf = t
    for d, (s, land, ts, te) in zip(st["deal"], sched):
        d.update({"s": s, "land": land, "ts": ts, "te": te})
    g = _geometry(rng, sc, 2 if sc == "two_piles" else 1)
    if g is None:
        return None
    scene = {"scenario": sc, **st, "timing": tm, "n_frames": nf, "duration": round(nf / FPS, 6),
             "geometry": g, "style": _style(rng), "labels": labels_for(st["ring"], g),
             "timer": {"box": list(TIMER_BOX), "size": TIMER_FONT}}
    return scene


def build(rng: random.Random) -> dict:
    sc = rng.choice(SCENARIOS)
    STATS["builds"] += 1
    while True:
        s = _try(rng, sc)
        if s is not None and not check(s):
            return s


# ---------------------------------------------------------------- checks

def _rect_circle_dist(rect, cx, cy) -> float:
    x0, y0, x1, y1 = rect
    nx, ny = min(max(cx, x0), x1), min(max(cy, y0), y1)
    return math.hypot(cx - nx, cy - ny)


def _overlap(a, b, m=0) -> bool:
    return a[0] - m < b[2] and b[0] - m < a[2] and a[1] - m < b[3] and b[1] - m < a[3]


def card_rect(g, x, y):
    return (x - g["cw"] / 2, y - g["ch"] / 2, x + g["cw"] / 2, y + g["ch"] / 2)


def sweep_rect(g, pile):
    ex, ey = entry_point(g, pile)
    a, b = card_rect(g, ex, ey), card_rect(g, *pile)
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def label_box(lb, angle_deg, g, font_box) -> tuple:
    """Pixel box of a label centred on its ring at this angle."""
    x = g["cx"] + lb["r"] * math.cos(math.radians(angle_deg))
    y = g["cy"] + lb["r"] * math.sin(math.radians(angle_deg))
    w, h = font_box
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


_GLYPH = {}


def glyph_size(text: str, size: int):
    """Ink width/height of a label (anchor "mm" centres this box)."""
    key = (text, size)
    if key not in _GLYPH:
        from common.video5 import font
        x0, y0, x1, y1 = font(size).getbbox(text, anchor="mm")
        _GLYPH[key] = (x1 - x0, y1 - y0)
    return _GLYPH[key]


def geometry_errors(g) -> list:
    errs = []
    cx, cy, r3 = g["cx"], g["cy"], g["r3"]
    if cx - r3 < 8 or cx + r3 > W - 8 or cy - r3 < 8 or cy + r3 > H - 8:
        errs.append("wheel leaves the frame")
    if _rect_circle_dist(TIMER_BOX, cx, cy) < r3 + 8:
        errs.append("wheel overlaps the timer")
    rects = []
    for p in g["piles"]:
        r = card_rect(g, *p)
        if r[0] < 8 or r[1] < 8 or r[2] > W - 8 or r[3] > H - 8:
            errs.append("pile leaves the frame")
        sw = sweep_rect(g, p)
        if _rect_circle_dist(sw, cx, cy) < r3 + 12:
            errs.append("a card's slide passes over the wheel")
        if _overlap(sw, TIMER_BOX, 8):
            errs.append("a card's slide passes over the timer")
        rects.append((r, sw))
    if len(rects) == 2:
        (a, sa), (b, sb) = rects
        if _overlap(a, b, 20) or _overlap(sa, b, 6) or _overlap(sb, a, 6):
            errs.append("the piles overlap or a slide crosses the other pile")
    return errs


def label_errors(scene) -> list:
    """Labels: >= 24 px tall, inside their ring band, never touching a
    neighbour (checked at 8 sub-positions of a turn), the two rings' labels
    radially apart, and nothing near the timer or the cards (the wheel is
    clear of both)."""
    g = scene["geometry"]
    errs = []
    outer = [lb for lb in scene["labels"] if lb["ring"] == "outer"]
    inner = [lb for lb in scene["labels"] if lb["ring"] == "inner"]
    for group, (rlo, rhi) in ((outer, (g["r2"], g["r3"])), (inner, (g["r1"], g["r2"]))):
        sizes = [glyph_size(lb["text"], lb["size"]) for lb in group]
        if min(h for _, h in sizes) < 24:
            errs.append("a label is under 24 px tall")
        for lb, (w, h) in zip(group, sizes):
            half = math.hypot(w / 2, h / 2)
            if lb["r"] - half < rlo + 2 or lb["r"] + half > rhi - 2:
                errs.append(f"label {lb['text']} leaves its ring band")
        n = len(group)
        for f in range(8):
            boxes = [label_box(lb, -90 + (lb["index"] + f / 8) * STEP_DEG, g, sz) for lb, sz in zip(group, sizes)]
            for k in range(n):
                if _overlap(boxes[k], boxes[(k + 1) % n], 2):
                    errs.append("two ring labels overlap")
                    break
    return sorted(set(errs))


def scenario_errors(scene) -> list:
    sc, d = scene["scenario"], scene["deal"]
    errs = []
    T = turners(scene)
    J = jokers(scene)
    turns = [d[i]["turn"] for i in T]
    if sc not in SCENARIOS:
        return ["unknown scenario"]
    if (scene["moving"] == "outer") != (sc == "letter_ring"):
        errs.append("wrong moving ring for the scenario")
    if sc == "trigger":
        if J or not T or any(d[i]["kind"] != "card" for i in T):
            errs.append("trigger: turns must come from cards, no jokers")
    else:
        if not J or set(J) != set(T):
            errs.append("every joker (and only a joker) turns the ring")
    if sc in ("clockwise", "trigger", "two_piles") and any(t < 0 for t in turns):
        errs.append("counter-clockwise turn in a clockwise scenario")
    if sc == "both_ways" and not (any(t > 0 for t in turns) and any(t < 0 for t in turns)):
        errs.append("both_ways needs both directions")
    piles = [x["pile"] for x in d]
    if sc == "two_piles":
        if min(piles.count(0), piles.count(1)) < 2:
            errs.append("a pile gets fewer than 2 items")
    elif any(piles):
        errs.append("a second pile outside two_piles")
    return errs


def timing_errors(scene) -> list:
    errs = []
    d, nf, tm = scene["deal"], scene["n_frames"], scene["timing"]
    if nf > MAX_FRAMES:
        errs.append("clip longer than 10 s")
    ev = []
    for x in d:
        if x["land"] - x["s"] != tm["slide_f"]:
            errs.append("slide length off")
        ev += [x["s"], x["land"]]
        if x["turn"]:
            if x["ts"] - x["land"] < 5 or x["te"] - x["ts"] != abs(x["turn"]) * tm["step_f"]:
                errs.append("turn timing off")
            ev += [x["ts"], x["te"]]
    if ev != sorted(ev) or any(b - a < 2 for a, b in zip(ev, ev[1:])):
        errs.append("events closer than 2 frames or out of order")
    if ev[0] < MIN_EDGE_F or nf - 1 - ev[-1] < MIN_EDGE_F:
        errs.append("event within 0.5 s of the clip edge")
    for i in range(len(d)):
        a, b = top_span(scene, i)
        if b - a + 1 < 9:
            errs.append("a card is on top for under 9 frames")
    for a, b, _ in ring_rests(scene):
        if b - a < 6:
            errs.append("ring rest under 6 frames")
    return errs


def check(scene) -> list:
    errs = []
    d = scene["deal"]
    if decode(scene) != scene["code"]:
        errs.append("cards don't spell the stored code")
    if not 4 <= len(scene["code"]) <= 6 or len(set(scene["code"])) != len(scene["code"]):
        errs.append("code must be 4-6 distinct letters")
    if sorted(scene["ring"]) != sorted(f"{c}:{r}" for c in ("red", "black") for r in RANKS):
        errs.append("ring isn't the 26 card symbols")
    C = cards(scene)
    names = [card_name(d[i]) for i in C]
    if len(set(names)) != len(names):
        errs.append("two dealt cards look the same")
    if any(d[i]["suit"] not in SUITS[d[i]["color"]] for i in C):
        errs.append("suit colour mismatch")
    if d[0]["kind"] == "joker" or d[-1]["kind"] == "joker" or d[-1]["turn"] or any(
            a["kind"] == b["kind"] == "joker" for a, b in zip(d, d[1:])):
        errs.append("jokers must sit between two cards; the last item can't turn the ring")
    if any(not 1 <= abs(d[i]["turn"]) <= MAX_TURN for i in turners(scene)):
        errs.append("turn size out of range")
    errs += scenario_errors(scene)
    errs += timing_errors(scene)
    errs += geometry_errors(scene["geometry"])
    errs += label_errors(scene)
    # counterfactual decodes must differ from the real code
    T = turners(scene)
    turns = [d[i]["turn"] for i in T]
    if decode(scene, [0] * len(T)) == scene["code"]:
        errs.append("no-turn decode equals the code")
    for k in range(len(T)):
        if decode(scene, turns[:k] + [0] + turns[k + 1:]) == scene["code"]:
            errs.append("skipping a turn changes nothing")
    if abs(scene["duration"] - scene["n_frames"] / FPS) > 1e-6:
        errs.append("duration mismatch")
    return errs
