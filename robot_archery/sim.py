"""Scene sampler and rules for robot_archery (after Jane Street's Robot Archery, Dec 2021).

Robots wait in a queue and take turns shooting one arrow each at a target
of ten rings (ring k = distance (k-1, k] from the centre, in ring units).
A shot strictly closer to the centre than every earlier shot is a new
record (a black-and-white dashed circle on the target moves to it). The
running record is the only thing that matters:

  classic     record -> the robot goes to the back of the queue;
              anything else -> the robot is eliminated (walks to the bench)
  two_lives   every robot starts with two life pips; a non-record shot
              costs a pip and the robot goes to the back of the queue,
              losing the last pip eliminates it
  stay_on     a record setter stays on the shooting spot and shoots again;
              anything else eliminates it
  big_field   classic rules with 5-7 robots and a faster pace

The last robot left wins; every game finishes inside the clip. Robots are
named by colour (unique per clip); each also has its own body shape and
pattern. Arrows stay where they land, drawn in the shooter's colour and
shape.

Timing: turn k starts at intro + k * T; the robot steps onto the spot, the
arrow flies and lands at intro + (k + 0.55) T (snapped 0.3-0.45 or
0.55-0.7 of the way through a frame interval). Landing is the only event:
the arrow, the record circle, a lost pip and the start of the walk to the
bench all appear on the first frame after it. Frame f is t = f / 24 and the
clock shows f * 10 // 24 tenths. Pure function of the rng passed in.
"""
from __future__ import annotations
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FPS = 24
SIZE = (1280, 720)
MAX_FRAMES = 240
EDGE_S = 0.6
MARGIN = 0.6                    # every shot is at least this far (ring units) from the record it is judged against
LAND = 0.55                     # landing, as a fraction of the turn
FRACS = ((0.3, 0.45), (0.55, 0.7))
SCENARIOS = ("classic", "two_lives", "stay_on", "big_field")
CLOCK_BOX = (18, 16, 214, 84)
LABEL_PX = 34
LABEL_RINGS = (2, 4, 6, 8, 10)
LABEL_ANG = 270.0               # labels sit on the ray straight down from the centre
NO_ARROW = (244.0, 296.0)       # arrows never land in this sector (keeps the labels clear)
MARK_R = 12                     # arrow marker radius (px)
SHAPES = ("circle", "square", "triangle", "diamond", "hexagon", "pentagon")
PATTERNS = ("plain", "stripes", "dots", "band")
PALETTE = {"red": (215, 40, 45), "orange": (240, 125, 20), "green": (35, 155, 65), "teal": (0, 150, 155),
           "blue": (40, 90, 225), "purple": (135, 60, 205), "pink": (230, 80, 170), "brown": (130, 82, 40)}
TARGETS = {
    "classic": [(250, 250, 250), (250, 250, 250), (60, 60, 64), (60, 60, 64), (90, 170, 230), (90, 170, 230),
                (235, 95, 85), (235, 95, 85), (250, 215, 80), (250, 215, 80)],
    "mono": [(245, 245, 245), (215, 215, 218), (245, 245, 245), (215, 215, 218), (245, 245, 245),
             (215, 215, 218), (245, 245, 245), (215, 215, 218), (245, 245, 245), (215, 215, 218)],
    "pastel": [(245, 240, 225), (225, 238, 225), (245, 240, 225), (225, 232, 245), (245, 240, 225),
               (245, 228, 232), (245, 240, 225), (232, 225, 245), (245, 240, 225), (250, 236, 200)],
    "sand": [(238, 226, 200), (226, 210, 178), (238, 226, 200), (226, 210, 178), (238, 226, 200),
             (226, 210, 178), (238, 226, 200), (226, 210, 178), (238, 226, 200), (226, 210, 178)],
}
BGS = [(232, 236, 228), (226, 232, 240), (238, 232, 226), (230, 230, 236), (222, 234, 230), (240, 236, 224)]
BG_PATTERNS = ("plain", "grass", "tiles", "gradient")


# ------------------------------------------------------------------ time
def duration(scene) -> float:
    return scene["frames"] / FPS


def t_last(scene) -> float:
    return (scene["frames"] - 1) / FPS


def clock_tenths(frame: int) -> int:
    return frame * 10 // FPS


def first_frame(t: float) -> int:
    return math.ceil(t * FPS)


def frac_ok(t: float, lo: float = 0.25) -> bool:
    x = t * FPS - math.floor(t * FPS)
    return lo <= x <= 1 - lo


def ring_of(d: float) -> int:
    return max(1, math.ceil(d - 1e-9))


# ------------------------------------------------------------------ rules
def play(order, draws, lives=1, stay_on=False) -> dict:
    """Replay a game: queue order (robot indices) and the distance of the shot on every turn.
    Stops when one robot is left or the draws run out (then 'winner' is None)."""
    queue = list(order)
    left = {r: lives for r in order}
    record, holder, current = None, None, None
    turns, eliminated = [], []
    for k, d in enumerate(draws):
        if len(queue) + (current is not None) <= 1:
            break
        if current is None:
            current = queue.pop(0)
        shooter = current
        is_rec = record is None or d < record
        t = {"turn": k, "shooter": shooter, "d": d, "record": is_rec, "prev_record": record,
             "prev_holder": holder}
        if is_rec:
            record, holder = d, shooter
            if not stay_on:
                queue.append(shooter)
                current = None
        else:
            left[shooter] -= 1
            current = None
            if left[shooter] == 0:
                eliminated.append(shooter)
            else:
                queue.append(shooter)
        t["lives_after"] = left[shooter]
        t["out"] = left[shooter] == 0
        turns.append(t)
    alive = queue + ([current] if current is not None else [])
    return {"turns": turns, "eliminated": eliminated, "winner": alive[0] if len(alive) == 1 else None}


def rules(scene):
    return {"lives": scene["lives"], "stay_on": scene["stay_on"]}


def land_time(scene, k) -> float:
    return scene["land"][k]


def turn_start(scene, k) -> float:
    return scene["intro"] + k * scene["T"]


# ------------------------------------------------------------------ layout
def target_xy(scene, d, ang):
    tg = scene["target"]
    a = math.radians(ang)
    return tg["cx"] + d * tg["unit"] * math.cos(a), tg["cy"] - d * tg["unit"] * math.sin(a)


def labels(scene) -> list[dict]:
    tg = scene["target"]
    a = math.radians(LABEL_ANG)
    return [{"text": str(k), "x": round(tg["cx"] + (k - 0.5) * tg["unit"] * math.cos(a), 2),
             "y": round(tg["cy"] - (k - 0.5) * tg["unit"] * math.sin(a), 2), "anchor": "mm", "size": LABEL_PX}
            for k in LABEL_RINGS]


def label_box(lab):
    from common.video5 import font
    l, t, r, b = font(lab["size"]).getbbox(lab["text"], anchor=lab["anchor"])
    return (lab["x"] + l, lab["y"] + t, lab["x"] + r, lab["y"] + b)


def _overlap(a, b, pad=0):
    return a[0] - pad < b[2] and b[0] - pad < a[2] and a[1] - pad < b[3] and b[1] - pad < a[3]


def _layout(rng, n, side, unit):
    """Target + robot area. Returns target dict and layout dict (spot, queue slots, bench slots)."""
    R = 10 * unit
    body = rng.randint(44, 54)
    if side == "left":
        cx = rng.uniform(R + 80, R + 125)
    else:
        cx = rng.uniform(SIZE[0] - R - 70, SIZE[0] - R - 30)
    cy = rng.uniform(max(R + 14, 330), min(SIZE[1] - R - 14, 390))
    tg = {"cx": round(cx, 2), "cy": round(cy, 2), "unit": unit, "R": R}
    # robot area: the other side of the target
    if side == "left":
        x_lo, x_hi, toward = cx + R + 40, SIZE[0] - 40, -1
    else:
        x_lo, x_hi, toward = 40, cx - R - 40, 1
    near = x_lo if side == "left" else x_hi
    far = x_hi if side == "left" else x_lo
    gap = rng.randint(body + 18, body + 34)
    queue_y = rng.uniform(560, 640)
    spot = [near + (-toward) * (body * 0.5 + 10), rng.uniform(cy - 40, cy + 60)]
    first = near + (-toward) * (body * 0.5 + 10 + rng.randint(0, 40))
    slots = [[round(first + (-toward) * gap * i, 2), round(queue_y, 2)] for i in range(n)]
    bench_y = rng.uniform(150, 230)
    bgap = rng.randint(body + 10, body + 24)
    bfirst = far + toward * (body * 0.5 + 10)
    bench = [[round(bfirst + toward * bgap * i, 2), round(bench_y, 2)] for i in range(n)]
    lay = {"spot": [round(spot[0], 2), round(spot[1], 2)], "slots": slots, "bench": bench, "body": body,
           "side": side}
    xs = [p[0] for p in slots + bench] + [spot[0]]
    if min(xs) - body < 20 or max(xs) + body > SIZE[0] - 20:
        return None
    return tg, lay


# ------------------------------------------------------------------ sampling
def _snap(rng, t):
    lo, hi = rng.choice(FRACS)
    return round((math.floor(t * FPS) + rng.uniform(lo, hi)) / FPS, 6)


def _draws(rng, order, lives, stay_on, sigma, max_turns, p_rec):
    """Shots drawn one at a time while the game is played; each is >= MARGIN from the record it faces."""
    draws = []
    for _ in range(max_turns):
        g = play(order, draws, lives, stay_on)
        if g["winner"] is not None:
            return draws
        rec = min(draws) if draws else None
        aim = rec is not None and rec - MARGIN > 0.8 and rng.random() < p_rec
        for _ in range(100):
            d = rng.uniform(0.35, rec - MARGIN) if aim else rng.weibullvariate(sigma * math.sqrt(2), 2.0)
            d = round(min(9.6, max(0.35, d)), 2)
            if rec is None or abs(d - rec) >= MARGIN:
                break
        else:
            return None
        draws.append(d)
    return draws if play(order, draws, lives, stay_on)["winner"] is not None else None


def _angles(rng, scene):
    pts, out = [], []
    for d in scene["draws"]:
        for _ in range(200):
            a = round(rng.uniform(0, 360), 2)
            if NO_ARROW[0] - 8 <= a <= NO_ARROW[1] + 8:
                continue
            p = target_xy(scene, d, a)
            box = (p[0] - MARK_R, p[1] - MARK_R, p[0] + MARK_R, p[1] + MARK_R)
            if all(math.dist(p, q) >= 2 * MARK_R + 6 for q in pts) and \
                    not any(_overlap(label_box(lab), box, 5) for lab in scene["labels"]):
                break
        else:
            return None
        pts.append(p)
        out.append(a)
    return out


def _draw(rng, scenario):
    lives = 2 if scenario == "two_lives" else 1
    stay_on = scenario == "stay_on"
    n = rng.randint(5, 7) if scenario == "big_field" else rng.randint(3, 4) if scenario == "two_lives" \
        else rng.randint(3, 6)
    T = rng.uniform(0.55, 0.68) if scenario == "big_field" else rng.uniform(0.6, 0.85)
    intro = rng.uniform(0.5, 0.9)
    outro = rng.uniform(0.7, 1.3)
    max_turns = int((MAX_FRAMES / FPS - 0.05 - intro - outro) / T - LAND + 1)
    names = rng.sample(sorted(PALETTE), n)
    shapes = rng.sample(SHAPES, n) if n <= len(SHAPES) and rng.random() < 0.5 else [rng.choice(SHAPES) for _ in range(n)]
    robots = [{"colour": names[i], "shape": shapes[i], "pattern": rng.choice(PATTERNS)} for i in range(n)]
    order = list(range(n))
    rng.shuffle(order)
    sigma = rng.uniform(2.0, 3.6)
    p_rec = rng.uniform(0.15, 0.45)
    draws = _draws(rng, order, lives, stay_on, sigma, max_turns, p_rec)
    if draws is None or len(draws) < n + 1:
        return None
    unit = rng.randint(28, 31)
    side = rng.choice(("left", "right"))
    lt = _layout(rng, n, side, unit)
    if lt is None:
        return None
    tg, lay = lt
    land = [_snap(rng, intro + (k + LAND) * T) for k in range(len(draws))]
    frames = math.ceil((land[-1] + outro) * FPS)
    if frames > MAX_FRAMES:
        return None
    scene = {"scenario": scenario, "frames": frames, "robots": robots, "order": order, "lives": lives,
             "stay_on": stay_on, "draws": draws, "intro": round(intro, 4), "T": round(T, 4), "land": land,
             "target": tg, "layout": lay,
             "style": {"bg": rng.choice(BGS), "bg_pat": rng.choice(BG_PATTERNS), "rings": rng.choice(sorted(TARGETS)),
                       "ink": (24, 24, 28), "pat_step": rng.randint(26, 48), "sigma": round(sigma, 3),
                       "p_rec": round(p_rec, 3)}}
    scene["labels"] = labels(scene)
    ang = _angles(rng, scene)
    if ang is None:
        return None
    scene["angles"] = ang
    g = play(order, draws, lives, stay_on)
    scene.update(turns=g["turns"], eliminated=g["eliminated"], winner=g["winner"])
    return scene


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    for attempt in range(1, 5000):
        scene = _draw(rng, scenario)
        if scene is None or check(scene) or not _ladder_ok(scene):
            continue
        scene["build_attempts"] = attempt
        return scene
    raise RuntimeError("no valid scene")


def _ladder_ok(scene):
    import questions                                   # late import: questions.py imports this module
    try:
        questions.build(scene, random.Random(0))
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------ consistency check
def robot_box(scene, xy):
    """A robot incl. legs and (two_lives) the pips over its head."""
    b = scene["layout"]["body"]
    top = b / 2 + (2 * 10 + 12 if scene["lives"] > 1 else 4)
    return (xy[0] - b / 2 - 4, xy[1] - top, xy[0] + b / 2 + 4, xy[1] + b / 2 + 8)


def check(scene: dict) -> list[str]:
    errs = []
    n = len(scene["robots"])
    g = play(scene["order"], scene["draws"], scene["lives"], scene["stay_on"])
    if g["turns"] != scene["turns"] or g["winner"] != scene["winner"] or g["eliminated"] != scene["eliminated"]:
        errs.append("stored game log doesn't match a replay of the draws")
    if g["winner"] is None or len(g["turns"]) != len(scene["draws"]):
        errs.append("game doesn't finish exactly on the last shot")
    if scene["scenario"] not in SCENARIOS:
        errs.append("unknown scenario")
    if len({r["colour"] for r in scene["robots"]}) != n:
        errs.append("two robots share a colour")
    if scene["frames"] > MAX_FRAMES:
        errs.append("clip longer than 10 s")
    tl = t_last(scene)
    land = scene["land"]
    for k, t in enumerate(land):
        if not EDGE_S <= t <= tl - EDGE_S:
            errs.append("a landing too near the start or the end")
        if not frac_ok(t):
            errs.append("a landing too near a frame instant")
        if abs(t - (scene["intro"] + (k + LAND) * scene["T"])) > 1 / FPS:
            errs.append("landing off its turn")
    if any(b - a < 0.5 for a, b in zip(land, land[1:])):
        errs.append("two landings too close")
    for t in g["turns"]:
        if t["prev_record"] is not None and abs(t["d"] - t["prev_record"]) < MARGIN - 1e-9:
            errs.append("a shot too close to the record it is judged against")
        if not 0.3 <= t["d"] <= 9.7:
            errs.append("shot off the target")
    if sum(t["record"] for t in g["turns"]) < 2:
        errs.append("fewer than two records")
    # drawing: arrows apart, labels clear of arrows, robots and the clock
    pts = [target_xy(scene, d, a) for d, a in zip(scene["draws"], scene["angles"])]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if math.dist(pts[i], pts[j]) < 2 * MARK_R + 6 - 1e-6:
                errs.append("two arrows overlap")
    labs = [label_box(lab) for lab in scene["labels"]]
    for k, lb in enumerate(labs):
        if lb[3] - lb[1] < 24:
            errs.append("label shorter than 24 px")
        if any(_overlap(lb, (p[0] - MARK_R, p[1] - MARK_R, p[0] + MARK_R, p[1] + MARK_R), 3) for p in pts):
            errs.append("an arrow covers a ring number")
        if any(_overlap(lb, o, 2) for o in labs[k + 1:]):
            errs.append("ring numbers overlap")
    tg, lay = scene["target"], scene["layout"]
    x0, y0, x1, y1 = CLOCK_BOX
    dx = max(x0 - tg["cx"], 0, tg["cx"] - x1)
    dy = max(y0 - tg["cy"], 0, tg["cy"] - y1)
    if math.hypot(dx, dy) < tg["R"] + 12:
        errs.append("target reaches the clock")
    boxes = [robot_box(scene, p) for p in lay["slots"] + lay["bench"] + [lay["spot"]]]
    for b in boxes:
        if _overlap(b, CLOCK_BOX, 8) or b[0] < 4 or b[2] > SIZE[0] - 4 or b[1] < 4 or b[3] > SIZE[1] - 4:
            errs.append("a robot position is off screen or under the clock")
        if math.dist(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), (tg["cx"], tg["cy"])) < tg["R"] + 40:
            errs.append("a robot position touches the target")
    return errs
