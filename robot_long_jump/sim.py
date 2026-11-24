"""Scene generation for robot_long_jump (after Jane Street, Mar 2023).

Each robot hops forward along its own lane in random steps. As soon as it
stands at or past its own hidden take-off threshold it makes one long jump
into the sand pit, and its score is take-off position + jump length (= where
it lands). A hop that carries it past the red foul line at 1.0 before it
has jumped is a foul: the robot topples over and scores 0.

All positions are integers in tenths of a track unit (the ruler shows
0.0 .. 2.0, gridlines every 0.1), so every footprint, take-off spot and
landing spot sits exactly on a gridline. Every hop leaves a footprint that
stays for the rest of the clip; a landing leaves a dent in the sand and the
robot stays standing on it.

Scenarios (scene["scenario"]):
  turns      robots go one after another, random hop lengths 0.1-0.4
  together   all robots hop at the same time on a shared beat, random hops
  staggered  robots go one after another from their own start blocks (0.0-0.3)
  steady     all robots hop together on a beat, each with one fixed hop length

Timing is in integer frames (24 fps). The on-screen timer shows
floor(t * 10) / 10 s. In the turns / staggered scenarios every sand landing
is put on a frame whose exact time is within 0.045 s above a tenth, so the
timer reading and the true time agree.

All randomness comes from the random.Random passed to build(); the scene is
plain JSON (ints, short lists, a few rounded floats).
"""
from __future__ import annotations
import random

FPS = 24
MAX_FRAMES = 240                     # 10 s
EDGE_F = 12                          # question events stay >= 0.5 s from both ends
FOUL = 10                            # foul line at 1.0 (tenths)
TRACK_END = 20                       # ruler / pit run to 2.0
SCENARIOS = ("turns", "together", "staggered", "steady")
COLOURS = ("orange", "yellow", "green", "teal", "blue", "purple", "pink")
SHAPES = ("circle", "square", "triangle", "hexagon", "diamond")
PATTERNS = ("plain", "striped", "ringed", "two-tone")
SEQUENTIAL = ("turns", "staggered")

THEMES = (  # background, runway, sand
    ((238, 234, 224), (150, 86, 72), (234, 212, 160)),
    ((226, 232, 238), (104, 112, 126), (230, 206, 152)),
    ((230, 238, 228), (88, 90, 98), (238, 220, 176)),
    ((240, 232, 236), (128, 98, 88), (226, 200, 146)),
    ((232, 230, 242), (96, 104, 118), (236, 214, 166)),
    ((244, 240, 230), (140, 80, 66), (222, 196, 140)),
)


def aligned(f: int) -> bool:
    """Frame whose exact time is within 0.045 s above a tenth (timer == true time to 0.1 s)."""
    return (f * 10) % FPS <= 10


def timer_tenths(f: int) -> int:
    return (f * 10) // FPS


def display(f: int) -> str:
    t = timer_tenths(f)
    return f"{t // 10}.{t % 10} s"


def fmt(tenths: int) -> str:
    return f"{tenths // 10}.{tenths % 10}"


# --------------------------------------------------------------------------- the rule


def run(tau: int, start: int, hops: list[int], foul: int = FOUL) -> dict:
    """Play one attempt over pre-drawn hops: stop at the first hop that lands
    past the foul line (foul) or at / past tau (jump). Positions in tenths."""
    pos, used, feet_ = start, [], []
    for h in hops:
        pos += h
        used.append(h)
        feet_.append(pos)
        if pos > foul:
            return {"hops": used, "feet": feet_, "fouled": True, "takeoff": None}
        if pos >= tau:
            return {"hops": used, "feet": feet_, "fouled": False, "takeoff": pos}
    return {"hops": used, "feet": feet_, "fouled": None, "takeoff": None}   # ran out of hops


def _draw_hops(rng, tau, start, fixed):
    """Draw hops one by one; a hop never lands exactly on the foul line."""
    pos, hops = start, []
    while True:
        if fixed:
            if pos + fixed == FOUL:
                return None
            h = fixed
        else:
            h = rng.choice([s for s in (1, 2, 3, 4) if pos + s != FOUL])
        hops.append(h)
        pos += h
        if pos > FOUL or pos >= tau:
            return hops


# --------------------------------------------------------------------------- build


def _robots(rng, scen, n):
    cols = rng.sample(COLOURS, n)
    lanes = list(range(n))
    rng.shuffle(lanes)
    starts = [0] * n
    if scen == "staggered":
        while len(set(starts)) < 2:
            starts = [rng.randint(0, 3) for _ in range(n)]
    robots = []
    for i in range(n):
        tau = rng.randint(5, 9)
        fixed = rng.randint(1, 4) if scen == "steady" else None
        hops = _draw_hops(rng, tau, starts[i], fixed)
        if hops is None:
            return None
        att = run(tau, starts[i], hops)
        jump = None
        if not att["fouled"]:
            lo, hi = max(11 - att["takeoff"], 2), min(19 - att["takeoff"], 10)
            jump = rng.randint(lo, hi)
        robots.append({"id": i, "colour": cols[i], "shape": rng.choice(SHAPES), "pattern": rng.choice(PATTERNS),
                       "size": rng.randint(22, 28), "lane": lanes[i], "tau": tau, "start": starts[i],
                       "hops": att["hops"], "fouled": att["fouled"], "jump": jump})
    return robots


def _schedule(rng, scen, robots):
    """Per-robot frames: hop start frames, jump start frame, landing / foul frame."""
    tm = {"hop_f": rng.randint(5, 7), "rest_f": rng.randint(2, 4), "jump_f": rng.randint(12, 16),
          "tip_f": 6, "gap_f": rng.randint(3, 6)}
    intro = rng.randint(EDGE_F, 20)
    if scen in SEQUENTIAL:
        order = list(range(len(robots)))
        rng.shuffle(order)
        t = intro
        for k, i in enumerate(order):
            r = robots[i]
            r["order"] = k
            r["hop_frames"] = []
            for _ in r["hops"]:
                r["hop_frames"].append(t)
                t += tm["hop_f"] + tm["rest_f"]
            if r["fouled"]:
                r["jump_frame"] = None
                r["end_frame"] = r["hop_frames"][-1] + tm["hop_f"]          # lands past the line
                t = r["end_frame"] + tm["tip_f"] + tm["gap_f"]
            else:
                while not aligned(t + tm["jump_f"]):
                    t += 1
                r["jump_frame"] = t
                r["end_frame"] = t + tm["jump_f"]                          # lands in the sand
                t = r["end_frame"] + tm["gap_f"]
    else:
        tm["rest_f"] = max(tm["rest_f"], 9 - tm["hop_f"])                  # beat >= 9 frames
        beat = tm["hop_f"] + tm["rest_f"]
        for r in robots:
            r["order"] = None
            r["hop_frames"] = [intro + k * beat for k in range(len(r["hops"]))]
            last = r["hop_frames"][-1]
            if r["fouled"]:
                r["jump_frame"] = None
                r["end_frame"] = last + tm["hop_f"]
            else:
                r["jump_frame"] = last + beat
                r["end_frame"] = r["jump_frame"] + tm["jump_f"]
    done = max(r["end_frame"] + (tm["tip_f"] if r["fouled"] else 0) for r in robots)
    return tm, done + rng.randint(EDGE_F, 22)


def _view(rng, n):
    x0 = rng.randint(150, 186)
    ppu = rng.randint(500, (1250 - x0) // 2)
    top, bot = 196, 600
    pitch = min(150, (bot - top) // (n - 1))
    mid = (top + bot) // 2 + rng.randint(-12, 12)
    lane_y = [round(mid + (k - (n - 1) / 2) * pitch) for k in range(n)]
    return {"x0": x0, "ppu": ppu, "pitch": pitch, "lane_y": lane_y, "strip_h": rng.randint(14, 18),
            "ruler_y": 648}


def _labels(view):
    """The only text besides the timer: ruler numbers at 0.0, 0.5, ... 2.0."""
    return [{"text": fmt(v), "x": view["x0"] + view["ppu"] * v // 10, "y": view["ruler_y"] + 40, "size": 34}
            for v in range(0, TRACK_END + 1, 5)]


def build(rng: random.Random) -> dict:
    scen = rng.choice(SCENARIOS)            # drawn once, so retries don't skew the scenario mix
    tries = 0
    while True:
        tries += 1
        n = rng.choice((3, 3, 4)) if scen in SEQUENTIAL else rng.choice((3, 4, 4, 5))
        robots = _robots(rng, scen, n)
        if robots is None:
            continue
        tm, n_frames = _schedule(rng, scen, robots)
        bg, runway, sand = rng.choice(THEMES)
        jit = lambda c: [max(0, min(255, v + rng.randint(-8, 8))) for v in c]
        view = _view(rng, n)
        scene = {"scenario": scen, "n_frames": n_frames, "foul": FOUL, "timing": tm,
                 # a jumping robot's body stays below the lane above (tallest body: 10 px legs + 28 + 31 px)
                 "jump_h": min(round(view["pitch"] * rng.uniform(0.42, 0.55)), view["pitch"] - 78),
                 "theme": {"bg": jit(bg), "runway": jit(runway), "sand": jit(sand),
                           "foot": rng.choice(("dot", "bar")), "grid_alpha": round(rng.uniform(0.3, 0.42), 2),
                           "eye": rng.choice(("round", "visor"))},
                 "view": view, "robots": robots, "labels": _labels(view),
                 "timer": {"box": [16, 14, 196, 66], "size": 36}, "tries": tries}
        scene["hop_h"] = min(round(view["pitch"] * rng.uniform(0.18, 0.26)), round(0.6 * scene["jump_h"]))
        if not check(scene):
            return scene


# --------------------------------------------------------------------------- derived facts (generator side)


def feet(r) -> list[int]:
    out, pos = [], r["start"]
    for h in r["hops"]:
        pos += h
        out.append(pos)
    return out


def takeoff(r):
    return None if r["fouled"] else feet(r)[-1]


def score(r) -> int:
    return 0 if r["fouled"] else takeoff(r) + r["jump"]


def landing_frames(scene, wait=None) -> dict:
    """{robot id: frame it lands in the sand} for the robots that jump. `wait`
    ({id: beats}, hop-together scenarios): that robot stands still for extra
    beats before its take-off (counterfactual)."""
    tm = scene["timing"]
    beat = tm["hop_f"] + tm["rest_f"]
    return {r["id"]: r["end_frame"] + beat * (wait or {}).get(r["id"], 0) for r in scene["robots"] if not r["fouled"]}


def leaders(scene, drop=None) -> list[tuple[int, int]]:
    """[(frame, leader id)] after each sand landing, in time order. `drop`: a
    robot whose landing is taken away (counterfactual: it scores 0)."""
    lands = sorted((f, i) for i, f in landing_frames(scene).items() if i != drop)
    sc = {r["id"]: score(r) for r in scene["robots"]}
    best, out = None, []
    for f, i in lands:
        if best is None or sc[i] > sc[best]:
            best = i
        out.append((f, best))
    return out


def lead_changes(scene, drop=None) -> int:
    seq = [i for _, i in leaders(scene, drop)]
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def lands_spread(scene, ids=None) -> int:
    """Smallest gap (frames) between the sand landings of the given robots."""
    fr = sorted(f for i, f in landing_frames(scene).items() if ids is None or i in ids)
    return min((b - a for a, b in zip(fr, fr[1:])), default=10 ** 6)


# --------------------------------------------------------------------------- checks


def check(scene: dict) -> list[str]:
    errs = []
    rs = scene["robots"]
    n = scene["n_frames"]
    tip = scene["timing"]["tip_f"]
    if n > MAX_FRAMES:
        errs.append(f"clip is {n} frames, over {MAX_FRAMES}")
    if len({r["colour"] for r in rs}) != len(rs):
        errs.append("robot colours repeat")
    if sorted(r["lane"] for r in rs) != list(range(len(rs))):
        errs.append("lanes are not a permutation")
    for r in rs:
        att = run(r["tau"], r["start"], r["hops"])
        if att["hops"] != r["hops"] or att["fouled"] != r["fouled"]:
            errs.append(f"robot {r['id']}: hops don't follow the take-off / foul rule")
        if FOUL in feet(r):
            errs.append(f"robot {r['id']}: a footprint sits on the foul line")
        if any(h < 1 or h > 4 for h in r["hops"]):
            errs.append(f"robot {r['id']}: hop length out of range")
        if scene["scenario"] == "steady" and len(set(r["hops"])) != 1:
            errs.append(f"robot {r['id']}: steady robot with unequal hops")
        if not r["fouled"]:
            land = takeoff(r) + r["jump"]
            if not (11 <= land <= 19 and 2 <= r["jump"] <= 10):
                errs.append(f"robot {r['id']}: landing {land} outside the pit")
        if r["hop_frames"][0] < EDGE_F or r["end_frame"] + (tip if r["fouled"] else 0) > n - EDGE_F:
            errs.append(f"robot {r['id']}: an event is within 0.5 s of the clip edge")
        if scene["scenario"] in SEQUENTIAL and not r["fouled"] and not aligned(r["end_frame"]):
            errs.append(f"robot {r['id']}: landing frame not timer-aligned")
    jumpers = [r for r in rs if not r["fouled"]]
    if len(jumpers) < 2:
        errs.append("fewer than two robots jump")
    sc = sorted(score(r) for r in jumpers)
    if any(b - a < 1 for a, b in zip(sc, sc[1:])):
        errs.append("two jumping robots tie on score (need >= 0.1 apart)")
    if scene["scenario"] in SEQUENTIAL:                     # attempts in turns never overlap in time
        spans = sorted((r["hop_frames"][0], r["end_frame"] + (tip if r["fouled"] else 0)) for r in rs)
        if any(b[0] < a[1] + 2 for a, b in zip(spans, spans[1:])):
            errs.append("attempts overlap in time")
    return errs
