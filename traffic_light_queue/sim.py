"""traffic_light_queue: cars queue at traffic lights.

Cars enter at the start of a lane (a gate) and drive at their own constant
speed. A car stops dead at a stop line while its light is red (if its front
has not crossed the line yet) and behind the car ahead (keeping a fixed
gap); it moves on at full speed as soon as it can. Lights cycle
red -> green with their own red / green durations and phase.

Scenarios (scene["scenario"]):
  single       one lane, one light
  two_lanes    two parallel lanes, each with its own light and cycle
  crossing     a horizontal and a vertical road crossing; their lights are
               opposite (one green while the other is red) with a short
               all-red gap
  mixed        one lane, cars with clearly different speeds (fast cars
               close up on slow ones)
  two_lights   one lane with two lights in a row
Cars are named by colour + body (unique colour per clip). The clock is the
only text. Fixed-step simulation (1/240 s); times are seconds; frame f is
shown at t = f / 24 and the clock shows f * 10 // 24 tenths.
Pure function of the rng passed in.
"""
from __future__ import annotations
import math
import random

FPS = 24
DT = 1 / 240
SUB = 10                         # simulation steps per frame
MAX_FRAMES = 240
EDGE_S = 0.5
AREA = (250, 20, 1262, 704)
CLOCK_BOX = (18, 16, 214, 84)
SCENARIOS = ("single", "two_lanes", "crossing", "mixed", "two_lights")
COLOURS = {
    "red": (222, 40, 40), "orange": (247, 130, 20), "yellow": (240, 200, 20), "green": (35, 165, 60),
    "teal": (0, 170, 170), "blue": (40, 90, 230), "purple": (140, 60, 210), "pink": (240, 100, 190),
    "brown": (135, 80, 35), "lime": (150, 220, 40),
}
BODIES = {"car": 1.0, "van": 1.25, "truck": 1.6, "bus": 1.9}          # length factor
THEMES = {
    "day": {"bg": [(110, 160, 90), (140, 150, 120), (170, 160, 130)], "road": [(70, 70, 76), (90, 88, 86)],
            "mark": (235, 235, 235)},
    "night": {"bg": [(20, 30, 26), (28, 26, 36), (16, 20, 30)], "road": [(46, 46, 52), (56, 54, 60)],
              "mark": (210, 210, 200)},
}


# ------------------------------------------------------------------ lights
def light_red(L, t):
    """Red at time t? Cycle: red for L['red'] s, then green for L['green'] s, shifted by L['phase']."""
    c = (t - L["phase"]) % (L["red"] + L["green"])
    return c < L["red"]


def light_changes(L, t0, t1):
    """[(t, 'green'|'red')] switch times in [t0, t1]."""
    P = L["red"] + L["green"]
    out = []
    k = math.floor((t0 - L["phase"]) / P) - 1
    while True:
        base = L["phase"] + k * P
        for t, kind in ((base + L["red"], "green"), (base + P, "red")):
            if t0 <= t <= t1:
                out.append((t, kind))
        if base > t1:
            break
        k += 1
    return sorted(out)


# ------------------------------------------------------------------ the simulation
def simulate(scene, t_end=None, override=None):
    """Positions of every car on a frame-aligned fixed-step grid.
    Returns {"pos": {car: [s or None per frame]}, "events": [(t, kind, car, extra)]} where s is the distance of
    the car's FRONT along its lane from the lane start, None before it enters / after it leaves.
    events: enter, stop, go, cross (front passes a stop line; extra = light index), leave."""
    sc = dict(scene)
    if override:
        sc.update(override)
    t_end = duration(sc) if t_end is None else t_end
    lanes, lights, cars = sc["lanes"], sc["lights"], sc["cars"]
    n_steps = int(round(t_end / DT))
    s = {c["id"]: None for c in cars}
    stopped = {c["id"]: False for c in cars}
    crossed = {c["id"]: set() for c in cars}
    arrive = {}
    pos = {c["id"]: [] for c in cars}
    events = []
    by_lane = {}
    for c in sorted(cars, key=lambda c: (c["t_in"], c["id"])):
        by_lane.setdefault(c["lane"], []).append(c)
    for k in range(n_steps + 1):
        t = k * DT
        for lane_i, lane_cars in by_lane.items():
            lane = lanes[lane_i]
            for j, c in enumerate(lane_cars):           # lane order = entry order; the car before is "ahead"
                i = c["id"]
                ahead = lane_cars[j - 1] if j > 0 else None
                sa = None if ahead is None else s[ahead["id"]]
                if s[i] == "gone":
                    continue
                if s[i] is None:
                    room = ahead is None or sa == "gone" or (sa is not None and sa - ahead["len"] - sc["gap"] >= 0)
                    if t + 1e-9 >= c["t_in"] and room:
                        s[i] = 0.0
                        events.append((t, "enter", i, None))
                    continue
                v = c["v"]
                lim = s[i] + v * DT
                if sa not in (None, "gone"):
                    lim = min(lim, sa - ahead["len"] - sc["gap"])
                for li in lane["lights"]:
                    line = lights[li]["s"]
                    if s[i] <= line + 1e-9 and light_red(lights[li], t):
                        lim = min(lim, line)
                new = max(s[i], lim)
                ds = new - s[i]
                moving = ds > 1e-9
                if moving and ds < v * DT - 1e-9 and not stopped[i]:
                    arrive[i] = t + ds / v                  # reached its limit part-way through the step
                if not moving and not stopped[i]:
                    stopped[i] = True
                    events.append((arrive.get(i, t) if arrive.get(i, -1) > t - 2 * DT else t, "stop", i,
                                   next((q for q in lane["lights"] if abs(new - lights[q]["s"]) < 1e-6), "queue")))
                elif moving and stopped[i]:
                    stopped[i] = False
                    events.append((t + DT - ds / v, "go", i, None))   # began moving part-way through the step
                if moving and ds >= v * DT - 1e-9:
                    arrive.pop(i, None)
                for li in lane["lights"]:
                    if li not in crossed[i] and s[i] <= lights[li]["s"] < new:
                        crossed[i].add(li)
                        events.append((t + (lights[li]["s"] - s[i]) / v, "cross", i, li))
                s[i] = new
                if s[i] - c["len"] > lane["length"]:
                    s[i] = "gone"
                    events.append((t, "leave", i, None))
        if k % SUB == 0:
            for c in cars:
                v = s[c["id"]]
                pos[c["id"]].append(None if v in (None, "gone") else v)
    return {"pos": pos, "events": sorted(events, key=lambda e: (e[0], e[1], e[2]))}


def n_frames(scene):
    return scene["frames"]


def duration(scene):
    return scene["frames"] / FPS


def clock_tenths(frame: int) -> int:
    return frame * 10 // FPS


def lane_xy(lane, s):
    """Screen point at distance s along a lane."""
    (x0, y0), (x1, y1) = lane["a"], lane["b"]
    L = math.dist((x0, y0), (x1, y1))
    u = s / L
    return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u


# ------------------------------------------------------------------ generation
def _style(rng):
    theme = rng.choice(("day", "day", "night"))
    T = THEMES[theme]
    return {"theme": theme, "bg": list(rng.choice(T["bg"])), "road": list(rng.choice(T["road"])),
            "mark": list(T["mark"]), "lane_w": rng.randint(54, 74), "dash": rng.choice(("dashed", "solid", "none")),
            "verge": rng.choice(("grass", "plain", "kerb")), "lamp_px": rng.randint(22, 30),
            "pole": rng.choice(("left", "right"))}


def _lanes(rng, scenario, st):
    W = st["lane_w"]
    x_lo, x_hi = AREA[0] + 10, AREA[2] - 10
    if scenario == "crossing":
        y = rng.uniform(260, 470)
        x = rng.uniform(560, 960)
        h = {"a": [x_lo, y], "b": [x_hi, y], "lights": [0], "dir": "h"}
        v = {"a": [x, AREA[1] + 10], "b": [x, AREA[3] - 10], "lights": [1], "dir": "v"}
        h["length"] = x_hi - x_lo
        v["length"] = v["b"][1] - v["a"][1]
        return [h, v], {"x": x, "y": y}
    ys = [rng.uniform(220, 520)] if scenario != "two_lanes" else sorted(rng.sample([260, 360, 460, 560], 2))
    if scenario == "two_lanes" and ys[1] - ys[0] < W + 30:
        ys[1] = ys[0] + W + 40
    lanes = []
    for i, y in enumerate(ys):
        lanes.append({"a": [x_lo, y], "b": [x_hi, y], "lights": [i] if scenario == "two_lanes" else [0],
                      "dir": "h", "length": x_hi - x_lo})
    if scenario == "two_lights":
        lanes[0]["lights"] = [0, 1]
    return lanes, None


def _lights(rng, scenario, lanes, centre):
    def cyc():
        return {"red": round(rng.uniform(1.6, 3.4), 2), "green": round(rng.uniform(1.6, 3.4), 2),
                "phase": round(rng.uniform(0, 3.0), 2)}
    if scenario == "crossing":
        a = cyc()
        allred = round(rng.uniform(0.5, 0.8), 2)
        g = round(a["red"] - 2 * allred, 2)            # the vertical light is green only while the other is red,
        if g < 0.8:                                    # with an all-red margin at both ends
            return None
        b = {"red": round(a["green"] + 2 * allred, 2), "green": g, "phase": round(a["phase"] + a["red"] - allred, 2)}
        half = lanes[0].get("w", 0) or 0
        a["s"] = round(centre["x"] - 100 - lanes[0]["a"][0], 2)    # lamps stay clear of the other road
        b["s"] = round(centre["y"] - 100 - lanes[1]["a"][1], 2)
        return [a, b]
    out = []
    n = 2 if scenario in ("two_lanes", "two_lights") else 1
    for i in range(n):
        L = cyc()
        lane = lanes[min(i, len(lanes) - 1)]
        if scenario == "two_lights":
            L["s"] = round(rng.uniform(0.30, 0.42) * lane["length"] if i == 0 else rng.uniform(0.66, 0.8) * lane["length"], 2)
        else:
            L["s"] = round(rng.uniform(0.45, 0.72) * lane["length"], 2)
        out.append(L)
    return out


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    for attempt in range(1, 5000):
        st = _style(rng)
        lanes, centre = _lanes(rng, scenario, st)
        lights = _lights(rng, scenario, lanes, centre)
        if lights is None:
            continue
        n = rng.randint(4, 8)
        cols = rng.sample(sorted(COLOURS), n)
        base_len = rng.uniform(38, 52)
        gap = round(rng.uniform(10, 22), 2)
        v0 = rng.uniform(130, 210)
        cars, t = [], round(rng.uniform(0.0, 0.4), 3)
        for i in range(n):
            body = rng.choice(sorted(BODIES))
            lane = rng.randrange(len(lanes))
            v = v0 if scenario != "mixed" else rng.choice((0.7, 0.85, 1.0, 1.3)) * v0
            cars.append({"id": i, "colour": cols[i], "body": body, "lane": lane, "len": round(base_len * BODIES[body], 2),
                         "v": round(v, 2), "t_in": round(t, 3)})
            t += rng.uniform(0.4, 1.4)
        frames = rng.randint(168, MAX_FRAMES)
        scene = {"scenario": scenario, "frames": frames, "lanes": lanes, "lights": lights, "cars": cars, "gap": gap,
                 "centre": centre, "style": st, "build_attempts": attempt}
        if check(scene) or not _ladder_ok(scene):
            continue
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
def check(scene: dict) -> list[str]:
    errs = []
    dur = duration(scene)
    res = simulate(scene)
    ev = [e for e in res["events"] if e[0] <= dur]
    if len({c["colour"] for c in scene["cars"]}) != len(scene["cars"]):
        errs.append("two cars share a colour")
    if sum(1 for e in ev if e[1] == "enter") < 4:
        errs.append("fewer than 4 cars enter")
    if sum(1 for e in ev if e[1] == "stop") < 2:
        errs.append("fewer than 2 stops")
    # every car event is clear of light changes (a car reaching the line as the light turns is ambiguous)
    ch = [[t for t, _ in light_changes(L, 0, dur + 1)] for L in scene["lights"]]
    waiting = {}                                       # car -> light it is waiting at (stop extra = light index)
    for t, kind, i, extra in ev:
        if kind == "stop":
            waiting[i] = extra
        if kind == "cross" and waiting.get(i) == extra:
            waiting.pop(i)
            continue                                   # a car that waited at this line pulls away on green
        if kind in ("stop", "cross") and isinstance(extra, int) and any(abs(t - c) < 0.15 for c in ch[extra]):
            errs.append("a car reaches a stop line as the light changes")
            break
    # a car's own events are >= 0.3 s apart; no event within 0.5 s of the clip ends unless it is an entry
    per, wait = {}, {}
    for t, kind, i, extra in ev:                       # pull-away + crossing of a car waiting at the line is one moment
        if kind == "stop":
            wait[i] = extra
        if kind == "cross" and wait.pop(i, None) == extra:
            continue
        per.setdefault(i, []).append(t)
    if any(b - a < 0.3 - 1e-9 for ts in per.values() for a, b in zip(ts, ts[1:])):
        errs.append("a car's events are too close together")
    if any((t < EDGE_S and kind != "enter") or t > dur - EDGE_S for t, kind, *_ in ev):
        errs.append("an event at a clip edge")
    # crossing: the two roads never hold cars inside the junction box at the same time
    if scene["scenario"] == "crossing":
        c = scene["centre"]
        half = scene["style"]["lane_w"] / 2
        for f in range(scene["frames"]):
            inside = set()
            for car in scene["cars"]:
                sv = res["pos"][car["id"]][f]
                if sv is None:
                    continue
                lane = scene["lanes"][car["lane"]]
                a = lane["a"][0] if lane["dir"] == "h" else lane["a"][1]
                mid = (c["x"] if lane["dir"] == "h" else c["y"]) - a
                if sv > mid - half and sv - car["len"] < mid + half:
                    inside.add(lane["dir"])
            if len(inside) > 1:
                errs.append("cars from both roads inside the junction")
                break
    if scene["scenario"] == "mixed" and len({c["v"] for c in scene["cars"]}) < 2:
        errs.append("mixed scenario with one speed")
    return errs
