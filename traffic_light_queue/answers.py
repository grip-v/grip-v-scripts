"""Independent answers for traffic_light_queue (validator, tests, frame check).

Factual answers come from what the renderer draws, frame by frame
(render.state, or the boxes / lamp colours found in the decoded MP4 by
frame_check.py): each vehicle's box (its front read off the box and the known
body length) and each lamp's colour. A vehicle is standing still over a frame
interval when its front does not move; a stop is the first still interval
after moving; it drives past a stop line in the first frame its front is
beyond the line.

What-ifs re-simulate with a separate fixed-step integrator here (1/480 s,
cars handled front-to-back by position), not with sim.py's simulation.
"""
from __future__ import annotations
import json

import render

FPS = 24
_CACHE: dict = {}


# ------------------------------------------------------------------ names / formats (shared with questions.py)
def name(scene, i):
    c = scene["cars"][i]
    return f"the {c['colour']} {c['body']}"


def light_name(scene, li):
    sc = scene["scenario"]
    if sc == "two_lanes":
        ys = sorted(l["a"][1] for l in scene["lanes"])
        lane = next(l for l in scene["lanes"] if li in l["lights"])
        return "the light on the " + ("upper" if lane["a"][1] == ys[0] else "lower") + " road"
    if sc == "crossing":
        lane = next(l for l in scene["lanes"] if li in l["lights"])
        return "the light on the " + ("left-to-right" if lane["dir"] == "h" else "top-to-bottom") + " road"
    if sc == "two_lights":
        order = sorted(range(len(scene["lights"])), key=lambda k: scene["lights"][k]["s"])
        return "the " + ("first" if order.index(li) == 0 else "second") + " light"
    return "the traffic light"


def fmt_clock(tenths):
    return f"{tenths / 10:.1f} s"


def clock_tenths(f):
    return f * 10 // FPS


def frame_of_clock(t):
    k = round(t * 10)
    f = (k * FPS + 9) // 10
    while clock_tenths(f) < k:
        f += 1
    return f


def order_text(names):
    return ", then ".join(names)


# ------------------------------------------------------------------ tracks from drawn frames
def drawn(scene):
    key = json.dumps(scene, sort_keys=True)
    if key not in _CACHE:
        if len(_CACHE) > 16:
            _CACHE.clear()
        _CACHE[key] = [render.state(scene, f) for f in range(render.n_frames(scene))]
    return _CACHE[key]


class Tracks:
    """front[i][f]: distance of car i's front along its lane (None when not on screen); red[li][f]."""

    def __init__(self, scene, frames, jitter=0.0):
        self.s, self.fr, self.j = scene, frames, jitter
        self.N = len(frames)
        self.n = len(scene["cars"])
        self.front = [[None] * self.N for _ in range(self.n)]
        for f, st in enumerate(frames):
            for c in st["cars"]:
                self.front[c["id"]][f] = self._front(c["id"], c["box"])
        self.red = [[st["lights"][li]["red"] for st in frames] for li in range(len(scene["lights"]))]

    def _front(self, i, box):
        car = self.s["cars"][i]
        lane = self.s["lanes"][car["lane"]]
        k = 0 if lane["dir"] == "h" else 1
        lo, hi = box[k], box[k + 2]
        if hi >= lane["b"][k] - 1:                     # cut off at the far end: the back is the true one
            return lo + car["len"] - lane["a"][k]
        return hi - lane["a"][k]

    def present(self, i, f):
        return 0 <= f < self.N and self.front[i][f] is not None

    def moving(self, i, f):
        """Front moves over frame interval f -> f+1 (None when unknown)."""
        if not (self.present(i, f) and self.present(i, f + 1)):
            return None
        return self.front[i][f + 1] - self.front[i][f] > 0.5 + 2 * self.j

    def seen(self, i):
        return any(p is not None for p in self.front[i])

    def stop_frames(self, i):
        return [f for f in range(1, self.N - 1) if self.moving(i, f - 1) is True and self.moving(i, f) is False]

    def still(self, i, f):
        return self.moving(i, f) is False

    def line(self, li):
        return self.s["lights"][li]["s"]

    def on_lane(self, i, li):
        return li in self.s["lanes"][self.s["cars"][i]["lane"]]["lights"]

    def cross_frame(self, i, li):
        if not self.on_lane(i, li):
            return None
        for f in range(self.N):
            p = self.front[i][f]
            if p is not None and p > self.line(li) + 0.5 + 3 * self.j:   # a car waiting at the line is exactly on it
                return f
        return None

    def changes(self, li):
        r = self.red[li]
        return [(f, "red" if r[f] else "green") for f in range(1, self.N) if r[f] != r[f - 1]]


# ------------------------------------------------------------------ independent what-if integrator
def _red(L, t):
    P = L["red"] + L["green"]
    return L["red"] > 0 and (t - L["phase"]) % P < L["red"]


def resim(scene, t_end):
    """Re-run the traffic with 1/480 s steps. Returns (crossings {(car, light): t}, stops {car: count})."""
    h = 1 / 480
    cars = scene["cars"]
    gap = scene["gap"]
    lights = scene["lights"]
    s = {c["id"]: None for c in cars}
    done = set()
    still = {c["id"]: False for c in cars}
    stops = {c["id"]: 0 for c in cars}
    cross = {}
    waiting = {}
    for c in sorted(cars, key=lambda c: (c["t_in"], c["id"])):
        waiting.setdefault(c["lane"], []).append(c)
    n = int(round(t_end / h)) + 1
    for k in range(n):
        t = k * h
        for lane_i, lane in enumerate(scene["lanes"]):
            on = sorted((c for c in cars if c["lane"] == lane_i and s[c["id"]] is not None and c["id"] not in done),
                        key=lambda c: -s[c["id"]])
            q = waiting.get(lane_i, [])
            if q and t + 1e-9 >= q[0]["t_in"]:
                last = on[-1] if on else None
                if last is None or s[last["id"]] - last["len"] - gap >= 0:
                    c = q.pop(0)
                    s[c["id"]] = 0.0                     # placed at the gate; it starts moving on the next step
                    on.append(c)
                    fresh = c["id"]
                else:
                    fresh = None
            else:
                fresh = None
            for j, c in enumerate(on):
                i = c["id"]
                if i == fresh:
                    continue
                x = s[i]
                nx = x + c["v"] * h
                if j > 0:
                    a = on[j - 1]
                    if a["id"] not in done:
                        nx = min(nx, s[a["id"]] - a["len"] - gap)
                for li in lane["lights"]:
                    if x <= lights[li]["s"] + 1e-9 and _red(lights[li], t):
                        nx = min(nx, lights[li]["s"])
                nx = max(nx, x)
                if nx - x <= 1e-9:
                    if not still[i]:
                        still[i] = True
                        stops[i] += 1
                else:
                    still[i] = False
                for li in lane["lights"]:
                    if (i, li) not in cross and x <= lights[li]["s"] < nx:
                        cross[(i, li)] = t + (lights[li]["s"] - x) / c["v"]
                s[i] = nx
                if nx - c["len"] > lane["length"]:
                    done.add(i)
    return cross, stops


def _with(scene, **kw):
    sc = json.loads(json.dumps(scene))
    if "green" in kw:
        sc["lights"][kw["green"]]["red"] = 0.0
    if "remove" in kw:
        sc["cars"] = [c for c in sc["cars"] if c["id"] != kw["remove"]]
    if "late" in kw:
        for c in sc["cars"]:
            c["t_in"] = c["t_in"] + kw["late"]
    return sc


# ------------------------------------------------------------------ the dispatcher
def answer(key: str, p: dict, scene: dict, frames=None, jitter=0.0):
    T = Tracks(scene, frames if frames is not None else drawn(scene), jitter)
    last = T.N - 1
    ids = [i for i in range(T.n) if T.seen(i)]
    colour = lambda red: "red" if red else "green"   # noqa: E731
    if key == "n_lights":
        return len(T.red)
    if key == "n_on_road_end":
        return sum(1 for i in ids if T.present(i, last))
    if key == "light_end":
        return colour(T.red[p["l"]][last])
    if key == "longest_end":
        on = [i for i in ids if T.present(i, last)]
        st = {c["id"]: c["box"] for c in T.fr[last]["cars"]}

        def length(i):
            b = st[i]
            return b[2] - b[0] if scene["lanes"][scene["cars"][i]["lane"]]["dir"] == "h" else b[3] - b[1]
        return name(scene, max(on, key=length))
    if key == "light_at_t":
        return colour(T.red[p["l"]][frame_of_clock(p["t"])])
    if key == "first_enter":
        return name(scene, min(ids, key=lambda i: next(f for f in range(T.N) if T.present(i, f))))
    if key == "first_stop":
        st = [(T.stop_frames(i)[0], i) for i in ids if T.stop_frames(i)]
        return name(scene, min(st)[1])
    if key == "front_of_queue_at_t":
        f = frame_of_clock(p["t"])
        at = [i for i in ids if T.on_lane(i, p["l"]) and T.still(i, f) and abs(T.front[i][f] - T.line(p["l"])) < 2 + T.j]
        return name(scene, at[0]) if len(at) == 1 else None
    if key == "n_stops":
        return len(T.stop_frames(p["o"]))
    if key == "time_first_stop":
        return fmt_clock(clock_tenths(T.stop_frames(p["o"])[0]))
    if key == "time_first_green":
        return fmt_clock(clock_tenths(next(f for f, k in T.changes(p["l"]) if k == "green")))
    if key == "n_crossed_by_t":
        f = frame_of_clock(p["t"])
        return sum(1 for i in ids if (cf := T.cross_frame(i, p["l"])) is not None and cf <= f)
    if key == "n_still_at_t":
        f = frame_of_clock(p["t"])
        return sum(1 for i in ids if T.still(i, f))
    if key == "n_turn_red":
        return sum(1 for f, k in T.changes(p["l"]) if k == "red")
    if key == "order_cross":
        return order_text([name(scene, i) for i in sorted(p["os"], key=lambda i: T.cross_frame(i, p["l"]))])
    if key == "n_crossed_end":
        return sum(1 for i in ids if T.cross_frame(i, p["l"]) is not None)
    if key == "never_cross":
        nv = [i for i in ids if T.on_lane(i, p["l"]) and T.cross_frame(i, p["l"]) is None]
        return name(scene, nv[0]) if len(nv) == 1 else None
    if key == "n_still_end":
        return sum(1 for i in ids if T.still(i, last - 1))
    # ---------------- what-ifs (re-simulated)
    dur = render.duration(scene)
    if key == "cf_always_green":
        cr, _ = resim(_with(scene, green=p["l"]), dur)
        return sum(1 for (i, li), t in cr.items() if li == p["l"] and t <= dur)
    if key == "cf_remove_stops":
        _, st = resim(_with(scene, remove=p["remove"]), dur)
        return st[p["o"]]
    if key == "cf_late":
        cr, _ = resim(_with(scene, late=p["d"]), dur)
        return sum(1 for (i, li), t in cr.items() if li == p["l"] and t <= dur)
    return None
