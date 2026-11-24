"""Independent answers for object_tracking_mot_v2 (validator, tests, frame check).

Factual answers come from what the renderer draws, frame by frame
(render.state, or centres found in the decoded MP4 by frame_check.py):
visible objects' centres only. Unique objects are followed by their look;
identical objects (targets scenario) by predicted-position matching from
the frames where the rings show which ones are targets. While an object is
under a screen it is not seen; its track is bridged in a straight line
(sim.check guarantees nothing happens behind a screen). A wall bounce is a
sign flip of one velocity component next to a wall; a collision is any
other velocity change, the partner being the object in contact then.

What-ifs re-simulate with a separate fixed-step integrator here (contact
times found by bisection inside each step), not with sim.py's event solver.
"""
from __future__ import annotations
import json
import math

import render

FPS = 24
_CACHE: dict = {}
QUADS = ("top-left", "top-right", "bottom-left", "bottom-right")


# ------------------------------------------------------------------ names / formats (shared with questions.py)
def name(scene, i):
    o = scene["objects"][i]
    return f"the {o['colour']} {o['shape']}"


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


def quad(scene, x, y):
    a = scene["arena"]
    cx, cy = (a["x0"] + a["x1"]) / 2, (a["y0"] + a["y1"]) / 2
    return ("top" if y < cy else "bottom") + "-" + ("left" if x < cx else "right")


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
    """pos[i][f] = (x, y) or None when not seen; for the targets scenario identities come from matching."""

    def __init__(self, scene, frames, jitter=0.0):
        self.s, self.fr, self.j = scene, frames, jitter
        self.n = len(scene["objects"])
        self.N = len(frames)
        if scene["scenario"] == "targets":
            self.pos = self._match()
        else:
            self.pos = [[None] * self.N for _ in range(self.n)]
            for f, st in enumerate(frames):
                for o in st["objects"]:
                    if o["visible"]:
                        self.pos[o["id"]][f] = (o["x"], o["y"])
        self._bridge()
        self.ringed = self._ringed()

    def _match(self):
        """Identical objects: link frame to frame by predicted position (frame 0 order is arbitrary but fixed)."""
        first = sorted(((o["x"], o["y"]) for o in self.fr[0]["objects"]), key=lambda p: (round(p[0]), round(p[1])))
        tr = [[p] for p in first]
        for f in range(1, self.N):
            pts = [(o["x"], o["y"]) for o in self.fr[f]["objects"]]
            pred = [(2 * t[-1][0] - t[-2][0], 2 * t[-1][1] - t[-2][1]) if len(t) > 1 else t[-1] for t in tr]
            pairs = sorted((math.dist(pred[i], p), i, j) for i in range(len(tr)) for j, p in enumerate(pts))
            ut, up, nxt = set(), set(), [None] * len(tr)
            for dd, i, j in pairs:
                if i not in ut and j not in up:
                    nxt[i] = pts[j]
                    ut.add(i)
                    up.add(j)
            for i, p in enumerate(nxt):
                tr[i].append(p if p is not None else tr[i][-1])
        self.track_ids = list(range(len(tr)))
        return tr

    def _bridge(self):
        for p in self.pos:
            f = 0
            while f < self.N:
                if p[f] is None:
                    a = f - 1
                    b = f
                    while b < self.N and p[b] is None:
                        b += 1
                    if a >= 1 and b < self.N:
                        va = (p[a][0] - p[a - 1][0], p[a][1] - p[a - 1][1])
                        for k in range(f, b):
                            p[k] = (p[a][0] + va[0] * (k - a), p[a][1] + va[1] * (k - a))
                    f = b
                else:
                    f += 1

    def _ringed(self):
        if self.s["scenario"] != "targets":
            return []
        f = 0
        rings = self.fr[0].get("ring_xy")
        if rings is None:                               # rendered state: the renderer rings scene["targets"]
            rings = [(o["x"], o["y"]) for o in self.fr[0]["objects"] if o["id"] in self.s["targets"]]
        return sorted(min(range(self.n), key=lambda i: math.dist(self.pos[i][f], r)) for r in rings)

    # ---------------- events (geometry: contact-range minima where the velocity really changes)
    def _v(self, i, f):
        """Velocity (px/s) over frame interval f -> f+1, or None if a position is missing."""
        if f < 0 or f + 1 >= self.N:
            return None
        a, b = self.pos[i][f], self.pos[i][f + 1]
        if a is None or b is None:
            return None
        return (b[0] - a[0]) * FPS, (b[1] - a[1]) * FPS

    def _changed(self, i, f):
        """Velocity before frame f-1 vs after frame f+1 differ clearly (the event lies near frame f)."""
        v0, v1 = self._v(i, f - 2), self._v(i, f + 1)
        if v0 is None or v1 is None:
            return None
        return math.hypot(v1[0] - v0[0], v1[1] - v0[1]) > 8 + 0.1 * math.hypot(*v0), v0, v1

    def wall_events(self, i):
        a = self.s["arena"]
        r = self.s["objects"][i]["r"]
        out = []
        for side, comp, lim, sgn in (("left", 0, a["x0"] + r, -1), ("right", 0, a["x1"] - r, 1),
                                     ("top", 1, a["y0"] + r, -1), ("bottom", 1, a["y1"] - r, 1)):
            d = [None if p is None else sgn * (lim - p[comp]) for p in self.pos[i]]   # >= 0 inside
            for f in range(1, self.N - 1):
                if d[f] is None or d[f - 1] is None or d[f + 1] is None:
                    continue
                if d[f] <= d[f - 1] and d[f] < d[f + 1] and d[f] < 30:
                    ch = self._changed(i, f)
                    if not ch or not ch[0]:
                        continue
                    v0, v1 = ch[1], ch[2]
                    # a real bounce: heading into this wall, heading out after, and reaching the wall line
                    if v0[comp] * sgn <= 0 or v1[comp] * sgn >= 0 or d[f] > abs(v0[comp]) / FPS + 2 + self.j:
                        continue
                    # contact time from the incoming motion
                    fa = f - 1
                    t = fa / FPS + (lim - self.pos[i][fa][comp]) / v0[comp]
                    out.append((t, "wall", side))
        return sorted(out)

    def hit_events(self):
        """[(t, i, j)] collisions: minima of the pair gap within contact range where both velocities change."""
        out = []
        ob = self.s["objects"]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                R = ob[i]["r"] + ob[j]["r"]
                g = [None if self.pos[i][f] is None or self.pos[j][f] is None
                     else math.dist(self.pos[i][f], self.pos[j][f]) - R for f in range(self.N)]
                for f in range(1, self.N - 1):
                    if None in (g[f - 1], g[f], g[f + 1]):
                        continue
                    if g[f] <= g[f - 1] and g[f] < g[f + 1] and g[f] < 30:
                        ci, cj = self._changed(i, f), self._changed(j, f)
                        if not ci or not cj or not ci[0] or not cj[0]:
                            continue
                        fa = f - 1 if g[f - 1] > 0 and g[f] <= 0 else f - 1
                        out.append((self._contact(i, j, fa), i, j))
        return sorted(out)

    def events(self, i):
        ev = [(t, "wall", side) for t, _, side in self.wall_events(i)]
        ev += [(t, "hit", j if a == i else a) for t, a, j in self.hit_events() if i in (a, j)]
        return sorted(ev)

    def _contact(self, i, j, f):
        """Time the two discs touch, extrapolating both from frame f on their incoming velocities."""
        R = self.s["objects"][i]["r"] + self.s["objects"][j]["r"]
        f = max(1, f)
        vi, vj = self._v(i, f - 1), self._v(j, f - 1)
        pi, pj = self.pos[i][f], self.pos[j][f]
        dx, dy = pj[0] - pi[0], pj[1] - pi[1]
        dvx, dvy = vj[0] - vi[0], vj[1] - vi[1]
        A = dvx * dvx + dvy * dvy
        B = 2 * (dx * dvx + dy * dvy)
        C = dx * dx + dy * dy - R * R
        disc = B * B - 4 * A * C
        if A < 1e-9 or disc < 0:
            return f / FPS
        return f / FPS + (-B - math.sqrt(disc)) / (2 * A)

    def hidden_frames(self, i):
        return [f for f, st in enumerate(self.fr) if not next(o for o in st["objects"] if o["id"] == i)["visible"]] \
            if self.s["scenario"] != "targets" else []


# ------------------------------------------------------------------ independent what-if integrator
def resim(scene, *, remove=None, speed=1.0, delay=0.0, t_end=None, hits=None):
    """Positions of every object at t_end under the change. Fixed 1 ms steps; within a step the earliest
    wall or disc contact is located by bisection and resolved (equal-mass elastic)."""
    a = scene["arena"]
    t_end = render.duration(scene) if t_end is None else t_end
    objs = []
    for o in scene["objects"]:
        if o["id"] == remove:
            continue
        objs.append({"id": o["id"], "r": o["r"], "x": o["x"], "y": o["y"], "vx": o["vx"] * speed, "vy": o["vy"] * speed})
    t = scene["start_t"] + delay
    h = 1e-3

    def gap(p, q, s):
        return math.hypot(q["x"] + q["vx"] * s - p["x"] - p["vx"] * s, q["y"] + q["vy"] * s - p["y"] - p["vy"] * s) \
            - p["r"] - q["r"]

    while t < t_end - 1e-12:
        step = min(h, t_end - t)
        first, what = step, None
        for k, o in enumerate(objs):
            for ax, lo, hi in (("x", a["x0"], a["x1"]), ("y", a["y0"], a["y1"])):
                v = o["v" + ax]
                if v > 0 and o[ax] + v * step > hi - o["r"]:
                    s = (hi - o["r"] - o[ax]) / v
                elif v < 0 and o[ax] + v * step < lo + o["r"]:
                    s = (lo + o["r"] - o[ax]) / v
                else:
                    continue
                if s < first:
                    first, what = max(s, 0.0), ("w", k, ax)
            for m in range(k + 1, len(objs)):
                q = objs[m]
                if gap(o, q, step) < 0 and gap(o, q, 0) >= -1e-9:
                    lo_, hi_ = 0.0, step
                    for _ in range(50):
                        mid = (lo_ + hi_) / 2
                        if gap(o, q, mid) < 0:
                            hi_ = mid
                        else:
                            lo_ = mid
                    if lo_ < first:
                        first, what = lo_, ("c", k, m)
        for o in objs:
            o["x"] += o["vx"] * first
            o["y"] += o["vy"] * first
        t += first
        if what and what[0] == "w":
            objs[what[1]]["v" + what[2]] *= -1
        elif what:
            p, q = objs[what[1]], objs[what[2]]
            if hits is not None:                         # count each contact for both objects
                hits[p["id"]] = hits.get(p["id"], 0) + 1
                hits[q["id"]] = hits.get(q["id"], 0) + 1
            nx, ny = q["x"] - p["x"], q["y"] - p["y"]
            n = math.hypot(nx, ny)
            nx, ny = nx / n, ny / n
            dp = (q["vx"] - p["vx"]) * nx + (q["vy"] - p["vy"]) * ny
            if dp < 0:
                p["vx"] += dp * nx
                p["vy"] += dp * ny
                q["vx"] -= dp * nx
                q["vy"] -= dp * ny
    return {o["id"]: (o["x"], o["y"]) for o in objs}


def resim_hits(scene, who, remove=None):
    """Number of collisions `who` has during the clip with `remove` taken out (same integrator as resim)."""
    hits = {}
    resim(scene, remove=remove, hits=hits)
    return hits.get(who, 0)


def orbit_at(o, t, speed=1.0):
    return (o["cx"] + o["ax"] * math.sin(o["wx"] * speed * t + o["px"]),
            o["cy"] + o["ay"] * math.sin(o["wy"] * speed * t + o["py"]))


# ------------------------------------------------------------------ the dispatcher
def answer(key: str, p: dict, scene: dict, frames=None, jitter=0.0):
    T = Tracks(scene, frames if frames is not None else drawn(scene), jitter)
    objs = scene["objects"]
    last = T.N - 1
    if key == "n_objects":
        return len(T.fr[0]["objects"])
    if key == "n_targets":
        return len(T.ringed)
    if key == "shape_of":
        return objs[p["o"]]["shape"]
    if key == "pattern_of":
        return objs[p["o"]]["pattern"]
    if key == "largest":
        return name(scene, max(objs, key=lambda o: o["r"])["id"])
    if key == "quadrant_at_t":
        f = frame_of_clock(p["t"])
        return quad(scene, *T.pos[p["o"]][f])
    if key == "leftmost_at_t":
        f = frame_of_clock(p["t"])
        return name(scene, min(range(T.n), key=lambda i: T.pos[i][f][0]))
    if key == "first_partner":
        return name(scene, next(e[2] for e in T.events(p["o"]) if e[1] == "hit"))
    if key == "first_wall":
        return next(e[2] for e in T.events(p["o"]) if e[1] == "wall")
    if key == "closest_at_end":
        me = T.pos[p["o"]][last]
        return name(scene, min((i for i in range(T.n) if i != p["o"]), key=lambda i: math.dist(T.pos[i][last], me)))
    if key == "targets_in_quad_at_t":
        f = frame_of_clock(p["t"])
        return sum(1 for i in T.ringed if quad(scene, *T.pos[i][f]) == p["q"])
    if key == "untargeted_in_quad_at_t":
        f = frame_of_clock(p["t"])
        return sum(1 for i in range(T.n) if i not in T.ringed and quad(scene, *T.pos[i][f]) == p["q"])
    if key == "n_walls":
        return sum(1 for e in T.events(p["o"]) if e[1] == "wall" and p["t0"] <= e[0] <= p["t1"])
    if key == "n_hits":
        return sum(1 for e in T.events(p["o"]) if e[1] == "hit" and p["t0"] <= e[0] <= p["t1"])
    if key == "time_first_hit":
        t = next(e[0] for e in T.events(p["o"]) if e[1] == "hit")
        return fmt_clock(clock_tenths(math.ceil(t * FPS - 1e-9)))
    if key == "time_first_wall":
        t = next(e[0] for e in T.events(p["o"]) if e[1] == "wall")
        return fmt_clock(clock_tenths(math.ceil(t * FPS - 1e-9)))
    if key == "time_hidden":
        return fmt_clock(clock_tenths(T.hidden_frames(p["o"])[0]))
    if key == "quad_changes":
        qs = [quad(scene, *T.pos[p["o"]][f]) for f in range(T.N) if T.pos[p["o"]][f] is not None]
        return sum(1 for a, b in zip(qs, qs[1:]) if a != b)
    if key == "total_hits":
        return sum(1 for t, i, j in T.hit_events() if p["t0"] <= t <= p["t1"])
    if key == "order_end":
        return order_text([name(scene, i) for i in sorted(p["os"], key=lambda i: T.pos[i][last][0])])
    if key == "hidden_at_end":
        st = T.fr[last]
        return name(scene, next(o["id"] for o in st["objects"] if not o["visible"]))
    if key == "quadrant_end":
        return quad(scene, *T.pos[p["o"]][last])
    if key == "targets_in_quad_end":
        return sum(1 for i in T.ringed if quad(scene, *T.pos[i][last]) == p["q"])
    # ---------------- what-ifs (re-simulated)
    if key == "cf_remove_quad":
        return quad(scene, *resim(scene, remove=p["remove"])[p["o"]])
    if key == "cf_speed_quad":
        if scene["scenario"] == "orbit":
            return quad(scene, *orbit_at(objs[p["o"]], render.duration(scene), p["speed"]))
        return quad(scene, *resim(scene, speed=p["speed"])[p["o"]])
    if key == "cf_remove_hits":
        return resim_hits(scene, p["o"], remove=p["remove"])
    if key == "cf_delay_targets":
        pos = resim(scene, delay=p["d"])
        return sum(1 for i in scene["targets"] if quad(scene, *pos[i]) == p["q"])
    return None
