"""Independent answers for robot_archery.

Never reads sim.py's game log (turns, eliminated, winner) or its draws /
angles / landing times. Everything comes from a per-frame *track* of what
is drawn: the landed arrow markers (colour + centre, in px), each robot's
position and whether it is shown eliminated, and its pips. The track is
built from `render.state` (or, in frame_check.py, from decoded MP4
pixels). The static drawing gives the rest: robot colours (names), the
target centre and ring size (px per ring), the shooting spot, and the
rule set of the scenario (what the clip shows happening on a record / a
miss).

From the track: a new marker on frame f is a shot by the robot of that
colour, landed on frame f, at distance |marker - centre| / (px per ring);
the starting queue is the robots at frame 0 sorted by distance from the
shooting spot; eliminations are the frames a robot is first shown out.
Records, holders and what-ifs come from a replay written here (a deque
plus a dict of pips), separate from sim.play.
"""
from __future__ import annotations
import functools
import json
import math
from collections import deque

FPS = 24
ORD = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")


def fmt_clock(tenths):
    return f"{tenths / 10:.1f} s"


def clock_of(frame):
    return frame * 10 // FPS


def track_from_render(scene) -> dict:
    import render
    F = scene["frames"]
    tr = {"arrows": [], "robots": []}
    for f in range(F):
        s = render.state(scene, f)
        tr["arrows"].append([(a["robot"], a["xy"]) for a in s["arrows"]])
        tr["robots"].append([(r["xy"], r["out"], r["lives"]) for r in s["robots"]])
    return tr


# ------------------------------------------------------------------ observations
class Obs:
    def __init__(self, scene, tr):
        tg = scene["target"]
        self.n = len(scene["robots"])
        self.cols = [r["colour"] for r in scene["robots"]]
        self.F = len(tr["arrows"])
        self.shots = []                                   # (robot, distance in rings, landing frame)
        seen = 0
        for f, arr in enumerate(tr["arrows"]):
            while len(arr) > seen:
                r, (x, y) = arr[seen]
                self.shots.append((r, math.hypot(x - tg["cx"], y - tg["cy"]) / tg["unit"], f))
                seen += 1
        sx, sy = scene["layout"]["spot"]
        r0 = tr["robots"][0]
        self.order = sorted(range(self.n), key=lambda r: math.hypot(r0[r][0][0] - sx, r0[r][0][1] - sy))
        self.out_frame = {}
        for f, rs in enumerate(tr["robots"]):
            for r, (_, out, _) in enumerate(rs):
                if out and r not in self.out_frame:
                    self.out_frame[r] = f
        self.lives_track = [[lv for _, _, lv in rs] for rs in tr["robots"]]
        self.lives0 = max(self.lives_track[0])
        self.stay = scene["scenario"] == "stay_on"
        self.log = replay(self.order, [d for _, d, _ in self.shots], self.lives0, self.stay)
        self.dist = [d for _, d, _ in self.shots]

    def nm(self, r):
        return f"the {self.cols[r]} robot"

    def shot_index(self, robot, nth):
        ks = [k for k, (r, _, _) in enumerate(self.shots) if r == robot]
        return ks[nth - 1]

    def shot_name(self, k):
        r = self.shots[k][0]
        mine = [j for j, (rr, _, _) in enumerate(self.shots) if rr == r]
        return f"{self.nm(r)}'s {ORD[mine.index(k)]} shot" if len(mine) > 1 else f"{self.nm(r)}'s only shot"

    def landed_at(self, f):
        return sum(1 for _, _, lf in self.shots if lf <= f)


def replay(order, dists, lives=1, stay=False):
    """Own replay: returns dict(log=[(shooter, d, is_record, prev_holder, out)], out=[...], winner)."""
    q = deque(order)
    pips = {r: lives for r in order}
    best, holder, on_spot = math.inf, None, None
    log, out = [], []
    for d in dists:
        if len(q) + (on_spot is not None) < 2:
            break
        who = on_spot if on_spot is not None else q.popleft()
        on_spot = None
        rec = d < best
        prev = holder
        if rec:
            best, holder = d, who
            if stay:
                on_spot = who
            else:
                q.append(who)
            log.append((who, d, True, prev, False))
            continue
        pips[who] -= 1
        if pips[who] <= 0:
            out.append(who)
            log.append((who, d, False, prev, True))
        else:
            q.append(who)
            log.append((who, d, False, prev, False))
    alive = list(q) + ([on_spot] if on_spot is not None else [])
    return {"log": log, "out": out, "winner": alive[0] if len(alive) == 1 else None}


def ring(d):
    return max(1, math.ceil(d - 1e-9))


@functools.lru_cache(maxsize=8)
def _obs_cached(key: str):
    scene = json.loads(key)
    return Obs(scene, track_from_render(scene))


def obs(scene):
    return _obs_cached(json.dumps(scene, sort_keys=True))


# ------------------------------------------------------------------ derivations
def _frames_of(o, x):
    return [f for f in range(o.F) if clock_of(f) == x]


def derive(key, p, o: Obs, scene):
    log = o.log["log"]
    if key == "n_robots":
        return o.n
    if key == "n_arrows":
        return len(o.shots)
    if key == "shots_by":
        return sum(1 for r, _, _ in o.shots if r == p["robot"])
    if key == "first_shooter":
        return o.nm(o.shots[0][0])
    if key == "winner":
        alive = [r for r in range(o.n) if r not in o.out_frame]
        return o.nm(alive[0]) if len(alive) == 1 else None
    if key == "record_beaten":
        k = o.shot_index(p["robot"], p["nth"])
        return o.nm(log[k][3]) if log[k][2] and log[k][3] is not None else None
    if key == "shooter_after":
        k = o.shot_index(p["robot"], p["nth"])
        return o.nm(o.shots[k + 1][0])
    if key == "out_after":
        seq = sorted(o.out_frame, key=o.out_frame.get)
        return o.nm(seq[seq.index(p["robot"]) + 1])
    if key == "holder_at_out":
        k = max(j for j, (r, _, f) in enumerate(o.shots) if r == p["robot"] and f <= o.out_frame[p["robot"]])
        return o.nm(log[k][3])
    if key == "closest_of":
        ks = [o.shot_index(s["robot"], s["nth"]) for s in p["shots"]]
        return o.shot_name(min(ks, key=lambda k: o.dist[k]))
    if key == "ring_of":
        return ring(o.dist[o.shot_index(p["robot"], p["nth"])])
    if key == "record_gain":
        k = o.shot_index(p["robot"], p["nth"])
        prev = min(o.dist[:k])
        return round(prev - o.dist[k])
    if key == "out_time":
        return fmt_clock(clock_of(o.out_frame[p["robot"]]))
    if key == "last_record_time":
        k = max(j for j, e in enumerate(log) if e[2])
        return fmt_clock(clock_of(o.shots[k][2]))
    if key == "arrows_at":
        return o.landed_at(_frames_of(o, p["x"])[0])
    if key == "record_ring_at":
        m = o.landed_at(_frames_of(o, p["x"])[0])
        return ring(min(o.dist[:m]))
    if key == "elim_order":
        return " → ".join(o.cols[r] for r in sorted(o.out_frame, key=o.out_frame.get))
    if key == "records_by":
        return sum(1 for e in log if e[0] == p["robot"] and e[2])
    if key == "most_records":
        cnt = {r: sum(1 for e in log if e[0] == r and e[2]) for r in range(o.n)}
        return o.nm(max(cnt, key=cnt.get))
    if key == "n_records":
        return sum(1 for e in log if e[2])
    if key == "arrows_at_out":
        return o.landed_at(o.out_frame[p["robot"]])
    if key == "full_pips_at":
        f = _frames_of(o, p["x"])[0]
        return sum(1 for lv in o.lives_track[f] if lv == o.lives0)
    if key in ("cf_shot_winner", "cf_shot_arrows"):
        k = o.shot_index(p["robot"], p["nth"])
        d = list(o.dist)
        d[k] = p["ring"] - 0.5
        g = replay(o.order, d, o.lives0, o.stay)
        if g["winner"] is None:
            return None
        return o.nm(g["winner"]) if key == "cf_shot_winner" else len(g["log"])
    if key == "cf_swap_winner":
        a, b = p["a"], p["b"]
        order = [b if r == a else a if r == b else r for r in o.order]
        g = replay(order, o.dist, o.lives0, o.stay)
        return o.nm(g["winner"]) if g["winner"] is not None else None
    if key in ("cf_rule_winner", "cf_rule_first_out"):
        lives, stay = {"stay": (1, True), "back": (1, False), "one_life": (1, False)}[p["rule"]]
        g = replay(o.order, o.dist, lives, stay)
        if g["winner"] is None:
            return None
        return o.nm(g["winner"]) if key == "cf_rule_winner" else o.nm(g["out"][0])
    return None


def answer(key: str, p: dict, scene: dict):
    return derive(key, p, obs(scene), scene)
