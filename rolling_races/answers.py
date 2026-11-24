"""Independent answers for rolling_races (validator, tests, frame check).

Nothing here reads sim.py's race runs. Factual answers are measured from what
the renderer draws, frame by frame (render.state, or the same read off decoded
MP4 frames by frame_check.py): each ball's screen centre per frame and which
rack slots are filled. A ball has finished from the first frame its rack slot
is filled; it has fallen if it ends up resting on the pit floor; it leaves its
track at a gap when its centre crosses the gap's near edge (the crossing time
is fitted to the centres of the frames around it); the race is released on
the first frame without the start gate. Clock readings are those of
the frame on which the event first shows (tenths = frame * 10 // 24).

What-ifs (and the final-speed question) are re-simulated with a separate
integrator written here: on each straight piece of the drawn polyline the
acceleration along the track is constant, so the exit speed and the time are
exact (v_b^2 = v_a^2 + 2 g dy, dt = 2 L / (v_a + v_b)); a flight is the
parabola from the lip, intersected with each polyline piece in closed form
(a quadratic per piece). sim.py instead integrates ds / v with Simpson's
rule and finds the landing by bisection.
"""
from __future__ import annotations
import json
import math

import sim
import render

FPS = 24
G = 9.8
ORD = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]
_CACHE: dict = {}


# ------------------------------------------------------------------ formats (shared with questions.py)
def ball(c):
    return f"the {c} ball"


def fmt_s(tenths):
    return f"{tenths / 10:.1f} s"


def clock_tenths(f):
    return f * 10 // FPS


def event_clock(te):
    """Clock reading (tenths) on the first frame drawn at or after an event at te seconds."""
    return clock_tenths(math.ceil(te * FPS - 1e-9))


# ------------------------------------------------------------------ the observed race
class Obs:
    """Per-ball observations from a list of per-frame states {"balls": {colour: (x, y)}, "rack": [colours]}."""

    def __init__(self, scene, frames, jitter=0.0):
        self.s, self.fr, self.jit = scene, frames, jitter
        st = scene["style"]
        self.R = st["R"]
        self.dir = -1 if st["mirror"] else 1
        sx, sy = sim.px(scene, 0, 0)
        self.start = (sx, sy - self.R)
        self.floor_y = sim.floor_px(scene) - self.R
        self.cols = sorted({c for fr in frames for c in fr["balls"]} | {c for fr in frames for c in fr["rack"]})
        self.fin = {c: next((f for f, fr in enumerate(frames) if c in fr["rack"]), None) for c in self.cols}
        self.rest = {c: next((f for f, fr in enumerate(frames) if c in fr["balls"]
                              and fr["balls"][c][1] >= self.floor_y - 1.0 - jitter), None) for c in self.cols}
        self.release = next((f for f, fr in enumerate(frames) if not fr["gate"]), None)

    def prog(self, p):
        return (p[0] - self.start[0]) * self.dir

    def finishers(self):
        return sorted((c for c in self.cols if self.fin[c] is not None), key=lambda c: self.fin[c])

    def fallen(self):
        return [c for c in self.cols if self.rest[c] is not None]

    def crossing(self, c, xw):
        """Clip time at which ball c's centre crosses world x = xw (fitted to the frames around it)."""
        target = self.prog(sim.px(self.s, xw, 0)) + 0.0
        f1 = next((f for f, fr in enumerate(self.fr) if c in fr["balls"] and self.prog(fr["balls"][c]) >= target),
                  None)
        if f1 is None:
            return None
        pts = [(f, self.prog(self.fr[f]["balls"][c])) for f in range(f1 - 3, f1 + 3)
               if 0 <= f < len(self.fr) and c in self.fr[f]["balls"]]
        n = len(pts)
        mf, mp = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
        sff = sum((p[0] - mf) ** 2 for p in pts)
        k = sum((p[0] - mf) * (p[1] - mp) for p in pts) / sff if sff > 0 else 0.0
        if k <= 1e-9:                                  # no usable fit: the first frame past the edge
            return f1 / FPS
        return (mf + (target - mp) / k) / FPS

    def fall_gap(self, c):
        """The gap a fallen ball drops through: the last gap whose near edge it crosses while still on its
        track line (a falling ball can drift past the next gap's edge far below the track)."""
        b = next(b for b in self.s["balls"] if b["colour"] == c)
        sh = sim.shape(self.s, b)
        out = None
        for gi, g in enumerate(b.get("gaps", ())):
            t = self.crossing(c, g["x0"])
            if t is None:
                break
            f = min(len(self.fr) - 1, math.ceil(t * FPS))
            near = [self.fr[k]["balls"][c] for k in (f, f + 1, f - 1, f + 2, f - 2)
                    if 0 <= k < len(self.fr) and c in self.fr[k]["balls"]]
            if not near:                              # not located around the crossing (decoded frames)
                break
            p = near[0]
            st = self.s["style"]                      # the track line (continued over the gap) under the ball
            xw = ((sim.SIZE[0] - p[0]) if st["mirror"] else p[0]) - st["ox"]
            xw = min(self.s["W"], max(0.0, xw / st["S"]))
            ty = sim.px(self.s, xw, sim.depth_at(sh, xw))[1] - self.R
            if abs(p[1] - ty) > 10 + self.jit:
                break
            out = gi
        return out

    def leave_t(self, c, gi=None):
        b = next(b for b in self.s["balls"] if b["colour"] == c)
        gi = self.fall_gap(c) if gi is None else gi
        return self.crossing(c, b["gaps"][gi]["x0"])

    def leader(self, f):
        """(leader, margin px) among the balls drawn on frame f that are not resting on the pit floor."""
        live = [(self.prog(p), c) for c, p in self.fr[f]["balls"].items() if p[1] < self.floor_y - 1.0 - self.jit]
        if len(live) < 2:
            return (live[0][1], 999.0) if live else (None, 0.0)
        live.sort(reverse=True)
        return live[0][1], live[0][0] - live[1][0]


def _frames(scene):
    key = json.dumps(scene, sort_keys=True)
    if key not in _CACHE:
        if len(_CACHE) > 8:
            _CACHE.clear()
        _CACHE[key] = [render.state(scene, f) for f in range(render.n_frames(scene))]
    return _CACHE[key]


def lead_changes(o: Obs, f_end: int, clear_px: float):
    """Leader on each frame from the release up to f_end where the lead is clear (>= clear_px); returns
    (number of changes, frames of the changes)."""
    seq = []
    for f in range(o.release, f_end):
        c, m = o.leader(f)
        if c is not None and m >= clear_px:
            seq.append((f, c))
    ch = [f for (fa, a), (f, b) in zip(seq, seq[1:]) if a != b]
    return len(ch), ch


# ------------------------------------------------------------------ independent integrator (what-ifs)
def _polyline(scene, b, straight=False):
    fam = "straight" if straight else b["family"]
    return sim.polyline((fam, 0.0 if straight else b.get("a", 0.0), 0 if straight else b.get("k", 0),
                         scene["W"], scene["D"]))


def _piece(pts, x):
    lo, hi = 0, len(pts) - 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if pts[mid][0] <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _y(pts, x):
    i = _piece(pts, x)
    (xa, ya), (xb, yb) = pts[i], pts[i + 1]
    return ya + (yb - ya) * (x - xa) / (xb - xa)


def _roll(pts, xa, xb, v, t):
    """Exact constant-acceleration roll from x = xa to x = xb starting at speed v: (t, v) at xb, or None."""
    ia, ib = _piece(pts, xa), _piece(pts, xb)
    p = (xa, _y(pts, xa))
    stops = [pts[j] for j in range(ia + 1, ib + 1) if xa < pts[j][0] < xb] + [(xb, _y(pts, xb))]
    for q in stops:
        L = math.dist(p, q)
        v2 = v * v + 2 * G * (q[1] - p[1])
        if v2 < 0.01:
            return None
        vb = math.sqrt(v2)
        t += 2 * L / (v + vb)
        v, p = vb, q
    return t, v


def _hit(pts, x0, y0, vx, vy):
    """First x > x0 where the parabola from (x0, y0) with velocity (vx, vy) meets the polyline."""
    A = G / (2 * vx * vx)
    B = vy / vx
    for j in range(_piece(pts, x0), len(pts) - 1):
        (xa, ya), (xb, yb) = pts[j], pts[j + 1]
        m = (yb - ya) / (xb - xa)
        # y0 + B u + A u^2 = ya + m (x0 + u - xa), u = x - x0
        a, bq, cq = A, B - m, y0 - ya - m * (x0 - xa)
        disc = bq * bq - 4 * a * cq
        if disc >= 0:
            for u in sorted(((-bq - math.sqrt(disc)) / (2 * a), (-bq + math.sqrt(disc)) / (2 * a))):
                x = x0 + u
                if u > 1e-6 and max(xa, x0) <= x <= xb:
                    return x
    return None


def race(scene, b, v0, no_gap=False, straight=False):
    """(finish physics time or None, final speed or None, fell)."""
    pts = _polyline(scene, b, straight)
    gp = [] if (no_gap or straight) else [(g["x0"], g["x1"], g["lip"]) for g in b.get("gaps", ())]
    x, v, t = 0.0, v0, 0.0
    for gx0, gx1, lip in gp:
        r = _roll(pts, x, gx0, v, t)
        if r is None:
            return None, None, False
        t, v = r
        i = _piece(pts, gx0 - 1e-9)
        phi = math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0]) - math.radians(lip)
        vx, vy = v * math.cos(phi), v * math.sin(phi)
        y0 = _y(pts, gx0)
        xc = _hit(pts, gx0, y0, vx, vy)
        if xc is None or xc < gx1:
            return None, None, True
        tau = (xc - gx0) / vx
        j = _piece(pts, xc)
        tx, ty = pts[j + 1][0] - pts[j][0], pts[j + 1][1] - pts[j][1]
        v = (vx * tx + (vy + G * tau) * ty) / math.hypot(tx, ty)
        t, x = t + tau, xc
    r = _roll(pts, x, scene["W"], v, t)
    if r is None:
        return None, None, False
    return r[0], r[1], False


def _only(cands):
    """The one option with the property; None (a validation failure) unless exactly one has it."""
    cands = list(cands)
    return ball(cands[0]) if len(cands) == 1 else None


def _cf_clock(scene, o, tau):
    return o.release / FPS + scene["slow"] * tau if tau is not None else None


def _rank(t, others):
    return ORD[sum(1 for u in others if u < t)]


# ------------------------------------------------------------------ the answers
def answer(key: str, p: dict, scene: dict, frames=None, jitter=0.0):
    fr = frames if frames is not None else _frames(scene)
    o = Obs(scene, fr, jitter)
    fin = o.finishers()
    fell = o.fallen()
    by = {b["colour"]: b for b in scene["balls"]}
    last = fr[-1]
    t_of = {c: o.fin[c] / FPS for c in fin}
    if key == "n_balls":
        return str(len(o.cols))
    if key == "n_gapped":
        return str(sum(1 for b in scene["balls"] if b.get("gaps")))
    if key == "n_fell":
        return str(len(fell))
    if key == "n_finish":
        return str(len(last["rack"]))
    if key == "winner":
        return ball(fin[0])
    if key == "last_finisher":
        return ball(fin[-1])
    if key == "lowest_track":
        low = {b["colour"]: max(y for run in render.track_runs(scene, b) for _, y in run) for b in scene["balls"]}
        return ball(max(low, key=low.get))
    if key == "straight_track":
        def bend(b):
            pts = [q for run in render.track_runs(scene, b) for q in run]
            (xa, ya), (xb, yb) = pts[0], pts[-1]
            return max(abs((xb - xa) * (y - ya) - (yb - ya) * (x - xa)) / math.dist(pts[0], pts[-1]) for x, y in pts)
        return ball(min(scene["balls"], key=bend)["colour"])
    if key == "two_gap_track":
        return ball(next(b["colour"] for b in scene["balls"] if len(render.track_runs(scene, b)) == 3))
    if key == "earlier":
        a, b = p["pair"]
        return ball(min((a, b), key=lambda c: t_of[c]))
    if key == "fell_first":
        a, b = p["pair"]
        return ball(min((a, b), key=o.leave_t))
    if key == "which_falls":
        return _only(c for c in p["options"] if c in fell)
    if key == "which_lands":
        landed = lambda c: (c in fin and by[c].get("gaps")) or (c in fell and (o.fall_gap(c) or 0) > 0)
        return _only(c for c in p["options"] if landed(c))
    if key == "leader_at":
        fs = [f for f in range(len(fr)) if clock_tenths(f) == p["tenths"]]
        lead = {o.leader(f)[0] for f in fs}
        return ball(lead.pop()) if len(lead) == 1 else None
    if key == "same_speed":
        sp = [race(scene, b, scene["v0"])[1] for b in scene["balls"]]
        if None in sp or max(sp) - min(sp) > 1e-3 * max(sp):
            return None
        return "they are all moving at the same speed"
    if key == "before_x":
        return _only(c for c in p["options"] if c in fin and t_of[c] < t_of[p["x"]])
    if key == "after_x":
        return _only(c for c in p["options"] if c in fin and t_of[c] > t_of[p["x"]])
    if key == "finish_clock":
        return fmt_s(clock_tenths(o.fin[p["x"]]))
    if key == "floor_clock":
        return fmt_s(clock_tenths(o.rest[p["x"]]))
    if key == "leave_clock":
        return fmt_s(event_clock(o.leave_t(p["x"], p["gap"])))
    if key == "release_clock":
        return fmt_s(clock_tenths(o.release))
    if key == "race_time":
        return fmt_s(clock_tenths(o.fin[p["x"]]) - clock_tenths(o.release))
    if key == "finish_gap":
        a, b = p["pair"]
        return fmt_s(clock_tenths(o.fin[b]) - clock_tenths(o.fin[a]))
    if key == "finish_order":
        return ", ".join(fin)
    if key == "place_of":
        return ORD[fin.index(p["x"])]
    if key == "next_after":
        return ball(fin[fin.index(p["x"]) + 1])
    if key == "fell_before_first":
        t1 = t_of[fin[0]]
        return str(sum(1 for c in fell if o.leave_t(c) < t1))
    if key == "n_done_at_rest":
        f = o.rest[p["x"]]
        return str(len(fr[f]["rack"]))
    if key == "n_after_x":
        return str(len(fin) - 1 - fin.index(p["x"]))
    if key == "n_racing_at_finish":
        f = o.fin[p["x"]]
        return str(sum(1 for c, q in fr[f]["balls"].items() if q[1] < o.floor_y - 1.0 - jitter))
    if key == "total_landings":
        return str(sum(len(by[c].get("gaps", ())) for c in fin) + sum(o.fall_gap(c) for c in fell))
    if key == "lead_changes":
        return str(lead_changes(o, o.fin[fin[0]], p["clear_px"])[0])
    # ---- what-ifs: own integrator
    if key in ("cf_no_gap", "cf_straight"):
        x = p["x"]
        tau = race(scene, by[x], scene["v0"], no_gap=key == "cf_no_gap", straight=key == "cf_straight")[0]
        return _rank(_cf_clock(scene, o, tau), [t_of[c] for c in fin if c != x])
    if key in ("cf_no_gap_clock", "cf_straight_clock", "cf_push_clock"):
        x = p["x"]
        v0 = scene["v0"] * (sim.PUSH_CF if key == "cf_push_clock" else 1.0)
        tau = race(scene, by[x], v0, no_gap=key == "cf_no_gap_clock", straight=key == "cf_straight_clock")[0]
        return fmt_s(event_clock(_cf_clock(scene, o, tau)))
    if key == "cf_push_falls":
        return str(sum(1 for b in scene["balls"] if race(scene, b, scene["v0"] * sim.PUSH_CF)[2]))
    if key == "cf_push_winner":
        ts = {b["colour"]: race(scene, b, scene["v0"] * sim.PUSH_CF)[0] for b in scene["balls"]}
        return ball(min((c for c in ts if ts[c] is not None), key=ts.get))
    if key == "cf_no_gaps_winner":
        ts = {b["colour"]: race(scene, b, scene["v0"], no_gap=True)[0] for b in scene["balls"]}
        return ball(min(ts, key=ts.get))
    if key == "cf_push_lands":
        return _only(c for c in p["options"] if not race(scene, by[c], scene["v0"] * sim.PUSH_CF)[2])
    return None
