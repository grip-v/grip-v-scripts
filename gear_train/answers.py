"""Independent answers for gear_train (used only by the validator and the tests).

Nothing here reads the generator's log (scene["log"]) or sim.py's closed-form
motion. From the drawn world alone (gear centres, tooth counts, module, the
slide path of the gears that move, the brake block's path, the driver's
motor programme) it re-runs the clip with a time-stepped integration at
8 sub-steps per frame:

  - at every step, which gears touch is decided from the geometry (their
    tooth-tip circles overlap); the driver's connected group turns, meshing
    neighbours turn opposite ways with |omega| * teeth shared;
  - a group cut off from the driver coasts down, stands still and spins up
    (the disengage spin profile); a group that the brake block touches
    slows to a stop and restarts when the block lifts off;
  - angles are integrated step by step (trapezoid rule) and read back at
    the frame times.

Counterfactuals re-run the same integration on a modified world.
"""
from __future__ import annotations
import hashlib
import json
import math
from fractions import Fraction

import numpy as np

import questions as Q
import render

FPS = 24
SUB = 8
EPS_W = 1e-6


def _ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class World:
    def __init__(self, scene, flip=False, remove=None, loose_mesh=None, cut=None, brake_on=None,
                 no_event=False, double=None):
        self.scene = scene
        m = scene["m"]
        self.m = m
        prog = scene["program"]
        sc = scene["scenario"]
        gs = [g for g in scene["gears"] if g["id"] != remove]
        self.ids = [g["id"] for g in gs]
        self.g = {g["id"]: g for g in gs}
        self.col = {g["id"]: g["colour"] for g in gs}
        D = scene["duration"]
        self.D = D
        self.END = (int(round(D * FPS)) - 1) / FPS            # the last frame shown
        self.nfr = int(round(D * FPS))
        N = self.nfr * SUB + 1
        dt = 1.0 / (FPS * SUB)
        t = np.arange(N) * dt
        self.t = t
        # --- where every gear is (the drawn slide path) ---
        pos = {k: (np.full(N, self.g[k]["x"]), np.full(N, self.g[k]["y"])) for k in self.ids}
        dis = prog.get("disengage")
        if dis and not no_event and cut is None:
            k_ = _ss((t - dis["t_out"]) / dis["slide"]) - _ss((t - (dis["still1"] + dis["ramp"])) / dis["slide"])
            for k in dis["moving"]:
                if k in pos:
                    pos[k] = (pos[k][0] + k_ * dis["shift"][0], pos[k][1] + k_ * dis["shift"][1])
        self.pos = pos
        # --- contacts from geometry: tooth-tip circles overlap ---
        rp = {k: m * self.g[k]["z"] / 2.0 for k in self.ids}
        self.rp = rp
        pairs = []
        for i, a in enumerate(self.ids):
            for b in self.ids[i + 1:]:
                d = np.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1])
                touch = d < rp[a] + rp[b] + 2 * m - 0.5
                if loose_mesh and {a, b} == {loose_mesh, _loose(scene)}:
                    touch = np.ones(N, bool)
                if touch.any():
                    pairs.append((a, b, touch))
        drv = scene["driver"]
        # contact graph at t = 0 -> distances from the driver, and the cut edge for the counterfactual
        adj0 = {k: [] for k in self.ids}
        for a, b, tch in pairs:
            if tch[0]:
                adj0[a].append(b)
                adj0[b].append(a)
        self.dist0 = _bfs(adj0, drv) if drv in self.ids else {}
        if cut is not None:
            par = min((n for n in adj0[cut] if n in self.dist0), key=lambda n: self.dist0[n])
            win = (dis["still0"] - dis["ramp"] - 0.2, dis["still1"] + dis["ramp"] + 0.2)
            new = []
            for a, b, tch in pairs:
                if {a, b} == {cut, par}:
                    tch = tch & ~((t > win[0]) & (t < win[1]))
                new.append((a, b, tch))
            pairs = new
        self.adj0 = adj0
        # --- per-step contact state -> driver-connected group and its signs ---
        code = np.zeros(N, dtype=np.int64)
        for bit, (_, _, tch) in enumerate(pairs):
            code |= tch.astype(np.int64) << bit
        zs = {k: self.g[k]["z"] * (2 if k == double else 1) for k in self.ids}
        sign = -scene["sign"] if flip else scene["sign"]
        w_d = scene["w_driver"]                                   # the motor turns the driver at this speed
        motor = np.ones(N)
        if not no_event and sc == "reversal":
            for rv in prog["reversals"]:
                a = rv["mid"] - rv["ramp"] / 2
                motor = motor * np.cos(np.pi * np.clip((t - a) / rv["ramp"], 0, 1))
        if not no_event and sc == "speed_change":
            sp = prog["speed"]
            motor = 1 + (sp["factor"] - 1) * _ss((t - (sp["mid"] - sp["ramp"] / 2)) / sp["ramp"])
        brake = np.ones(N)
        self.brake_gear = None
        if prog.get("brake") and not no_event:
            b = prog["brake"]
            bg = brake_on or b["gear"]
            self.brake_gear = bg
            frac = _ss((t - (b["t_touch"] - b["approach"])) / b["approach"]) - _ss((t - b["release"]) / b["approach"])
            gap = b["travel"] * (1 - frac)                          # brake face to the gear's tooth tips
            touching = gap <= 0.02 * m
            idx = np.nonzero(touching)[0]
            self.touch = (t[idx[0]], t[idx[-1]])
            r = b["ramp"]
            brake = 1 - _ss((t - self.touch[0]) / r) + _ss((t - self.touch[1]) / r)
        free = np.ones(N)
        if dis and not no_event:
            free = 1 - _ss((t - (dis["still0"] - dis["ramp"])) / dis["ramp"]) + _ss((t - dis["still1"]) / dis["ramp"])
        w = {k: np.zeros(N) for k in self.ids}
        sign0 = {}
        if drv in self.ids:
            for k, dd in self.dist0.items():
                sign0[k] = sign * (-1) ** dd
        for cval in np.unique(code):
            sel = code == cval
            adj = {k: [] for k in self.ids}
            for bit, (a, b, _) in enumerate(pairs):
                if (int(cval) >> bit) & 1:
                    adj[a].append(b)
                    adj[b].append(a)
            dist = _bfs(adj, drv) if drv in self.ids else {}
            braked = self.brake_gear in dist
            for k in self.ids:
                mag = w_d * zs[drv] / zs[k] if drv in self.ids else 0.0
                if k in dist:
                    s = sign * (-1) ** dist[k]
                    w[k][sel] = (s * mag * motor * (brake if braked else 1.0))[sel]
                elif k in sign0:          # cut off from the driver after being driven: coasts on the spin profile
                    w[k][sel] = (sign0[k] * mag * free)[sel]
        self.w_fine = w
        ang = {}
        for k in self.ids:
            inc = (w[k][1:] + w[k][:-1]) / 2 * dt
            ang[k] = np.concatenate([[0.0], np.cumsum(inc)])
        self.ang_fine = ang
        self.path_fine = {k: np.concatenate([[0.0], np.cumsum(np.abs((w[k][1:] + w[k][:-1]) / 2 * dt))])
                          for k in self.ids}

    # -- readings (t in seconds, snapped to the step grid) --
    def idx(self, t):
        return int(round(t * FPS * SUB))

    def angle(self, k, t):
        return self.ang_fine[k][self.idx(t)]

    def rot(self, k, t0, t1):
        return self.angle(k, t1) - self.angle(k, t0)

    def w_at(self, k, t):
        return self.w_fine[k][self.idx(t)]

    def speed(self, k, t0, t1):
        return abs(self.rot(k, t0, t1)) / (t1 - t0)

    def dir(self, k, t0, t1):
        r = self.rot(k, t0, t1)
        return 0 if abs(r) < 1e-6 else (1 if r > 0 else -1)

    def start_dir(self, k):
        return self.dir(k, 0.0, 1.0)

    def end_dir(self, k):
        return self.dir(k, self.END - 1.0, self.END)

    def moved(self, k):
        return self.path_fine[k][-1] > 1e-6

    def still_run(self, k):
        """Longest stretch (t0, t1) during which gear k stands still while it otherwise turns."""
        z = np.abs(self.w_fine[k]) < EPS_W
        best, cur0, out = 0, None, None
        for i, v in enumerate(np.append(z, False)):
            if v and cur0 is None:
                cur0 = i
            elif not v and cur0 is not None:
                if i - cur0 > best:
                    best, out = i - cur0, (self.t[cur0], self.t[i - 1])
                cur0 = None
        return out


def _loose(scene):
    return next((g["id"] for g in scene["gears"] if g["loose"]), None)


def _bfs(adj, root):
    dist = {root: 0}
    q = [root]
    while q:
        u = q.pop(0)
        for v in sorted(adj[u]):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


_CACHE: dict = {}


def world(scene, **mods) -> World:
    key = hashlib.sha1(json.dumps([scene, mods], sort_keys=True).encode()).hexdigest()
    if key not in _CACHE:
        if len(_CACHE) > 24:
            _CACHE.clear()
        _CACHE[key] = World(scene, **mods)
    return _CACHE[key]


def _combo(w, a, b, da, db):
    if da == db:
        return f"both {Q.dir_word(da)}"
    return f"{w.col[a]} {Q.dir_word(da)}, {w.col[b]} {Q.dir_word(db)}"


def _setx(w, ks):
    return Q.set_text([w.col[k] for k in ks])


def _event_gear(scene, w):
    if scene["scenario"] == "brake":
        return w.brake_gear
    # disengage: the gear that loses contact with a neighbour that stays put, while it moves itself
    for a in w.ids:
        if np.ptp(w.pos[a][0]) + np.ptp(w.pos[a][1]) < 1:
            continue
        for b in w.adj0[a]:
            if np.ptp(w.pos[b][0]) + np.ptp(w.pos[b][1]) > 1:
                continue
            d = np.hypot(w.pos[a][0] - w.pos[b][0], w.pos[a][1] - w.pos[b][1])
            if (d > w.rp[a] + w.rp[b] + 2 * w.m).any():
                return a
    return None


def _event_partner(w, x):
    for b in w.adj0[x]:
        if np.ptp(w.pos[b][0]) + np.ptp(w.pos[b][1]) < 1:
            return b
    return None


def answer(key: str, params: dict, scene: dict):
    w = world(scene)
    p = params
    D = w.END
    sc = scene["scenario"]
    turning = [k for k in w.ids if w.moved(k)]
    if key == "gear_count":
        return len(w.ids)
    if key == "teeth":
        g = w.g[p["g"]]
        return render.count_teeth(g["z"], scene["m"])
    if key in ("start_dirs", "flip_driver"):
        ww = w if key == "start_dirs" else world(scene, flip=True)
        return _combo(w, p["a"], p["b"], ww.start_dir(p["a"]), ww.start_dir(p["b"]))
    if key == "loose_which":
        still = [k for k in w.ids if not w.moved(k)]
        return w.col[still[0]] if len(still) == 1 else None
    if key in ("largest", "smallest"):
        rs = sorted(w.ids, key=lambda k: w.rp[k])
        return w.col[rs[-1] if key == "largest" else rs[0]]
    if key == "event_gear":
        g = _event_gear(scene, w)
        return w.col[g] if g else None
    if key == "reversal_count":
        s = np.sign(w.w_fine[scene["driver"]])
        s = s[s != 0]
        return int(np.sum(s[1:] != s[:-1]))
    if key == "change_kind":
        r = {k: w.speed(k, D - 1.0, D) / w.speed(k, 0.0, 1.0) for k in turning}
        drv = scene["driver"]
        others_same = all(abs(r[k] - 1) < 1e-3 for k in turning if k != drv)
        if all(abs(r[k] - r[drv]) < 1e-3 for k in turning):
            return Q.CHANGE_OPTS[0] if r[drv] > 1 else Q.CHANGE_OPTS[1]
        if others_same:
            return Q.CHANGE_OPTS[2] if r[drv] > 1 else Q.CHANGE_OPTS[3]
        return None
    if key == "same_dir_as":
        return w.col[p["s"]] if w.start_dir(p["s"]) == w.start_dir(p["g"]) != 0 else None
    if key in ("fastest_of", "slowest_of"):
        sp = sorted(p["gs"], key=lambda k: w.speed(k, 0.0, 1.0))
        return w.col[sp[-1] if key == "fastest_of" else sp[0]]
    if key == "meshes_with":
        return w.col[p["nb"]] if p["nb"] in w.adj0[p["g"]] else None
    if key == "keeps_turning":
        x = _event_gear(scene, w)
        r = w.still_run(x)
        tm = (r[0] + r[1]) / 2
        turners = [k for k in w.ids if abs(w.w_at(k, tm)) > EPS_W]
        if p["k"] is None:
            return Q.NONE_OF if not turners else None
        return w.col[p["k"]] if p["k"] in turners else None
    if key == "slides_from":
        x = _event_gear(scene, w)
        b = _event_partner(w, x)
        return w.col[b] if b else None
    if key == "start_vs_end":
        k = p["g"]
        r = w.speed(k, D - 1.0, D) / w.speed(k, 0.0, 1.0)
        sp = Q.SPEED_WORDS[2] if abs(r - 1) < 0.02 else (Q.SPEED_WORDS[0] if r > 1 else Q.SPEED_WORDS[1])
        dd = Q.DIR_WORDS[0] if w.start_dir(k) == w.end_dir(k) else Q.DIR_WORDS[1]
        return f"{sp}, {dd}"
    if key == "turns":
        return int(math.floor(abs(w.rot(p["g"], p["t0"], p["t1"])) / (2 * math.pi)))
    if key == "ratio":
        r = w.speed(p["b"], 0.0, 1.0) / w.speed(p["a"], 0.0, 1.0)
        return str(Fraction(r).limit_denominator(6))
    if key == "period":
        return int(round(2 * math.pi / w.speed(p["g"], 0.0, 1.0)))
    if key == "event_time":
        drv = scene["driver"]
        if sc == "reversal":
            s = np.sign(w.w_fine[drv])
            idx = [i for i in range(1, len(s)) if s[i] != s[i - 1] and s[i] != 0]
            # a zero crossing shows up as (+, 0, -): take the step where it settles on the new sign
            zs = [w.t[i] for i in idx]
            flips, last = [], None
            for tz in zs:
                if last is None or tz - last > 0.5:
                    flips.append(tz)
                last = tz
            return int(round(flips[p["i"]]))
        if sc == "speed_change":
            a, b = w.speed(drv, 0.0, 1.0), w.speed(drv, D - 1.0, D)
            mid = (a + b) / 2
            ws = np.abs(w.w_fine[drv])
            i = int(np.argmax(ws > mid) if b > a else np.argmax(ws < mid))
            return int(round(w.t[i]))
        if sc == "disengage":
            x = _event_gear(scene, w)
            mv = np.hypot(w.pos[x][0] - w.pos[x][0][0], w.pos[x][1] - w.pos[x][1][0]) > 0.5
            return int(round(w.t[int(np.argmax(mv))]))
        if sc == "brake":
            return int(round(w.touch[0]))
        return None
    if key == "stop_duration":
        r = w.still_run(p["g"])
        return int(round(r[1] - r[0])) if r else None
    if key == "speed_factor":
        drv = scene["driver"]
        f = w.speed(drv, D - 1.0, D) / w.speed(drv, 0.0, 1.0)
        best = min(Q.FACTOR_TEXT, key=lambda v: abs(math.log(v) - math.log(f)))
        return Q.FACTOR_TEXT[best]
    if key in ("end_dirs", "no_reversal"):
        ww = w if key == "end_dirs" else world(scene, no_event=True)
        return _combo(w, p["a"], p["b"], ww.end_dir(p["a"]), ww.end_dir(p["b"]))
    if key == "stopped_followers":
        x = _event_gear(scene, w)
        r = w.still_run(x)
        tm = (r[0] + r[1]) / 2
        return _setx(w, [k for k in turning if k != x and abs(w.w_at(k, tm)) < EPS_W])
    if key == "still_count":
        return int(sum(abs(w.w_at(k, float(p["T"]))) < EPS_W for k in w.ids))
    if key == "speed_rank":
        order = sorted(p["gs"], key=lambda k: -w.speed(k, 0.0, 1.0))
        return ", ".join(w.col[k] for k in order)
    if key == "most_turns":
        best = max(p["gs"], key=lambda k: w.path_fine[k][w.idx(D)])
        return w.col[best]
    if key == "drive_path":
        k = p["g"]
        d = w.dist0
        par = [n for n in w.adj0[k] if d.get(n, 99) == d[k] - 1]
        return w.col[par[0]] if len(par) == 1 else None
    if key == "cw_count":
        if p["when"] == "start":
            return int(sum(w.start_dir(k) < 0 for k in w.ids))
        return int(sum(w.end_dir(k) < 0 for k in w.ids))
    if key == "remove":
        ww = world(scene, remove=p["g"])
        return _setx(w, [k for k in turning if k != p["g"] and not ww.moved(k)])
    if key == "loose_mesh":
        lz = _loose(scene)
        ww = world(scene, loose_mesh=p["g"])
        d = Q.dir_word(ww.start_dir(lz))
        sp = "faster" if ww.speed(lz, 0.0, 1.0) > ww.speed(p["g"], 0.0, 1.0) else "slower"
        return f"{d}, {sp} than the {w.col[p['g']]} gear"
    if key == "disengage_instead":
        ww = world(scene, cut=p["g"])
        dis = scene["program"]["disengage"]
        tm = (dis["still0"] + dis["still1"]) / 2
        return _setx(w, [k for k in turning if k != p["g"] and abs(ww.w_at(k, tm)) < EPS_W])
    if key == "brake_other":
        ww = world(scene, brake_on=p["g"])
        r = ww.still_run(p["g"])
        tm = (r[0] + r[1]) / 2
        return int(sum(1 for k in turning if abs(ww.w_at(k, tm)) < EPS_W))
    if key == "turns_cf":
        ww = world(scene, no_event=True)
        return int(math.floor(abs(ww.rot(p["g"], 0.0, D)) / (2 * math.pi)))
    if key == "teeth_double":
        ww = world(scene, double=p["x"])
        a, b = w.speed(p["y"], 0.0, 1.0), ww.speed(p["y"], 0.0, 1.0)
        if b < 1e-9:
            return Q.DOUBLE_OPTS[3]
        r = b / a
        if abs(r - 2) < 0.01:
            return Q.DOUBLE_OPTS[0]
        if abs(r - 0.5) < 0.01:
            return Q.DOUBLE_OPTS[1]
        if abs(r - 1) < 0.01:
            return Q.DOUBLE_OPTS[2]
        return None
    return None
