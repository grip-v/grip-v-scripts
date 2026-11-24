"""Independent answers for spring_chain_waves.

Never reads the pulse list, the launches, the speeds or the section table
of the scene. Everything comes from `render.track` (what is drawn: the
chain's height at every 2 px on every frame, which samples carry thick
beads, what holds each end, the ruler labels):

1. read the ruler length from the labels and the thickness changes from the
   bead flags (snapped to ruler marks);
2. on every frame find the separate bumps (runs above a small threshold,
   single-peaked, symmetric, clear of the ends and thickness changes) and
   link them frame to frame into tracks; fit a straight line to each track
   (position against time) and take its height;
3. speeds: the thin-bead speed is the median track speed there (a quarter
   mark per second grid), the thick-bead speed a half or a third of it;
4. launches: tracks leaving a paddle that no arriving track explains, timed
   back to the paddle (whole frames), with their sign and size;
5. re-trace the chain with its own depth-first tracer from those readings
   (clamp / paddle: back upside down, ring: same way up, thickness change:
   impedance-step split, parts under 0.3 dropped), and check the trace
   against every observed track and every observed track against the trace;
6. answer from the re-trace (counts at a moment are counted directly as
   bumps on the drawn frame); what-ifs re-trace under the changed rule.
"""
from __future__ import annotations
import functools
import json
import math
from fractions import Fraction as Fr

import numpy as np

import render

FPS = 24
CUT = Fr(3, 10)
SMALL = Fr(3, 5)
THR = 0.12
ORD = ["first", "second", "third", "fourth", "fifth", "sixth"]
HARDWARE = {"paddle": "a paddle", "clamped": "a clamp", "free": "a ring that slides on a pole"}
SPLIT = ["a smaller upside-down pulse bounces back and the rest goes on",
         "a smaller pulse bounces back the same way up and the rest goes on"]
RATIO = {Fr(1, 3): "about 1/3 as tall", Fr(1, 2): "about 1/2 as tall", Fr(2, 3): "about 2/3 as tall",
         Fr(1): "about as tall", Fr(4, 3): "about 4/3 as tall", Fr(3, 2): "about 3/2 as tall"}


class Mismatch(Exception):
    pass


def _clock(t: Fr) -> str:
    f = math.floor(t * FPS + Fr(1, 2))
    k = (10 * f) // FPS
    return f"{k // 10}.{k % 10} s"


def _word(sign, big, sized):
    shape = "bump up" if sign > 0 else "dip down"
    return f"a {'big' if big else 'small'} {shape}" if sized else f"a {shape}"


def _short(sign, big, sized):
    shape = "bump" if sign > 0 else "dip"
    return f"{'big' if big else 'small'} {shape}" if sized else shape


# ---------------------------------------------------------------- reading the drawn chain

def _runs(row, cuts):
    """Separate single bumps on one frame: (centre x index float, signed peak)."""
    out = []
    a = np.abs(row)
    on = a > THR
    n = len(row)
    i = 0
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and on[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


class Model:
    def __init__(self, tr):
        self.tr = tr
        self.L = int(tr["labels"][-1]["text"])
        self.xs = tr["xs"]
        self.dx = float(self.xs[1] - self.xs[0])
        self.nf = tr["n_frames"]
        self.tl = Fr(self.nf - 1, FPS)
        self.y = -tr["dy"] / tr["amp_px"]
        hv = tr["heavy"]
        cuts = [round(float((self.xs[i] + self.xs[i + 1]) / 2)) for i in range(len(hv) - 1) if hv[i] != hv[i + 1]]
        bounds = [0] + cuts + [self.L]
        self.parts = [(bounds[i], bounds[i + 1], bool(hv[np.searchsorted(self.xs, (bounds[i] + bounds[i + 1]) / 2)]))
                      for i in range(len(bounds) - 1) if bounds[i] < bounds[i + 1]]
        self.cuts = bounds
        self.ends = tr["ends"]
        paddles = [s for s in ("left", "right") if self.ends[s] == "paddle"]
        self.home = paddles[0] if len(paddles) == 1 else None
        self.far = None if self.home is None else ("right" if self.home == "left" else "left")
        self._observe()
        self._speeds()
        self._launches()
        self.segs = trace(self)
        self._verify()

    # -- observation
    def part_of(self, x):
        for k, (lo, hi, _) in enumerate(self.parts):
            if lo <= x < hi or (x == hi == self.L):
                return k
        return len(self.parts) - 1

    def bumps(self, f):
        """Clean single bumps on frame f: (x, height)."""
        row = self.y[f]
        xs = self.xs
        out = []
        for i, j in _runs(row, None):
            if i <= 1 or j >= len(row) - 2:
                continue
            lo_x, hi_x = xs[i], xs[j]
            if any(lo_x - 0.1 <= b <= hi_x + 0.1 for b in self.cuts):
                continue
            seg = row[i:j + 1]
            a = np.abs(seg)
            k = int(np.argmax(a))
            sign = 1 if seg[k] > 0 else -1
            if np.any(seg * sign < -THR / 2):
                continue
            # one peak, symmetric about it
            d = np.diff(a)
            if np.any(d[:k] < -0.02) or np.any(d[k:] > 0.02):
                continue
            wts = np.clip(a - THR, 0, None)
            cx = float((wts * xs[i:j + 1]).sum() / wts.sum())
            # symmetric about its peak at two heights (a neighbour's tail, or the mirror image at an
            # end, would skew one flank)
            sym = True
            for lev in (0.4, 0.7):
                above = xs[i:j + 1][a > a[k] * lev]
                if abs((xs[i + k] - above.min()) - (above.max() - xs[i + k])) > 0.02 + 1.5 * self.dx:
                    sym = False
            if not sym or abs(cx - xs[i + k]) > 0.015 + self.dx:
                continue
            out.append((cx, float(seg[k])))
        return out

    def count(self, f) -> int:
        """Separate bumps (runs above the threshold reaching 0.2) on frame f."""
        return sum(1 for i, j in _runs(self.y[f], None) if np.abs(self.y[f][i:j + 1]).max() >= 0.2)

    def _observe(self):
        tracks = []
        for f in range(self.nf):
            for x, h in self.bumps(f):
                part = self.part_of(x)
                best, bd = None, None
                for tk in tracks:
                    lf, lx, lh = tk["obs"][-1]
                    if tk["part"] != part or f - lf > 2 or f == lf or (lh > 0) != (h > 0) or abs(lh - h) > 0.2:
                        continue
                    if len(tk["obs"]) >= 2:
                        pf, px, _ = tk["obs"][-2]
                        v = (lx - px) / (lf - pf)
                        dist = abs(lx + v * (f - lf) - x)
                        if dist > 0.08:
                            continue
                    else:
                        dist = abs(x - lx)
                        if dist > 0.2 * (f - lf):
                            continue
                    if bd is None or dist < bd:
                        best, bd = tk, dist
                if best is None:
                    tracks.append({"part": part, "obs": [(f, x, h)]})
                else:
                    best["obs"].append((f, x, h))
        self.tracks = []
        for tk in tracks:
            if len(tk["obs"]) < 5:
                continue
            obs = tk["obs"]
            while len(obs) >= 5:                     # robust line: drop the worst point until it fits
                fs = np.array([o[0] for o in obs], float) / FPS
                xs = np.array([o[1] for o in obs])
                v, a = np.polyfit(fs, xs, 1)
                res = np.abs(a + v * fs - xs)
                if res.max() <= 0.03:
                    break
                obs = [o for n, o in enumerate(obs) if n != int(np.argmax(res))]
            if len(obs) < 5 or abs(v) < 0.3:
                continue
            hs = np.array([o[2] for o in obs])
            self.tracks.append({"part": tk["part"], "v": float(v), "a": float(a), "h": float(np.median(hs)), "n": len(obs),
                                "t0": float(fs[0]), "t1": float(fs[-1]), "x0": float(xs[0]), "x1": float(xs[-1])})

    def _speeds(self):
        def wmedian(tracks):                      # longer tracks weigh more
            vals = sorted((abs(t["v"]), t["n"]) for t in tracks)
            half, acc = sum(n for _, n in vals) / 2, 0
            for v, n in vals:
                acc += n
                if acc >= half:
                    return v
        thin = [t for t in self.tracks if not self.parts[t["part"]][2]]
        thick = [t for t in self.tracks if self.parts[t["part"]][2]]
        if not thin:
            raise Mismatch("no pulse seen on thin beads")
        self.c = Fr(round(wmedian(thin) * 4), 4)
        self.ratio = 1
        if any(p[2] for p in self.parts):
            if not thick:
                raise Mismatch("no pulse seen on thick beads")
            self.ratio = 2 if float(self.c) / wmedian(thick) < 2.5 else 3
        # a bump that doesn't move at its section's speed is two pulses overlapping, not a pulse
        self.tracks = [t for t in self.tracks if abs(abs(t["v"]) / float(self.speed_of(t["part"])) - 1) < 0.04]

    def speed_of(self, part, k=Fr(1)):
        return self.c * k / (self.ratio if self.parts[part][2] else 1)

    def _launches(self):
        found = []
        for side in ("left", "right"):
            if self.ends[side] != "paddle":
                continue
            xe = 0 if side == "left" else self.L
            part = 0 if side == "left" else len(self.parts) - 1
            c = float(self.speed_of(part))
            away = [t for t in self.tracks if t["part"] == part and (t["v"] > 0) == (side == "left")]
            come = [t for t in self.tracks if t["part"] == part and (t["v"] > 0) != (side == "left")]
            arrive = [t["t1"] + abs(xe - t["x1"]) / c for t in come]
            for t in away:
                t0 = t["t0"] - abs(t["x0"] - xe) / c
                if any(abs(t0 - ta) < 2.5 / FPS for ta in arrive):
                    continue
                f = int(round(t0 * FPS))
                if any(g["end"] == side and abs(g["f"] - f) <= 1 for g in found):
                    continue
                big = abs(t["h"]) > 0.8
                found.append({"f": f, "end": side, "sign": 1 if t["h"] > 0 else -1, "big": big})
        if not found:
            raise Mismatch("no launch seen")
        self.launches = sorted(found, key=lambda g: (g["f"], g["end"]))
        self.sized = len({g["big"] for g in self.launches}) == 2

    def _verify(self):
        """Every observed track lies on a traced pulse, and every traced pulse
        that starts in the clip is observed."""
        hit = set()
        for t in self.tracks:
            tm = (t["t0"] + t["t1"]) / 2
            x = t["a"] + t["v"] * tm
            ok = False
            for i, s in enumerate(self.segs):
                if s["sec"] != t["part"] or (s["d"] > 0) != (t["v"] > 0) or (s["a"] > 0) != (t["h"] > 0):
                    continue
                if not float(s["tb"]) - 0.05 <= tm <= float(s["te"]) + 0.05:
                    continue
                if abs(float(s["x0"]) + s["d"] * float(s["c"]) * (tm - float(s["tb"])) - x) > 0.08:
                    continue
                if abs(abs(float(s["a"])) - abs(t["h"])) > 0.15:
                    continue
                ok = True
                hit.add(i)
            if not ok:
                raise Mismatch(f"a drawn pulse at x={x:.2f}, t={tm:.2f} is not in the re-trace")
        for i, s in enumerate(self.segs):
            if s["tb"] <= self.tl - Fr(1, 2) and i not in hit:
                raise Mismatch(f"re-traced pulse {i} (t={float(s['tb']):.2f}) is never seen")


# ---------------------------------------------------------------- own tracer (depth first)

def trace(M, *, k=Fr(1), ends=None, launches=None, swap=False, uniform=False, length=None, horizon=Fr(11)):
    if uniform:
        parts = [(0, M.L, False)]
    else:
        parts = [(lo, hi, (not hv) if swap else hv) for lo, hi, hv in M.parts]
    if length is not None:
        parts = [(0, length, False)]
    Lx = Fr(parts[-1][1])
    speeds = [M.c * k / (M.ratio if hv else 1) for _, _, hv in parts]
    ends = ends or M.ends
    launches = launches if launches is not None else M.launches
    home = M.home
    out = []

    def where(x, d):
        for i, (lo, hi, _) in enumerate(parts):
            if (lo <= x < hi) if d > 0 else (lo < x <= hi):
                return i
        raise Mismatch("pulse outside the chain")

    def go(t, x, d, a, org, far, par, how):
        if len(out) > 2000:
            raise Mismatch("runaway trace")
        i = where(x, d)
        lo, hi = Fr(parts[i][0]), Fr(parts[i][1])
        c = speeds[i]
        x1 = hi if d > 0 else lo
        t1 = t + abs(x1 - x) / c
        me = len(out)
        out.append({"tb": t, "te": t1, "x0": x, "x1": x1, "d": d, "a": a, "c": c, "sec": i, "org": org, "far": far,
                    "par": par, "how": how})
        if t1 > horizon:
            return
        if x1 == 0 or x1 == Lx:
            side = "left" if x1 == 0 else "right"
            go(t1, x1, -d, a * (1 if ends[side] == "free" else -1), org, far or (home is not None and side != home),
               me, "bounce")
            return
        c2 = speeds[i + 1] if d > 0 else speeds[i - 1]
        for aa, dd, hw in ((a * (c2 - c) / (c2 + c), -d, "back"), (a * 2 * c2 / (c + c2), d, "on")):
            if abs(aa) >= CUT:
                go(t1, x1, dd, aa, org, far, me, hw)

    for n, g in enumerate(launches):
        x = Fr(0) if g["end"] == "left" else Lx
        go(Fr(g["f"], FPS), x, 1 if g["end"] == "left" else -1, Fr(g["sign"]) * (1 if g["big"] else SMALL),
           n, False, -1, "launch")
    return out


# ---------------------------------------------------------------- queries

def _model(scene) -> Model:
    return _cached(json.dumps(scene, sort_keys=True))


@functools.lru_cache(maxsize=16)
def _cached(key):
    return Model(render.track(json.loads(key)))


def _ex(M, side, L=None):
    return Fr(0) if side == "left" else Fr(L if L is not None else M.L)


def _arr(M, segs, x, tl=None):
    tl = M.tl if tl is None else tl
    return sorted((s for s in segs if s["x1"] == x and s["te"] <= tl), key=lambda s: s["te"])


def _first_seg(segs, org=0):
    return next(s for s in segs if s["org"] == org and s["how"] == "launch")


def _kid(segs, s, how):
    i = segs.index(s)
    return next((z for z in segs if z["par"] == i and z["how"] == how), None)


def _home_hits(M, segs, org=0, tl=None, hx=None):
    hx = _ex(M, M.home) if hx is None else hx
    return [s for s in _arr(M, segs, hx, tl) if s["org"] == org and s["far"]]


def _first_of(M, side):
    return next(n for n, g in enumerate(M.launches) if g["end"] == side)


def _cross(segs):
    out = []
    for i, a in enumerate(segs):
        for b in segs[i + 1:]:
            if a["sec"] != b["sec"] or a["d"] == b["d"] or a["org"] == b["org"]:
                continue
            # meeting: positions equal; both move at the same speed in one section
            t = (b["x0"] - a["x0"] + a["d"] * a["c"] * a["tb"] - b["d"] * b["c"] * b["tb"]) / (a["d"] * a["c"] - b["d"] * b["c"])
            if max(a["tb"], b["tb"]) < t < min(a["te"], b["te"]):
                out.append((t, a["x0"] + a["d"] * a["c"] * (t - a["tb"]), a["org"], b["org"]))
    return sorted(out, key=lambda z: z[0])


def _meets(M, segs, ka, kb, tl=None):
    tl = M.tl if tl is None else tl
    return [z for z in _cross(segs) if {z[2], z[3]} == {ka, kb} and z[0] <= tl]


def _active(segs, f):
    t = Fr(f, FPS)
    return [s for s in segs if s["tb"] <= t <= s["te"]]


def _frames(k):
    lo = -(-k * FPS // 10)
    hi = -(-(k + 1) * FPS // 10) - 1
    return (lo + hi) // 2


def _num(v):
    return f"{float(v):g}"


def answer(key: str, p: dict, scene: dict):
    try:
        return _answer(key, p, scene)
    except (Mismatch, StopIteration, RuntimeError, IndexError, KeyError, ValueError, ZeroDivisionError):
        return None


def _answer(key, p, scene):
    M = _model(scene)
    S = M.segs
    far = M.far
    fx = None if far is None else _ex(M, far)
    first = _first_seg(S)
    if key == "far_kind":
        return HARDWARE[M.ends[far]]
    if key == "first_shape":
        g = M.launches[_first_of(M, p["side"]) if "side" in p else 0]
        if M.sized:
            return _word(g["sign"], g["big"], True)
        return _word(g["sign"], True, False)
    if key == "n_launch":
        return len(M.launches)
    if key == "change_mark":
        if len(M.parts) == 2:
            return str(M.parts[0][1])
        lo, hi, _ = next(pt for pt in M.parts if pt[2])
        return f"{lo} and {hi}"
    if key == "first_sender":
        fl, fr = (min(g["f"] for g in M.launches if g["end"] == s) for s in ("left", "right"))
        return "both at the same moment" if fl == fr else ("the left paddle" if fl < fr else "the right paddle")
    if key == "end_count":
        return M.count(M.nf - 1)
    if key == "far_bounce":
        return "upside down" if M.ends[far] == "clamped" else "the same way up"
    if key == "first_back":
        s = _home_hits(M, S)[0]
        return "the same way up" if s["a"] * M.launches[0]["sign"] > 0 else "upside down"
    if key == "second_back":
        s = _home_hits(M, S)[1]
        return "the same way up" if s["a"] * M.launches[0]["sign"] > 0 else "upside down"
    if key in ("junction_split", "trans_taller", "amp_ratio", "refl_ratio", "first_to_end", "junction_return"):
        back, on = _kid(S, first, "back"), _kid(S, first, "on")
        if key == "junction_split":
            return SPLIT[0] if back["a"] * first["a"] < 0 else SPLIT[1]
        if key == "trans_taller":
            return "taller" if abs(on["a"]) > abs(first["a"]) else "shorter"
        if key == "amp_ratio":
            return RATIO[abs(on["a"] / first["a"])]
        if key == "refl_ratio":
            return RATIO[abs(back["a"] / first["a"])]
        hx = _ex(M, M.home)
        if key == "junction_return":
            return _clock(back["te"])
        ta = back["te"] if back["x1"] == hx and back["te"] <= M.tl else None
        fa = _arr(M, S, fx)
        tb = fa[0]["te"] if fa else None
        if ta is not None and (tb is None or ta < tb):
            return "the part that bounced back reaches the paddle first"
        return f"the part that went on reaches the {far} end first"
    if key == "slow_part":
        lo, hi, _ = next(pt for pt in M.parts if pt[2])
        return f"between marks {lo} and {hi}"
    if key == "order_back":
        bounced = sorted((s for s in S if s["how"] == "bounce" and s["far"] and s["x0"] == fx
                          and S[s["par"]]["te"] <= M.tl and not S[s["par"]]["far"]), key=lambda s: s["tb"])
        return ", then ".join(_short(1 if s["a"] > 0 else -1, abs(s["a"]) == 1, M.sized) for s in bounced)
    if key in ("which_first_far", "cross_time"):
        def reach(side):
            k = _first_of(M, side)
            x = _ex(M, "right" if side == "left" else "left")
            return next(s for s in _arr(M, S, x) if s["org"] == k)
        if key == "cross_time":
            return _clock(reach(p["side"])["te"])
        ta, tb = reach("left")["te"], reach("right")["te"]
        if ta == tb:
            return "both at the same moment"
        return "the pulse from the left paddle" if ta < tb else "the pulse from the right paddle"
    if key in ("meet_side", "meet_mark", "second_meet"):
        ms = _meets(M, S, _first_of(M, "left"), _first_of(M, "right"))
        x = ms[1 if key == "second_meet" else 0][1]
        if key != "meet_side":
            return str(round(x))
        half = Fr(M.L, 2)
        if abs(x - half) < Fr(3, 20):
            return "in the middle of the chain"
        return "nearer the left paddle" if x < half else "nearer the right paddle"
    if key == "bigger_sender":
        bl, br = (M.launches[_first_of(M, s)]["big"] for s in ("left", "right"))
        return "the same height" if bl == br else ("the left paddle's" if bl else "the right paddle's")
    if key == "dir_at":
        (s,) = _active(S, _frames(p["k"]))
        return "toward the right end" if s["d"] > 0 else "toward the left end"
    if key == "speed":
        return _num(M.c)
    if key == "speed_ratio":
        return {2: "about half", 3: "about a third"}[M.ratio]
    if key == "first_far_time":
        return _clock(_arr(M, S, fx)[0]["te"])
    if key == "return_time":
        return _clock(_home_hits(M, S)[0]["te"])
    if key == "count_at":
        return M.count(_frames(p["k"]))
    if key == "toward_at":
        return sum(1 for s in _active(S, _frames(p["k"])) if (s["d"] < 0) == (p["side"] == "left"))
    if key == "launch_gap":
        return f"{(M.launches[1]['f'] - M.launches[0]['f']) / FPS:g} s"
    if key == "spacing":
        v = M.c * Fr(M.launches[1]["f"] - M.launches[0]["f"], FPS)
        return f"{round(2 * v) / 2:g} marks"
    if key == "far_hits":
        return len(_arr(M, S, fx))
    if key == "paddle_hits":
        return len(_arr(M, S, _ex(M, p["side"])))
    if key == "kth_far":
        s = _arr(M, S, fx)[p["k"] - 1]
        return _word(1 if s["a"] > 0 else -1, abs(s["a"]) == 1, True)
    if key == "time_kth_far":
        return _clock(_arr(M, S, fx)[p["k"] - 1]["te"])
    if key == "meet_count":
        return sum(1 for z in _cross(S) if z[0] <= M.tl)
    # ---- what-ifs
    if key == "cf_switch_far":
        ends = dict(M.ends)
        ends[far] = "free" if ends[far] == "clamped" else "clamped"
        s = _home_hits(M, trace(M, ends=ends), tl=Fr(10**6))[0]
        return "the same way up" if s["a"] * M.launches[0]["sign"] > 0 else "upside down"
    if key == "cf_double_speed":
        return len(_arr(M, trace(M, k=Fr(2)), fx))
    if key == "cf_uniform":
        return _clock(_home_hits(M, trace(M, uniform=True), tl=Fr(10**6))[0]["te"])
    if key == "cf_heavy_swap":
        segs = trace(M, swap=True)
        f0 = _first_seg(segs)
        back = _kid(segs, f0, "back")
        return SPLIT[0] if back["a"] * f0["a"] < 0 else SPLIT[1]
    if key == "cf_shorter":
        m = p["m"]
        length = m if M.home == "left" else M.L - m
        lm = Model.__new__(Model)
        lm.__dict__.update(M.__dict__)
        lm.parts, lm.home, lm.L = [(0, length, False)], "left", length
        lm.ends = {"left": "paddle", "right": M.ends[far]}
        ls = [dict(M.launches[0], end="left")]
        return _clock(_home_hits(lm, trace(lm, launches=ls), tl=Fr(11), hx=Fr(0))[0]["te"])
    if key == "cf_slow_far":
        return _clock(_arr(M, trace(M, k=Fr(1, 2)), fx, tl=Fr(11))[0]["te"])
    if key == "cf_later":
        k = _first_of(M, p["side"])
        other = _first_of(M, "right" if p["side"] == "left" else "left")
        ls = [dict(g) for g in M.launches]
        ls[k]["f"] += int(Fr(p["dt"]) * FPS)
        return str(round(_meets(M, trace(M, launches=ls), k, other, tl=Fr(11))[0][1]))
    if key == "cf_end_swap":
        side = p["side"]
        other = "right" if side == "left" else "left"
        k = _first_of(M, side)
        segs = trace(M, ends={side: "paddle", other: p["kind"]})
        s = next(z for z in sorted(segs, key=lambda z: z["te"]) if z["org"] == k and z["x1"] == _ex(M, side))
        return "the same way up" if s["a"] * M.launches[k]["sign"] > 0 else "upside down"
    if key == "cf_drop_last":
        return len(_arr(M, trace(M, launches=M.launches[:-1]), fx))
    if key == "cf_double_meet":
        return sum(1 for z in _cross(trace(M, k=Fr(2))) if z[0] <= M.tl)
    return None
