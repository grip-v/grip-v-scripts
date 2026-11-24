"""Independent answers for conveyor_belt_sort.

Reads only `render.track(scene)`: how each object is drawn (colour, shape,
pattern, radius), its centre on every frame, and the drawn geometry (tunnel
mouth, belt, gate, trays and their slots). Never the rule, the bins, the
slots, the frame counts or the keyframes in the scene.

From the track it finds, per object, the frame it comes out of the tunnel,
the frame it reaches the gate, the frame it comes to rest and the tray it
rests in; the belt speed from the motion; lane lengths from the geometry
(checked against every object that lands); the sorting rule by trying every
simple rule (one property -> bin, or taking turns) on the sorted objects -
exactly one must fit; and a separate re-simulation for the what-ifs.
"""
from __future__ import annotations
import math
import statistics
from fractions import Fraction as Fr

import render

FPS = 24
BIN = {"upper": "the upper bin", "middle": "the middle bin", "lower": "the lower bin"}
STATES = ["still inside the tunnel", "riding on the belt", "sliding toward a bin", "already resting in a bin"]
SIZE_WORDS = {1: ["medium"], 2: ["small", "large"], 3: ["small", "medium", "large"]}
EPS = 1e-6


def clock(f) -> str:
    k = (10 * int(f)) // FPS
    return f"{k // 10}.{k % 10} s"


def frames_showing(k):
    return -(-k * FPS // 10), -(-(k + 1) * FPS // 10) - 1


def half_s(frames):
    s = Fr(frames, FPS)
    h = round(s * 2) / 2
    if abs(s - Fr(h)) > Fr(1, 10) or h <= 0:
        return None
    return f"{h:g} s"


class Model:
    def __init__(self, tr):
        self.tr = tr
        P = tr["pos"]
        self.n = tr["n_frames"]
        self.d = tr["dir"]
        self.N = P.shape[1]
        mouth, gx = tr["mouth"], tr["gate"][0]
        # what each object looks like
        rs = sorted({o["r"] for o in tr["objects"]})
        words = SIZE_WORDS[len(rs)]
        self.look = []
        for o in tr["objects"]:
            self.look.append({"size": words[rs.index(o["r"])], "colour": o["colour"], "shape": o["shape"],
                              "pattern": o["pattern"], "ext": 1.22 * o["r"]})
        self.names = [f"the {lk['pattern']} {lk['colour']} {lk['shape']}" for lk in self.look]
        if len(set(self.names)) != self.N:
            raise ValueError("two objects look the same")
        self.sizes = words
        # events from the drawn centres
        self.out, self.gate, self.land, self.bin, self.first = [], [], [], [], []
        for j in range(self.N):
            xs = P[:, j, 0]
            seen = [f for f in range(self.n) if not math.isnan(xs[f])]
            self.first.append(seen[0] if seen else None)
            self.out.append(next((f for f in seen if (xs[f] - mouth) * self.d >= -EPS), None))
            self.gate.append(next((f for f in seen if (xs[f] - gx) * self.d >= -EPS), None))
            last = P[self.n - 1, j]
            b = self._tray(last) if seen else None
            if b is not None:
                f = self.n - 1
                while f > 0 and abs(P[f - 1, j, 0] - last[0]) < EPS and abs(P[f - 1, j, 1] - last[1]) < EPS:
                    f -= 1
                self.land.append(f)
            else:
                self.land.append(None)
            self.bin.append(b)
        if any(o is None for o in self.out):
            raise ValueError("an object never comes out of the tunnel")
        self.order = sorted(range(self.N), key=lambda j: (self.out[j], j))
        if self.order != list(range(self.N)):
            raise ValueError("objects are not listed in the order they come out")
        self.done = [j for j in range(self.N) if self.land[j] is not None]
        self.pend = [j for j in range(self.N) if self.gate[j] is None]
        if not self.done or not self.pend or len(self.done) + len(self.pend) != self.N:
            raise ValueError("objects in transit at the end")
        # belt speed from the motion along the belt (px per frame, half-pixel grid)
        steps = []
        for j in range(self.N):
            for f in range(self.out[j], (self.gate[j] if self.gate[j] is not None else self.n) - 1):
                steps.append(abs(P[f + 1, j, 0] - P[f, j, 0]))
        self.v = Fr(round(2 * statistics.median(steps)), 2)
        self.belt = self._frames(abs(gx - mouth))
        for j in range(self.N):
            if self.gate[j] is not None and self.gate[j] - self.out[j] != self.belt:
                raise ValueError("belt time differs between objects")
        self.bins = [b for b in ("upper", "middle", "lower") if b in tr["bins"]]
        # lanes: frames from the gate to rest in each slot, from the drawn geometry
        self.lane = {}
        for b in self.bins:
            g = tr["bins"][b]
            sx, sy = g["start"]
            c = self._frames(math.hypot(sx - gx, sy - tr["gate"][1]))
            self.lane[b] = [c + self._frames(math.hypot(px - sx, py - sy)) for px, py in g["slots"]]
        # check: landed objects fill their trays from slot 0 in the order they reach the gate
        for b in self.bins:
            got = sorted((j for j in self.done if self.bin[j] == b), key=lambda j: self.gate[j])
            for k, j in enumerate(got):
                sx, sy = tr["bins"][b]["slots"][k]
                if abs(P[self.n - 1, j, 0] - sx) > 0.01 or abs(P[self.n - 1, j, 1] - sy) > 0.01:
                    raise ValueError("objects do not fill the tray in order")
                if self.land[j] - self.gate[j] != self.lane[b][k]:
                    raise ValueError(f"lane time mismatch for {self.names[j]}")
        self.rule = self._induce()
        # where the pending objects will go
        self.future = self.simulate()
        for j in self.done:
            if self.future[j][2] != self.bin[j] or self.future[j][1] != self.land[j]:
                raise ValueError("re-simulation does not reproduce the clip")

    def _tray(self, p):
        for b, g in self.tr["bins"].items():
            x0, y0, x1, y1 = g["rect"]
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                return b
        return None

    def _frames(self, length) -> int:
        q = length / float(self.v)
        if abs(q - round(q)) > 0.02:
            raise ValueError("a length is not a whole number of frames at the belt speed")
        return int(round(q))

    def _induce(self):
        seq = sorted(self.done, key=lambda j: self.gate[j])
        bins = [self.bin[j] for j in seq]
        fits = []
        for a in ("size", "colour", "shape", "pattern"):
            m = {}
            if all(m.setdefault(self.look[j][a], b) == b for j, b in zip(seq, bins)):
                fits.append(("attr", a, m))
        nb = len(self.bins)
        if len(bins) >= nb and len(set(bins[:nb])) == nb and all(bins[i] == bins[i - nb] for i in range(nb, len(bins))):
            fits.append(("turns", bins[:nb], None))
        if len(fits) != 1:
            raise ValueError(f"{len(fits)} simple rules fit the sorted objects")
        return fits[0]

    def rule_bin(self, look, pos_in_order):
        kind, a, m = self.rule
        if kind == "turns":
            return a[pos_in_order % len(a)]
        return m[look[a]]

    def simulate(self, entries=None, speed=Fr(1), drop=(), flip=False):
        """[(gate, land, bin, slot)] per object, None for dropped ones."""
        entries = self.out if entries is None else entries
        keep = [j for j in range(self.N) if j not in drop]
        order = sorted(keep, key=lambda j: (entries[j], j))
        fill = {b: 0 for b in self.bins}
        out = [None] * self.N
        for pos, j in enumerate(order):
            b = self.rule_bin(self.look[j], pos)
            if flip:
                b = {"upper": "lower", "lower": "upper"}[b]
            k = fill[b]
            fill[b] += 1
            g = Fr(entries[j]) + Fr(self.belt) / speed
            land = g + Fr(self.lane[b][k]) / speed if k < len(self.lane[b]) else None
            out[j] = (g, land, b, k)
        return out

    def obj(self, nm):
        return self.names.index(nm)

    def resting(self, b, f):
        return sum(1 for j in self.done if self.bin[j] == b and self.land[j] <= f)

    def state(self, j, f):
        P = self.tr["pos"]
        if math.isnan(P[f, j, 0]):
            return 0
        x = P[f, j, 0]
        if (x - self.tr["mouth"]) * self.d < 0:
            return None
        if self.gate[j] is None or f < self.gate[j]:
            return 1
        return 2 if self.land[j] is None or f < self.land[j] else 3


def answer(key: str, p: dict, scene: dict):
    try:
        M = Model(render.track(scene))
        return _answer(M, key, p)
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def _answer(M, key, p):
    n = M.n
    if key == "n_objects":
        return M.N
    if key == "in_bin_end":
        got = [nm for nm in p["among"] if M.bin[M.obj(nm)] == p["bin"] and M.land[M.obj(nm)] is not None]
        return got[0] if len(got) == 1 else None
    if key == "count_end":
        return M.resting(p["bin"], n - 1)
    if key == "on_belt_end":
        got = [nm for nm in p["among"] if M.obj(nm) in M.pend]
        return got[0] if len(got) == 1 else None
    if key == "first_out":
        return M.names[M.order[0]]
    if key == "same_bin":
        return "yes" if M.bin[M.obj(p["x"])] == M.bin[M.obj(p["y"])] else "no"
    if key == "land_order":
        i, j = M.obj(p["x"]), M.obj(p["y"])
        return M.names[i] if M.land[i] < M.land[j] else M.names[j]
    if key == "next_out":
        i = M.obj(p["x"])
        return M.names[M.order[M.order.index(i) + 1]]
    if key == "bin_after":
        i = M.obj(p["x"])
        return BIN[M.bin[M.order[M.order.index(i) + 1]]]
    if key == "rule_value":
        kind, a, m = M.rule
        return BIN[m[p["value"]]] if kind == "attr" and a == p["attr"] else None
    if key == "state_at_land":
        s = M.state(M.obj(p["y"]), M.land[M.obj(p["x"])])
        return None if s is None else STATES[s]
    if key == "first_bin":
        return BIN[M.bin[min(M.done, key=lambda j: M.land[j])]]
    if key == "land_time":
        return clock(M.land[M.obj(p["x"])])
    if key == "gate_time":
        return clock(M.gate[M.obj(p["x"])])
    if key == "count_at":
        return M.resting(p["bin"], frames_showing(p["k"])[0])
    if key == "belt_count_at":
        f = frames_showing(p["k"])[0]
        return sum(1 for j in range(M.N) if M.state(j, f) == 1)
    if key == "belt_time":
        return half_s(M.belt)
    if key == "entry_gap":
        return half_s(M.out[M.obj(p["y"])] - M.out[M.obj(p["x"])])
    if key == "ride_time":
        i = M.obj(p["x"])
        return half_s(M.land[i] - M.gate[i])
    if key == "pending_bin":
        return BIN[M.future[M.obj(p["x"])][2]]
    if key == "final_count":
        return sum(1 for r in M.future if r[2] == p["bin"])
    if key == "pending_land":
        return clock(math.ceil(M.future[M.obj(p["x"])][1]))
    if key == "landed_between":
        f1, f2 = frames_showing(p["k1"])[0], frames_showing(p["k2"])[0]
        return sum(1 for j in M.done if f1 < M.land[j] <= f2)
    if key == "kth_land":
        return M.names[sorted(M.done, key=lambda j: M.land[j])[p["k"] - 1]]
    if key == "last_land_bin":
        return BIN[M.bin[max(M.done, key=lambda j: M.land[j])]]
    if key == "cf_reverse":
        res = M.simulate(flip=True)
        return sum(1 for g, land, b, k in res if b == p["bin"] and land is not None and land <= n - 1)
    if key == "cf_reverse_time":
        return clock(math.ceil(M.simulate(flip=True)[M.obj(p["x"])][1]))
    if key == "count_kind":
        return sum(1 for lk in M.look if lk[p["attr"]] == p["value"])
    if key == "cf_fast":
        res = M.simulate(speed=Fr(2))
        return sum(1 for g, land, b, k in res if land is not None and land <= n - 1)
    if key == "cf_remove_time":
        res = M.simulate(drop=(M.obj(p["x"]),))
        return clock(math.ceil(res[M.obj(p["y"])][1]))
    if key == "cf_remove_bin":
        res = M.simulate(drop=(M.obj(p["x"]),))
        return BIN[res[M.obj(p["y"])][2]]
    if key == "cf_new":
        kind, a, m = M.rule
        if kind == "turns":
            return BIN[a[M.N % len(a)]]
        look = {"size": p["size"] or "medium", "colour": p["colour"], "shape": p["shape"], "pattern": p["pattern"]}
        return BIN[m[look[a]]]
    if key == "cf_delay":
        i = M.obj(p["x"])
        ent = list(M.out)
        ent[i] += p["dt"]
        return clock(math.ceil(M.simulate(entries=ent)[i][1]))
    if key == "cf_swap":
        kind, a, m = M.rule
        return BIN[m[p["value"]]] if kind == "attr" else None
    return None
