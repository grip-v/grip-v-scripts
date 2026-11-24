"""Independent answers for stacking_collapse.

Reads only `render.track(scene)`: the drawn outline of every block and of
the plank on every frame, the pillars and the grid unit. It never reads the
actions, the event list or sim.play. From the outlines it finds when each
block comes to rest (landing), slides away sideways (pulled out), drops onto
a lower block, or starts to turn (falls); what each resting block sits on
(its bottom edge on another top edge or a pillar top); and the grid width of
each block. For the what-ifs it rebuilds the sequence of drops / pulls from
those observations and re-simulates it with its own rule, in px: at every
resting surface the turning effect sum(area * (x - edge)) of everything
above is taken about the left and the right edge of the contact; a negative
sum about the left edge tips the part over to the left, a positive one about
the right edge to the right. It also checks that this rule reproduces every
collapse seen in the clip.
"""
from __future__ import annotations
import math
import sys
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render  # noqa: E402

FPS = 24
TOL = 0.6            # px: edges closer than this touch


def clock_text(f: int) -> str:
    k = (10 * f) // FPS
    return f"{k // 10}.{k % 10} s"


def frames_of_reading(k: int) -> int:
    """First frame whose clock shows k tenths."""
    f = 0
    while (10 * f) // FPS < k:
        f += 1
    return f


def half_s(frames) -> str | None:
    s = Fr(frames, FPS)
    h = round(s * 2) / 2
    if abs(s - Fr(h)) > Fr(1, 10) or h <= 0:
        return None
    return f"{h:g} s"


class Model:
    def __init__(self, tr):
        P = np.asarray(tr["poly"], dtype=float)
        self.P = P
        self.n, self.nb = P.shape[0], P.shape[1]
        self.unit = float(tr["unit"])
        self.pillars = sorted([tuple(p) for p in tr["pillars"]])
        self.names = [f"the {b['pattern']} {b['colour']} block" for b in tr["blocks"]]
        self.idx = {nm: j for j, nm in enumerate(self.names)}
        self.pattern = [b["pattern"] for b in tr["blocks"]]
        self.vis = ~np.isnan(P[:, :, 0, 0])
        self.C = P.mean(axis=2)
        self.ang = np.arctan2(P[:, :, 1, 1] - P[:, :, 0, 1], P[:, :, 1, 0] - P[:, :, 0, 0])
        self.dims = []
        for j in range(self.nb):
            f = int(np.argmax(self.vis[:, j]))
            q = P[f, j]
            self.dims.append((math.hypot(*(q[1] - q[0])), math.hypot(*(q[3] - q[0]))))
        PL = tr.get("plank")
        self.plank = None if PL is None else np.asarray(PL, dtype=float)
        if self.plank is not None:
            self.scenario = "plank"
        elif len(self.pillars) == 2:
            self.scenario = "twin"
        elif self.nb and self.vis[0].any() and any(self._rest(0, j) for j in range(self.nb)):
            self.scenario = "pull_out"
        else:
            self.scenario = "tower"
        self._events()

    # ------------------------------------------------------------ geometry per frame
    def _same(self, f, j):
        return (0 < f < self.n and self.vis[f, j] and self.vis[f - 1, j]
                and np.allclose(self.P[f, j], self.P[f - 1, j], atol=1e-6))

    def _rest(self, f, j):
        if not self.vis[f, j] or abs(self.ang[f, j]) > 1e-9:
            return False
        return self._same(f, j) or (f + 1 < self.n and self._same(f + 1, j))

    def box(self, f, j):
        q = self.P[f, j]
        return q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()

    def plank_rest(self, f):
        return self.plank is not None and not np.isnan(self.plank[f, 0, 0]) and abs(
            math.atan2(self.plank[f, 1, 1] - self.plank[f, 0, 1], self.plank[f, 1, 0] - self.plank[f, 0, 0])) < 1e-9

    def support(self, f, j):
        """What resting block j stands on at frame f: a block index, ('pillar', b) or 'plank' (None if nothing)."""
        x0, _, x1, yb = self.box(f, j)
        for k in range(self.nb):
            if k != j and self._rest(f, k):
                a0, at, a1, _ = self.box(f, k)
                if abs(at - yb) < TOL and min(x1, a1) - max(x0, a0) > TOL:
                    return k
        if self.plank_rest(f):
            q = self.plank[f]
            if abs(q[:, 1].min() - yb) < TOL and min(x1, q[:, 0].max()) - max(x0, q[:, 0].min()) > TOL:
                return "plank"
        for b, (p0, pt, p1) in enumerate(self.pillars):
            if abs(pt - yb) < TOL and min(x1, p1) - max(x0, p0) > TOL:
                return ("pillar", b)
        return None

    def ground(self, f, j):
        """Follow the supports down: ('pillar', b), 'plank', or None."""
        seen = set()
        it = j
        while isinstance(it, (int, np.integer)):
            if it in seen:
                return None
            seen.add(it)
            it = self.support(f, it)
        return it

    def standing(self, f, base=None):
        out = []
        for j in range(self.nb):
            if self._rest(f, j):
                g = self.ground(f, j)
                if g is not None and (base is None or g == ("pillar", base)):
                    out.append(j)
        return out

    def item_name(self, it):
        if isinstance(it, (int, np.integer)):
            return self.names[it]
        if it == "plank":
            return "the plank"
        if self.scenario == "plank":
            return "the support"
        if self.scenario == "twin":
            return ("the left pedestal", "the right pedestal")[it[1]]
        return "the pedestal"

    # ------------------------------------------------------------ events
    def _events(self):
        self.land, self.pull, self.fall_of, self.settle = {}, {}, {}, {}
        self.initial = []
        runs = {}
        for j in range(self.nb):
            rs, f = [], 0
            while f < self.n:
                if self._rest(f, j):
                    g = f
                    while g + 1 < self.n and self._same(g + 1, j) and abs(self.ang[g + 1, j]) < 1e-9:
                        g += 1
                    rs.append((f, g))
                    f = g + 1
                else:
                    f += 1
            runs[j] = rs
            if not rs:
                continue
            if rs[0][0] == 0:
                self.initial.append(j)
            else:
                self.land[j] = rs[0][0]
            for a, b in rs:
                if b + 1 >= self.n:
                    continue
                if not self.vis[b + 1, j]:
                    continue
                d = self.C[b + 1, j] - self.C[b, j]
                if abs(self.ang[b + 1, j]) > 1e-9:
                    self.fall_of[j] = b + 1
                elif abs(d[1]) < 1e-9 and abs(d[0]) > 0:
                    self.pull[j] = b + 1
                elif d[1] > 0 and abs(d[0]) < 1e-9:
                    self.settle.setdefault(j, []).append(b + 1)
        self.runs = runs
        groups = {}
        for j, f in self.fall_of.items():
            groups.setdefault(f, []).append(j)
        self.plank_fall = None
        if self.plank is not None:
            for f in range(1, self.n):
                if not np.isnan(self.plank[f, 0, 0]) and not self.plank_rest(f):
                    self.plank_fall = f
                    groups.setdefault(f, [])
                    break
        self.falls = []
        for f in sorted(groups):
            js = sorted(groups[f], key=lambda j: -self.box(f - 1, j)[3])        # bottom first
            if self.plank_fall == f:
                a = self.plank[f]
                side = "right" if math.atan2(a[1, 1] - a[0, 1], a[1, 0] - a[0, 0]) > 0 else "left"
                frm = ("pillar", 0)
            else:
                side = "right" if self.ang[f, js[0]] > 0 else "left"
                frm = self.support(f - 1, js[0])
            gone = f
            while gone < self.n and (any(self.vis[gone, j] for j in js) or (
                    self.plank_fall == f and not np.isnan(self.plank[gone, 0, 0]))):
                gone += 1
            self.falls.append({"f": f, "blocks": js, "side": side, "from": frm, "gone": gone,
                               "base": self.ground(f - 1, js[0]) if js else ("pillar", 0)})
        self.order = sorted(self.land, key=lambda j: self.land[j])
        self.pulled = sorted(self.pull, key=lambda j: self.pull[j])
        steps = [(self.land[j], "drop", j) for j in self.land] + [(self.pull[j], "pull", j) for j in self.pull]
        self.steps = sorted(steps)
        self.actions = []
        for f, kind, j in self.steps:
            if kind == "drop":
                g = self.ground(f, j)
                tgt = "plank" if g == "plank" else (g[1] if g else None)
                self.actions.append({"kind": "drop", "block": j, "x": float(self.C[f, j, 0]), "to": tgt})
            else:
                self.actions.append({"kind": "pull", "block": j})
        self.init_stack = []
        if self.initial:
            js = sorted(self.initial, key=lambda j: -self.box(0, j)[3])
            self.init_stack = [(j, float(self.C[0, j, 0])) for j in js]
        res = self.simulate(self.actions)
        seen = []
        for e in self.falls:
            prev = [s for s in self.steps if s[0] < e["f"]]
            seen.append((len(prev) - 1, len(e["blocks"]), e["side"]))
        self.consistent = res is not None and [(i, k, s) for i, k, s, _ in res["falls"]] == seen

    # ------------------------------------------------------------ own re-simulation (px)
    def simulate(self, actions, widen=None):
        """Replay drops / pulls with the torque rule. None if any contact is under one
        grid square or any turning effect is within a quarter square of the edge."""
        u = self.unit
        pil = [list(p) for p in self.pillars]
        if widen is not None:
            pil[widen][0] -= u
            pil[widen][2] += u
        span = lambda j, x: (x - self.dims[j][0] / 2, x + self.dims[j][0] / 2)
        mass = lambda j: self.dims[j][0] * self.dims[j][1]
        stacks = [[] for _ in pil]
        if self.init_stack:
            stacks[0] = list(self.init_stack)
        on_plank = []
        plank_up = self.plank is not None
        falls = []

        def test(b, st):
            for i in range(len(st)):
                j, x = st[i]
                l, r = span(j, x)
                if i == 0:
                    bl, br = pil[b][0], pil[b][2]
                else:
                    bl, br = span(*st[i - 1])
                lo, hi = max(l, bl), min(r, br)
                if hi - lo < u - 1e-6:
                    return "bad", None
                M = sum(mass(k) for k, _ in st[i:])
                tl = sum(mass(k) * (xk - lo) for k, xk in st[i:])
                trr = sum(mass(k) * (xk - hi) for k, xk in st[i:])
                if min(abs(tl), abs(trr)) / M < u / 4 - 1e-6:
                    return "bad", None
                if tl < 0:
                    return i, "left"
                if trr > 0:
                    return i, "right"
            return None, None

        def test_plank():
            q = self.plank[0]
            px0, px1 = q[:, 0].min(), q[:, 0].max()
            pm = (px1 - px0) * (q[:, 1].max() - q[:, 1].min())
            pc = (px0 + px1) / 2
            lo, hi = pil[0][0], pil[0][2]
            items = [(pm, pc)] + [(mass(j), x) for j, x in on_plank]
            M = sum(m for m, _ in items)
            tl = sum(m * (x - lo) for m, x in items)
            trr = sum(m * (x - hi) for m, x in items)
            if min(abs(tl), abs(trr)) / M < u / 4 - 1e-6:
                return "bad"
            return "left" if tl < 0 else "right" if trr > 0 else None

        if self.init_stack and test(0, stacks[0])[0] is not None:
            return None
        if plank_up and test_plank() is not None:
            return None
        for i, a in enumerate(actions):
            j = a["block"]
            if a["kind"] == "drop" and a["to"] == "plank":
                if not plank_up:
                    return None
                q = self.plank[0]
                l, r = span(j, a["x"])
                if l < q[:, 0].min() + u / 2 - 1e-6 or r > q[:, 0].max() - u / 2 + 1e-6:
                    return None
                for k, xk in on_plank:
                    kl, kr = span(k, xk)
                    if l < kr + u / 2 - 1e-6 and kl < r + u / 2 - 1e-6:
                        return None
                on_plank.append((j, a["x"]))
                s = test_plank()
                if s == "bad":
                    return None
                if s:
                    falls.append((i, len(on_plank), s, 0))
                    on_plank, plank_up = [], False
                continue
            if a["kind"] == "drop":
                b = a["to"]
                if b is None:
                    return None
                st = stacks[b]
                st.append((j, a["x"]))
            else:
                st = stacks[0]
                ks = [t for t, (k, _) in enumerate(st) if k == j]
                if not ks:
                    return None
                del st[ks[0]]
                b = 0
                if not st:
                    continue
            k, side = test(b, st)
            if k == "bad":
                return None
            if k is not None:
                falls.append((i, len(st) - k, side, b))
                del st[k:]
        standing = sum(len(s) for s in stacks) + len(on_plank)
        return {"falls": falls, "standing": standing, "plank_up": plank_up}


def _model(scene) -> Model:
    return Model(render.track(scene))


def answer(key: str, p: dict, scene: dict):
    try:
        return _answer(key, p, _model(scene))
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def _answer(key, p, M: Model):
    I = M.idx
    last = M.n - 1
    if key == "n_dropped":
        return len(M.land)
    if key == "n_init":
        return len(M.initial)
    if key == "n_pulled":
        return len(M.pull)
    if key == "n_falls":
        return len(M.falls)
    if key == "plank_tips":
        return "yes" if M.plank_fall is not None else "no"
    if key == "end_count":
        return len(M.standing(last))
    if key == "top_end":
        st = M.standing(last, p["base"])
        return M.names[min(st, key=lambda j: M.box(last, j)[1])]
    if key == "first_land":
        return M.names[M.order[0]]
    if key == "first_pull":
        return M.names[M.pulled[0]]
    if key == "count_look":
        return sum(1 for q in M.pattern if q == p["pattern"])
    if key == "land_order":
        return min((p["x"], p["y"]), key=lambda nm: M.land[I[nm]])
    if key == "pull_order":
        return min((p["x"], p["y"]), key=lambda nm: M.pull[I[nm]])
    if key == "on_what":
        j = I[p["x"]]
        return M.item_name(M.support(M.land[j], j))
    if key == "settle_on":
        j = I[p["x"]]
        f = M.pull[j]
        above = [k for k in range(M.nb) if M._rest(f - 1, k) and M.support(f - 1, k) == j]
        k = above[0]
        g = min(t for t in M.settle[k] if t > f)
        a = [r for r in M.runs[k] if r[0] >= g][0][0]
        return M.item_name(M.support(a, k))
    if key == "fall_side":
        return f"to the {M.falls[p['k']]['side']}"
    if key == "cause":
        f = M.falls[p["k"]]["f"]
        return M.names[max((j for j in M.land if M.land[j] < f), key=lambda j: M.land[j])]
    if key == "cause_pull":
        f = M.falls[0]["f"]
        return M.names[max((j for j in M.pull if M.pull[j] < f), key=lambda j: M.pull[j])]
    if key == "wider":
        return max((p["x"], p["y"]), key=lambda nm: M.dims[I[nm]][0])
    if key == "twin_tower":
        j = I[p["x"]]
        return M.item_name(M.ground(M.land[j], j))
    if key == "twin_first_fall":
        return M.item_name(M.falls[0]["base"]) if M.falls else "neither"
    if key == "twin_more":
        a, b = len(M.standing(last, 0)), len(M.standing(last, 1))
        return "both the same" if a == b else "the left pedestal" if a > b else "the right pedestal"
    if key == "plank_land_side":
        j = I[p["x"]]
        p0, _, p1 = M.pillars[0]
        return "left of the support" if M.C[M.land[j], j, 0] < (p0 + p1) / 2 else "right of the support"
    if key == "above":
        j = I[p["x"]]
        return [M.names[k] for k in M.order if M.support(M.land[k], k) == j][0]
    if key == "land_time":
        return clock_text(M.land[I[p["x"]]])
    if key == "pull_time":
        return clock_text(M.pull[I[p["x"]]])
    if key == "count_at":
        return len(M.standing(frames_of_reading(p["k"]), p["base"]))
    if key == "n_fall":
        return len(M.falls[p["k"]]["blocks"])
    if key == "gap":
        ev = M.pull if M.scenario == "pull_out" else M.land
        return half_s(ev[I[p["y"]]] - ev[I[p["x"]]])
    if key == "width":
        return int(round(M.dims[I[p["x"]]][0] / M.unit))
    if key == "max_count":
        return max(len(M.standing(f, p["base"])) for f in range(M.n))
    if key == "total_fallen":
        return sum(len(e["blocks"]) for e in M.falls)
    if key == "break_from":
        return M.item_name(M.falls[p["k"]]["from"])
    if key == "kth_land":
        return M.names[M.order[p["k"]]]
    if key == "kth_pull":
        return M.names[M.pulled[p["k"]]]
    if key == "count_after":
        j = I[p["x"]]
        f = M.land[j]
        base = M.ground(f, j)[1] if M.scenario == "twin" else None
        return len(M.standing(f, base))
    if key == "pull_fall_n":
        f = M.pull[I[p["x"]]]
        nxt = min([M.pull[k] for k in M.pull if M.pull[k] > f] + [M.n])
        fl = [e for e in M.falls if f < e["f"] < nxt]
        return len(fl[0]["blocks"]) if fl else 0
    if key == "between":
        a = frames_of_reading(p["k1"] + 1) - 1
        b = frames_of_reading(p["k2"])
        ev = M.pull if M.scenario == "pull_out" else M.land
        return sum(1 for f in ev.values() if a < f < b)
    if key == "which_fell":
        fell = {M.names[j] for e in M.falls for j in e["blocks"]}
        got = [c for c in p["cands"] if c in fell]
        return got[0] if len(got) == 1 else None
    # ---- what-ifs: own re-simulation of the observed sequence
    if not M.consistent:
        return None
    acts = M.actions
    if key in ("cf_skip_any", "cf_skip_end", "cf_no_pull"):
        j = I[p["x"]]
        r = M.simulate([a for a in acts if a["block"] != j])
        if r is None:
            return None
        if key == "cf_skip_any":
            return "yes" if r["falls"] else "no"
        return r["standing"]
    if key in ("cf_shift", "cf_shift_n"):
        j = I[p["x"]]
        i = [t for t, a in enumerate(acts) if a["block"] == j][0]
        alt = [dict(a) for a in acts[:i + 1]]
        alt[i]["x"] += p["d"] * M.unit
        r = M.simulate(alt)
        if r is None:
            return None
        fl = [x for x in r["falls"] if x[0] == i]
        if key == "cf_shift":
            return "yes" if fl else "no"
        return fl[0][1] if fl else 0
    if key == "cf_swap":
        i = [t for t, a in enumerate(acts) if a["block"] == I[p["x"]]][0]
        if acts[i + 1]["block"] != I[p["y"]]:
            return None
        r = M.simulate(acts[:i] + [acts[i + 1], acts[i]] + acts[i + 2:])
        return None if r is None else r["standing"]
    if key == "cf_pull_other":
        alt = [dict(a) for a in acts]
        for a in alt:
            if a["block"] == I[p["x"]]:
                a["block"] = I[p["y"]]
        r = M.simulate(alt)
        return None if r is None else sum(x[1] for x in r["falls"])
    if key == "cf_wide_base":
        r = M.simulate(acts, widen=p["base"])
        if r is None:
            return None
        if M.scenario == "plank":
            return "yes" if r["falls"] else "no"
        return "yes" if any(x[3] == p["base"] for x in r["falls"]) else "no"
    if key == "cf_mirror":
        j = I[p["x"]]
        p0, _, p1 = M.pillars[0]
        alt = [dict(a) for a in acts]
        for a in alt:
            if a["block"] == j:
                a["x"] = (p0 + p1) - a["x"]
        r = M.simulate(alt)
        return None if r is None else ("yes" if r["falls"] else "no")
    return None
