"""Independent answers for sandpile_cascade, read from what the renderer draws.

The decoder sees only `render.track(scene)`: the centre and scale of every
grain on every frame, the cells (centre, kind, slot centres) and whether the
board is drawn from the side. It never reads the drops, the timing, the
event list or sim.play. From the drawing it finds:

  * grains at rest: drawn at full size exactly on a slot centre and still
    there on the frame before or after (a falling or hopping grain only
    passes through a slot);
  * landings: a grain that appears after the start and first comes to rest;
  * topples: grains leaving the slots of one cell on the same frame; where
    each one goes (the next cell it rests in, or nowhere: it is lost);
  * the board's wiring from the cell positions alone (side view -> a row;
    a hole -> a ring; equal rows -> a grid; rows growing by one -> a
    pyramid wired downwards) and the threshold T from the size of the
    topples seen; it then checks that every topple in the clip matches the
    wiring, that every drop's cascade re-simulates to what was seen, and
    that the end state matches.

What-ifs are re-simulated with a sequential toppling loop (one unstable cell
at a time, lowest index first). The generator topples in parallel waves;
both give the same counts because toppling is order-independent.
"""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import render  # noqa: E402
from sim import ORD, clock_text, tenths, frames_showing  # noqa: E402  (text formatting only)

FPS = 24
TOL = 0.05          # px: a grain sits on a slot
WIN = 7


def _ord(k):
    return ORD[k]


class Model:
    def __init__(self, tr):
        self.tr = tr
        P = np.asarray(tr["poses"], dtype=float)
        self.n, self.ng = P.shape[0], P.shape[1]
        cells = tr["cells"]
        self.kinds = [c["kind"] for c in cells]
        self.nc = len(cells)
        self.live = [i for i, k in enumerate(self.kinds) if k == "cell"]
        slots = [(i, k, s[0], s[1]) for i, c in enumerate(cells) for k, s in enumerate(c["slots"])]
        sx = np.array([s[2] for s in slots])
        sy = np.array([s[3] for s in slots])
        sc = np.array([s[0] for s in slots])
        # which cell's slot each grain sits on (or -1), per frame
        X, Y, Z = P[:, :, 0], P[:, :, 1], P[:, :, 2]
        d = np.abs(X[:, :, None] - sx[None, None, :]) + np.abs(Y[:, :, None] - sy[None, None, :])
        j = np.argmin(np.nan_to_num(d, nan=1e9), axis=2)
        dmin = np.take_along_axis(np.nan_to_num(d, nan=1e9), j[:, :, None], axis=2)[:, :, 0]
        on = np.where((dmin < TOL) & (np.abs(np.nan_to_num(Z, nan=0.0) - 1.0) < 1e-9), sc[j], -1)
        still = np.zeros_like(on, dtype=bool)
        same_prev = np.zeros_like(still)
        same_prev[1:] = (np.abs(P[1:, :, 0] - P[:-1, :, 0]) < 1e-9) & (np.abs(P[1:, :, 1] - P[:-1, :, 1]) < 1e-9)
        same_prev[0] = True
        same_next = np.zeros_like(still)
        same_next[:-1] = same_prev[1:]
        same_next[-1] = True
        still = same_prev | same_next
        on[~still] = -1
        self.on = on
        self.occ = np.zeros((self.n, self.nc), dtype=int)
        for c in range(self.nc):
            self.occ[:, c] = (on == c).sum(axis=1)
        self._events(P)
        self._wiring()
        self._verify()

    # -------------------------------------------------------------- what happened
    def _events(self, P):
        on, n = self.on, self.n
        vis = ~np.isnan(P[:, :, 0])
        self.lands = []                                   # (frame, cell, grain)
        for g in range(self.ng):
            fv = np.nonzero(vis[:, g])[0]
            if len(fv) == 0 or fv[0] == 0:
                continue
            fr = np.nonzero(on[:, g] >= 0)[0]
            self.lands.append((int(fr[0]), int(on[fr[0], g]), g, int(fv[0])))
        self.lands.sort()
        dep = {}
        for g in range(self.ng):
            for f in range(1, n):
                if on[f - 1, g] >= 0 and on[f, g] < 0:
                    dest, arr = -1, None
                    nx = np.nonzero(on[f:, g] >= 0)[0]
                    if len(nx):
                        dest, arr = int(on[f + nx[0], g]), f + int(nx[0])
                    dep.setdefault((f, int(on[f - 1, g])), []).append((dest, arr, g))
        self.tops = [{"f": f, "cell": c, "to": sorted(d for d, _, _ in v), "arr": [a for _, a, _ in v]}
                     for (f, c), v in sorted(dep.items())]
        sizes = {len(t["to"]) for t in self.tops}
        self.T = sizes.pop() if len(sizes) == 1 else None
        # which dropped grain's cascade each topple belongs to
        lf = [l[0] for l in self.lands]
        for t in self.tops:
            t["drop"] = max(i for i, x in enumerate(lf) if x <= t["f"])
        self.first_top = {}
        self.n_top = {c: 0 for c in self.live}
        for t in self.tops:
            self.first_top.setdefault(t["cell"], t["f"])
            self.n_top[t["cell"]] += 1

    # -------------------------------------------------------------- the board, from the drawing
    def _wiring(self):
        cells = self.tr["cells"]
        cx = [c["centre"][0] for c in cells]
        cy = [c["centre"][1] for c in cells]
        nc = self.nc
        T = self.T or 0
        self.out = [[] for _ in range(nc)]
        self.desc = {}
        if self.tr["side_view"]:
            self.scenario = "row"
            order = sorted(range(nc), key=lambda i: cx[i])
            m = len(order)
            for r, i in enumerate(order):
                l = order[r - 1] if r > 0 else -1
                rr = order[r + 1] if r < m - 1 else -1
                self.out[i] = [l] * (T // 2) + [rr] * (T // 2)
                self.desc[i] = ("L", r + 1) if r < (m + 1) // 2 else ("R", m - r)
        elif "hole" in self.kinds:
            self.scenario = "ring"
            mx, my = sum(cx) / nc, sum(cy) / nc
            hole = self.kinds.index("hole")
            a0 = math.atan2(cy[hole] - my, cx[hole] - mx)
            order = sorted(range(nc), key=lambda i: (math.atan2(cy[i] - my, cx[i] - mx) - a0) % (2 * math.pi))
            for r, i in enumerate(order):
                if i == hole:
                    continue
                self.out[i] = [order[r - 1]] * (T // 2) + [order[(r + 1) % nc]] * (T // 2)
                self.desc[i] = (r,)
            self.out = [[-1 if (t >= 0 and self.kinds[t] == "hole") else t for t in o] for o in self.out]
        else:
            ys = sorted(set(round(y, 3) for y in cy))
            rows = [sorted([i for i in range(nc) if round(cy[i], 3) == y], key=lambda i: cx[i]) for y in ys]
            lens = [len(r) for r in rows]
            if len(set(lens)) == 1:
                self.scenario = "grid"
                for r, row in enumerate(rows):
                    for c, i in enumerate(row):
                        nb = []
                        for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                            rr, cc = r + dr, c + dc
                            nb.append(rows[rr][cc] if 0 <= rr < len(rows) and 0 <= cc < len(row) else -1)
                        self.out[i] = nb
                        self.desc[i] = (r + 1, c + 1)
            else:
                self.scenario = "pyramid"
                self.w0 = lens[0]
                for r, row in enumerate(rows):
                    for c, i in enumerate(row):
                        nxt = rows[r + 1] if r + 1 < len(rows) else None
                        self.out[i] = [nxt[c], nxt[c + 1]] if nxt else [-1, -1]
                        self.desc[i] = (r + 1, c + 1)
        self.by_desc = {tuple(v): k for k, v in self.desc.items()}

    def _verify(self):
        ok = self.T is not None and len(self.tops) > 0
        if ok:
            for t in self.tops:
                if sorted(self.out[t["cell"]]) != t["to"]:
                    ok = False
        if ok:
            h = [int(v) for v in self.occ[0]]
            for i, (f, c, g, _) in enumerate(self.lands):
                h[c] += 1
                per = self.relax(h)
                seen = [t["cell"] for t in self.tops if t["drop"] == i]
                if sorted(per["cells"]) != sorted(seen):
                    ok = False
            if h != [int(v) for v in self.occ[-1]]:
                ok = False
        self.consistent = ok

    # -------------------------------------------------------------- the rule (sequential)
    def relax(self, h):
        cells, lost = [], 0
        while True:
            u = [c for c in self.live if h[c] >= self.T]
            if not u:
                return {"cells": cells, "lost": lost, "n": len(cells)}
            if len(cells) > 5000:
                raise RuntimeError("no settling")
            c = u[0]
            h[c] -= self.T
            cells.append(c)
            for t in self.out[c]:
                if t >= 0:
                    h[t] += 1
                else:
                    lost += 1

    def run(self, init, drops):
        h = list(init)
        per = []
        for c in drops:
            h[c] += 1
            per.append(self.relax(h))
        return per, h

    # -------------------------------------------------------------- naming
    def cell(self, d):
        return self.by_desc[tuple(d)]

    def name(self, c):
        d = self.desc[c]
        if self.scenario == "row":
            return f"the {_ord(d[1] - 1)} tray from the {'left' if d[0] == 'L' else 'right'}"
        if self.scenario == "grid":
            return f"the cell in row {d[0]}, column {d[1]}"
        if self.scenario == "ring":
            return f"the {_ord(d[0] - 1)} cell clockwise from the hole"
        if d[0] == 1 and self.w0 == 1:
            return "the top cell"
        return f"the {_ord(d[1] - 1)} cell from the left in the {_ord(d[0] - 1)} row"

    def lost_in(self, i):
        return sum(sum(1 for d in t["to"] if d < 0) for t in self.tops if t["drop"] == i)


def grain_name(i):
    return f"the {ORD[i]} grain to be dropped"


@functools.lru_cache(maxsize=16)
def _model_key(key: str) -> Model:
    return Model(render.track(json.loads(key)))


def _model(scene) -> Model:
    return _model_key(json.dumps(scene, sort_keys=True))


def _half(frames):
    s = frames / FPS
    h = round(s * 2) / 2
    if abs(s - h) > 0.1 + 1e-9 or h <= 0:
        return None
    return f"{h:g} s"


def answer(key: str, p: dict, scene: dict):
    M = _model(scene)
    if not M.consistent:
        return None
    return _answer(key, p, M)


def _answer(key, p, M: Model):
    if not M.consistent:
        return None
    c = M.cell(p["cell"]) if "cell" in p else None
    init = [int(v) for v in M.occ[0]]
    drops = [l[1] for l in M.lands]
    per_seen = [[t for t in M.tops if t["drop"] == i] for i in range(len(M.lands))]
    if key == "n_drops":
        return len(M.lands)
    if key == "n_cells":
        return len(M.live)
    if key == "start_count":
        return int(M.occ[0, c])
    if key == "start_total":
        return int(M.occ[0].sum())
    if key == "end_count":
        return int(M.occ[-1, c])
    if key == "first_drop_cell":
        return M.name(M.lands[0][1])
    if key == "first_topple_cell":
        return M.name(M.tops[0]["cell"])
    if key == "ever_topples":
        return "yes" if M.n_top[c] else "no"
    if key == "most_end":
        xs = [M.cell(d) for d in p["cells"]]
        v = sorted(((int(M.occ[-1, x]), x) for x in xs), reverse=True)
        return M.name(v[0][1]) if v[0][0] > v[1][0] else None
    if key in ("topple_order",):
        a, b = M.cell(p["a"]), M.cell(p["b"])
        fa, fb = M.first_top.get(a), M.first_top.get(b)
        if fa is None or fb is None or fa == fb:
            return None
        return M.name(a) if fa < fb else M.name(b)
    if key == "more_topples":
        a, b = M.cell(p["a"]), M.cell(p["b"])
        if M.n_top[a] == M.n_top[b]:
            return None
        return M.name(a) if M.n_top[a] > M.n_top[b] else M.name(b)
    if key == "bigger_cascade":
        i, j = p["i"], p["j"]
        a, b = len(per_seen[i]), len(per_seen[j])
        return None if a == b else grain_name(i if a > b else j)
    if key == "trigger":
        t = next(t for t in M.tops if t["cell"] == c)
        return grain_name(t["drop"])
    if key == "start_vs_end":
        a, b = int(M.occ[0, c]), int(M.occ[-1, c])
        return "more" if b > a else "fewer" if b < a else "the same number"
    if key == "more_end":
        a, b = M.cell(p["a"]), M.cell(p["b"])
        va, vb = M.occ[-1, a], M.occ[-1, b]
        return None if va == vb else M.name(a if va > vb else b)
    if key == "receives":
        t = next(t for t in M.tops if t["cell"] == c)
        hit = [M.cell(d) for d in p["opts"] if M.cell(d) in t["to"]]
        return M.name(hit[0]) if len(hit) == 1 else None
    if key == "lost_side":
        return _lost_side(M)
    if key == "topple_time":
        return clock_text(tenths(M.first_top[c]))
    if key == "land_time":
        return clock_text(tenths(M.lands[p["i"]][0]))
    if key == "total_topples":
        return len(M.tops)
    if key == "cell_topples":
        return M.n_top[c]
    if key == "count_at":
        a, b = frames_showing(p["t"])
        vals = {int(M.occ[f, c]) for f in range(a - WIN, b + WIN + 1)}
        return vals.pop() if len(vals) == 1 else None
    if key == "lost_total":
        return sum(sum(1 for d in t["to"] if d < 0) for t in M.tops)
    if key == "end_total":
        return int(M.occ[-1].sum())
    if key == "cascade_span":
        i = p["i"]
        return _half(max(t["f"] for t in per_seen[i]) - M.lands[i][0])
    if key == "cascade_size":
        return len(per_seen[p["i"]])
    if key == "n_toppled_cells":
        return len(M.first_top)
    if key == "biggest":
        v = sorted(((len(per_seen[i]), i) for i in p["ids"]), reverse=True)
        return grain_name(v[0][1]) if v[0][0] > v[1][0] else None
    if key == "drops_before":
        return M.tops[0]["drop"] + 1
    if key == "last_topple_cell":
        last = [t for t in M.tops if t["f"] == M.tops[-1]["f"]]
        return M.name(last[0]["cell"]) if len(last) == 1 else None
    if key == "topples_between":
        f1, f2 = frames_showing(p["t1"])[1], frames_showing(p["t2"])[0]
        return sum(1 for t in M.tops if f1 < t["f"] < f2)
    if key == "cascade_lost":
        return M.lost_in(p["i"])
    if key == "count_after":
        i = p["i"]
        f = M.lands[i + 1][3] - 1 if i + 1 < len(M.lands) else M.n - 1      # before the next grain shows up
        return int(M.occ[f, c])
    # what-ifs: re-simulated from the drawn start, the drawn drops and the wiring
    if key in ("cf_skip_total", "cf_lost", "cf_skip_cell"):
        dd = [x for k, x in enumerate(drops) if k != p["i"]]
        per, _ = M.run(init, dd)
        if key == "cf_skip_total":
            return sum(r["n"] for r in per)
        if key == "cf_lost":
            return sum(r["lost"] for r in per)
        return "yes" if any(c in r["cells"] for r in per) else "no"
    if key == "cf_move":
        dd = list(drops)
        dd[p["i"]] = c
        per, _ = M.run(init, dd)
        return per[p["i"]]["n"]
    if key == "cf_extra_end":
        h = [int(v) for v in M.occ[-1]]
        h[c] += 1
        return M.relax(h)["n"]
    if key == "cf_init":
        h = list(init)
        h[c] += p["d"]
        pre = M.relax(h)
        per, _ = M.run(h, drops)
        return pre["n"] + sum(r["n"] for r in per)
    return None


def _lost_side(M):
    for t in M.tops:
        if any(d < 0 for d in t["to"]):
            wave = [x for x in M.tops if x["f"] == t["f"]]
            sides = set()
            cx = [cc["centre"][0] for cc in M.tr["cells"]]
            cy = [cc["centre"][1] for cc in M.tr["cells"]]
            P = M.tr["poses"]
            for w in wave:
                # where the lost grains were last drawn
                for g in range(M.ng):
                    if M.on[w["f"] - 1, g] == w["cell"] and M.on[w["f"], g] < 0:
                        nx = np.nonzero(M.on[w["f"]:, g] >= 0)[0]
                        if len(nx):
                            continue
                        fr = np.nonzero(~np.isnan(P[:, g, 0]))[0]
                        x, y = P[fr[-1], g, 0], P[fr[-1], g, 1]
                        if M.scenario == "row":
                            sides.add("left" if x < min(cx) else "right")
                        else:
                            sides.add("left" if x < min(cx) - 1 else "right" if x > max(cx) + 1
                                      else "top" if y < min(cy) else "bottom")
            if len(sides) != 1:
                return None
            s = sides.pop()
            return f"the {s} end" if M.scenario == "row" else f"the {s} edge"
    return None
