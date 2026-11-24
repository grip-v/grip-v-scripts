"""Independent answers for mirror_beam_reflection (validator, tests, frame check).

Factual answers come from what the renderer draws, frame by frame
(render.state: the pulse heads, the trails, the mirror angles; or the same
read off decoded MP4 frames by frame_check.py). Heads are linked into
tracks by nearest neighbour; a turn is where two straight runs of a track
meet (the corner is the intersection of the incoming and outgoing lines,
its time follows from the measured speed); a track that starts inside the
board is a split; a head that vanishes at the board edge has left it.

Predictions (the path after the clip) and what-ifs use a separate tracer
written here: complex-number directions, pulses processed in time order
from a heap, mirrors read from the drawn geometry (cell, diagonal, kind).
It does not import sim.py's tracer.
"""
from __future__ import annotations
import heapq
import json
import math

import render

FPS = 24
LETTERS = "ABCDEFGHIJ"
_CACHE: dict = {}
DIR_WORDS = {(1, 0): "right", (-1, 0): "left", (0, 1): "down", (0, -1): "up"}


# ------------------------------------------------------------------ names / formats (shared with questions.py)
def cell_name(cell):
    return f"{LETTERS[cell[0]]}{cell[1] + 1}"


def end_text(end):
    if end[0] == "exit":
        edge, i = end[1], end[2]
        where = f"row {i + 1}" if edge in ("left", "right") else f"column {LETTERS[i]}"
        return f"leaves through the {edge} edge at {where}"
    return f"is stopped in cell {cell_name(end[1])}"


def fmt_s(v):
    return f"{v:.1f} s"


def fmt_clock(tenths):
    return f"{tenths / 10:.1f} s"


def fmt_speed(v):
    return f"{v:.1f} cells per second"


def clock_tenths(f):
    return f * 10 // FPS


def frame_of_clock(t):
    k = round(t * 10)
    f = (k * FPS + 9) // 10
    while clock_tenths(f) < k:
        f += 1
    return f


def event_clock(te):
    """Clock reading on the first frame drawn after an event at te seconds."""
    return clock_tenths(math.ceil(te * FPS - 1e-9))


# ------------------------------------------------------------------ the drawn track
def drawn(scene):
    key = json.dumps(scene, sort_keys=True)
    if key not in _CACHE:
        if len(_CACHE) > 16:
            _CACHE.clear()
        _CACHE[key] = [render.state(scene, f) for f in range(render.n_frames(scene))]
    return _CACHE[key]


class Track:
    """Measurements from per-frame drawn state (heads, trails, mirror angles)."""

    def __init__(self, scene, frames, jitter=0.0):
        self.s, self.fr, self.j = scene, frames, jitter
        st = scene["style"]
        self.x0, self.y0, self.c = st["x0"], st["y0"], st["cell"]
        self.tracks = self._link()

    def cell_of(self, x, y):
        return (math.floor((x - self.x0) / self.c), math.floor((y - self.y0) / self.c))

    def _link(self):
        out = {}
        for f, st in enumerate(self.fr):
            by_col = {}
            for colour, x, y in st["heads"]:
                by_col.setdefault(colour, []).append((x, y))
            for colour, heads in by_col.items():
                ts = out.setdefault(colour, [])
                live = [t for t in ts if t["pts"][-1][0] == f - 1]
                pairs = sorted((math.dist(self._predict(t), h), i, j) for i, t in enumerate(live)
                               for j, h in enumerate(heads))
                used_t, used_h = set(), set()
                for dist, i, j in pairs:                # nearest predicted position first, one head per track
                    if dist < 0.35 * self.c and i not in used_t and j not in used_h:
                        live[i]["pts"].append((f,) + tuple(heads[j]))
                        used_t.add(i)
                        used_h.add(j)
                for j, h in enumerate(heads):
                    if j not in used_h:
                        ts.append({"pts": [(f,) + tuple(h)]})
        for ts in out.values():
            for t in ts:
                t.update(self._analyse(t["pts"]))
        return out

    @staticmethod
    def _predict(t):
        p = t["pts"]
        if len(p) < 2:
            return p[-1][1:]
        return (2 * p[-1][1] - p[-2][1], 2 * p[-1][2] - p[-2][2])

    def _analyse(self, pts):
        moves = []
        for (fa, xa, ya), (fb, xb, yb) in zip(pts, pts[1:]):
            dx, dy = xb - xa, yb - ya
            tol = 0.5 + self.j
            if abs(dy) <= tol < abs(dx):
                moves.append((fa, (1 if dx > 0 else -1, 0), abs(dx)))
            elif abs(dx) <= tol < abs(dy):
                moves.append((fa, (0, 1 if dy > 0 else -1), abs(dy)))
            else:
                moves.append((fa, None, math.hypot(dx, dy)))
        pure = [m for m in moves if m[1] is not None]
        speed = sum(m[2] for m in pure) / len(pure) * FPS if pure else 0.0   # px / s (mean: jitter telescopes)
        corners, last = [], None
        by_f = {f: (x, y) for f, x, y in pts}
        for f, d, _ in moves:
            if d is None:
                continue
            if last is not None and d != last[1]:
                a = by_f[last[0]]                       # a point on the incoming line
                b = by_f[f]                             # a point on the outgoing line
                u = last[1]
                corner = (b[0] if u[1] == 0 else a[0], a[1] if u[1] == 0 else b[1])
                t = last[0] / FPS + math.dist(a, corner) / speed
                corners.append({"t": t, "cell": self.cell_of(*corner), "din": u, "dout": d, "xy": corner})
            last = (f + 1, d)                           # the frame ending this run is on the incoming line
        start, end = pts[0], pts[-1]
        return {"speed": speed, "corners": corners, "start": start, "end": end, "first_dir": pure[0][1] if pure else None}

    def colour_tracks(self, colour):
        return sorted(self.tracks.get(colour, []), key=lambda t: t["start"][0])

    def all_corners(self, colour):
        return sorted((c for t in self.colour_tracks(colour) for c in t["corners"]), key=lambda c: c["t"])

    def speed_cells(self):
        sp = [t["speed"] for ts in self.tracks.values() for t in ts if t["speed"] > 0]
        return sorted(sp)[len(sp) // 2] / self.c

    def entry_edge(self, colour):
        f, x, y = self.colour_tracks(colour)[0]["start"]
        W, H = self.s["cols"], self.s["rows"]
        d = {"left": abs(x - self.x0), "right": abs(x - (self.x0 + W * self.c)),
             "top": abs(y - self.y0), "bottom": abs(y - (self.y0 + H * self.c))}
        return min(d, key=d.get)

    def lit_cells(self):
        out = set()
        for colour, pts in self.fr[-1]["trails"]:
            for a, b in zip(pts, pts[1:]):
                L = math.dist(a, b)
                n = max(1, int(L / 2))
                for k in range(n + 1):
                    x, y = a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n
                    fx, fy = (x - self.x0) / self.c, (y - self.y0) / self.c
                    if abs(fx - round(fx)) * self.c < 2 or abs(fy - round(fy)) * self.c < 2:
                        continue                        # on a grid line: belongs to neither cell
                    cx, cy = math.floor(fx), math.floor(fy)
                    if 0 <= cx < self.s["cols"] and 0 <= cy < self.s["rows"]:
                        out.add((cx, cy))
        return out

    def at_edge(self, x, y):
        W, H = self.s["cols"], self.s["rows"]
        return min(abs(x - self.x0), abs(x - self.x0 - W * self.c), abs(y - self.y0), abs(y - self.y0 - H * self.c)) \
            < 0.25 * self.c

    def first_reach(self, cell):
        """Earliest time any head is at the centre of `cell`: a corner there, or a straight pass (a track
        whose runs pass the centre), from the per-frame positions."""
        cx, cy = self.x0 + (cell[0] + 0.5) * self.c, self.y0 + (cell[1] + 0.5) * self.c
        best = math.inf
        for ts in self.tracks.values():
            for t in ts:
                for cn in t["corners"]:
                    if cn["cell"] == cell:
                        best = min(best, cn["t"])
                pts = t["pts"]
                for (fa, xa, ya), (fb, xb, yb) in zip(pts, pts[1:]):
                    on_x = abs(ya - cy) < 1 + self.j and abs(yb - cy) < 1 + self.j and (xa - cx) * (xb - cx) <= 0 and xa != xb
                    on_y = abs(xa - cx) < 1 + self.j and abs(xb - cx) < 1 + self.j and (ya - cy) * (yb - cy) <= 0 and ya != yb
                    if (on_x or on_y) and t["speed"] > 0:
                        # from the incoming side only (the frame before the centre), at the measured speed:
                        # right after a split the outgoing frames can hold two overlapping heads
                        best = min(best, fa / FPS + math.dist((xa, ya), (cx, cy)) / t["speed"])
                if pts and math.dist(pts[0][1:], (cx, cy)) < 0.5 * self.c and t["speed"] > 0:
                    best = min(best, pts[0][0] / FPS - math.dist(pts[0][1:], (cx, cy)) / t["speed"])
        return best

    def flip_start(self, i):
        """First frame at which mirror i is drawn at a different angle than at frame 0 (None if never)."""
        a0 = self.fr[0]["angles"][i]
        return next((f for f, st in enumerate(self.fr) if abs(st["angles"][i] - a0) > 1e-6), None)


# ------------------------------------------------------------------ independent tracer (complex numbers, time-ordered)
def _turn(kind, d):
    # screen coords (y down) as complex x + iy: '/' maps right (1) -> up (-i): d -> -i * conj(d)
    return -1j * d.conjugate() if kind == "/" else 1j * d.conjugate()


def run(scene, *, remove=None, rotate=None, add=None, step=None, no_flip=False, two_sided=False,
        no_absorbers=False, delay=None):
    """Every pulse's (end, t_end, turns) under the given change. end = ("exit", edge, index) or
    ("absorbed", cell); turns counts reflections (a split counts for the reflected copy)."""
    W, H = scene["cols"], scene["rows"]
    s = scene["step"] if step is None else step
    mir = {}
    for m in scene["mirrors"]:
        c = tuple(m["cell"])
        if c == remove:
            continue
        mm = dict(m)
        if c == rotate:
            mm["kind"] = "\\" if mm["kind"] == "/" else "/"
        if no_flip and mm["type"] == "flip":
            mm["type"] = "two"
        if two_sided and mm["type"] == "one":
            mm["type"] = "two"
        mir[c] = mm
    if add is not None:
        mir[tuple(add[0])] = {"cell": list(add[0]), "kind": add[1], "type": "two"}
    absb = set() if no_absorbers else {tuple(a) for a in scene["absorbers"]}
    heap, out, seq = [], [], 0
    for i, em in enumerate(scene["emitters"]):
        e, k = em["edge"], em["index"]
        pos = {"left": complex(0, k), "right": complex(W - 1, k), "top": complex(k, 0), "bottom": complex(k, H - 1)}[e]
        d = {"left": 1, "right": -1, "top": 1j, "bottom": -1j}[e]
        t0 = em["t0"] + (delay[1] if delay and delay[0] == em["colour"] else 0.0)
        heapq.heappush(heap, (t0 + s / 2, seq, em["colour"], pos, d, 0, 0))
        seq += 1
    visited = set()
    while heap:
        t, _, colour, pos, d, turns, n = heapq.heappop(heap)
        c, r = int(pos.real), int(pos.imag)
        if not (0 <= c < W and 0 <= r < H):
            edge = "left" if c < 0 else "right" if c >= W else "top" if r < 0 else "bottom"
            out.append({"colour": colour, "end": ("exit", edge, r if edge in ("left", "right") else c),
                        "t_end": t - s / 2, "turns": turns})
            continue
        if (c, r, d) in visited or n > 400:
            return None
        visited.add((c, r, d))
        if (c, r) in absb:
            out.append({"colour": colour, "end": ("absorbed", (c, r)), "t_end": t, "turns": turns})
            continue
        m = mir.get((c, r))
        if m is not None:
            kind = m["kind"]
            if m["type"] == "flip" and t >= m["flip_t"]:
                kind = "\\" if kind == "/" else "/"
            if m["type"] == "one":
                face = complex(*m["face"])
                if (d.real * face.real + d.imag * face.imag) >= 0:
                    out.append({"colour": colour, "end": ("absorbed", (c, r)), "t_end": t, "turns": turns})
                    continue
            nd = _turn(kind, d)
            if m["type"] == "split":
                heapq.heappush(heap, (t + s, seq, colour, pos + nd, nd, turns + 1, n + 1))
                seq += 1
            else:
                d, turns = nd, turns + 1
        heapq.heappush(heap, (t + s, seq, colour, pos + d, d, turns, n + 1))
        seq += 1
    return sorted(out, key=lambda o: (o["t_end"], o["colour"]))


def _one(scene, colour, **kw):
    r = run(scene, **kw)
    ends = [o for o in r if o["colour"] == colour]
    return ends[0] if len(ends) == 1 else None


# ------------------------------------------------------------------ the dispatcher
def answer(key: str, p: dict, scene: dict, frames=None, jitter=0.0):
    T = Track(scene, frames if frames is not None else drawn(scene), jitter)
    mirrors = scene["mirrors"]
    ang0 = (frames if frames is not None else drawn(scene))[0]["angles"]
    if key == "n_mirrors":
        return len(ang0)
    if key == "n_slash":
        return sum(1 for a in ang0 if a < 0)
    if key == "n_absorbers":
        return len(scene["absorbers"])
    if key == "n_special":
        return sum(1 for m in mirrors if m["type"] == p["type"])
    if key == "entry_edge":
        return T.entry_edge(p["colour"])
    if key == "first_launch":
        return min(T.tracks, key=lambda c: T.colour_tracks(c)[0]["start"][0])
    if key == "first_mirror":
        return cell_name(T.all_corners(p["colour"])[0]["cell"])
    if key == "turn_dir_first":
        return DIR_WORDS[T.all_corners(p["colour"])[0]["dout"]]
    if key == "head_cell_at_t":
        f = frame_of_clock(p["t"])
        hs = [h for h in T.fr[f]["heads"] if h[0] == p["colour"]]
        return cell_name(T.cell_of(hs[0][1], hs[0][2]))
    if key == "n_turns_by_t":
        return sum(1 for c in T.all_corners(p["colour"]) if c["t"] < p["t"])
    if key == "first_turn_beam":
        return min(T.tracks, key=lambda c: T.all_corners(c)[0]["t"])
    if key == "flip_before_after":
        i = next(k for k, m in enumerate(mirrors) if tuple(m["cell"]) == tuple(p["cell"]))
        hit = next(c["t"] for col in T.tracks for c in T.all_corners(col) if c["cell"] == tuple(p["cell"]))
        fs = T.flip_start(i)
        return "after it turned" if fs is not None and hit > fs / FPS else "before it turned"
    if key == "time_turn_at":
        return fmt_clock(event_clock(T.first_reach(tuple(p["cell"]))))
    if key == "split_cell":
        starts = [t for col in T.tracks for t in T.colour_tracks(col)[1:] if T.colour_tracks(col)]
        inner = [t for t in starts if not T.at_edge(*t["start"][1:])]
        first = min(inner, key=lambda t: t["start"][0])
        return cell_name(T.cell_of(*first["start"][1:]))
    if key == "launch_time":
        return fmt_clock(clock_tenths(T.colour_tracks(p["colour"])[0]["start"][0]))
    if key == "time_between_mirrors":
        return fmt_s(T.first_reach(tuple(p["b"])) - T.first_reach(tuple(p["a"])))
    if key == "n_heads_at_t":
        return len(T.fr[frame_of_clock(p["t"])]["heads"])
    if key == "speed":
        return fmt_speed(T.speed_cells())
    if key == "time_between_turns":
        cs = T.all_corners(p["colour"])
        return fmt_s(cs[p["k"] + 1]["t"] - cs[p["k"]]["t"])
    if key == "cells_between":
        cs = T.all_corners(p["colour"])
        a, b = cs[p["k"]]["xy"], cs[p["k"] + 1]["xy"]
        return round(math.dist(a, b) / T.c)
    if key == "n_lit_cells":
        return len(T.lit_cells())
    # ---------------- predictions (independent tracer)
    if key == "exit_point":
        return end_text(_one(scene, p["colour"])["end"])
    if key == "exit_time":
        return fmt_clock(math.floor(_one(scene, p["colour"])["t_end"] * 10 + 1e-9))
    if key == "n_turns_total":
        return _one(scene, p["colour"])["turns"]
    if key == "first_out":
        return run(scene)[0]["colour"]
    if key == "n_exits":
        return sum(1 for o in run(scene) if o["end"][0] == "exit")
    if key == "time_after_end":
        return fmt_s(_one(scene, p["colour"])["t_end"] - render.duration(scene))
    # ---------------- counterfactuals (re-traced)
    kw = {}
    if key.startswith("cf_"):
        ch = p["change"]
        if ch == "remove":
            kw["remove"] = tuple(p["cell"])
        elif ch == "rotate":
            kw["rotate"] = tuple(p["cell"])
        elif ch == "add":
            kw["add"] = (tuple(p["cell"]), p["kind"])
        elif ch == "faster":
            kw["step"] = scene["step"] / 2
        elif ch == "no_flip":
            kw["no_flip"] = True
        elif ch == "two_sided":
            kw["two_sided"] = True
        elif ch == "no_absorbers":
            kw["no_absorbers"] = True
        elif ch == "delay":
            kw["delay"] = (p["colour"], p["d"])
        r = run(scene, **kw)
        if r is None:
            return None
        if key == "cf_exit_point":
            return end_text([o for o in r if o["colour"] == p["colour"]][0]["end"])
        if key == "cf_exit_time":
            return fmt_clock(math.floor([o for o in r if o["colour"] == p["colour"]][0]["t_end"] * 10 + 1e-9))
        if key == "cf_n_exits":
            return sum(1 for o in r if o["end"][0] == "exit")
        if key == "cf_first_out":
            return r[0]["colour"]
    return None
