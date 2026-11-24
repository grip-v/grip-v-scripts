"""Independent answers for hidden_mirrors (validator, tests, frame check).

Nothing here reads sim.py's traces. Factual answers are measured from what
the renderer draws, frame by frame (render.state, or the same read off
decoded MP4 frames by frame_check.py): one head per pulse colour, linked
into a track; a turn is where two straight runs of a track meet (the
corner is the intersection of the incoming and outgoing lines, its time
follows from the measured speed); the entry port is where the track starts
and the exit port is the edge the last head is heading into (a head that
stops more than ~0.4 cell short of the edge was stopped by a block).

Predictions and what-ifs re-simulate with a separate tracer written here
(complex-number directions) on the mirror map INFERRED from the tracks:
a corner cell holds a mirror whose slant follows from the turn, a cell a
track crosses straight is empty, blocks and drawn mirrors are read from
the drawing. If a hypothetical path enters a cell no pulse has visited the
tracer gives up (None), so a what-if that cannot be worked out from the
video fails validation.
"""
from __future__ import annotations
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


def port_text(port):
    edge, i = port
    if edge in ("top", "bottom"):
        return f"{edge} of column {LETTERS[i]}"
    return f"{edge} end of row {i + 1}"


def end_text(end):
    """end: ("exit", (edge, index)) or ("stopped", cell)."""
    if end[0] == "exit":
        return f"leaves at the {port_text(end[1])}"
    return f"is stopped by the block in {cell_name(end[1])}"


def fmt_s(v):
    return f"{v:.1f} s"


def fmt_clock(tenths):
    return f"{tenths / 10:.1f} s"


def fmt_speed(v):
    """Speeds are asked to the nearest half cell per second."""
    return f"{round(v * 2) / 2:.1f} cells per second"


def clock_tenths(f):
    return f * 10 // FPS


def frame_of_clock(t):
    k = round(t * 10)
    f = (k * FPS + 9) // 10
    while clock_tenths(f) < k:
        f += 1
    return f


def event_clock(te):
    """Clock reading on the first frame drawn at or after an event at te seconds."""
    return clock_tenths(math.ceil(te * FPS - 1e-9))


def mirror_kind(din, dout):
    """'/' turns right-moving light up: (dx, dy) -> (-dy, -dx)."""
    return "/" if tuple(dout) == (-din[1], -din[0]) else "\\"


# ------------------------------------------------------------------ the drawn track
def drawn(scene):
    key = json.dumps(scene, sort_keys=True)
    if key not in _CACHE:
        if len(_CACHE) > 16:
            _CACHE.clear()
        _CACHE[key] = [render.state(scene, f) for f in range(render.n_frames(scene))]
    return _CACHE[key]


class Track:
    """Measurements from per-frame drawn heads (one head per colour per frame)."""

    def __init__(self, scene, frames, jitter=0.0):
        self.s, self.j = scene, jitter
        st = scene["style"]
        self.x0, self.y0, self.c = st["x0"], st["y0"], st["cell"]
        self.W, self.H = scene["cols"], scene["rows"]
        pts = {}
        for f, fr in enumerate(frames):
            for colour, x, y in fr["heads"]:
                pts.setdefault(colour, []).append((f, x, y))
        self.t = {colour: self._analyse(p) for colour, p in pts.items()}

    def cell_of(self, x, y):
        return (math.floor((x - self.x0) / self.c), math.floor((y - self.y0) / self.c))

    def _edge_dist(self, x, y, d):
        """Distance (px) from (x, y) to the board edge in direction d."""
        if d == (1, 0):
            return self.x0 + self.W * self.c - x
        if d == (-1, 0):
            return x - self.x0
        if d == (0, 1):
            return self.y0 + self.H * self.c - y
        return y - self.y0

    def _analyse(self, pts):
        moves = []
        tol = 0.5 + self.j
        for (fa, xa, ya), (fb, xb, yb) in zip(pts, pts[1:]):
            dx, dy = xb - xa, yb - ya
            n = fb - fa                                  # frames between detections (a decoded head can be missed)
            if abs(dy) <= tol < abs(dx):
                moves.append((fa, (1 if dx > 0 else -1, 0), abs(dx) / n, fb))
            elif abs(dx) <= tol < abs(dy):
                moves.append((fa, (0, 1 if dy > 0 else -1), abs(dy) / n, fb))
            else:
                moves.append((fa, None, math.hypot(dx, dy) / n, fb))
        pure = [m for m in moves if m[1] is not None]
        by_f = {f: (x, y) for f, x, y in pts}
        # speed from the moves away from the board edge (a head on the edge may be partly hidden)
        inner = [m for m in pure if min(self._edge_dist(*by_f[m[0]], d) for d in DIR_WORDS) > 0.3 * self.c
                 and min(self._edge_dist(*by_f[m[3]], d) for d in DIR_WORDS) > 0.3 * self.c] or pure
        speed = sum(m[2] for m in inner) / len(inner) * FPS           # px / s
        corners, last = [], None
        for f, d, _, f_next in moves:
            if d is None:
                continue
            if last is not None and d != last[1]:
                a, b, u = by_f[last[0]], by_f[f], last[1]
                corner = (b[0] if u[1] == 0 else a[0], a[1] if u[1] == 0 else b[1])
                t = last[0] / FPS + math.dist(a, corner) / speed
                corners.append({"t": t, "cell": self.cell_of(*corner), "din": u, "dout": d, "xy": corner})
            last = (f_next, d)
        d0, d1 = pure[0][1], pure[-1][1]
        sg = lambda v: 1 if v > 0 else -1
        # a turn in the first or last cell: the head may show only one frame on that half-cell leg, a point
        # off the line of the neighbouring straight run (the corner is where the two lines meet)
        fa = pure[0][0]
        if fa > pts[0][0]:
            a, b = pts[0][1:], by_f[fa]
            off = a[0] - b[0] if d0[0] == 0 else a[1] - b[1]
            if abs(off) > tol:
                corner = (b[0], a[1]) if d0[0] == 0 else (a[0], b[1])
                din = (sg(corner[0] - a[0]), 0) if d0[0] == 0 else (0, sg(corner[1] - a[1]))
                t = pts[0][0] / FPS + math.dist(a, corner) / speed
                corners.insert(0, {"t": t, "cell": self.cell_of(*corner), "din": din, "dout": d0, "xy": corner})
                d0 = din
        if last[0] < pts[-1][0]:
            a, b = by_f[last[0]], pts[-1][1:]
            off = b[0] - a[0] if d1[0] == 0 else b[1] - a[1]
            if abs(off) > tol:
                corner = (a[0], b[1]) if d1[0] == 0 else (b[0], a[1])
                dout = (sg(off), 0) if d1[0] == 0 else (0, sg(off))
                t = last[0] / FPS + math.dist(a, corner) / speed
                corners.append({"t": t, "cell": self.cell_of(*corner), "din": d1, "dout": dout, "xy": corner})
                d1 = dout
        f0, sx, sy = pts[0]
        back = (-d0[0], -d0[1])
        t_start = f0 / FPS - self._edge_dist(sx, sy, back) / speed
        edge0 = DIR_WORDS[back].replace("up", "top").replace("down", "bottom")
        port0 = (edge0, self.cell_of(sx, sy)[1] if d0[1] == 0 else self.cell_of(sx, sy)[0])
        fl, ex, ey = pts[-1]
        rest = self._edge_dist(ex, ey, d1)
        if rest < 0.39 * self.c:
            edge1 = DIR_WORDS[d1].replace("up", "top").replace("down", "bottom")
            end = ("exit", (edge1, self.cell_of(ex, ey)[1] if d1[1] == 0 else self.cell_of(ex, ey)[0]))
            t_end = fl / FPS + rest / speed
        else:
            cell = self.cell_of(ex + d1[0] * 0.3 * self.c, ey + d1[1] * 0.3 * self.c)
            cx, cy = self.x0 + (cell[0] + 0.5) * self.c, self.y0 + (cell[1] + 0.5) * self.c
            end = ("stopped", cell)
            t_end = fl / FPS + math.dist((ex, ey), (cx, cy)) / speed
        # path: edge start -> corners -> end point, in px
        start_xy = (sx + back[0] * self._edge_dist(sx, sy, back), sy + back[1] * self._edge_dist(sx, sy, back))
        end_xy = (ex + d1[0] * rest, ey + d1[1] * rest) if end[0] == "exit" else (cx, cy)
        poly = [start_xy] + [cn["xy"] for cn in corners] + [end_xy]
        return {"speed": speed, "corners": corners, "t_start": t_start, "t_end": t_end, "port": port0,
                "end": end, "poly": poly, "length": sum(math.dist(a, b) for a, b in zip(poly, poly[1:]))}

    # -------------------------------------------------- derived facts
    def colours(self):
        return sorted(self.t, key=lambda k: (self.t[k]["t_start"], k))

    def speed_cells(self):
        sp = sorted(v["speed"] for v in self.t.values())
        return sp[len(sp) // 2] / self.c

    def n_cells(self, colour):
        return int(round(self.t[colour]["length"] / self.c))

    def visited(self):
        """Cells some track passes through: from the polylines, sampled every 0.05 cell."""
        out = set()
        for v in self.t.values():
            for a, b in zip(v["poly"], v["poly"][1:]):
                n = max(1, int(math.dist(a, b) / (0.05 * self.c)))
                for k in range(n + 1):
                    x, y = a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n
                    fx, fy = (x - self.x0) / self.c, (y - self.y0) / self.c
                    if abs(fx - round(fx)) < 0.1 or abs(fy - round(fy)) < 0.1:
                        continue                        # near a grid line: belongs to neither cell
                    cl = (math.floor(fx), math.floor(fy))
                    if 0 <= cl[0] < self.W and 0 <= cl[1] < self.H:
                        out.add(cl)
        return out

    def turn_cells(self):
        return {cn["cell"]: mirror_kind(cn["din"], cn["dout"]) for v in self.t.values() for cn in v["corners"]}

    def known_map(self):
        """cell -> '/', '\\', 'empty' or 'block' for every cell the video tells us about."""
        m = {cell: "empty" for cell in self.visited()}
        m.update(self.turn_cells())
        for mm in self.s["mirrors"]:
            if mm["shown"]:
                m[tuple(mm["cell"])] = mm["kind"]
        for b in self.s["blocks"]:
            m[tuple(b)] = "block"
        return m


# ------------------------------------------------------------------ an independent tracer
def _entry(W, H, port):
    edge, i = port
    if edge == "left":
        return complex(0, i), 1
    if edge == "right":
        return complex(W - 1, i), -1
    if edge == "top":
        return complex(i, 0), 1j
    return complex(i, H - 1), -1j


def walk(W, H, known, port):
    """Trace on a known map. Returns {"end": ..., "turns": n, "cells": n} or None if the path enters a cell
    whose content the video does not show."""
    z, d = _entry(W, H, tuple(port))
    turns = cells = 0
    for _ in range(4 * W * H + 4):
        c, r = int(z.real), int(z.imag)
        if not (0 <= c < W and 0 <= r < H):
            if c < 0:
                end = ("exit", ("left", r))
            elif c >= W:
                end = ("exit", ("right", r))
            elif r < 0:
                end = ("exit", ("top", c))
            else:
                end = ("exit", ("bottom", c))
            return {"end": end, "turns": turns, "cells": cells}
        k = known.get((c, r))
        if k is None:
            return None
        cells += 1
        if k == "block":
            return {"end": ("stopped", (c, r)), "turns": turns, "cells": cells}
        if k == "/":
            d, turns = -1j * d.conjugate(), turns + 1
        elif k == "\\":
            d, turns = 1j * d.conjugate(), turns + 1
        z += d
    return None


def _cf_map(tr, p):
    m = dict(tr.known_map())
    if "cell" in p:
        cell = tuple(p["cell"])
        if p.get("change") == "flip":
            m[cell] = {"/": "\\", "\\": "/"}[m[cell]]
        elif p.get("change") in ("remove", "unblock"):
            m[cell] = "empty"
    return m


# ------------------------------------------------------------------ answers
def answer(key: str, p: dict, scene: dict, frames=None, jitter=0.0):
    tr = Track(scene, frames if frames is not None else drawn(scene), jitter)
    T = tr.t
    col = p.get("colour")
    if key == "n_pulses":
        return str(len(T))
    if key == "entry_port":
        return port_text(T[col]["port"])
    if key == "first_colour":
        return tr.colours()[0]
    if key == "n_blocks":
        return str(len(scene["blocks"]))
    if key == "n_shown":
        return str(sum(1 for m in scene["mirrors"] if m["shown"]))
    if key == "launch_order":
        return ", ".join(sorted(p["colours"], key=lambda k: T[k]["t_start"]))
    if key == "exit_order":
        return ", ".join(sorted(p["colours"], key=lambda k: T[k]["t_end"]))
    if key == "exit_port":
        return end_text(T[col]["end"])
    if key == "first_turn_dir":
        return DIR_WORDS[T[col]["corners"][0]["dout"]]
    if key == "first_turn_cell":
        return cell_name(T[col]["corners"][0]["cell"])
    if key == "n_straight":
        return str(sum(1 for v in T.values() if not v["corners"]))
    if key == "n_turns":
        return str(len(T[col]["corners"]))
    if key == "n_cells":
        return str(tr.n_cells(col))
    if key == "time_on_board":
        return fmt_s(T[col]["t_end"] - T[col]["t_start"])
    if key == "exit_clock":
        return fmt_clock(event_clock(T[col]["t_end"]))
    if key == "turn_clock":
        return fmt_clock(event_clock(T[col]["corners"][0]["t"]))
    if key == "launch_gap":
        return fmt_s(T[p["later"]]["t_start"] - T[p["earlier"]]["t_start"])
    if key == "speed":
        return fmt_speed(tr.speed_cells())
    if key == "n_turn_cells":
        return str(len(tr.turn_cells()))
    if key == "total_turns":
        return str(sum(len(v["corners"]) for v in T.values()))
    if key == "locate_mirror":
        cn = T[col]["corners"][p["k"] - 1]
        return f"{cell_name(cn['cell'])}, {mirror_kind(cn['din'], cn['dout'])}"
    if key in ("must_mirror", "certainly_empty"):
        km = tr.known_map()
        want = ("/", "\\") if key == "must_mirror" else ("empty",)
        hits = [c for c in p["cells"] if km.get(tuple(c)) in want]
        return cell_name(hits[0]) if len(hits) == 1 else None
    if key == "n_unvisited":
        blocks = {tuple(b) for b in scene["blocks"]}
        return str(tr.W * tr.H - len(blocks) - len(tr.visited() - blocks))
    if key == "cf_reverse":
        end = T[col]["end"]
        if end[0] != "exit":
            return None
        r = walk(tr.W, tr.H, tr.known_map(), end[1])
        return end_text(r["end"]) if r else None
    if key == "cf_new_port":
        r = walk(tr.W, tr.H, tr.known_map(), p["port"])
        return end_text(r["end"]) if r else None
    if key in ("cf_change", "cf_change_turns"):
        r = walk(tr.W, tr.H, _cf_map(tr, p), T[col]["port"])
        if r is None:
            return None
        return end_text(r["end"]) if key == "cf_change" else str(r["turns"])
    if key == "cf_fast":
        v = T[col]
        return fmt_s(v["t_start"] + (v["t_end"] - v["t_start"]) / 2)
    return None
