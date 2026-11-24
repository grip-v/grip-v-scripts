"""Independent answers for moving_shadows.

sim.py finds coverage events by projecting outline vertices onto the ground
and bisecting in continuous time. This module never projects onto the
ground and never reads sim's event lists. It works per rendered frame from
what is drawn (render.state: the light centre and each object's outline):

  * a marker is shaded by an object in frame f when the segment from the
    marker (on the ground) to the light crosses one of the object's outline
    edges (orientation test); for the sun, which is at infinity, the segment
    runs from the marker to a far point in the sun's direction, read off the
    drawn disc's angle on its sky arc;
  * events are runs of shaded frames; times are the clock text of the first
    frame of a run (or of the first lit frame after it), durations are
    frame counts / 24;
  * a shadow's position (for "which way does it move") is found from the
    angles of the outline vertices as seen from the light;
  * what-if questions rebuild the per-frame track with an object removed,
    the light path run backwards, or the moving object held still, and
    re-run the same tests.

`answer(..., tr=...)` takes a decoded track instead (frame_check.py: light
centres, object offsets and marker shading read from MP4 pixels).
"""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render  # noqa: E402
import sim  # noqa: E402

FPS = 24
NONE_OPT = "no object's shadow"
TWO_OPT = "two objects' shadows"


def clock(f: int) -> str:
    return f"{(f * 10 // FPS) / 10:.1f} s"


def frame_of_clock(t: float) -> int:
    """First frame whose clock shows t (t on the 0.1 s grid)."""
    k = round(t * 10)
    f = 0
    while f * 10 // FPS < k:
        f += 1
    return f


def _orient(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


class Track:
    """Per-frame light, object outlines, and marker shading."""

    def __init__(self, scene, light, polys, shaded=None):
        self.scene = scene
        self.G = scene["ground"]
        self.light = np.asarray(light, dtype=float)                # (F, 2) drawn light centre
        lt = scene["light"]
        if lt["kind"] == "arc":
            # the sun is at infinity: read its elevation off the drawn disc on the sky arc; rays are parallel
            self.elev = np.arctan2((lt["cy"] - self.light[:, 1]) / lt["Ry"], (self.light[:, 0] - lt["cx"]) / lt["Rx"])
        else:
            self.elev = None
        self.polys = [np.asarray(p, dtype=float) for p in polys]    # per object (F, V, 2)
        self.F = len(self.light)
        self.mx = np.array(scene["markers"], dtype=float) + 0.5
        self.cov = [self._covers(p) for p in self.polys]            # per object (F, M) bool
        if shaded is None:
            sh = np.zeros((self.F, len(self.mx)), dtype=bool)
            for c in self.cov:
                sh |= c
            self.shaded = sh
        else:
            self.shaded = np.asarray(shaded, dtype=bool)

    def _covers(self, poly):
        px = self.mx[None, :, None]
        py = float(self.G)
        if self.elev is None:
            lx, ly = self.light[:, 0][:, None, None], self.light[:, 1][:, None, None]
        else:                                                       # a far point straight towards the sun
            far = 1e5
            lx = px + far * np.cos(self.elev)[:, None, None]
            ly = py - far * np.sin(self.elev)[:, None, None]
        ax, ay = poly[:, :, 0][:, None, :], poly[:, :, 1][:, None, :]
        b = np.roll(poly, -1, axis=1)
        bx, by = b[:, :, 0][:, None, :], b[:, :, 1][:, None, :]
        d1 = _orient(px, py, lx, ly, ax, ay)
        d2 = _orient(px, py, lx, ly, bx, by)
        d3 = _orient(ax, ay, bx, by, px, py)
        d4 = _orient(ax, ay, bx, by, lx, ly)
        hit = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))
        return hit.any(axis=2)

    # ---- runs and derived quantities
    @staticmethod
    def runs(flags):
        out, start = [], None
        for f, v in enumerate(list(flags) + [False]):
            if v and start is None:
                start = f
            elif not v and start is not None:
                out.append((start, f))
                start = None
        return out

    def m_runs(self, j):
        return self.runs(self.shaded[:, j])

    def c_runs(self, i, j):
        return self.runs(self.cov[i][:, j])

    def first(self, j):
        r = self.m_runs(j)
        return r[0][0] if r else None

    def frames_shaded(self, j):
        return int(self.shaded[:, j].sum())

    def firsts(self):
        return {j: self.first(j) for j in range(len(self.mx)) if self.first(j) is not None}

    def shadow_centre(self, i):
        """Centre of object i's ground shadow per frame, from the extreme angles of its outline seen from the light."""
        p = self.polys[i]
        if self.elev is not None:                                   # slide each vertex down the parallel rays
            xs = p[:, :, 0] - (self.G - p[:, :, 1]) / np.tan(self.elev)[:, None]
            return (xs.min(axis=1) + xs.max(axis=1)) / 2
        lx, ly = self.light[:, 0][:, None], self.light[:, 1][:, None]
        ang = np.arctan2(p[:, :, 0] - lx, p[:, :, 1] - ly)          # 0 = straight down
        h = self.G - self.light[:, 1]
        a, b = h * np.tan(ang.min(axis=1)), h * np.tan(ang.max(axis=1))
        return self.light[:, 0] + (a + b) / 2

    # ---- what-if variants
    def variant(self, remove=None, reverse=False, still=None):
        light = self.light[::-1] if reverse else self.light
        polys = [p for k, p in enumerate(self.polys) if k != remove]
        scene = self.scene if remove is None else dict(self.scene, casters=[o for k, o in enumerate(
            self.scene["casters"]) if k != remove])
        if still is not None:
            polys = [np.repeat(p[:1], self.F, axis=0) if k == still else p for k, p in enumerate(polys)]
        return Track(scene, light, polys)


def drawn_track(scene) -> Track:
    st = [render.state(scene, f) for f in range(sim.n_frames(scene))]
    light = [s["light"] for s in st]
    polys = [[s["casters"][i] for s in st] for i in range(len(scene["casters"]))]
    return Track(scene, light, polys)


@functools.lru_cache(maxsize=16)
def _cached(key: str) -> Track:
    return drawn_track(json.loads(key))


def decoded_track(scene, tr) -> Track:
    """tr = {"light": [(x, y)] per frame, "shift": [[(dx, dy)] per frame] per object, "shaded": [[bool] * M] per
    frame}: the object outline at t = 0 moved by the decoded shift."""
    polys = []
    for i, o in enumerate(scene["casters"]):
        base = np.asarray(sim.base_outline(o), dtype=float)
        polys.append(np.stack([base + np.asarray(d) for d in tr["shift"][i]]))
    return Track(scene, tr["light"], polys, tr.get("shaded"))


def _legs(xs, h):
    """Direction of each leg of a 1-D track: a new leg starts once it retreats h px from its extreme."""
    dirs, ext, d = [], xs[0], 0
    for x in xs:
        if d == 0:
            if abs(x - ext) >= h:
                d = 1 if x > ext else -1
                ext = x
        elif (x - ext) * d > 0:
            ext = x
        elif abs(x - ext) >= h:
            dirs.append(d)
            d, ext = -d, x
    if d:
        dirs.append(d)
    return tuple(dirs)


WAY = {(1,): "to the right", (-1,): "to the left", (-1, 1): "left, then back right", (1, -1): "right, then back left"}


def _name(j):
    return f"marker {j + 1}"


PATTERN_WORD = {"solid": "", "striped": "striped ", "dotted": "dotted ", "checked": "checked ", "ringed": "ringed ",
                "banded": "banded "}


def _lbl(scene, i):
    o = scene["casters"][i]
    return f"the {PATTERN_WORD[o['pattern']]}{o['color']} {o['shape']}"


def _half(x):
    return f"{round(x * 2) / 2:.1f} s"


def _first_of(T: Track):
    fs = T.firsts()
    return min(fs, key=fs.get) if fs else None


def answer(key: str, p: dict, scene: dict, tr=None):
    T = decoded_track(scene, tr) if tr is not None else _cached(render.static_key(scene))
    nm = len(scene["markers"])
    if key == "n_casters":
        return len(T.polys)
    if key == "light_dir":
        return "left to right" if T.light[-1, 0] > T.light[0, 0] else "right to left"
    if key == "first_marker":
        return _name(_first_of(T))
    if key == "n_ever":
        return len(T.firsts())
    if key == "never":
        nev = [j for j in p["ms"] if T.first(j) is None]
        return _name(nev[0]) if len(nev) == 1 else None
    if key in ("shaded_at_t", "n_shaded_at_t", "who_covers_at_t"):
        f = frame_of_clock(p["t"])
        if key == "n_shaded_at_t":
            return int(T.shaded[f].sum())
        if key == "shaded_at_t":
            on = [j for j in p["ms"] if T.shaded[f, j]]
            return _name(on[0]) if len(on) == 1 else None
        cov = [i for i in range(len(T.polys)) if T.cov[i][f, p["j"]]]
        return NONE_OPT if not cov else _lbl(scene, cov[0]) if len(cov) == 1 else TWO_OPT if len(cov) == 2 else None
    if key in ("moving_object", "move_dir"):
        disp = [p_[-1].mean(axis=0) - p_[0].mean(axis=0) for p_ in T.polys]
        mv = [i for i, d in enumerate(disp) if math.hypot(*d) > 20]
        if len(mv) != 1:
            return None
        if key == "moving_object":
            return _lbl(scene, mv[0])
        dx, dy = disp[mv[0]]
        return ("to the right" if dx > 0 else "to the left") if abs(dx) > abs(dy) else ("down" if dy > 0 else "up")
    if key == "caster_of":
        fs = {i: T.c_runs(i, p["j"])[0][0] for i in range(len(T.polys)) if T.c_runs(i, p["j"])}
        return _lbl(scene, min(fs, key=fs.get))
    if key == "shadow_way":
        return WAY.get(_legs(list(T.shadow_centre(p["i"])), 25.0))
    if key == "earliest_of":
        return _name(min(p["ms"], key=lambda j: T.first(j) if T.first(j) is not None else 10 ** 9))
    if key == "last_marker":
        fs = T.firsts()
        return _name(max(fs, key=fs.get))
    if key == "longest_of":
        return _name(max(p["ms"], key=T.frames_shaded))
    if key == "cover_time":
        return clock(T.m_runs(p["j"])[0][0])
    if key == "uncover_time":
        return clock(T.m_runs(p["j"])[0][1])
    if key == "shade_dur":
        return _half(T.frames_shaded(p["j"]) / FPS)
    if key == "gap_time":
        return _half((T.first(p["b"]) - T.first(p["a"])) / FPS)
    if key == "n_entries":
        return len(T.m_runs(p["j"]))
    if key == "order":
        return " → ".join(str(j + 1) for j in sorted(p["ms"], key=T.first))
    if key in ("most_shaded", "least_shaded"):
        ev = list(T.firsts())
        f = max if key == "most_shaded" else min
        return _name(f(ev, key=T.frames_shaded))
    if key == "n_casters_for":
        return sum(1 for i in range(len(T.polys)) if T.cov[i][:, p["j"]].any())
    if key == "caster_markers":
        return int(T.cov[p["i"]].any(axis=0).sum())
    if key == "total_entries":
        return sum(len(T.m_runs(j)) for j in range(nm))
    if key == "max_simul":
        return int(T.shaded.sum(axis=1).max())
    if key.startswith("cf_remove"):
        A = T.variant(remove=p["x"])
    elif key.startswith("cf_reverse"):
        A = T.variant(reverse=True)
    elif key.startswith("cf_still"):
        A = T.variant(still=p["i"])
    else:
        return None
    if key.endswith("_count"):
        return len(A.firsts())
    if key.endswith("_first"):
        return _name(_first_of(A))
    if key == "cf_remove_never":
        nev = [j for j in p["ms"] if A.first(j) is None]
        return _name(nev[0]) if len(nev) == 1 else None
    if key == "cf_remove_dur":
        return _half(A.frames_shaded(p["j"]) / FPS)
    return None
