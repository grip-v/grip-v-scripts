"""Independent answers for nested_shapes.

This never reads the scene's event lists (spin segments, appear / vanish
frames, recolor list, mark add / remove frames, wall hits) or the stored
scenario order. It walks every frame of the renderer's drawing state
(render.state: centre, size, rotation, colour and visibility of every
outline; centre and visibility of every mark; the nest centre) and derives:

  * appear / vanish moments from visibility runs;
  * turning from frame-to-frame rotation, reduced modulo the outline's visible
    symmetry period (what a viewer can tell apart), grouped into runs;
    one-sample blips at a change are merged away;
  * colours by nearest palette colour, and colour changes from them;
  * which outline a mark sits on from geometry (nearest drawn outline);
  * nesting depth from drawn polygon areas;
  * wall hits as turning points of the nest centre's x / y track.
Counterfactuals are answered from those derived events (e.g. reversed
appearance order, one outline removed from the depth order, first turning
run kept, marks never removed, last colour change undone, hits before the
nest would cross a missing wall).

`track(scene)` is the full 24 fps track; `subsample(tr, k, phase)` keeps every
k-th frame, which the frame-sampling checks feed back through `answer`.
"""
from __future__ import annotations
import functools
import json
import math

import render
import sim
from questions import HEADINGS, ORDERS, ordinal, pair_text, pattern_text, seq_text

FOLDS = {"circle": 0, "ellipse": 2, "square": 4, "rectangle": 2, "diamond": 2, "triangle": 3, "pentagon": 5,
         "hexagon": 6, "octagon": 8, "star": 5, "cross": 4, "trapezoid": 1}


def _nearest(rgb):
    return min(sim.COLORS, key=lambda k: sum((rgb[j] - sim.COLORS[k][j]) ** 2 for j in range(3)))


def _area(pts):
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))) / 2


class Track:
    """Per-sample drawing state: frames F, ring and mark lists indexed [k][i]."""

    def __init__(self, scene, frames, sts):
        self.nf = scene["n_frames"]
        self.F = list(frames)
        st0 = sts[0]
        self.nr = len(st0["rings"])
        self.shape = [r["shape"] for r in st0["rings"]]
        self.tab = [r["tab"] for r in st0["rings"]]
        self.vis = [[st["rings"][k]["vis"] for st in sts] for k in range(self.nr)]
        self.rot = [[st["rings"][k]["rot"] for st in sts] for k in range(self.nr)]
        self.col = [[_nearest(st["rings"][k]["rgb"]) for st in sts] for k in range(self.nr)]
        self.geo = [[(st["rings"][k]["x"], st["rings"][k]["y"], st["rings"][k]["size"], st["rings"][k]["aspect"])
                     for st in sts] for k in range(self.nr)]
        self.nm = len(st0["marks"])
        self.mvis = [[st["marks"][m]["vis"] for st in sts] for m in range(self.nm)]
        self.mxy = [[(st["marks"][m]["x"], st["marks"][m]["y"]) for st in sts] for m in range(self.nm)]
        self.nest = [st["nest"] for st in sts]
        self.first_col = [_nearest(st0["rings"][k]["rgb"]) for k in range(self.nr)]
        self.area = [_area(render.ring_points(st0["rings"][k])) for k in range(self.nr)]
        self._sts = sts


@functools.lru_cache(maxsize=6)
def _full(key: str) -> Track:
    scene = json.loads(key)
    nf = scene["n_frames"]
    return Track(scene, range(nf), [render.state(scene, f) for f in range(nf)])


def track(scene) -> Track:
    key = json.dumps({k: scene[k] for k in ("scenario", "n_frames", "rings", "marks", "nest", "arena", "R", "style")},
                     sort_keys=True)
    return _full(key)


def subsample(tr: Track, k: int, phase: int) -> Track:
    idx = list(range(phase, tr.nf, k))
    out = object.__new__(Track)
    out.__dict__.update(tr.__dict__)
    out.F = [tr.F[i] for i in idx]
    for name in ("vis", "rot", "col", "geo", "mvis", "mxy"):
        setattr(out, name, [[row[i] for i in idx] for row in getattr(tr, name)])
    out.nest = [tr.nest[i] for i in idx]
    out._sts = [tr._sts[i] for i in idx]
    return out


# ---- derived events -------------------------------------------------------------------------
class View:
    def __init__(self, scene, tr: Track):
        self.s, self.t = scene, tr
        self.F = tr.F
        self.last = len(tr.F) - 1
        self.n = tr.nr
        self._runs = {}
        self._marks = None

    def label(self, k):
        if self.s["scenario"] == "recolor":
            return self.t.shape[k]
        return f"{self.t.first_col[k]} {self.t.shape[k]}"

    def on_at(self, k, i) -> bool:
        return self.t.vis[k][i]

    def appear(self, k):
        for i, v in enumerate(self.t.vis[k]):
            if v:
                return self.F[i]
        return None

    def vanish(self, k):
        seen = False
        for i, v in enumerate(self.t.vis[k]):
            seen = seen or v
            if seen and not v:
                return self.F[i]
        return None

    def ends(self):
        """Rings on screen in the last sample, outer to inner (by drawn area)."""
        on = [k for k in range(self.n) if self.t.vis[k][self.last]]
        return sorted(on, key=lambda k: -self.t.area[k])

    def period(self, k):
        """Symmetry period a viewer can see: 360 with a tab; none when the drawn outline's
        corners stick out less than 7 px past its flat parts (turning can't be seen)."""
        if self.t.tab[k]:
            return 360.0
        x, y, size, aspect = self.t.geo[k][0]
        unit = render.unit_shape(self.t.shape[k], aspect)
        rad = [math.hypot(px, py) for px, py in unit]
        bulge = size * (max(rad) - min(sim._seg_dist(unit[i], unit[(i + 1) % len(unit)]) for i in range(len(unit))))
        f = FOLDS[self.t.shape[k]] if bulge >= 7.0 else 0
        return 360.0 / f if f else 0.0

    def runs(self, k):
        """[(first frame, last frame, direction, n intervals, speeds)] of ring k's turning."""
        if k not in self._runs:
            self._runs[k] = self._make_runs(k)
        return self._runs[k]

    def _make_runs(self, k):
        P = self.period(k)
        iv = []
        for i in range(1, len(self.F)):
            if self.t.vis[k][i] and self.t.vis[k][i - 1]:
                d = self.t.rot[k][i] - self.t.rot[k][i - 1]
                if P:
                    d = (d + P / 2) % P - P / 2
                else:
                    d = 0.0
                sg = 0 if abs(d) < 1e-6 else (1 if d > 0 else -1)
                iv.append((self.F[i], sg, abs(d) / (self.F[i] - self.F[i - 1])))
        out = []
        for f, sg, sp in iv:
            if out and out[-1][2] == sg:
                a, _, _, n_, sps = out[-1]
                out[-1] = (a, f, sg, n_ + 1, sps + [sp])
            else:
                out.append((f, f, sg, 1, [sp]))
        # one-interval blips (a sample straddling a change) join the run before them
        changed = True
        while changed and len(out) > 1:
            changed = False
            for j, r in enumerate(out):
                if r[3] == 1:
                    if j > 0:
                        a, _, sg, n_, sps = out[j - 1]
                        out[j - 1] = (a, r[1], sg, n_ + 1, sps)
                    else:
                        a, b, sg, n_, sps = out[1]
                        out[1] = (r[0], b, sg, n_ + 1, sps)
                    del out[j]
                    if 0 < j < len(out) and out[j - 1][2] == out[j][2]:
                        a, _, sg, n_, sps = out[j - 1]
                        out[j - 1] = (a, out[j][1], sg, n_ + out[j][3], sps + out[j][4])
                        del out[j]
                    changed = True
                    break
        return tuple((a, b, sg, n_, tuple(sps)) for a, b, sg, n_, sps in out)

    def dirs(self, k):
        return [r[2] for r in self.runs(k)]

    def changes(self, k):
        return [r[0] for r in self.runs(k)[1:]]

    def turns(self, k):
        return any(r[2] for r in self.runs(k))

    def dir_at(self, k, ft):
        """Direction of ring k around frame ft (the first sample at or after ft)."""
        rs = self.runs(k)
        s = next((f for f in self.F if f >= ft), self.F[-1])
        for a, b, sg, _, _ in rs:
            if a <= s <= b:
                return sg
        nxt = [r for r in rs if r[0] > s]
        return nxt[0][2] if nxt else (rs[-1][2] if rs else None)

    def dir_end(self, k):
        rs = self.runs(k)
        return rs[-1][2] if rs else 0

    def speed(self, k):
        sps = [sp for r in self.runs(k) if r[2] for sp in r[4]]
        return sum(sps) / len(sps) if sps else 0.0

    def idx_at(self, ft):
        return next((i for i, f in enumerate(self.F) if f >= ft), self.last)

    def on_screen(self, ft):
        i = self.idx_at(ft)
        return [k for k in range(self.n) if self.t.vis[k][i]]

    def colors(self, k):
        """[(frame, colour)] of ring k: its first colour, then every change."""
        out = []
        for i, c in enumerate(self.t.col[k]):
            if not out or out[-1][1] != c:
                out.append((self.F[i], c))
        return out

    def color_at(self, k, ft):
        return self.t.col[k][self.idx_at(ft)]

    # marks
    def mark_info(self):
        if self._marks is None:
            self._marks = self._make_marks()
        return self._marks

    def _make_marks(self):
        out = []
        for m in range(self.t.nm):
            vis = self.t.mvis[m]
            if not any(vis):
                continue
            i0 = vis.index(True)
            rem = next((self.F[i] for i in range(i0, len(vis)) if not vis[i]), None)
            x, y = self.t.mxy[m][i0]
            best, bd = None, 1e9
            for k in range(self.n):
                if not self.t.vis[k][i0]:
                    continue
                pts = render.ring_points(self.t._sts[i0]["rings"][k])
                d = min(sim._seg_dist((ax - x, ay - y), (bx - x, by - y))
                        for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]))
                if d < bd:
                    best, bd = k, d
            out.append({"ring": best, "add": self.F[i0], "remove": rem})
        return tuple(tuple(sorted(d.items())) for d in out)

    def marks(self):
        return [dict(x) for x in self.mark_info()]

    # nest path
    def hits(self):
        out = []
        for ax, walls in ((0, ("left", "right")), (1, ("top", "bottom"))):
            prev = 0
            for i in range(1, len(self.F)):
                d = self.t.nest[i][ax] - self.t.nest[i - 1][ax]
                sg = 0 if abs(d) < 1e-9 else (1 if d > 0 else -1)
                if sg and prev and sg != prev:
                    out.append((self.F[i - 1], walls[1] if prev > 0 else walls[0]))
                if sg:
                    prev = sg
        return sorted(out)

    def heading_end(self):
        (x0, y0), (x1, y1) = self.t.nest[-2], self.t.nest[-1]
        return HEADINGS[(2 if y1 > y0 else 0) + (1 if x1 > x0 else 0)]


def _rd(f):
    return sim.fmt_reading(sim.tenths(f))


def _top(vals: dict):
    best = max(vals.values())
    ks = [k for k, v in vals.items() if v == best]
    return ks[0] if len(ks) == 1 else None


def _only(v: View, opts, pred):
    hits = [k for k in opts if pred(k)]
    return v.label(hits[0]) if len(hits) == 1 else None


def answer(key: str, p: dict, scene: dict, tr: Track | None = None):
    v = View(scene, tr or track(scene))
    nf_last = v.F[-1]
    ends = v.ends()
    lab = v.label
    rt = (lambda tn: sim.frame_of_reading(tn))
    # ---- L1
    if key == "n_end":
        return len(ends)
    if key == "color_count_end":
        return sum(1 for k in ends if v.t.col[k][v.last] == p["c"])
    if key == "shape_at_depth":
        return v.t.shape[ends[p["d"]]]
    if key == "depth_color":
        return v.t.col[ends[p["d"]]][v.last]
    if key == "first_appear":
        app = {k: v.appear(k) for k in range(v.n)}
        return lab(_top({k: -a for k, a in app.items()}))
    if key == "never_turns":
        return _only(v, p["opts"], lambda k: not v.turns(k))
    if key == "first_mark":
        ms = v.marks()
        m0 = min(ms, key=lambda m: m["add"])
        if sum(1 for m in ms if m["add"] == m0["add"]) > 1:
            return None
        return lab(m0["ring"])
    if key == "start_color":
        return v.t.col[p["x"]][0]
    if key == "end_color":
        return v.t.col[p["x"]][v.last]
    if key == "first_wall":
        return v.hits()[0][1]
    if key == "end_heading":
        return v.heading_end()
    # ---- L2
    if key == "inside_of":
        e = ends
        return lab(e[e.index(p["x"]) + 1])
    if key == "inside_count":
        return len(ends) - ends.index(p["x"]) - 1
    if key == "pair_dir":
        ft = rt(p["t"])
        return pair_text(v.dir_at(p["x"], ft), v.dir_at(p["y"], ft), lab(p["x"]), lab(p["y"]))
    if key == "appear_earliest":
        app = {k: v.appear(k) for k in p["opts"]}
        return lab(_top({k: -a for k, a in app.items()}))
    if key == "vanishes":
        return _only(v, p["opts"], lambda k: v.vanish(k) is not None)
    if key == "same_dir":
        ft = rt(p["t"])
        dx = v.dir_at(p["x"], ft)
        return _only(v, p["opts"], lambda k: v.dir_at(k, ft) == dx)
    if key == "fastest":
        return lab(_top({k: round(v.speed(k), 6) for k in p["opts"]}))
    if key == "first_change":
        fc = {k: (v.changes(k) or [10 ** 6])[0] for k in p["opts"]}
        return lab(_top({k: -f for k, f in fc.items()}))
    if key == "lost_mark":
        ms = v.marks()
        return _only(v, p["opts"], lambda k: any(m["ring"] == k and m["remove"] is not None for m in ms))
    if key == "more_added":
        ms = v.marks()
        return sum(m["ring"] == p["x"] for m in ms) - sum(m["ring"] == p["y"] for m in ms)
    if key == "more_at":
        i = v.idx_at(rt(p["t"]))
        ms = [m for m in range(v.t.nm) if v.t.mvis[m][i]]
        info = {mm: d for mm, d in zip([m for m in range(v.t.nm) if any(v.t.mvis[m])], v.marks())}
        return sum(info[m]["ring"] == p["x"] for m in ms) - sum(info[m]["ring"] == p["y"] for m in ms)
    if key == "wall_before":
        hs = v.hits()
        i = next(j for j, (_, w) in enumerate(hs) if w == p["wall"])
        return hs[i - 1][1]
    if key == "color_at_time":
        return v.color_at(p["x"], rt(p["t"]))
    if key == "which_recolors":
        return _only(v, p["opts"], lambda k: len(v.colors(k)) > 1)
    # ---- L3
    if key == "appear_time":
        return _rd(v.appear(p["x"]))
    if key == "change_time":
        ch = v.changes(p["x"])
        return _rd(ch[0]) if ch else None
    if key == "n_changes":
        return len(v.changes(p["x"]))
    if key == "count_at":
        return len(v.on_screen(rt(p["t"])))
    if key == "turning_at":
        ft = rt(p["t"])
        return sum(1 for k in v.on_screen(ft) if v.dir_at(k, ft))
    if key == "mark_time":
        return _rd(min(m["add"] for m in v.marks() if m["ring"] == p["x"]))
    if key == "marks_window":
        fa, fb = rt(p["a"]), rt(p["b"])
        return sum(fa <= m["add"] < fb for m in v.marks())
    if key == "recolor_time":
        cs = v.colors(p["x"])
        return _rd(cs[1][0]) if len(cs) > 1 else None
    if key == "n_recolors":
        return len(v.colors(p["x"])) - 1
    if key == "hit_time":
        return _rd(v.hits()[0][0])
    if key == "hit_wall_time":
        return _rd(next(f for f, w in v.hits() if w == p["wall"]))
    if key == "hits_window":
        fa, fb = rt(p["a"]), rt(p["b"])
        return sum(fa <= f < fb for f, _ in v.hits())
    if key == "n_hits":
        return len(v.hits())
    # ---- L4
    if key == "appear_pattern":
        app = {k: v.appear(k) for k in range(v.n)}
        if len(set(app.values())) == 1:
            return ORDERS["together"]
        seq = sorted(range(v.n), key=lambda k: app[k])
        if len(set(app.values())) != v.n:
            return None
        ar = [v.t.area[k] for k in seq]
        if all(a > b for a, b in zip(ar, ar[1:])):
            return ORDERS["outside-in"]
        if all(a < b for a, b in zip(ar, ar[1:])):
            return ORDERS["inside-out"]
        return ORDERS["mixed"]
    if key == "appear_after":
        app = {k: v.appear(k) for k in range(v.n)}
        return sum(1 for k in range(v.n) if app[k] > app[p["x"]])
    if key == "last_depth":
        app = {k: v.appear(k) for k in range(v.n)}
        last = max(range(v.n), key=lambda k: app[k])
        if sum(1 for k in app if app[k] == app[last]) > 1:
            return None
        depth = sorted(range(v.n), key=lambda k: -v.t.area[k])
        return ordinal(depth.index(last), v.n)
    if key == "max_on_screen":
        return max(sum(v.t.vis[k][i] for k in range(v.n)) for i in range(len(v.F)))
    if key == "spin_pattern":
        return pattern_text(v.dirs(p["x"]))
    if key == "cw_end":
        return sum(1 for k in ends if v.dir_end(k) == 1)
    if key == "most_changes":
        return lab(_top({k: len(v.changes(k)) for k in range(v.n)}))
    if key == "total_added":
        return len(v.marks())
    if key == "total_removed":
        return sum(m["remove"] is not None for m in v.marks())
    if key == "most_added":
        ms = v.marks()
        return lab(_top({k: sum(m["ring"] == k for m in ms) for k in range(v.n)}))
    if key == "color_sequence":
        return seq_text([c for _, c in v.colors(p["x"])])
    if key == "total_recolors":
        return sum(len(v.colors(k)) - 1 for k in range(v.n))
    if key == "hits_axis":
        walls = ("left", "right") if p["axis"] == "x" else ("top", "bottom")
        return sum(1 for _, w in v.hits() if w in walls)
    # ---- L5
    if key == "cf_stop_count":
        return len(v.on_screen(rt(p["t"]) - 1))
    if key == "cf_reverse_order":
        app = {k: v.appear(k) for k in range(v.n)}
        seq = sorted(range(v.n), key=lambda k: app[k])
        return lab(list(reversed(seq))[p["pos"]])
    if key == "cf_remove":
        e = [k for k in ends if k != p["x"]]
        return lab(e[e.index(p["y"]) + 1])
    if key == "cf_keep_dir":
        return sum(1 for k in ends if v.dirs(k)[0] == 1)
    if key == "cf_flip":
        dx = -v.dir_end(p["x"])
        return sum(1 for k in ends if k != p["x"] and v.dir_end(k) == dx)
    if key == "cf_stop_turning":
        ft = rt(p["t"])
        return sum(1 for k in v.on_screen(ft) if v.dir_at(k, ft) == 1)
    if key == "cf_no_removal":
        return sum(1 for m in v.marks() if m["ring"] == p["x"])
    if key == "cf_stop_marks":
        ft = rt(p["t"])
        return sum(1 for m in v.marks() if m["add"] < ft and (m["remove"] is None or m["remove"] >= ft))
    if key == "cf_no_last_recolor":
        ch = sorted((f, k) for k in range(v.n) for f, _ in v.colors(k)[1:])
        f_last, k_last = ch[-1]
        endc = {k: v.colors(k)[-1][1] for k in range(v.n)}
        cs = v.colors(k_last)
        endc[k_last] = cs[-2][1]
        return sum(1 for k in range(v.n) if endc[k] == p["c"])
    if key == "cf_stop_color":
        return v.color_at(p["x"], rt(p["t"]) - 1)
    if key == "cf_no_wall":
        hs = v.hits()
        cross = [f for f, w in hs if w == p["wall"]]
        return sum(1 for f, _ in hs if f < cross[0])
    if key == "cf_stop_hits":
        ft = rt(p["t"])
        return sum(1 for f, _ in v.hits() if f < ft)
    return None


__all__ = ["answer", "track", "subsample", "Track", "View", "ordinal"]
