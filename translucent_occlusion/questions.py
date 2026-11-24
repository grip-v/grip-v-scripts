"""10-question ladder for translucent_occlusion (common/ladder5.py, LADDER = "ten").

A shape is named by its colour and shape at the start ("the red triangle");
looks are unique within a clip, including the new look of any shape that
changes while hidden. Shapes that change colour or shape are only asked about
in the change questions and in counts. "Completely hidden" means the whole
shape is inside a panel. Times are clock readings (the clock is the only text
in the frame). Margins: hide/reveal events are >= 6 frames apart and >= 0.5 s
from the clip ends (sim.check); order comparisons need >= 8 frames (0.33 s);
clock options are >= 0.6 s apart; "when the clock shows X" questions keep
every event >= 6 frames outside the frames that show X. What-ifs re-run the
track with the change (sim.track(scene, mods)) and are only asked when the
re-run passes the same edge/gap checks. Every question has four distinct
options.
"""
from __future__ import annotations
import itertools
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
TEN_SPLITS = [c for c in itertools.product(*(range(lo, hi + 1) for lo, hi in L.TEN_RANGES)) if sum(c) == L.TEN_TOTAL]
FPS = sim.FPS
SIDES = ("left", "right", "top", "bottom")
PANEL_SHAPES = ("rectangle", "circle", "triangle", "hexagon")


# ------------------------------------------------------------------ helpers
def fps_for(gaps, floor=1):
    g = min(gaps) if gaps else 99.0
    return int(min(24, max(floor, math.ceil(1.0 / max(g, 1e-6) - 1e-9))))


def fmt_clock(tenths):
    return f"{tenths / 10:.1f} s"


def mc4(rng, correct, distractors, fmt=str):
    seen, opts = {fmt(correct)}, [correct]
    for d in distractors:
        if len(opts) == 4:
            break
        if fmt(d) not in seen:
            opts.append(d)
            seen.add(fmt(d))
    if len(opts) < 4:
        return None
    rng.shuffle(opts)
    texts = [fmt(o) for o in opts]
    return texts, L.LETTERS[texts.index(fmt(correct))]


def int_opts(rng, gt, lo, hi, extra=()):
    near = [v for v in (gt + 1, gt - 1, gt + 2, gt - 2, gt + 3, gt - 3, gt + 4, gt + 5) if lo <= v <= hi]
    rng.shuffle(near)
    near.sort(key=lambda v: abs(v - gt))
    ex = [e for e in extra if lo <= e <= hi and e != gt]
    return mc4(rng, gt, ex[:1] + near)


def clock_opts(rng, tenths, extra=(), step=6):
    cands = [round(e) for e in extra] + [tenths + d for d in (-12, -6, 6, 12, 18, -18, 9, -9)]
    keep = []
    for c in cands:
        if c >= 0 and all(abs(c - u) >= step for u in keep + [tenths]):
            keep.append(c)
    head, tail = keep[:len(extra)], keep[len(extra):]
    rng.shuffle(tail)
    return mc4(rng, fmt_clock(tenths), [fmt_clock(c) for c in head + tail])


def Q(key, params, text, gt, opts, motion, fps):
    if opts is None:
        return None
    return (key, dict(params, min_fps=fps), text, gt, opts[0], opts[1], motion)


def look_text(color, shape):
    return f"a {color} {shape}"


class Ctx:
    def __init__(self, scene, mods=()):
        self.scene = scene
        self.acts = scene["actors"]
        self.n = len(self.acts)
        self.frames = scene["frames"]
        self.panels = scene["panels"]
        self.lbl = [sim.label(a["color"], a["shape"]) for a in self.acts]
        self.morph = [bool(a["changes"]) and a["changes"][0][1] != "r" for a in self.acts]
        self.tr = sim.track(scene)
        self.st = sim.states(scene)
        self.ev = sim.events(scene)
        self.ef = [e[0] for e in self.ev]
        self.hide = {i: [f for f, j, k in self.ev if j == i and k == "hide"] for i in range(self.n)}
        self.rev = {i: [f for f, j, k in self.ev if j == i and k == "reveal"] for i in range(self.n)}
        self.ever = [i for i in range(self.n) if any(self.st[f][i] == 2 for f in range(self.frames))]
        self.never = [i for i in range(self.n) if i not in self.ever]
        self.plain = [i for i in range(self.n) if not self.morph[i]]
        if len(self.panels) == 1:
            self.pname = ["the panel"]
        else:
            self.pname = ["the left panel", "the right panel"]

    def gaps_around(self, f, pts=None):
        pts = [0] + (self.ef if pts is None else pts) + [self.frames - 1]
        return [abs(f - g) / FPS for g in pts if g != f] or [99.0]

    def all_gaps(self):
        pts = [0] + self.ef + [self.frames - 1]
        return [(b - a) / FPS for a, b in zip(pts, pts[1:]) if b > a] or [99.0]

    def clock_safe(self, tenths, ef=None, pad=6):
        ef = self.ef if ef is None else ef
        fr = [f for f in range(self.frames) if sim.clock_tenths(f) == tenths]
        if not fr or fr[0] < sim.EDGE_F or fr[-1] > self.frames - 1 - sim.EDGE_F:
            return None
        if any(fr[0] - pad < g <= fr[-1] + pad for g in ef):
            return None
        return fr

    def panel_of(self, i, f):
        a = self.tr[f][i]
        for k, p in enumerate(self.panels):
            if sim.cover_state(p, a[0], a[1], a[2]) == 2:
                return k
        return None

    def label_opts(self, rng, gt, pool):
        rest = [k for k in pool if k != gt]
        others = [k for k in range(self.n) if k != gt and k not in rest and not self.morph[k]]
        rng.shuffle(others)
        return mc4(rng, self.lbl[gt], [self.lbl[k] for k in rest + others])


def exit_side(c: Ctx, i):
    """Side of the panel where actor i is when it is next fully clear after its first reveal."""
    f0 = c.rev[i][0]
    f = next((g for g in range(f0, c.frames) if c.st[g][i] == 0), None)
    if f is None:
        return None
    p = c.panels[c.panel_of(i, f0 - 1)]
    cx, cy = ((p["x0"] + p["x1"]) / 2, (p["y0"] + p["y1"]) / 2) if p["kind"] == "rect" else (p["cx"], p["cy"])
    x, y = c.tr[f][i][:2]
    dx, dy = x - cx, y - cy
    if p["kind"] == "rect":
        hx, hy = (p["x1"] - p["x0"]) / 2, (p["y1"] - p["y0"]) / 2
        dx, dy = dx / hx, dy / hy
    if abs(dx) >= 1.5 * abs(dy):
        return "right" if dx > 0 else "left"
    if abs(dy) >= 1.5 * abs(dx):
        return "bottom" if dy > 0 else "top"
    return None


# ------------------------------------------------------------------ levels
def level1(c: Ctx, rng):
    out = []
    n = c.n
    hidden0 = any(c.st[0][i] == 2 for i in range(n))
    out.append(Q("n_total", {}, "How many shapes appear in the video in total, counting any that are hidden behind "
                 "a panel?", n, int_opts(rng, n, 1, 9), hidden0, 1))
    k = sum(c.st[0][i] == 0 for i in range(n))
    out.append(Q("n_clear_start", {}, "At the start of the video, how many shapes are completely outside the "
                 "panels (not behind a panel at all)?", k, int_opts(rng, k, 0, 9), False, 1))
    shapes = [a["shape"] for a in c.acts]
    colors = [a["color"] for a in c.acts]
    for i in range(n):
        a = c.acts[i]
        if shapes.count(a["shape"]) == 1 and c.st[0][i] == 0:
            pool = [x for x in sim.COLORS if x != a["color"]]
            rng.shuffle(pool)
            out.append(Q("color_of", {"x": i}, f"What colour is the {a['shape']} at the start of the video?",
                         a["color"], mc4(rng, a["color"], pool), False, 1))
        if colors.count(a["color"]) == 1 and c.st[0][i] == 0:
            pool = [x for x in sim.SHAPES if x != a["shape"]]
            rng.shuffle(pool)
            out.append(Q("shape_of", {"x": i}, f"What shape is the {a['color']} shape at the start of the video?",
                         a["shape"], mc4(rng, a["shape"], pool), False, 1))
    for k, p in enumerate(c.panels):
        gt = "rectangle" if p["kind"] == "rect" else "circle"
        pool = [s for s in PANEL_SHAPES if s != gt]
        out.append(Q("panel_shape", {"p": k}, f"What shape is {c.pname[k]}?", gt, mc4(rng, gt, pool), False, 1))
    if len(c.panels) == 2:
        out.append(Q("n_panels", {}, "How many translucent panels are there?", 2, int_opts(rng, 2, 0, 5), False, 1))
    return out


def level2(c: Ctx, rng):
    out = []
    hid = [i for i in c.plain if c.hide[i]]
    for x, y in itertools.combinations(hid, 2):
        fx, fy = c.hide[x][0], c.hide[y][0]
        if abs(fx - fy) < sim.ORDER_F:
            continue
        if rng.random() < 0.5:
            x, y = y, x
        gt = c.lbl[x] if c.hide[x][0] < c.hide[y][0] else c.lbl[y]
        opts = mc4(rng, gt, [c.lbl[x], c.lbl[y], "both at the same moment", "neither goes completely behind a panel"])
        out.append(Q("hid_first", {"x": x, "y": y}, f"Which goes completely behind a panel first: {c.lbl[x]} or "
                     f"{c.lbl[y]}?", gt, opts, True, fps_for([abs(fx - fy) / FPS])))
    out_ = [i for i in c.plain if c.rev[i]]
    for x, y in itertools.combinations(out_, 2):
        fx, fy = c.rev[x][0], c.rev[y][0]
        if abs(fx - fy) < sim.ORDER_F:
            continue
        if rng.random() < 0.5:
            x, y = y, x
        gt = c.lbl[x] if c.rev[x][0] < c.rev[y][0] else c.lbl[y]
        opts = mc4(rng, gt, [c.lbl[x], c.lbl[y], "both at the same moment", "neither comes out"])
        out.append(Q("out_first", {"x": x, "y": y}, f"Which starts to come out from behind a panel first: "
                     f"{c.lbl[x]} or {c.lbl[y]}?", gt, opts, True, fps_for([abs(fx - fy) / FPS])))
    for i in c.plain:
        if not c.rev[i] or not c.hide[i]:
            continue
        side = exit_side(c, i)
        if side is None:
            continue
        k = c.panel_of(i, c.rev[i][0] - 1)
        out.append(Q("exit_side", {"x": i}, f"{c.lbl[i][0].upper() + c.lbl[i][1:]} goes behind {c.pname[k]}. "
                     f"On which side of {c.pname[k]} does it come back out?", side,
                     mc4(rng, side, [s for s in SIDES if s != side]), True,
                     fps_for(c.gaps_around(c.rev[i][0], c.hide[i] + c.rev[i]))))
    if c.never:
        for gt in [i for i in c.plain if c.hide[i]]:
            pool = [c.lbl[k] for k in c.never if not c.morph[k]]
            opts = mc4(rng, c.lbl[gt], pool[:2] + ["none of these"] + pool[2:])
            out.append(Q("went_behind", {"x": gt}, "Which of these shapes goes completely behind a panel at some "
                         "point?", c.lbl[gt], opts, True, 1))
        hidden = [c.lbl[k] for k in c.ever if not c.morph[k]]
        for gt in [i for i in c.never if not c.morph[i]]:
            rng.shuffle(hidden)
            out.append(Q("never_behind", {"x": gt}, "Which of these shapes never goes completely behind a panel?",
                         c.lbl[gt], mc4(rng, c.lbl[gt], hidden), True, 1))
    if len(c.panels) == 2:
        for i in c.plain:
            if not c.hide[i]:
                continue
            k = c.panel_of(i, c.hide[i][0])
            gt = c.pname[k]
            opts = mc4(rng, gt, ["the left panel", "the right panel", "both panels", "neither panel"])
            out.append(Q("which_panel", {"x": i}, f"Behind which panel does {c.lbl[i]} become completely hidden?", gt,
                         opts, True, 1))
    return out


def level3(c: Ctx, rng):
    out = []
    for i in c.plain:
        if c.hide[i]:
            f = c.hide[i][0]
            tt = sim.clock_tenths(f)
            out.append(Q("hide_time", {"x": i}, f"What does the clock show when {c.lbl[i]} becomes completely hidden "
                         "behind a panel?", fmt_clock(tt), clock_opts(rng, tt), True, fps_for(c.gaps_around(f), 2)))
        if c.rev[i] and (c.hide[i] or c.st[0][i] == 2):
            f = c.rev[i][0]
            tt = sim.clock_tenths(f)
            out.append(Q("reveal_time", {"x": i}, f"What does the clock show when {c.lbl[i]} first starts to come out "
                         "from behind a panel?", fmt_clock(tt), clock_opts(rng, tt), True,
                         fps_for(c.gaps_around(f), 2)))
        if c.hide[i] and c.rev[i] and c.rev[i][0] > c.hide[i][0]:
            d = sim.clock_tenths(c.rev[i][0]) - sim.clock_tenths(c.hide[i][0])
            out.append(Q("hidden_for", {"x": i}, f"For how long is {c.lbl[i]} completely hidden (read the clock)?",
                         fmt_clock(d), clock_opts(rng, d), True,
                         fps_for(c.gaps_around(c.hide[i][0]) + c.gaps_around(c.rev[i][0]), 2)))
    for tt in range(5, sim.clock_tenths(c.frames - 1)):
        fr = c.clock_safe(tt)
        if not fr:
            continue
        k = sum(s == 2 for s in c.st[fr[0]])
        out.append(Q("n_hidden_at", {"t": tt}, f"When the clock shows {fmt_clock(tt)}, how many shapes are completely "
                     "hidden behind a panel?", k, int_opts(rng, k, 0, 9), True,
                     fps_for([abs(g - fr[0]) / FPS for g in c.ef] or [99.0], 2)))
    return out


def _seq_opts(rng, seq):
    wrong = []
    for i in range(len(seq) - 1):
        alt = list(seq)
        alt[i], alt[i + 1] = alt[i + 1], alt[i]
        wrong.append(alt)
    wrong.append(seq[::-1])
    wrong.append(seq[1:] + seq[:1])
    rng.shuffle(wrong)
    j = ", then ".join
    return mc4(rng, j(seq), [j(w) for w in wrong])


def level4(c: Ctx, rng):
    out = []
    for i in range(c.n):
        ch = c.acts[i]["changes"]
        if not ch:
            continue
        cf, what, val = ch[0]
        a = c.acts[i]
        k = c.panel_of(i, cf)
        lead = f"{c.lbl[i][0].upper() + c.lbl[i][1:]} goes behind {c.pname[k]}. "
        fps = fps_for(c.gaps_around(c.hide[i][0]) + c.gaps_around(c.rev[i][0]))
        if what == "r":
            gt = "bigger" if val > a["r"] else "smaller"
            out.append(Q("change_size", {"x": i}, lead + "When it comes back out, how has its size changed?", gt,
                         mc4(rng, gt, ["bigger", "smaller", "the same size", "it never comes out"]), True, fps))
            continue
        new = (val, a["shape"]) if what == "color" else (a["color"], val)
        others = [(col, a["shape"]) for col in sim.COLORS if col not in (a["color"], new[0])]
        others += [(a["color"], s) for s in sim.SHAPES if s not in (a["shape"], new[1])]
        rng.shuffle(others)
        gt = look_text(*new)
        out.append(Q("change_look", {"x": i}, lead + "What does it look like when it comes back out?", gt,
                     mc4(rng, gt, [look_text(a["color"], a["shape"])] + [look_text(*o) for o in others]), True, fps))
        gt = look_text(a["color"], a["shape"])
        out.append(Q("before_look", {"x": i}, f"{look_text(*new)[0].upper() + look_text(*new)[1:]} comes out from "
                     f"behind {c.pname[k]}. What did it look like when it went in?", gt,
                     mc4(rng, gt, [look_text(*new)] + [look_text(*o) for o in others]), True, fps))
    for kind, name, ev in (("reveal", "start to come out from behind", c.rev), ("hide", "become completely hidden "
                                                                                     "behind", c.hide)):
        who = sorted([i for i in c.plain if ev[i]], key=lambda i: ev[i][0])
        fs = [ev[i][0] for i in who]
        if 2 <= len(who) <= 4 and all(b - a >= sim.ORDER_F for a, b in zip(fs, fs[1:])):
            seq = [c.lbl[i] for i in who]
            out.append(Q(f"{kind}_order", {}, f"In what order do these shapes {name} a panel: "
                         f"{', '.join(sorted(seq))}?", ", then ".join(seq), _seq_opts(rng, seq), True,
                         fps_for([(b - a) / FPS for a, b in zip(fs, fs[1:])])))
    k = len(c.ever)
    out.append(Q("n_ever_hidden", {}, "How many different shapes are completely hidden behind a panel at some point "
                 "in the video?", k, int_opts(rng, k, 0, 9, [len(c.never)]), True, fps_for(c.all_gaps())))
    k = sum(s == 2 for s in c.st[-1])
    last = c.ef[-1] if c.ef else 0
    out.append(Q("n_hidden_end", {}, "At the end of the video, how many shapes are completely hidden behind a panel?",
                 k, int_opts(rng, k, 0, 9), True, fps_for([(c.frames - 1 - last) / FPS])))
    cnt = [sum(s == 2 for s in row) for row in c.st]
    m = max(cnt)
    runs = [len(list(g)) for v, g in itertools.groupby(cnt) if v == m]
    if m >= 2 and max(runs) >= sim.ORDER_F:
        out.append(Q("max_hidden", {}, "What is the largest number of shapes that are completely hidden behind panels "
                     "at the same moment?", m, int_opts(rng, m, 0, 9), True, fps_for([max(runs) / FPS])))
    return out


def cf_mods(c: Ctx):
    out = [(("nopanel",), None)]
    for i in c.plain:
        a = c.acts[i]
        if len(a["kf"]) >= 2 and a["kf"][-1][0] + FPS <= c.frames - sim.EDGE_F - 2:
            out.append((("delay", i), i))
    return out


def level5(c: Ctx, rng):
    out = []
    many = len(c.panels) > 1
    for m, x in cf_mods(c):
        mods = (m,)
        if sim.run_errors(c.scene, mods, strict=False):
            continue
        cst = sim.states(c.scene, mods)
        cev = sim.events(c.scene, mods)
        cef = [e[0] for e in cev]
        params = {"tag": m[0], "x": x}
        if m[0] == "nopanel":
            lead = f"Suppose the panel{'s were' if many else ' were'} removed (everything else the same). "
            k = c.n
            real = sum(s != 2 for s in c.st[-1])
            if real != k:
                out.append(Q("cf_n_seen_end", params, lead + "At the end of the video, how many shapes would you see "
                             "in full?", k, int_opts(rng, k, 1, 9, [real]), True, 1))
            continue
        lead = f"Suppose {c.lbl[x]} had started moving 1 s later (everything else the same). "
        gaps = [(b - a) / FPS for a, b in zip([0] + cef, cef + [c.frames - 1]) if b > a] or [99.0]
        safe = [t for t in range(5, sim.clock_tenths(c.frames - 1)) if c.clock_safe(t, cef)]
        diff = [t for t in safe if sum(s == 2 for s in cst[c.clock_safe(t, cef)[0]]) !=
                sum(s == 2 for s in c.st[c.clock_safe(t, cef)[0]])]
        for t in (diff or safe)[:1] + (diff[-1:] if len(diff) > 1 else []):
            fr = c.clock_safe(t, cef)
            k = sum(s == 2 for s in cst[fr[0]])
            real = sum(s == 2 for s in c.st[fr[0]])
            out.append(Q("cf_n_hidden_at", dict(params, t=t), lead + f"When the clock shows {fmt_clock(t)}, how many "
                         "shapes would be completely hidden behind a panel?", k, int_opts(rng, k, 0, 9, [real]), True,
                         fps_for([abs(g - fr[0]) / FPS for g in cef] or [99.0], 2)))
        k = sum(s == 2 for s in cst[-1])
        out.append(Q("cf_n_hidden_end", params, lead + "At the end of the video, how many shapes would be completely "
                     "hidden behind a panel?", k, int_opts(rng, k, 0, 9, [sum(s == 2 for s in c.st[-1])]), True,
                     fps_for(gaps)))
        revs = [(f, i) for f, i, kind in cev if kind == "reveal" and not c.morph[i]]
        if len(revs) >= 2 and revs[1][0] - revs[0][0] >= sim.ORDER_F:
            gt = revs[0][1]
            out.append(Q("cf_first_out", params, lead + "Which shape would be the first to start coming out from "
                         "behind a panel?", c.lbl[gt], c.label_opts(rng, gt, [i for _, i in revs[1:]]), True,
                         fps_for(gaps)))
    return out


def _pick(rng, cands, n):
    by_key = {}
    for cand in cands:
        by_key.setdefault(cand[0], []).append(cand)
    keys = sorted(by_key)
    rng.shuffle(keys)
    keys.sort(key=lambda k: k not in PRIORITY)
    chosen, used = [], set()
    while len(chosen) < n:
        avail = [k for k in keys if by_key[k]]
        if not avail:
            break
        k = ([k for k in avail if k not in used] or avail)[0]
        cand = by_key[k].pop(rng.randrange(len(by_key[k])))
        if any(cand[2] == ch[2] for ch in chosen):
            continue
        chosen.append(cand)
        used.add(k)
        keys.remove(k)
        keys.append(k)
    return chosen


PRIORITY = {"change_look", "before_look", "change_size", "cf_n_seen_end"}
MAKERS = {1: level1, 2: level2, 3: level3, 4: level4, 5: level5}


def candidates(scene, rng):
    c = Ctx(scene)
    made = {}
    for lv in range(1, 6):
        seen, ok = set(), []
        for q in MAKERS[lv](c, random.Random(rng.random())):
            if q is None:
                continue
            sig = (q[0], json.dumps(q[1], sort_keys=True))
            if sig not in seen:
                seen.add(sig)
                ok.append(q)
        made[lv] = ok
    return made


def feasible(scene) -> bool:
    made = candidates(scene, random.Random(0))
    texts = {lv: len({cand[2] for cand in made[lv]}) for lv in range(1, 6)}
    return all(texts[lv] >= L.TEN_RANGES[lv - 1][1] for lv in range(1, 6))


def build(scene: dict, rng: random.Random) -> list[dict]:
    made = candidates(scene, rng)
    texts = {lv: len({cand[2] for cand in made[lv]}) for lv in range(1, 6)}
    ok = [cmb for cmb in TEN_SPLITS if all(texts[lv] >= cmb[lv - 1] for lv in range(1, 6))]
    if not ok:
        raise ValueError("not enough question candidates for this scene")
    counts = rng.choice(ok)
    qs = []
    for lv in range(1, 6):
        for key, params, text, gt, opts, letter, motion in _pick(rng, made[lv], counts[lv - 1]):
            qs.append(L.question(lv, key, text, gt, opts, letter, motion, params, idx=len(qs) + 1))
    return qs
