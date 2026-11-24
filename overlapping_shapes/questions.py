"""10-question, 5-rung ladder for overlapping_shapes (see common/ladder5.py).

Per seed up to three questions are built per level, then the level counts
are drawn at random among the 1-3 / 2-3 / 2-3 / 1-3 / 1-3 splits (total 10)
that the scene can fill (in practice every split). Each level has a pool of templates; which ones apply depends on
the scenario. Templates that need the motion are drawn three times as often
as ones that can be read off a single frame. Every question stores its
derivation_key and the params answers.py needs, plus params["min_fps"]: the
sampling rate needed to catch the events the question depends on."""
from __future__ import annotations
import itertools
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
FPS = sim.FPS
STYLE_WORD = {"solid": "a solid", "dashed": "a dashed", "dotted": "a dotted", "double": "a double-line"}
FILL_WORD = {"plain": "a plain fill with no pattern", "stripes": "stripes that run one way",
             "hatch": "crossed lines", "dots": "dots"}
DIRS = ["right", "down and to the right", "down", "down and to the left", "left", "up and to the left", "up",
        "up and to the right"]
NONE = "none of them"


def secs(x: float) -> str:
    return f"{x:g} s"


class Ctx:
    def __init__(self, scene):
        self.sc = scene
        self.kind = scene["scenario"]
        self.shapes = scene["shapes"]
        self.n = len(self.shapes)
        self.lab = [sim.label(s) for s in self.shapes]
        self.st = sim.statics(scene)
        self.mv = sim.movers(scene)
        self.nb = sim.neighbours(scene)
        self.prs = sim.pairs(scene)
        self.N = scene["n_frames"]
        self.A = scene["anim_f"]
        self.ev = scene["events"]
        self.order = [e["shape"] for e in self.ev]
        self.f = {e["shape"]: e["f"] for e in self.ev}
        if self.kind in ("arrive", "paper"):
            last = self.ev[-1]["f"] + self.A - 1
            self.state_win = (self.N - 1 - last) / FPS           # the final, complete board
            self.when = " at the end"
        elif self.kind == "depart":
            self.state_win = self.ev[0]["f"] / FPS                # the full board at the start
            self.when = " at the start"
        else:
            self.state_win = self.N / FPS
            self.when = ""
        self.pre = "Not counting the moving shapes, " if self.kind == "glide" else ""
        self.used = set()

    def gap(self, shapes) -> float:
        """Smallest gap (s) between the start frames of the given events."""
        fs = sorted(self.f[i] for i in shapes)
        return min((b - a for a, b in zip(fs, fs[1:])), default=self.N) / FPS

    def all_gap(self) -> float:
        return self.gap(self.order)

    def T_choices(self, rng, lo_f=0, hi_f=None, events=None):
        """Clock readings (multiples of 0.5 s) at least SEP_F frames from every change."""
        hi_f = self.N - 1 - sim.EDGE_F if hi_f is None else hi_f
        out = []
        for k in range(1, 40):
            f = k * 12
            if f < max(lo_f, sim.EDGE_F) or f > hi_f:
                continue
            if events is None:
                w = sim.stable_window(self.sc, f)
                if w is None or f - w[0] < sim.SEP_F or w[1] - f < sim.SEP_F:
                    continue
                out.append((f, (w[1] - w[0] + 1) / FPS))
            else:
                if all(abs(f / FPS - e) >= sim.SEP_F / FPS for e in events):
                    prev = max([e for e in events if e < f / FPS], default=0.0)
                    nxt = min([e for e in events if e > f / FPS], default=self.N / FPS)
                    out.append((f, nxt - prev))
        rng.shuffle(out)
        return out


def _labels(ctx, rng, correct: int, pool, k=3):
    others = [i for i in pool if i != correct]
    rng.shuffle(others)
    return [ctx.lab[i] for i in others[:k]]


def _q(ctx, rng, level, key, text, gt, distractors, motion, params, fps):
    params = dict(params)
    params["min_fps"] = round(fps, 2)
    sig = (key, tuple(sorted((k, str(v)) for k, v in params.items() if k != "min_fps")))
    if sig in ctx.used:
        return None
    if isinstance(gt, int) and not isinstance(gt, bool):
        opts, a = L.mc(rng, gt, distractors)
    else:
        if len({str(d) for d in distractors if str(d) != str(gt)}) < 3:
            return None
        opts, a = L.mc(rng, gt, distractors)
    if len(opts) != 4:
        return None
    ctx.used.add(sig)
    return L.question(level, key, text, gt, opts, a, motion, params)


def _ints(v, lo=0):
    return L.near_ints(v, lo)


# ===================================================================== L1
def t_n_shapes(ctx, rng):
    where = {"arrive": "on the board at the end of the video", "paper": "on the board at the end of the video",
             "depart": "on the board at the start of the video", "glide": "on the board, counting the moving ones"}
    return _q(ctx, rng, 1, "n_shapes", f"How many shapes are {where[ctx.kind]}?", ctx.n, _ints(ctx.n, 1), False,
              {}, sim.fps_for(ctx.state_win))


def t_n_types(ctx, rng):
    v = len({s["type"] for s in ctx.shapes})
    return _q(ctx, rng, 1, "n_types", "How many different kinds of shape appear in the video (circle, star, "
              "heart and so on), ignoring colour?", v, _ints(v, 1), False, {}, sim.fps_for(ctx.state_win))


def t_n_color(ctx, rng):
    c = rng.choice(sorted({s["color"] for s in ctx.shapes}))
    v = sum(s["color"] == c for s in ctx.shapes)
    return _q(ctx, rng, 1, "n_color", f"How many {c} shapes appear in the video?", v, _ints(v), False,
              {"color": c}, sim.fps_for(ctx.state_win))


def t_n_style(ctx, rng):
    st = rng.choice(sorted({s["style"] for s in ctx.shapes}))
    v = sum(s["style"] == st for s in ctx.shapes)
    return _q(ctx, rng, 1, "n_style", f"How many shapes have {STYLE_WORD[st]} border?", v, _ints(v), False,
              {"style": st}, sim.fps_for(ctx.state_win))


def t_n_fill(ctx, rng):
    fills = sorted({s["fill"] for s in ctx.shapes})
    if fills == ["plain"]:
        return None
    fl = rng.choice(fills)
    v = sum(s["fill"] == fl for s in ctx.shapes)
    return _q(ctx, rng, 1, "n_fill", f"How many shapes are filled with {FILL_WORD[fl]}?", v, _ints(v), False,
              {"fill": fl}, sim.fps_for(ctx.state_win))


def t_n_kind(ctx, rng):
    k = rng.choice(sorted({s["type"] for s in ctx.shapes}))
    v = sum(s["type"] == k for s in ctx.shapes)
    return _q(ctx, rng, 1, "n_kind", f"How many {sim.TYPE_PLURAL[k]} appear in the video?", v, _ints(v), False,
              {"kind": k}, sim.fps_for(ctx.state_win))


def t_first_shape(ctx, rng):
    if ctx.kind not in ("arrive", "paper"):
        return None
    x = ctx.order[0]
    return _q(ctx, rng, 1, "first_shape", "Which shape appears first?", ctx.lab[x],
              _labels(ctx, rng, x, ctx.order[1:4] + ctx.order[4:]), True, {},
              sim.fps_for(ctx.gap(ctx.order[:2])))


def t_last_shape(ctx, rng):
    if ctx.kind not in ("arrive", "paper"):
        return None
    x = ctx.order[-1]
    return _q(ctx, rng, 1, "last_shape", "Which shape is the last one to appear?", ctx.lab[x],
              _labels(ctx, rng, x, ctx.order[-4:-1] + ctx.order[:-4]), True, {},
              sim.fps_for(ctx.gap(ctx.order[-2:])))


def t_first_gone(ctx, rng):
    if ctx.kind != "depart":
        return None
    x = ctx.order[0]
    pool = ctx.order[1:] + [i for i in range(ctx.n) if i not in ctx.f]
    return _q(ctx, rng, 1, "first_gone", "Which shape is the first to disappear?", ctx.lab[x],
              _labels(ctx, rng, x, pool), True, {}, sim.fps_for(ctx.gap(ctx.order[:2])))


def t_n_left(ctx, rng):
    if ctx.kind != "depart":
        return None
    v = ctx.n - len(ctx.order)
    last = ctx.ev[-1]["f"] + ctx.A - 1
    return _q(ctx, rng, 1, "n_left", "How many shapes are still on the board at the end of the video?", v,
              [ctx.n] + _ints(v, 1), True, {}, sim.fps_for((ctx.N - 1 - last) / FPS))


def t_mover_which(ctx, rng):
    if ctx.kind != "glide":
        return None
    m = rng.choice(ctx.mv)
    text = "Which shape slides across the board?" if len(ctx.mv) == 1 else \
        "Which of these shapes slides across the board?"
    p = ctx.shapes[m]["path"]
    return _q(ctx, rng, 1, "mover_which", text, ctx.lab[m], _labels(ctx, rng, m, ctx.st), True, {"m": m},
              sim.fps_for((p["f1"] - p["f0"]) / FPS / 2))


def direction(shape) -> str | None:
    p = shape["path"]
    ang = math.degrees(math.atan2(p["y1"] - shape["y"], p["x1"] - shape["x"])) % 360
    k = int(round(ang / 45)) % 8
    d = abs((ang - 45 * k + 180) % 360 - 180)
    return DIRS[k] if d <= 12 else None


def t_mover_dir(ctx, rng):
    if ctx.kind != "glide":
        return None
    m = rng.choice(ctx.mv)
    d = direction(ctx.shapes[m])
    if d is None:
        return None
    k = DIRS.index(d)
    dis = [DIRS[(k + 4) % 8], DIRS[(k + 2) % 8], DIRS[(k + 6) % 8], DIRS[(k + 1) % 8]]
    p = ctx.shapes[m]["path"]
    return _q(ctx, rng, 1, "mover_dir", f"In which direction does the {ctx.lab[m]} move?", d, dis, True,
              {"m": m}, sim.fps_for((p["f1"] - p["f0"]) / FPS / 2))


# ===================================================================== L2
def t_overlap_which(ctx, rng):
    xs = [i for i in ctx.st if ctx.nb[i] and len(set(ctx.st) - ctx.nb[i] - {i}) >= 2]
    if not xs:
        return None
    x = rng.choice(xs)
    y = rng.choice(sorted(ctx.nb[x]))
    far = sorted(set(ctx.st) - ctx.nb[x] - {x})
    rng.shuffle(far)
    cands = sorted([y] + far[:3])
    when = " before anything moves" if ctx.kind == "glide" else ctx.when
    return _q(ctx, rng, 2, "overlap_which", f"Which of these shapes overlaps the {ctx.lab[x]}{when}?", ctx.lab[y],
              [ctx.lab[i] for i in far[:3]] + [NONE], False, {"x": x, "cands": cands}, sim.fps_for(ctx.state_win))


def t_overlap_none(ctx, rng):
    alone = [i for i in ctx.st if not ctx.nb[i]]
    busy = [i for i in ctx.st if ctx.nb[i]]
    if not alone or len(busy) < 3:
        return None
    x = rng.choice(alone)
    rng.shuffle(busy)
    cands = sorted([x] + busy[:3])
    text = (f"{ctx.pre}which of these shapes overlaps no other shape{ctx.when}?")
    text = text[0].upper() + text[1:]
    return _q(ctx, rng, 2, "overlap_none", text, ctx.lab[x], [ctx.lab[i] for i in busy[:3]], False,
              {"cands": cands}, sim.fps_for(ctx.state_win))


def t_earlier4(ctx, rng):
    if ctx.kind == "glide" or len(ctx.order) < 4:
        return None
    cands = sorted(rng.sample(ctx.order, 4))
    x = min(cands, key=lambda i: ctx.f[i])
    verb = "disappears" if ctx.kind == "depart" else "appears"
    return _q(ctx, rng, 2, "earlier4", f"Of these shapes, which one {verb} first?", ctx.lab[x],
              [ctx.lab[i] for i in cands if i != x], True, {"cands": cands}, sim.fps_for(ctx.gap(cands)))


def t_earlier2(ctx, rng):
    if ctx.kind == "glide":
        return None
    a, b = sorted(rng.sample(ctx.order, 2))
    if rng.random() < 0.5:
        a, b = b, a
    x = min((a, b), key=lambda i: ctx.f[i])
    verb = "disappears" if ctx.kind == "depart" else "appears"
    same = "they disappear at the same moment" if ctx.kind == "depart" else "they appear at the same moment"
    other = b if x == a else a
    neither = "neither of them disappears" if ctx.kind == "depart" else "neither of them appears"
    return _q(ctx, rng, 2, "earlier2", f"Which {verb} first: the {ctx.lab[a]} or the {ctx.lab[b]}?", ctx.lab[x],
              [ctx.lab[other], same, neither], True, {"pair": sorted([a, b])}, sim.fps_for(ctx.gap([a, b])))


def _present_before(ctx, x):
    """Arrive / paper: the shapes already on the board when x starts to appear."""
    k = ctx.order.index(x)
    return set(ctx.order[:k])


def t_arrival_partner(ctx, rng):
    if ctx.kind not in ("arrive", "paper"):
        return None
    xs = [x for x in ctx.order if len(ctx.nb[x] & _present_before(ctx, x)) == 1]
    if not xs:
        return None
    x = rng.choice(xs)
    (y,) = ctx.nb[x] & _present_before(ctx, x)
    later_nb = sorted(ctx.nb[x] - {y})
    pool = later_nb + [i for i in ctx.order if i not in ctx.nb[x] and i != x]
    dis = [ctx.lab[i] for i in pool if i != y][:3]
    return _q(ctx, rng, 2, "arrival_partner", f"When the {ctx.lab[x]} appears, it overlaps exactly one shape "
              f"that is already on the board. Which one?", ctx.lab[y], dis, True, {"x": x},
              sim.fps_for(ctx.gap([x] + sorted(ctx.nb[x]))))


def t_on_top(ctx, rng):
    if ctx.kind != "paper":
        return None
    pos = {i: k for k, i in enumerate(ctx.order)}
    xs = [x for x in ctx.order if any(pos[y] > pos[x] for y in ctx.nb[x])]
    rng.shuffle(xs)
    for x in xs:
        above = [y for y in ctx.nb[x] if pos[y] > pos[x]]
        below = [y for y in ctx.nb[x] if pos[y] < pos[x]]
        far = [i for i in ctx.order if i not in ctx.nb[x] and i != x]
        y = rng.choice(sorted(above))
        rng.shuffle(below)
        rng.shuffle(far)
        others = (below + far)[:3]
        if len(others) < 3:
            continue
        cands = sorted([y] + others)
        return _q(ctx, rng, 2, "on_top", f"Which of these shapes lies on top of part of the {ctx.lab[x]}?",
                  ctx.lab[y], [ctx.lab[i] for i in others], False, {"x": x, "cands": cands},
                  sim.fps_for(ctx.state_win))
    return None


def t_gone_after(ctx, rng):
    if ctx.kind != "depart":
        return None
    k = rng.randrange(len(ctx.order) - 1)
    x, y = ctx.order[k], ctx.order[k + 1]
    pool = ctx.order[k + 2:] + ctx.order[:k] + [i for i in range(ctx.n) if i not in ctx.f]
    return _q(ctx, rng, 2, "gone_after", f"Which shape is the next to disappear after the {ctx.lab[x]}?",
              ctx.lab[y], _labels(ctx, rng, y, pool), True, {"x": x}, sim.fps_for(ctx.gap([x, y])))


def t_passes_over(ctx, rng):
    if ctx.kind != "glide":
        return None
    m = rng.choice(ctx.mv)
    tt = sim.touches(ctx.sc, m)
    hit = {s for s, _, _ in tt}
    miss = [s for s in ctx.st if s not in hit]
    if not tt or len(miss) < 2:
        return None
    s, a, b = rng.choice(tt)
    rng.shuffle(miss)
    cands = sorted([s] + miss[:3])
    return _q(ctx, rng, 2, "passes_over", f"Which of these shapes does the moving {ctx.lab[m]} slide over?",
              ctx.lab[s], [ctx.lab[i] for i in miss[:3]] + [NONE], True, {"m": m, "cands": cands},
              sim.fps_for(b - a))


def t_first_touch(ctx, rng):
    if ctx.kind != "glide":
        return None
    m = rng.choice(ctx.mv)
    tt = sim.touches(ctx.sc, m)
    if len(tt) < 2:
        return None
    s = tt[0][0]
    pool = [r[0] for r in tt[1:]] + [i for i in ctx.st if i not in {r[0] for r in tt}]
    return _q(ctx, rng, 2, "first_touch", f"Which shape does the moving {ctx.lab[m]} overlap first?", ctx.lab[s],
              [ctx.lab[i] for i in pool[:3]], True, {"m": m}, sim.fps_for(tt[1][1] - tt[0][1]))


# ===================================================================== L3
def t_partners(ctx, rng):
    x = rng.choice([i for i in ctx.st if ctx.nb[i]] or ctx.st)
    v = len(ctx.nb[x])
    text = f"{ctx.pre}how many other shapes does the {ctx.lab[x]} overlap{ctx.when}?"
    return _q(ctx, rng, 3, "partners", text[0].upper() + text[1:], v, _ints(v), False, {"x": x},
              sim.fps_for(ctx.state_win))


def t_n_pairs(ctx, rng):
    v = len(ctx.prs)
    text = f"{ctx.pre}how many pairs of shapes overlap each other{ctx.when}?"
    return _q(ctx, rng, 3, "n_pairs", text[0].upper() + text[1:], v, _ints(v), False, {},
              sim.fps_for(ctx.state_win))


def t_max_depth(ctx, rng):
    if ctx.kind == "paper":
        return None
    d, _ = sim.max_depth(ctx.sc)
    text = (f"{ctx.pre}at the most crowded spot{ctx.when}, how many shapes are stacked on top of each other?")
    return _q(ctx, rng, 3, "max_depth", text[0].upper() + text[1:], d, _ints(d, 1), False, {},
              sim.fps_for(ctx.state_win))


def t_area_ratio(ctx, rng):
    if ctx.kind == "paper":
        return None
    areas = [sim.polygon(ctx.shapes[i]).area for i in ctx.st]
    r = max(areas) / min(areas)
    if r < 1.4 or r > 4.0 or abs(2 * r - round(2 * r)) > 0.2:      # >= 0.15 from a rounding boundary
        return None
    base = round(r * 2) / 2
    gt = f"{base:.1f}"
    dis = [f"{v:.1f}" for v in (base + 0.5, base - 0.5, base + 1.0, base - 1.0, base + 1.5) if v >= 1]
    text = f"{ctx.pre}about how many times larger (by area) is the biggest shape than the smallest one (to the " \
           f"nearest 0.5)?"
    return _q(ctx, rng, 3, "area_ratio", text[0].upper() + text[1:], gt, dis, False, {},
              sim.fps_for(ctx.state_win))


def t_count_at_T(ctx, rng):
    if ctx.kind == "glide":
        return None
    for f, win in ctx.T_choices(rng):
        v = len(sim.on_board(ctx.sc, f))
        return _q(ctx, rng, 3, "count_at_T", f"How many shapes are on the board when the clock reads "
                  f"{sim.timer_text(f)}?", v, _ints(v, 1), True, {"f": f}, sim.fps_for(win))
    return None


def t_time_of(ctx, rng):
    if ctx.kind == "glide":
        return None
    xs = list(ctx.order)
    rng.shuffle(xs)
    verb = "start to disappear" if ctx.kind == "depart" else "start to appear"
    for x in xs:
        t = ctx.f[x] / FPS
        margin = 0.5 - abs(t - round(t))
        if margin < 0.25 or round(t) < 1:
            continue
        v = int(round(t))
        dis = [secs(u) for u in (v + 1, v - 1, v + 2, v - 2, v + 3) if u >= 0]
        return _q(ctx, rng, 3, "time_of", f"To the nearest second, when does the {ctx.lab[x]} {verb}?", secs(v),
                  dis, True, {"x": x}, sim.fps_for(margin))
    return None


def t_n_touched(ctx, rng):
    if ctx.kind != "glide":
        return None
    m = rng.choice(ctx.mv)
    tt = sim.touches(ctx.sc, m)
    v = len(tt)
    short = min([b - a for _, a, b in tt], default=ctx.N / FPS)
    return _q(ctx, rng, 3, "n_touched", f"How many different shapes does the moving {ctx.lab[m]} overlap on its "
              f"way?", v, _ints(v), True, {"m": m}, sim.fps_for(short))


def _mover_events(ctx, m):
    p = ctx.shapes[m]["path"]
    ev = [p["f0"] / FPS, p["f1"] / FPS]
    for _, a, b in sim.touches(ctx.sc, m):
        ev += [a, b]
    return sorted(ev)


def t_overlap_at_T(ctx, rng):
    if ctx.kind != "glide":
        return None
    m = rng.choice(ctx.mv)
    p = ctx.shapes[m]["path"]
    for f, win in ctx.T_choices(rng, p["f0"], p["f1"], _mover_events(ctx, m)):
        v = sum(a < f / FPS < b for _, a, b in sim.touches(ctx.sc, m))
        return _q(ctx, rng, 3, "overlap_at_T", f"How many shapes is the moving {ctx.lab[m]} overlapping when the "
                  f"clock reads {sim.timer_text(f)}?", v, _ints(v), True, {"m": m, "f": f}, sim.fps_for(win))
    return None


# ===================================================================== L4
def arrivals_overlap(ctx, order):
    seen, out = set(), []
    for i in order:
        out.append(ctx.nb[i] & seen)
        seen.add(i)
    return out


def t_isolated_arrivals(ctx, rng):
    if ctx.kind not in ("arrive", "paper"):
        return None
    v = sum(1 for x in arrivals_overlap(ctx, ctx.order)[1:] if not x)
    alone = sum(1 for i in ctx.st if not ctx.nb[i])
    return _q(ctx, rng, 4, "isolated_arrivals", "Not counting the first shape, how many shapes overlapped nothing "
              "already on the board at the moment they appeared?", v, [alone] + _ints(v), True, {},
              sim.fps_for(ctx.all_gap()))


def t_most_new(ctx, rng):
    if ctx.kind not in ("arrive", "paper"):
        return None
    counts = [len(x) for x in arrivals_overlap(ctx, ctx.order)]
    top = max(counts)
    if counts.count(top) != 1 or top == 0:
        return None
    x = ctx.order[counts.index(top)]
    most_nb = max(ctx.st, key=lambda i: (len(ctx.nb[i]), -i))
    pool = ([most_nb] if most_nb != x else []) + [i for i in ctx.order if i != x and i != most_nb]
    return _q(ctx, rng, 4, "most_new", "Which shape, at the moment it appeared, overlapped the most shapes that "
              "were already on the board?", ctx.lab[x], [ctx.lab[i] for i in pool[:3]], True, {},
              sim.fps_for(ctx.all_gap()))


def pairs_on(ctx, members):
    return sum(1 for i, j in ctx.prs if i in members and j in members)


def t_arrival_count(ctx, rng):
    if ctx.kind not in ("arrive", "paper"):
        return None
    x = rng.choice(ctx.order[1:])
    v = len(ctx.nb[x] & _present_before(ctx, x))
    verb = "is laid down" if ctx.kind == "paper" else "appears"
    return _q(ctx, rng, 4, "arrival_count", f"At the moment the {ctx.lab[x]} {verb}, how many of the shapes "
              f"already on the board does it overlap?", v, [len(ctx.nb[x])] + _ints(v), True, {"x": x},
              sim.fps_for(ctx.gap([x] + sorted(ctx.nb[x]))))


def t_pairs_at_T(ctx, rng):
    if ctx.kind == "glide":
        return None
    for f, win in ctx.T_choices(rng):
        members = set(sim.on_board(ctx.sc, f))
        if len(members) == ctx.n and ctx.kind != "depart":
            continue
        v = pairs_on(ctx, members)
        return _q(ctx, rng, 4, "pairs_at_T", f"How many pairs of shapes overlap each other when the clock reads "
                  f"{sim.timer_text(f)}?", v, [len(ctx.prs)] + _ints(v), True, {"f": f}, sim.fps_for(win))
    return None


def t_pairs_before_last(ctx, rng):
    if ctx.kind != "arrive":
        return None
    v = len(ctx.prs) - len(ctx.nb[ctx.order[-1]])
    return _q(ctx, rng, 4, "pairs_before_last", "Just before the last shape appears, how many pairs of shapes "
              "overlap each other?", v, [len(ctx.prs)] + _ints(v), True, {},
              sim.fps_for(ctx.gap(ctx.order[-2:])))


def t_partners_when_gone(ctx, rng):
    if ctx.kind != "depart":
        return None
    k = rng.randrange(1, len(ctx.order))
    x = ctx.order[k]
    gone = set(ctx.order[:k])
    v = len(ctx.nb[x] - gone)
    return _q(ctx, rng, 4, "partners_when_gone", f"Just before the {ctx.lab[x]} starts to disappear, how many "
              f"of the shapes still on the board does it overlap?", v, [len(ctx.nb[x])] + _ints(v), True,
              {"x": x}, sim.fps_for(ctx.gap(ctx.order[:k + 1])))


def t_lost_pairs(ctx, rng):
    if ctx.kind != "depart":
        return None
    left = set(range(ctx.n)) - set(ctx.order)
    v = len(ctx.prs) - pairs_on(ctx, left)
    last = ctx.ev[-1]["f"] + ctx.A - 1
    return _q(ctx, rng, 4, "lost_pairs", "Between the start and the end of the video, how many pairs of "
              "overlapping shapes are lost?", v, [len(ctx.prs)] + _ints(v), True, {},
              sim.fps_for(min(ctx.state_win, (ctx.N - 1 - last) / FPS)))


def t_newly_alone(ctx, rng):
    if ctx.kind != "depart":
        return None
    left = set(range(ctx.n)) - set(ctx.order)
    v = sum(1 for i in left if ctx.nb[i] and not (ctx.nb[i] & left))
    last = ctx.ev[-1]["f"] + ctx.A - 1
    return _q(ctx, rng, 4, "newly_alone", "How many of the shapes still on the board at the end overlapped another "
              "shape at the start but overlap nothing at the end?", v, _ints(v), True, {},
              sim.fps_for(min(ctx.state_win, (ctx.N - 1 - last) / FPS)))


def t_last_touch(ctx, rng):
    if ctx.kind != "glide":
        return None
    m = rng.choice(ctx.mv)
    tt = sim.touches(ctx.sc, m)
    if len(tt) < 2:
        return None
    s = tt[-1][0]
    pool = [r[0] for r in tt[:-1]][::-1] + [i for i in ctx.st if i not in {r[0] for r in tt}]
    return _q(ctx, rng, 4, "last_touch", f"Which shape does the moving {ctx.lab[m]} overlap last?", ctx.lab[s],
              [ctx.lab[i] for i in pool[:3]], True, {"m": m}, sim.fps_for(tt[-1][1] - tt[-2][1]))


def t_touch_order(ctx, rng):
    if ctx.kind != "glide":
        return None
    m = rng.choice(ctx.mv)
    tt = sim.touches(ctx.sc, m)
    if len(tt) < 3:
        return None
    pick = sorted(rng.sample(range(len(tt)), 3))
    ids = [tt[k][0] for k in pick]
    fmt = lambda seq: ", then ".join(f"the {ctx.lab[i]}" for i in seq)  # noqa: E731
    gt = fmt(ids)
    perms = [list(p) for p in itertools.permutations(ids) if list(p) != ids]
    rng.shuffle(perms)
    names = ", ".join(f"the {ctx.lab[i]}" for i in sorted(ids))
    gap = min(tt[pick[1]][1] - tt[pick[0]][1], tt[pick[2]][1] - tt[pick[1]][1])
    return _q(ctx, rng, 4, "touch_order", f"In what order does the moving {ctx.lab[m]} start to overlap "
              f"{names}?", gt, [fmt(p) for p in perms], True, {"m": m, "ids": sorted(ids)}, sim.fps_for(gap))


def t_touches_between(ctx, rng):
    if ctx.kind != "glide":
        return None
    m = rng.choice(ctx.mv)
    p = ctx.shapes[m]["path"]
    tt = sim.touches(ctx.sc, m)
    if len(tt) < 2:
        return None
    ts = sorted(f for f, _ in ctx.T_choices(rng, p["f0"], p["f1"], _mover_events(ctx, m)))
    for a, b in itertools.combinations(ts, 2):
        v = sum(a / FPS < e < b / FPS for _, e, _ in tt)
        if v == 0 and rng.random() < 0.7:
            continue
        ev = sorted([a / FPS, b / FPS] + [e for _, e, _ in tt])
        gap = min(y - x for x, y in zip(ev, ev[1:]))
        return _q(ctx, rng, 4, "touches_between", f"Between the moments the clock reads {sim.timer_text(a)} and "
                  f"{sim.timer_text(b)}, how many shapes does the moving {ctx.lab[m]} start to overlap?", v,
                  [len(tt)] + _ints(v), True, {"m": m, "fa": a, "fb": b}, sim.fps_for(gap))
    return None


def t_longest_overlap(ctx, rng):
    if ctx.kind != "glide":
        return None
    m = rng.choice(ctx.mv)
    tt = sim.touches(ctx.sc, m)
    if len(tt) < 2:
        return None
    d = sorted(((b - a), s) for s, a, b in tt)
    if d[-1][0] - d[-2][0] < 0.3 or d[-1][0] < 1.15 * d[-2][0]:
        return None
    s = d[-1][1]
    pool = [x for _, x in d[:-1]][::-1] + [i for i in ctx.st if i not in {r[0] for r in tt}]
    return _q(ctx, rng, 4, "longest_overlap", f"Which shape does the moving {ctx.lab[m]} stay on top of for the "
              f"longest time?", ctx.lab[s], [ctx.lab[i] for i in pool[:3]], True, {"m": m},
              sim.fps_for((d[-1][0] - d[-2][0]) / 2))


def max_simul(intervals):
    best = 0
    for _, a, _ in intervals:
        best = max(best, sum(a2 <= a + 1e-9 and a < b2 for _, a2, b2 in intervals))
    return best


def t_max_simul(ctx, rng):
    if ctx.kind != "glide":
        return None
    m = rng.choice(ctx.mv)
    tt = sim.touches(ctx.sc, m)
    if len(tt) < 2:
        return None
    v = max_simul(tt)
    ev = _mover_events(ctx, m)
    gap = min(b - a for a, b in zip(ev, ev[1:]))
    return _q(ctx, rng, 4, "max_simul", f"What is the largest number of shapes the moving {ctx.lab[m]} overlaps "
              f"at the same moment?", v, [len(tt)] + _ints(v), True, {"m": m}, sim.fps_for(gap))


def t_which_at_T(ctx, rng):
    if ctx.kind != "glide":
        return None
    m = rng.choice(ctx.mv)
    p = ctx.shapes[m]["path"]
    tt = sim.touches(ctx.sc, m)
    for f, win in ctx.T_choices(rng, p["f0"], p["f1"], _mover_events(ctx, m)):
        now = [s for s, a, b in tt if a < f / FPS < b]
        if len(now) != 1:
            continue
        s = now[0]
        pool = [r[0] for r in tt if r[0] != s] + [i for i in ctx.st if i not in {r[0] for r in tt}]
        return _q(ctx, rng, 4, "which_at_T", f"Which shape is the moving {ctx.lab[m]} overlapping when the clock "
                  f"reads {sim.timer_text(f)}?", ctx.lab[s], [ctx.lab[i] for i in pool[:3]], True,
                  {"m": m, "f": f}, sim.fps_for(win))
    return None


def t_more_touched(ctx, rng):
    if ctx.kind != "glide" or len(ctx.mv) != 2:
        return None
    a, b = ctx.mv
    ca, cb = len(sim.touches(ctx.sc, a)), len(sim.touches(ctx.sc, b))
    gt = ctx.lab[a] if ca > cb else ctx.lab[b] if cb > ca else "they overlap the same number"
    opts = [ctx.lab[a], ctx.lab[b], "they overlap the same number", NONE]
    short = min([y - x for m in ctx.mv for _, x, y in sim.touches(ctx.sc, m)])
    return _q(ctx, rng, 4, "more_touched", "Of the two moving shapes, which one overlaps more different shapes "
              "on its way?", gt, [o for o in opts if o != gt], True, {}, sim.fps_for(short))


# ===================================================================== L5
def t_cf_remove_pairs(ctx, rng):
    x = rng.choice([i for i in ctx.st if ctx.nb[i]] or ctx.st)
    v = len(ctx.prs) - len(ctx.nb[x])
    when = {"arrive": "at the end", "paper": "at the end", "depart": "at the start",
            "glide": "before anything moves"}[ctx.kind]
    return _q(ctx, rng, 5, "cf_remove_pairs", f"If the {ctx.lab[x]} had never been on the board (everything else "
              f"the same), how many pairs of shapes would overlap {when}?", v, [len(ctx.prs)] + _ints(v), False,
              {"x": x}, sim.fps_for(ctx.state_win))


def t_cf_remove_isolated(ctx, rng):
    x = rng.choice([i for i in ctx.st if ctx.nb[i]] or ctx.st)
    v = sum(1 for i in ctx.st if i != x and not (ctx.nb[i] - {x}))
    alone = sum(1 for i in ctx.st if not ctx.nb[i])
    text = (f"If the {ctx.lab[x]} were taken away, how many of the other "
            f"{'non-moving ' if ctx.kind == 'glide' else ''}shapes would overlap no other shape"
            f"{' before anything moves' if ctx.kind == 'glide' else ctx.when}?")
    return _q(ctx, rng, 5, "cf_remove_isolated", text, v, [alone] + _ints(v), False, {"x": x},
              sim.fps_for(ctx.state_win))


def t_cf_reverse(ctx, rng):
    if ctx.kind != "arrive":
        return None
    v = sum(1 for y in arrivals_overlap(ctx, ctx.order[::-1])[1:] if not y)
    now = sum(1 for y in arrivals_overlap(ctx, ctx.order)[1:] if not y)
    return _q(ctx, rng, 5, "cf_reverse", "If the same shapes had appeared in the reverse order, how many of them "
              "(not counting the first) would overlap nothing already on the board at the moment they appeared?",
              v, [now] + _ints(v), True, {}, sim.fps_for(ctx.all_gap()))


def t_cf_skip_first(ctx, rng):
    if ctx.kind not in ("arrive", "paper"):
        return None
    rest = ctx.order[1:]
    v = sum(1 for y in arrivals_overlap(ctx, rest)[1:] if not y)
    now = sum(1 for y in arrivals_overlap(ctx, ctx.order)[1:] if not y)
    return _q(ctx, rng, 5, "cf_skip_first", "If the first shape to appear had never appeared, how many of the "
              "other shapes (not counting the new first one) would overlap nothing already on the board at the "
              "moment they appeared?", v, [now] + _ints(v), True, {}, sim.fps_for(ctx.all_gap()))


def t_cf_reverse_top(ctx, rng):
    if ctx.kind != "paper":
        return None
    pos = {i: k for k, i in enumerate(ctx.order)}
    xs = [x for x in ctx.order if any(pos[y] < pos[x] for y in ctx.nb[x])]
    rng.shuffle(xs)
    for x in xs:
        below = [y for y in ctx.nb[x] if pos[y] < pos[x]]
        above = [y for y in ctx.nb[x] if pos[y] > pos[x]]
        far = [i for i in ctx.order if i not in ctx.nb[x] and i != x]
        y = rng.choice(sorted(below))
        rng.shuffle(above)
        rng.shuffle(far)
        others = (above + far)[:3]
        if len(others) < 3:
            continue
        cands = sorted([y] + others)
        return _q(ctx, rng, 5, "cf_reverse_top", f"If the same shapes had been laid down in the reverse order, "
                  f"which of these shapes would lie on top of part of the {ctx.lab[x]}?", ctx.lab[y],
                  [ctx.lab[i] for i in others], True, {"x": x, "cands": cands}, sim.fps_for(ctx.state_win))
    return None


def t_cf_keep(ctx, rng):
    if ctx.kind != "depart":
        return None
    x = rng.choice(ctx.order)
    left = set(range(ctx.n)) - set(ctx.order)
    v = pairs_on(ctx, left | {x})
    last = ctx.ev[-1]["f"] + ctx.A - 1
    return _q(ctx, rng, 5, "cf_keep", f"If the {ctx.lab[x]} had never disappeared (everything else the same), how "
              f"many pairs of shapes would overlap at the end?", v, [pairs_on(ctx, left), len(ctx.prs)] + _ints(v),
              True, {"x": x}, sim.fps_for(min(ctx.state_win, (ctx.N - 1 - last) / FPS)))


def t_cf_keep_alone(ctx, rng):
    if ctx.kind != "depart":
        return None
    x = rng.choice(ctx.order)
    left = set(range(ctx.n)) - set(ctx.order)
    keep = left | {x}
    v = sum(1 for i in keep if not (ctx.nb[i] & keep))
    now = sum(1 for i in left if not (ctx.nb[i] & left))
    last = ctx.ev[-1]["f"] + ctx.A - 1
    return _q(ctx, rng, 5, "cf_keep_alone", f"If the {ctx.lab[x]} had never disappeared, how many of the shapes on "
              f"the board at the end would overlap no other shape?", v, [now] + _ints(v), True, {"x": x},
              sim.fps_for(min(ctx.state_win, (ctx.N - 1 - last) / FPS)))


def t_cf_stop(ctx, rng):
    if ctx.kind != "glide":
        return None
    m = rng.choice(ctx.mv)
    p = ctx.shapes[m]["path"]
    tt = sim.touches(ctx.sc, m)
    if len(tt) < 2:
        return None
    for f, win in ctx.T_choices(rng, p["f0"] + 12, p["f1"] - 12, _mover_events(ctx, m)):
        v = sum(a < f / FPS for _, a, _ in tt)
        if v == len(tt):
            continue
        return _q(ctx, rng, 5, "cf_stop", f"If the moving {ctx.lab[m]} had stopped for good when the clock read "
                  f"{sim.timer_text(f)}, how many different shapes would it have overlapped?", v,
                  [len(tt)] + _ints(v), True, {"m": m, "f": f}, sim.fps_for(win))
    return None


def t_cf_reverse_path(ctx, rng):
    if ctx.kind != "glide":
        return None
    m = rng.choice(ctx.mv)
    tt = sim.touches(ctx.sc, m)
    if len(tt) < 2:
        return None
    s = max(tt, key=lambda r: r[2])[0]
    pool = [r[0] for r in tt if r[0] != s] + [i for i in ctx.st if i not in {r[0] for r in tt}]
    exits = sorted(r[2] for r in tt)
    return _q(ctx, rng, 5, "cf_reverse_path", f"If the {ctx.lab[m]} had made the same trip backwards (starting "
              f"where it stops and ending where it starts), which shape would it overlap first?", ctx.lab[s],
              [ctx.lab[i] for i in pool[:3]], True, {"m": m}, sim.fps_for(exits[-1] - exits[-2]))


def t_cf_half_speed(ctx, rng):
    if ctx.kind != "glide":
        return None
    m = rng.choice(ctx.mv)
    p = ctx.shapes[m]["path"]
    tt = sim.touches(ctx.sc, m)
    t0, end = p["f0"] / FPS, (ctx.N - 1) / FPS
    slow = [t0 + 2 * (a - t0) for _, a, _ in tt]
    if not tt or any(abs(x - end) < sim.SEP_F / FPS for x in slow):
        return None
    v = sum(x < end for x in slow)
    if v == len(tt):
        return None
    short = min(b - a for _, a, b in tt)
    return _q(ctx, rng, 5, "cf_half_speed", f"If the {ctx.lab[m]} had moved at half its speed (starting at the "
              f"same moment, along the same line), how many different shapes would it have overlapped by the end "
              f"of the video?", v, [len(tt)] + _ints(v), True, {"m": m}, sim.fps_for(short))


LEVELS = {
    1: [t_n_shapes, t_n_types, t_n_color, t_n_style, t_n_fill, t_n_kind, t_first_shape, t_last_shape,
        t_first_gone, t_n_left, t_mover_which, t_mover_dir],
    2: [t_overlap_which, t_overlap_none, t_earlier4, t_earlier2, t_arrival_partner, t_on_top, t_gone_after, t_passes_over,
        t_first_touch],
    3: [t_partners, t_n_pairs, t_max_depth, t_area_ratio, t_count_at_T, t_time_of, t_n_touched, t_overlap_at_T],
    4: [t_isolated_arrivals, t_most_new, t_arrival_count, t_pairs_at_T, t_pairs_before_last, t_partners_when_gone, t_lost_pairs,
        t_newly_alone, t_last_touch, t_touch_order, t_max_simul, t_which_at_T, t_more_touched,
        t_touches_between, t_longest_overlap],
    5: [t_cf_remove_pairs, t_cf_remove_isolated, t_cf_reverse, t_cf_skip_first, t_cf_reverse_top, t_cf_keep,
        t_cf_keep_alone, t_cf_stop, t_cf_reverse_path, t_cf_half_speed],
}
STATIC = {t_n_shapes, t_n_types, t_n_color, t_n_style, t_n_fill, t_n_kind, t_overlap_which, t_overlap_none,
          t_on_top, t_partners, t_n_pairs, t_max_depth, t_area_ratio, t_cf_remove_pairs, t_cf_remove_isolated}


def _weighted_order(rng, temps):
    """Templates in a random order; motion templates are three times as likely to come early."""
    keyed = [(rng.random() ** (1.0 / (1 if t in STATIC else 3)), t) for t in temps]
    return [t for _, t in sorted(keyed, key=lambda kt: -kt[0])]


def build(scene: dict, rng: random.Random) -> list[dict]:
    ctx = Ctx(scene)
    pools = {}
    for lv in range(1, 6):                         # up to 3 questions per level, motion templates first
        got = []
        for rnd in range(4):                       # later rounds may reuse a template with new params
            for t in _weighted_order(rng, LEVELS[lv]):
                if len(got) == 3:
                    break
                if rnd == 0 and any(q["derivation_key"] == t.__name__[2:] for q in got):
                    continue
                q = t(ctx, rng)
                if q is not None:
                    got.append(q)
            if len(got) == 3:
                break
        pools[lv] = got
    combos = [c for c in L._count_combos() if sum(c) == L.TEN_TOTAL
              and all(k <= len(pools[lv]) for lv, k in zip(range(1, 6), c))]
    if not combos:
        raise RuntimeError(f"not enough questions for a {scene['scenario']} scene")
    counts = rng.choice(combos)
    qs = [q for lv, k in zip(range(1, 6), counts) for q in pools[lv][:k]]
    return L.number(qs)
