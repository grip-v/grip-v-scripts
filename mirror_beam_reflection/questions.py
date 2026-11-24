"""10-question ladder for mirror_beam_reflection (common/ladder5.py, mode "ten").

Cells are named by the on-screen labels (column letter + row number, "C4");
beams by their colour; mirrors by their cell. Times are what the clock
shows. Answers come from sim.py's tracer; answers.py re-derives them from
the drawn frames (factual) or its own tracer (predictions, what-ifs).

Templates per level (derivation_key):
  L1 n_mirrors, n_slash, n_absorbers, n_special, entry_edge
  L2 first_mirror, turn_dir_first, head_cell_at_t, n_turns_by_t, n_heads_at_t, split_cell
  L3 time_turn_at, time_between_mirrors, launch_time, speed, time_between_turns, cells_between, n_lit_cells
  L4 exit_point, exit_time, n_turns_total, time_after_end, n_exits
  L5 cf_exit_point (remove / rotate / add a mirror, pivots frozen, two-sided, no absorbers),
     cf_exit_time (twice as fast, launched later), cf_n_exits
"""
from __future__ import annotations
import itertools
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ladder5 as L  # noqa: E402
import sim  # noqa: E402
from answers import (cell_name, end_text, fmt_s, fmt_clock, fmt_speed, clock_tenths, frame_of_clock,  # noqa: E402
                     DIR_WORDS, LETTERS)

LADDER = "ten"
TEN_SPLITS = [c for c in itertools.product(*(range(lo, hi + 1) for lo, hi in L.TEN_RANGES)) if sum(c) == L.TEN_TOTAL]
FPS = sim.FPS
EV = 0.3
RULES = ("(Rules: a pulse moves one cell at a time at a steady speed; on reaching a mirror's cell it turns 90 degrees "
         "off the mirror's diagonal; a half-mirror (dashed) lets it through AND sends a copy off the diagonal; a "
         "pulse hitting the dark hatched back of a one-sided mirror, or a dark block, stops there; a pivoting mirror "
         "(round hub) acts with the diagonal it shows when the pulse arrives.)")


def fps_for(gaps, floor=1):
    g = min(gaps) if gaps else 99.0
    return int(min(24, max(floor, math.ceil(1.0 / g - 1e-9))))


def align(t):
    return 1 if abs(t - round(t)) < 1e-9 else 2


def safe1(v):
    x = v * 10 - math.floor(v * 10)
    return abs(x - 0.5) >= 0.15


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


def int_opts(rng, gt, lo, hi):
    near = [v for v in (gt + 1, gt - 1, gt + 2, gt - 2, gt + 3, gt - 3, gt + 4) if lo <= v <= hi]
    rng.shuffle(near)
    near.sort(key=lambda v: abs(v - gt))
    return mc4(rng, gt, near)


def secs_opts(rng, gt, cands, spacing, fmt=fmt_s):
    g = round(gt, 1)
    cands = [round(c, 1) for c in cands]
    rng.shuffle(cands)
    keep = []
    for c in cands:
        if c >= 0.1 and all(abs(c - u) >= spacing - 1e-9 for u in keep + [g]):
            keep.append(c)
    return mc4(rng, g, keep, fmt)


def clock_opts(rng, tenths, extra=()):
    cands = [tenths + d for d in (-15, -10, -5, 5, 10, 15, 20, -20)] + [round(e) for e in extra]
    rng.shuffle(cands)
    keep = []
    for c in cands:
        if c >= 0 and all(abs(c - u) >= 5 for u in keep + [tenths]):
            keep.append(c)
    return mc4(rng, tenths, keep, fmt_clock)


def event_ok(te):
    x = te * FPS
    fa = math.ceil(x - 1e-9)
    return 0.25 <= x - math.floor(x) <= 0.75 and clock_tenths(fa - 1) == clock_tenths(fa)


def Q(key, params, text, gt, opts, motion, fps):
    if opts is None:
        return None
    return (key, dict(params, min_fps=fps), text, gt, opts[0], opts[1], motion)


# ------------------------------------------------------------------ context
class Ctx:
    def __init__(self, scene):
        self.s = scene
        self.res = sim.trace(sim.world_of(scene))
        self.dur = sim.duration(scene)
        self.t_last = (scene["frames"] - 1) / FPS
        self.step = scene["step"]
        self.colours = [e["colour"] for e in scene["emitters"]]
        self.splits = sorted(e[0] for p in self.res["pulses"] for e in p["events"] if e[1] == "split")
        self.split_t = self.splits[0] if self.splits else math.inf
        self.mcells = [tuple(m["cell"]) for m in scene["mirrors"]]

    def pulses(self, colour):
        return [p for p in self.res["pulses"] if p["colour"] == colour]

    def turns(self, colour):
        """[(t, cell, din, dout)] reflections of the colour's first pulse (no splits), in time order."""
        p = self.pulses(colour)[0]
        out = []
        for t, kind, cell in p["events"]:
            if kind != "bounce":
                continue
            i = next(k for k, q in enumerate(p["pts"]) if abs(q[0] - t) < 1e-9)
            a, b, c = p["pts"][i - 1], p["pts"][i], p["pts"][i + 1]
            sg = lambda v: (v > 1e-9) - (v < -1e-9)
            din = (sg(b[1] - a[1]), sg(b[2] - a[2]))
            dout = (sg(c[1] - b[1]), sg(c[2] - b[2]))
            out.append((t, tuple(cell), din, dout))
        return out

    def seen_turns(self, colour):
        """Turns in the clip, at least 0.5 s from both ends and before any split."""
        return [x for x in self.turns(colour) if sim.EDGE_S <= x[0] <= self.dur - sim.EDGE_S and x[0] < self.split_t - EV]

    def single(self, colour):
        return len(self.pulses(colour)) == 1

    def end(self, colour):
        return self.pulses(colour)[0]["end"]

    def grid_t(self):
        return [k / 2 for k in range(2, int(2 * (self.dur - 1.0)) + 1)]

    def nm(self, cell):
        return cell_name(cell)


def _end_tuple(e):
    return ("exit", e[1], e[2]) if e[0] == "exit" else ("absorbed", tuple(e[1]))


def end_distractors(c: Ctx, colour, gt_end):
    """Plausible wrong endpoints: straight on from the head, neighbouring exit points, other mirrors' changes."""
    import answers as A
    out = []
    p = c.pulses(colour)[0]
    W, H = c.s["cols"], c.s["rows"]
    h = sim.head_at(p, c.t_last)
    if h is not None:
        later = [q for q in p["pts"] if q[0] > c.t_last and (abs(q[1] - h[0]) > 1e-9 or abs(q[2] - h[1]) > 1e-9)]
        if later:                                       # the next waypoint the head is not already sitting on
            dx, dy = later[0][1] - h[0], later[0][2] - h[1]
            d = (int(math.copysign(1, dx)) if abs(dx) > 1e-9 else 0, int(math.copysign(1, dy)) if abs(dy) > 1e-9 else 0)
            x, y = math.floor(h[0]), math.floor(h[1])
            while 0 <= x < W and 0 <= y < H:
                x, y = x + d[0], y + d[1]
            edge = "left" if x < 0 else "right" if x >= W else "top" if y < 0 else "bottom"
            out.append(("exit", edge, y if edge in ("left", "right") else x))
    if gt_end[0] == "exit":
        n = H if gt_end[1] in ("left", "right") else W
        for k in (1, -1, 2, -2):
            if 0 <= gt_end[2] + k < n:
                out.append(("exit", gt_end[1], gt_end[2] + k))
    for cell in c.mcells:
        for kw in ({"remove": cell}, {"rotate": cell}):
            r = A.run(c.s, **kw)
            if r:
                o = [x for x in r if x["colour"] == colour]
                if len(o) == 1:
                    out.append(o[0]["end"])
    for a in c.s["absorbers"]:
        out.append(("absorbed", tuple(a)))
    return [end_text(e) for e in out]


# ------------------------------------------------------------------ L1
SPECIAL = {"one": "have a dark hatched backing on one side", "split": "are drawn dashed (half-mirrors)",
           "flip": "have a round pivot hub at their centre"}


def level1(c: Ctx, rng):
    out = []
    n = len(c.s["mirrors"])
    out.append(Q("n_mirrors", {}, "How many mirrors are on the board?", n, int_opts(rng, n, 0, 40), False, 1))
    k = sum(1 for m in c.s["mirrors"] if m["kind"] == "/")
    out.append(Q("n_slash", {}, "At the start of the clip, how many mirrors lean like '/' (rising to the right)?", k,
                 int_opts(rng, k, 0, 40), False, 1))
    k = len(c.s["absorbers"])
    out.append(Q("n_absorbers", {}, "How many dark blocks (which stop the light) are on the board?", k,
                 int_opts(rng, k, 0, 9), False, 1))
    for typ, words in SPECIAL.items():
        k = sum(1 for m in c.s["mirrors"] if m["type"] == typ)
        if k:
            out.append(Q("n_special", {"type": typ}, f"How many mirrors {words}?", k, int_opts(rng, k, 0, 20), False, 1))
    for col in c.colours:
        e = next(em for em in c.s["emitters"] if em["colour"] == col)["edge"]
        out.append(Q("entry_edge", {"colour": col}, f"From which edge of the board does the {col} beam come in?", e,
                     mc4(rng, e, [x for x in ("left", "right", "top", "bottom") if x != e]), False, 1))
    return out


# ------------------------------------------------------------------ L2
def _cell_opts(c: Ctx, rng, gt, extra=()):
    cand = [x for x in list(extra) + c.mcells if x != gt]
    near = [(gt[0] + dx, gt[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    cand += [x for x in near if 0 <= x[0] < c.s["cols"] and 0 <= x[1] < c.s["rows"]]
    cand = list(dict.fromkeys(cand))
    rng.shuffle(cand)
    cand.sort(key=lambda x: abs(x[0] - gt[0]) + abs(x[1] - gt[1]))
    return mc4(rng, gt, cand[:6], cell_name)


def level2(c: Ctx, rng):
    out = heads_at_t(c, rng)
    for col in c.colours:
        ts = c.seen_turns(col)
        if ts and ts[0] == c.turns(col)[0]:
            gt = ts[0][1]
            out.append(Q("first_mirror", {"colour": col}, f"At which mirror does the {col} beam turn for the first time?",
                         cell_name(gt), _cell_opts(c, rng, gt, [x[1] for x in c.turns(col)[1:3]]), False, 1))
            w = DIR_WORDS[ts[0][3]]
            out.append(Q("turn_dir_first", {"colour": col}, f"After its first turn, which way does the {col} beam travel?",
                         w, mc4(rng, w, [x for x in ("up", "down", "left", "right") if x != w]), False, 1))
    for col in c.colours:
        p = c.pulses(col)[0]
        ts = [t for t in c.grid_t() if t < c.split_t - EV]
        rng.shuffle(ts)
        made = 0
        for t in ts:
            if made == 2:
                break
            fr = [f for f in range(frame_of_clock(t), frame_of_clock(t) + 4) if clock_tenths(f) == round(t * 10)]
            hs = [sim.head_at(p, f / FPS) for f in fr]
            if any(h is None for h in hs):
                continue
            cells = {(math.floor(h[0]), math.floor(h[1])) for h in hs}
            if len(cells) != 1 or any(min(h[0] % 1, 1 - h[0] % 1, h[1] % 1, 1 - h[1] % 1) < 0.2 for h in hs):
                continue
            if any(len([q for q in c.res["pulses"] if q["colour"] == col and sim.head_at(q, f / FPS)]) != 1 for f in fr):
                continue
            gt = cells.pop()
            made += 1
            out.append(Q("head_cell_at_t", {"colour": col, "t": t}, f"At {t:.1f} s on the clock, in which cell is the "
                         f"head of the {col} beam?", cell_name(gt), _cell_opts(c, rng, gt), True,
                         align(t)))
        allt = [x[0] for x in c.turns(col)]
        for t in rng.sample(c.grid_t(), min(6, len(c.grid_t()))):
            if any(abs(e - t) < EV or abs(e - t - 0.1) < EV for e in allt) or t > c.split_t - EV:
                continue
            k = sum(1 for e in allt if e < t)
            gaps = [abs(e - t) for e in allt]
            out.append(Q("n_turns_by_t", {"colour": col, "t": t}, f"By {t:.1f} s on the clock, how many times has the "
                         f"{col} beam turned at a mirror?", k, int_opts(rng, k, 0, 30), True,
                         max(align(t), fps_for(gaps))))
            break
    return out


def heads_at_t(c: Ctx, rng):
    out = []
    starts = [p["pts"][0][0] for p in c.res["pulses"]]
    ends = [p["end"][-1] for p in c.res["pulses"]]
    for t in rng.sample(c.grid_t(), min(8, len(c.grid_t()))):
        ev = starts + ends
        if any(abs(e - t) < EV or abs(e - t - 0.1) < EV for e in ev):
            continue
        k = sum(1 for p in c.res["pulses"] if sim.head_at(p, t) is not None)
        if k == 0:
            continue
        out.append(Q("n_heads_at_t", {"t": t}, f"At {t:.1f} s on the clock, how many moving light heads (round dots at "
                     "the front of a beam) are on the board?", k, int_opts(rng, k, 0, 9), True,
                     max(align(t), fps_for([abs(e - t) for e in ev]))))
        if len(out) == 2:
            break
    if c.splits and sim.EDGE_S <= c.splits[0] <= c.dur - sim.EDGE_S:
        cell = next(tuple(e[2]) for p in c.res["pulses"] for e in p["events"] if e[1] == "split" and e[0] == c.splits[0])
        out.append(Q("split_cell", {}, "At which mirror does the light first split into two beams?", cell_name(cell),
                     _cell_opts(c, rng, cell, [m for m in c.mcells]), False, 1))
    return out


# ------------------------------------------------------------------ L3
def level3(c: Ctx, rng):
    out = []
    first_hit = {}
    for p in c.res["pulses"]:
        for te, kind, cell in p["events"]:
            first_hit[tuple(cell)] = min(first_hit.get(tuple(cell), math.inf), te)
    for col in c.colours:
        ts = c.seen_turns(col)
        first_hit = {}
        for p in c.res["pulses"]:
            for te, kind, cell in p["events"]:
                first_hit[tuple(cell)] = min(first_hit.get(tuple(cell), math.inf), te)
        hits = [(te, cell, None, None) for cell, te in first_hit.items()
                if cell in c.mcells and sim.EDGE_S <= te <= c.dur - sim.EDGE_S]
        if col != c.colours[0]:
            hits = []
        for t, cell, din, dout in rng.sample(hits, min(3, len(hits))):
            if not event_ok(t) or first_hit[cell] != t:
                continue
            tenths = clock_tenths(math.ceil(t * FPS - 1e-9))
            out.append(Q("time_turn_at", {"cell": list(cell)}, f"When does light first reach the mirror in cell "
                         f"{cell_name(cell)}? Give the clock reading.", fmt_clock(tenths),
                         clock_opts(rng, tenths, [10 * (t + c.step), 10 * (t - c.step)]), True, 4))
        for k in range(len(ts) - 1):
            a, b = ts[k], ts[k + 1]
            if c.turns(col).index(a) + 1 != c.turns(col).index(b):
                continue
            dt = b[0] - a[0]
            cells = round(dt / c.step)
            if safe1(dt):
                out.append(Q("time_between_turns", {"colour": col, "k": c.turns(col).index(a)},
                             f"How much time passes between the {col} beam's turn at {cell_name(a[1])} and its next turn "
                             f"at {cell_name(b[1])}?", fmt_s(dt),
                             secs_opts(rng, dt, [dt + c.step, dt - c.step, dt * 2, dt / 2, dt + 0.5, dt - 0.5, dt + 1],
                                       max(0.3, 0.15 * dt)), True, 4))
            out.append(Q("cells_between", {"colour": col, "k": c.turns(col).index(a)},
                         f"How many cells does the {col} beam move from its turn at {cell_name(a[1])} to its next turn at "
                         f"{cell_name(b[1])}?", cells, int_opts(rng, cells, 1, 12), False, 1))
            break
    pairs = []
    for p in c.res["pulses"]:
        evs = [(t, tuple(cell)) for t, kind, cell in p["events"] if kind in ("bounce", "split")]
        for (ta, a), (tb, b) in zip(evs, evs[1:]):
            if first_hit[a] == ta and first_hit[b] == tb and sim.EDGE_S <= ta and tb <= c.dur - sim.EDGE_S:
                pairs.append((ta, a, tb, b))
    for ta, a, tb, b in rng.sample(pairs, min(2, len(pairs))):
        dt = tb - ta
        if safe1(dt):
            out.append(Q("time_between_mirrors", {"a": list(a), "b": list(b)}, f"How long does the light take to get from "
                         f"the mirror in {cell_name(a)} to the mirror in {cell_name(b)}?", fmt_s(dt),
                         secs_opts(rng, dt, [dt + c.step, dt - c.step, dt * 2, dt / 2, dt + 0.5, dt - 0.5, dt + 1],
                                   max(0.3, 0.15 * dt)), True, 4))
    for em in c.s["emitters"][1:]:
        t0 = em["t0"]
        if event_ok(t0) and sim.EDGE_S <= t0 <= c.dur - sim.EDGE_S:
            tenths = clock_tenths(math.ceil(t0 * FPS - 1e-9))
            out.append(Q("launch_time", {"colour": em["colour"]}, f"When does the {em['colour']} beam start? Give the "
                         "clock reading of the first frame it shows.", fmt_clock(tenths), clock_opts(rng, tenths), True, 4))
    v = 1 / c.step
    if safe1(v):
        out.append(Q("speed", {}, "How fast does the light advance, in cells per second?", fmt_speed(v),
                     secs_opts(rng, v, [v * 2, v / 2, v * 1.5, v / 1.5, v + 1, v - 1, v + 2], max(0.3, 0.15 * v),
                               fmt_speed), True, 2))
    lit = lit_cells(c)
    if lit is not None:
        k = len(lit)
        out.append(Q("n_lit_cells", {}, "At the end of the clip, how many cells does the lit trail pass through "
                     "(count each cell once)?", k, int_opts(rng, k, 1, 99), False, 1))
    return out


def lit_cells(c: Ctx):
    """Cells the trails cover in the last frame, or None if a head sits too close to a cell border."""
    out = set()
    for p in c.res["pulses"]:
        if p["pts"][0][0] > c.t_last:
            continue
        pts = [(x, y) for t, x, y in p["pts"] if t <= c.t_last]
        h = sim.head_at(p, c.t_last)
        if h is not None:
            fx, fy = h[0] % 1, h[1] % 1
            if min(fx, 1 - fx, fy, 1 - fy) < 0.15:     # a head right at a cell border: ambiguous count
                return None
            pts.append(h)
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            n = max(1, int(math.dist((xa, ya), (xb, yb)) * 20))
            for k in range(n + 1):
                x, y = xa + (xb - xa) * k / n, ya + (yb - ya) * k / n
                if abs(x - round(x)) < 0.03 or abs(y - round(y)) < 0.03:
                    continue
                cx, cy = math.floor(x), math.floor(y)
                if 0 <= cx < c.s["cols"] and 0 <= cy < c.s["rows"]:
                    out.add((cx, cy))
    return out


# ------------------------------------------------------------------ L4
def level4(c: Ctx, rng):
    out = []
    for col in c.colours:
        if not c.single(col):
            continue
        e = c.end(col)
        gt = end_text(e)
        out.append(Q("exit_point", {"colour": col}, f"The clip stops before the {col} beam is done. Following the rules, "
                     f"where does the {col} beam end up? " + RULES, gt,
                     mc4(rng, gt, end_distractors(c, col, _end_tuple(e))), False, 1))
        t_end = e[-1]
        if t_end > c.dur:
            frac = t_end * 10 - math.floor(t_end * 10)
            if 0.15 <= frac <= 0.85:
                tenths = math.floor(t_end * 10)
                word = "leave the board" if e[0] == "exit" else "be stopped"
                out.append(Q("exit_time", {"colour": col}, f"If the clip went on (and the clock kept running), at what "
                             f"clock reading would the {col} beam {word}? " + RULES, fmt_clock(tenths),
                             clock_opts(rng, tenths, [10 * (t_end + c.step), 10 * (t_end - c.step)]), True, 4))
            dt = t_end - sim.duration(c.s)
            if safe1(dt) and dt >= 0.3:
                out.append(Q("time_after_end", {"colour": col}, f"How long after the clip ends would the {col} beam "
                             f"{'leave the board' if e[0] == 'exit' else 'be stopped'}? " + RULES, fmt_s(dt),
                             secs_opts(rng, dt, [dt + c.step, dt - c.step, dt * 2, dt / 2, dt + 1, dt + 0.5, dt - 0.5],
                                       max(0.3, 0.15 * dt)), True, 4))
        k = len(c.turns(col))
        out.append(Q("n_turns_total", {"colour": col}, f"Counting the turns in the clip and after it, how many times in "
                     f"total does the {col} beam turn at a mirror before it is done? " + RULES, k,
                     int_opts(rng, k, 1, 40), False, 1))
    if c.s["scenario"] == "splitter":
        k = sum(1 for p in c.res["pulses"] if p["end"][0] == "exit")
        out.append(Q("n_exits", {}, "Once all the light is done, at how many places does light leave the board? " + RULES,
                     k, int_opts(rng, k, 0, 9), False, 1))
    return out


# ------------------------------------------------------------------ L5
def _cf_world(c: Ctx, change, p):
    sc = json.loads(json.dumps(c.s))
    if change == "remove":
        sc["mirrors"] = [m for m in sc["mirrors"] if m["cell"] != list(p["cell"])]
    elif change == "rotate":
        for m in sc["mirrors"]:
            if m["cell"] == list(p["cell"]):
                m["kind"] = sim.flipped(m["kind"])
    elif change == "add":
        sc["mirrors"].append({"cell": list(p["cell"]), "kind": p["kind"], "type": "two"})
    elif change == "faster":
        sc["step"] = c.step / 2
    elif change == "no_flip":
        for m in sc["mirrors"]:
            if m["type"] == "flip":
                m["type"] = "two"
                m.pop("flip_t", None)
    elif change == "two_sided":
        for m in sc["mirrors"]:
            if m["type"] == "one":
                m["type"] = "two"
                m.pop("face", None)
    elif change == "no_absorbers":
        sc["absorbers"] = []
    elif change == "delay":
        for em in sc["emitters"]:
            if em["colour"] == p["colour"]:
                em["t0"] = em["t0"] + p["d"]
    return sc


def _cf_ok(sc, res):
    if res["loop"] or any(q["end"][0] == "open" for q in res["pulses"]):
        return False
    for m in sc["mirrors"]:                       # a pivot's turn must not be within two steps of a hit
        if m["type"] == "flip":
            for q in res["pulses"]:
                for t, _, cell in q["events"]:
                    if tuple(cell) == tuple(m["cell"]) and abs(t - m["flip_t"]) < 2 * sc["step"]:
                        return False
    return True


CF_TEXT = {"remove": "the mirror in cell {cell} were taken away", "rotate": "the mirror in cell {cell} leaned the other way",
           "add": "an extra two-sided mirror leaning like '{kind}' stood in the empty cell {cell}",
           "no_flip": "the pivoting mirror(s) never turned", "two_sided": "every mirror were silvered on both sides",
           "no_absorbers": "the dark blocks were removed", "faster": "the light moved twice as fast",
           "delay": "the {colour} beam were launched {d:g} s later"}


def level5(c: Ctx, rng):
    out = []
    changes = []
    for cell in c.mcells:
        changes += [("remove", {"cell": list(cell)}), ("rotate", {"cell": list(cell)})]
    path_cells = [(math.floor(x), math.floor(y)) for p in c.res["pulses"] for t, x, y in p["pts"]]
    empty = [x for x in dict.fromkeys(path_cells) if 0 <= x[0] < c.s["cols"] and 0 <= x[1] < c.s["rows"]
             and x not in c.mcells and list(x) not in c.s["absorbers"]]
    first_cells = {tuple(sim.edge_entry(c.s, em)[0]) for em in c.s["emitters"]}
    for cell in empty:
        if cell not in first_cells:
            changes.append(("add", {"cell": list(cell), "kind": rng.choice(("/", "\\"))}))
    rng.shuffle(changes)
    changes = changes[:10]
    scen = c.s["scenario"]
    if scen == "flip":
        changes.insert(0, ("no_flip", {}))
    if scen == "one_sided":
        changes.insert(0, ("two_sided", {}))
    if c.s["absorbers"]:
        changes.insert(rng.randint(0, len(changes)), ("no_absorbers", {}))
    made = 0
    base_ends = {col: _end_tuple(c.end(col)) for col in c.colours if c.single(col)}
    for ch, p in changes:
        if made == 4:
            break
        sc = _cf_world(c, ch, p)
        res = sim.trace(sim.world_of(sc))
        if not _cf_ok(sc, res):
            continue
        txt = CF_TEXT[ch].format(cell=cell_name(p["cell"]) if "cell" in p else "", kind=p.get("kind", ""))
        if scen == "splitter":
            k = sum(1 for q in res["pulses"] if q["end"][0] == "exit")
            out.append(Q("cf_n_exits", dict(p, change=ch), f"Suppose {txt}. Once all the light is done, at how many places "
                         "would light leave the board? " + RULES, k, int_opts(rng, k, 0, 9), False, 1))
            made += 1
            continue
        for col in c.colours:
            qs = [q for q in res["pulses"] if q["colour"] == col]
            if len(qs) != 1:
                continue
            e = _end_tuple(qs[0]["end"])
            gt = end_text(e)
            cands = [end_text(base_ends[col])] if col in base_ends else []
            cands += end_distractors(c, col, e)
            out.append(Q("cf_exit_point", dict(p, change=ch, colour=col), f"Suppose {txt}, with everything else the same. "
                         f"Where would the {col} beam end up? " + RULES, gt, mc4(rng, gt, cands), ch == "no_flip", 1))
            made += 1
            break
    for ch, p in [("faster", {})] + [("delay", {"colour": col, "d": d}) for col in c.colours for d in (0.5, 1.0, 1.5)]:
        if ch == "delay" and scen != "two_beams" and scen != "flip":
            continue
        sc = _cf_world(c, ch, p)
        res = sim.trace(sim.world_of(sc))
        if not _cf_ok(sc, res):
            continue
        for col in ([p["colour"]] if "colour" in p else c.colours):
            qs = [q for q in res["pulses"] if q["colour"] == col]
            if len(qs) != 1:
                continue
            te = qs[0]["end"][-1]
            frac = te * 10 - math.floor(te * 10)
            if not 0.15 <= frac <= 0.85:
                continue
            tenths = math.floor(te * 10)
            word = "leave the board" if qs[0]["end"][0] == "exit" else "be stopped"
            base = c.pulses(col)[0]["end"][-1] * 10 if c.single(col) else tenths + 10
            txt = CF_TEXT[ch].format(colour=p.get("colour", ""), d=p.get("d", 0))
            out.append(Q("cf_exit_time", dict(p, change=ch, colour=col), f"Suppose {txt}, with everything else the same. "
                         f"At what clock reading would the {col} beam {word}? " + RULES, fmt_clock(tenths),
                         clock_opts(rng, tenths, [base, tenths + 10 * c.step]), True, 4))
            break
    return out


# ------------------------------------------------------------------ assembly
def _pick(rng, cands, n):
    """Distinct keys first; templates that need the motion are preferred (70 %)."""
    by_key = {}
    for cand in cands:
        by_key.setdefault(cand[0], []).append(cand)
    keys = list(by_key)
    rng.shuffle(keys)
    if rng.random() < 0.7:
        keys.sort(key=lambda k: not any(q[6] for q in by_key[k]))
    chosen, used = [], set()
    while len(chosen) < n:
        avail = [k for k in keys if by_key[k]]
        if not avail:
            break
        k = ([k for k in avail if k not in used] or avail)[0]
        chosen.append(by_key[k].pop(rng.randrange(len(by_key[k]))))
        used.add(k)
        keys.remove(k)
        keys.append(k)
    return chosen


def build(scene: dict, rng: random.Random) -> list[dict]:
    c = Ctx(scene)
    makers = {1: level1, 2: level2, 3: level3, 4: level4, 5: level5}
    for attempt in range(20):
        made = {}
        for lv in range(1, 6):
            seen, ok = set(), []
            for q in makers[lv](c, random.Random(rng.random())):
                if q is None:
                    continue
                sig = (q[0], json.dumps(q[1], sort_keys=True))
                if sig not in seen:
                    seen.add(sig)
                    ok.append(q)
            made[lv] = ok
        feasible = [cmb for cmb in TEN_SPLITS if all(len(made[lv]) >= cmb[lv - 1] for lv in range(1, 6))]
        if feasible:
            break
    else:
        raise ValueError("not enough question candidates for this scene")
    counts = rng.choice(feasible)
    qs = []
    for lv in range(1, 6):
        for key, params, text, gt, opts, letter, motion in _pick(rng, made[lv], counts[lv - 1]):
            qs.append(L.question(lv, key, text, gt, opts, letter, motion, params, idx=len(qs) + 1))
    return qs
