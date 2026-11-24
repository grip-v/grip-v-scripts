"""10-question ladder for hidden_mirrors (common/ladder5.py, mode "ten").

Pulses are named by colour, cells by the on-screen labels (column letter +
row number, "C4"), ports by the label of their column or row ("top of
column C", "left end of row 4"). Times are what the clock shows. Answers
come from sim.py's tracer; answers.py re-derives them from the drawn frames
and re-simulates what-ifs on the mirror map it infers from the tracks.

Templates per level (derivation_key):
  L1 n_pulses, entry_port, first_colour, n_straight, n_blocks, n_shown
  L2 exit_port, first_turn_dir, first_turn_cell, launch_order, exit_order
  L3 n_turns, n_cells, time_on_board, exit_clock, turn_clock, launch_gap, speed
  L4 n_turn_cells, total_turns, locate_mirror, must_mirror, certainly_empty, n_unvisited
  L5 cf_reverse, cf_new_port, cf_change (flip / remove a mirror, remove a block),
     cf_change_turns, cf_fast
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
from answers import (cell_name, port_text, end_text, fmt_s, fmt_clock, fmt_speed, clock_tenths,  # noqa: E402
                     event_clock, DIR_WORDS)

LADDER = "ten"
TEN_SPLITS = [c for c in itertools.product(*(range(lo, hi + 1) for lo, hi in L.TEN_RANGES)) if sum(c) == L.TEN_TOTAL]
FPS = sim.FPS
GAP = 0.3                       # order questions: consecutive events at least this far apart (s)


def fps_for(gaps, floor=1):
    g = min(gaps) if gaps else 99.0
    return int(min(24, max(floor, math.ceil(1.0 / g - 1e-9))))


def safe1(v, m=0.25):
    """v sits at least m tenths (default 0.025 s) away from a 0.1 rounding boundary."""
    x = v * 10 - math.floor(v * 10)
    return abs(x - 0.5) >= m


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


def int_opts(rng, gt, lo, hi=99):
    near = [v for v in (gt + 1, gt - 1, gt + 2, gt - 2, gt + 3, gt - 3, gt + 4, gt + 5) if lo <= v <= hi]
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
    """An event read off the clock: 0.3-0.7 of the way through a frame interval, and both frames around it
    show the same tenth."""
    x = te * FPS
    fa = math.ceil(x - 1e-9)
    return 0.3 <= x - math.floor(x) <= 0.7 and clock_tenths(fa - 1) == clock_tenths(fa)


def Q(key, params, text, gt, opts, motion, fps):
    if opts is None:
        return None
    return (key, dict(params, min_fps=fps), text, gt, opts[0], opts[1], motion)


def _end(run):
    e = run["end"]
    return ("exit", (e[1], e[2])) if e[0] == "exit" else ("stopped", tuple(e[1]))


# ------------------------------------------------------------------ context
class Ctx:
    def __init__(self, scene):
        self.s = scene
        self.rs = sim.runs(scene)
        self.by = {r["colour"]: r for r in self.rs}
        self.dur = sim.duration(scene)
        self.W, self.H = scene["cols"], scene["rows"]
        self.mir = {tuple(m["cell"]): m["kind"] for m in scene["mirrors"]}
        self.shown = {tuple(m["cell"]) for m in scene["mirrors"] if m["shown"]}
        self.blocks = {tuple(b) for b in scene["blocks"]}
        self.visited = {tuple(c) for r in self.rs for c in r["cells"]}
        self.known = self.visited | self.blocks | self.shown
        self.turn_cells = {tuple(t[1]) for r in self.rs for t in r["turns"]}
        self.hidden_turns = sorted(self.turn_cells - self.shown)
        self.empty = sorted(c for c in self.visited if c not in self.mir and c not in self.blocks)
        self.used_ports = {r["port"] for r in self.rs} | {_end(r)[1] for r in self.rs if _end(r)[0] == "exit"}
        self.stop_word = " (or which block stops it)" if scene["blocks"] else ""

    def exits(self):
        return [r for r in self.rs if r["end"][0] == "exit"]

    def hyp(self, port, mirrors=None, blocks=None):
        """A hypothetical pulse; None unless every cell it enters is one the video shows."""
        r = sim.trace_one(self.s, tuple(port), 0.0, mirrors=mirrors if mirrors is not None else self.mir,
                          blocks=blocks if blocks is not None else self.blocks)
        if any(tuple(c) not in self.known for c in r["cells"]):
            return None
        r["port"] = tuple(port)
        return r

    def end_distractors(self, run, gt_end, rng):
        """Plausible wrong ends: what a flipped / missing mirror on the path would give, straight on from
        the entry, neighbouring ports on the same edge, back out of the entry port, the blocks."""
        out = []
        port = run["port"]
        straight = sim.trace_one(self.s, port, 0.0, mirrors={}, blocks=self.blocks)
        out.append(_end(straight))
        for t in run["turns"]:
            cell = tuple(t[1])
            for kind in (sim.flipped(self.mir[cell]), None):
                m2 = dict(self.mir)
                if kind is None:
                    m2.pop(cell)
                else:
                    m2[cell] = kind
                out.append(_end(sim.trace_one(self.s, port, 0.0, mirrors=m2, blocks=self.blocks)))
        rng.shuffle(out)
        near = []
        if gt_end[0] == "exit":
            edge, i = gt_end[1]
            n = self.H if edge in ("left", "right") else self.W
            near = [("exit", (edge, i + k)) for k in (1, -1, 2, -2) if 0 <= i + k < n]
        near.append(("exit", tuple(port)))
        near += [("stopped", b) for b in sorted(self.blocks)]
        rng.shuffle(near)
        return [end_text(e) for e in out[:3] + near + out[3:]]


def _port_label(port):
    return f"the {port_text(port)}"


# ------------------------------------------------------------------ L1
def level1(c: Ctx, rng):
    out = []
    k = len(c.rs)
    out.append(Q("n_pulses", {}, "How many light pulses are fired into the board during the clip?", str(k),
                 int_opts(rng, k, 1), True, 1))
    for r in c.rs:
        gt = port_text(r["port"])
        e, i = r["port"]
        n = c.H if e in ("left", "right") else c.W
        cands = [(e, i + d) for d in (1, -1, 2, -2) if 0 <= i + d < n]
        opp = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[e]
        cands.append((opp, i))
        if _end(r)[0] == "exit" and _end(r)[1] != r["port"]:
            cands.insert(0, _end(r)[1])
        out.append(Q("entry_port", {"colour": r["colour"]}, f"Where is the {r['colour']} pulse fired into the board?",
                     gt, mc4(rng, gt, [port_text(x) for x in cands]), False, 1))
    t0s = sorted(r["t0"] for r in c.rs)
    if len(c.rs) >= 4 and t0s[1] - t0s[0] >= GAP:
        first = min(c.rs, key=lambda r: r["t0"])["colour"]
        others = [r["colour"] for r in sorted(c.rs, key=lambda r: r["t0"])[1:]]
        out.append(Q("first_colour", {}, "Which pulse is fired first?", first, mc4(rng, first, others), True,
                     fps_for([t0s[1] - t0s[0]])))
    k = sum(1 for r in c.rs if not r["turns"])
    out.append(Q("n_straight", {}, "How many of the pulses cross the board without ever changing direction?", str(k),
                 int_opts(rng, k, 0, len(c.rs)), True, 1))
    if c.s["blocks"]:
        k = len(c.s["blocks"])
        out.append(Q("n_blocks", {}, "How many dark blocks are on the board?", str(k), int_opts(rng, k, 0), False, 1))
    if c.shown:
        k = len(c.shown)
        out.append(Q("n_shown", {}, "How many mirrors are drawn on the board?", str(k), int_opts(rng, k, 0), False, 1))
    return out


# ------------------------------------------------------------------ L2
def _order_q(c: Ctx, rng, key, runs, tkey, text):
    out = []
    for trio in itertools.combinations(runs, 3):
        ts = sorted(tkey(r) for r in trio)
        if min(b - a for a, b in zip(ts, ts[1:])) < GAP:
            continue
        cols = [r["colour"] for r in sorted(trio, key=tkey)]
        perms = [list(p) for p in itertools.permutations(cols) if list(p) != cols]
        rng.shuffle(perms)
        # the most plausible wrong orders swap one neighbouring pair
        perms.sort(key=lambda p: sum(a != b for a, b in zip(p, cols)))
        gt = ", ".join(cols)
        names = sorted(cols)
        out.append(Q(key, {"colours": names},
                     text.format(", ".join(names[:-1]) + " and " + names[-1]), gt,
                     mc4(rng, gt, [", ".join(p) for p in perms]), True,
                     fps_for([b - a for a, b in zip(ts, ts[1:])])))
    rng.shuffle(out)
    return out[:3]


def _cell_opts(c: Ctx, rng, gt, extra=()):
    cand = [x for x in list(extra) if x != gt]
    near = [(gt[0] + dx, gt[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1))]
    near = [x for x in near if 0 <= x[0] < c.W and 0 <= x[1] < c.H]
    rng.shuffle(near)
    cand = list(dict.fromkeys(cand + near))
    return mc4(rng, gt, cand, cell_name)


def level2(c: Ctx, rng):
    out = []
    for r in c.rs:
        col, gt_end = r["colour"], _end(r)
        gt = end_text(gt_end)
        out.append(Q("exit_port", {"colour": col}, f"Where does the {col} pulse leave the board{c.stop_word}?", gt,
                     mc4(rng, gt, c.end_distractors(r, gt_end, rng)), True, 1))
        if r["turns"]:
            t, cell, din, dout = r["turns"][0]
            w = DIR_WORDS[tuple(dout)]
            others = [DIR_WORDS[tuple(din)]] + [x for x in ("up", "down", "left", "right") if x not in (w, DIR_WORDS[tuple(din)])]
            out.append(Q("first_turn_dir", {"colour": col},
                         f"After its first change of direction, which way does the {col} pulse travel?", w,
                         mc4(rng, w, others), True, 1))
            cells = [tuple(x) for x in r["cells"]]
            i = cells.index(tuple(cell))
            extra = [cells[j] for j in (i - 1, i + 1) if 0 <= j < len(cells)]
            extra += [tuple(x[1]) for x in r["turns"][1:3]]
            out.append(Q("first_turn_cell", {"colour": col},
                         f"In which cell does the {col} pulse change direction for the first time?", cell_name(cell),
                         _cell_opts(c, rng, tuple(cell), extra), True, 1))
    out += _order_q(c, rng, "launch_order", c.rs, lambda r: r["t0"], "In what order are the {} pulses fired?")
    out += _order_q(c, rng, "exit_order", c.exits(), sim.end_time, "In what order do the {} pulses leave the board?")
    return out


# ------------------------------------------------------------------ L3
def level3(c: Ctx, rng):
    out = []
    step = c.s["step"]
    for r in c.rs:
        col = r["colour"]
        k = len(r["turns"])
        out.append(Q("n_turns", {"colour": col}, f"How many times does the {col} pulse change direction?", str(k),
                     int_opts(rng, k, 0), False, 1))
        if r["turns"] and sim.EDGE_S <= r["turns"][0][0] <= c.dur - sim.EDGE_S and event_ok(r["turns"][0][0]):
            te = event_clock(r["turns"][0][0])
            out.append(Q("turn_clock", {"colour": col},
                         f"What does the clock show when the {col} pulse changes direction for the first time?",
                         fmt_clock(te), clock_opts(rng, te, [round(r["t0"] * 10)]), True, 4))
        if r["end"][0] != "exit":
            continue
        n = len(r["cells"])
        out.append(Q("n_cells", {"colour": col},
                     f"How many cells does the {col} pulse pass through, counting the cells where it turns?", str(n),
                     int_opts(rng, n, 1), False, 1))
        D = n * step
        if safe1(D):
            sp = max(0.3, 0.15 * D)
            out.append(Q("time_on_board", {"colour": col},
                         f"For how long is the {col} pulse on the board, from entering it to leaving it?", fmt_s(D),
                         secs_opts(rng, D, [D + k * sp for k in (-2, -1, 1, 2, 3)] + [(n + 3) * step, (n - 3) * step],
                                   sp), True, 4))
        te_ = sim.end_time(r)
        if event_ok(te_):
            te = event_clock(te_)
            out.append(Q("exit_clock", {"colour": col}, f"What does the clock show when the {col} pulse leaves the board?",
                         fmt_clock(te), clock_opts(rng, te, [round(r["t0"] * 10)]), True, 4))
    srt = sorted(c.rs, key=lambda r: r["t0"])
    for a, b in zip(srt, srt[1:]):
        g = b["t0"] - a["t0"]
        if g >= 0.4 and safe1(g):
            sp = max(0.3, 0.15 * g)
            out.append(Q("launch_gap", {"earlier": a["colour"], "later": b["colour"]},
                         f"How much time passes between the {a['colour']} pulse being fired and the {b['colour']} "
                         f"pulse being fired?", fmt_s(g),
                         secs_opts(rng, g, [g + k * sp for k in (-2, -1, 1, 2, 3)], sp), True, 4))
    v = 1 / step
    x = 2 * v - math.floor(2 * v)
    if abs(x - 0.5) >= 0.2:                          # >= 0.1 cells/s from a half-cell rounding boundary
        g = round(2 * v) / 2
        cand = [g + d for d in (0.5, -0.5, 1.0, -1.0, 1.5, -1.5) if g + d >= 1.0]
        rng.shuffle(cand)                            # so the answer is not always the second-smallest option
        out.append(Q("speed", {}, "About how fast do the pulses travel, to the nearest half cell per second?",
                     fmt_speed(v), mc4(rng, g, cand, fmt_speed), True, 4))
    return out


# ------------------------------------------------------------------ L4
def level4(c: Ctx, rng):
    out = []
    k = len(c.turn_cells)
    out.append(Q("n_turn_cells", {}, "Counting every pulse, in how many different cells does some pulse change "
                 "direction?", str(k), int_opts(rng, k, 1), True, 1))
    k = sum(len(r["turns"]) for r in c.rs)
    out.append(Q("total_turns", {}, "How many changes of direction do all the pulses make altogether?", str(k),
                 int_opts(rng, k, 1), True, 1))
    ordn = {1: "first", 2: "second", 3: "third", 4: "fourth"}
    for r in c.rs:
        cells = [tuple(x) for x in r["cells"]]
        for j, (t, cell, din, dout) in enumerate(r["turns"][:4]):
            cell = tuple(cell)
            if cell in c.shown:
                continue
            kind = c.mir[cell]
            other = sim.flipped(kind)
            i = cells.index(cell)
            prev = cells[i - 1] if i > 0 else None
            nxt = cells[i + 1] if i + 1 < len(cells) else None
            cand = [(cell, other)] + [(x, y) for x in (prev, nxt) if x is not None for y in (kind, other)]
            rng.shuffle(cand)
            cand.sort(key=lambda x: x[0] != cell)
            fmt = lambda x: f"{cell_name(x[0])}, {x[1]}"
            out.append(Q("locate_mirror", {"colour": r["colour"], "k": j + 1},
                         f"The {r['colour']} pulse changes direction for the {ordn[j + 1]} time at a hidden mirror. "
                         f"In which cell is that mirror, and which way does it lean ('/' rises to the right, "
                         f"'\\' falls to the right)?", fmt((cell, kind)), mc4(rng, (cell, kind), cand, fmt), True, 1))
    if c.hidden_turns and len(c.empty) >= 3:
        for _ in range(2):
            gt = rng.choice(c.hidden_turns)
            e = sorted(c.empty, key=lambda x: (abs(x[0] - gt[0]) + abs(x[1] - gt[1]), rng.random()))[:5]
            rng.shuffle(e)
            cells = sorted([gt] + e[:3])
            out.append(Q("must_mirror", {"cells": [list(x) for x in cells]},
                         f"Which of these cells must hold a hidden mirror: {', '.join(cell_name(x) for x in cells)}?",
                         cell_name(gt), mc4(rng, gt, e[:3], cell_name), True, 1))
    if len(c.hidden_turns) >= 3 and c.empty:
        for _ in range(2):
            gt = rng.choice(c.empty)
            m = list(c.hidden_turns)
            rng.shuffle(m)
            cells = sorted([gt] + m[:3])
            out.append(Q("certainly_empty", {"cells": [list(x) for x in cells]},
                         f"Which of these cells certainly has no mirror in it: "
                         f"{', '.join(cell_name(x) for x in cells)}?", cell_name(gt), mc4(rng, gt, m[:3], cell_name),
                         True, 1))
    open_cells = c.W * c.H - len(c.blocks)
    k = open_cells - len(c.visited - c.blocks)
    where = " (not counting the dark blocks)" if c.blocks else ""
    out.append(Q("n_unvisited", {}, f"Over the whole clip, how many cells{where} does no pulse ever enter?", str(k),
                 int_opts(rng, k, 0), True, 1))
    return out


# ------------------------------------------------------------------ L5
def level5(c: Ctx, rng):
    out = []
    for r in c.exits():
        h = c.hyp(_end(r)[1])
        if h is None:
            continue
        gt = end_text(_end(h))
        out.append(Q("cf_reverse", {"colour": r["colour"]},
                     f"If a new pulse were fired into the board exactly where the {r['colour']} pulse leaves it, "
                     f"where would the new pulse leave the board{c.stop_word}?", gt,
                     mc4(rng, gt, [end_text(_end(r))] + c.end_distractors(h, _end(h), rng)), True, 1))
    free = [p for p in sim.all_ports(c.s) if p not in c.used_ports]
    rng.shuffle(free)
    made = 0
    for port in free:
        h = c.hyp(port)
        if h is None or made == 2:
            continue
        made += 1
        gt = end_text(_end(h))
        out.append(Q("cf_new_port", {"port": list(port)},
                     f"If one more pulse were fired in at {_port_label(port)}, where would it leave the board"
                     f"{c.stop_word}?", gt, mc4(rng, gt, c.end_distractors(h, _end(h), rng)), True, 1))
    changes = []
    for r in c.rs:
        for cell in sorted({tuple(t[1]) for t in r["turns"]}):
            changes += [(r, cell, "flip"), (r, cell, "remove")]
        if r["end"][0] == "stopped":
            changes.append((r, tuple(r["end"][1]), "unblock"))
    rng.shuffle(changes)
    texts = {"flip": "If the mirror in {} leaned the other way", "remove": "If there were no mirror in {}",
             "unblock": "If the dark block in {} were taken away"}
    n_ch = n_tu = 0
    for r, cell, ch in changes:
        m2, b2 = dict(c.mir), set(c.blocks)
        if ch == "flip":
            m2[cell] = sim.flipped(m2[cell])
        elif ch == "remove":
            m2.pop(cell)
        else:
            b2.discard(cell)
        h = c.hyp(r["port"], m2, b2)
        if h is None:
            continue
        col = r["colour"]
        params = {"colour": col, "cell": list(cell), "change": ch}
        lead = texts[ch].format(cell_name(cell))
        gt_end = _end(h)
        if gt_end != _end(r) and n_ch < 3:
            n_ch += 1
            gt = end_text(gt_end)
            out.append(Q("cf_change", params, f"{lead}, where would the {col} pulse leave the board{c.stop_word}?", gt,
                         mc4(rng, gt, [end_text(_end(r))] + c.end_distractors(h, gt_end, rng)), True, 1))
        k = len(h["turns"])
        if k != len(r["turns"]) and n_tu < 2:
            n_tu += 1
            out.append(Q("cf_change_turns", params, f"{lead}, how many times would the {col} pulse change direction?",
                         str(k), int_opts(rng, k, 0), True, 1))
    for r in c.exits():
        v = r["t0"] + len(r["cells"]) * c.s["step"] / 2
        if safe1(v):
            te = sim.end_time(r)
            out.append(Q("cf_fast", {"colour": r["colour"]},
                         f"If every pulse moved twice as fast (each fired at the same moment as in the clip), at what "
                         f"time on the clock would the {r['colour']} pulse leave the board?", fmt_s(v),
                         secs_opts(rng, v, [te, v - 0.5, v + 0.5, v - 1.0, v + 1.0, r["t0"] + (te - r["t0"]) / 4], 0.4),
                         True, 4))
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
    if not feasible:
        raise ValueError("not enough question candidates for this scene")
    counts = rng.choice(feasible)
    qs = []
    for lv in range(1, 6):
        for key, params, text, gt, opts, letter, motion in _pick(rng, made[lv], counts[lv - 1]):
            qs.append(L.question(lv, key, text, gt, opts, letter, motion, params, idx=len(qs) + 1))
    return qs
