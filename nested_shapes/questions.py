"""10-question, 5-rung ladder for nested_shapes (common/ladder5.py, LADDER = "ten").

Nothing in the frame is labelled except the timer. Outlines are named by
border colour + shape ("the teal hexagon"; every colour + shape is unique in a
clip), or by shape alone in the recolor scenario (shapes are unique there,
colours change), or by their place in the nest ("the 2nd outline from the
outside"). Moments are named by timer readings. Every question has 4 options
and stores params["min_fps"] = 24 / k for the largest frame step k at which
sampling every k-th frame still shows every relied-on event in the right
order, keeps every relied-on turn unambiguous (a turn per sample under 0.4 of
the outline's symmetry period) and every relied-on turning stretch at least
three samples long.

Templates per level (derivation keys):
  L1 n_end, color_count_end, shape_at_depth, depth_color, first_appear, never_turns, first_mark,
     start_color, end_color, first_wall, end_heading
  L2 inside_of, appear_earliest, vanishes, same_dir, fastest, first_change, lost_mark, more_added,
     color_at_time, which_recolors
  L3 appear_time, change_time, mark_time, recolor_time, hit_time, count_at, turning_at, marks_window,
     n_changes, n_recolors, n_hits
  L4 appear_pattern, max_on_screen, spin_pattern, cw_end, most_changes, total_added, total_removed,
     most_added, color_sequence, total_recolors, hits_axis
  L5 cf_stop_count, cf_reverse_order, cf_remove, cf_keep_dir, cf_flip, cf_stop_turning, cf_no_removal,
     cf_stop_marks, cf_no_last_recolor, cf_stop_color, cf_no_wall, cf_stop_hits
Counterfactuals are re-simulated on an edited copy of the scene.
"""
from __future__ import annotations
import copy
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
FPS = sim.FPS
KMAX = 24
DIRW = {1: "clockwise", -1: "counterclockwise", 0: "not turning"}
HEADINGS = ("up and to the left", "up and to the right", "down and to the left", "down and to the right")
ORDERS = {"outside-in": "from the outside in", "inside-out": "from the inside out",
          "mixed": "in a mixed order, neither outside-in nor inside-out",
          "together": "all at the same moment"}


def fps_of(k) -> float:
    k = max(1, min(KMAX, int(k)))
    return math.ceil(FPS / k * 10) / 10


def ordinal(d: int, n: int) -> str:
    if d == 0:
        return "outermost outline"
    if d == n - 1:
        return "innermost outline"
    suf = {1: "2nd", 2: "3rd"}.get(d, f"{d + 1}th")
    return f"{suf} outline from the outside"


def pattern_text(dirs) -> str:
    first = {1: "turns clockwise", -1: "turns counterclockwise", 0: "stays still"}
    later = {1: "turns clockwise", -1: "turns counterclockwise", 0: "stops"}
    if len(dirs) == 1:
        return first[dirs[0]] + " the whole time"
    return ", then ".join([first[dirs[0]]] + [later[d] for d in dirs[1:]])


def pattern_alts(dirs) -> list[list[int]]:
    """Plausible wrong turning patterns: one stretch changed, one dropped / added, reversed."""
    out = []
    for i in range(len(dirs)):
        for v in (1, -1, 0):
            alt = list(dirs)
            alt[i] = v
            if alt != list(dirs) and all(a != b for a, b in zip(alt, alt[1:])):
                out.append(alt)
    if len(dirs) > 1:
        out += [list(dirs[:-1]), list(dirs[1:]), list(reversed(dirs))]
    for v in (1, -1, 0):
        if v != dirs[-1]:
            out.append(list(dirs) + [v])
    seen, uniq = set(), []
    for p in out:
        if tuple(p) not in seen and p != list(dirs) and not (len(p) == 1 and False):
            seen.add(tuple(p))
            uniq.append(p)
    return uniq


def pair_text(a: int, b: int, la: str, lb: str) -> str:
    if a and b:
        return "both turn the same way" if a == b else "they turn opposite ways"
    if a or b:
        return f"only the {la if a else lb} turns"
    return "neither turns"


def seq_text(cols) -> str:
    return ", then ".join(cols)


def margin_ok(a: int, b: int) -> bool:
    return a - b >= max(1, math.ceil(0.1 * a))


def unique_top(vals: dict):
    order = sorted(vals, key=lambda k: (-vals[k], k))
    if len(order) > 1 and margin_ok(vals[order[0]], vals[order[1]]):
        return order[0], order
    return None, order


class Ctx:
    def __init__(self, scene):
        s = self.s = scene
        self.nf = nf = scene["n_frames"]
        self.n = n = len(scene["rings"])
        self.sc = scene["scenario"]
        self.lab = [sim.label(scene, k) for k in range(n)]
        self.all = list(range(n))
        self.endv = sim.on_screen(scene, nf - 1)
        self.runs = [sim.runs(scene, k) for k in self.all]
        self.chg = [sim.changes(scene, k) for k in self.all]
        self.ev = sim.events(scene)
        self.evf = [f for f, _ in self.ev]
        self.hits = sim.bounce_frames(scene)
        self.turns = [any(d for _, _, d in self.runs[k]) for k in self.all]
        self.ring_ev = {k: sorted([r for r in [scene["rings"][k]["appear"]] if r > 0] +
                                  ([scene["rings"][k]["vanish"]] if scene["rings"][k]["vanish"] is not None else []) +
                                  [f for f, _ in scene["rings"][k]["recolor"]] + self.chg[k]) for k in self.all}

    # ---- frame-step limits -----------------------------------------------------------------
    def k_events(self, frames) -> int:
        """Largest step keeping these events in separate samples and seen before the end."""
        fr = sorted(frames)
        if not fr:
            return KMAX
        ks = [b - a for a, b in zip(fr, fr[1:])] + [self.nf - fr[-1], fr[0]]
        return max(1, min(ks))

    def k_turn(self, rings) -> int:
        """Largest step keeping the turning of these rings unambiguous and every stretch >= 3 samples."""
        k = KMAX
        for i in rings:
            r = self.s["rings"][i]
            P = sim.period(r)
            wmax = max([abs(w) for _, w in r["spin"]] + [0])
            if P and wmax:
                k = min(k, int(sim.ALIAS * P * FPS / wmax))
            k = min(k, min(b - a + 1 for a, b, _ in self.runs[i]) // 3)
        return max(1, k)

    def k_end(self, frames) -> int:
        """End-state question: the last relied-on event happens >= k frames before the end."""
        return max(1, min([KMAX] + [self.nf - f for f in frames]))

    def k_at(self, ft: int, frames) -> int:
        """State at frame ft: no relied-on event within k frames of it."""
        return max(1, min([KMAX] + [abs(f - ft) - 1 for f in frames]))

    def readings(self, frames, lo_s=1.0):
        """Timer readings (tenths, multiples of 0.5 s) with no relied-on event within GAP_F frames."""
        out = []
        top = sim.tenths(self.nf - 1)
        for tn in range(int(lo_s * 10), top - 9, 5):
            ft = sim.frame_of_reading(tn)
            if all(abs(f - ft) >= sim.GAP_F for f in frames) and ft < self.nf - sim.EDGE_F:
                out.append(tn)
        return out

    def time_opts(self, rng, fr: int, pool_frames):
        tn = sim.tenths(fr)
        top = sim.tenths(self.nf - 1)
        pool = sorted({sim.tenths(f) for f in pool_frames})
        rng.shuffle(pool)
        pool += [tn + d for d in (10, -10, 15, -15, 20, -20, 25, -25, 30, -30, 8, -8, 13, -13, 18, -18, 35, -35, 40, -40)]
        ds = []
        for v in pool:
            if 3 <= v <= top - 3 and abs(v - tn) >= 8 and all(abs(v - w) >= 5 for w in ds) and len(ds) < 3:
                ds.append(v)
        gt = sim.fmt_reading(tn)
        o, a = L.mc(rng, gt, [sim.fmt_reading(v) for v in ds])
        kt = int((min(abs(v - tn) for v in ds) / 20 - 0.1) * FPS)
        return gt, o, a, kt


def _q(level, key, text, gt, opts, letter, motion, params):
    return L.question(level, key, text, gt, opts, letter, motion, params)


def _num(rng, gt, extra=(), lo=0):
    return L.mc(rng, gt, list(extra) + L.near_ints(gt, lo))


def _rings(rng, c: Ctx, correct, pool):
    pick = [k for k in pool if k != correct][:3]
    o, a = L.mc(rng, c.lab[correct], [c.lab[k] for k in pick])
    return o, a, [correct] + pick


def _words(rng, gt, pool):
    return L.mc(rng, gt, [w for w in pool if w != gt])


# ---- L1 observation ---------------------------------------------------------------------
def level1(c: Ctx):
    s, out = c.s, []
    ends = c.endv
    vis_ev = [f for k in c.all for f in c.ring_ev[k] if f in ([s["rings"][k]["appear"]] + [s["rings"][k]["vanish"]])]

    def mk(rng):
        o, a = _num(rng, len(ends), lo=1)
        return _q(1, "n_end", "How many nested outlines are on screen at the end of the video?", len(ends), o, a,
                  False, {"min_fps": fps_of(c.k_end(vis_ev))})
    out.append(("n_end", None, mk, False))
    end_cols = {k: sim.color_at(s["rings"][k], c.nf - 1) for k in ends}
    pal = sim.THEMES[s["style"]["theme"]]["colors"]
    rec_ev = [f for k in c.all for f, _ in s["rings"][k]["recolor"]]
    for col in pal:
        v = sum(1 for k in ends if end_cols[k] == col)
        if v or col in [s["rings"][k]["color"] for k in c.all]:
            def mk(rng, col=col, v=v):
                o, a = _num(rng, v, lo=0)
                return _q(1, "color_count_end", f"At the end of the video, how many outlines have a {col} border?",
                          v, o, a, False, {"c": col, "min_fps": fps_of(c.k_end(vis_ev + rec_ev))})
            out.append(("color_count_end", col, mk, False))
    shapes_here = sorted({s["rings"][k]["shape"] for k in c.all})
    for d, k in enumerate(ends):
        def mk(rng, d=d, k=k):
            gt = s["rings"][k]["shape"]
            pool = [x for x in shapes_here if x != gt] + [x for x in sim.TYPES if x not in shapes_here]
            o, a = L.mc(rng, gt, pool[:2] + rng.sample(pool[2:], 1) if len(pool) > 2 else pool)
            return _q(1, "shape_at_depth", f"At the end of the video, what shape is the {ordinal(d, len(ends))}?",
                      gt, o, a, False, {"d": d, "min_fps": fps_of(c.k_end(vis_ev))})
        out.append(("shape_at_depth", d, mk, False))
        if c.sc != "recolor":
            def mk(rng, d=d, k=k):
                gt = end_cols[k]
                others = [end_cols[j] for j in ends if end_cols[j] != gt]
                pool = list(dict.fromkeys(others + [x for x in pal if x != gt and x not in others]))
                o, a = L.mc(rng, gt, pool[:2] + rng.sample(pool[2:], 1))
                return _q(1, "depth_color", f"At the end of the video, what colour is the border of the "
                          f"{ordinal(d, len(ends))}?", gt, o, a, False, {"d": d, "min_fps": fps_of(c.k_end(vis_ev))})
            out.append(("depth_color", d, mk, False))
    if c.sc == "assemble" and c.n >= 4:
        app = sorted(c.all, key=lambda k: s["rings"][k]["appear"])

        def mk(rng, app=app):
            rest = app[1:]
            rng.shuffle(rest)
            o, a, ids = _rings(rng, c, app[0], rest)
            kk = c.k_events([s["rings"][k]["appear"] for k in app[:2]])
            return _q(1, "first_appear", "Which outline appears first?", c.lab[app[0]], o, a, True,
                      {"min_fps": fps_of(kk)})
        out.append(("first_appear", None, mk, True))
    still = [k for k in c.all if not c.turns[k]]
    moving = [k for k in c.all if c.turns[k]]
    if len(moving) >= 3:
        for x in still:
            def mk(rng, x=x):
                o, a, ids = _rings(rng, c, x, rng.sample(moving, 3))
                return _q(1, "never_turns", "Which of these outlines never turns?", c.lab[x], o, a, True,
                          {"opts": ids, "min_fps": fps_of(c.k_turn(ids))})
            out.append(("never_turns", x, mk, True))
    if s["marks"]:
        m0 = min(s["marks"], key=lambda m: m["add"])
        others = [k for k in c.all if k != m0["ring"]]
        if len(others) >= 3:
            def mk(rng, m0=m0, others=others):
                rng.shuffle(others)
                o, a, ids = _rings(rng, c, m0["ring"], others)
                adds = sorted(m["add"] for m in s["marks"])
                return _q(1, "first_mark", "Which outline gets a mark first?", c.lab[m0["ring"]], o, a, True,
                          {"opts": ids, "min_fps": fps_of(c.k_events(adds[:2]))})
            out.append(("first_mark", None, mk, True))
    if c.sc == "recolor":
        for x in c.all:
            r = s["rings"][x]
            for key, f, word in (("start_color", 0, "start"), ("end_color", c.nf - 1, "end")):
                def mk(rng, x=x, key=key, f=f, word=word):
                    gt = sim.color_at(s["rings"][x], f)
                    seen = [col for _, col in s["rings"][x]["recolor"]] + [s["rings"][x]["color"]]
                    pool = list(dict.fromkeys([col for col in seen if col != gt] +
                                              [col for col in pal if col != gt and col not in seen]))
                    o, a = L.mc(rng, gt, pool[:3])
                    rec = [ff for ff, _ in s["rings"][x]["recolor"]]
                    kk = c.k_end(rec) if word == "end" else min([KMAX] + rec)
                    return _q(1, key, f"What colour is the border of the {c.lab[x]} at the {word} of the video?", gt,
                              o, a, False, {"x": x, "min_fps": fps_of(kk)})
                out.append((key, x, mk, False))
    if c.hits:
        def mk(rng):
            gt = c.hits[0][1]
            o, a = _words(rng, gt, list(sim.WALLS))
            return _q(1, "first_wall", "Which edge of the box does the nest hit first?", gt, o, a, True,
                      {"min_fps": fps_of(c.k_events([h[0] for h in c.hits[:2]]) // 2)})
        out.append(("first_wall", None, mk, True))
        n = s["nest"]
        vx = n["vx"] * (-1) ** sum(1 for _, w in c.hits if w in ("left", "right"))
        vy = n["vy"] * (-1) ** sum(1 for _, w in c.hits if w in ("top", "bottom"))
        gt = HEADINGS[(2 if vy > 0 else 0) + (1 if vx > 0 else 0)]

        def mk(rng, gt=gt):
            o, a = _words(rng, gt, list(HEADINGS))
            return _q(1, "end_heading", "At the end of the video, which way is the nest moving?", gt, o, a, True,
                      {"min_fps": fps_of(c.k_end([h[0] for h in c.hits]) // 2)})
        out.append(("end_heading", None, mk, True))
    return out


# ---- L2 relational ----------------------------------------------------------------------
def level2(c: Ctx):
    s, out = c.s, []
    ends = c.endv
    for d, x in enumerate(ends[:-1]):
        y = ends[d + 1]

        def mk(rng, x=x, y=y):
            pool = [k for k in ends if k not in (x, y)]
            rng.shuffle(pool)
            o, a, ids = _rings(rng, c, y, pool + [k for k in c.all if k not in ends])
            vis_ev = [f for k in c.all for f in ([s["rings"][k]["appear"]] + [s["rings"][k]["vanish"]]) if f]
            return _q(2, "inside_of", f"At the end of the video, which outline is directly inside the {c.lab[x]}?",
                      c.lab[y], o, a, False, {"x": x, "min_fps": fps_of(c.k_end(vis_ev))})
        if c.n - 2 >= 3:
            out.append(("inside_of", x, mk, False))
    for d, x in enumerate(ends[:-1]):
        v = len(ends) - d - 1

        def mk(rng, x=x, v=v):
            o, a = _num(rng, v, lo=0)
            vis_ev = [f for k in c.all for f in ([s["rings"][k]["appear"]] + [s["rings"][k]["vanish"]]) if f]
            return _q(2, "inside_count", f"At the end of the video, how many outlines are inside the {c.lab[x]}?",
                      v, o, a, False, {"x": x, "min_fps": fps_of(c.k_end(vis_ev))})
        out.append(("inside_count", x, mk, False))
    if c.sc == "assemble":
        app = {k: s["rings"][k]["appear"] for k in c.all}
        first = min(c.all, key=lambda k: app[k])
        for x in c.all:
            later = [k for k in c.all if app[k] > app[x]]
            if x != first and len(later) >= 3:
                def mk(rng, x=x, later=later):
                    o, a, ids = _rings(rng, c, x, rng.sample(later, 3))
                    return _q(2, "appear_earliest", "Of these outlines, which one appears earliest?", c.lab[x], o, a,
                              True, {"opts": ids, "min_fps": fps_of(c.k_events([app[k] for k in ids]))})
                out.append(("appear_earliest", x, mk, True))
        gone = [k for k in c.all if s["rings"][k]["vanish"] is not None]
        stay = [k for k in c.all if s["rings"][k]["vanish"] is None]
        if len(stay) >= 3:
            for x in gone:
                def mk(rng, x=x):
                    o, a, ids = _rings(rng, c, x, rng.sample(stay, 3))
                    kk = c.k_events([s["rings"][x]["appear"], s["rings"][x]["vanish"]])
                    return _q(2, "vanishes", "Which of these outlines disappears before the end of the video?",
                              c.lab[x], o, a, True, {"opts": ids, "min_fps": fps_of(kk)})
                out.append(("vanishes", x, mk, True))
    # turning relations
    turn_ev = [f for k in c.all for f in c.chg[k]]
    for tn in c.readings(turn_ev + [s["rings"][k]["appear"] for k in c.all] +
                         [s["rings"][k]["vanish"] for k in c.all if s["rings"][k]["vanish"] is not None])[::2]:
        ft = sim.frame_of_reading(tn)
        vis = sim.on_screen(s, ft)
        dirs = {k: sim.dir_at(s, k, ft) for k in vis}
        for x in vis:
            if not dirs[x]:
                continue
            same = [k for k in vis if k != x and dirs[k] == dirs[x]]
            diff = [k for k in vis if dirs[k] != dirs[x]]
            if len(same) >= 1 and len(diff) >= 3:
                y = same[0]

                def mk(rng, x=x, y=y, diff=diff, tn=tn, ft=ft):
                    o, a, ids = _rings(rng, c, y, rng.sample(diff, 3))
                    kk = min(c.k_turn(ids + [x]), c.k_at(ft, turn_ev))
                    return _q(2, "same_dir", f"When the timer reads {sim.fmt_reading(tn)}, which of these outlines is "
                              f"turning the same way as the {c.lab[x]}?", c.lab[y], o, a, True,
                              {"x": x, "t": tn, "opts": ids, "min_fps": fps_of(kk)})
                out.append(("same_dir", (x, tn), mk, True))
                break
    for tn in c.readings(turn_ev + [s["rings"][k]["appear"] for k in c.all] +
                         [s["rings"][k]["vanish"] for k in c.all if s["rings"][k]["vanish"] is not None])[1::2]:
        ft = sim.frame_of_reading(tn)
        vis = sim.on_screen(s, ft)
        dirs = {k: sim.dir_at(s, k, ft) for k in vis}
        for x in vis:
            for y in vis:
                if x < y and (dirs[x] or dirs[y]):
                    gt = pair_text(dirs[x], dirs[y], c.lab[x], c.lab[y])

                    def mk(rng, x=x, y=y, tn=tn, ft=ft, gt=gt):
                        opts = [pair_text(a_, b_, c.lab[x], c.lab[y]) for a_, b_ in ((1, 1), (1, -1), (1, 0), (0, 1))]
                        o, a = L.mc(rng, gt, opts)
                        kk = min(c.k_turn([x, y]), c.k_at(ft, turn_ev))
                        return _q(2, "pair_dir", f"When the timer reads {sim.fmt_reading(tn)}, how are the {c.lab[x]} "
                                  f"and the {c.lab[y]} turning?", gt, o, a, True,
                                  {"x": x, "y": y, "t": tn, "min_fps": fps_of(kk)})
                    out.append(("pair_dir", (x, y, tn), mk, True))
    steady = {k: abs(s["rings"][k]["spin"][0][1]) for k in c.all
              if len(s["rings"][k]["spin"]) == 1 and c.turns[k]}
    for x in steady:
        slow = [k for k in steady if k != x and steady[x] >= 1.5 * steady[k]]
        if len(slow) >= 3:
            def mk(rng, x=x, slow=slow):
                o, a, ids = _rings(rng, c, x, rng.sample(slow, 3))
                return _q(2, "fastest", "Which of these outlines turns the fastest?", c.lab[x], o, a, True,
                          {"opts": ids, "min_fps": fps_of(c.k_turn(ids))})
            out.append(("fastest", x, mk, True))
    changers = [k for k in c.all if c.chg[k]]
    for x in changers:
        later = [k for k in changers if c.chg[k][0] > c.chg[x][0]]
        if len(later) >= 3:
            def mk(rng, x=x, later=later):
                o, a, ids = _rings(rng, c, x, rng.sample(later, 3))
                kk = min(c.k_turn(ids), c.k_events([c.chg[k][0] for k in ids]) // 2)
                return _q(2, "first_change", "Of these outlines, which one is the first to change how it turns "
                          "(reverse, stop or start)?", c.lab[x], o, a, True, {"opts": ids, "min_fps": fps_of(kk)})
            out.append(("first_change", x, mk, True))
    if s["marks"]:
        adds = {k: [m for m in s["marks"] if m["ring"] == k] for k in c.all}
        losers = [k for k in c.all if any(m["remove"] is not None for m in adds[k])]
        keep = [k for k in c.all if k not in losers]
        if len(keep) >= 3:
            for x in losers:
                def mk(rng, x=x, keep=keep):
                    o, a, ids = _rings(rng, c, x, rng.sample(keep, 3))
                    fr = [m["add"] for m in s["marks"]] + [m["remove"] for m in s["marks"] if m["remove"]]
                    return _q(2, "lost_mark", "Which of these outlines loses a mark during the video?", c.lab[x], o, a,
                              True, {"opts": ids, "min_fps": fps_of(c.k_events(fr))})
                out.append(("lost_mark", x, mk, True))
        for x in c.all:
            for y in c.all:
                v = len(adds[x]) - len(adds[y])
                if x != y and v > 0 and (any(m["remove"] for m in adds[x] + adds[y])):
                    def mk(rng, x=x, y=y, v=v):
                        o, a = _num(rng, v, [0])
                        fr = [m["add"] for m in adds[x] + adds[y]]
                        return _q(2, "more_added", f"Over the whole video, how many more marks are placed on the "
                                  f"{c.lab[x]} than on the {c.lab[y]}?", v, o, a, True,
                                  {"x": x, "y": y, "min_fps": fps_of(c.k_events(fr))})
                    out.append(("more_added", (x, y), mk, True))
    if s["marks"]:
        mk_ev = [m["add"] for m in s["marks"]] + [m["remove"] for m in s["marks"] if m["remove"]]
        for tn in c.readings(mk_ev)[::2]:
            ft = sim.frame_of_reading(tn)
            on = {k: sum(1 for m in s["marks"] if m["ring"] == k and sim.mark_visible(m, ft)) for k in c.all}
            endn = {k: sum(1 for m in s["marks"] if m["ring"] == k and sim.mark_visible(m, c.nf - 1)) for k in c.all}
            for x in c.all:
                for y in c.all:
                    v = on[x] - on[y]
                    if x != y and v > 0 and v != endn[x] - endn[y]:
                        def mk(rng, x=x, y=y, v=v, tn=tn, ft=ft):
                            o, a = _num(rng, v, [0, endn[x] - endn[y]] if endn[x] - endn[y] >= 0 else [0])
                            return _q(2, "more_at", f"When the timer reads {sim.fmt_reading(tn)}, how many more marks "
                                      f"does the {c.lab[x]} carry than the {c.lab[y]}?", v, o, a, True,
                                      {"x": x, "y": y, "t": tn,
                                       "min_fps": fps_of(min(c.k_at(ft, mk_ev), c.k_events(mk_ev)))})
                        out.append(("more_at", (x, y, tn), mk, True))
    if len(c.hits) >= 2:
        hf = [h[0] for h in c.hits]
        for wall in sim.WALLS:
            idx = [i for i, (_, w) in enumerate(c.hits) if w == wall]
            if idx and idx[0] > 0:
                gt = c.hits[idx[0] - 1][1]

                def mk(rng, wall=wall, gt=gt):
                    o, a = _words(rng, gt, list(sim.WALLS))
                    return _q(2, "wall_before", f"Which edge of the box does the nest hit just before it first hits the "
                              f"{wall} edge?", gt, o, a, True, {"wall": wall, "min_fps": fps_of(c.k_events(hf) // 2)})
                out.append(("wall_before", wall, mk, True))
    if c.sc == "recolor":
        pal = sim.THEMES[s["style"]["theme"]]["colors"]
        rec = [f for k in c.all for f, _ in s["rings"][k]["recolor"]]
        for x in c.all:
            r = s["rings"][x]
            if not r["recolor"]:
                continue
            endc = sim.color_at(r, c.nf - 1)
            for tn in c.readings(rec):
                ft = sim.frame_of_reading(tn)
                gt = sim.color_at(r, ft)
                if gt == endc:
                    continue

                def mk(rng, x=x, tn=tn, ft=ft, gt=gt):
                    seen = [col for _, col in s["rings"][x]["recolor"]] + [s["rings"][x]["color"]]
                    pool = list(dict.fromkeys([col for col in seen if col != gt] +
                                              [col for col in pal if col not in seen]))
                    o, a = L.mc(rng, gt, pool[:3])
                    return _q(2, "color_at_time", f"What colour is the border of the {c.lab[x]} when the timer reads "
                              f"{sim.fmt_reading(tn)}?", gt, o, a, True,
                              {"x": x, "t": tn, "min_fps": fps_of(c.k_at(ft, rec))})
                out.append(("color_at_time", (x, tn), mk, True))
                break
        fixed = [k for k in c.all if not s["rings"][k]["recolor"]]
        if len(fixed) >= 3:
            for x in c.all:
                if s["rings"][x]["recolor"]:
                    def mk(rng, x=x, fixed=fixed):
                        o, a, ids = _rings(rng, c, x, rng.sample(fixed, 3))
                        return _q(2, "which_recolors", "Which of these outlines changes colour during the video?",
                                  c.lab[x], o, a, True,
                                  {"opts": ids, "min_fps": fps_of(c.k_events([f for f, _ in s["rings"][x]["recolor"]]))})
                    out.append(("which_recolors", x, mk, True))
    return out


# ---- L3 measurement ---------------------------------------------------------------------
def level3(c: Ctx):
    s, out = c.s, []
    allev = c.evf
    if c.sc == "assemble":
        for x in c.all:
            def mk(rng, x=x):
                fr = s["rings"][x]["appear"]
                gt, o, a, kt = c.time_opts(rng, fr, [s["rings"][k]["appear"] for k in c.all if k != x])
                return _q(3, "appear_time", f"What does the timer read when the {c.lab[x]} appears?", gt, o, a, True,
                          {"x": x, "min_fps": fps_of(kt)})
            out.append(("appear_time", x, mk, True))
        vis_ev = [f for k in c.all for f in ([s["rings"][k]["appear"]] + [s["rings"][k]["vanish"]]) if f]
        for tn in c.readings(vis_ev):
            ft = sim.frame_of_reading(tn)
            v = len(sim.on_screen(s, ft))
            if v != len(c.endv):
                def mk(rng, tn=tn, v=v, ft=ft):
                    o, a = _num(rng, v, [len(c.endv)], lo=0)
                    return _q(3, "count_at", f"How many outlines are on screen when the timer reads "
                              f"{sim.fmt_reading(tn)}?", v, o, a, True,
                              {"t": tn, "min_fps": fps_of(c.k_at(ft, vis_ev))})
                out.append(("count_at", tn, mk, True))
    for x in c.all:
        if c.chg[x]:
            def mk(rng, x=x):
                fr = c.chg[x][0]
                gt, o, a, kt = c.time_opts(rng, fr, [f for k in c.all for f in c.chg[k] if f != fr])
                # a change shows up to 2k - 1 frames late when sampling every k-th frame (the sample
                # that straddles it can still read the old direction)
                return _q(3, "change_time", f"What does the timer read when the {c.lab[x]} first changes how it "
                          f"turns (reverses, stops or starts)?", gt, o, a, True,
                          {"x": x, "min_fps": fps_of(min((kt + 1) // 2, c.k_turn([x])))})
            out.append(("change_time", x, mk, True))

            def mk(rng, x=x):
                v = len(c.chg[x])
                o, a = _num(rng, v, lo=0)
                return _q(3, "n_changes", f"How many times does the {c.lab[x]} change how it turns (reversing, "
                          f"stopping or starting)?", v, o, a, True, {"x": x, "min_fps": fps_of(c.k_turn([x]))})
            out.append(("n_changes", x, mk, True))
    turn_ev = [f for k in c.all for f in c.chg[k]]
    vis_ev = [f for k in c.all for f in ([s["rings"][k]["appear"]] + [s["rings"][k]["vanish"]]) if f]
    if any(c.turns):
        for tn in c.readings(turn_ev + vis_ev)[::2]:
            ft = sim.frame_of_reading(tn)
            vis = sim.on_screen(s, ft)
            v = sum(1 for k in vis if sim.dir_at(s, k, ft))

            def mk(rng, tn=tn, v=v, ft=ft, vis=vis):
                o, a = _num(rng, v, [len(vis)], lo=0)
                kk = min(c.k_turn(vis), c.k_at(ft, turn_ev + vis_ev))
                return _q(3, "turning_at", f"When the timer reads {sim.fmt_reading(tn)}, how many outlines are "
                          f"turning?", v, o, a, True, {"t": tn, "min_fps": fps_of(kk)})
            out.append(("turning_at", tn, mk, True))
    if s["marks"]:
        for x in c.all:
            mine = [m for m in s["marks"] if m["ring"] == x]
            if mine:
                def mk(rng, x=x, mine=mine):
                    fr = min(m["add"] for m in mine)
                    gt, o, a, kt = c.time_opts(rng, fr, [m["add"] for m in s["marks"] if m["add"] != fr])
                    return _q(3, "mark_time", f"What does the timer read when the {c.lab[x]} gets its first mark?",
                              gt, o, a, True, {"x": x, "min_fps": fps_of(kt)})
                out.append(("mark_time", x, mk, True))
        adds = sorted(m["add"] for m in s["marks"])
        top = sim.tenths(c.nf - 1)
        for a_ in range(5, top - 15, 5):
            for ln in (20, 30, 40):
                b_ = a_ + ln
                if b_ > top - 5:
                    continue
                fa, fb = sim.frame_of_reading(a_), sim.frame_of_reading(b_)
                if any(abs(o - fa) < sim.GAP_F or abs(o - fb) < sim.GAP_F for o in c.evf):
                    continue
                v = sum(fa <= o < fb for o in adds)
                if v >= 1:
                    def mk(rng, a_=a_, b_=b_, v=v, fa=fa, fb=fb):
                        o, a = _num(rng, v, lo=0)
                        kk = min(c.k_events(adds), c.k_at(fa, c.evf), c.k_at(fb, c.evf))
                        return _q(3, "marks_window", f"How many marks are placed between {sim.fmt_reading(a_)} and "
                                  f"{sim.fmt_reading(b_)} on the timer?", v, o, a, True,
                                  {"a": a_, "b": b_, "min_fps": fps_of(kk)})
                    out.append(("marks_window", (a_, b_), mk, True))
    if c.sc == "recolor":
        for x in c.all:
            rec = [f for f, _ in s["rings"][x]["recolor"]]
            if rec:
                def mk(rng, x=x, rec=rec):
                    gt, o, a, kt = c.time_opts(rng, rec[0], [f for k in c.all for f, _ in s["rings"][k]["recolor"]
                                                             if f != rec[0]])
                    return _q(3, "recolor_time", f"What does the timer read when the {c.lab[x]} first changes colour?",
                              gt, o, a, True, {"x": x, "min_fps": fps_of(kt)})
                out.append(("recolor_time", x, mk, True))

                def mk(rng, x=x, rec=rec):
                    o, a = _num(rng, len(rec), lo=0)
                    return _q(3, "n_recolors", f"How many times does the {c.lab[x]} change colour?", len(rec), o, a,
                              True, {"x": x, "min_fps": fps_of(c.k_events(rec))})
                out.append(("n_recolors", x, mk, True))
    if c.hits:
        hf = [h[0] for h in c.hits]

        def mk(rng):
            gt, o, a, kt = c.time_opts(rng, hf[0], hf[1:])
            return _q(3, "hit_time", "What does the timer read when the nest first hits an edge of the box?", gt, o, a,
                      True, {"min_fps": fps_of(min(kt, c.k_events(hf) // 2))})
        out.append(("hit_time", None, mk, True))

        for wall in sim.WALLS:
            fw = [f for f, w in c.hits if w == wall]
            if fw and fw[0] != hf[0]:
                def mk(rng, wall=wall, fw=fw):
                    gt, o, a, kt = c.time_opts(rng, fw[0], [f for f in hf if f != fw[0]])
                    return _q(3, "hit_wall_time", f"What does the timer read when the nest first hits the {wall} edge "
                              f"of the box?", gt, o, a, True, {"wall": wall, "min_fps": fps_of(min(kt, c.k_events(hf) // 2))})
                out.append(("hit_wall_time", wall, mk, True))
        top = sim.tenths(c.nf - 1)
        for a_ in range(5, top - 15, 10):
            for ln in (20, 30, 40):
                b_ = a_ + ln
                fa, fb = sim.frame_of_reading(a_), sim.frame_of_reading(b_)
                if b_ > top - 5 or any(abs(o - fa) < sim.GAP_F or abs(o - fb) < sim.GAP_F for o in hf):
                    continue
                v = sum(fa <= o < fb for o in hf)
                if v and v != len(hf):
                    def mk(rng, a_=a_, b_=b_, v=v, fa=fa, fb=fb):
                        o, a = _num(rng, v, [len(hf)], lo=0)
                        kk = min(c.k_events(hf) // 2, c.k_at(fa, hf), c.k_at(fb, hf))
                        return _q(3, "hits_window", f"How many times does the nest hit an edge of the box between "
                                  f"{sim.fmt_reading(a_)} and {sim.fmt_reading(b_)} on the timer?", v, o, a, True,
                                  {"a": a_, "b": b_, "min_fps": fps_of(kk)})
                    out.append(("hits_window", (a_, b_), mk, True))

        def mk(rng):
            o, a = _num(rng, len(hf), lo=0)
            return _q(3, "n_hits", "How many times does the nest hit an edge of the box?", len(hf), o, a, True,
                      {"min_fps": fps_of(c.k_events(hf) // 2)})
        out.append(("n_hits", None, mk, True))
    return out


# ---- L4 integration ---------------------------------------------------------------------
def level4(c: Ctx):
    s, out = c.s, []
    if c.sc == "assemble" and not any(s["rings"][k]["vanish"] is not None for k in c.all):
        def mk(rng):
            gt = ORDERS[s["order"]]
            o, a = _words(rng, gt, list(ORDERS.values()))
            app = [s["rings"][k]["appear"] for k in c.all]
            return _q(4, "appear_pattern", "In which order do the outlines appear?", gt, o, a, True,
                      {"min_fps": fps_of(c.k_events(app))})
        out.append(("appear_pattern", None, mk, True))
    if c.sc == "assemble" and any(s["rings"][k]["vanish"] is not None for k in c.all):
        vis_ev = sorted(f for k in c.all for f in ([s["rings"][k]["appear"]] + [s["rings"][k]["vanish"]]) if f)
        v = max(len(sim.on_screen(s, f)) for f in vis_ev)
        if v != len(c.endv):
            def mk(rng, v=v):
                o, a = _num(rng, v, [len(c.endv), c.n], lo=1)
                return _q(4, "max_on_screen", "What is the largest number of outlines on screen at the same time?",
                          v, o, a, True, {"min_fps": fps_of(c.k_events(vis_ev))})
            out.append(("max_on_screen", None, mk, True))
    if c.sc == "assemble":
        app = {k: s["rings"][k]["appear"] for k in c.all}
        order = sorted(c.all, key=lambda k: app[k])
        for x in order[:-1]:
            v = sum(1 for k in c.all if app[k] > app[x])

            def mk(rng, x=x, v=v):
                o, a = _num(rng, v, lo=0)
                return _q(4, "appear_after", f"How many outlines appear after the {c.lab[x]}?", v, o, a, True,
                          {"x": x, "min_fps": fps_of(c.k_events(list(app.values())))})
            out.append(("appear_after", x, mk, True))
        if not any(s["rings"][k]["vanish"] is not None for k in c.all) and c.n >= 4:
            d = order[-1]

            def mk(rng, d=d):
                gt = ordinal(d, c.n)
                o, a = L.mc(rng, gt, [ordinal(j, c.n) for j in rng.sample([j for j in c.all if j != d], 3)])
                return _q(4, "last_depth", "Where in the finished nest is the outline that appears last?", gt, o, a,
                          True, {"min_fps": fps_of(c.k_events(list(app.values())))})
            out.append(("last_depth", None, mk, True))
    for x in c.all:
        if not sim.fold(s["rings"][x]):
            continue
        dirs = [d for _, _, d in c.runs[x]]
        if len(dirs) == 1 and dirs[0] == 0:
            continue

        def mk(rng, x=x, dirs=dirs):
            gt = pattern_text(dirs)
            near = pattern_alts(dirs)
            rng.shuffle(near)
            near.sort(key=lambda p: (abs(len(p) - len(dirs)), -sum(1 for u, v in zip(p, dirs) if u == v)))
            o, a = L.mc(rng, gt, [pattern_text(p) for p in near])
            when = " while it is on screen" if c.sc == "assemble" else " over the whole video"
            return _q(4, "spin_pattern", f"Which describes how the {c.lab[x]} turns{when}?", gt, o, a, True,
                      {"x": x, "min_fps": fps_of(c.k_turn([x]))})
        out.append(("spin_pattern", x, mk, True))
    ends = c.endv
    if any(c.turns[k] for k in ends):
        v = sum(1 for k in ends if sim.dir_at(s, k, c.nf - 1) == 1)

        def mk(rng, v=v):
            o, a = _num(rng, v, lo=0)
            return _q(4, "cw_end", "At the end of the video, how many outlines are turning clockwise?", v, o, a, True,
                      {"min_fps": fps_of(c.k_turn(ends))})
        out.append(("cw_end", None, mk, True))
    nch = {k: len(c.chg[k]) for k in c.all}
    top, order = unique_top(nch)
    if top is not None and nch[top] >= 2 and c.n >= 4:
        def mk(rng, top=top, order=order):
            o, a, ids = _rings(rng, c, top, order[1:])
            return _q(4, "most_changes", "Which outline changes how it turns (reverses, stops or starts) the most "
                      "times?", c.lab[top], o, a, True, {"min_fps": fps_of(c.k_turn(c.all))})
        out.append(("most_changes", None, mk, True))
    if s["marks"]:
        mk_ev = [m["add"] for m in s["marks"]] + [m["remove"] for m in s["marks"] if m["remove"]]
        tot = len(s["marks"])
        rem = sum(1 for m in s["marks"] if m["remove"] is not None)

        def mk(rng):
            o, a = _num(rng, tot, [tot - rem] if rem else [], lo=1)
            return _q(4, "total_added", "Counting every outline, how many marks are placed during the video?", tot, o,
                      a, True, {"min_fps": fps_of(c.k_events(mk_ev))})
        out.append(("total_added", None, mk, True))
        if rem:
            def mk(rng):
                o, a = _num(rng, rem, lo=0)
                return _q(4, "total_removed", "Counting every outline, how many marks are taken off during the video?",
                          rem, o, a, True, {"min_fps": fps_of(c.k_events(mk_ev))})
            out.append(("total_removed", None, mk, True))
        cnt = {k: sum(1 for m in s["marks"] if m["ring"] == k) for k in c.all}
        top, order = unique_top(cnt)
        endc = {k: sum(1 for m in s["marks"] if m["ring"] == k and m["remove"] is None) for k in c.all}
        if top is not None and len(c.all) >= 4:
            def mk(rng, top=top, order=order):
                o, a, ids = _rings(rng, c, top, order[1:])
                return _q(4, "most_added", "Over the whole video, which outline has the most marks placed on it?",
                          c.lab[top], o, a, True, {"min_fps": fps_of(c.k_events(mk_ev))})
            out.append(("most_added", None, mk, True))
    if c.sc == "recolor":
        pal = sim.THEMES[s["style"]["theme"]]["colors"]
        for x in c.all:
            r = s["rings"][x]
            if r["recolor"]:
                seq = [r["color"]] + [col for _, col in r["recolor"]]

                def mk(rng, x=x, seq=seq):
                    alts = []
                    if len(seq) > 2:
                        alts.append(list(reversed(seq)))
                    alts.append(seq[:-1])
                    alts.append(seq + [rng.choice([p for p in pal if p != seq[-1]])])
                    for i in range(len(seq)):
                        alt = list(seq)
                        alt[i] = rng.choice([p for p in pal if p not in (seq[i], seq[i - 1] if i else None,
                                                                         seq[i + 1] if i + 1 < len(seq) else None)])
                        alts.append(alt)
                    alts.append(list(reversed(seq)))
                    texts = [seq_text(a_) for a_ in alts if a_ != seq and len(a_) >= 1]
                    rng.shuffle(texts)
                    o, a = L.mc(rng, seq_text(seq), texts)
                    return _q(4, "color_sequence", f"Which border colours does the {c.lab[x]} show, in order?",
                              seq_text(seq), o, a, True,
                              {"x": x, "min_fps": fps_of(c.k_events([f for f, _ in s["rings"][x]["recolor"]]))})
                out.append(("color_sequence", x, mk, True))
        rec = [f for k in c.all for f, _ in s["rings"][k]["recolor"]]

        def mk(rng, rec=rec):
            o, a = _num(rng, len(rec), lo=1)
            return _q(4, "total_recolors", "Counting every outline, how many colour changes are there in the video?",
                      len(rec), o, a, True, {"min_fps": fps_of(c.k_events(rec))})
        out.append(("total_recolors", None, mk, True))
    if c.hits:
        hf = [h[0] for h in c.hits]
        for axis, walls in (("x", ("left", "right")), ("y", ("top", "bottom"))):
            v = sum(1 for _, w in c.hits if w in walls)

            def mk(rng, v=v, walls=walls, axis=axis):
                o, a = _num(rng, v, [len(hf)] if len(hf) != v else [], lo=0)
                return _q(4, "hits_axis", f"How many times does the nest hit the {walls[0]} or {walls[1]} edge of the "
                          f"box?", v, o, a, True, {"axis": axis, "min_fps": fps_of(c.k_events(hf) // 2)})
            out.append(("hits_axis", axis, mk, True))
    return out


# ---- L5 counterfactual (re-simulated on an edited copy) ---------------------------------------
def level5(c: Ctx):
    s, out = c.s, []
    ends = c.endv
    vis_ev = [f for k in c.all for f in ([s["rings"][k]["appear"]] + [s["rings"][k]["vanish"]]) if f]
    if c.sc == "assemble":
        for tn in c.readings(vis_ev, lo_s=1.5)[::2]:
            ft = sim.frame_of_reading(tn)
            v = len(sim.on_screen(s, ft - 1))
            if v != len(ends):
                def mk(rng, tn=tn, v=v, ft=ft):
                    o, a = _num(rng, v, [len(ends)], lo=0)
                    return _q(5, "cf_stop_count", f"If the video had stopped the moment the timer first read "
                              f"{sim.fmt_reading(tn)}, how many outlines would have been on screen?", v, o, a, True,
                              {"t": tn, "min_fps": fps_of(c.k_at(ft, vis_ev))})
                out.append(("cf_stop_count", tn, mk, True))
        if not any(s["rings"][k]["vanish"] is not None for k in c.all) and c.n >= 4:
            app = sorted(c.all, key=lambda k: s["rings"][k]["appear"])
            for pos in range(len(app)):
                # re-simulate: the same appearance moments, handed out in the reverse order
                alt = copy.deepcopy(s)
                times = [s["rings"][k]["appear"] for k in app]
                for k, t in zip(reversed(app), times):
                    alt["rings"][k]["appear"] = t
                who = sorted(c.all, key=lambda k: alt["rings"][k]["appear"])[pos]
                if who == app[pos]:
                    continue

                def mk(rng, pos=pos, who=who, app=app):
                    pool = [app[pos]] + [k for k in app if k not in (who, app[pos])]
                    o, a, ids = _rings(rng, c, who, pool)
                    word = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh")[pos]
                    return _q(5, "cf_reverse_order", f"If the outlines had appeared in the reverse order (at the same "
                              f"moments), which one would have been the {word} to appear?", c.lab[who], o, a, True,
                              {"pos": pos, "min_fps": fps_of(c.k_events([s["rings"][k]["appear"] for k in app]))})
                out.append(("cf_reverse_order", pos, mk, True))
    for d in range(len(ends) - 2 if c.n >= 4 else 0):
        y, x, z = ends[d], ends[d + 1], ends[d + 2]
        alt = copy.deepcopy(s)
        alt["rings"][x]["vanish"] = 0
        alt["rings"][x]["appear"] = 0
        rest = sim.on_screen(alt, c.nf - 1)
        who = rest[rest.index(y) + 1]

        def mk(rng, x=x, y=y, who=who):
            pool = [x] + [k for k in ends if k not in (x, y, who)] + [k for k in c.all if k not in ends]
            o, a, ids = _rings(rng, c, who, pool)
            return _q(5, "cf_remove", f"If the {c.lab[x]} were taken out of the nest (everything else the same), which "
                      f"outline would be directly inside the {c.lab[y]} at the end?", c.lab[who], o, a, False,
                      {"x": x, "y": y, "min_fps": fps_of(c.k_end(vis_ev))})
        if 1 + sum(1 for k in c.all if k not in (x, y, who)) >= 3:
            out.append(("cf_remove", x, mk, False))
    turn_ev = [f for k in c.all for f in c.chg[k]]
    if any(c.chg) and c.sc != "assemble":
        alt = copy.deepcopy(s)
        for r in alt["rings"]:
            r["spin"] = r["spin"][:1]
        v = sum(1 for k in sim.on_screen(alt, c.nf - 1) if sim.dir_at(alt, k, c.nf - 1) == 1)
        real = sum(1 for k in ends if sim.dir_at(s, k, c.nf - 1) == 1)
        if v != real:
            def mk(rng, v=v, real=real):
                o, a = _num(rng, v, [real], lo=0)
                return _q(5, "cf_keep_dir", "If every outline had kept turning the way it turned at the start (no "
                          "reversing, stopping or starting), how many would be turning clockwise at the end?", v, o, a,
                          True, {"min_fps": fps_of(c.k_turn(c.all))})
            out.append(("cf_keep_dir", None, mk, True))
    for x in ends:
        if not sim.dir_at(s, x, c.nf - 1):
            continue
        alt = copy.deepcopy(s)
        alt["rings"][x]["spin"] = [[f0, -w] for f0, w in alt["rings"][x]["spin"]]
        dx = sim.dir_at(alt, x, c.nf - 1)
        v = sum(1 for k in ends if k != x and sim.dir_at(alt, k, c.nf - 1) == dx)

        def mk(rng, x=x, v=v):
            o, a = _num(rng, v, lo=0)
            return _q(5, "cf_flip", f"If the {c.lab[x]} had always turned the opposite way (everything else the same), "
                      f"how many of the other outlines would be turning the same way as it at the end?", v, o, a, True,
                      {"x": x, "min_fps": fps_of(c.k_turn(ends))})
        out.append(("cf_flip", x, mk, True))
    if any(c.turns):
        for tn in c.readings(turn_ev + vis_ev, lo_s=1.5)[1::2]:
            ft = sim.frame_of_reading(tn)
            vis = sim.on_screen(s, ft)
            v = sum(1 for k in vis if sim.dir_at(s, k, ft) == 1)

            def mk(rng, tn=tn, v=v, ft=ft, vis=vis):
                o, a = _num(rng, v, lo=0)
                kk = min(c.k_turn(vis), c.k_at(ft, turn_ev + vis_ev))
                return _q(5, "cf_stop_turning", f"If the video had stopped the moment the timer first read "
                          f"{sim.fmt_reading(tn)}, how many outlines would have been turning clockwise at that "
                          f"moment?", v, o, a, True, {"t": tn, "min_fps": fps_of(kk)})
            out.append(("cf_stop_turning", tn, mk, True))
    if s["marks"]:
        mk_ev = [m["add"] for m in s["marks"]] + [m["remove"] for m in s["marks"] if m["remove"]]
        alt = copy.deepcopy(s)
        for m in alt["marks"]:
            m["remove"] = None
        for x in c.all:
            lost = [m for m in s["marks"] if m["ring"] == x and m["remove"] is not None]
            if lost:
                v = sum(1 for m in alt["marks"] if m["ring"] == x and sim.mark_visible(m, c.nf - 1))
                real = sum(1 for m in s["marks"] if m["ring"] == x and sim.mark_visible(m, c.nf - 1))

                def mk(rng, x=x, v=v, real=real):
                    o, a = _num(rng, v, [real], lo=0)
                    return _q(5, "cf_no_removal", f"If no mark had ever been taken off, how many marks would the "
                              f"{c.lab[x]} carry at the end?", v, o, a, True, {"x": x, "min_fps": fps_of(c.k_events(mk_ev))})
                out.append(("cf_no_removal", x, mk, True))
        for tn in c.readings(mk_ev, lo_s=1.5)[::2]:
            ft = sim.frame_of_reading(tn)
            v = sum(1 for m in s["marks"] if sim.mark_visible(m, ft - 1))
            real = sum(1 for m in s["marks"] if sim.mark_visible(m, c.nf - 1))
            if v != real and v:
                def mk(rng, tn=tn, v=v, ft=ft, real=real):
                    o, a = _num(rng, v, [real], lo=0)
                    return _q(5, "cf_stop_marks", f"If the video had stopped the moment the timer first read "
                              f"{sim.fmt_reading(tn)}, how many marks would have been on the outlines?", v, o, a, True,
                              {"t": tn, "min_fps": fps_of(min(c.k_at(ft, mk_ev), c.k_events(mk_ev)))})
                out.append(("cf_stop_marks", tn, mk, True))
    if c.sc == "recolor":
        rec = sorted((f, k) for k in c.all for f, _ in s["rings"][k]["recolor"])
        pal = sim.THEMES[s["style"]["theme"]]["colors"]
        if rec:
            f_last, k_last = rec[-1]
            alt = copy.deepcopy(s)
            alt["rings"][k_last]["recolor"] = [e for e in alt["rings"][k_last]["recolor"] if e[0] != f_last]
            for col in pal:
                v = sum(1 for k in c.all if sim.color_at(alt["rings"][k], c.nf - 1) == col)
                real = sum(1 for k in c.all if sim.color_at(s["rings"][k], c.nf - 1) == col)
                if v != real:
                    def mk(rng, col=col, v=v, real=real):
                        o, a = _num(rng, v, [real], lo=0)
                        return _q(5, "cf_no_last_recolor", f"If the last colour change of the video had never happened, "
                                  f"how many outlines would have a {col} border at the end?", v, o, a, True,
                                  {"c": col, "min_fps": fps_of(c.k_events([f for f, _ in rec]))})
                    out.append(("cf_no_last_recolor", col, mk, True))
        recf = [f for f, _ in rec]
        for x in c.all:
            r = s["rings"][x]
            if not r["recolor"]:
                continue
            endc = sim.color_at(r, c.nf - 1)
            for tn in c.readings(recf, lo_s=1.5)[::-1]:
                ft = sim.frame_of_reading(tn)
                gt = sim.color_at(r, ft - 1)
                if gt == endc:
                    continue

                def mk(rng, x=x, tn=tn, ft=ft, gt=gt):
                    seen = [col for _, col in s["rings"][x]["recolor"]] + [s["rings"][x]["color"]]
                    pool = list(dict.fromkeys([col for col in seen if col != gt] +
                                              [col for col in pal if col not in seen]))
                    o, a = L.mc(rng, gt, pool[:3])
                    return _q(5, "cf_stop_color", f"If the video had stopped the moment the timer first read "
                              f"{sim.fmt_reading(tn)}, what colour would the border of the {c.lab[x]} have been?",
                              gt, o, a, True, {"x": x, "t": tn, "min_fps": fps_of(c.k_at(ft, recf))})
                out.append(("cf_stop_color", (x, tn), mk, True))
                break
    if c.hits:
        hf = [h[0] for h in c.hits]
        for wall in sim.WALLS:
            t_cross = [t for t, w in sim.hits(s) if w == wall]
            if not t_cross:
                continue
            v = sum(1 for t, w in sim.hits(s, drop=wall) if t < t_cross[0])
            if v == len(hf):
                continue

            def mk(rng, wall=wall, v=v):
                o, a = _num(rng, v, [len(hf)], lo=0)
                return _q(5, "cf_no_wall", f"If the {wall} edge of the box were missing, the nest would leave the box "
                          f"there. How many times would it hit an edge of the box before leaving?", v, o, a, True,
                          {"wall": wall, "min_fps": fps_of(c.k_events(hf) // 2)})
            out.append(("cf_no_wall", wall, mk, True))
        for tn in c.readings(hf, lo_s=1.5)[::2]:
            ft = sim.frame_of_reading(tn)
            v = sum(1 for f in hf if f < ft)
            if 0 < v < len(hf):
                def mk(rng, tn=tn, v=v, ft=ft):
                    o, a = _num(rng, v, [len(hf)], lo=0)
                    return _q(5, "cf_stop_hits", f"If the video had stopped the moment the timer first read "
                              f"{sim.fmt_reading(tn)}, how many times would the nest have hit an edge of the box?",
                              v, o, a, True, {"t": tn, "min_fps": fps_of(min(c.k_at(ft, hf), c.k_events(hf) // 2))})
                out.append(("cf_stop_hits", tn, mk, True))
    return out


LEVELS = (level1, level2, level3, level4, level5)


def supply(scene) -> list[int]:
    """Distinct candidate questions per level (sim.check needs >= 3 everywhere)."""
    c = Ctx(scene)
    return [len(f(c)) for f in LEVELS]


GENERIC = ("pair_dir", "same_dir", "turning_at", "cw_end", "spin_pattern", "cf_flip", "cf_stop_turning",
           "inside_count", "inside_of", "fastest", "never_turns")


def _pick(rng, cands, n, scenario=None):
    """Round-robin over shuffled keys. Outside the spin scenario the generic turning
    and nesting keys usually (60 %) queue behind the scenario's own keys."""
    bykey = {}
    for key, ident, mk, motion in cands:
        bykey.setdefault(key, []).append((ident, mk))
    keys = list(bykey)
    rng.shuffle(keys)
    if scenario not in (None, "spin"):
        late = {k: k in GENERIC and rng.random() < 0.6 for k in keys}
        keys.sort(key=lambda k: late[k])
    chosen = []
    while len(chosen) < n and any(bykey.values()):
        for key in keys:
            if bykey[key] and len(chosen) < n:
                ident, mk = bykey[key].pop(rng.randrange(len(bykey[key])))
                chosen.append(mk)
    return chosen


def build(scene: dict, rng: random.Random) -> list[dict]:
    """Per-level counts from ten_counts; within a level, keys round-robin. On L2-L5
    the static templates (inside_of, cf_remove) are used at most once per clip
    (30 %) or when a level runs out of motion templates."""
    counts = L.ten_counts(rng)
    c = Ctx(scene)
    static_left = 1 if rng.random() < 0.3 else 0
    qs = []
    for lv, fn in enumerate(LEVELS):
        cands = fn(c)
        if lv == 0:
            picks = _pick(rng, cands, counts[lv], c.sc)
        else:
            mot = [cd for cd in cands if cd[3]]
            sta = [cd for cd in cands if not cd[3]]
            use_static = static_left > 0 and sta and rng.random() < 0.5
            picks = _pick(rng, mot + (sta if use_static else []), counts[lv], c.sc)
            if use_static:
                static_left -= 1
            if len(picks) < counts[lv]:
                picks += _pick(rng, sta if not use_static else [], counts[lv] - len(picks))
        qs += [mk(rng) for mk in picks]
    return L.number(qs)
