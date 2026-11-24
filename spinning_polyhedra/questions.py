"""10-question, 5-rung ladder for spinning_polyhedra (common/ladder5.py, LADDER = "ten").

Nothing in the frame is labelled except the timer, so questions name things
by what is drawn:
  * solids: "the solid", or "the solid on the left / right" (pair scenario);
  * markers: colour + shape, "the red dot", "the blue square", "the green
    diamond" (colours are unique in a clip);
  * "corners" are vertices; "crossing in front of the spin axis" means a
    marker passing through the middle of its solid on the side nearest the
    viewer (azimuth 0 about the upright axis);
  * moments: "at the very start / end", "when it first stops", or a timer
    reading ("at 4.0 s").

The generator side answers from sim's closed-form motion (segment angles,
analytic crossings); answers.py re-derives everything from the renderer's
per-frame geometry. Every question stores params["min_fps"].

Templates per level (derivation keys):
  L1 front_dir, n_stops, spin_axis, mark_side
  L2 more_fv, face_kinds, common_face, speed_trend, halves, start_end_dir,
     first_front, wider_orbit, speed_compare, faster, dir_pair, vf_pair
  L3 n_faces, n_corners, n_edges, n_turns, turns_by_time, period, stop_time,
     resume_time, count_passes, first_pass_time, turns_before_stop
  L4 name, mark_degree, pass_time, passes_before_stop, extra_turn,
     side_at_time, mark_at_stop, pair_sum, turn_diff, turns_total, net_dir,
     axis_seq
  L5 cf_truncate_faces, cf_truncate_corners, cf_dual_corners, cf_more_sides,
     cf_speed, cf_speed_first, cf_other_way, cf_stop_side, cf_no_stop,
     cf_no_switch, cf_swap, cf_passes_double
At most one question per level on L2-L5 has requires_motion=False (the pure
shape-count templates).
"""
from __future__ import annotations
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
ORD = {1: "first", 2: "second", 3: "third"}
PASS = "cross in front of the spin axis (through the middle of its solid, on the side nearest you)"
QUAD = {("left", "near"): "left of the spin axis, on the side nearest you",
        ("left", "far"): "left of the spin axis, on the far side",
        ("right", "near"): "right of the spin axis, on the side nearest you",
        ("right", "far"): "right of the spin axis, on the far side"}
DIRS = ["to the left", "to the right", "upward", "downward"]
AXIS = {"y": "an upright axis (it turns like a turntable)", "x": "a left-right axis (it tumbles forward or backward)",
        "z": "a front-back axis (it turns like a clock hand)"}
AXIS_OPTS = list(AXIS.values()) + ["a slanted axis, tilted to one side"]
TREND = ["speeding up", "slowing down", "keeping a steady speed", "speeding up, then slowing down"]
HALVES = ["further in the first half", "further in the second half", "about the same in both halves",
          "it does not turn at all in the second half"]
SED = ["the same way as at the start", "the opposite way", "up or down instead of sideways",
       "round like a clock hand instead"]
SPEEDCMP = ["faster before the stop", "faster after the stop", "the same speed before and after",
            "it never stops"]
FACE_KINDS = ["all the same shape", "two different shapes", "three different shapes", "four different shapes"]
FV = ["more faces than corners", "more corners than faces", "the same number of each",
      "twice as many faces as corners"]
FASTER = ["the solid on the left", "the solid on the right", "they spin at the same speed",
          "the left one at first, then the right one"]
DIRPAIR = {("left", "left"): "both with their near side moving left",
           ("right", "right"): "both with their near side moving right",
           ("left", "right"): "the left solid's near side moves left, the right solid's moves right",
           ("right", "left"): "the left solid's near side moves right, the right solid's moves left"}
VFPAIR = {(0, 0): "the left one has more of both", (1, 1): "the right one has more of both",
          (0, 1): "the left one has more corners, the right one more faces",
          (1, 0): "the right one has more corners, the left one more faces"}
NETDIR = ["with its near side moving left", "with its near side moving right", "the same amount each way",
          "it only ever turns one way"]
EXTRA = ["less than a quarter turn", "between a quarter and a half turn", "between a half and three quarters of a turn",
         "more than three quarters of a turn"]
AXSEQ = {("y", "x"): "first about an upright axis, then about a left-right axis",
         ("y", "z"): "first about an upright axis, then about a front-back axis",
         ("x", "y"): "first about a left-right axis, then about an upright axis",
         ("z", "y"): "first about a front-back axis, then about an upright axis"}
FAM_TEXT = {
    "prism": "Suppose the two matching end faces of {X} each had one more side, with the faces around the sides "
             "still four-sided.",
    "antiprism": "Suppose the two matching end faces of {X} each had one more side, with the band of triangles "
                 "between them built the same way.",
    "pyramid": "Suppose the one face of {X} that is not a triangle had one more side, with triangles still meeting "
               "at a single top corner.",
    "bipyramid": "Suppose the middle ring of corners of {X} had one more corner, with triangles still running up to "
                 "the top corner and down to the bottom corner.",
}
FAM_EDGES = {"prism": 3, "antiprism": 4, "pyramid": 2, "bipyramid": 3}


# ---- phrasing (shared with answers.py, formatting only) ---------------------------
def bn(scene, b) -> str:
    if len(scene["bodies"]) == 1:
        return "the solid"
    return "the solid on the left" if b == 0 else "the solid on the right"


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def mn(scene, m) -> str:
    mk = scene["marks"][m]
    return f"the {mk['colour']} {mk['shape']}"


def tstr(tn: int) -> str:
    return f"{sim.fmt_t(tn)} s"


def lr(w) -> str:
    return "right" if w > 0 else "left"


def dir_word(axis, w):
    if axis == "y":
        return "to the right" if w > 0 else "to the left"
    if axis == "x":
        return "downward" if w > 0 else "upward"
    return None


def extra_label(frac):
    return EXTRA[min(3, int(frac * 4))]


# ---- helpers -------------------------------------------------------------------------
def ceil1(x: float) -> float:
    return max(0.1, math.ceil(x * 10 - 1e-9) / 10)


def rot_fps(body, f0=0, f1=None) -> float:
    """Samples per second so the solid turns at most 90 degrees between samples."""
    return sim.max_speed(body, f0, f1) / 90


def _q(level, key, text, gt, opts, letter, motion, params):
    return L.question(level, key, text, gt, opts, letter, motion, params=params)


def _opts(rng, gt, distractors):
    opts, a = L.mc(rng, gt, distractors)
    if len(opts) != 4:
        raise ValueError(f"need 4 options for {gt!r}: {opts}")
    return opts, a


def _num(rng, gt, extra=(), lo=0):
    return _opts(rng, gt, [x for x in extra if x != gt and x >= lo] + L.near_ints(gt, lo))


def time_options(rng, tn: int, n: int):
    """Correct timer reading tn (tenths) plus 3 readings a whole multiple of `step` away, all >= 0.3 s
    from either end of the clip. Returns (options, letter, step in s)."""
    last = sim.tenths(n - 1)
    for step in rng.sample([10, 12, 15, 20], 4) + [10, 8]:
        cands = [tn + k * step for k in (-4, -3, -2, -1, 1, 2, 3, 4) if 3 <= tn + k * step <= last - 3]
        if len(cands) >= 3:
            pick = rng.sample(cands, 3)
            opts, a = _opts(rng, tstr(tn), [tstr(x) for x in pick])
            return opts, a, step / 10
    raise ValueError("no time options")


def trend(run) -> str | None:
    """Speed profile of one stretch of turning (closed form)."""
    if len(run) == 2:
        a, b = run
        if abs(a["w1"]) >= 1.5 * abs(a["w0"]) and abs(b["w0"]) >= 1.5 * abs(b["w1"]):
            return TREND[3]
        return None
    s = run[0]
    r = abs(s["w1"]) / abs(s["w0"])
    if r == 1:
        return TREND[2]
    if r >= 1.5:
        return TREND[0]
    if r <= 1 / 1.5:
        return TREND[1]
    return None


def half_angles(body, n):
    """Degrees turned (unsigned, all axes) in the first and second half of the clip."""
    h = (n - 1) / 2
    first = second = 0.0
    for s in body["motion"]:
        if sim.is_pause(s):
            continue
        a_mid = sim.seg_angle(s, h) if s["f0"] < h else 0.0
        if s["f1"] <= h:
            first += abs(sim.seg_total(s))
        elif s["f0"] >= h:
            second += abs(sim.seg_total(s))
        else:
            first += abs(a_mid)
            second += abs(sim.seg_total(s) - a_mid)
    return first, second


def halves_label(body, n):
    a, b = half_angles(body, n)
    runs = sim.runs(body)
    if len(runs) == 1 and len(runs[0]) == 1 and runs[0][0]["w0"] == runs[0][0]["w1"]:
        return HALVES[2]
    if b < 1e-9:
        return HALVES[3]
    if a >= 1.4 * b:
        return HALVES[0]
    if b >= 1.4 * a:
        return HALVES[1]
    return None


def start_end_label(body):
    s0, s1 = body["motion"][0], body["motion"][-1]
    if s1["axis"] == "y":
        return SED[0] if (s0["w0"] > 0) == (s1["w1"] > 0) else SED[1]
    return SED[2] if s1["axis"] == "x" else SED[3]


def side_window(scene, m, f, body=None):
    """Frames around f during which marker m keeps the same (side, depth) label."""
    body = body or scene["bodies"][scene["marks"][m]["body"]]
    last = min([s["f0"] for s in body["motion"] if s["axis"] != "y"] + [scene["n_frames"] - 1])
    lab = sim.side_at(scene, m, f, body)
    lo = hi = f
    while lo > 0 and sim.side_at(scene, m, lo - 1, body) == lab:
        lo -= 1
    while hi < last and sim.side_at(scene, m, hi + 1, body) == lab:
        hi += 1
    return lo, hi


class Ctx:
    def __init__(self, scene):
        self.s = scene
        self.n = scene["n_frames"]
        self.B = scene["bodies"]
        self.nb = len(self.B)
        self.marks_of = [[i for i, m in enumerate(scene["marks"]) if m["body"] == b] for b in range(self.nb)]
        self.counts = [sim.SOLIDS[b["solid"]] for b in self.B]
        self.rot_all = [rot_fps(b) for b in self.B]
        self.safe = [sim.pass_safe(scene, i) if sim.y_only(self.B[m["body"]]) else False
                     for i, m in enumerate(scene["marks"])]
        self.passes = [sim.passes(scene, i) if self.safe[i] else [] for i in range(len(scene["marks"]))]

    def body_of(self, m):
        return self.s["marks"][m]["body"]

    def fps(self, b, *extra, f0=0, f1=None):
        return ceil1(max([rot_fps(self.B[b], f0, f1)] + list(extra)))

    def plain(self, b):
        """y-only, one stretch of turning (no pauses)."""
        body = self.B[b]
        return sim.y_only(body) and not sim.pauses(body)

    def min_pause_s(self, b):
        ps = sim.pauses(self.B[b])
        return min((f2 - f1) / sim.FPS for f1, f2 in ps) if ps else None


# ---- level 1: observation --------------------------------------------------------------
def level1(c: Ctx):
    s, out = c.s, []
    for b, body in enumerate(c.B):
        for when in ("start", "end"):
            seg = body["motion"][0] if when == "start" else body["motion"][-1]
            w = seg["w0"] if when == "start" else seg["w1"]
            gt = dir_word(seg["axis"], w)
            if gt is None:
                continue
            f0, f1 = (0, 24) if when == "start" else (c.n - 25, c.n - 1)

            def mk(rng, b=b, when=when, gt=gt, f0=f0, f1=f1):
                o, a = _opts(rng, gt, DIRS)
                text = f"At the very {when}, which way do the edges of {bn(s, b)} nearest to you move?"
                return _q(1, "front_dir", text, gt, o, a, True, {"b": b, "when": when, "min_fps": c.fps(b, f0=f0, f1=f1)})
            out.append(("front_dir", (b, when), mk))

        def mk(rng, b=b, body=body):
            gt = len(sim.pauses(body))
            o, a = _num(rng, gt)
            mp = c.min_pause_s(b)
            text = f"How many times does {bn(s, b)} stop turning for a moment and then turn again?"
            return _q(1, "n_stops", text, gt, o, a, True,
                      {"b": b, "min_fps": c.fps(b, 2 / mp if mp else 0)})
        out.append(("n_stops", b, mk))
        for when in ("start", "end"):
            seg = body["motion"][0] if when == "start" else body["motion"][-1]
            gt = AXIS[seg["axis"]]
            f0, f1 = (0, 24) if when == "start" else (c.n - 25, c.n - 1)

            def mk(rng, b=b, when=when, gt=gt, f0=f0, f1=f1):
                o, a = _opts(rng, gt, AXIS_OPTS)
                text = f"At the very {when}, about which kind of axis is {bn(s, b)} turning?"
                return _q(1, "spin_axis", text, gt, o, a, True, {"b": b, "when": when, "min_fps": c.fps(b, f0=f0, f1=f1)})
            out.append(("spin_axis", (b, when), mk))
    for m, mk_ in enumerate(s["marks"]):
        b = mk_["body"]
        for when in ("start", "end"):
            if when == "end" and not sim.y_only(c.B[b]):
                continue
            f = 0 if when == "start" else c.n - 1
            lab = sim.side_at(s, m, f)
            if lab is None:
                continue
            lo, hi = side_window(s, m, f)
            if hi - lo < 12:
                continue

            def mk(rng, m=m, b=b, when=when, lab=lab, lo=lo, hi=hi):
                gt = QUAD[lab]
                o, a = _opts(rng, gt, list(QUAD.values()))
                text = f"At the very {when} of the clip, where is {mn(s, m)}?"
                return _q(1, "mark_side", text, gt, o, a, True,
                          {"m": m, "when": when, "min_fps": c.fps(b, 2 / ((hi - lo) / sim.FPS),
                                                          f0=max(0, lo - 12), f1=min(c.n - 1, hi + 12))})
            out.append(("mark_side", (m, when), mk))
    return out


# ---- level 2: relational -------------------------------------------------------------
def level2(c: Ctx):
    s, out = c.s, []
    for b, body in enumerate(c.B):
        V, E, F, shapes = c.counts[b]

        def mk(rng, b=b, V=V, F=F):
            gt = FV[0] if F > V else FV[1] if V > F else FV[2]
            o, a = _opts(rng, gt, FV)
            text = f"Compared with its number of corners, how many faces does {bn(s, b)} have?"
            return _q(2, "more_fv", text, gt, o, a, False, {"b": b, "min_fps": c.fps(b)})
        out.append(("more_fv", b, mk))

        def mk(rng, b=b, shapes=shapes):
            gt = FACE_KINDS[len(shapes) - 1]
            o, a = _opts(rng, gt, FACE_KINDS)
            text = f"Which is true of the faces of {bn(s, b)}?"
            return _q(2, "face_kinds", text, gt, o, a, False, {"b": b, "min_fps": c.fps(b)})
        out.append(("face_kinds", b, mk))
        top = max(shapes.values())
        majors = [k for k, v in shapes.items() if v == top]
        if len(majors) == 1:
            def mk(rng, b=b, k=majors[0]):
                gt = sim.SHAPE[k]
                o, a = _opts(rng, gt, list(sim.SHAPE.values()))
                text = f"What shape are most of the faces of {bn(s, b)}?"
                return _q(2, "common_face", text, gt, o, a, False, {"b": b, "min_fps": c.fps(b)})
            out.append(("common_face", b, mk))
        if c.plain(b):
            gt = trend(sim.runs(body)[0])
            if gt:
                def mk(rng, b=b, gt=gt):
                    o, a = _opts(rng, gt, TREND)
                    text = f"How does the turning speed of {bn(s, b)} change during the clip?"
                    return _q(2, "speed_trend", text, gt, o, a, True, {"b": b, "min_fps": c.fps(b)})
                out.append(("speed_trend", b, mk))
        gt = halves_label(body, c.n)
        if gt:
            def mk(rng, b=b, gt=gt):
                o, a = _opts(rng, gt, HALVES)
                text = f"Does {bn(s, b)} turn further in the first half of the clip or in the second half?"
                return _q(2, "halves", text, gt, o, a, True, {"b": b, "min_fps": c.fps(b)})
            out.append(("halves", b, mk))

        def mk(rng, b=b, gt=start_end_label(body)):
            o, a = _opts(rng, gt, SED)
            text = f"At the very end, how do the edges of {bn(s, b)} nearest to you move, compared with the very start?"
            return _q(2, "start_end_dir", text, gt, o, a, True,
                      {"b": b, "min_fps": c.fps(b, 2 / c.min_pause_s(b) if c.min_pause_s(b) else 0)})
        out.append(("start_end_dir", b, mk))
        ms = c.marks_of[b]
        if len(ms) >= 2 and all(c.safe[m] and c.passes[m] for m in ms):
            firsts = sorted((c.passes[m][0], m) for m in ms)
            gap = firsts[1][0] - firsts[0][0]
            if gap >= 12:
                def mk(rng, b=b, ms=ms, firsts=firsts, gap=gap):
                    gt = mn(s, firsts[0][1])
                    extra = [mn(s, m) for m in ms] + ["they cross at the same moment", "none of them crosses"]
                    o, a = _opts(rng, gt, extra)
                    text = f"Which marker is the first to {PASS}?"
                    return _q(2, "first_front", text, gt, o, a, True,
                              {"b": b, "min_fps": c.fps(b, 2 / (gap / sim.FPS))})
                out.append(("first_front", b, mk))
        if len(ms) >= 2:
            rs = sorted((sim.orbit(s, m)[0], m) for m in ms)
            if rs[-1][0] >= 1.3 * rs[-2][0]:
                def mk(rng, b=b, ms=ms, rs=rs):
                    gt = mn(s, rs[-1][1])
                    extra = [mn(s, m) for m in ms] + ["they all sweep the same size of circle",
                                                      "none of them moves"]
                    o, a = _opts(rng, gt, extra)
                    text = (f"As {bn(s, b)} turns about the upright axis, which marker sweeps around the widest "
                            f"circle?")
                    return _q(2, "wider_orbit", text, gt, o, a, True, {"b": b, "ms": ms, "min_fps": c.fps(b)})
                if sim.y_only(body):
                    out.append(("wider_orbit", b, mk))
        rr = sim.runs(body)
        if len(rr) >= 2:
            w1, w2 = abs(rr[0][-1]["w1"]), abs(rr[1][0]["w0"])
            if w1 == w2 or max(w1, w2) >= 1.5 * min(w1, w2):
                gt = SPEEDCMP[2] if w1 == w2 else SPEEDCMP[0] if w1 > w2 else SPEEDCMP[1]

                def mk(rng, b=b, gt=gt):
                    o, a = _opts(rng, gt, SPEEDCMP)
                    text = (f"Compare how fast {bn(s, b)} turns just before it first stops with how fast it turns "
                            f"just after it starts again.")
                    return _q(2, "speed_compare", text, gt, o, a, True,
                              {"b": b, "min_fps": c.fps(b, 2 / c.min_pause_s(b))})
                out.append(("speed_compare", b, mk))
    if c.nb == 2:
        w = [abs(b["motion"][0]["w0"]) for b in c.B]
        if max(w) >= 1.5 * min(w):
            def mk(rng, w=w):
                gt = FASTER[0] if w[0] > w[1] else FASTER[1]
                o, a = _opts(rng, gt, FASTER)
                return _q(2, "faster", "Which solid spins faster?", gt, o, a, True,
                          {"min_fps": ceil1(max(c.rot_all))})
            out.append(("faster", 0, mk))

        def mk(rng):
            gt = DIRPAIR[(lr(c.B[0]["motion"][0]["w0"]), lr(c.B[1]["motion"][0]["w0"]))]
            o, a = _opts(rng, gt, list(DIRPAIR.values()))
            return _q(2, "dir_pair", "Which way do the two solids spin?", gt, o, a, True,
                      {"min_fps": ceil1(max(c.rot_all))})
        out.append(("dir_pair", 0, mk))
        (V0, _, F0, _), (V1, _, F1, _) = c.counts
        if V0 != V1 and F0 != F1:
            def mk(rng, key=(0 if V0 > V1 else 1, 0 if F0 > F1 else 1)):
                gt = VFPAIR[key]
                o, a = _opts(rng, gt, list(VFPAIR.values()))
                return _q(2, "vf_pair", "Which solid has more corners, and which has more faces?", gt, o, a, False,
                          {"min_fps": ceil1(max(c.rot_all))})
            out.append(("vf_pair", 0, mk))
    return out


# ---- level 3: measurement --------------------------------------------------------------
def level3(c: Ctx):
    s, out = c.s, []
    for b, body in enumerate(c.B):
        V, E, F, _ = c.counts[b]
        for key, word, gt in (("n_faces", "faces", F), ("n_corners", "corners", V), ("n_edges", "edges", E)):
            def mk(rng, b=b, key=key, word=word, gt=gt, V=V, E=E, F=F):
                o, a = _num(rng, gt, [V, E, F], lo=1)
                return _q(3, key, f"How many {word} does {bn(s, b)} have?", gt, o, a, False,
                          {"b": b, "min_fps": c.fps(b)})
            out.append((key, b, mk))
        if c.plain(b):
            k, fr = sim.turns_value(sim.run_angle(sim.runs(body)[0]))
            if sim.frac_ok(fr):
                def mk(rng, b=b, k=k):
                    o, a = _num(rng, k)
                    return _q(3, "n_turns", f"How many complete turns does {bn(s, b)} make during the clip?", k, o, a,
                              True, {"b": b, "min_fps": c.fps(b)})
                out.append(("n_turns", b, mk))
            for sec in range(2, int((c.n - 1) / sim.FPS) + 1):
                f = sec * sim.FPS
                if f > c.n - 1 - sim.EDGE_F:
                    continue
                k, fr = sim.turns_value(sim.theta_y(body, f))
                if sim.frac_ok(fr) and k >= 1:
                    def mk(rng, b=b, k=k, sec=sec):
                        o, a = _num(rng, k)
                        text = f"How many complete turns has {bn(s, b)} made by the time the timer reads {sec}.0 s?"
                        return _q(3, "turns_by_time", text, k, o, a, True, {"b": b, "t": sec * 10,
                                                                          "min_fps": c.fps(b, 1.0)})
                    out.append(("turns_by_time", (b, sec), mk))
            seg = body["motion"][0]
            if len(body["motion"]) == 1 and seg["w0"] == seg["w1"]:
                P = round(360 / abs(seg["w0"]), 1)

                def mk(rng, b=b, P=P):
                    gt = f"{P:.1f} s"
                    cands = []
                    for fct in rng.sample([0.5, 0.67, 1.5, 2.0, 2.5], 5):
                        x = round(P * fct, 1)
                        if abs(x - P) >= max(0.6, 0.3 * P) and all(abs(x - y) >= 0.5 for y in cands) and \
                                0.5 <= x <= max(2 * c.n / sim.FPS, 1.6 * P):
                            cands.append(x)
                    o, a = _opts(rng, gt, [f"{x:.1f} s" for x in cands])
                    return _q(3, "period", f"About how long does {bn(s, b)} take to make one full turn?", gt, o, a,
                              True, {"b": b, "min_fps": c.fps(b)})
                out.append(("period", b, mk))
        stops, res = sim.stop_frames(body), sim.resume_frames(body)
        for k, f in enumerate(stops, 1):
            def mk(rng, b=b, k=k, f=f, many=len(stops) > 1):
                o, a, step = time_options(rng, sim.tenths(f), c.n)
                which = f" for the {ORD[k]} time" if many else ""
                text = f"At about what time does {bn(s, b)} stop turning{which}?"
                return _q(3, "stop_time", text, tstr(sim.tenths(f)), o, a, True,
                          {"b": b, "k": k, "min_fps": c.fps(b, 2 / step)})
            out.append(("stop_time", (b, k), mk))
        for k, f in enumerate(res, 1):
            def mk(rng, b=b, k=k, f=f, many=len(res) > 1):
                o, a, step = time_options(rng, sim.tenths(f), c.n)
                which = f"its {ORD[k]} stop" if many else "it stops"
                text = f"At about what time does {bn(s, b)} start turning again after {which}?"
                return _q(3, "resume_time", text, tstr(sim.tenths(f)), o, a, True,
                          {"b": b, "k": k, "min_fps": c.fps(b, 2 / step)})
            out.append(("resume_time", (b, k), mk))
        rr = sim.runs(body)
        if len(rr) >= 2 and rr[0][0]["axis"] == "y":
            k, fr = sim.turns_value(sim.run_angle(rr[0]))
            if sim.frac_ok(fr):
                def mk(rng, b=b, k=k):
                    o, a = _num(rng, k)
                    return _q(3, "turns_before_stop", f"How many complete turns does {bn(s, b)} make before it "
                              f"first stops?", k, o, a, True, {"b": b, "min_fps": c.fps(b, f1=sim.stop_frames(body)[0])})
                out.append(("turns_before_stop", b, mk))
    for m in range(len(s["marks"])):
        if not c.safe[m]:
            continue
        b, ps = c.body_of(m), c.passes[m]
        gap = min([ps[0], c.n - 1 - ps[-1]] + [y - x for x, y in zip(ps, ps[1:])]) if ps else c.n

        def mk(rng, m=m, b=b, k=len(ps), gap=gap):
            o, a = _num(rng, k)
            return _q(3, "count_passes", f"How many times does {mn(s, m)} {PASS}?", k, o, a, True,
                      {"m": m, "min_fps": c.fps(b, 2 / (gap / sim.FPS))})
        out.append(("count_passes", m, mk))
        if ps:
            def mk(rng, m=m, b=b, f=ps[0]):
                o, a, step = time_options(rng, sim.tenths(f), c.n)
                return _q(3, "first_pass_time", f"At about what time does {mn(s, m)} first {PASS}?",
                          tstr(sim.tenths(f)), o, a, True, {"m": m, "min_fps": c.fps(b, 2 / step)})
            out.append(("first_pass_time", m, mk))
    return out


# ---- level 4: integration ---------------------------------------------------------------
def similar_solids(name):
    V, E, F, _ = sim.SOLIDS[name]
    return sorted((o for o in sim.SOLIDS if o != name),
                  key=lambda o: (abs(sim.SOLIDS[o][0] - V) + abs(sim.SOLIDS[o][2] - F), o))


def level4(c: Ctx):
    s, out = c.s, []
    for b, body in enumerate(c.B):
        name = body["solid"]

        def mk(rng, b=b, name=name):
            o, a = _opts(rng, name, similar_solids(name)[:5])
            return _q(4, "name", f"Which solid is {bn(s, b)}?", name, o, a, False, {"b": b, "min_fps": c.fps(b)})
        out.append(("name", b, mk))
        if c.plain(b):
            k, fr = sim.turns_value(sim.run_angle(sim.runs(body)[0]))
            if 0.06 <= min(abs(fr - x) for x in (0.0, 0.25, 0.5, 0.75, 1.0)) and k >= 1:
                def mk(rng, b=b, fr=fr):
                    gt = extra_label(fr)
                    o, a = _opts(rng, gt, EXTRA)
                    text = (f"After the last complete turn that {bn(s, b)} finishes, how much further does it turn "
                            f"before the clip ends?")
                    return _q(4, "extra_turn", text, gt, o, a, True, {"b": b, "min_fps": c.fps(b)})
                out.append(("extra_turn", b, mk))
        rr = sim.runs(body)
        if len(rr) >= 2 and sim.y_only(body):
            ks = [sim.turns_value(sim.run_angle(r)) for r in rr]
            if all(sim.frac_ok(fr) for _, fr in ks):
                def mk(rng, b=b, gt=sum(k for k, _ in ks), parts=[k for k, _ in ks]):
                    o, a = _num(rng, gt, [gt + 1, max(0, gt - 1), parts[0]])
                    text = (f"Count the complete turns {bn(s, b)} makes in each stretch of turning between stops, "
                            f"then add them up. What is the total?")
                    return _q(4, "turns_total", text, gt, o, a, True,
                              {"b": b, "min_fps": c.fps(b, 2 / c.min_pause_s(b))})
                out.append(("turns_total", b, mk))
            net = sum(sim.run_angle(r) for r in rr)
            if abs(net) >= 90:
                def mk(rng, b=b, net=net):
                    gt = NETDIR[0] if net < 0 else NETDIR[1]
                    o, a = _opts(rng, gt, NETDIR)
                    text = f"Over the whole clip, which way has {bn(s, b)} turned further in total?"
                    return _q(4, "net_dir", text, gt, o, a, True, {"b": b, "min_fps": c.fps(b, 2 / c.min_pause_s(b))})
                out.append(("net_dir", b, mk))
        if len(rr) >= 2 and not sim.y_only(body):
            key = (rr[0][0]["axis"], rr[-1][-1]["axis"])
            if key in AXSEQ:
                def mk(rng, b=b, key=key):
                    gt = AXSEQ[key]
                    o, a = _opts(rng, gt, list(AXSEQ.values()))
                    return _q(4, "axis_seq", f"Which describes how {bn(s, b)} turns during the clip?", gt, o, a, True,
                              {"b": b, "min_fps": c.fps(b, 2 / c.min_pause_s(b))})
                out.append(("axis_seq", b, mk))
    for m, mk_ in enumerate(s["marks"]):
        b = mk_["body"]
        body = c.B[b]

        def mk(rng, m=m, b=b):
            gt = sim.degree(c.B[b]["solid"], s["marks"][m]["v"])
            o, a = _num(rng, gt, lo=2)
            return _q(4, "mark_degree", f"How many edges meet at the corner with {mn(s, m)}?", gt, o, a, False,
                      {"m": m, "min_fps": c.fps(b)})
        out.append(("mark_degree", m, mk))
        ps = c.passes[m]
        for k in (2, 3):
            if c.safe[m] and len(ps) >= k:
                def mk(rng, m=m, b=b, k=k, f=ps[k - 1]):
                    o, a, step = time_options(rng, sim.tenths(f), c.n)
                    return _q(4, "pass_time", f"At about what time does {mn(s, m)} {PASS} for the {ORD[k]} time?",
                              tstr(sim.tenths(f)), o, a, True, {"m": m, "k": k, "min_fps": c.fps(b, 2 / step)})
                out.append(("pass_time", (m, k), mk))
        if sim.y_only(body):
            for sec in range(1, int((c.n - 1) / sim.FPS)):
                if sec * sim.FPS < sim.EDGE_F or sec * sim.FPS > c.n - 1 - sim.EDGE_F:
                    continue
                fs = [f for f in range(c.n) if sim.tenths(f) == sec * 10]
                labs = {sim.side_at(s, m, f) for f in fs}
                if len(labs) != 1 or None in labs:
                    continue
                lo, hi = side_window(s, m, fs[0])
                lo2, hi2 = side_window(s, m, fs[-1])
                if min(fs[0] - lo, hi2 - fs[-1]) < 3 or hi2 - lo < 12:
                    continue

                def mk(rng, m=m, b=b, sec=sec, lab=labs.pop(), win=(hi2 - lo) / sim.FPS):
                    gt = QUAD[lab]
                    o, a = _opts(rng, gt, list(QUAD.values()))
                    text = f"Where is {mn(s, m)} when the timer reads {sec}.0 s?"
                    return _q(4, "side_at_time", text, gt, o, a, True,
                              {"m": m, "t": sec * 10, "min_fps": c.fps(b, 2 / win)})
                out.append(("side_at_time", (m, sec), mk))
        stops = sim.stop_frames(body)
        if stops and body["motion"][0]["axis"] == "y" and sim.pass_safe(s, m, f_end=stops[0]):
            ps = sim.passes(s, m, f_end=stops[0])
            gap = min([stops[0] - ps[-1]] + [y - x for x, y in zip(ps, ps[1:])]) if ps else stops[0]

            def mk(rng, m=m, b=b, k=len(ps), gap=max(gap, 12)):
                o, a = _num(rng, k)
                text = f"How many times does {mn(s, m)} {PASS} before {bn(s, b)} first stops?"
                return _q(4, "passes_before_stop", text, k, o, a, True,
                          {"m": m, "min_fps": c.fps(b, 2 / (gap / sim.FPS))})
            if gap >= 12:
                out.append(("passes_before_stop", m, mk))
        if stops and body["motion"][0]["axis"] == "y":
            f = stops[0]
            lab = sim.side_at(s, m, f)
            if lab is not None:
                def mk(rng, m=m, b=b, lab=lab):
                    gt = QUAD[lab]
                    o, a = _opts(rng, gt, list(QUAD.values()))
                    return _q(4, "mark_at_stop", f"Where is {mn(s, m)} when {bn(s, b)} first stops?", gt, o, a, True,
                              {"m": m, "min_fps": c.fps(b, 2 / c.min_pause_s(b))})
                out.append(("mark_at_stop", m, mk))
    if c.nb == 2:
        for key, word, i in (("faces", "faces", 2), ("corners", "corners", 0), ("edges", "edges", 1)):
            def mk(rng, word=word, i=i):
                a0, a1 = c.counts[0][i], c.counts[1][i]
                o, a = _num(rng, a0 + a1, [a0, a1, 2 * max(a0, a1)], lo=1)
                return _q(4, "pair_sum", f"How many {word} do the two solids have altogether?", a0 + a1, o, a, False,
                          {"what": word, "min_fps": ceil1(max(c.rot_all))})
            out.append(("pair_sum", key, mk))
        w = [abs(b["motion"][0]["w0"]) for b in c.B]
        ks = [sim.turns_value(sim.run_angle(sim.runs(b)[0])) for b in c.B]
        if max(w) >= 1.5 * min(w) and all(sim.frac_ok(fr) for _, fr in ks):
            fast, slow = (0, 1) if w[0] > w[1] else (1, 0)

            def mk(rng, gt=ks[fast][0] - ks[slow][0], kk=ks):
                o, a = _num(rng, gt, [kk[0][0], kk[1][0]])
                return _q(4, "turn_diff", "How many more complete turns does the faster solid make than the slower "
                          "one?", gt, o, a, True, {"min_fps": ceil1(max(c.rot_all))})
            out.append(("turn_diff", 0, mk))
    return out


# ---- level 5: counterfactual ------------------------------------------------------------
def level5(c: Ctx):
    s, out = c.s, []
    for b, body in enumerate(c.B):
        V, E, F, _ = c.counts[b]
        name = body["solid"]

        def mk(rng, b=b, V=V, E=E, F=F):
            o, a = _num(rng, F + V, [F, F + E, V + E, F + V + 2], lo=1)
            text = (f"Suppose every corner of {bn(s, b)} were sliced off flat (a small cut at each corner, each cut "
                    f"making a new face). How many faces would the new solid have?")
            return _q(5, "cf_truncate_faces", text, F + V, o, a, False, {"b": b, "min_fps": c.fps(b)})
        out.append(("cf_truncate_faces", b, mk))

        def mk(rng, b=b, V=V, E=E):
            o, a = _num(rng, 2 * E, [E, V + E, 2 * V, 3 * V], lo=1)
            text = (f"Suppose every corner of {bn(s, b)} were sliced off flat (a small cut at each corner). How many "
                    f"corners would the new solid have?")
            return _q(5, "cf_truncate_corners", text, 2 * E, o, a, False, {"b": b, "min_fps": c.fps(b)})
        out.append(("cf_truncate_corners", b, mk))

        def mk(rng, b=b, V=V, E=E, F=F):
            o, a = _num(rng, F, [V, E, F + 2, abs(F - 2)], lo=1)
            text = (f"Suppose a new solid were made by putting one corner at the centre of each face of {bn(s, b)} "
                    f"and joining the corners of faces that share an edge. How many corners would it have?")
            return _q(5, "cf_dual_corners", text, F, o, a, False, {"b": b, "min_fps": c.fps(b)})
        out.append(("cf_dual_corners", b, mk))
        if name in sim.FAMILY:
            fam, nn = sim.FAMILY[name]

            def mk(rng, b=b, fam=fam, nn=nn, E=E):
                gt = FAM_EDGES[fam] * (nn + 1)
                o, a = _num(rng, gt, [E, gt + 1, gt - 1, gt + 2], lo=1)
                text = FAM_TEXT[fam].replace("{X}", bn(s, b)) + " How many edges would it then have?"
                return _q(5, "cf_more_sides", text, gt, o, a, False, {"b": b, "min_fps": c.fps(b)})
            out.append(("cf_more_sides", b, mk))
        if c.plain(b):
            ang = sim.run_angle(sim.runs(body)[0])
            real = sim.turns_value(ang)[0]
            for fct, word in ((2, "twice as fast"), (0.5, "half as fast")):
                k, fr = sim.turns_value(ang * fct)
                if sim.frac_ok(fr):
                    def mk(rng, b=b, fct=fct, word=word, k=k, real=real):
                        o, a = _num(rng, k, [real])
                        text = (f"If {bn(s, b)} had turned {word} the whole time (same start, same direction), how "
                                f"many complete turns would it make during the clip?")
                        return _q(5, "cf_speed", text, k, o, a, True, {"b": b, "factor": fct, "min_fps": c.fps(b)})
                    out.append(("cf_speed", (b, fct), mk))
        rr = sim.runs(body)
        if len(rr) >= 2 and body["motion"][0]["axis"] == "y":
            alt = sim.no_stop_body(body, c.n)
            k, fr = sim.turns_value(sim.run_angle(sim.runs(alt)[0]))
            if sim.frac_ok(fr):
                def mk(rng, b=b, k=k):
                    o, a = _num(rng, k)
                    text = (f"If {bn(s, b)} had never stopped, and had kept turning about the upright axis at its "
                            f"starting speed and direction for the whole clip, how many complete turns would it make?")
                    return _q(5, "cf_no_stop", text, k, o, a, True, {"b": b, "min_fps": c.fps(b)})
                out.append(("cf_no_stop", b, mk))
        if len(rr) >= 2 and body["motion"][0]["axis"] == "y":
            for fct, word in ((2, "twice as fast"), (0.5, "half as fast")):
                k, fr = sim.turns_value(sim.run_angle(sim.runs(sim.scale_body(body, fct))[0]))
                if sim.frac_ok(fr):
                    def mk(rng, b=b, fct=fct, word=word, k=k):
                        o, a = _num(rng, k)
                        text = (f"If {bn(s, b)} had turned {word} until its first stop (stopping at the same moment), "
                                f"how many complete turns would it make before that stop?")
                        return _q(5, "cf_speed_first", text, k, o, a, True, {"b": b, "factor": fct,
                                                                           "min_fps": c.fps(b)})
                    out.append(("cf_speed_first", (b, fct), mk))
            for m in c.marks_of[b]:
                lab = sim.side_at(s, m, sim.stop_frames(body)[0], sim.scale_body(body, -1))
                if lab is not None:
                    def mk(rng, m=m, b=b, lab=lab):
                        gt = QUAD[lab]
                        o, a = _opts(rng, gt, list(QUAD.values()))
                        text = (f"If {bn(s, b)} had turned the other way at the same speed, where would {mn(s, m)} be "
                                f"when it first stops?")
                        return _q(5, "cf_stop_side", text, gt, o, a, True,
                                  {"m": m, "min_fps": c.fps(b, 2 / c.min_pause_s(b))})
                    out.append(("cf_stop_side", m, mk))
        if len(rr) >= 2 and not sim.y_only(body):
            k, fr = sim.turns_value(sum(abs(sim.run_angle(r)) for r in sim.runs(sim.upright_body(body))))
            if sim.frac_ok(fr):
                def mk(rng, b=b, k=k):
                    o, a = _num(rng, k)
                    text = (f"If, after its stop, {bn(s, b)} had gone back to turning about the upright axis in its "
                            f"starting direction (at its later speed, for the same time), how many complete turns "
                            f"would it make in the whole clip?")
                    return _q(5, "cf_no_switch", text, k, o, a, True, {"b": b, "min_fps": c.fps(b)})
                out.append(("cf_no_switch", b, mk))
        if c.nb == 2:
            o_ = 1 - b
            alt = sim.swap_speed_body(body, c.B[o_])
            k, fr = sim.turns_value(sim.run_angle(sim.runs(alt)[0]))
            real = sim.turns_value(sim.run_angle(sim.runs(body)[0]))[0]
            if sim.frac_ok(fr):
                def mk(rng, b=b, k=k, real=real):
                    o, a = _num(rng, k, [real])
                    text = (f"If {bn(s, b)} had turned at the same speed as the other solid (in its own direction), "
                            f"how many complete turns would it make during the clip?")
                    return _q(5, "cf_swap", text, k, o, a, True, {"b": b, "min_fps": ceil1(max(c.rot_all))})
                out.append(("cf_swap", b, mk))
    for m, mk_ in enumerate(s["marks"]):
        b = mk_["body"]
        body = c.B[b]
        if not sim.y_only(body):
            continue
        flipped = sim.scale_body(body, -1)
        lab = sim.side_at(s, m, c.n - 1, flipped)
        if lab is not None:
            lo, hi = side_window(s, m, c.n - 1, flipped)
            if hi - lo >= 12:
                def mk(rng, m=m, b=b, lab=lab):
                    gt = QUAD[lab]
                    o, a = _opts(rng, gt, list(QUAD.values()))
                    text = (f"If {bn(s, b)} had turned the other way at the same speeds (same start, same stops), "
                            f"where would {mn(s, m)} be at the very end?")
                    return _q(5, "cf_other_way", text, gt, o, a, True, {"m": m, "min_fps": c.fps(b)})
                out.append(("cf_other_way", m, mk))
        fast = sim.scale_body(body, 2)
        if sim.pass_safe(s, m, fast) and c.safe[m]:
            k = len(sim.passes(s, m, fast))

            def mk(rng, m=m, b=b, k=k):
                o, a = _num(rng, k, [len(c.passes[m])])
                text = (f"If {bn(s, b)} had turned twice as fast the whole time (same start, same stops), how many "
                        f"times would {mn(s, m)} {PASS}?")
                return _q(5, "cf_passes_double", text, k, o, a, True, {"m": m, "min_fps": c.fps(b)})
            out.append(("cf_passes_double", m, mk))
    return out


LEVELS = (level1, level2, level3, level4, level5)
STATIC_KEYS = {"more_fv", "face_kinds", "common_face", "vf_pair", "n_faces", "n_corners", "n_edges", "name",
               "mark_degree", "pair_sum", "cf_truncate_faces", "cf_truncate_corners", "cf_dual_corners",
               "cf_more_sides"}


def _supply_level(lv, cands) -> int:
    if lv == 1:
        return len(cands)
    motion = sum(1 for k, _, _ in cands if k not in STATIC_KEYS)
    return min(len(cands), motion + 1)


def supply(scene) -> list[int]:
    """Number of usable distinct candidate questions per level (sim.check needs >= 3)."""
    c = Ctx(scene)
    return [_supply_level(i, f(c)) for i, f in enumerate(LEVELS, 1)]


def _pick(rng, cands, n, lv):
    bykey = {}
    for key, ident, mk in cands:
        bykey.setdefault(key, []).append(mk)
    keys = list(bykey)
    rng.shuffle(keys)
    chosen, static = [], 0
    for _ in range(10):
        if len(chosen) == n:
            break
        for key in keys:
            if not bykey[key] or len(chosen) == n:
                continue
            if lv > 1 and key in STATIC_KEYS:
                if static >= 1:
                    continue
                static += 1
            chosen.append(bykey[key].pop(rng.randrange(len(bykey[key]))))
    if len(chosen) != n:
        raise ValueError(f"level {lv}: only {len(chosen)} of {n} questions")
    return chosen


def build(scene: dict, rng: random.Random) -> list[dict]:
    counts = L.ten_counts(rng)
    c = Ctx(scene)
    qs = []
    for lv, (fn, n) in enumerate(zip(LEVELS, counts), 1):
        for mk in _pick(rng, fn(c), n, lv):
            qs.append(mk(rng))
    return L.number(qs)
