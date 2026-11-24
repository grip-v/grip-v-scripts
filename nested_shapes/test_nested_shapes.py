"""Tests for nested_shapes:  python -m pytest nested_shapes/test_nested_shapes.py

determinism (in-process, across PYTHONHASHSEED values, raw frames), ladder
counts, scene uniqueness, sim.check on every scene, answers agreement, the
tamper test, min_fps on every question (and that every-k-th-frame sampling at
that rate still gives the stored answer), the timer as the only text, and
hand-built scenes with known answers (turning, aliasing, appearance order,
marks, colours, wall hits).
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import ladder5 as L5  # noqa: E402
from common.video5 import build_sample  # noqa: E402
import answers  # noqa: E402
import checks  # noqa: E402
import questions  # noqa: E402
import render  # noqa: E402
import sim  # noqa: E402

SEEDS = range(0, 150)


def _sample(seed):
    return json.loads(json.dumps(build_sample("nested_shapes", sim, questions, seed)))


@pytest.fixture(scope="module")
def samples():
    return [_sample(s) for s in SEEDS]


def test_same_seed_same_json(samples):
    for s in samples[:30]:
        assert _sample(s["seed"]) == s


def test_determinism_across_processes():
    code = ("import sys, json, hashlib; sys.path[:0]=[%r, %r]; import sim, questions; "
            "from common.video5 import build_sample; "
            "print(hashlib.sha256(json.dumps([build_sample('nested_shapes', sim, questions, s) "
            "for s in range(60)], sort_keys=True).encode()).hexdigest())") % (str(HERE), str(HERE.parent))
    outs = set()
    for hs in ("0", "1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                check=True).stdout.strip())
    assert len(outs) == 1


def test_frames_deterministic(samples):
    for s in samples[:5]:
        sc = s["params"]
        h1 = [hashlib.sha1(render.frame_at(sc, f).tobytes()).hexdigest() for f in range(0, sc["n_frames"], 7)]
        render._BG_CACHE.clear()
        h2 = [hashlib.sha1(render.frame_at(json.loads(json.dumps(sc)), f).tobytes()).hexdigest()
              for f in range(0, sc["n_frames"], 7)]
        assert h1 == h2


def test_ladder_and_min_fps(samples):
    for s in samples:
        assert L5.ladder_errors(s["questions"], "ten") == []
        for q in s["questions"]:
            assert q["params"]["min_fps"] > 0 and len(q["options"]) == 4
            assert q["derivation_key"] and isinstance(q["params"], dict)


def test_scenes_unique_checked_and_short(samples):
    sigs = set()
    for s in samples:
        sigs.add(sim.signature(s["params"]))
        assert sim.check(s["params"]) == []
        assert render.duration(s["params"]) <= 10.0
        nf = s["params"]["n_frames"]
        for f, _ in sim.events(s["params"]):
            assert 12 <= f <= nf - 12                        # >= 0.5 s from both ends
    assert len(sigs) == len(samples)
    assert {s["params"]["scenario"] for s in samples} == set(sim.SCENARIOS)


def test_names_unique(samples):
    for s in samples:
        sc = s["params"]
        labs = [sim.label(sc, k) for k in range(len(sc["rings"]))]
        assert len(set(labs)) == len(labs)


def test_answers_match(samples):
    for s in samples:
        for q in s["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], s["params"])) == str(q["groundtruth"]), q


def test_tamper_flags_exactly_three(samples):
    tampered = copy.deepcopy(samples[:20])
    for s in tampered[:3]:
        q = s["questions"][6]
        wrong = next(o for o in q["options"] if o != q["answer_text"])
        q["groundtruth"], q["answer_text"] = wrong, wrong
        q["answer"] = "ABCD"[q["options"].index(wrong)]
    flagged = sum(str(answers.answer(q["derivation_key"], q["params"], s["params"])) != str(q["groundtruth"])
                  for s in tampered for q in s["questions"])
    assert flagged == 3


def test_frame_sampling_at_min_fps(samples):
    """Every k-th frame (k = floor(24 / min_fps), every phase) still gives the answer."""
    for s in samples[:80]:
        assert checks.sampling_failures(s) == []


def test_timer_is_the_only_text_and_clear_of_the_nest(samples):
    assert sim.AREA[1] > render.TIMER_PLATE[3]
    for s in samples[:40]:
        sc = s["params"]
        for f in range(0, sc["n_frames"], 12):
            x, y = sim.nest_pos(sc, f)
            assert y - sc["R"] >= sim.AREA[1] - 0.5
    assert sim.timer_text(0) == "0.0 s" and sim.timer_text(23) == "0.9 s" and sim.timer_text(24) == "1.0 s"
    assert sim.frame_of_reading(25) == 60 and sim.tenths(59) == 24
    # the timer plate is the only non-background, non-shape ink: blank scene -> only the plate differs
    sc = copy.deepcopy(samples[0]["params"])
    for r in sc["rings"]:
        r["appear"], r["vanish"] = 0, 0
    sc["marks"] = []
    import numpy as np
    a = np.asarray(render.frame_at(sc, 30), dtype=int)
    b = np.asarray(render.background(sc), dtype=int)
    diff = np.argwhere(np.abs(a - b).sum(axis=2) > 0)
    x0, y0, x1, y1 = render.TIMER_PLATE
    assert diff[:, 0].min() >= y0 and diff[:, 0].max() <= y1 and diff[:, 1].min() >= x0 and diff[:, 1].max() <= x1


# ---- hand-built scenes ----------------------------------------------------------
def _ring(shape, size, color, spin=((0, 0.0),), appear=0, vanish=None, recolor=(), tab=False, line="solid", w=5):
    r = {"shape": shape, "aspect": 1.0, "width": w, "line": line, "tab": tab, "dx": 0.0, "dy": 0.0, "rot0": 0.0,
         "size": size, "color": color, "appear": appear, "vanish": vanish, "recolor": [list(x) for x in recolor],
         "spin": [list(x) for x in spin]}
    if tab:
        r["tab_r"] = 8
    return r


def _scene(rings, scenario="spin", nf=200, marks=(), arena=None, nest=None):
    sc = {"scenario": scenario, "n_frames": nf, "style": {"theme": "light", "bg_name": "cream", "texture": "plain",
                                                          "tex_step": 40, "tex_amp": 6},
          "rings": rings, "marks": list(marks), "arena": arena, "R": 0.0, "order": None,
          "nest": nest or {"x": 640.0, "y": 400.0, "vx": 0.0, "vy": 0.0}}
    sc["R"] = math.ceil(max(sim.reach_out(r) + sim.pad(r) for r in rings) * 10) / 10
    return sc


def test_turning_patterns_and_counterfactuals():
    sc = _scene([_ring("square", 250, "red", spin=[(0, 60.0), (80, -60.0)]),
                 _ring("hexagon", 160, "blue", spin=[(0, -50.0)]),
                 _ring("pentagon", 100, "green", spin=[(0, 0.0), (120, 40.0)]),
                 _ring("circle", 50, "black")])
    lab = lambda k: sim.label(sc, k)  # noqa: E731
    assert answers.answer("spin_pattern", {"x": 0}, sc) == "turns clockwise, then turns counterclockwise"
    assert answers.answer("spin_pattern", {"x": 2}, sc) == "stays still, then turns clockwise"
    assert answers.answer("change_time", {"x": 0}, sc) == "3.3 s"            # frame 80
    assert answers.answer("n_changes", {"x": 0}, sc) == 1
    assert answers.answer("cw_end", {}, sc) == 1                             # only the pentagon
    assert answers.answer("cf_keep_dir", {}, sc) == 1                        # the square, kept clockwise
    assert answers.answer("cf_flip", {"x": 1}, sc) == 1                      # hexagon flipped -> cw like the pentagon
    assert answers.answer("never_turns", {"opts": [3, 0, 1, 2]}, sc) == lab(3)
    assert answers.answer("turning_at", {"t": 20}, sc) == 2                  # 2.0 s: square + hexagon
    assert answers.answer("pair_dir", {"x": 0, "y": 1, "t": 20}, sc) == "they turn opposite ways"
    assert answers.answer("inside_of", {"x": 1}, sc) == lab(2)
    assert answers.answer("inside_count", {"x": 0}, sc) == 3
    c = questions.Ctx(sc)
    assert c.chg[0] == [80] and c.chg[2] == [120]


def test_aliasing_flips_apparent_direction():
    # a hexagon turning 96 deg/s: 4 deg per frame, 32 deg per 8 frames > half its 60-degree period
    sc = _scene([_ring("square", 250, "red"), _ring("hexagon", 150, "blue", spin=[(0, 96.0)])])
    tr = answers.track(sc)
    assert answers.answer("spin_pattern", {"x": 1}, sc) == "turns clockwise the whole time"
    assert answers.answer("spin_pattern", {"x": 1}, sc, answers.subsample(tr, 8, 0)) == \
        "turns counterclockwise the whole time"
    # with a tab the same turn is unambiguous
    sc2 = _scene([_ring("square", 250, "red"), _ring("hexagon", 150, "blue", spin=[(0, 96.0)], tab=True)])
    tr2 = answers.track(sc2)
    assert answers.answer("spin_pattern", {"x": 1}, sc2, answers.subsample(tr2, 8, 0)) == \
        "turns clockwise the whole time"


def test_assemble_order_and_counts():
    rings = [_ring("octagon", 250, "red", appear=20), _ring("hexagon", 180, "blue", appear=50),
             _ring("square", 120, "green", appear=80), _ring("circle", 60, "black", appear=110)]
    sc = _scene(rings, scenario="assemble")
    assert answers.answer("appear_pattern", {}, sc) == questions.ORDERS["outside-in"]
    assert answers.answer("first_appear", {}, sc) == sim.label(sc, 0)
    assert answers.answer("count_at", {"t": 30}, sc) == 2                    # 3.0 s = frame 72
    assert answers.answer("count_at", {"t": 40}, sc) == 3                    # 4.0 s = frame 96
    assert answers.answer("appear_time", {"x": 2}, sc) == "3.3 s"
    assert answers.answer("cf_reverse_order", {"pos": 0}, sc) == sim.label(sc, 3)
    assert answers.answer("appear_after", {"x": 1}, sc) == 2
    assert answers.answer("last_depth", {}, sc) == "innermost outline"
    sc["rings"][1]["vanish"] = 150
    assert answers.answer("n_end", {}, sc) == 3
    assert answers.answer("max_on_screen", {}, sc) == 4
    assert answers.answer("cf_remove", {"x": 2, "y": 0}, sc) == sim.label(sc, 3)


def test_marks_attach_by_geometry():
    rings = [_ring("square", 250, "red", spin=[(0, 30.0)]), _ring("hexagon", 150, "blue"), _ring("circle", 70, "black")]
    for r in rings:
        r["mark_r"] = 14
    marks = [{"ring": 0, "u": 0.1, "kind": "heart", "color": "green", "add": 30, "remove": None},
             {"ring": 1, "u": 0.3, "kind": "star", "color": "orange", "add": 60, "remove": 120},
             {"ring": 0, "u": 0.6, "kind": "dot", "color": "blue", "add": 90, "remove": None},
             {"ring": 2, "u": 0.5, "kind": "diamond", "color": "pink", "add": 140, "remove": None}]
    sc = _scene(rings, scenario="marks", marks=marks)
    assert answers.answer("total_added", {}, sc) == 4
    assert answers.answer("total_removed", {}, sc) == 1
    assert answers.answer("first_mark", {}, sc) == sim.label(sc, 0)
    assert answers.answer("more_added", {"x": 0, "y": 1}, sc) == 1
    assert answers.answer("cf_no_removal", {"x": 1}, sc) == 1
    assert answers.answer("cf_stop_marks", {"t": 50}, sc) == 3               # 5.0 s = frame 120: heart, dot (+star gone)
    assert answers.answer("lost_mark", {"opts": [1, 0, 2]}, sc) == sim.label(sc, 1)
    assert answers.answer("mark_time", {"x": 2}, sc) == "5.8 s"


def test_colours_and_last_change():
    rings = [_ring("octagon", 250, "red", recolor=[(40, "blue"), (120, "green")]),
             _ring("square", 150, "blue"), _ring("triangle", 80, "black", recolor=[(160, "blue")])]
    sc = _scene(rings, scenario="recolor")
    assert answers.answer("color_sequence", {"x": 0}, sc) == "red, then blue, then green"
    assert answers.answer("color_count_end", {"c": "blue"}, sc) == 2
    assert answers.answer("cf_no_last_recolor", {"c": "blue"}, sc) == 1
    assert answers.answer("total_recolors", {}, sc) == 3
    assert answers.answer("color_at_time", {"x": 0, "t": 30}, sc) == "blue"
    assert answers.answer("recolor_time", {"x": 2}, sc) == "6.6 s"


def test_wall_hits_match_closed_form():
    sc = _scene([_ring("square", 80, "red"), _ring("circle", 40, "blue")], scenario="drift", nf=230,
                arena=[100, 120, 900, 680], nest={"x": 400.0, "y": 300.0, "vx": 7.3, "vy": -3.1})
    hs = sim.bounce_frames(sc)
    v = answers.View(sc, answers.track(sc))
    assert v.hits() == hs
    assert answers.answer("n_hits", {}, sc) == len(hs)
    first_w = hs[0][1]
    assert answers.answer("first_wall", {}, sc) == first_w
    for wall in sim.WALLS:
        cross = [t for t, w in sim.hits(sc) if w == wall]
        if cross:
            want = sum(1 for t, _ in sim.hits(sc, drop=wall) if t < cross[0])
            assert answers.answer("cf_no_wall", {"wall": wall}, sc) == want
