"""Tests for turntable_voxels:  python -m pytest turntable_voxels/test_turntable_voxels.py"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import answers  # noqa: E402
import questions  # noqa: E402
import render  # noqa: E402
import sim  # noqa: E402
from common import ladder5 as L5  # noqa: E402
from common.video5 import build_sample, _roundtrip  # noqa: E402

SEEDS = range(30)


def _sample(seed):
    return _roundtrip(build_sample("turntable_voxels", sim, questions, seed))


# ------------------------------------------------------------ hand-built worlds
def _world(heights, legs, events=(), marked=(), elev=30.0, duration=None, scenario="spin"):
    gy, gx = len(heights), len(heights[0])
    sc = {"scenario": scenario, "layout": "random", "gx": gx, "gy": gy, "heights": heights, "elev": elev,
          "legs": [dict(lg) for lg in legs], "events": [dict(e) for e in events],
          "duration": duration or legs[-1]["t1"] + 1.2, "marked": [dict(m) for m in marked],
          "theme": {"base": "white", "base_rgb": [228, 228, 222], "bg": "cream", "bg_rgb": [238, 234, 224],
                    "dark": False, "bg_style": "plain", "disc": "slate", "disc_rgb": [82, 92, 110],
                    "marker_rgb": [250, 250, 250], "outline_w": 3, "grid_w": 3},
          "S": 90.0, "cx": 560.0, "cy": 420.0}
    return sc


SPIN = [{"t0": 1.05, "t1": 5.5, "a0": 0, "a1": 360}]


def test_ray_casting_by_hand():
    # column x=1: a 2-stack in front (y=0), a 1-stack behind it (y=1), a 3-stack at the back (y=2).
    # At 30 degrees: the 1-stack is fully hidden; of the back stack the top cube shows its top and front,
    # the middle cube shows the upper 58% of its front face (rows in front stop rays below z = 1.42 + f),
    # the bottom cube is hidden.
    h = [[0, 2, 0], [0, 1, 0], [0, 3, 0]]
    vis = answers._visible(h, 30.0)
    assert vis == {(1, 0, 0), (1, 0, 1), (1, 2, 1), (1, 2, 2)}
    sc = _world(h, SPIN)
    assert sim.visible(sc, 0.0, h) == vis           # the ID-buffer render agrees
    assert not sim.ambiguous(sc, 0.0, h)
    assert answers.answer("hidden_start", {}, sc) == 2
    assert answers.answer("visible_start", {}, sc) == 4


def test_quarter_turn_brings_the_left_side_to_the_front():
    h = [[3, 0, 0], [1, 0, 0], [2, 0, 0]]          # cubes only in the left column
    r = sim.rotate_heights(h, 1)
    assert r[0] == [2, 1, 3] and r[1] == [0, 0, 0]   # after +90 (CCW from above) that column is the front row
    assert answers._rot(h, 1) == r and answers._rot(h, 4) == h
    sc = _world(h, SPIN)
    # projected depth: the left column is nearest the camera at +90 degrees
    d_left = sim.project(sc, 90.0, (0.5, 1.5, 0.5))[2]
    d_right = sim.project(sc, 90.0, (2.5, 1.5, 0.5))[2]
    assert d_left < d_right
    # positive turn = near edge moves right on screen
    x0 = sim.project(sc, 0.0, (1.5, 0.0, 0.0))[0]
    x1 = sim.project(sc, 10.0, (1.5, 0.0, 0.0))[0]
    assert x1 > x0
    assert questions.FACING[sim.quarter(90)] == "the side that was on your left at the start"


def test_visibility_after_turning_maps_back_to_world_cells():
    h = [[1, 0, 0], [1, 0, 0], [3, 0, 0]]
    sc = _world(h, SPIN)
    for a in (0, 90, 180, 270, -90):
        assert sim.visible(sc, a, h) == answers._vis_world(sc, h, a), a
    # seen from the left side (+90) the column becomes a front row: everything in it is visible
    assert answers._vis_world(sc, h, 90) == {(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 2, 1), (0, 2, 2)}


def test_timing_and_turn_answers_from_the_frame_track():
    legs = [{"t0": 1.05, "t1": 3.9, "a0": 0, "a1": -180}, {"t0": 5.1, "t1": 7.0, "a0": -180, "a1": -450}]
    sc = _world([[1, 2, 1], [2, 0, 1], [1, 3, 2]], legs, scenario="stops", duration=8.4)
    assert answers.answer("t_start", {}, sc) == 1
    assert answers.answer("t_stop", {}, sc) == 4
    assert answers.answer("stop_len", {}, sc) == 1
    assert answers.answer("total_quarters", {}, sc) == 5
    assert answers.answer("turn_dirs", {}, sc) == "always to the left"
    assert answers.answer("first_leg", {}, sc) == "half a turn, with its near edge moving left"
    # -450 = -90 mod 360: the side that was on the right faces you; turning the other way, the left side
    assert answers.answer("facing_end", {}, sc) == questions.FACING[3]
    assert answers.answer("cf_other_dir", {}, sc) == questions.FACING[1]
    assert answers.answer("facing_stop", {}, sc) == questions.FACING[2]


def test_reverse_and_events():
    legs = [{"t0": 1.0, "t1": 5.04, "a0": 0, "a1": 360}, {"t0": 5.04, "t1": 7.0, "a0": 360, "a1": 180}]
    sc = _world([[1, 2, 1], [2, 0, 1], [1, 3, 2]], legs, scenario="reverse", duration=8.2)
    assert answers.answer("turn_dirs", {}, sc) == "first to the right, later to the left"
    assert answers.answer("t_reverse", {}, sc) == 5
    assert answers.answer("total_quarters", {}, sc) == 6
    ev = [{"kind": "drop", "x": 1, "y": 1, "z": 0, "t0": 2.35, "t1": 2.95},
          {"kind": "lift", "x": 1, "y": 2, "z": 2, "t0": 4.1, "t1": 4.7}]
    sc = _world([[1, 2, 1], [2, 0, 1], [1, 3, 2]], SPIN, events=ev, scenario="build")
    assert answers.answer("t_event", {"i": 0}, sc) == 3          # lands at 2.95
    assert answers.answer("t_event", {"i": 1}, sc) == 4          # starts to rise at 4.1
    assert answers.answer("total_end", {}, sc) == 13
    assert sim.end_heights(sc) == [[1, 2, 1], [2, 1, 1], [1, 2, 2]]
    assert answers.answer("cf_add_top", {}, sc) == 13 + 8


def test_marked_cube_answers():
    mk = [{"colour": "red", "x": 1, "y": 2, "z": 0}, {"colour": "yellow", "x": 1, "y": 2, "z": 2}]
    sc = _world([[0, 2, 0], [0, 1, 0], [1, 3, 1]], SPIN, marked=mk)
    assert answers.answer("marked_vis_top", {"c": "red"}, sc) == questions.VISTOP[3]
    assert answers.answer("marked_vis_top", {"c": "yellow"}, sc) == questions.VISTOP[0]
    assert answers.answer("above_marked", {"c": "red"}, sc) == 2
    assert answers.answer("layer_diff", {"a": "yellow", "b": "red"}, sc) == 2
    assert answers.answer("marked_set_start", {}, sc) == "only the yellow cube"
    assert answers.answer("cf_remove_fv", {"c": "red"}, sc) == "1, 2, 1"


# ------------------------------------------------------------ generated samples
def test_ladder_counts_uniqueness_min_fps_and_answers():
    levels = set()
    for seed in SEEDS:
        s = _sample(seed)
        qs = s["questions"]
        assert L5.ladder_errors(qs, "ten") == []
        assert [q["qid"] for q in qs] == [f"q{i:02d}" for i in range(1, 11)]
        lv = [q["difficulty_level"] for q in qs]
        assert lv == sorted(lv)
        levels.add(tuple(lv.count(k) for k in range(1, 6)))
        sigs = [(q["derivation_key"], json.dumps(q["params"], sort_keys=True)) for q in qs]
        assert len(set(sigs)) == len(sigs)
        for q in qs:
            assert 0 < q["params"]["min_fps"] <= 4
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4
            assert str(answers.answer(q["derivation_key"], q["params"], s["params"])) == str(q["groundtruth"])
        assert sim.check(s["params"]) == []
    assert len(levels) > 3


def test_scene_signatures_unique_and_scenarios_mixed():
    rows = [_sample(seed)["params"] for seed in SEEDS]
    assert len({sim.sig_hash(r) for r in rows}) == len(rows)
    assert {r["scenario"] for r in rows} == set(sim.SCENARIOS)


def test_tamper_is_flagged():
    flagged = 0
    for seed in range(5):
        s = _sample(seed)
        q = copy.deepcopy(s["questions"][seed])
        wrong = next(o for o in q["options"] if o != q["answer_text"])
        flagged += str(answers.answer(q["derivation_key"], q["params"], s["params"])) != str(wrong)
    assert flagged == 5


def test_determinism_across_processes():
    code = ("import sys, json; sys.path.insert(0, %r); sys.path.insert(0, %r);"
            "import sim, questions; from common.video5 import build_sample;"
            "print(json.dumps([build_sample('turntable_voxels', sim, questions, s) for s in (3, 17, 42)], "
            "sort_keys=True))") % (str(HERE), str(HERE.parent))
    outs = []
    for hs in ("1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                                   check=True).stdout)
    assert outs[0] == outs[1] and len(outs[0]) > 1000


def test_frames_deterministic_and_clock_is_the_only_text():
    sc = _sample(5)["params"]
    for t in (0.0, 2.5, sc["duration"] - 0.05):
        a = np.asarray(render.frame(sc, t))
        b = np.asarray(render.frame(sc, t))
        assert hashlib.sha256(a.tobytes()).hexdigest() == hashlib.sha256(b.tobytes()).hexdigest()
    # the clock box is the only place two frames with the same scene state differ (the clock reading)
    t = 0.2
    a = np.asarray(render.frame(sc, t)).astype(int)
    b = np.asarray(render.frame(sc, t + 0.3)).astype(int)            # still in the start view
    diff = np.abs(a - b).sum(axis=2) > 0
    ys, xs = np.nonzero(diff)
    x0, y0, x1, y1 = sim.TIMER_BOX
    assert len(ys) and xs.min() >= x0 and xs.max() <= x1 and ys.min() >= y0 and ys.max() <= y1


def test_cubes_large_and_still_views_unambiguous():
    for seed in SEEDS:
        sc = _sample(seed)["params"]
        assert sc["S"] >= 40
        for t0, _t1, a in sim.holds(sc):
            h = sim.heights_after(sc, t0 + 1e-6)
            assert not sim.ambiguous(sc, a, h)
            assert sim.visible(sc, a, h) == answers._vis_world(sc, h, a)
