"""Tests for spinning_polyhedra:  python -m pytest spinning_polyhedra/test_spinning_polyhedra.py

determinism (in-process and across PYTHONHASHSEED values), frame determinism,
ladder counts, scene uniqueness, answer agreement, the tamper test, min_fps
on every question, the geometry table, "timer is the only text", and
hand-checked scenes with known answers.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import ladder5 as L5  # noqa: E402
from common.video5 import build_sample  # noqa: E402
import answers  # noqa: E402
import questions  # noqa: E402
import render  # noqa: E402
import sim  # noqa: E402

SEEDS = range(0, 150)


def _sample(seed):
    return json.loads(json.dumps(build_sample("spinning_polyhedra", sim, questions, seed)))


@pytest.fixture(scope="module")
def samples():
    return [_sample(s) for s in SEEDS]


def test_same_seed_same_json(samples):
    for s in samples[:30]:
        assert _sample(s["seed"]) == s


def test_determinism_across_processes():
    code = ("import sys, json, hashlib; sys.path[:0]=[%r, %r]; import sim, questions; "
            "from common.video5 import build_sample; "
            "print(hashlib.sha256(json.dumps([build_sample('spinning_polyhedra', sim, questions, s) "
            "for s in range(40)], sort_keys=True).encode()).hexdigest())") % (str(HERE), str(HERE.parent))
    outs = set()
    for hs in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                check=True).stdout.strip())
    assert len(outs) == 1


def test_frames_deterministic():
    sc = _sample(7)["params"]
    n = sc["n_frames"]
    h1 = [hashlib.sha1(render.frame_at(sc, f).tobytes()).hexdigest() for f in range(0, n, 9)]
    render._background.cache_clear()
    h2 = [hashlib.sha1(render.frame_at(sc, f).tobytes()).hexdigest() for f in range(0, n, 9)]
    assert h1 == h2


def test_ladder_and_min_fps(samples):
    for s in samples:
        assert L5.ladder_errors(s["questions"], "ten") == []
        assert all(q["params"]["min_fps"] > 0 for q in s["questions"])
        assert all(len(q["options"]) == 4 for q in s["questions"])
        for lv in (2, 3, 4, 5):
            static = [q for q in s["questions"] if q["difficulty_level"] == lv and not q["requires_motion"]]
            assert len(static) <= 1


def test_scenes_unique_and_checked(samples):
    sigs = set()
    for s in samples:
        sigs.add(sim.signature(s["params"]))
        assert sim.check(s["params"]) == []
        assert render.duration(s["params"]) <= 10.0
    assert len(sigs) == len(samples)
    assert {s["params"]["scenario"] for s in samples} == set(sim.SCENARIOS)


def test_answers_match(samples):
    for s in samples:
        for q in s["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], s["params"])) == str(q["groundtruth"]), q


def test_tamper_flags_exactly_three(samples):
    tampered = copy.deepcopy(samples[:20])
    for s in tampered[:3]:
        q = s["questions"][4]
        wrong = next(o for o in q["options"] if o != q["answer_text"])
        q["groundtruth"], q["answer_text"] = wrong, wrong
        q["answer"] = "ABCD"[q["options"].index(wrong)]
    flagged = sum(str(answers.answer(q["derivation_key"], q["params"], s["params"])) != str(q["groundtruth"])
                  for s in tampered for q in s["questions"])
    assert flagged == 3


def test_geometry_table():
    sigs = set()
    for name, (V, E, F, shapes) in sim.SOLIDS.items():
        bV, bE, bF, sizes = answers.counts(name)[:4]
        assert (bV, bE, bF) == (V, E, F), name
        assert sizes == tuple(sorted(k for k, c in shapes.items() for _ in range(c))), name
        assert answers.identify(name) == name
        sigs.add((V, E, F, sizes))
        assert answers.truncated_counts(name) == (2 * E, 3 * E, F + V), name
        assert answers.dual_corners(name) == F, name
        assert F != 2 * V                                   # the "twice as many faces" option is never true
    assert len(sigs) == len(sim.SOLIDS)
    assert answers.more_sides_edges("pentagonal prism") == 18
    assert answers.more_sides_edges("square pyramid") == 10
    assert answers.more_sides_edges("pentagonal bipyramid") == 18
    assert answers.more_sides_edges("hexagonal antiprism") == 28


def test_timer_is_the_only_text(samples):
    for s in samples[:6]:
        sc = s["params"]
        assert sc["labels"] == []
        f = sc["n_frames"] // 2
        img = np.asarray(render.frame_at(sc, f), dtype=int)
        bg = np.asarray(render._background(render.bg_key(sc)).resize((sim.W, sim.H)), dtype=int)
        yy, xx = np.mgrid[0:sim.H, 0:sim.W]
        out = np.ones((sim.H, sim.W), bool)
        for b, body in enumerate(sc["bodies"]):
            out &= np.hypot(xx - body["cx"], yy - body["cy"]) > sim.body_radius(sc, b)
        x0, y0, x1, y1 = sim.timer_box(sc)
        tb = np.zeros_like(out)
        tb[y0:y1, x0:x1] = True
        assert (np.abs(img - bg).sum(axis=2)[out & ~tb] > 30).mean() < 1e-4
        assert (np.abs(img - bg).sum(axis=2)[tb] > 100).sum() > 200     # the timer is there


# ---- hand-checked scenes ------------------------------------------------------------
def _scene(solid, motion, n, verts=(0,), scenario="steady"):
    sc = {"scenario": scenario, "n_frames": n, "theme": "dark", "bg_name": "ink", "bg": [18, 20, 28],
          "bg2": [30, 32, 40], "bg_style": "flat", "camera": {"elev": 15.0, "dist": 4.5},
          "timer": {"corner": "tl", "size": 40, "ink": [236, 238, 242]}, "line_style": "shade", "tint": 0.0,
          "corner_dots": False, "labels": [],
          "bodies": [{"solid": solid, "cx": 640.0, "cy": 380.0, "scale": 200.0, "tilt": [0.0, 0.0], "phase": 10.0,
                      "edge_name": "silver", "edge_rgb": [222, 228, 238], "edge_w": 4, "motion": motion}],
          "marks": [{"body": 0, "v": v, "colour": c, "rgb": rgb, "shape": "dot", "r": 13}
                    for v, (c, rgb) in zip(verts, [("red", [232, 36, 36]), ("blue", [36, 104, 255])])]}
    return sc


def _seg(f0, f1, axis, w0, w1=None):
    return {"f0": f0, "f1": f1, "axis": axis, "w0": w0, "w1": w0 if w1 is None else w1}


def A(key, sc, **p):
    return answers.answer(key, p, sc)


def test_steady_cube():
    # cube corner (1,1,1)/sqrt3 starts 45 deg round from the front; phase 10 -> azimuth 55 deg; 90 deg/s
    sc = _scene("cube", [_seg(0, 219, "y", 90.0)], 220, verts=(7,))
    assert sim.orbit(sc, 0)[1] == pytest.approx(55.0)
    assert sim.passes(sc, 0) == [82, 178] == answers.crossings(sc, 0)       # 55 + 90 t crosses 360, 720
    assert A("count_passes", sc, m=0) == 2
    assert A("first_pass_time", sc, m=0) == "3.4 s"                        # frame 82 -> 82*10//24 = 34
    assert A("n_turns", sc, b=0) == 2                                       # 90 * 219/24 = 821 deg
    assert A("period", sc, b=0) == "4.0 s"
    assert A("front_dir", sc, b=0, when="start") == "to the right"
    assert A("speed_trend", sc, b=0) == "keeping a steady speed"
    assert A("halves", sc, b=0) == "about the same in both halves"
    assert A("n_stops", sc, b=0) == 0
    assert A("extra_turn", sc, b=0) == questions.EXTRA[1]                  # 0.28 of a turn
    assert A("cf_speed", sc, b=0, factor=2) == 4
    assert A("cf_passes_double", sc, m=0) == 4
    assert (A("n_faces", sc, b=0), A("n_corners", sc, b=0), A("n_edges", sc, b=0)) == (6, 8, 12)
    assert A("cf_truncate_faces", sc, b=0) == 14 and A("cf_truncate_corners", sc, b=0) == 24
    assert A("cf_dual_corners", sc, b=0) == 6 and A("mark_degree", sc, m=0) == 3
    assert A("name", sc, b=0) == "cube"


def test_reverse_stop_and_restart():
    motion = [_seg(0, 60, "y", 120.0), _seg(60, 78, "y", 0.0), _seg(78, 200, "y", -60.0)]
    sc = _scene("square pyramid", motion, 201, scenario="reverse")
    assert sim.stop_frames(sc["bodies"][0]) == [60] and sim.resume_frames(sc["bodies"][0]) == [79]
    assert A("n_stops", sc, b=0) == 1
    assert A("stop_time", sc, b=0, k=1) == "2.5 s" and A("resume_time", sc, b=0, k=1) == "3.2 s"
    assert A("speed_compare", sc, b=0) == "faster before the stop"
    assert A("start_end_dir", sc, b=0) == "the opposite way"
    assert A("front_dir", sc, b=0, when="end") == "to the left"
    assert A("turns_before_stop", sc, b=0) == 0                             # 120 * 2.5 s = 300 deg
    assert A("cf_no_stop", sc, b=0) == 2                                    # 120 * 200/24 = 1000 deg = 2.8 turns
    assert A("cf_more_sides", sc, b=0) == 10


def test_axis_switch():
    motion = [_seg(0, 72, "y", 90.0), _seg(72, 90, "y", 0.0), _seg(90, 200, "x", 90.0)]
    sc = _scene("hexagonal prism", motion, 201, scenario="axis_switch")
    assert A("spin_axis", sc, b=0, when="start") == questions.AXIS["y"]
    assert A("spin_axis", sc, b=0, when="end") == questions.AXIS["x"]
    assert A("axis_seq", sc, b=0) == questions.AXSEQ[("y", "x")]
    assert A("front_dir", sc, b=0, when="end") == "downward"
    assert A("start_end_dir", sc, b=0) == "up or down instead of sideways"
    assert A("cf_no_switch", sc, b=0) == 1                                  # 90 * (72 + 110)/24 = 682 deg = 1.9 turns


def test_speed_change_profiles():
    sc = _scene("octahedron", [_seg(0, 200, "y", 30.0, 100.0)], 201, scenario="speed_change")
    assert A("speed_trend", sc, b=0) == "speeding up"
    assert A("halves", sc, b=0) == "further in the second half"
    sc = _scene("octahedron", [_seg(0, 100, "y", -30.0, -100.0), _seg(100, 200, "y", -100.0, -30.0)], 201)
    assert A("speed_trend", sc, b=0) == "speeding up, then slowing down"
    assert A("front_dir", sc, b=0, when="start") == "to the left"


def test_events_clear_of_ends(samples):
    for s in samples:
        sc = s["params"]
        n = sc["n_frames"]
        for q in s["questions"]:
            if q["answer_text"].endswith(" s") and q["derivation_key"] != "period":
                fs = [f for f in range(n) if sim.timer_text(f) == q["answer_text"]]
                assert fs and fs[0] >= sim.EDGE_F - 2 and fs[-1] <= n - 1 - sim.EDGE_F + 2
