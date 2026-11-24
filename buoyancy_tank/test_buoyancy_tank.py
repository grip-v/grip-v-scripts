"""Tests for buoyancy_tank: python -m pytest buoyancy_tank/test_buoyancy_tank.py"""
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import answers  # noqa: E402
import questions  # noqa: E402
import render  # noqa: E402
import sim  # noqa: E402
from common import ladder5 as L5  # noqa: E402
from common.video5 import build_sample  # noqa: E402

SEEDS = range(12)


def _sample(seed):
    return json.loads(json.dumps(build_sample("buoyancy_tank", sim, questions, seed)))


def test_deterministic_same_process():
    for seed in (0, 7):
        assert _sample(seed) == _sample(seed)


def test_deterministic_across_hash_seeds():
    code = ("import sys, json; sys.path[:0] = [%r, %r]; import sim, questions; "
            "from common.video5 import build_sample; "
            "print(json.dumps([build_sample('buoyancy_tank', sim, questions, s) for s in (3, 11)], sort_keys=True))"
            % (str(HERE), str(HERE.parent)))
    outs = []
    for hs in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                   check=True).stdout)
    assert outs[0] == outs[1]


def test_render_deterministic():
    sc = _sample(5)["params"]
    k = sc["n_frames"] // 2
    a = hashlib.sha1(render.frame(sc, k / 24).tobytes()).hexdigest()
    render._assets.cache_clear()
    sim._simulate.cache_clear()
    b = hashlib.sha1(render.frame(sc, k / 24).tobytes()).hexdigest()
    assert a == b


def test_ladder_min_fps_and_answers():
    for seed in SEEDS:
        s = _sample(seed)
        qs = s["questions"]
        assert L5.ladder_errors(qs, "ten") == []
        for q in qs:
            assert q["params"]["min_fps"] > 0
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4
            assert str(answers.answer(q["derivation_key"], q["params"], s["params"])) == str(q["groundtruth"])
        assert sim.check(s["params"]) == []
        assert s["params"]["n_frames"] <= 240


def test_objects_uniquely_named():
    for seed in SEEDS:
        objs = _sample(seed)["params"]["objects"]
        assert len({sim.describe(o) for o in objs}) == len(objs)
        assert len({o["colour"] for o in objs}) == len(objs)


def test_unique_scenes():
    sigs = {json.dumps(_sample(s)["params"], sort_keys=True) for s in SEEDS}
    assert len(sigs) == len(SEEDS)


def test_tamper_detected():
    s = _sample(2)
    flagged = 0
    for q in s["questions"][:3]:
        wrong = next(o for o in q["options"] if o != q["answer_text"])
        q["groundtruth"] = wrong
    for q in s["questions"]:
        if str(answers.answer(q["derivation_key"], q["params"], s["params"])) != str(q["groundtruth"]):
            flagged += 1
    assert flagged == 3


def test_geometry_hand_checked():
    sq = {"shape": "square", "w": 10.0, "h": 10.0}
    assert sim.sub_area(sq, 4.0) == 40.0
    ball = {"shape": "ball", "w": 10.0, "h": 10.0}
    assert abs(sim.area(ball) - math.pi * 25) < 1e-9
    assert abs(sim.sub_area(ball, 5.0) - math.pi * 25 / 2) < 1e-9
    tri = {"shape": "triangle", "w": 10.0, "h": 10.0}
    assert abs(sim.area(tri) - 50.0) < 1e-9
    hexa = {"shape": "hexagon", "w": 10.0, "h": 8.66}
    assert abs(sim.area(hexa) - answers.area(hexa)) < 1e-6
    # a half-density square floats half under; a ball at half density too (by symmetry)
    assert abs(sim.draft(dict(sq, rho=0.5)) - 5.0) < 1e-9
    assert abs(sim.draft(dict(ball, rho=0.5)) - 5.0) < 1e-9
    # one 10x10 square of density 0.25 in a 100 cm tank with 2000 cm^2 of water: level 20.25 cm
    L, ys = sim.equilibrium([dict(sq, rho=0.25)], 2000.0, 100.0)
    assert abs(L - 20.25) < 1e-9 and abs(ys[0] - 17.75) < 1e-9


def _hand_scene():
    """Drop scene: a 0.4-density 10x10 square, and a 2.0-density 8x8 square, 1000 cm tank area."""
    objs = [{"shape": "square", "w": 10.0, "h": 10.0, "rho": 0.4, "colour": "red", "pattern": "plain", "x": 20.0,
             "release": 14, "y0": 30.0},
            {"shape": "square", "w": 8.0, "h": 8.0, "rho": 2.0, "colour": "green", "pattern": "striped",
             "x": 60.0, "release": 40, "y0": 32.0}]
    sc = {"scenario": "drop", "tank": {"w": 100.0, "h": 50.0}, "flow": None, "objects": objs,
          "water0": 1500.0,
          "style": {"scale": 10.0, "x0": 100.0, "floor": 680, "bg": "#F3F0E8", "wall": "#3A3F4B",
                    "back": "#FBFBF8", "water": [70, 150, 225], "alpha": 55, "grid": 6, "dashed": False,
                    "wall_px": 6}}
    sf = sim.settle_frame(sc)
    sc["n_frames"] = max(sf, 40) + sim.HOLD_FRAMES + 1
    return sc


def test_hand_checked_scene():
    sc = _hand_scene()
    assert sim.check(sc) == []
    # final level: (1500 + 0.4*100 + 64) / 100 = 16.04 cm; red floats with 4 cm under, green on the floor
    L, ys = sim.final_state(sc)
    assert abs(L - 16.04) < 1e-9 and abs(ys[0] - 12.04) < 1e-9 and ys[1] == 0.0
    assert answers.answer("n_float_end", {}, sc) == 1
    assert answers.answer("n_floor_end", {}, sc) == 1
    assert answers.answer("end_state", {"obj": 1}, sc) == "resting on the floor, completely under water"
    # three times as heavy: 1.2 > 1, so it sinks; level (1500 + 100 + 64)/100 = 16.64 > 10, under water
    assert answers.answer("cf_mass", {"obj": 0, "factor": 3}, sc) == "resting on the floor, completely under water"
    # without the red square the level is (1500 + 64)/100 = 15.64 cm: nearest line 3 (18 cm)? 15.64/6 = 2.61 -> None
    assert answers.answer("cf_remove", {"obj": 0, "ask": "line"}, sc) is None
    assert answers.answer("cf_remove", {"obj": 0, "ask": "floor"}, sc) == 1
    # releases are at frames 14 and 40 -> clock 0.5 s and 1.6 s
    assert answers.answer("event_time", {"obj": 0, "event": "release"}, sc) == "0.5 s"
    assert answers.answer("event_time", {"obj": 1, "event": "release"}, sc) == "1.6 s"
    assert answers.answer("first_event", {"event": "enter", "objs": [0, 1]}, sc) == "the red square"
