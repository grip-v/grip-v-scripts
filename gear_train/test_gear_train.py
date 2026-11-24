"""Tests for gear_train:  python -m pytest gear_train/test_gear_train.py"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import random
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

SEEDS = range(40)


def _sample(seed):
    return _roundtrip(build_sample("gear_train", sim, questions, seed))


# ------------------------------------------------------------ hand-built worlds
def _gear(i, colour, z, x, y, parent=None, dir_=0, loose=False):
    return {"id": f"g{i}", "colour": colour, "rgb": list(sim.COLOURS[colour]), "z": z, "x": x, "y": y,
            "pattern": "plain", "pattern_n": 3, "loose": loose, "parent": parent, "dir": dir_, "phase": 0.0}


def _world(scenario, gears, K=20.0, sign=1, duration=8.0, program=None):
    sc = {"scenario": scenario, "m": 10.0, "K": K, "sign": sign, "driver": "g0", "w_driver": K / gears[0]["z"],
          "gears": gears, "duration": duration, "program": program or {},
          "theme": {"name": "cream", "bg": [238, 232, 216], "dark": False, "style": "plain", "texture_seed": 1,
                    "outline_w": 2}}
    for g in sc["gears"][1:]:                 # teeth phase along the tree, as sim does
        if g["parent"] is None:
            continue
        p = next(h for h in sc["gears"] if h["id"] == g["parent"])
        f = sim.phi((p["x"], p["y"]), (g["x"], g["y"]))
        up = (f - p["phase"]) * p["z"] / (2 * math.pi)
        g["phase"] = (f + math.pi - 2 * math.pi / g["z"] * (0.5 - up)) % (2 * math.pi)
    sc["log"] = sim.make_log(sc)
    return sc


def _chain3(scenario="steady", **kw):
    # driver 20 teeth (r 100) -> idler 10 teeth (r 50) -> 40 teeth (r 200), all in a row
    gs = [_gear(0, "red", 20, 300, 360, None, 1), _gear(1, "blue", 10, 450, 360, "g0", -1),
          _gear(2, "green", 40, 700, 360, "g1", 1)]
    return _world(scenario, gs, **kw)


def test_speed_ratio_through_an_idler():
    sc = _chain3()
    # |omega| * z = K for every gear: 1.0, 2.0, 0.5 rad/s
    assert [abs(sim.omega(sc, g, 1.0)) for g in ("g0", "g1", "g2")] == [1.0, 2.0, 0.5]
    assert answers.answer("ratio", {"a": "g0", "b": "g2"}, sc) == "1/2"
    assert answers.answer("ratio", {"a": "g2", "b": "g1"}, sc) == "4"
    assert answers.answer("start_dirs", {"a": "g0", "b": "g2"}, sc) == "both counter-clockwise"
    assert answers.answer("start_dirs", {"a": "g0", "b": "g1"}, sc) == "red counter-clockwise, blue clockwise"
    # 8 s at 0.5 rad/s = 4 rad = 0.64 turns; the driver makes 8 / (2 pi) = 1.27 turns
    assert answers.answer("turns", {"g": "g2", "t0": 0.0, "t1": 8.0}, sc) == 0
    assert answers.answer("turns", {"g": "g0", "t0": 0.0, "t1": 8.0}, sc) == 1
    assert sim.check(sc) == []


def test_counterfactuals_on_a_chain():
    sc = _chain3()
    assert answers.answer("remove", {"g": "g1"}, sc) == "green"
    assert answers.answer("flip_driver", {"a": "g1", "b": "g2"}, sc) == "blue counter-clockwise, green clockwise"
    # doubling the driver's teeth with the motor speed fixed doubles every other gear
    assert answers.answer("teeth_double", {"x": "g0", "y": "g2"}, sc) == questions.DOUBLE_OPTS[0]
    assert answers.answer("teeth_double", {"x": "g1", "y": "g2"}, sc) == questions.DOUBLE_OPTS[2]
    assert answers.answer("teeth_double", {"x": "g1", "y": "g1"}, sc) == questions.DOUBLE_OPTS[1]


def test_reversal_twice():
    prog = {"reversals": [{"mid": 2.0, "ramp": 0.6}, {"mid": 5.0, "ramp": 0.6}]}
    sc = _chain3("reversal", program=prog)
    assert answers.answer("reversal_count", {}, sc) == 2
    assert answers.answer("event_time", {"i": 1}, sc) == 5
    assert answers.answer("end_dirs", {"a": "g0", "b": "g1"}, sc) == "red counter-clockwise, blue clockwise"
    assert answers.answer("start_vs_end", {"g": "g1"}, sc).endswith("in the same direction")
    assert abs(sim.omega(sc, "g0", 2.0)) < 1e-9          # momentarily still at the turning point


def test_brake_stops_the_whole_train():
    prog = {"brake": {"gear": "g2", "angle": 0.0, "travel": 34.0, "t_touch": 2.0, "approach": 0.3,
                      "ramp": 0.35, "still0": 2.35, "release": 4.35}}
    sc = _chain3("brake", program=prog)
    for g in ("g0", "g1", "g2"):
        assert sim.omega(sc, g, 3.0) == 0.0
    assert answers.answer("still_count", {"T": 3}, sc) == 3
    assert answers.answer("stop_duration", {"g": "g0"}, sc) == 2
    assert answers.answer("keeps_turning", {"k": None}, sc) == questions.NONE_OF
    assert answers.answer("brake_other", {"g": "g1"}, sc) == 3
    assert answers.answer("event_time", {"i": 0}, sc) == 2


def test_disengage_stops_only_the_downstream_gears():
    K = 20.0
    period = 2 * math.pi / K
    still = 8 * period - 0.4                               # 8 whole teeth lost: 2.11 s still
    prog = {"disengage": {"gear": "g1", "parent": "g0", "shift": [46.0, 0.0], "moving": ["g1", "g2"],
                          "t_out": 2.0, "slide": 0.5, "ramp": 0.4, "still0": 2.9, "still1": 2.9 + still,
                          "teeth_lag": 8}}
    sc = _chain3("disengage", K=K, program=prog, duration=9.0)
    assert sim.check(sc) == []
    assert answers.answer("stopped_followers", {}, sc) == "green"
    assert answers.answer("still_count", {"T": 4}, sc) == 2
    assert answers.answer("slides_from", {}, sc) == "red"
    assert answers.answer("stop_duration", {"g": "g2"}, sc) == 2
    assert answers.answer("disengage_instead", {"g": "g2"}, sc) == "none"
    # teeth phase-lock again after the gear slides back in
    t = 8.5
    p, c = sc["gears"][0], sc["gears"][1]
    up = (0.0 - sim.theta(sc, "g0", t)) * p["z"] / (2 * math.pi)
    uc = (math.pi - sim.theta(sc, "g1", t)) * c["z"] / (2 * math.pi)
    x = (up + uc - 0.5) % 1.0
    assert min(x, 1 - x) < 1e-6


def test_teeth_counted_off_the_outline():
    for z in (10, 13, 18, 27, 36):
        assert render.count_teeth(z, 10.0) == z


# ---------------------------------------------------------------- generated
def test_ladder_counts_order_and_min_fps():
    for seed in SEEDS:
        s = _sample(seed)
        assert L5.ladder_errors(s["questions"], "ten") == []
        assert [q["qid"] for q in s["questions"]] == [f"q{i:02d}" for i in range(1, 11)]
        for q in s["questions"]:
            assert q["params"]["min_fps"] > 0
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4


def test_answers_independent_and_check_clean():
    for seed in SEEDS:
        s = _sample(seed)
        assert sim.check(s["params"]) == []
        for q in s["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], s["params"])) == str(q["groundtruth"])
            assert q["options"]["ABCD".index(q["answer"])] == q["answer_text"] == str(q["groundtruth"])


def test_uniqueness_of_scenes_and_names():
    sigs = set()
    for seed in range(200):
        sc = sim.build(random.Random(seed))
        sigs.add(sim.signature(sc))
        cols = [g["colour"] for g in sc["gears"]]
        assert len(set(cols)) == len(cols)
    assert len(sigs) == 200


def test_scene_timing_rules():
    for seed in SEEDS:
        sc = _sample(seed)["params"]
        assert sc["duration"] <= 10.0
        for a, b in sc["log"]["windows"]["events"]:
            assert a >= 1.5 and b <= sc["duration"] - 1.2


def test_tamper_is_caught():
    rows = [_sample(seed) for seed in range(10)]
    rng = random.Random(3)
    targets = rng.sample([(i, j) for i in range(10) for j in range(10)], 3)
    for i, j in targets:
        q = rows[i]["questions"][j]
        wrong = next(x for x in "ABCD" if x != q["answer"])
        q["groundtruth"] = q["options"]["ABCD".index(wrong)]
    flagged = sum(str(answers.answer(q["derivation_key"], q["params"], r["params"])) != str(q["groundtruth"])
                  for r in rows for q in r["questions"])
    assert flagged == 3


def test_answers_ignore_the_generator_log():
    s = _sample(11)
    sc = copy.deepcopy(s["params"])
    sc["log"] = {"poisoned": True}
    for q in s["questions"]:
        assert str(answers.answer(q["derivation_key"], q["params"], sc)) == str(q["groundtruth"])


def test_deterministic_across_processes():
    code = ("import sys, json; sys.path[:0] = [%r, %r]; import sim, questions; "
            "from common.video5 import build_sample; "
            "print(json.dumps([build_sample('gear_train', sim, questions, s) for s in range(12)], "
            "sort_keys=True))") % (str(HERE), str(HERE.parent))
    outs = []
    for hs in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                   check=True).stdout)
    assert outs[0] == outs[1]


def test_render_deterministic_and_clock_kept_clear():
    sc = _sample(5)["params"]
    h = []
    for _ in range(2):
        render._CACHE.clear()
        m = hashlib.sha256()
        for t in (0.0, 1.0, sc["duration"] / 2, sc["duration"] - 0.05):
            m.update(np.asarray(render.frame(sc, t)).tobytes())
        h.append(m.hexdigest())
    assert h[0] == h[1]
    for seed in SEEDS:
        sc = _sample(seed)["params"]
        for g in sc["gears"]:
            assert sim._box_gap(g["x"], g["y"], sim.tip_r(sc, g), sim.TIMER_BOX) >= sim.TIMER_KEEP


def test_markers_drawn_where_the_scene_says():
    import checks
    sc = _sample(7)["params"]
    for t in (0.0, 1.3, sc["duration"] / 2):
        img = np.asarray(render.frame(sc, t), dtype=np.int16)
        for g in sc["gears"]:
            a = checks.marker_angle(img, sc, g, t)
            assert a is not None
            assert abs(checks._adiff(a, sim.theta(sc, g["id"], t))) < math.radians(6)


def test_stale_state_between_seeds():
    a = json.dumps(_sample(17), sort_keys=True)
    for s in (3, 99, 4):
        _sample(s)
    assert json.dumps(_sample(17), sort_keys=True) == a


def test_check_rejects_a_stop_that_is_not_a_whole_tooth():
    K = 20.0
    still = 8 * 2 * math.pi / K - 0.4 + 0.05          # 0.05 s too long: the teeth would clash on re-entry
    prog = {"disengage": {"gear": "g1", "parent": "g0", "shift": [46.0, 0.0], "moving": ["g1", "g2"],
                          "t_out": 2.0, "slide": 0.5, "ramp": 0.4, "still0": 2.9, "still1": 2.9 + still,
                          "teeth_lag": 8}}
    errs = sim.check(_chain3("disengage", K=K, program=prog, duration=9.0))
    assert any("out of phase" in e for e in errs) and any("whole tooth" in e for e in errs)


def test_check_rejects_gears_that_look_meshed_but_are_not():
    sc = _chain3()
    sc["gears"].append(_gear(3, "pink", 12, 300, 360 - 100 - 60 - 15, None, 0, loose=True))   # tips 15 px apart
    sc["log"] = sim.make_log(sc)
    assert any("look meshed" in e for e in sim.check(sc))
