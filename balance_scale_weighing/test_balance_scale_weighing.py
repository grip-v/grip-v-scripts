"""Tests for balance_scale_weighing: determinism, ladder, uniqueness, closed loop, tamper, edge cases.

    cd balance_scale_weighing && python -m pytest -q test_balance_scale_weighing.py
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import answers  # noqa: E402
import questions  # noqa: E402
import render  # noqa: E402
import sim  # noqa: E402
from common import ladder5 as L5  # noqa: E402
from common.video5 import build_sample, run_generate, run_validate  # noqa: E402

DS = "balance_scale_weighing"
SEEDS = range(0, 150)


def _sample(seed):
    return json.loads(json.dumps(build_sample(DS, sim, questions, seed)))


def test_same_seed_same_json_across_processes():
    code = ("import json,sys; sys.path[:0]=[%r,%r]; import sim,questions; from common.video5 import build_sample; "
            "print(json.dumps([build_sample('x', sim, questions, s) for s in range(40)], sort_keys=True))"
            % (str(HERE), str(HERE.parent)))
    outs = []
    for hs in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                                   check=True).stdout)
    assert outs[0] == outs[1] and len(outs[0]) > 1000


def test_render_deterministic_frames():
    for seed in (1, 2):
        scene = _sample(seed)["params"]
        for f in (0, 30, scene["n_frames"] // 2, scene["n_frames"] - 1):
            a = np.asarray(render.frame(scene, f / 24)).tobytes()
            render._sprite.cache_clear()
            render._background.cache_clear()
            b = np.asarray(render.frame(copy.deepcopy(scene), f / 24)).tobytes()
            assert hashlib.sha1(a).digest() == hashlib.sha1(b).digest()


def test_ladder_counts_order_and_min_fps():
    for seed in SEEDS:
        qs = _sample(seed)["questions"]
        assert L5.ladder_errors(qs, "ten") == []
        for q in qs:
            assert 0 < q["params"]["min_fps"] <= 4
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4
            assert q["options"].count(q["answer_text"]) == 1


def test_scenes_unique_and_checked():
    sigs = set()
    for seed in SEEDS:
        s = _sample(seed)["params"]
        assert sim.check(s) == []
        assert s["n_frames"] <= 240
        sigs.add(json.dumps(s, sort_keys=True))
        looks = [(o["colour"], o["shape"], o["pattern"]) for o in s["objects"]]
        assert len(set(looks)) == len(looks)
    assert len(sigs) == len(SEEDS)


def test_independent_answers_match():
    for seed in SEEDS:
        s = _sample(seed)
        for q in s["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], s["params"])) == str(q["groundtruth"]), \
                (seed, q["qid"], q["derivation_key"])


def test_all_scenarios_present():
    got = {_sample(seed)["params"]["scenario"] for seed in range(60)}
    assert got == set(sim.SCENARIOS)


def test_objects_inside_frame():
    for seed in range(40):
        scene = _sample(seed)["params"]
        for f in range(0, scene["n_frames"], 3):
            lay = render.layout(scene, f)
            for st in lay["objs"].values():
                if st["state"] == "rest":
                    assert 20 <= st["x"] <= 1260 and 60 <= st["y"] <= 640
                    assert not (st["x"] - st["size"] < render.TIMER_BOX[2] and st["y"] > render.TIMER_BOX[1])


def test_resting_weights_never_overlap():
    for seed in range(60):
        scene = _sample(seed)["params"]
        objs = {o["id"]: o for o in scene["objects"]}
        for f in range(0, scene["n_frames"], 4):
            lay = render.layout(scene, f)
            rest = [(o, st) for o, st in lay["objs"].items() if st["state"] == "rest"]
            for i, (a, sa) in enumerate(rest):
                for b, sb in rest[i + 1:]:
                    half = (render.WIDTH[objs[a]["shape"]] * sa["size"] + render.WIDTH[objs[b]["shape"]] * sb["size"]) / 2
                    assert abs(sa["x"] - sb["x"]) >= half + 2 or abs(sa["y"] - sb["y"]) > 40, (seed, f, a, b)


def test_tamper(tmp_path):
    run_generate(DS, HERE, sim, questions, render, ["--n", "12", "--seed", "3", "--no-render",
                                                     "--output-dir", str(tmp_path)])
    run_validate(DS, HERE, sim, questions, answers, render, ["--dataset-dir", str(tmp_path), "--skip-video"])
    lines = (tmp_path / "annotations.jsonl").read_text().splitlines()
    for i in (0, 4, 9):
        s = json.loads(lines[i])
        q = s["questions"][5]
        wrong = next(o for o in q["options"] if o != q["answer_text"])
        q["groundtruth"], q["answer_text"] = wrong, wrong
        q["answer"] = "ABCD"[q["options"].index(wrong)]
        lines[i] = json.dumps(s)
    (tmp_path / "t.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(SystemExit):
        run_validate(DS, HERE, sim, questions, answers, render, ["--dataset-dir", str(tmp_path), "--skip-video",
                                                                 "--ann", "t.jsonl", "--report", "t.txt"])
    rep = (tmp_path / "t.txt").read_text()
    assert sum(1 for ln in rep.splitlines() if "independent answer" in ln) == 3


# --------------------------------------------------------------------------- hand-checked edge cases


def _hand_scene(scenario="drop"):
    """Left gets 3 dots at 1.0 s, right gets 5 at 2.0 s, left gets 2 at 3.0 s: left, right, level."""
    return {"scenario": scenario, "n_frames": 120, "duration": 5.0, "trans_f": 12, "fall_f": 8, "lift_f": 10,
            "deg_per_unit": 5.0, "max_deg": 14.0, "camera": {"cx": 640, "cy": 425, "sc": 1.0},
            "theme": {"bg_top": [236, 233, 224], "bg_bot": [214, 208, 196], "floor": [150, 118, 86],
                      "stand": [92, 88, 96], "beam": [70, 74, 84], "pan": [176, 180, 188], "bg_pattern": "plain",
                      "floor_pattern": "plain", "floor_y": 620, "stand_style": "pillar", "pan_style": "tray"},
            "objects": [{"id": 0, "colour": "red", "shape": "square", "pattern": "plain", "mass": 3, "size": 70,
                         "mystery": False},
                        {"id": 1, "colour": "blue", "shape": "circle", "pattern": "striped", "mass": 5, "size": 70,
                         "mystery": False},
                        {"id": 2, "colour": "green", "shape": "hexagon", "pattern": "ringed", "mass": 2, "size": 70,
                         "mystery": False}],
            "initial": [],
            "events": [{"kind": "drop", "obj": 0, "side": "left", "peg": None, "f0": 16, "f1": 24},
                       {"kind": "drop", "obj": 1, "side": "right", "peg": None, "f0": 40, "f1": 48},
                       {"kind": "drop", "obj": 2, "side": "left", "peg": None, "f0": 64, "f1": 72}]}


def test_hand_drop_scene():
    s = _hand_scene()
    assert sim.check(s) == []
    assert sim.merged_dirs(sim.timeline(s)) == ["level", "left", "right", "level"]
    A = answers.answer
    assert A("changes", {}, s) == 3
    assert A("level_count", {}, s) == 1
    assert A("end_dir", {}, s) == "the beam is level"
    assert A("event_time", {"obj": 1, "act": "arrive", "occ": 1}, s) == "2.0 s"
    assert A("gap", {"a": {"obj": 0, "act": "arrive", "occ": 1}, "b": {"obj": 2, "act": "arrive", "occ": 1}}, s) \
        == "2.0 s"
    assert A("first_turn", {"dir": "right"}, s) == "the blue circle"
    assert A("cf_remove", {"obj": 2}, s) == "the right side is lower"
    assert A("cf_other_side", {"obj": 0}, s) == "the right side is lower"
    assert A("cf_heavier", {"obj": 2, "delta": 1}, s) == "the left side is lower"
    assert A("cf_extra", {"side": "right", "mass": 1}, s) == "the right side is lower"
    assert A("cf_changes_remove", {"obj": 1}, s) == 1
    assert A("side_mass_end", {"side": "left"}, s) == 5
    assert A("mass_after", {"obj": 1, "act": "arrive", "occ": 1, "side": "left"}, s) == 3


def test_hand_lever_and_mystery():
    s = _hand_scene("lever")
    s["deg_per_unit"] = 3.0
    s["events"][0].update(peg=2)            # left 3x2 = 6
    s["events"][1].update(peg=1)            # right 5x1 = 5 -> still left lower
    s["events"][2].update(side="right", peg=4)   # right 5 + 8 = 13 vs 6 -> right
    assert answers.answer("dir_after", {"obj": 1, "act": "arrive", "occ": 1}, s) == "the left side is lower"
    assert answers.answer("end_dir", {}, s) == "the right side is lower"
    assert answers.answer("peg", {"obj": 2}, s) == "the 4th peg"
    assert answers.answer("torque_end", {"side": "right"}, s) == 13
    assert answers.answer("cf_shift_peg", {"obj": 2, "delta": -1}, s) == "the right side is lower"
    m = _hand_scene("mystery")
    m["objects"][2]["mystery"] = True        # green hexagon: 2 units, beam ends level -> mass 2
    assert answers.answer("mystery", {}, m) == 2
    m["objects"][2]["mass"] = 3              # no longer level anywhere with it on: not derivable
    assert answers.answer("mystery", {}, m) is None


def test_timer_matches_event_frames():
    for seed in range(40):
        s = _sample(seed)
        for ev in s["params"]["events"]:
            f = sim.moment(ev)
            assert sim.aligned(f)
            assert render.timer_text(f) == sim.display(f)
            assert abs(f / 24 - (f * 10 // 24) / 10) < 0.05
