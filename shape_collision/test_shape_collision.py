"""Tests for shape_collision.  Run:  python -m pytest test_shape_collision.py -q"""
from __future__ import annotations
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
from common.video5 import build_sample, font, run_generate, run_validate  # noqa: E402
import sim, questions, answers, render  # noqa: E402

DS = "shape_collision"
SEEDS = range(0, 120)


def _sample(seed):
    return json.loads(json.dumps(build_sample(DS, sim, questions, seed)))


def test_same_seed_same_sample():
    for s in range(10):
        assert _sample(s) == _sample(s)


def test_determinism_across_processes_and_hash_seeds():
    code = ("import sys, json; sys.path.insert(0, %r); sys.path.insert(0, %r);"
            "from common.video5 import build_sample; import sim, questions;"
            "print(json.dumps([build_sample('%s', sim, questions, s) for s in range(20)], sort_keys=True))"
            % (str(HERE), str(HERE.parent), DS))
    outs = []
    for hs in ("1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                   check=True).stdout)
    assert outs[0] == outs[1]


def test_render_deterministic():
    scene = _sample(3)["params"]
    idx = (0, 60, scene["n_frames"] - 1)
    h = [hashlib.sha1(np.asarray(render.frame_at(scene, i)).tobytes()).hexdigest() for i in idx]
    render._static.cache_clear()
    render.sprite.cache_clear()
    render._track.cache_clear()
    sim._simulate.cache_clear()
    h2 = [hashlib.sha1(np.asarray(render.frame_at(scene, i)).tobytes()).hexdigest() for i in idx]
    assert h == h2


def test_ladder_counts_order_and_min_fps():
    for s in SEEDS:
        qs = _sample(s)["questions"]
        assert L5.ladder_errors(qs, "ten") == []
        for q in qs:
            assert q["params"]["min_fps"] in questions.FPS_STEPS
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4


def test_independent_answers_agree_and_check_passes():
    for s in SEEDS:
        smp = _sample(s)
        assert sim.check(smp["params"]) == []
        for q in smp["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], smp["params"])) == str(q["groundtruth"])


def test_uniqueness_of_scenes_and_names():
    sigs = set()
    for s in SEEDS:
        sc = _sample(s)["params"]
        sigs.add(json.dumps(sc, sort_keys=True))
        assert len({b["color"] for b in sc["bodies"]}) == len(sc["bodies"])
    assert len(sigs) == len(SEEDS)


def test_all_scenarios_appear():
    assert {_sample(s)["params"]["scenario"] for s in range(40)} == set(sim.SCENARIOS)


def test_timing_limits():
    for s in SEEDS:
        sc = _sample(s)["params"]
        assert sim.MIN_FRAMES <= sc["n_frames"] <= 240
        tl = sim.t_last(sc)
        for t, kind, *_ in sim.shown(sc):
            assert t >= 0.5
            if kind in "ch":
                assert t <= tl - 0.5


def test_timer_is_only_text_and_big_enough():
    sc = _sample(5)["params"]
    assert sc["labels"] == []
    for k in (0, 37, 99):
        x0, y0, x1, y1 = font(sim.TIMER_FONT).getbbox(sim.clock_text(k))
        assert y1 - y0 >= 24
    L, T, R, B = sc["box"]
    assert sim.TIMER_BOX[3] < T - sc["style"]["wall_w"]


def test_clock_hand_checked():
    assert sim.tenths(23) == 9 and sim.tenths(24) == 10 and sim.tenths(239) == 99
    assert sim.frames_showing(10) == (24, 26)
    assert sim.frame_of(1.0) == 24 and sim.frame_of(1.01) == 25
    assert sim.clock_text(43) == "4.3 s"


def test_physics_hand_checked():
    # equal discs head-on exchange velocities; a light disc bounces off a heavy one
    a, b = sim.bounce((0, 0), (100, 0), 1, (60, 0), (0, 0), 1)
    assert a == pytest.approx([0, 0]) and b == pytest.approx([100, 0])
    a, b = sim.bounce((0, 0), (100, 0), 1, (60, 0), (0, 0), 3)
    assert a == pytest.approx([-50, 0]) and b == pytest.approx([50, 0])
    # first contact of two discs 100 px apart closing at 50 px/s with radii summing to 60
    assert sim._first_contact(100, 0, -50, 0, 60) == pytest.approx(0.8)
    assert sim._first_contact(100, 0, 50, 0, 60) == float("inf")


def test_rules_hand_checked():
    sc = {"rule": "swap", "tag": None}
    st = ["dots", "grid", "solid"]
    sim.apply_rule(sc, st, 0, 2)
    assert st == ["solid", "grid", "dots"]
    sc = {"rule": "infect", "tag": "checks"}
    st = ["checks", "solid", "solid"]
    sim.apply_rule(sc, st, 1, 2)
    assert st == ["checks", "solid", "solid"]
    sim.apply_rule(sc, st, 0, 2)
    assert st == ["checks", "solid", "checks"]


def test_edge_seed_hand_checked():
    # seed 5: a rails clip; checked by eye on the render: shapes never leave their rails
    sc = _sample(5)["params"]
    assert sc["scenario"] == "rails"
    assert all(b["vy"] == 0 and b["y"] in sc["lanes"] for b in sc["bodies"])
    D = answers._D(sc)
    assert [(i, j) for _, i, j in D["coll"]] == [(i, j) for _, i, j in sim.collisions(sc)]
    for (t1, *_), (t2, *_) in zip(D["coll"], sim.collisions(sc)):
        assert abs(t1 - t2) < 1e-6
    # every collision on a rail is between two shapes on the same rail
    assert all(sc["bodies"][i]["y"] == sc["bodies"][j]["y"] for _, i, j in sim.collisions(sc))


def test_holes_and_counterfactuals_agree():
    for s in SEEDS:
        sc = _sample(s)["params"]
        D = answers._D(sc)
        assert [(i, k) for _, i, k in D["caps"]] == [(i, k) for _, i, k in sim.captures(sc)]
        assert D["bad"] == []


@pytest.mark.filterwarnings("ignore::ResourceWarning")
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_tamper_detected(tmp_path):
    run_generate(DS, HERE, sim, questions, render, ["--n", "12", "--seed", "500", "--no-render",
                                                     "--output-dir", str(tmp_path)])
    rows = [json.loads(x) for x in (tmp_path / "annotations.jsonl").read_text().splitlines()]
    for k in (1, 5, 9):
        q = rows[k]["questions"][k % 10]
        wrong = next(o for o in q["options"] if o != q["answer_text"])
        q["groundtruth"], q["answer_text"] = wrong, wrong
        q["answer"] = "ABCD"[q["options"].index(wrong)]
    (tmp_path / "annotations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit):
        run_validate(DS, HERE, sim, questions, answers, render, ["--dataset-dir", str(tmp_path), "--skip-video"])
    rep = (tmp_path / "validation_report.txt").read_text()
    flagged = {ln.split(" (")[0] for ln in rep.splitlines() if "independent answer" in ln}
    assert len(flagged) == 3
