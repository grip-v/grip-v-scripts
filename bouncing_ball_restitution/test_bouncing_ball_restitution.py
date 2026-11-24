"""Tests for bouncing_ball_restitution.  Run:  python -m pytest test_bouncing_ball_restitution.py -q"""
from __future__ import annotations
import hashlib
import json
import math
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
from common import ladder5 as L5  # noqa: E402
from common.video5 import build_sample, run_generate, run_validate  # noqa: E402
import sim, questions, answers, render  # noqa: E402

DS = "bouncing_ball_restitution"
SEEDS = range(0, 120)


def _sample(seed):
    return json.loads(json.dumps(build_sample(DS, sim, questions, seed)))


def test_same_seed_same_sample():
    for s in range(20):
        assert _sample(s) == _sample(s)


def test_determinism_across_processes_and_hash_seeds():
    code = ("import sys, json; sys.path.insert(0, %r); sys.path.insert(0, %r);"
            "from common.video5 import build_sample; import sim, questions;"
            "print(json.dumps([build_sample('%s', sim, questions, s) for s in range(15)], sort_keys=True))"
            % (str(HERE), str(HERE.parent), DS))
    outs = []
    for hs in ("1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                   check=True).stdout)
    assert outs[0] == outs[1]


def test_render_deterministic():
    scene = _sample(3)["params"]
    h = [hashlib.sha1(np.asarray(render.frame(scene, t)).tobytes()).hexdigest() for t in (0.0, 1.25, 3.5)]
    render._background.cache_clear()
    render.sprite.cache_clear()
    h2 = [hashlib.sha1(np.asarray(render.frame(scene, t)).tobytes()).hexdigest() for t in (0.0, 1.25, 3.5)]
    assert h == h2


def test_ladder_counts_order_and_min_fps():
    for s in SEEDS:
        qs = _sample(s)["questions"]
        assert L5.ladder_errors(qs, "ten") == []
        assert len(qs) == 10
        for q in qs:
            assert q["params"]["min_fps"] in questions.FPS_STEPS
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4


def test_independent_answers_agree():
    for s in SEEDS:
        smp = _sample(s)
        assert sim.check(smp["params"]) == []
        for q in smp["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], smp["params"])) == str(q["groundtruth"])


def test_uniqueness_of_scenes_and_objects():
    sigs = set()
    for s in SEEDS:
        sc = _sample(s)["params"]
        sigs.add(json.dumps(sc, sort_keys=True))
        desc = [(o["color"], o["pattern"], o["shape"]) for o in sc["objects"]]
        assert len(set(desc)) == len(desc)
        assert len({o["color"] for o in sc["objects"]}) == len(desc)
    assert len(sigs) == len(SEEDS)


def test_all_scenarios_appear():
    assert {_sample(s)["params"]["scenario"] for s in range(40)} == set(sim.SCENARIOS)


def test_bounce_hand_checked():
    # h0=200 px, e=0.5, g=1000: first hit at sqrt(0.4) s, rebound apex 50 px,
    # second hit 2*316.23/1000 s later, next rebound apex 12.5 px < 30 -> rests.
    h = sim.bounce(200.0, 0.5, 1000.0, 0.0, 20.0)
    assert len(h["impacts"]) == 2
    assert math.isclose(h["impacts"][0], math.sqrt(0.4), rel_tol=1e-12)
    assert math.isclose(h["apexes"][0], 50.0, rel_tol=1e-9)
    assert math.isclose(h["impacts"][1], math.sqrt(0.4) + 2 * math.sqrt(2 * 1000 * 50) / 1000, rel_tol=1e-9)
    assert h["rest"] == h["impacts"][1]
    # the independent fixed-step integrator agrees
    r = answers.resim(200.0, 0.5, 1000.0, 0.0, 20.0)
    assert len(r) == 2 and abs(r[1] - h["impacts"][1]) < 1e-3


def test_wall_reflection_hand_checked():
    x_at, hits = sim.x_track(100.0, 100.0, 50.0, 150.0)
    assert [(round(t, 9), s) for t, s in hits(2.0)] == [(0.5, "right"), (1.5, "left")]
    assert math.isclose(x_at(1.0), 100.0) and math.isclose(x_at(0.5), 150.0)


def test_clock_hand_checked():
    assert sim.timer_tenths(23) == 9 and sim.timer_tenths(24) == 10 and sim.timer_tenths(239) == 99
    assert sim.clock_text(1.0) == "1.0 s" and sim.clock_text(2.36) == "2.3 s"
    assert sim.clean_time(2.35) and not sim.clean_time(2.399)


def test_seed_100_hand_checked():
    # checked by eye on the rendered clip: the cyan ball is clearly the smallest
    qs = _sample(100)["questions"]
    q = qs[0]
    assert q["derivation_key"] == "smallest" and q["answer_text"] == "the ringed cyan ball"


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
