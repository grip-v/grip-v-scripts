"""Tests for stacking_collapse.  Run:  python -m pytest test_stacking_collapse.py -q"""
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

DS = "stacking_collapse"
SEEDS = range(0, 60)


def _sample(seed):
    return json.loads(json.dumps(build_sample(DS, sim, questions, seed)))


def test_same_seed_same_sample():
    for s in range(8):
        assert _sample(s) == _sample(s)


def test_determinism_across_processes_and_hash_seeds():
    code = ("import sys, json; sys.path.insert(0, %r); sys.path.insert(0, %r);"
            "from common.video5 import build_sample; import sim, questions;"
            "print(json.dumps([build_sample('%s', sim, questions, s) for s in range(16)], sort_keys=True, default=str))"
            % (str(HERE), str(HERE.parent), DS))
    outs = []
    for hs in ("1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                   check=True).stdout)
    assert outs[0] == outs[1]


def test_render_deterministic():
    scene = _sample(3)["params"]
    idx = (0, scene["n_frames"] // 2, scene["n_frames"] - 1)
    h = [hashlib.sha1(np.asarray(render.frame_at(scene, i)).tobytes()).hexdigest() for i in idx]
    for fn in (render._track, render._static, render.sprite, render.plank_sprite, sim._timeline):
        fn.cache_clear()
    h2 = [hashlib.sha1(np.asarray(render.frame_at(scene, i)).tobytes()).hexdigest() for i in idx]
    assert h == h2


def test_ladder_counts_order_and_min_fps():
    for s in SEEDS:
        qs = _sample(s)["questions"]
        assert L5.ladder_errors(qs, "ten") == []
        for q in qs:
            assert q["params"]["min_fps"] in questions.FPS_STEPS and q["params"]["min_fps"] <= 4
            assert 2 <= len(q["options"]) <= 4 and len(set(q["options"])) == len(q["options"])


def test_independent_answers_agree_and_check_passes():
    for s in SEEDS:
        smp = _sample(s)
        assert sim.check(smp["params"]) == []
        assert answers._model(smp["params"]).consistent
        for q in smp["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], smp["params"])) == str(q["groundtruth"])


def test_uniqueness():
    sigs = {json.dumps(_sample(s)["params"], sort_keys=True) for s in SEEDS}
    assert len(sigs) == len(SEEDS)


def test_all_scenarios_appear():
    assert {_sample(s)["params"]["scenario"] for s in range(20)} == set(sim.SCENARIOS)


def test_blocks_distinct_big_enough_and_apart():
    for s in range(20):
        sc = _sample(s)["params"]
        looks = [(b["colour"], b["pattern"]) for b in sc["blocks"]]
        assert len(set(looks)) == len(looks)
        assert all(min(b["w"], b["h"]) * sc["S"] >= 20 for b in sc["blocks"])
        assert sim.geometry_errors(sc, sim.timeline(sc)) == []


def test_timing_limits():
    for s in SEEDS:
        sc = _sample(s)["params"]
        n = sc["n_frames"]
        assert n <= sim.MAX_FRAMES
        for e in sim.timeline(sc)["events"]:
            assert sim.EDGE <= e["f"] <= n - 1 - sim.EDGE
            if e["kind"] == "fall":
                assert e["gone"] <= n - 5


def test_rule_margins():
    # every compared centre of mass is a quarter square clear of the edges; touching layers overlap >= 1 square
    for s in SEEDS:
        sc = _sample(s)["params"]
        r = sim.play(sc)
        assert r["ok"] and r["worst"] >= sim.MARGIN


def test_no_text_but_the_clock():
    for s in range(10):
        sc = _sample(s)["params"]
        assert sc["labels"] == [] and render.track(sc)["labels"] == []
    for k in (0, 37, 99):
        x0, y0, x1, y1 = font(sim.TIMER_FONT).getbbox(sim.clock_text(k))
        assert y1 - y0 >= 24                                           # readable at half size


def test_hand_checked_seeds():
    # seed 6: tower on a 5-wide pedestal [-2.5, 2.5]; six drops, all to the left. After the sixth (the
    # striped yellow block, 3 x 1.25 at x = -6.5) the centre of mass of all six is -71.25 / 25.5 = -2.794,
    # outside the contact [-2.5, 0] of the bottom block with the pedestal: everything tips over to the left.
    sc = _sample(6)["params"]
    A = lambda k, p: answers.answer(k, p, sc)
    assert sc["scenario"] == "tower" and sc["bases"][0]["w"] == 5
    assert A("n_falls", {}) == 1 and A("total_fallen", {}) == 6
    assert A("fall_side", {"k": 0}) == "to the left"
    assert A("break_from", {"k": 0}) == "the pedestal"
    assert A("cause", {"k": 0}) == "the striped yellow block"
    assert A("end_count", {}) == 0
    assert A("land_time", {"x": "the plain purple block"}) == "1.4 s"          # 12 + 22 = frame 34
    assert A("land_time", {"x": "the striped yellow block"}) == "5.3 s"        # frame 129
    # without the yellow block: -46.875 / 21.75 = -2.155, inside [-2.5, 0] -> five blocks stand
    assert A("cf_skip_end", {"x": "the striped yellow block"}) == 5
    assert A("cf_skip_any", {"x": "the striped yellow block"}) == "no"
    # yellow one square to the right: -67.5 / 25.5 = -2.647, only 0.147 past the edge -> not clear-cut
    assert A("cf_shift", {"x": "the striped yellow block", "d": 1}) is None
    # seed 12: plank 10 wide on a 1-wide support [-0.5, 0.5]; five 1-wide dotted blocks; the plank stands
    sc = _sample(12)["params"]
    A = lambda k, p: answers.answer(k, p, sc)
    assert sc["scenario"] == "plank" and sc["plank"] == {"x": 0.0, "w": 10}
    assert A("plank_tips", {}) == "no" and A("end_count", {}) == 5
    # centre of mass at the end: (0 - 0.5 + 0.75 - 1.5 + 3.125 - 3.5) / 9.75 = -0.167
    assert A("cf_skip_any", {"x": "the dotted red block"}) == "no"      # 1.875 / 8.75 = 0.214
    assert A("cf_skip_any", {"x": "the dotted green block"}) is None    # -4.75 / 8.5 = -0.559: too close
    # without the pink block the green one lands third: (-0.5 + 0.75 + 3.125) / 8 = 0.422, 0.078 from the edge
    assert A("cf_skip_any", {"x": "the dotted pink block"}) is None
    assert A("cf_wide_base", {"base": 0}) == "no"
    assert A("plank_land_side", {"x": "the dotted red block"}) == "left of the support"
    assert A("plank_land_side", {"x": "the dotted green block"}) == "right of the support"
    assert A("land_time", {"x": "the dotted red block"}) == "4.5 s"            # frame 110
    assert A("width", {"x": "the dotted blue block"}) == 1


def test_decoder_reads_the_drawing_not_the_actions():
    # answers.py reads render.track only: blank out the actions' bookkeeping it must not need
    for s in (1, 5, 6, 12):
        smp = _sample(s)
        sc = smp["params"]
        tr = render.track(sc)
        M = answers.Model(tr)
        assert M.scenario == sc["scenario"] and M.consistent
        for q in smp["questions"]:
            assert str(answers._answer(q["derivation_key"], q["params"], M)) == str(q["groundtruth"])
        assert set(tr) == {"blocks", "poly", "plank", "n_frames", "unit", "origin", "pillars", "labels"}


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
