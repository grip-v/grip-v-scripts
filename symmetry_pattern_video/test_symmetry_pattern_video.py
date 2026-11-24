"""Tests for symmetry_pattern_video.  Run:  python -m pytest test_symmetry_pattern_video.py -q"""
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

DS = "symmetry_pattern_video"
SEEDS = range(0, 100)


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
    idx = (0, scene["n_frames"] // 2, scene["n_frames"] - 1)
    h = [hashlib.sha1(np.asarray(render.frame_at(scene, i)).tobytes()).hexdigest() for i in idx]
    render._track.cache_clear()
    render._background.cache_clear()
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


def test_uniqueness_and_group_names():
    sigs = set()
    for s in SEEDS:
        sc = _sample(s)["params"]
        sigs.add(json.dumps(sc, sort_keys=True))
        names = {(b["color"], b["type"]) for b in sc["base"]}
        assert len(names) == 1 + max(b["orbit"] for b in sc["base"])        # one name per group
    assert len(sigs) == len(SEEDS)


def test_all_scenarios_appear():
    assert {_sample(s)["params"]["scenario"] for s in range(30)} == set(sim.SCENARIOS)


def test_timing_limits():
    for s in SEEDS:
        sc = _sample(s)["params"]
        assert sc["n_frames"] <= 240
        ev = sim.event_frames(sc)
        assert ev[0] >= 12 and ev[-1] <= sc["n_frames"] - 13


def test_timer_is_only_text_and_big_enough():
    sc = _sample(4)["params"]
    assert sc["labels"] == []
    for k in (0, 37, 99):
        x0, y0, x1, y1 = font(sim.TIMER_FONT).getbbox(sim.clock_text(k))
        assert y1 - y0 >= 24


def test_group_algebra_hand_checked():
    # left-right mirror sends (x, y) to (-x, y); up-down to (x, -y); 90 deg turn (1, 0) -> (0, 1)
    M = sim.elem_matrix(sim.group_elements("mirror_left_right")[1])
    assert sim.apply(M, (3, 2)) == pytest.approx((-3, 2))
    M = sim.elem_matrix(sim.group_elements("mirror_up_down")[1])
    assert sim.apply(M, (3, 2)) == pytest.approx((3, -2))
    assert sim.apply(sim.elem_matrix((90.0, 0)), (1, 0)) == pytest.approx((0, 1))
    assert sim.mirror_axes("dihedral_4") == [0.0, 45.0, 90.0, 135.0]
    assert sim.octant(1, 1) == "top-right" and sim.octant(-1, 0) == "left"
    assert sim.clock_text(sim.tenths(25)) == "1.0 s" and sim.frames_showing(10) == (24, 26)


def test_point_along_equal_edges():
    # regression: the tip dot must go round shapes whose edges are all the same length
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert sim.point_along(sq, 0.5) == pytest.approx((10, 10))
    assert sim.point_along(sq, 0.875) == pytest.approx((0, 5))


def test_symmetry_measure_hand_checked():
    base = {"type": "kite", "color": "red", "fill": "solid", "dash": False, "size": 30.0, "cap": 1.0}
    shapes = [dict(base, x=100 * np.cos(np.radians(a)), y=100 * np.sin(np.radians(a)), rot=a + 30, mirrored=False)
              for a in (10, 130, 250)]
    assert sim.detect(shapes) == (3, 0)        # kites tilted off the radius: a pinwheel, no mirror lines
    radial = [dict(s, rot=s["rot"] - 30) for s in shapes]
    assert sim.detect(radial) == (3, 3)        # pointing straight out: three mirror lines
    D = [((s["color"],), [(x, y) for x, y in sim.outline(s)]) for s in shapes]
    assert answers.measure(D) == (3, 0)
    D = [((s["color"],), [(x, y) for x, y in sim.outline(s)]) for s in radial]
    assert answers.measure(D) == (3, 3)


def test_edge_seeds_hand_checked():
    # seed 0 (turns, left-right mirror pattern): flip, flip, 120 deg clockwise turn
    sc = _sample(0)["params"]
    assert sc["scenario"] == "turns" and sc["group"] == "mirror_left_right"
    assert answers.answer("kth_move", {"k": 1}, sc) == "a left-right flip"
    assert answers.answer("kth_move", {"k": 3}, sc) == "a clockwise turn"
    assert answers.answer("total_turn", {}, sc) == "120°"
    assert answers.answer("match_after", {}, sc) == "after the first move"
    assert answers.answer("n_match", {}, sc) == 2
    # seed 2 (waves, 3-fold, two waves starting on frames 20 and 58, 21-frame traces)
    sc = _sample(2)["params"]
    assert sc["scenario"] == "waves"
    g0 = [sc["base"][0]["color"], sc["base"][0]["type"]]
    assert answers.answer("wave_time", {"g": g0}, sc) == "0.8 s"
    assert answers.answer("done_time", {}, sc) == "3.2 s"
    assert answers.answer("cf_slow", {}, sc) == "5.0 s"
    assert answers.answer("cf_more_wave", {}, sc) == 9
    assert answers.answer("dir_count", {}, sc) == 6


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
