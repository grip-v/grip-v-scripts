"""Tests for sandpile_cascade.  Run:  python -m pytest test_sandpile_cascade.py -q"""
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

DS = "sandpile_cascade"
SEEDS = range(0, 80)


def _sample(seed):
    return json.loads(json.dumps(build_sample(DS, sim, questions, seed)))


def test_same_seed_same_sample():
    for s in range(8):
        assert _sample(s) == _sample(s)


def test_determinism_across_processes_and_hash_seeds():
    code = ("import sys, json; sys.path.insert(0, %r); sys.path.insert(0, %r);"
            "from common.video5 import build_sample; import sim, questions;"
            "print(json.dumps([build_sample('%s', sim, questions, s) for s in range(20)], sort_keys=True, default=str))"
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
    for fn in (render._track, render._static, render.grain_sprite, sim._timeline, sim._geom):
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


def test_sizes_and_timing_limits():
    for s in SEEDS:
        sc = _sample(s)["params"]
        n = sc["n_frames"]
        assert n <= sim.MAX_FRAMES
        assert sim.geometry(sc)["g"] >= 20 and sc["style"]["outline"] >= 3
        for e in sim.play(sc)["events"]:
            assert sim.EDGE <= e["f"] <= n - 1 - sim.EDGE


def test_cells_named_uniquely():
    for s in range(30):
        sc = _sample(s)["params"]
        kinds, _ = sim.topology(sc)
        names = [sim.cell_name(sc, c) for c, k in enumerate(kinds) if k == "cell"]
        assert len(set(names)) == len(names)


def test_no_text_but_the_clock():
    for s in range(10):
        sc = _sample(s)["params"]
        assert sc["labels"] == [] and render.track(sc)["labels"] == []
    for k in (0, 37, 99):
        x0, y0, x1, y1 = font(sim.TIMER_FONT).getbbox(sim.clock_text(k))
        assert y1 - y0 >= 24                                           # readable at half size


def test_hand_checked_seed():
    # seed 2: a row of 5 trays, T = 2 (one grain to each side), start [0, 1, 1, 1, 0]; grains dropped on
    # trays 3, 4, 2 (0-based). Drop 1 -> [0,1,1,2,0]: tray 3 topples -> [0,1,2,0,1], tray 2 -> [0,2,0,1,1],
    # tray 1 -> [1,0,1,1,1]: 3 topples, none lost. Drop 2 -> [1,0,1,1,2]: tray 4 (one grain falls off the
    # right end) -> [1,0,1,2,0], tray 3 -> [1,0,2,0,1], tray 2 -> [1,1,0,1,1]: 3 topples, 1 lost.
    # Drop 3 -> [1,1,1,1,1]: nothing topples.
    sc = _sample(2)["params"]
    A = lambda k, p: answers.answer(k, p, sc)
    assert sc["scenario"] == "row" and sc["T"] == 2 and sc["init"] == [0, 1, 1, 1, 0]
    assert sim.cell_name(sc, 3) == "the second tray from the right"
    assert sim.cell_name(sc, 0) == "the first tray from the left"
    assert A("n_drops", {}) == 3 and A("total_topples", {}) == 6 and A("lost_total", {}) == 1
    assert A("cascade_size", {"i": 0}) == 3 and A("cascade_size", {"i": 2}) == 0
    assert A("cascade_lost", {"i": 1}) == 1
    assert A("first_topple_cell", {}) == "the second tray from the right"
    assert A("end_count", {"cell": ["L", 1]}) == 1 and A("end_total", {}) == 5
    assert A("land_time", {"i": 0}) == "1.1 s"                     # frame 28
    assert A("topple_time", {"cell": ["R", 2]}) == "1.4 s"         # frame 34
    assert A("cell_topples", {"cell": ["L", 3]}) == 2               # tray 2 topples in both cascades
    assert A("lost_side", {}) == "the right end"
    # without the second grain: [1,0,1,1,1] + tray 2 -> 3 topples down to the right end, one grain lost
    assert A("cf_skip_total", {"i": 1}) == 6 and A("cf_lost", {"i": 1}) == 1
    # one more grain on the leftmost tray at the end: [2,1,1,1,1] topples all the way along: 5 topples
    assert A("cf_extra_end", {"cell": ["L", 1]}) == 5


def test_decoder_reads_the_drawing_not_the_rule():
    for s in (1, 2, 5, 7):
        smp = _sample(s)
        sc = smp["params"]
        tr = render.track(sc)
        M = answers.Model(tr)
        assert M.scenario == sc["scenario"] and M.consistent and M.T == sc["T"]
        for q in smp["questions"]:
            assert str(answers._answer(q["derivation_key"], q["params"], M)) == str(q["groundtruth"])
        assert set(tr) == {"poses", "n_frames", "grain", "cells", "side_view", "links", "labels"}


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
