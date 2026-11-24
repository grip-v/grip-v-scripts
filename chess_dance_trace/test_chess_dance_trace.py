"""Tests for chess_dance_trace.  Run:  python -m pytest test_chess_dance_trace.py -q"""
from __future__ import annotations
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
from common import ladder5 as L5  # noqa: E402
from common.video5 import build_sample, font, run_generate, run_validate  # noqa: E402
import sim, questions, answers, render  # noqa: E402

DS = "chess_dance_trace"
SEEDS = range(0, 150)


def _sample(seed):
    return json.loads(json.dumps(build_sample(DS, sim, questions, seed)))


def test_same_seed_same_sample():
    for s in range(20):
        assert _sample(s) == _sample(s)


def test_determinism_across_processes_and_hash_seeds():
    code = ("import sys, json; sys.path.insert(0, %r); sys.path.insert(0, %r);"
            "from common.video5 import build_sample; import sim, questions;"
            "print(json.dumps([build_sample('%s', sim, questions, s) for s in range(25)], sort_keys=True))"
            % (str(HERE), str(HERE.parent), DS))
    outs = []
    for hs in ("1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                   check=True).stdout)
    assert outs[0] == outs[1]


def test_render_deterministic():
    scene = _sample(3)["params"]
    idx = (0, 30, scene["n_frames"] - 1)
    h = [hashlib.sha1(np.asarray(render.frame_at(scene, i)).tobytes()).hexdigest() for i in idx]
    render._static.cache_clear()
    render.icon.cache_clear()
    render._artwork.cache_clear()
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


def test_uniqueness_of_scenes_and_pieces():
    sigs = set()
    for s in SEEDS:
        sc = _sample(s)["params"]
        sigs.add(json.dumps(sc, sort_keys=True))
        assert len({p["color"] for p in sc["pieces"]}) == len(sc["pieces"])
    assert len(sigs) == len(SEEDS)


def test_all_scenarios_appear():
    assert {_sample(s)["params"]["scenario"] for s in range(60)} == set(sim.SCENARIOS)


def test_timing_limits():
    for s in SEEDS:
        sc = _sample(s)["params"]
        assert sc["n_frames"] <= 240
        assert sim.round_frames(sc, 1)[0] >= 12
        assert sc["n_frames"] - sim.round_frames(sc, sc["rounds"])[1] >= 12


def test_labels_match_renderer_and_are_big_enough():
    sc = _sample(5)["params"]
    img = np.asarray(render.frame_at(sc, 0), dtype=int)
    bg = np.array(sc["style"]["bg"])
    for lb in sc["labels"]:
        assert lb["size"] == sim.LABEL_FONT
        x0, y0, x1, y1 = font(lb["size"]).getbbox(lb["text"])
        assert y1 - y0 >= 24
        x, y = int(lb["x"]), int(lb["y"])
        crop = img[y - 16:y + 16, x - 16:x + 16]
        assert (np.abs(crop - bg).sum(axis=2) > 200).sum() > 40      # ink is drawn there
    for k in range(10):
        x0, y0, x1, y1 = font(sim.TIMER_FONT).getbbox(sim.clock_text(k * 11))
        assert y1 - y0 >= 24
    assert sim.check(sc) == []


def test_clock_hand_checked():
    assert sim.timer_tenths(23) == 9 and sim.timer_tenths(24) == 10 and sim.timer_tenths(239) == 99
    assert sim.frames_showing(10) == (24, 26)
    assert sim.clock_text(43) == "4.3 s"


def test_geometry_hand_checked():
    # rook a1 on an empty 8x8 board reaches 14 squares; a piece on a4 stops it at a4
    assert len(sim.reach("R", (0, 0), [], 8)) == 14
    assert (0, 4) not in sim.reach("R", (0, 0), [(0, 3)], 8) and (0, 3) in sim.reach("R", (0, 0), [(0, 3)], 8)
    assert answers.targets("R", (0, 0), {(0, 3)}, 8) == sim.reach("R", (0, 0), [(0, 3)], 8)
    # a hole on c3 stops a bishop on a1 at b2 and cannot be landed on; a knight jumps
    assert sim.reach("B", (0, 0), [], 8, [(2, 2)]) == {(1, 1)}
    assert (2, 2) not in sim.reach("N", (0, 1), [], 8, [(2, 2)]) and (1, 3) in sim.reach("N", (0, 1), [], 8, [(2, 2)])
    # 6x6 board: knight b1 has 3 squares (a3, c3, d2)
    assert answers.targets("N", (1, 0), set(), 6) == {(0, 2), (2, 2), (3, 1)}
    assert sim.classify("R", (0, 0), (1, 1), set(), 8) == "pattern"
    assert sim.classify("R", (0, 0), (0, 5), {(0, 3)}, 8) == "blocked"


def test_edge_seed_hand_checked():
    # seed 7 (relay, 7x7, checked by eye on the rendered clip): the sky-blue king
    # makes the first move D7-D6 and lands at 1.4 s; the pink queen's first move is F2-B2.
    smp = _sample(7)
    sc = smp["params"]
    assert sc["scenario"] == "relay" and sc["N"] == 7
    assert sc["moves"][0] == [{"p": 2, "from": "D7", "to": "D6", "illegal": None, "capture": None}]
    assert answers.answer("land_time", {"x": 2, "k": 1}, sc) == "1.4 s"
    assert answers.answer("first_mover", {"x": 2, "options": [1, 2, 4, 5]}, sc) == "the sky-blue king"
    assert answers.answer("illegal_piece", {}, sc) == "no piece does"


def test_scenario_rules():
    for s in SEEDS:
        sc = _sample(s)["params"]
        caps = sum(1 for mv in sc["moves"] for m in mv if m["capture"] is not None)
        assert (sc["illegal"] is not None) == (sc["scenario"] == "rogue")
        assert (caps > 0) == (sc["scenario"] == "capture")
        assert bool(sc["holes"]) == (sc["scenario"] == "holes")
        if sc["scenario"] == "relay":
            assert all(len(mv) == 1 for mv in sc["moves"])


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
