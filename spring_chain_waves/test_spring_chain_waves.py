"""Tests for spring_chain_waves.  Run:  python -m pytest test_spring_chain_waves.py -q"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import sys
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import ladder5 as L5  # noqa: E402
from common.video5 import build_sample, font, run_generate, run_validate  # noqa: E402
import sim, questions, answers, render  # noqa: E402

DS = "spring_chain_waves"
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
    render._track.cache_clear()
    render._static.cache_clear()
    h2 = [hashlib.sha1(np.asarray(render.frame_at(scene, i)).tobytes()).hexdigest() for i in idx]
    assert h == h2


def test_ladder_counts_order_and_min_fps():
    for s in SEEDS:
        qs = _sample(s)["questions"]
        assert L5.ladder_errors(qs, "ten") == []
        for q in qs:
            assert q["params"]["min_fps"] in questions.FPS_STEPS
            assert 3 <= len(q["options"]) <= 4 and len(set(q["options"])) == len(q["options"])


def test_independent_answers_agree_and_check_passes():
    for s in SEEDS:
        smp = _sample(s)
        assert sim.check(smp["params"]) == []
        for q in smp["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], smp["params"])) == str(q["groundtruth"])


def test_uniqueness():
    sigs = {json.dumps(_sample(s)["params"], sort_keys=True) for s in SEEDS}
    assert len(sigs) == len(SEEDS)


def test_all_scenarios_appear():
    assert {_sample(s)["params"]["scenario"] for s in range(20)} == set(sim.SCENARIOS)


def test_timing_limits():
    for s in SEEDS:
        sc = _sample(s)["params"]
        assert sc["n_frames"] <= 240
        segs = sim.visible(sc, sim.trace(sc))
        tl = sim.t_last(sc)
        first = min(g["tb"] for g in segs if g["kind"] == "launch")
        S, tau = sim._support(sc)
        assert (first - S * tau) * sim.FPS >= sim.EDGE
        for g in segs:
            for t in (g["tb"], g["te"]):
                assert not tl - Fr(1, 2) < t < tl + sim.END_AFTER


def test_labels_match_renderer_and_readable():
    for s in range(10):
        sc = _sample(s)["params"]
        tr = render.track(sc)
        assert [lb["text"] for lb in sc["labels"]] == [str(i) for i in range(sc["L"] + 1)]
        assert tr["labels"] == sc["labels"]
        for lb in sc["labels"]:
            x0, y0, x1, y1 = font(lb["size"]).getbbox(lb["text"], anchor="mt")
            assert y1 - y0 >= 24                             # readable at half size
            assert abs(lb["x"] - (sc["x0"] + int(lb["text"]) * sc["scale"])) < 0.06   # under its tick
            assert lb["y"] + y1 <= sim.H - 8
            assert lb["y"] > sc["y0"] + sc["amp_px"] * sc["reach"][1]     # below the lowest dip
    for k in (0, 37, 99):
        x0, y0, x1, y1 = font(sim.TIMER_FONT).getbbox(sim.clock_text(k))
        assert y1 - y0 >= 24


def test_reflection_rules_hand_checked():
    # clamp / paddle: back upside down; ring: same way up; into slower beads: a third / half back inverted
    sc = {"L": 10, "c4": 12, "ratio": 2, "sections": [[0, 5, False], [5, 10, True]], "home": "left",
          "ends": {"left": "paddle", "right": "clamped"}, "launches": [{"f": 24, "end": "left", "sign": 1, "big": True}]}
    segs = sim.trace(sc, t_stop=Fr(8))
    assert [(g["kind"], g["amp"]) for g in segs[:3]] == [("launch", 1), ("split_back", Fr(-1, 3)),
                                                         ("split_on", Fr(2, 3))]
    assert segs[0]["te"] == Fr(1) + Fr(5, 3)                  # 5 marks at 3 marks/s from t = 1 s
    on = segs[2]
    assert on["c"] == Fr(3, 2) and on["te"] == segs[0]["te"] + Fr(10, 3)
    bounce = next(g for g in segs if g["parent"] == 2)
    assert bounce["amp"] == Fr(-2, 3) and bounce["via_far"]
    sc["ends"]["right"] = "free"
    assert next(g for g in sim.trace(sc, t_stop=Fr(8)) if g["parent"] == 2)["amp"] == Fr(2, 3)


def test_edge_seeds_hand_checked():
    # seed 6: reflect, 8 marks at 3.75 marks/s from the left paddle (launch frame 28), a ring on the right
    sc = _sample(6)["params"]
    assert sc["scenario"] == "reflect" and sc["ends"]["right"] == "free" and sc["c4"] == 15
    assert answers.answer("first_far_time", {}, sc) == "3.2 s"         # 28/24 + 8/3.75 = 3.30 s -> frame 79
    assert answers.answer("return_time", {}, sc) == "5.4 s"            # + 8/3.75 = 5.43 s -> frame 130
    assert answers.answer("first_back", {}, sc) == "the same way up"
    assert answers.answer("cf_switch_far", {}, sc) == "upside down"
    assert answers.answer("cf_slow_far", {}, sc) == "5.4 s"            # 8 / 1.875 + 1.17 s
    assert answers.answer("cf_shorter", {"m": 4}, sc) == "3.2 s"       # 2 * 4 / 3.75 + 1.17 s
    # seed 7: collide, 8 marks at 2.25 marks/s, left launch frame 23, right frame 44
    sc = _sample(7)["params"]
    assert sc["scenario"] == "collide"
    assert answers.answer("meet_mark", {}, sc) == "5"                  # meet at t = 3.17 s, x = 4.98
    assert answers.answer("second_meet", {}, sc) == "3"                # L - m
    assert answers.answer("cross_time", {"side": "right"}, sc) == "5.3 s"
    assert answers.answer("which_first_far", {}, sc) == "the pulse from the left paddle"
    assert answers.answer("cf_later", {"side": "left", "dt": "1"}, sc) == "4"   # x = 3.86
    # seed 1: junction, thick band 3-7 (a third of 3.25 marks/s), paddle on the right
    sc = _sample(1)["params"]
    assert sc["scenario"] == "junction" and sc["sections"] == [[0, 3, False], [3, 7, True], [7, 10, False]]
    assert answers.answer("refl_ratio", {}, sc) == "about 1/2 as tall"
    assert answers.answer("trans_taller", {}, sc) == "shorter"
    assert answers.answer("speed_ratio", {}, sc) == "about a third"
    assert answers.answer("junction_return", {}, sc) == "2.6 s"        # 20/24 + 2 * 3/3.25 = 2.68 s -> frame 64


def test_decoder_reads_the_drawing_not_the_scene():
    # answers.py must not depend on the stored pulse list: blank it and the answers stay the same
    smp = _sample(9)
    sc = smp["params"]
    tr = render.track(sc)
    M = answers.Model(tr)
    blank = dict(sc, segs=[], launches=[], sections=[], c4=0, ratio=0)
    with pytest.raises(Exception):
        answers.Model(render.track(blank))
    assert M.launches == [dict(g) for g in sc["launches"]]
    assert M.c == Fr(sc["c4"], 4)


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
