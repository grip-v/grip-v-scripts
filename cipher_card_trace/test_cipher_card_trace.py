"""Tests for cipher_card_trace.  Run:  python -m pytest test_cipher_card_trace.py -q"""
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
from common import ladder5 as L5  # noqa: E402
from common.video5 import build_sample, font, run_generate, run_validate  # noqa: E402
import sim, questions, answers, render  # noqa: E402

DS = "cipher_card_trace"
SEEDS = range(0, 200)


def _sample(seed):
    return json.loads(json.dumps(build_sample(DS, sim, questions, seed)))


def test_same_seed_same_sample():
    for s in range(20):
        assert _sample(s) == _sample(s)


def test_determinism_across_processes_and_hash_seeds():
    code = ("import sys, json; sys.path.insert(0, %r); sys.path.insert(0, %r);"
            "from common.video5 import build_sample; import sim, questions;"
            "print(json.dumps([build_sample('%s', sim, questions, s) for s in range(40)], sort_keys=True))"
            % (str(HERE), str(HERE.parent), DS))
    outs = []
    for hs in ("1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                   check=True).stdout)
    assert outs[0] == outs[1]


def test_render_deterministic():
    scene = _sample(3)["params"]
    idx = (0, 40, scene["deal"][2]["land"] + 3, scene["n_frames"] - 1)
    h = [hashlib.sha1(np.asarray(render.frame_at(scene, i)).tobytes()).hexdigest() for i in idx]
    render._static.cache_clear()
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


def test_uniqueness_of_scenes_and_cards():
    sigs = set()
    for s in SEEDS:
        sc = _sample(s)["params"]
        sigs.add(json.dumps(sc, sort_keys=True))
        names = [sim.card_name(sc["deal"][i]) for i in sim.cards(sc)]
        assert len(set(names)) == len(names)
    assert len(sigs) == len(SEEDS)


def test_all_scenarios_appear_and_follow_their_rules():
    seen = set()
    for s in SEEDS:
        sc = _sample(s)["params"]
        seen.add(sc["scenario"])
        assert sim.scenario_errors(sc) == []
    assert seen == set(sim.SCENARIOS)


def test_timing_limits():
    for s in SEEDS:
        sc = _sample(s)["params"]
        assert sc["n_frames"] <= 240
        assert sc["deal"][0]["s"] >= 12
        assert sc["n_frames"] - 1 - sim.end_frame(sc["deal"][-1]) >= 12


def test_labels_match_renderer_and_are_big_enough():
    sc = _sample(5)["params"]
    img = np.asarray(render.frame_at(sc, 0), dtype=int)
    rot = render.ring_rotation(sc, 0)
    g = sc["geometry"]
    for lb in sc["labels"]:
        x0, y0, x1, y1 = font(lb["size"]).getbbox(lb["text"])
        assert y1 - y0 >= 24
        a = np.radians(-90 + lb["index"] * sim.STEP_DEG + rot[lb["ring"]])
        x, y = int(g["cx"] + lb["r"] * np.cos(a)), int(g["cy"] + lb["r"] * np.sin(a))
        crop = img[y - 10:y + 10, x - 10:x + 10]
        assert (crop.sum(axis=2) < 330).sum() > 20          # ink is drawn there
    assert sim.label_errors(sc) == []
    x0, y0, x1, y1 = font(sim.TIMER_FONT).getbbox(sim.clock_text(37))
    assert y1 - y0 >= 24


def test_clock_hand_checked():
    assert sim.timer_tenths(23) == 9 and sim.timer_tenths(24) == 10 and sim.timer_tenths(239) == 99
    assert sim.frames_showing(10) == (24, 26)
    assert sim.clock_text(43) == "4.3 s"


def test_decode_hand_checked():
    # inner ring A..K red then black, inner offset 2, letter ring offset 0: red 3 (index 2) sits under E;
    # a clockwise inner turn of 3 moves it under H.
    ring = [f"red:{r}" for r in sim.RANKS] + [f"black:{r}" for r in sim.RANKS]
    sc = {"ring": ring, "start": {"inner": 2, "outer": 0}, "moving": "inner",
          "deal": [{"kind": "card", "color": "red", "rank": "3", "suit": "hearts", "turn": 0},
                   {"kind": "joker", "turn": 3},
                   {"kind": "card", "color": "red", "rank": "3", "suit": "diamonds", "turn": 0}]}
    assert sim.decode(sc) == "EH"
    assert sim.decode(sc, [0]) == "EE"
    sc["moving"] = "outer"           # the letter ring turning clockwise moves letters away: E -> B
    assert sim.decode(sc) == "EB"


def test_edge_seed_hand_checked():
    # seed 3 (both_ways, checked by eye on the rendered clip): code RELTPU; the first joker turns
    # the inner ring 5 positions clockwise, the second 5 counter-clockwise, so the ring ends where it began.
    smp = _sample(3)
    sc = smp["params"]
    assert sc["scenario"] == "both_ways" and sc["code"] == "RELTPU"
    assert answers.answer("turn_size", {"k": 0}, sc) == "5 positions clockwise"
    assert answers.answer("turn_size", {"k": 1}, sc) == "5 positions counter-clockwise"
    assert answers.answer("net_turn", {}, sc) == "no net turn"
    assert answers.answer("code", {}, sc) == "RELTPU"
    assert answers.answer("n_cards", {}, sc) == 6


def test_frame_check_catches_a_wrong_code():
    import frame_check
    sc = _sample(4)["params"]
    frames = [np.asarray(render.frame_at(sc, i), dtype=np.int16) for i in range(sc["n_frames"])]
    assert frame_check.check_clip(sc, [], frames) == []
    bad = copy.deepcopy(sc)
    bad["code"] = bad["code"][::-1]
    assert any("letter read off the wheel" in e for e in frame_check.check_clip(bad, [], frames))


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
