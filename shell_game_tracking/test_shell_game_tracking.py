"""Tests for shell_game_tracking.  Run:  python -m pytest test_shell_game_tracking.py -q"""
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

DS = "shell_game_tracking"
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
    for fn in (render._track, render._static, render.cup_sprite, render.token_sprite, sim._poses):
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
        assert answers._model(smp["params"]).ok
        for q in smp["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], smp["params"])) == str(q["groundtruth"])


def test_uniqueness():
    sigs = {json.dumps(_sample(s)["params"], sort_keys=True) for s in SEEDS}
    assert len(sigs) == len(SEEDS)


def test_all_scenarios_appear():
    assert {_sample(s)["params"]["scenario"] for s in range(25)} == set(sim.SCENARIOS)


def test_sizes_timing_and_separation():
    for s in SEEDS:
        sc = _sample(s)["params"]
        n = sc["n_frames"]
        assert n <= sim.MAX_FRAMES and sc["cup"]["w"] >= 20 and sc["tok_size"] >= 20 and sc["style"]["outline"] >= 3
        for _, f in sim.events(sc):
            assert sim.EDGE <= f <= n - 1 - sim.EDGE
        assert all(m["dur"] >= 12 for m in sc["moves"])


def test_tokens_unique_and_never_shown_while_hidden():
    for s in range(30):
        sc = _sample(s)["params"]
        looks = [(t["colour"], t["shape"]) for t in sc["tokens"]]
        assert len(set(t["colour"] for t in sc["tokens"])) == len(looks)
        tr = render.track(sc)
        first = sc["moves"][0]["start"]
        for f in range(sc["n_frames"]):
            if f >= first:
                lifted = any(p["start"] <= f < p["start"] + p["up"] + p["hold"] + p["down"] for p in sc["peeks"])
                assert lifted or tr["tokens"][f] == []


def test_no_text_but_the_clock():
    for s in range(10):
        sc = _sample(s)["params"]
        assert sc["labels"] == [] and render.track(sc)["labels"] == []
    for k in (0, 37, 99):
        x0, y0, x1, y1 = font(sim.TIMER_FONT).getbbox(sim.clock_text(k))
        assert y1 - y0 >= 24                                           # readable at half size


def test_hand_checked_seeds():
    # seed 2: classic, 3 cups, the blue triangle under the leftmost cup. Swaps: middle<->right (frame 38),
    # left<->middle (frame 68), left<->middle (frame 91). Token: left, left, middle, left.
    sc = _sample(2)["params"]
    A = lambda k, p: answers.answer(k, p, sc)
    tok = {"tok": "the blue triangle"}
    assert sc["scenario"] == "classic" and sc["n"] == 3
    assert A("n_moves", {}) == 3 and A("n_cups", {}) == 3
    assert A("start_slot", tok) == "the leftmost position"
    assert A("final_slot", tok) == "the leftmost position"
    assert A("tok_moves", tok) == 2 and A("tok_visits", tok) == 2
    assert A("returned", tok) == "no"                     # back home only with the last swap
    assert A("in_move", dict(tok, k=0)) == "no" and A("in_move", dict(tok, k=1)) == "yes"
    assert A("first_swap", {}) == "the middle position and the rightmost position"
    assert A("move_time", {"k": 0}) == "1.5 s"            # frame 38
    assert A("first_move_time", tok) == "2.8 s"           # frame 69
    assert A("cf_skip", dict(tok, k=2)) == "the middle position"
    assert A("cf_skip", dict(tok, k=1)) == "the middle position"
    assert A("cf_start", {"slot": "the middle position"}) == "the rightmost position"
    assert A("cup_end", {"slot": "the middle position"}) == "the rightmost position"
    # seed 36: rotate, 4 cups, red triangle at slot 2. Moves: double swap 1<->3 and 2<->0 (triangle to 0);
    # turn 1->2, 2->3, 3->1 (triangle stays at 0); turn 0->2, 2->3, 3->0 (triangle to 2)
    sc = _sample(36)["params"]
    A = lambda k, p: answers.answer(k, p, sc)
    tok = {"tok": "the red triangle"}
    assert sc["scenario"] == "rotate" and sc["n"] == 4
    assert A("n_moves", {}) == 3
    assert A("start_slot", tok) == "the third position from the left"
    assert A("final_slot", tok) == "the third position from the left"
    assert A("tok_moves", tok) == 2 and A("tok_visits", tok) == 2
    assert A("in_move", dict(tok, k=1)) == "no"
    assert A("cf_skip", dict(tok, k=2)) == "the leftmost position"
    # without the double swap it stays at 2, the first turn takes it 2 -> 3, the second 3 -> 0
    assert A("cf_skip", dict(tok, k=0)) == "the leftmost position"


def test_decoder_reads_the_drawing_not_the_moves():
    for s in (1, 2, 5, 7, 9):
        smp = _sample(s)
        sc = smp["params"]
        tr = render.track(sc)
        M = answers.Model(tr)
        assert M.ok and len(M.moves) == len(sc["moves"]) and len(M.peeks) == len(sc["peeks"])
        for q in smp["questions"]:
            assert str(answers._answer(q["derivation_key"], q["params"], M)) == str(q["groundtruth"])
        assert set(tr) == {"cups", "tokens", "slots", "n_frames", "cup", "tok_size", "labels"}
        # cups come unordered on every frame
        assert all(c == sorted(c) for c in tr["cups"])


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
