"""Tests for conveyor_belt_sort.  Run:  python -m pytest test_conveyor_belt_sort.py -q"""
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

DS = "conveyor_belt_sort"
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
    for fn in (render._track, render._static, render._tunnel, render.sprite):
        fn.cache_clear()
    h2 = [hashlib.sha1(np.asarray(render.frame_at(scene, i)).tobytes()).hexdigest() for i in idx]
    assert h == h2


def test_ladder_counts_order_and_min_fps():
    for s in SEEDS:
        qs = _sample(s)["questions"]
        assert L5.ladder_errors(qs, "ten") == []
        for q in qs:
            assert q["params"]["min_fps"] in questions.FPS_STEPS
            assert 2 <= len(q["options"]) <= 4 and len(set(q["options"])) == len(q["options"])


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


def test_objects_distinct_visible_and_apart():
    for s in range(20):
        sc = _sample(s)["params"]
        looks = [(o["colour"], o["pattern"], o["shape"]) for o in sc["objects"]]
        assert len(set(looks)) == len(looks)
        assert all(2 * o["r"] >= 40 for o in sc["objects"])           # >= 20 px objects, here >= 40 px
        assert sim.clearance(sc) >= 6


def test_timing_limits():
    for s in SEEDS:
        sc = _sample(s)["params"]
        n = sc["n_frames"]
        assert n <= sim.MAX_FRAMES
        v = sc["v2"] / 2
        for o in sc["objects"]:
            e, gate, land = o["keys"][1][0], o["keys"][2][0], o["keys"][4][0]
            assert e - sim.ext(o) / v >= sim.EDGE - 1e-9                # starts coming out after 0.5 s
            if o["pending"]:
                assert gate > n - 1 and e + sim.ext(o) / v <= n - 1 - sim.EDGE
            else:
                assert land <= n - 1 - sim.EDGE


def test_exactly_one_rule_fits():
    for s in SEEDS:
        sc = _sample(s)["params"]
        done = [o for o in sc["objects"] if not o["pending"]]
        nb = len(sim.bin_names(sc))
        fits = sim.hypotheses(done, [o["bin"] for o in done], nb)
        assert len(fits) == 1


def test_no_text_but_the_clock():
    for s in range(10):
        sc = _sample(s)["params"]
        assert sc["labels"] == [] and render.track(sc)["labels"] == []
    for k in (0, 37, 99):
        x0, y0, x1, y1 = font(sim.TIMER_FONT).getbbox(sim.clock_text(k))
        assert y1 - y0 >= 24                                           # readable at half size


def test_hand_checked_seeds():
    # seed 2: size rule (small -> lower, large -> upper); belt 59 frames, chute 25, in 10, slot 16 frames;
    # trays: upper 2 slots, lower 3; objects out of the tunnel at frames 20, 54, 77, 112 and 191 (pending)
    sc = _sample(2)["params"]
    assert sc["scenario"] == "size" and sc["rule"]["map"] == {"small": "lower", "large": "upper"}
    A = lambda k, p: answers.answer(k, p, sc)
    assert A("land_time", {"x": "the dotted orange star"}) == "6.0 s"        # 20 + 59 + 25 + 10 + 2 * 16 = 146
    assert A("land_time", {"x": "the plain purple star"}) == "6.8 s"         # 54 + 59 + 35 + 16 = 164
    assert A("gate_time", {"x": "the striped green star"}) == "7.1 s"        # 112 + 59 = 171
    assert A("pending_bin", {"x": "the striped purple triangle"}) == "the lower bin"
    assert A("pending_land", {"x": "the striped purple triangle"}) == "11.8 s"   # 191 + 59 + 35 + 0 = 285
    assert A("final_count", {"bin": "lower"}) == 3
    # without the orange star the green star takes the far slot: 171 + 35 + 32 = 238 -> 9.9 s
    assert A("cf_remove_time", {"x": "the dotted orange star", "y": "the striped green star"}) == "9.9 s"
    # sizes reversed: the purple star is the first large object, far lower slot: 113 + 35 + 32 = 180 -> 7.5 s
    assert A("cf_reverse_time", {"x": "the plain purple star"}) == "7.5 s"
    # twice as fast: rests at 83, 109, 124, 167 and 238 (pending), all <= frame 239
    assert A("cf_fast", {}) == 5
    # seed 5: alternate, starting with the lower bin; 6 sorted, 2 pending
    sc = _sample(5)["params"]
    A = lambda k, p: answers.answer(k, p, sc)
    assert sc["scenario"] == "alternate"
    assert A("pending_bin", {"x": "the striped orange circle"}) == "the lower bin"
    assert A("pending_bin", {"x": "the striped green square"}) == "the upper bin"
    assert A("cf_remove_bin", {"x": "the striped yellow star", "y": "the striped purple square"}) == "the lower bin"
    assert A("cf_new", {"size": "", "colour": "red", "shape": "star", "pattern": "plain"}) == "the lower bin"
    assert A("land_time", {"x": "the striped purple circle"}) == "7.0 s"     # 77 + 44 + 29 + 8 + 11 = 169
    assert A("count_at", {"bin": "lower", "k": 60}) == 1                    # 6.0 s = frame 144: rest at 134 only
    assert A("count_at", {"bin": "lower", "k": 64}) == 2                    # 6.4 s = frame 154: also 151


def test_decoder_reads_the_drawing_not_the_rule():
    # answers.py reads render.track only: strip everything else and the answers stay the same
    smp = _sample(9)
    sc = smp["params"]
    bare = dict(sc, rule={"kind": "attr", "attr": "pattern", "map": {}}, frames={},
                objects=[{k: v for k, v in o.items() if k in ("colour", "shape", "pattern", "r", "keys", "size")}
                         for o in sc["objects"]])
    bare["bins"] = {b: {k: v for k, v in g.items() if k != "cap"} for b, g in sc["bins"].items()}
    for q in smp["questions"]:
        assert str(answers.answer(q["derivation_key"], q["params"], bare)) == str(q["groundtruth"])
    M = answers.Model(render.track(sc))
    assert M.rule[0] == "attr" and M.rule[1] == sc["rule"]["attr"] and M.rule[2] == sc["rule"]["map"]


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
