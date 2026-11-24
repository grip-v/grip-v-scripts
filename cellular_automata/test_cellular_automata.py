"""Tests for cellular_automata:  python -m pytest cellular_automata/test_cellular_automata.py

determinism (in-process and across PYTHONHASHSEED values), frame determinism,
ladder counts, scene uniqueness, the tamper test, min_fps on every
question, and hand-checked scenes with known answers.
"""
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
from common.video5 import build_sample  # noqa: E402
import answers  # noqa: E402
import questions  # noqa: E402
import render  # noqa: E402
import sim  # noqa: E402

SEEDS = range(0, 120)


def _sample(seed):
    return json.loads(json.dumps(build_sample("cellular_automata", sim, questions, seed)))


@pytest.fixture(scope="module")
def samples():
    return [_sample(s) for s in SEEDS]


def test_same_seed_same_json(samples):
    for s in samples[:30]:
        assert _sample(s["seed"]) == s


def test_determinism_across_processes():
    code = ("import sys, json, hashlib; sys.path[:0]=[%r, %r]; import sim, questions; "
            "from common.video5 import build_sample; "
            "print(hashlib.sha256(json.dumps([build_sample('cellular_automata', sim, questions, s) "
            "for s in range(40)], sort_keys=True).encode()).hexdigest())") % (str(HERE), str(HERE.parent))
    outs = set()
    for hs in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                check=True).stdout.strip())
    assert len(outs) == 1


def test_frames_deterministic():
    sc = _sample(7)["params"]
    n = sim.n_frames(sc)
    h1 = [hashlib.sha1(render.frame(sc, i / sim.FPS).tobytes()).hexdigest() for i in range(0, n, 5)]
    h2 = [hashlib.sha1(render.frame(sc, i / sim.FPS).tobytes()).hexdigest() for i in range(0, n, 5)]
    assert h1 == h2


def test_ladder_and_min_fps(samples):
    for s in samples:
        assert L5.ladder_errors(s["questions"], "ten") == []
        assert all(q["params"]["min_fps"] > 0 for q in s["questions"])
        assert all(len(q["options"]) == 4 for q in s["questions"])


def test_scenes_unique_and_checked(samples):
    sigs = set()
    for s in samples:
        p = {k: v for k, v in s["params"].items() if k != "tries"}
        sigs.add(json.dumps(p, sort_keys=True))
        assert sim.check(s["params"]) == []
        assert render.duration(s["params"]) <= 10.0
    assert len(sigs) == len(samples)
    assert {s["params"]["scenario"] for s in samples} == set(sim.SCENARIOS)


def test_answers_match(samples):
    for s in samples:
        for q in s["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], s["params"])) == str(q["groundtruth"]), q


def test_tamper_flags_exactly_three(samples):
    tampered = copy.deepcopy(samples[:20])
    for s in tampered[:3]:
        q = s["questions"][4]
        wrong = next(o for o in q["options"] if o != q["answer_text"])
        q["groundtruth"], q["answer_text"] = wrong, wrong
        q["answer"] = "ABCD"[q["options"].index(wrong)]
    flagged = sum(str(answers.answer(q["derivation_key"], q["params"], s["params"])) != str(q["groundtruth"])
                  for s in tampered for q in s["questions"])
    assert flagged == 3


# ---- hand-checked scenes ------------------------------------------------------
def _scene(scenario, rule, start, gens=6):
    pool, nbhd, wrap, states = sim.SCENARIO_SETUP[scenario]
    sc = {"scenario": scenario, "rule": rule, "nbhd": nbhd, "wrap": wrap, "states": states,
          "rows": len(start), "cols": len(start[0]), "gens": gens, "start": start,
          "timing": {"start_f": 24, "gen_f": 24, "fade_f": 4, "end_f": 30},
          "markers": [], "style": {"live_name": "blue", "dying_name": "pink"}}
    sc["history"] = sim.as_lists(sim.history(sc))
    return sc


def _grid(R, C, cells):
    g = [[0] * C for _ in range(R)]
    for r, c in cells:
        g[r][c] = 1
    return g


def test_blinker_life():
    sc = _scene("life_like", "B3/S23", _grid(7, 7, [(3, 2), (3, 3), (3, 4)]))
    h = sim.history(sc)
    assert [sim.live_count(g) for g in h] == [3] * 7
    assert np.array_equal(h[1], np.array(_grid(7, 7, [(2, 3), (3, 3), (4, 3)])))
    assert answers.answer("cf_next", {"n": 1}, sc) == 3
    assert answers.answer("total_born", {}, sc) == 12          # 2 births per step, 6 steps
    assert answers.answer("ever_filled", {}, sc) == 5
    assert answers.answer("always_filled", {}, sc) == 1        # the centre


def test_torus_wraps():
    # a vertical blinker on the left edge: under wrap its row neighbours include the right edge
    sc = _scene("torus", "B3/S23", _grid(6, 6, [(2, 0), (3, 0), (4, 0)]))
    g1 = sim.history(sc)[1]
    assert g1[3, 5] == 1 and g1[3, 1] == 1 and g1[3, 0] == 1 and sim.live_count(g1) == 3


def test_brians_brain_pair():
    sc = _scene("generations", "B2/S", _grid(9, 9, [(4, 4), (4, 5)]))
    g1 = sim.history(sc, gens=1)[1]
    assert int((g1 == 1).sum()) == 4 and int((g1 == 2).sum()) == 2
    assert answers.answer("cf_rule", {"rule": "B2/S", "k": 1}, sc) == 4


def test_elementary_xor_sierpinski():
    start = [[0] * 21]
    start[0][10] = 1
    sc = _scene("elementary", "B1/S1", start, gens=7)       # new = left XOR right (Wolfram rule 90)
    assert [sim.live_count(g) for g in sim.history(sc)] == [1, 2, 2, 4, 2, 4, 4, 8]
    # from a single seed a filled cell never has 0 filled neighbours above, so S1 and S01 look the same:
    # the rule is not identifiable, answers.py refuses to answer and sim.check rejects the scene
    assert answers.answer("rule", {}, sc) is None
    sc["style"] = {"cell_px": 30}
    sc["markers"] = []
    assert any("indistinguishable" in e for e in sim.check(sc))


def test_time_refs_inside_still_windows(samples):
    for s in samples[:40]:
        sc = s["params"]
        for k in range(1, sc["gens"]):
            t = sim.time_ref(sc, k)
            fr = [f for f in range(sim.n_frames(sc)) if render.timer_text(f) == f"{t} s"]
            lo, hi = sim.stable_window(sc, k)
            assert fr and lo + 2 <= fr[0] and fr[-1] <= hi - 2
            assert all(render.gen_at_frame(sc, f) == (k, 1.0) for f in fr)
