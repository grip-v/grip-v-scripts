"""Tests for blinking_networks:  python -m pytest blinking_networks/test_blinking_networks.py"""
from __future__ import annotations
import hashlib
import math
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

import answers  # noqa: E402
import questions  # noqa: E402
import render  # noqa: E402
import sim  # noqa: E402
from common import ladder5 as L5  # noqa: E402
from common.video5 import build_sample, _roundtrip  # noqa: E402

SEEDS = range(40)


def _sample(seed):
    return _roundtrip(build_sample("blinking_networks", sim, questions, seed))


# ------------------------------------------------------------ hand-built worlds
def _node(i, x, y, colour, shape, thr=1):
    return {"id": f"n{i}", "x": x, "y": y, "colour": colour, "rgb": list(sim.COLOURS[colour]), "shape": shape,
            "pattern": "solid", "threshold": thr, "name": f"{colour} {shape}"}


def _world(scenario, nodes, edges, starts, speed=100.0):
    return {"scenario": scenario, "R": 24, "nodes": nodes, "speed": speed, "starts": starts,
            "edges": [{"a": a, "b": b, "slow": slow, "oneway": ow} for a, b, slow, ow in edges]}


def test_chain_with_hidden_slow_link():
    # A(200)--B(400)--C(600); B-C slow. speed 100 px/s: A->B 2 s, B->C 4 s.
    w = _world("hidden_slow", [_node(0, 200, 300, "red", "circle"), _node(1, 400, 300, "blue", "square"),
                               _node(2, 600, 300, "green", "star")],
               [("n0", "n1", False, False), ("n1", "n2", True, False)], [["n0", 1.0]])
    r = sim.simulate(w)
    assert r["fire"] == {"n0": 1.0, "n1": 3.0, "n2": 7.0}
    assert r["parent"] == {"n0": None, "n1": "n0", "n2": "n1"}
    ind = answers.run(w)["lit"]
    assert all(abs(ind[k] - r["fire"][k]) < 1e-6 for k in r["fire"])
    assert answers.run(w, unslow=True)["lit"]["n2"] == pytest.approx(5.0, abs=1e-6)


def test_triangle_shortcut_and_fizzle():
    # A-B 300 px, A-C 400 px, B-C 500 px at 100 px/s: B lit at 3, C at 4; B->C pulse fizzles at 8.
    w = _world("hidden_slow", [_node(0, 100, 100, "red", "circle"), _node(1, 400, 100, "blue", "square"),
                               _node(2, 100, 500, "green", "star")],
               [("n0", "n1", False, False), ("n0", "n2", False, False), ("n1", "n2", False, False)],
               [["n0", 0.0]])
    r = sim.simulate(w)
    assert r["fire"] == {"n0": 0.0, "n1": 3.0, "n2": 4.0}
    res = {(p["from"], p["to"]): p["result"] for p in r["pulses"]}
    assert res[("n1", "n2")] == "fizzle" and res[("n2", "n1")] == "fizzle"
    assert answers.run(w, cut=["n0", "n2"])["lit"]["n2"] == pytest.approx(8.0, abs=1e-6)


def test_one_way_blocks_and_reverse():
    w = _world("one_way", [_node(0, 100, 100, "red", "circle"), _node(1, 300, 100, "blue", "square"),
                           _node(2, 500, 100, "green", "star")],
               [("n0", "n1", False, True), ("n2", "n1", False, True)], [["n0", 0.0]])
    assert set(sim.simulate(w)["fire"]) == {"n0", "n1"}
    assert set(sim.simulate(w, reverse=["n1", "n2"])["fire"]) == {"n0", "n1", "n2"}
    assert len(answers.run(w, reverse=["n1", "n2"])["lit"]) == 3


def test_double_trigger_needs_two_pulses():
    # D (double) hears A at 2 s and B at 3 s -> lights at 3 s; with one pulse it never lights.
    w = _world("double_trigger", [_node(0, 100, 300, "red", "circle"), _node(1, 300, 300, "blue", "square", 2),
                                  _node(2, 300, 600, "green", "star")],
               [("n0", "n1", False, False), ("n0", "n2", False, False), ("n2", "n1", False, False)],
               [["n0", 0.0]], speed=100.0)
    r = sim.simulate(w)
    t_a, t_c = 2.0, (math.dist((100, 300), (300, 600)) + 300) / 100     # 2.0 s and 6.61 s
    assert r["fire"]["n1"] == pytest.approx(max(t_a, t_c), abs=1e-6)
    assert r["half"]["n1"] == pytest.approx(t_a, abs=1e-6)
    assert "n1" not in sim.simulate(w, cut=["n2", "n1"])["fire"]
    assert answers.run(w, single=True)["lit"]["n1"] == pytest.approx(t_a, abs=1e-6)


def test_two_sources_origin():
    w = _world("two_sources", [_node(0, 100, 100, "red", "circle"), _node(1, 300, 100, "blue", "square"),
                               _node(2, 500, 100, "green", "star"), _node(3, 700, 100, "pink", "diamond")],
               [("n0", "n1", False, False), ("n1", "n2", False, False), ("n2", "n3", False, False)],
               [["n0", 1.0], ["n3", 1.5]])
    r = sim.simulate(w)
    assert r["fire"] == {"n0": 1.0, "n1": 3.0, "n2": 3.5, "n3": 1.5}
    assert r["parent"]["n2"] == "n3"
    assert answers.answer("origin", {"x": "n2"}, w) == "pink diamond"


# ---------------------------------------------------------------- generated
def test_ladder_counts_order_and_min_fps():
    for seed in SEEDS:
        s = _sample(seed)
        assert L5.ladder_errors(s["questions"], "ten") == []
        assert [q["qid"] for q in s["questions"]] == [f"q{i:02d}" for i in range(1, 11)]
        for q in s["questions"]:
            assert q["params"]["min_fps"] > 0
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4


def test_answers_independent_and_check_clean():
    for seed in SEEDS:
        s = _sample(seed)
        assert sim.check(s["params"]) == []
        for q in s["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], s["params"])) == str(q["groundtruth"])
            assert q["options"]["ABCD".index(q["answer"])] == q["answer_text"] == str(q["groundtruth"])


def test_uniqueness_of_scenes_and_names():
    sigs = set()
    for seed in range(200):
        sc = sim.build(random.Random(seed))
        sigs.add(sim.signature(sc))
        names = [n["name"] for n in sc["nodes"]]
        assert len(set(names)) == len(names)
        assert len({(n["colour"], n["shape"]) for n in sc["nodes"]}) == len(names)
    assert len(sigs) == 200


def test_scene_timing_rules():
    for seed in SEEDS:
        sc = _sample(seed)["params"]
        assert sc["end_t"] <= 10.0
        ev = list(sc["fire"].values()) + [t for a in sc["arrivals"].values() for t, _ in a]
        assert min(ev) >= 0.5 and max(ev) <= sc["end_t"] - 0.5


def test_tamper_is_caught():
    rows = [_sample(seed) for seed in range(10)]
    rng = random.Random(3)
    targets = rng.sample([(i, j) for i in range(10) for j in range(10)], 3)
    for i, j in targets:
        q = rows[i]["questions"][j]
        wrong = next(x for x in "ABCD" if x != q["answer"])
        q["groundtruth"] = q["options"]["ABCD".index(wrong)]
    flagged = sum(str(answers.answer(q["derivation_key"], q["params"], r["params"])) != str(q["groundtruth"])
                  for r in rows for q in r["questions"])
    assert flagged == 3


def test_deterministic_across_processes():
    code = ("import sys, json; sys.path[:0] = [%r, %r]; import sim, questions; "
            "from common.video5 import build_sample; "
            "print(json.dumps([build_sample('blinking_networks', sim, questions, s) for s in range(12)], "
            "sort_keys=True))") % (str(HERE), str(HERE.parent))
    outs = []
    for hs in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                   check=True).stdout)
    assert outs[0] == outs[1]


def test_render_deterministic_and_timer_only_text_region():
    sc = _sample(5)["params"]
    h = []
    for _ in range(2):
        render._CACHE.clear()
        m = hashlib.sha256()
        for t in (0.0, 1.0, sc["end_t"] / 2, sc["end_t"] - 0.05):
            m.update(np.asarray(render.frame(sc, t)).tobytes())
        h.append(m.hexdigest())
    assert h[0] == h[1]
    x0, y0, x1, y1 = sim.TIMER_BOX
    for n in sc["nodes"]:      # nothing of the network comes near the clock
        assert n["y"] - sc["R"] - 40 > y1 or n["x"] + sc["R"] + 40 < x0


def test_stale_state_between_seeds():
    a = json.dumps(_sample(17), sort_keys=True)
    for s in (3, 99, 4):
        _sample(s)
    assert json.dumps(_sample(17), sort_keys=True) == a
