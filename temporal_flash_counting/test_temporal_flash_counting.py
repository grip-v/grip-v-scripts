"""Tests for temporal_flash_counting:  python -m pytest temporal_flash_counting/test_temporal_flash_counting.py

determinism (in-process, across PYTHONHASHSEED values, raw frames), ladder
counts, scene uniqueness, sim.check on every scene, answers agreement, the
tamper test, min_fps on every question (and that every-k-th-frame sampling
at that rate still gives the stored answer), the timer as the only text,
and hand-built scenes with known answers.
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

SEEDS = range(0, 150)


def _sample(seed):
    return json.loads(json.dumps(build_sample("temporal_flash_counting", sim, questions, seed)))


@pytest.fixture(scope="module")
def samples():
    return [_sample(s) for s in SEEDS]


def test_same_seed_same_json(samples):
    for s in samples[:30]:
        assert _sample(s["seed"]) == s


def test_determinism_across_processes():
    code = ("import sys, json, hashlib; sys.path[:0]=[%r, %r]; import sim, questions; "
            "from common.video5 import build_sample; "
            "print(hashlib.sha256(json.dumps([build_sample('temporal_flash_counting', sim, questions, s) "
            "for s in range(60)], sort_keys=True).encode()).hexdigest())") % (str(HERE), str(HERE.parent))
    outs = set()
    for hs in ("0", "1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                check=True).stdout.strip())
    assert len(outs) == 1


def test_frames_deterministic(samples):
    for s in samples[:5]:
        sc = s["params"]
        h1 = [hashlib.sha1(render.frame_at(sc, f).tobytes()).hexdigest() for f in range(0, sc["n_frames"], 7)]
        render._BG_CACHE.clear()
        h2 = [hashlib.sha1(render.frame_at(json.loads(json.dumps(sc)), f).tobytes()).hexdigest()
              for f in range(0, sc["n_frames"], 7)]
        assert h1 == h2


def test_ladder_and_min_fps(samples):
    for s in samples:
        assert L5.ladder_errors(s["questions"], "ten") == []
        for q in s["questions"]:
            assert q["params"]["min_fps"] > 0 and len(q["options"]) == 4
            assert q["derivation_key"] and isinstance(q["params"], dict)


def test_scenes_unique_checked_and_short(samples):
    sigs = set()
    for s in samples:
        p = {k: v for k, v in s["params"].items() if k != "tries"}
        sigs.add(json.dumps(p, sort_keys=True))
        assert sim.check(s["params"]) == []
        assert render.duration(s["params"]) <= 10.0
        for f in s["params"]["flashes"]:
            assert f["f0"] >= 12 and f["f1"] <= s["params"]["n_frames"] - 12      # >= 0.5 s from both ends
    assert len(sigs) == len(samples)
    assert {s["params"]["scenario"] for s in samples} == set(sim.SCENARIOS)


def test_answers_match(samples):
    for s in samples:
        for q in s["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], s["params"])) == str(q["groundtruth"]), q


def test_tamper_flags_exactly_three(samples):
    tampered = copy.deepcopy(samples[:20])
    for s in tampered[:3]:
        q = s["questions"][6]
        wrong = next(o for o in q["options"] if o != q["answer_text"])
        q["groundtruth"], q["answer_text"] = wrong, wrong
        q["answer"] = "ABCD"[q["options"].index(wrong)]
    flagged = sum(str(answers.answer(q["derivation_key"], q["params"], s["params"])) != str(q["groundtruth"])
                  for s in tampered for q in s["questions"])
    assert flagged == 3


def _state_looks(scene):
    nf, n = scene["n_frames"], len(scene["shapes"])
    pals = [render.palette(scene, i) for i in range(n)]
    sts = [render.state(scene, f) for f in range(nf)]
    modes = [[answers._nearest(sts[f][i]["fill"], pals[i]) for f in range(nf)] for i in range(n)]
    grown = [[sts[f][i]["r"] - scene["shapes"][i]["r"] >= 9 for f in range(nf)] for i in range(n)]
    return modes, grown, [(s["x"], s["y"]) for s in sts[0]]


def test_frame_sampling_at_min_fps(samples):
    """Every k-th frame (k = floor(24 / min_fps), every phase) still gives the answer."""
    for s in samples[:60]:
        sc = s["params"]
        modes, grown, cents = _state_looks(sc)
        for q in s["questions"]:
            k = max(1, int(sim.FPS / q["params"]["min_fps"] + 1e-9))
            for ph in range(k):
                ev, pul = answers.events_from(modes, grown, range(ph, sc["n_frames"], k), sc["n_frames"])
                old = answers.track
                answers.track = lambda scene: (ev, pul, cents)
                try:
                    val = answers.answer(q["derivation_key"], q["params"], sc)
                finally:
                    answers.track = old
                if q["derivation_key"] == "first_time":
                    v = float(val.split()[0])
                    val = min(q["options"], key=lambda o: abs(float(o.split()[0]) - v))
                    assert val == q["answer_text"], (s["seed"], q["qid"], k, ph)
                else:
                    assert str(val) == str(q["groundtruth"]), (s["seed"], q["qid"], q["derivation_key"], k, ph)


def test_timer_is_the_only_text_and_clear_of_shapes(samples):
    for s in samples[:40]:
        sc = s["params"]
        for f in range(0, sc["n_frames"], 12):
            for i, st in enumerate(render.state(sc, f)):
                R = sim.max_radius(sc, i)
                assert st["y"] - R >= sim.AREA[1] > render.TIMER_PLATE[3]
    assert sim.timer_text(0) == "0.0 s" and sim.timer_text(23) == "0.9 s" and sim.timer_text(24) == "1.0 s"
    assert sim.frame_of_reading(25) == 60 and sim.tenths(59) == 24


# ---- hand-built scenes ----------------------------------------------------------
def _scene(flashes, n=4, nf=200, tempo="fast", pulses=(), chain=None, scenario="scatter"):
    shapes = []
    for i in range(n):
        shapes.append({"x": 200.0 + 250 * i, "y": 400.0, "r": 40.0, "rot": 0.0, "type": sim.TYPES[i],
                       "color": sorted(sim.COLORS)[i], "aspect": 1.6, "rhythm": "random"})
    fl, ids = [], {}
    for k, (i, f0, f1, white, cause) in enumerate(sorted(flashes, key=lambda x: (x[1], x[0]))):
        fl.append({"i": i, "f0": f0, "f1": f1, "white": white, "cause": cause, "id": k})
        ids[(i, f0)] = k
    for f in fl:                                        # causes are given as (shape, onset frame)
        f["cause"] = ids[f["cause"]] if f["cause"] is not None else None
    return {"scenario": scenario, "n_frames": nf, "tempo": tempo,
            "style": {"bg_name": "navy", "texture": "plain", "tex_step": 40, "tex_amp": 6, "outline": 4,
                      "rest_mix": 0.25, "white_mix": 0.85, "glow_mix": 0.5, "plate": "none", "rails": False},
            "shapes": shapes, "flashes": fl, "pulses": [{"i": i, "f0": a, "f1": b} for i, a, b in pulses],
            "causal": {"chain": chain} if chain else None, "pair": None, "pair_kind": None}


def test_counts_lengths_whites():
    sc = _scene([(0, 20, 27, False, None), (0, 60, 85, True, None), (0, 120, 128, False, None),
                 (1, 30, 38, True, None), (1, 100, 108, True, None),
                 (2, 40, 48, False, None), (2, 150, 175, False, None), (3, 50, 58, False, None),
                 (3, 90, 98, False, None)], pulses=[(3, 120, 136)])
    assert answers.answer("flash_count", {"x": 0}, sc) == 3
    assert answers.answer("long_count", {"x": 0, "cut": 0.6}, sc) == 1          # 25 frames > 14.4
    assert answers.answer("white_count", {"x": 1}, sc) == 2
    assert answers.answer("sequence", {"x": 0, "cut": 0.6}, sc) == "short, long, short"
    assert answers.answer("first_flasher", {}, sc) == sim.label(sc, 0)
    assert answers.answer("last_flasher", {}, sc) == sim.label(sc, 2)
    assert answers.answer("total_flashes", {}, sc) == 9
    assert answers.answer("first_time", {"x": 3}, sc) == "2.0 s"                # frame 50 -> 2.08 s
    assert answers.answer("cf_stop", {"x": 0, "t": 30}, sc) == 2                 # 3.0 s = frame 72
    assert answers.answer("count_window", {"a": 10, "b": 30}, sc) == 4          # onsets in frames [24, 72)
    assert answers.answer("pulse_shape", {"opts": [3, 0, 1, 2]}, sc) == sim.label(sc, 3)
    assert answers.answer("more_by", {"pair": [0, 1]}, sc) == 1
    c = questions.Ctx(sc)
    assert c.longs == {0: 1, 1: 0, 2: 1, 3: 0} and c.whites[1] == 2


def test_relations():
    base = [(2, 150, 160, False, None), (3, 160, 170, False, None)]
    together = _scene(base + [(0, 40, 48, False, None), (1, 40, 60, False, None)])
    overlap = _scene(base + [(0, 40, 70, False, None), (1, 52, 80, False, None)])
    seq = _scene(base + [(0, 40, 48, False, None), (1, 60, 68, False, None)])
    for sc, want in ((together, 0), (overlap, 1), (seq, 2)):
        assert answers.answer("relation", {"pair": [0, 1]}, sc) == sim.RELATIONS[want]
        x, y = sim.flashes_of(sc, 0)[0], sim.flashes_of(sc, 1)[0]
        assert sim.relation(x, y, 8) == sim.RELATIONS[want]


def test_relay_counterfactual_chain():
    # A (0) -> B (1) -> C (2), lag 24 frames; B has one flash of its own (C reacts to it too)
    a_on = (20, 80, 140, 200)
    fl = [(0, t, t + 8, False, None) for t in a_on]

    b_fl = [(1, t + 24, t + 32, False, (0, t)) for t in a_on] + [(1, 15, 23, False, None)]
    c_fl = [(2, t + 48, t + 56, False, (1, t + 24)) for t in a_on] + [(2, 39, 47, False, (1, 15))]
    sc = _scene(fl + b_fl + c_fl, nf=300, chain=[0, 1, 2], scenario="relay")
    assert answers.answer("trigger", {"resp": 1, "opts": [0, 2, 3]}, sc) == sim.label(sc, 0)
    assert answers.answer("trigger", {"resp": 2, "opts": [1, 0, 3]}, sc) == sim.label(sc, 1)
    # without A: B keeps only its own flash; C keeps only its reaction to that flash
    assert answers.answer("cf_no_trigger", {"trig": 0, "resp": 1}, sc) == 1
    assert answers.answer("cf_chain", {"trig": 0, "via": 1, "resp": 2}, sc) == 1
    assert sum(1 for f in sim.resimulate(sc, {0}) if f["i"] == 2) == 1
    assert answers.answer("cf_no_trigger", {"trig": 1, "resp": 2}, sc) == 0


def test_steady_and_period():
    fl = [(0, 20 + 36 * k, 28 + 36 * k, False, None) for k in range(5)]           # every 1.5 s
    fl += [(1, 20, 28, False, None), (1, 50, 58, False, None), (1, 140, 148, False, None)]
    fl += [(2, 30, 38, False, None), (2, 120, 128, False, None), (2, 160, 168, False, None)]
    fl += [(3, 40, 48, False, None), (3, 70, 78, False, None), (3, 175, 183, False, None)]
    sc = _scene(fl)
    assert answers.answer("period", {"x": 0}, sc) == "1.5 s"
    assert answers.answer("steady", {"opts": [1, 2, 0, 3]}, sc) == sim.label(sc, 0)
