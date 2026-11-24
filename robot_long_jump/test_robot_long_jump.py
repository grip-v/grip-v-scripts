"""Tests for robot_long_jump: determinism, ladder, uniqueness, closed loop, labels, tamper, edge cases.

    cd robot_long_jump && python -m pytest -q test_robot_long_jump.py
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

import answers  # noqa: E402
import questions  # noqa: E402
import render  # noqa: E402
import sim  # noqa: E402
from common import ladder5 as L5  # noqa: E402
from common.video5 import build_sample, run_generate, run_validate  # noqa: E402

DS = "robot_long_jump"
SEEDS = range(0, 200)


def _sample(seed):
    return json.loads(json.dumps(build_sample(DS, sim, questions, seed)))


def test_same_seed_same_json_across_processes():
    code = ("import json,sys; sys.path[:0]=[%r,%r]; import sim,questions; from common.video5 import build_sample; "
            "print(json.dumps([build_sample('x', sim, questions, s) for s in range(40)], sort_keys=True))"
            % (str(HERE), str(HERE.parent)))
    outs = []
    for hs in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                                   check=True).stdout)
    assert outs[0] == outs[1] and len(outs[0]) > 1000


def test_render_deterministic_frames():
    for seed in (1, 2, 3):
        scene = _sample(seed)["params"]
        for f in (0, 30, scene["n_frames"] // 2, scene["n_frames"] - 1):
            a = np.asarray(render.frame(scene, f / 24)).tobytes()
            render._sprite.cache_clear()
            render._static.cache_clear()
            b = np.asarray(render.frame(copy.deepcopy(scene), f / 24)).tobytes()
            assert hashlib.sha1(a).digest() == hashlib.sha1(b).digest()


def test_ladder_counts_order_and_min_fps():
    for seed in SEEDS:
        qs = _sample(seed)["questions"]
        assert L5.ladder_errors(qs, "ten") == []
        for q in qs:
            assert 0 < q["params"]["min_fps"] <= 4
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4
            assert q["options"].count(q["answer_text"]) == 1


def test_scenes_unique_checked_and_identifiable():
    sigs = set()
    for seed in SEEDS:
        s = _sample(seed)["params"]
        assert sim.check(s) == []
        assert s["n_frames"] <= 240
        sigs.add(json.dumps(s, sort_keys=True))
        cols = [r["colour"] for r in s["robots"]]
        assert len(set(cols)) == len(cols) and "red" not in cols
        sc = sorted(sim.score(r) for r in s["robots"] if not r["fouled"])
        assert all(b - a >= 1 for a, b in zip(sc, sc[1:]))          # no score ties (old seed-81 bug)
    assert len(sigs) == len(SEEDS)


def test_independent_answers_match():
    for seed in SEEDS:
        s = _sample(seed)
        for q in s["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], s["params"])) == str(q["groundtruth"]), \
                (seed, q["qid"], q["derivation_key"])


def test_all_scenarios_present():
    assert {_sample(seed)["params"]["scenario"] for seed in range(40)} == set(sim.SCENARIOS)


def test_labels_match_renderer():
    """The ruler numbers drawn in the frame are exactly scene['labels'] (text, place, size >= 24 px)."""
    import verify_frames as VF
    for seed in (0, 5, 9):
        scene = _sample(seed)["params"]
        fr = np.asarray(render.frame(scene, 0.0))
        errs = []
        got = VF._read_labels(fr, scene, errs)
        assert errs == [] and set(got) == {lab["text"] for lab in scene["labels"]}
        assert [lab["text"] for lab in scene["labels"]] == ["0.0", "0.5", "1.0", "1.5", "2.0"]


def test_tamper(tmp_path):
    run_generate(DS, HERE, sim, questions, render, ["--n", "12", "--seed", "3", "--no-render",
                                                     "--output-dir", str(tmp_path)])
    run_validate(DS, HERE, sim, questions, answers, render, ["--dataset-dir", str(tmp_path), "--skip-video"])
    lines = (tmp_path / "annotations.jsonl").read_text().splitlines()
    for i in (0, 4, 9):
        s = json.loads(lines[i])
        q = s["questions"][5]
        wrong = next(o for o in q["options"] if o != q["answer_text"])
        q["groundtruth"], q["answer_text"] = wrong, wrong
        q["answer"] = "ABCD"[q["options"].index(wrong)]
        lines[i] = json.dumps(s)
    (tmp_path / "t.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(SystemExit):
        run_validate(DS, HERE, sim, questions, answers, render, ["--dataset-dir", str(tmp_path), "--skip-video",
                                                                 "--ann", "t.jsonl", "--report", "t.txt"])
    rep = (tmp_path / "t.txt").read_text()
    assert sum(1 for ln in rep.splitlines() if "independent answer" in ln) == 3


# --------------------------------------------------------------------------- hand-checked edge cases


def _hand_scene():
    """Turns, order orange -> blue -> green.
    orange: hops 0.3 0.2 0.2 -> take-off 0.7, jump 0.6 -> lands 1.3 at frame 53 (clock 2.2 s)
    blue:   hops 0.4 0.4 0.3 -> 0.8 is short of its threshold 0.9, 1.1 is past the line: foul after 3 hops
    green:  hops 0.2 0.2 0.2 -> take-off 0.6, jump 0.9 -> lands 1.5 at frame 132"""
    view = {"x0": 160, "ppu": 520, "pitch": 150, "lane_y": [248, 398, 548], "strip_h": 16, "ruler_y": 648}
    rb = lambda i, col, lane, tau, hops, fouled, jump, hf, jf, ef: {
        "id": i, "colour": col, "shape": "circle", "pattern": "plain", "size": 25, "lane": lane, "tau": tau,
        "start": 0, "hops": hops, "fouled": fouled, "jump": jump, "order": i, "hop_frames": hf, "jump_frame": jf,
        "end_frame": ef}
    return {"scenario": "turns", "n_frames": 146, "foul": 10,
            "timing": {"hop_f": 6, "rest_f": 3, "jump_f": 14, "tip_f": 6, "gap_f": 4},
            "jump_h": 70, "hop_h": 30,
            "theme": {"bg": [238, 234, 224], "runway": [150, 86, 72], "sand": [234, 212, 160], "foot": "dot",
                      "grid_alpha": 0.35, "eye": "round"},
            "view": view, "labels": sim._labels(view), "timer": {"box": [16, 14, 196, 66], "size": 36}, "tries": 1,
            "robots": [rb(0, "orange", 0, 7, [3, 2, 2], False, 6, [12, 21, 30], 39, 53),
                       rb(1, "blue", 1, 9, [4, 4, 3], True, None, [57, 66, 75], None, 81),
                       rb(2, "green", 2, 6, [2, 2, 2], False, 9, [91, 100, 109], 118, 132)]}


def test_hand_scene():
    s = _hand_scene()
    assert sim.check(s) == []
    A = lambda k, p=None: answers.answer(k, p or {}, s)
    assert A("hops_jump", {"r": 0}) == 3
    assert A("hops_foul", {"r": 1}) == 3            # the hop that crosses the line counts
    assert A("n_fouls") == 1
    assert A("who_fouls") == "the blue robot"
    assert A("takeoff", {"r": 0}) == "0.7"
    assert A("jump_length", {"r": 2}) == "0.9"
    assert A("score", {"r": 2}) == "1.5"
    assert A("top_scorer") == "the green robot"
    assert A("ranking", {"ids": [0, 1, 2]}) == "green > orange > blue"
    assert A("place", {"r": 0}) == "2nd"
    assert A("win_margin") == "0.2"
    assert A("land_time", {"r": 0}) == "2.2 s"
    assert A("takeoff_time", {"r": 0}) == "1.6 s"
    assert A("first_land", {"a": 2, "b": 0}) == "the orange robot"
    assert A("longest_hop", {"r": 1}) == "0.4"
    assert A("lead_changes") == 1
    assert A("leader_after", {"r": 0}) == "the orange robot"
    assert A("leader_after", {"r": 2}) == "the green robot"
    assert A("landed_before", {"r": 2}) == 1
    assert A("cf_jump_earlier", {"r": 0}) == "1.1"
    assert A("cf_threshold", {"r": 0, "p": 3}) == "0.9"
    assert A("cf_foul_line", {"line": 9}) == 1
    assert A("cf_swap", {"a": 0, "b": 2}) == "1.6"
    assert A("cf_longer_place", {"r": 0, "d": 3}) == "1st"
    assert A("cf_lead_changes", {"r": 2}) == 0
    assert A("cf_leader_after", {"r": 0, "s": 2}) == "the green robot"


def test_hand_scene_rejections():
    s = _hand_scene()
    s["robots"][2]["jump"] = 7                       # green lands on 1.3 like orange: a tie -> rejected
    assert any("tie" in e for e in sim.check(s))
    s = _hand_scene()
    s["robots"][1]["hops"] = [4, 4, 2]               # a footprint on the foul line itself -> rejected
    assert sim.check(s)
    s = _hand_scene()
    s["robots"][0]["end_frame"] = 54                 # landing not on a timer-aligned frame -> rejected
    assert any("aligned" in e for e in sim.check(s))
    s = _hand_scene()
    s["n_frames"] = 140                              # last landing within 0.5 s of the end -> rejected
    assert any("edge" in e for e in sim.check(s))


def test_hop_question_wording():
    """Old wording bug: 'steps before jumping or fouling' was asked of fouled robots."""
    for seed in SEEDS:
        s = _sample(seed)
        fouled = {r["id"] for r in s["params"]["robots"] if r["fouled"]}
        for q in s["questions"]:
            if q["derivation_key"] == "hops_jump":
                assert q["params"]["r"] not in fouled
            if q["derivation_key"] == "hops_foul":
                assert q["params"]["r"] in fouled and "counting the hop that carries it past" in q["question"]


TOP = {"circle": 1.0, "square": 0.86, "triangle": 1.08, "hexagon": 0.87, "diamond": 1.12}


def robot_box(r, st):
    """Bounding box of a robot (body + legs) in output pixels, rotated with its tilt."""
    import math
    s = r["size"]
    pts = [(-1.15 * s, -TOP[r["shape"]] * s), (1.15 * s, -TOP[r["shape"]] * s), (1.15 * s, s + render.LEG),
           (-1.15 * s, s + render.LEG)]
    a = math.radians(st["tilt"])
    q = [(st["x"] + dx * math.cos(a) - dy * math.sin(a), st["y"] + dx * math.sin(a) + dy * math.cos(a)) for dx, dy in pts]
    xs, ys = zip(*q)
    return min(xs), min(ys), max(xs), max(ys)


def test_robots_never_overlap_and_stay_clear_of_text():
    for seed in range(80):
        scene = _sample(seed)["params"]
        for f in range(scene["n_frames"]):
            lay = render.layout(scene, f)
            B = [robot_box(r, lay["robots"][r["id"]]) for r in scene["robots"]]
            for i in range(len(B)):
                assert B[i][1] > scene["timer"]["box"][3] or B[i][0] > scene["timer"]["box"][2]
                assert B[i][3] < render.RULER_BAND[1]
                for j in range(i + 1, len(B)):
                    a, b = B[i], B[j]
                    assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]), (seed, f, i, j)
