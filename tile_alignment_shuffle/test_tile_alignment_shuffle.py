"""Tests for tile_alignment_shuffle: determinism, ladder, uniqueness, closed loop, labels, tamper, edge cases.

    cd tile_alignment_shuffle && python -m pytest -q test_tile_alignment_shuffle.py
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

DS = "tile_alignment_shuffle"
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
        for f in (0, 20, scene["n_frames"] // 2, scene["n_frames"] - 1):
            a = np.asarray(render.frame(scene, f / 24)).tobytes()
            render.sprite.cache_clear()
            render._background.cache_clear()
            render._boards.cache_clear()
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


def test_scenes_unique_checked_and_short():
    sigs = set()
    for seed in SEEDS:
        s = _sample(seed)["params"]
        assert sim.check(s) == []
        assert s["n_frames"] <= 240
        sigs.add(json.dumps(s, sort_keys=True))
        cols = [d["color"] for d in s["designs"]]
        assert len(set(cols)) == len(cols)                     # every kind is identifiable by colour
        gs = sim.grids(s)                                      # no line before the final one forms (fix #2)
        for k, g in enumerate(gs):
            assert not sim.alignments(s["tiles"], g) or (s["line"] and k >= s["line_move"])
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
    """The row/column numbers drawn in the frame are exactly scene['labels'] (text, place, >= 24 px tall)."""
    import verify_frames as VF
    for seed in (0, 5, 9):
        scene = _sample(seed)["params"]
        fr = np.asarray(render.frame(scene, 0.0))
        errs = []
        VF.check_labels(scene, [fr], errs)
        assert errs == []
        n = scene["n"]
        assert sorted(lab["text"] for lab in scene["labels"]) == sorted([str(i + 1) for i in range(n)] * 2)


def test_tamper(tmp_path):
    run_generate(DS, HERE, sim, questions, render, ["--n", "12", "--seed", "3", "--no-render",
                                                     "--output-dir", str(tmp_path)])
    run_validate(DS, HERE, sim, questions, answers, render, ["--dataset-dir", str(tmp_path), "--skip-video"])
    lines = (tmp_path / "annotations.jsonl").read_text(encoding="utf8").splitlines()
    for i in (0, 4, 9):
        s = json.loads(lines[i])
        q = s["questions"][5]
        wrong = next(o for o in q["options"] if o != q["answer_text"])
        q["groundtruth"], q["answer_text"] = wrong, wrong
        q["answer"] = "ABCD"[q["options"].index(wrong)]
        lines[i] = json.dumps(s)
    (tmp_path / "t.jsonl").write_text("\n".join(lines) + "\n", encoding="utf8")
    with pytest.raises(SystemExit):
        run_validate(DS, HERE, sim, questions, answers, render, ["--dataset-dir", str(tmp_path), "--skip-video",
                                                                 "--ann", "t.jsonl", "--report", "t.txt"])
    rep = (tmp_path / "t.txt").read_text(encoding="utf8")
    assert sum(1 for ln in rep.splitlines() if "independent answer" in ln) == 3


# --------------------------------------------------------------------------- hand-checked edge case


def _hand_scene():
    """3x3, designs A red circle (ids 0,1,2), B blue square (3,5,7), C green triangle (4,6,8).
    start  [[4,7,2],[3,0,5],[6,1,8]]   designs C B A / B A B / C A C   (no line)
    move 1 swaps (0,1)<->(2,1): [[4,1,2],[3,0,5],[6,7,8]]              (no line)
    move 2 swaps (0,0)<->(1,1): [[0,1,2],[3,4,5],[6,7,8]]              row 1 = A A A, the only line
    timing: intro 12, move 10, pause 6, hold 14 -> rest after move 1 at frame 22 (0.9 s), after move 2 at 38."""
    n = 3
    lay = {"x0": 420, "y0": 140, "span": 420, "pad": 10}
    return {"scenario": "swap", "n": n,
            "designs": [{"kind": "circle", "color": "red", "false": False},
                        {"kind": "square", "color": "blue", "false": False},
                        {"kind": "triangle", "color": "green", "false": False}],
            "tiles": [0, 0, 0, 1, 2, 1, 2, 1, 2],
            "start": [[4, 7, 2], [3, 0, 5], [6, 1, 8]],
            "moves": [sim._swap_move(n, (0, 1), (2, 1), "swap"), sim._swap_move(n, (0, 0), (1, 1), "swap")],
            "line": True, "line_move": 2, "near_miss": False,
            "timing": {"intro": 12, "move": 10, "pause": 6, "hold": 14}, "n_frames": 58,
            "layout": lay, "labels": sim.label_list(n, lay["x0"], lay["y0"], lay["span"]),
            "timer": {"box": list(sim.TIMER_BOX), "size": sim.TIMER_SIZE}, "theme": "paper", "card": "ivory",
            "style": {"radius": 0.1, "gap": 0.07, "scale": 0.3, "border": 3, "lift": 0.08}}


HAND = [
    ("n_moves", {}, 2),
    ("moves_of", {"cell": [0, 0]}, 1),
    ("moves_of", {"cell": [1, 0]}, 0),
    ("end_cell", {"cell": [0, 0]}, "row 2, column 2"),
    ("start_of", {"cell": [0, 1]}, "row 3, column 2"),
    ("never_moved", {}, 5),
    ("back_home", {}, 0),
    ("line_after", {}, "the 2nd move"),
    ("line_where", {}, "row 1"),
    ("aligned_design", {}, "red circle"),
    ("partner", {"cell": [0, 0]}, "red circle"),
    ("partner_origin", {"cell": [0, 0]}, "row 2, column 2"),
    ("line_origin", {}, 1),
    ("cells_visited", {"cell": [0, 0]}, 2),
    ("first_rest_time", {"cell": [0, 1]}, "0.9 s"),
    ("cf_skip", {"cell": [2, 1], "k": 0}, "row 3, column 2"),
    ("cf_skip_lines", {"k": 1}, 0),
    ("cf_skip_lines", {"k": 0}, 0),
    ("cf_skip_moves", {"cell": [0, 0], "k": 1}, 0),
    ("count_design", {"design": 1}, 3),
]


def test_hand_scene_answers():
    scene = _hand_scene()
    errs = [e for e in sim.check(scene) if "move count" not in e]     # only 2 moves: below the dataset range
    assert errs == []
    for key, p, want in HAND:
        assert str(answers.answer(key, p, scene)) == str(want), key
        if key == "line_after":          # a 2-move clip can't give 4 "which move" options (real clips have >= 4)
            continue
        q = questions.make(questions.Ctx(scene), random.Random(0), 3, key, dict(p))
        assert str(q["groundtruth"]) == str(want), key
