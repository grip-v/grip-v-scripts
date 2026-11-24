"""Tests for colored_route_puzzle.  Run:  python -m pytest test_colored_route_puzzle.py -q"""
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
import sim, questions, answers, render, frame_check  # noqa: E402

DS = "colored_route_puzzle"
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
    assert outs[0] == outs[1] and len(outs[0]) > 1000


def test_render_deterministic():
    scene = _sample(3)["params"]
    idx = (0, 30, scene["n_frames"] // 2, scene["n_frames"] - 1)
    h = [hashlib.sha1(np.asarray(render.frame_at(scene, i)).tobytes()).hexdigest() for i in idx]
    render._static.cache_clear()
    render.marker_sprite.cache_clear()
    render.ring_sprite.cache_clear()
    render._draw_path.cache_clear()
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


def test_uniqueness_of_scenes_routes_and_stops():
    sigs = set()
    for s in SEEDS:
        sc = _sample(s)["params"]
        sigs.add(json.dumps(sc, sort_keys=True))
        assert len({r["color"] for r in sc["routes"]}) == len(sc["routes"])
        assert len({st["shape"] for st in sc["stops"]}) == len(sc["stops"])
    assert len(sigs) == len(SEEDS)


def test_all_scenarios_appear_and_follow_their_rules():
    seen = set()
    for s in SEEDS:
        sc = _sample(s)["params"]
        seen.add(sc["scenario"])
        dead = sum(1 for r in sc["routes"] if r["b"] is None)
        assert (dead > 0) == (sc["scenario"] == "dead_ends")
        assert any(r["erase"] for r in sc["routes"]) == (sc["scenario"] == "erase")
        assert (sc["token"] is not None) == (sc["scenario"] == "traveller")
    assert seen == set(sim.SCENARIOS)


def test_timing_limits():
    for s in SEEDS:
        sc = _sample(s)["params"]
        fr = sim.event_frames(sc)
        assert sc["n_frames"] <= 240
        assert fr[0] >= 12 and sc["n_frames"] - 1 - fr[-1] >= 12
        assert all(b - a >= 2 for a, b in zip(fr, fr[1:]))


def test_only_text_is_the_clock_and_it_is_big_enough():
    for k in range(0, 100, 7):
        x0, y0, x1, y1 = font(sim.TIMER_FONT).getbbox(sim.clock_text(k))
        assert y1 - y0 >= 24
    for s in range(20):
        sc = _sample(s)["params"]
        assert sc["labels"] == []
        assert sc["timer"] == {"box": list(sim.TIMER_BOX), "size": sim.TIMER_FONT}
    assert render.timer_text(23) == "0.9 s" and render.timer_text(24) == "1.0 s" and render.timer_text(239) == "9.9 s"
    assert sim.frames_showing(10) == (24, 26)


def test_track_matches_pixels():
    """The renderer's track is what it draws: route colour on the drawn path, ink at every stop."""
    for s in (2, 5, 7):
        sc = _sample(s)["params"]
        i = sc["n_frames"] - 1
        img = np.asarray(render.frame_at(sc, i), dtype=int)
        tr = render.track(sc, i)
        for r, t in zip(sc["routes"], tr["routes"]):
            if not t["pts"]:
                continue
            pts, _ = frame_check.samples(sc, r)
            ok = frame_check.inked(img.astype(np.int16), pts, r["rgb"])
            assert ok.mean() > {"solid": 0.95, "dashed": 0.55, "dotted": 0.35}[r["style"]]
        for m in render.stop_marks(sc):
            shp, iou = frame_check.read_shape(sc, img, m)
            assert shp == m["shape"] and iou > 0.8


def test_geometry_hand_checked():
    r = {"nodes": [[2, 2], [3, 2], [4, 2], [4, 3], [4, 4], [5, 4]]}
    assert sim.n_corners(r) == 2 and sim.length_steps(r) == 5
    assert answers._corners([(0, 0), (10, 0), (20, 0), (20, 10), (30, 10)]) == 2
    assert answers.events_from_lengths([0, 0, 0, 5, 10, 10, 10, 4, 0, 0], 10) == (2, 4, 6, 8)


def test_edge_seed_hand_checked():
    # seed 2 (sequential; checked by eye on the rendered clip): 5 routes, 4 stops. The star stop has
    # 3 routes (blue dotted to the diamond, lime dotted to the square, purple dashed to the circle);
    # the green route (circle -> square, 42 steps) is the longest and turns 9 corners; blue is drawn
    # first, then lime, purple, green, pink, so the circle and the square first become connected when
    # the purple route completes (circle-star-square). Without the lime route the star still reaches
    # all 3 other stops (via the circle-square green route).
    sc = _sample(2)["params"]
    assert sc["scenario"] == "sequential" and len(sc["routes"]) == 5 and len(sc["stops"]) == 4
    sid = {s["shape"]: s["id"] for s in sc["stops"]}
    rid = {r["color"]: r["id"] for r in sc["routes"]}
    assert answers.answer("degree", {"stop": sid["star"]}, sc) == 3
    assert answers.answer("longest", {}, sc) == "the green route"
    assert answers.answer("corners", {"route": rid["green"]}, sc) == 9
    assert answers.answer("corners", {"route": rid["lime"]}, sc) == 5
    assert answers.answer("first_drawn", {}, sc) == "the blue route"
    assert answers.answer("connect_by", {"pair": [sid["circle"], sid["square"]]}, sc) == "the purple route"
    assert answers.answer("cf_remove_reach", {"route": rid["lime"], "stop": sid["star"]}, sc) == 3
    assert questions.first_connect(sc, (sid["circle"], sid["square"]))[0] == rid["purple"]


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


def test_rendered_clip_passes_frame_check(tmp_path):
    run_generate(DS, HERE, sim, questions, render, ["--n", "1", "--seed", "11", "--output-dir", str(tmp_path)])
    s = json.loads((tmp_path / "annotations.jsonl").read_text().splitlines()[0])
    frames = frame_check.decode(tmp_path / s["media_path"])
    assert frame_check.check_clip(s["params"], s["questions"], frames) == []
    # a wrong stored answer is caught on the decoded frames too
    q = json.loads(json.dumps(s["questions"][0]))
    q["groundtruth"] = next(o for o in q["options"] if o != q["answer_text"])
    assert frame_check.check_clip(s["params"], [q], frames) != []
