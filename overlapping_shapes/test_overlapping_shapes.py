"""Tests for overlapping_shapes:  python -m pytest overlapping_shapes/test_overlapping_shapes.py"""
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
from common.video5 import build_sample, _roundtrip  # noqa: E402

SEEDS = range(32)
_CACHE = {}


def _sample(seed):
    if seed not in _CACHE:
        _CACHE[seed] = _roundtrip(build_sample("overlapping_shapes", sim, questions, seed))
    return copy.deepcopy(_CACHE[seed])


# ------------------------------------------------------------ hand-built worlds
def _shape(t, color, x, y, size=60.0, rot=0.0, **kw):
    d = {"type": t, "color": color, "style": "solid", "fill": "plain", "x": x, "y": y, "size": size, "rot": rot}
    d.update(kw)
    return d


def _world(scenario, shapes, events=(), n_frames=200, anim_f=6):
    return {"scenario": scenario, "theme": dict(sim.THEMES[0], seed=1), "board": [40, 100, 1240, 690],
            "zoom": 1.0, "line_w": 4, "alpha": 255 if scenario == "paper" else 100, "pattern": "scatter",
            "shapes": shapes, "anim": "none" if scenario == "glide" else "fade", "anim_f": 0 if scenario == "glide"
            else anim_f, "events": [{"shape": i, "f": f} for i, f in events], "n_frames": n_frames,
            "duration": round(n_frames / sim.FPS, 4)}


def _row():
    # red circle and blue circle overlap; green square far away; yellow star overlaps the blue circle only
    return [_shape("circle", "red", 300, 400), _shape("circle", "blue", 390, 400),
            _shape("square", "green", 900, 400), _shape("star", "yellow", 470, 400, size=55, rot=0.0)]


def test_arrive_by_hand():
    sc = _world("arrive", _row(), [(2, 20), (0, 45), (3, 70), (1, 95)], n_frames=130)
    assert sim.check(sc) == []
    assert [tuple(p) for p in sim.pairs(sc)] == [(0, 1), (1, 3)]
    A = answers.answer
    assert A("n_pairs", {}, sc) == 2
    # green, then red (nothing on the board overlaps it), then yellow (nothing yet), then blue (red + yellow)
    assert A("isolated_arrivals", {}, sc) == 2
    assert A("most_new", {}, sc) == "blue circle"
    assert A("arrival_count", {"x": 1}, sc) == 2
    assert A("pairs_before_last", {}, sc) == 0
    assert A("count_at_T", {"f": 60}, sc) == 2
    assert A("pairs_at_T", {"f": 120}, sc) == 2
    assert A("time_of", {"x": 3}, sc) == "3 s"                    # frame 70 = 2.92 s
    # reversed: blue, yellow (overlaps blue), red (overlaps blue), green -> only green is alone
    assert A("cf_reverse", {}, sc) == 1
    assert A("cf_remove_pairs", {"x": 1}, sc) == 0
    assert A("cf_remove_isolated", {"x": 1}, sc) == 3
    assert A("earlier4", {"cands": [0, 1, 2, 3]}, sc) == "green square"


def test_paper_on_top_by_hand():
    sc = _world("paper", _row(), [(0, 20), (3, 45), (1, 70), (2, 95)], n_frames=130)
    assert sim.check(sc) == []
    # blue is laid last of the three, so it sits on top of both red and yellow
    assert answers.answer("on_top", {"x": 0, "cands": [1, 2, 3]}, sc) == "blue circle"
    assert answers.answer("on_top", {"x": 3, "cands": [0, 1, 2]}, sc) == "blue circle"
    # laid in reverse (green, blue, yellow, red), red and yellow would both cover part of the blue circle
    assert answers.answer("cf_reverse_top", {"x": 1, "cands": [0, 2]}, sc) == "red circle"
    assert answers.answer("cf_reverse_top", {"x": 1, "cands": [0, 3]}, sc) is None


def test_depart_by_hand():
    sc = _world("depart", _row(), [(1, 30), (3, 60), (0, 90)], n_frames=130)
    assert sim.check(sc) == []
    A = answers.answer
    assert A("first_gone", {}, sc) == "blue circle"
    assert A("n_left", {}, sc) == 1
    assert A("lost_pairs", {}, sc) == 2
    assert A("partners_when_gone", {"x": 0}, sc) == 0          # blue (its only partner) has already gone
    assert A("cf_keep", {"x": 1}, sc) == 0                     # blue + green left: no overlap
    assert A("newly_alone", {}, sc) == 0
    assert A("count_at_T", {"f": 72}, sc) == 2


def _glide_world(x1=1100.0, f1=150, spin=0.0):
    st = [_shape("square", "green", 400, 400, size=55), _shape("hexagon", "blue", 700, 400, size=55),
          _shape("circle", "red", 400, 600, size=50), _shape("heart", "pink", 700, 600, size=50),
          _shape("octagon", "gray", 1000, 600, size=50)]
    mv = _shape("circle", "orange", 150, 400, size=45)
    mv["path"] = {"x1": x1, "y1": 400.0, "f0": 20, "f1": f1, "spin": spin}
    return _world("glide", st + [mv], n_frames=200)


def test_glide_by_hand():
    sc = _glide_world()
    assert sim.check(sc) == []
    tt = sim.touches(sc, 5)
    assert [s for s, _, _ in tt] == [0, 1]
    A = answers.answer
    assert A("n_touched", {"m": 5}, sc) == 2
    assert A("first_touch", {"m": 5}, sc) == "green square"
    assert A("last_touch", {"m": 5}, sc) == "blue hexagon"
    assert A("cf_reverse_path", {"m": 5}, sc) == "blue hexagon"
    assert A("mover_dir", {"m": 5}, sc) == "right"
    # 950 px in 130 frames: ~175 px/s. Half speed: by the last frame it has covered 475 px -> x = 625:
    # past the green square, not yet far into the blue hexagon (which starts near x = 600)
    assert A("cf_half_speed", {"m": 5}, sc) == len([1 for _, a, _ in tt if 20 / 24 + 2 * (a - 20 / 24) < 199 / 24])
    assert A("cf_stop", {"m": 5, "f": 60}, sc) == sum(a < 60 / 24 for _, a, _ in tt)
    f_mid = int(round((tt[0][1] + tt[0][2]) / 2 * 24))
    assert A("overlap_at_T", {"m": 5, "f": f_mid}, sc) == 1
    assert A("which_at_T", {"m": 5, "f": f_mid}, sc) == "green square"


def test_check_rejects_slivers_and_near_touches():
    sh = _row()
    sh[2] = _shape("square", "green", 300 + 60 + 55 * 0.7071 + 8, 400, size=55)      # 8 px from the red circle
    errs = sim.check(_world("arrive", sh, [(0, 20), (1, 45), (2, 70), (3, 95)], n_frames=130))
    assert any("nearly touch" in e for e in errs)
    sh = _row()
    sh[1] = _shape("circle", "blue", 300 + 118, 400)          # circles of r 60, 118 apart: a 2 px deep sliver
    errs = sim.check(_world("arrive", sh, [(0, 20), (1, 45), (2, 70), (3, 95)], n_frames=130))
    assert any("sliver" in e for e in errs)


def test_check_rejects_events_too_close():
    sc = _world("arrive", _row(), [(2, 20), (0, 30), (3, 70), (1, 95)], n_frames=130)
    assert any("too close" in e for e in sim.check(sc))
    sc = _world("arrive", _row(), [(2, 6), (0, 45), (3, 70), (1, 95)], n_frames=130)
    assert any("clip edge" in e for e in sim.check(sc))


# ---------------------------------------------------------------- generated
def test_ladder_counts_order_and_min_fps():
    for seed in SEEDS:
        s = _sample(seed)
        assert L5.ladder_errors(s["questions"], "ten") == []
        assert [q["qid"] for q in s["questions"]] == [f"q{i:02d}" for i in range(1, 11)]
        for q in s["questions"]:
            assert q["params"]["min_fps"] > 0
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4
            assert q["derivation_key"] and isinstance(q["params"], dict)


def test_answers_independent_and_check_clean():
    for seed in SEEDS:
        s = _sample(seed)
        assert sim.check(s["params"]) == []
        for q in s["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], s["params"])) == str(q["groundtruth"])
            assert q["options"]["ABCD".index(q["answer"])] == q["answer_text"] == str(q["groundtruth"])


def test_distractors_are_never_correct():
    """For 'which of these' questions, exactly one candidate satisfies the question."""
    for seed in SEEDS:
        s = _sample(seed)
        sc = s["params"]
        labels = {sim.label(x): i for i, x in enumerate(sc["shapes"])}
        for q in s["questions"]:
            if q["derivation_key"] in ("overlap_which", "overlap_none", "on_top", "cf_reverse_top", "passes_over"):
                assert answers.answer(q["derivation_key"], q["params"], sc) is not None
                for o in q["options"]:
                    if o != q["answer_text"] and o in labels:
                        alt = dict(q["params"], cands=[labels[o]])
                        assert answers.answer(q["derivation_key"], alt, sc) is None


def test_scenarios_names_and_uniqueness():
    sigs, scen = set(), set()
    for seed in range(120):
        sc = sim.build(random.Random(seed))
        sigs.add(sim.signature(sc))
        scen.add(sc["scenario"])
        labs = [sim.label(x) for x in sc["shapes"]]
        assert len(set(labs)) == len(labs)
    assert len(sigs) == 120
    assert scen == set(sim.SCENARIOS)


def test_scene_timing_rules():
    for seed in SEEDS:
        sc = _sample(seed)["params"]
        assert sc["n_frames"] <= 240 and abs(render.duration(sc) - sc["duration"]) < 1e-3
        ev = ([e["f"] / 24 for e in sc["events"]] + [(e["f"] + sc["anim_f"] - 1) / 24 for e in sc["events"]]
              if sc["scenario"] != "glide" else sim.glide_events(sc))
        assert min(ev) >= 0.5 and max(ev) <= sc["duration"] - 0.5


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


def test_answers_do_not_use_the_generator_geometry(monkeypatch):
    rows = [_sample(seed) for seed in range(8)]

    def boom(*a, **k):
        raise AssertionError("answers.py called the generator's geometry")
    for name in ("pairs", "neighbours", "max_depth", "touches", "glide_info", "glide_events", "on_board",
                 "polygon", "poly_at", "stable_window"):
        monkeypatch.setattr(sim, name, boom)
    answers._static_masks.cache_clear()
    answers._graph_key.cache_clear()
    answers._trip.cache_clear()
    for r in rows:
        for q in r["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], r["params"])) == str(q["groundtruth"])


def test_deterministic_across_processes():
    code = ("import sys, json; sys.path[:0] = [%r, %r]; import sim, questions; "
            "from common.video5 import build_sample; "
            "print(json.dumps([build_sample('overlapping_shapes', sim, questions, s) for s in range(10)], "
            "sort_keys=True))") % (str(HERE), str(HERE.parent))
    outs = []
    for hs in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                   check=True).stdout)
    assert outs[0] == outs[1]


def test_render_deterministic_clock_clear_and_no_other_text():
    for seed in (1, 2, 3, 4):
        sc = _sample(seed)["params"]
        h = []
        for _ in range(2):
            render._board.cache_clear()
            render._tile.cache_clear()
            m = hashlib.sha256()
            for f in (0, 13, sc["n_frames"] // 2, sc["n_frames"] - 1):
                m.update(np.asarray(render.frame_at(sc, f)).tobytes())
            h.append(m.hexdigest())
        assert h[0] == h[1]
        img = np.asarray(render.frame_at(sc, sc["n_frames"] - 1), dtype=int)
        x0, y0, x1, y1 = sc["board"]
        outside = np.ones(img.shape[:2], bool)
        outside[y0 - 2:y1 + 3, x0 - 2:x1 + 3] = False
        tx0, ty0, tx1, ty1 = sim.TIMER_BOX
        outside[ty0 - 1:ty1 + 2, tx0 - 1:tx1 + 2] = False
        assert np.abs(img[outside] - np.array(sc["theme"]["bg"])).max() == 0      # nothing but the clock
        assert y0 > ty1                                                            # board stays below it


def test_timer_matches_frame():
    import checks
    sc = _sample(5)["params"]
    for f in (0, 12, 37, sc["n_frames"] - 1):
        ok, _ = checks.timer_ok(np.asarray(render.frame_at(sc, f), dtype=np.int16), f)
        assert ok


def test_stale_state_between_seeds():
    a = json.dumps(_roundtrip(build_sample("overlapping_shapes", sim, questions, 17)), sort_keys=True)
    for s in (3, 99, 4):
        build_sample("overlapping_shapes", sim, questions, s)
    assert json.dumps(_roundtrip(build_sample("overlapping_shapes", sim, questions, 17)), sort_keys=True) == a


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_shapes_drawn_where_the_scene_says(seed):
    """Every fully drawn shape colours the pixels well inside its outline."""
    import checks
    sc = _sample(seed)["params"]
    f = sc["n_frames"] - 1 if sc["scenario"] != "depart" else 0
    img = np.asarray(render.frame_at(sc, f), dtype=np.int16)
    bg = checks._board_img(sc)
    for i in sim.statics(sc):
        pr = checks._probe(sc, i, sim.statics(sc))
        if pr is not None:
            assert checks._present(img, bg, pr)
