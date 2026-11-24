"""Tests for rainbow_bagel: python -m pytest rainbow_bagel/test_rainbow_bagel.py"""
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import answers  # noqa: E402
import questions  # noqa: E402
import render  # noqa: E402
import sim  # noqa: E402
from common import ladder5 as L5  # noqa: E402
from common.video5 import build_sample  # noqa: E402

SEEDS = range(10)


def _sample(seed):
    return json.loads(json.dumps(build_sample("rainbow_bagel", sim, questions, seed)))


def test_deterministic_same_process():
    for seed in (0, 7):
        assert _sample(seed) == _sample(seed)


def test_deterministic_across_hash_seeds():
    code = ("import sys, json; sys.path[:0] = [%r, %r]; import sim, questions; "
            "from common.video5 import build_sample; "
            "print(json.dumps([build_sample('rainbow_bagel', sim, questions, s) for s in (3, 11)], sort_keys=True))"
            % (str(HERE), str(HERE.parent)))
    outs = []
    for hs in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                   check=True).stdout)
    assert outs[0] == outs[1]


def test_render_deterministic():
    sc = _sample(5)["params"]
    for f in (0, sc["gold_frame"], sc["n_frames"] - 1):
        a = hashlib.sha1(render.frame(sc, f / 24).tobytes()).hexdigest()
        b = hashlib.sha1(render.frame(json.loads(json.dumps(sc)), f / 24).tobytes()).hexdigest()
        assert a == b


def test_ladder_min_fps_and_answers():
    for seed in SEEDS:
        s = _sample(seed)
        qs = s["questions"]
        assert L5.ladder_errors(qs, "ten") == []
        for q in qs:
            assert 0 < q["params"]["min_fps"] <= sim.MAX_TRACK_FPS
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4
            assert str(answers.answer(q["derivation_key"], q["params"], s["params"])) == str(q["groundtruth"])
        assert sim.check(s["params"]) == []
        assert s["params"]["n_frames"] <= 240


def test_unique_scenes():
    sigs = {json.dumps(_sample(s)["params"], sort_keys=True) for s in SEEDS}
    assert len(sigs) == len(SEEDS)


def test_tamper_detected():
    s = _sample(2)
    for q in s["questions"][:3]:
        q["groundtruth"] = next(o for o in q["options"] if o != q["answer_text"])
    flagged = sum(1 for q in s["questions"]
                  if str(answers.answer(q["derivation_key"], q["params"], s["params"])) != str(q["groundtruth"]))
    assert flagged == 3


def test_clock_is_the_only_text_and_matches_renderer():
    for seed in SEEDS:
        sc = _sample(seed)["params"]
        assert sc["text"]["labels"] == []
        assert tuple(sc["text"]["timer"]["box"]) == render.layout(sc)["clock_box"] == sim.TIMER_BOX
    assert [sim.clock_text(f) for f in (0, 2, 3, 23, 24, 239)] == ["0.0 s", "0.0 s", "0.1 s", "0.9 s", "1.0 s",
                                                                   "9.9 s"]


def test_gold_piece_needs_tracking():
    """Review fix: the gold piece is NOT gold at the start; it turns gold mid-flight, far from its slice,
    and every piece is completely hidden behind the screen for at least MIN_HIDE frames."""
    for seed in SEEDS:
        sc = _sample(seed)["params"]
        m = sc["marked"]
        assert all(render.fill_of(sc, i, 0) != sim.GOLD for i in range(len(sc["pieces"])))
        assert render.fill_of(sc, m, sc["gold_frame"] - 1) != sim.GOLD
        assert render.fill_of(sc, m, sc["gold_frame"]) == sim.GOLD
        p = sc["pieces"][m]
        assert sc["gold_frame"] >= p["lift"] + sim.LIFT_F + 12
        rest = sim.rest_xy(sc, p, sim.SPREAD * p["ring"])
        x, y, _ = sim.state(sc, m, sc["gold_frame"])
        assert math.hypot(x - rest[0], y - rest[1]) >= sim.GOLD_AWAY
        hid = sim.hidden_mask(sc)
        assert hid.sum(0).min() >= sim.MIN_HIDE


def test_pieces_of_a_ring_look_alike():
    sc = _sample(4)["params"]
    for ring in range(1, sc["rings"] + 1):
        ids = [i for i, p in enumerate(sc["pieces"]) if p["ring"] == ring and i != sc["marked"]]
        cols = {render.fill_of(sc, i, sc["n_frames"] - 1) for i in ids}
        shapes = {tuple(sim.local_pts(sc["pieces"][i]).round(6).ravel()) for i in ids}
        assert len(cols) == 1 and len(shapes) == 1


def test_geometry_hand_checked():
    # hole radius 1: ring 1 runs from 1 to 2 units -> a quarter is 3*pi/4
    assert abs(sim.piece_area(1, 1, 4) - 3 * math.pi / 4) < 1e-12
    # hole radius 2, ring 3 runs from 4 to 5 -> a sixth is 9*pi/6
    assert abs(sim.piece_area(2, 3, 6) - 1.5 * math.pi) < 1e-12
    # centroid of a half annulus r=1..2: 4(R^3-r^3)/(3 pi (R^2-r^2)) = 28/(9 pi)
    r, R, half, d = sector(1, 1.0, 1, 2)
    assert abs(d - 28 / (9 * math.pi)) < 1e-4
    assert questions.safe_area(2.32) and not questions.safe_area(2.353) and not questions.safe_area(2.347)
    assert questions.fps_gap(8) == 4 and questions.fps_gap(25) == 1 and questions.fps_window(12) == 2


def sector(r0, u, ring, n):
    return sim.sector(r0, u, ring, n)


def test_hand_checked_seed():
    """Seed 0 checked by hand from the scene: counts, the tray-comparison wording and areas."""
    s = _sample(0)
    sc = s["params"]
    n, k, r0 = sc["slices"], sc["rings"], sc["hole"]
    assert len(sc["pieces"]) == n * k
    by_tray = {t: [p for p in sc["pieces"] if p["tray"] == t] for t in sc["trays"]}
    for t, ps in by_tray.items():
        area = sum(math.pi * ((r0 + p["ring"]) ** 2 - (r0 + p["ring"] - 1) ** 2) / n for p in ps)
        if questions.safe_area(area):
            assert answers.answer("tray_area", {"tray": t}, sc) == f"{area:.1f}"
        assert answers.answer("tray_count", {"tray": t}, sc) == len(ps)
    if len(sc["trays"]) == 2:
        a, b = (len(by_tray[t]) for t in sc["trays"])
        want = "same number" if a == b else (f"left, by {a - b}" if a > b else f"right, by {b - a}")
        assert answers.answer("tray_more_pieces", {}, sc) == want


def test_tie_wording():
    assert questions._cmp_two({"left": 5, "right": 5}, random.Random(0))[0] == "same number"
    assert questions._cmp_two({"left": 3, "right": 7}, random.Random(0))[0] == "right, by 4"
    assert questions._cmp_many({"left": 4, "middle": 4, "right": 2})[0] == "a tie for the most"
    assert answers._cmp({"left": 4, "middle": 4, "right": 2}) == "a tie for the most"
    assert answers._cmp({"left": 6, "right": 6}) == "same number"


def test_original_setup_and_scenarios():
    """Every clip: 3-4 rings cut into quarters, two trays; the scenario sets the hole, the lift-off
    pattern and how long pieces hide; the gold piece is from the innermost or outermost ring."""
    seen = set()
    for seed in range(24):
        sc = _sample(seed)["params"]
        seen.add(sc["scenario"])
        assert sc["slices"] == 4 and sc["n_trays"] == 2 and sc["rings"] in (3, 4)
        assert sc["hole"] == (2 if sc["scenario"] == "wide_hole" else 1)
        lifts = [p["lift"] for p in sc["pieces"]]
        if sc["scenario"] == "one_by_one":
            assert len(set(lifts)) == len(lifts)
        hid = sim.hidden_mask(sc).sum(0)
        assert hid.min() >= (21 if sc["scenario"] == "long_hide" else sim.MIN_HIDE)
        if sc["scenario"] != "long_hide":
            assert hid.max() <= 20
        g = sc["pieces"][sc["marked"]]["ring"]
        assert g == (1 if sc["gold_ring"] == "inner" else sc["rings"])
    assert seen == set(sim.SCENARIOS)
