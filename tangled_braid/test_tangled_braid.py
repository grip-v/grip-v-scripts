"""Tests for tangled_braid.  Run:  python -m pytest test_tangled_braid.py -q"""
from __future__ import annotations
import copy
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
from common.video5 import build_sample, font, hex_rgb, run_generate, run_validate  # noqa: E402
import sim, questions, answers, render  # noqa: E402

DS = "tangled_braid"
SEEDS = range(0, 200)


def _sample(seed):
    return json.loads(json.dumps(build_sample(DS, sim, questions, seed)))


def test_same_seed_same_sample():
    for s in range(20):
        assert _sample(s) == _sample(s)


def test_determinism_across_processes_and_hash_seeds():
    code = ("import sys, json; sys.path.insert(0, %r); sys.path.insert(0, %r);"
            "from common.video5 import build_sample; import sim, questions;"
            "print(json.dumps([build_sample('%s', sim, questions, s) for s in range(40)], sort_keys=True))"
            % (str(HERE), str(HERE.parent), DS))
    outs = []
    for hs in ("1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                   check=True).stdout)
    assert outs[0] == outs[1]


def test_render_deterministic():
    scene = _sample(3)["params"]
    idx = (0, scene["steps"][1]["m"], scene["steps"][2]["m"] + 3, scene["n_frames"] - 1)
    h = [hashlib.sha1(np.asarray(render.frame_at(scene, i)).tobytes()).hexdigest() for i in idx]
    render._static.cache_clear()
    render._geom.cache_clear()
    h2 = [hashlib.sha1(np.asarray(render.frame_at(scene, i)).tobytes()).hexdigest() for i in idx]
    assert h == h2


def test_ladder_counts_order_and_min_fps():
    for s in SEEDS:
        qs = _sample(s)["questions"]
        assert L5.ladder_errors(qs, "ten") == []
        for q in qs:
            assert q["params"]["min_fps"] in questions.FPS_STEPS
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4
            if q["difficulty_level"] >= 2:
                assert q["requires_motion"]


def test_independent_answers_agree_and_check_passes():
    for s in SEEDS:
        smp = _sample(s)
        assert sim.check(smp["params"]) == []
        for q in smp["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], smp["params"])) == str(q["groundtruth"])


def test_uniqueness_of_scenes_and_strands():
    sigs = set()
    for s in SEEDS:
        sc = _sample(s)["params"]
        sigs.add(sim.signature(sc))
        cols = [x["color_name"] for x in sc["strands"]]
        assert len(set(cols)) == len(cols)
    assert len(sigs) == len(SEEDS)


def test_all_scenarios_appear_and_follow_their_rules():
    seen = set()
    for s in SEEDS:
        sc = _sample(s)["params"]
        seen.add(sc["scenario"])
        assert sim.scenario_errors(sc) == []
        if sc["scenario"] == "pure":
            assert sim.final_order(sc) == sc["start"]
    assert seen == set(sim.SCENARIOS)


def test_timing_limits():
    for s in SEEDS:
        sc = _sample(s)["params"]
        assert sc["n_frames"] <= 240
        assert sc["steps"][0]["a"] >= 12
        last = sc["steps"][-1]
        assert sc["n_frames"] - 1 - (last["a"] + last["c"]) >= 12
        ms = [st["m"] for st in sc["steps"]]
        assert all(b - a >= 12 for a, b in zip(ms, ms[1:]))


def test_timer_is_the_only_text_and_readable():
    sc = _sample(5)["params"]
    assert sc["labels"] == []
    x0, y0, x1, y1 = font(sc["timer"]["size"]).getbbox(sim.clock_text(37))
    assert y1 - y0 >= 24
    assert sim.clock_text(sim.timer_tenths(23)) == "0.9 s" and sim.timer_tenths(24) == 10
    assert sim.frames_showing(10) == (24, 26)


def test_track_matches_drawn_pixels():
    """At a still moment every strand's colour is drawn where track() says it
    leaves the bar."""
    for seed in (2, 7, 11):
        sc = _sample(seed)["params"]
        lo, hi = sim.quiet_spans(sc)[2]
        f = (lo + hi) // 2
        img = np.asarray(render.frame_at(sc, f), dtype=int)
        tr = render.track(sc, f)
        lay = sc["layout"]
        for name, u in tr["head"].items():
            c = np.array(hex_rgb(sim.strand(sc, name)["color"]))
            a = int(lay["loom"] - lay["grow"] * 2)
            best = 1e9
            for off in range(0, 40, 2):
                aa = int(lay["loom"] - lay["grow"] * (2 + off))
                px = img[aa, int(u) - 6:int(u) + 7] if lay["orient"] == "v" else img[int(u) - 6:int(u) + 7, aa]
                best = min(best, np.abs(px - c).sum(axis=1).min())
            assert best < 30, (seed, name)
            assert a > 0


def test_replay_hand_checked():
    # four strands A B C D; crossings at places 0 (low over), 1 (high over), 0 (high over)
    plan = [(0, 0), (1, 1), (0, 1)]
    final, cr = answers._play(["A", "B", "C", "D"], plan)
    # 1: A over B -> B A C D; 2: C over A -> B C A D; 3: C over B -> C B A D
    assert final == ["C", "B", "A", "D"]
    assert cr == [("A", "B", "A"), ("A", "C", "C"), ("B", "C", "C")]
    assert answers._net(cr, "C", "A") == 1 and answers._net(cr, "B", "C") == -1
    # leaving out crossing 2: A over B, then place 0 high over: B A -> A over B again -> A B C D
    f2, _ = answers._play(["A", "B", "C", "D"], plan[:1] + plan[2:])
    assert f2 == ["A", "B", "C", "D"]


def test_edge_seed_hand_checked():
    # seed 0 (pauses, checked by eye on the rendered last frame): 7 strands, four crossings
    # one at a time at places 3, 4, 4, 5 (1-based); gray goes over purple, then blue over
    # purple, purple back over blue, blue over brown; red stays rightmost.
    sc = _sample(0)["params"]
    M = answers.measure(sc)
    assert sc["scenario"] == "pauses"
    assert [e["place"] for e in M["events"]] == [2, 3, 3, 4]
    assert [st["pause"] for st in sc["steps"]] == [False, False, True, True]
    assert answers.answer("first_cross_of", {"x": "gray banded"}, sc) == "it passes over the purple strand"
    assert answers.answer("partner_seq", {"x": "purple dashed"}, sc) == "gray, then blue, then blue"
    assert answers.answer("net_over", {"a": "blue solid", "b": "purple dashed"}, sc) == "0"
    assert answers.answer("end_edge", {"end": "high"}, sc) == "red"
    assert answers.answer("n_crossings", {}, sc) == 4
    assert answers.answer("cross_time", {"a": "gray banded", "b": "purple dashed"}, sc) == "1.0 s"


def test_frame_check_on_rendered_frames_and_a_wrong_scene():
    import checks
    sc = _sample(4)["params"]
    qs = _sample(4)["questions"]
    frames = [np.asarray(render.frame_at(sc, i), dtype=np.int16) for i in range(sc["n_frames"])]
    assert checks.check_clip(sc, qs, frames) == []
    bad = copy.deepcopy(sc)
    bad["steps"][1]["moves"][0]["over"] = "low" if bad["steps"][1]["moves"][0]["over"] == "high" else "high"
    bad_qs = _sample(4)["questions"]
    errs = checks.check_clip(bad, bad_qs, frames)
    assert any("step 1: over strand read" in e for e in errs)


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
