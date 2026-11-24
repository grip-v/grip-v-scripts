"""Tests for object_tracking_mot_v2: python test_object_tracking_mot_v2.py  (or pytest)

Covers determinism (same seed in two processes with different PYTHONHASHSEED
-> identical JSON; identical raw frames), the 10-question ladder counts,
scene uniqueness, the tamper test, min_fps on every question, the clock being
the only text, hand-checked physics (wall bounce time, head-on collision,
glancing collision conserving momentum and energy), the what-if integrator
agreeing with the exact event solver, and pinned seeds.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import ladder5 as L  # noqa: E402
from common import video5 as V  # noqa: E402
import sim, questions, answers, render  # noqa: E402

DS = "object_tracking_mot_v2"


def sample(seed):
    return V._roundtrip(V.build_sample(DS, sim, questions, seed))


def test_determinism_across_processes():
    code = ("import sys,json; sys.path[:0]=[r'%s', r'%s']; from common import video5 as V; "
            "import sim, questions, answers; out=[V.build_sample('%s', sim, questions, s) for s in (0, 7, 123)]; "
            "print(json.dumps([out, [[answers.answer(q['derivation_key'], q['params'], o['params']) "
            "for q in o['questions']] for o in out]], sort_keys=True))") % (HERE, HERE.parent, DS)
    outs = []
    for hs in ("1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                   check=True).stdout)
    assert outs[0] == outs[1] and len(outs[0]) > 1000


def test_frames_deterministic():
    for seed in (11, 12):
        sc = sample(seed)["params"]
        hs = []
        for _ in range(2):
            h = hashlib.sha256()
            for f in range(0, render.n_frames(sc), 7):
                h.update(render.frame_at(sc, f).tobytes())
            hs.append(h.hexdigest())
            render._static.cache_clear()
            render.sprite.cache_clear()
            sim._MEMO.clear()
        assert hs[0] == hs[1]


def test_ladder_uniqueness_minfps_answers():
    sigs = set()
    for seed in range(60):
        s = sample(seed)
        sc, qs = s["params"], s["questions"]
        assert L.ladder_errors(qs, "ten") == [], seed
        assert sim.check(sc) == [], seed
        sigs.add(json.dumps(sc, sort_keys=True))
        for q in qs:
            mf = q["params"].get("min_fps")
            assert isinstance(mf, int) and 1 <= mf <= 24
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4
            assert str(answers.answer(q["derivation_key"], q["params"], sc)) == str(q["groundtruth"])
            for k in ("t", "t0", "t1"):
                if k in q["params"]:
                    assert sim.EDGE_S <= q["params"][k] <= sim.duration(sc) - sim.EDGE_S
    assert len(sigs) == 60


def test_tamper_flags_exactly_the_corrupted():
    s = sample(5)
    sc = s["params"]
    bad = copy.deepcopy(s["questions"])
    for i in (0, 4, 9):
        q = bad[i]
        k = (L.LETTERS.index(q["answer"]) + 1) % 4
        q["answer"], q["answer_text"] = L.LETTERS[k], q["options"][k]
        q["groundtruth"] = q["options"][k]
    flagged = [q["qid"] for q in bad
               if str(answers.answer(q["derivation_key"], q["params"], sc)) != str(q["groundtruth"])]
    assert flagged == [bad[0]["qid"], bad[4]["qid"], bad[9]["qid"]]


def test_clock_is_the_only_text():
    assert [render.clock_text(f) for f in (0, 3, 24, 36, 239)] == ["0.0 s", "0.1 s", "1.0 s", "1.5 s", "9.9 s"]
    src = (HERE / "render.py").read_text()
    assert src.count("d.text(") == 1                    # the clock is the only text drawn


def test_hand_checked_physics():
    box = (0, 0, 400, 300)
    # wall: a disc of radius 10 at x = 100 moving right at 100 px/s touches x = 390 after 2.9 s
    r = sim.simulate([{"id": 0, "x": 100, "y": 150, "vx": 100, "vy": 0, "r": 10}], box, 5.0)
    assert abs(r["events"][0][0] - 2.9) < 1e-9 and r["events"][0][3] == "right"
    assert abs(sim.seg_pos(r["segs"][0], 4.0)[0] - (390 - 110)) < 1e-9
    # head-on: equal discs swap velocities; they touch when the 80 px gap closes at 200 px/s -> 0.4 s
    r = sim.simulate([{"id": 0, "x": 100, "y": 150, "vx": 100, "vy": 0, "r": 10},
                      {"id": 1, "x": 200, "y": 150, "vx": -100, "vy": 0, "r": 10}], box, 1.0)
    t, kind, ids, _ = r["events"][0]
    assert kind == "hit" and abs(t - 0.4) < 1e-9
    assert r["segs"][0][-1][3] == -100 and r["segs"][1][-1][3] == 100
    # glancing hit: momentum and kinetic energy are conserved
    r = sim.simulate([{"id": 0, "x": 100, "y": 150, "vx": 120, "vy": 0, "r": 15},
                      {"id": 1, "x": 200, "y": 165, "vx": 0, "vy": 0, "r": 15}], box, 0.9)
    a, b = r["segs"][0][-1], r["segs"][1][-1]
    assert abs(a[3] + b[3] - 120) < 1e-9 and abs(a[4] + b[4]) < 1e-9
    assert abs(a[3] ** 2 + a[4] ** 2 + b[3] ** 2 + b[4] ** 2 - 120 ** 2) < 1e-6


def test_whatif_integrator_agrees():
    """answers.resim (fixed steps + bisection) vs sim.simulate (exact events) on real scenes."""
    worst = 0.0
    for seed in range(30):
        sc = sample(seed)["params"]
        if sc["scenario"] == "orbit":
            continue
        T = sim.duration(sc)
        got = answers.resim(sc)
        for o in sc["objects"]:
            worst = max(worst, math.dist(got[o["id"]], sim.position(sc, o["id"], T)))
    assert worst < 1.0, worst


def test_tracker_follows_identical_objects():
    for seed in range(40):
        s = sample(seed)
        sc = s["params"]
        if sc["scenario"] != "targets":
            continue
        T = answers.Tracks(sc, answers.drawn(sc))
        # every linked track ends where some true object is, and the ringed tracks end at the targets
        last = T.N - 1
        ends = sorted(T.pos[i][last] for i in T.ringed)
        want = sorted(sim.position(sc, i, last / 24) for i in sc["targets"])
        assert all(math.dist(a, b) < 1e-6 for a, b in zip(ends, want)), seed


def test_pinned_seeds():
    got = {}
    for seed in PINNED:
        s = sample(seed)
        got[seed] = (s["params"]["scenario"], [(q["derivation_key"], q["answer_text"]) for q in s["questions"]])
    assert got == PINNED, json.dumps(got)


# Hand check of seed 0 (orbit; arena centre (737, 288.5)): green circle at t = 1.0 s is at
# (763.8 + 330.8 sin(1.962), 331.5 + 101.8 sin(6.556)) = (1069.5, 359.0) -> bottom-right; at 4.0 s
# (597.4, 230.1) -> top-left. Teal square in the last frame (t = 9.25 s): (875.3, 451.0) -> bottom-right.
PINNED: dict = {
    0: ('orbit', [
        ('pattern_of', 'ringed'),
        ('n_objects', '3'),
        ('quadrant_at_t', 'bottom-right'),
        ('quadrant_at_t', 'top-left'),
        ('quad_changes', '6'),
        ('n_walls', '0'),
        ('quadrant_end', 'bottom-right'),
        ('cf_speed_quad', 'bottom-right'),
        ('cf_speed_quad', 'top-right'),
        ('cf_speed_quad', 'top-left'),
    ]),
    1: ('targets', [
        ('n_objects', '5'),
        ('n_targets', '3'),
        ('targets_in_quad_at_t', '1'),
        ('targets_in_quad_at_t', '1'),
        ('untargeted_in_quad_at_t', '0'),
        ('untargeted_in_quad_at_t', '2'),
        ('untargeted_in_quad_at_t', '0'),
        ('targets_in_quad_end', '0'),
        ('targets_in_quad_end', '0'),
        ('cf_delay_targets', '0'),
    ]),
    2: ('colours', [
        ('largest', 'the red square'),
        ('pattern_of', 'solid'),
        ('first_wall', 'top'),
        ('leftmost_at_t', 'the blue diamond'),
        ('quadrant_at_t', 'top-left'),
        ('total_hits', '1'),
        ('quad_changes', '0'),
        ('order_end', 'the red square, then the blue diamond, then the yellow circle'),
        ('cf_remove_quad', 'top-left'),
        ('cf_remove_hits', '0'),
    ]),
    5: ('occluders', [
        ('shape_of', 'square'),
        ('pattern_of', 'solid'),
        ('quadrant_at_t', 'top-left'),
        ('first_wall', 'bottom'),
        ('quadrant_at_t', 'top-right'),
        ('time_first_wall', '2.3 s'),
        ('n_walls', '1'),
        ('quadrant_end', 'top-right'),
        ('quadrant_end', 'top-right'),
        ('cf_speed_quad', 'top-right'),
    ]),
}


if __name__ == "__main__":
    fails = 0
    for name_, fn in list(globals().items()):
        if name_.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name_)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name_, repr(e)[:300])
    sys.exit(1 if fails else 0)
