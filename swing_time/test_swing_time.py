"""Tests for swing_time: python test_swing_time.py  (or pytest test_swing_time.py)

Covers determinism (same seed in two processes with different PYTHONHASHSEED
-> identical JSON; identical raw frames), the 10-question ladder counts,
scene uniqueness, the tamper test, min_fps on every question, and
hand-checked edge cases for the swing rules and one pinned seed.
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

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import ladder5 as L  # noqa: E402
from common import video5 as V  # noqa: E402
import sim, questions, answers, render  # noqa: E402

DS = "swing_time"


def sample(seed):
    return V._roundtrip(V.build_sample(DS, sim, questions, seed))


def test_determinism_across_processes():
    code = ("import sys,json; sys.path[:0]=[r'%s', r'%s']; from common import video5 as V; "
            "import sim, questions; print(json.dumps([V.build_sample('swing_time', sim, questions, s) "
            "for s in (0, 7, 123, 4999)], sort_keys=True))") % (HERE, HERE.parent)
    outs = []
    for hs in ("1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                   check=True).stdout)
    assert outs[0] == outs[1] and len(outs[0]) > 1000


def test_frames_deterministic():
    sc = sample(11)["params"]
    n = render.n_frames(sc)
    for _ in range(2):
        h = hashlib.sha256()
        for f in range(0, n, 7):
            h.update(render.frame(sc, f / render.FPS).tobytes())
        if _ == 0:
            first = h.hexdigest()
    assert h.hexdigest() == first


def test_ladder_uniqueness_minfps_answers():
    sigs = set()
    for seed in range(200):
        s = sample(seed)
        sc, qs = s["params"], s["questions"]
        assert L.ladder_errors(qs, "ten") == [], seed
        assert sim.check(sc) == [], seed
        sigs.add(json.dumps(sc, sort_keys=True))
        for q in qs:
            assert isinstance(q["params"].get("min_fps"), int) and 1 <= q["params"]["min_fps"] <= 24
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4
            assert str(answers.answer(q["derivation_key"], q["params"], sc)) == str(q["groundtruth"])
        assert render.n_frames(sc) <= 240
        ev = sim.events(sc)
        assert ev[0][0] >= 12 and render.n_frames(sc) - ev[-1][1] >= 12
    assert len(sigs) == 200


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


def test_wrong_option_labels():
    """Every wrong option carries a known mistake tag that the frames re-derive; a tampered tag is caught."""
    import run
    for seed in range(60):
        s = sample(seed)
        for q in s["questions"]:
            assert run.label_errors(q, s["params"]) == [], (seed, q["qid"])
    s = sample(3)
    q = copy.deepcopy(s["questions"][7])            # cf_short_rope with answer "no route" (see test_pinned_seed)
    assert q["option_errors"] == [None, "claims_a_route", "claims_a_route", "claims_a_route"]
    k = next(i for i, t in enumerate(q["option_errors"]) if t)
    q["option_errors"][k] = "off_by_one"
    assert run.label_errors(q, s["params"]) != []


def _world(w, h, posts, cap=5.0, pits=()):
    return {"w": w, "h": h, "cap": cap, "posts": [list(p) for p in posts], "pits": [list(p) for p in pits],
            "start": [0, 0], "goal": [w - 1, h - 1]}


def _both(src, dst, post, w, h, posts):
    a = sim.swing_dir(src, dst, post, [tuple(p) for p in posts], w, h)
    b = answers._legal(src, dst, post, [tuple(p) for p in posts], w, h, 4 * 25)
    assert a == b, (a, b)
    return a


def test_hand_checked_swing_rules():
    # half turn about (2,2) from square (0,1) [centre (0.5,1.5)] to square (3,2) [centre (3.5,2.5)], rope 1.58
    assert _both((0, 1), (3, 2), (2, 2), 5, 5, [(2, 2)]) == 1            # free: counter-clockwise (under the post)
    assert _both((0, 1), (3, 2), (2, 2), 5, 5, [(2, 2), (2, 1)]) == -1   # (2,1) is under it -> go clockwise, over the top
    assert _both((0, 1), (3, 2), (2, 2), 5, 5, [(2, 2), (2, 1), (2, 3)]) == 0   # blocked both ways
    # (3,1) and (1,3) are both inside the rope (1.41 < 1.58), one under and one over the post: blocked both ways
    # (a post can never sit exactly on a rope circle: rope^2 is always k + 1/2 on this lattice)
    assert _both((0, 1), (3, 2), (2, 2), 5, 5, [(2, 2), (3, 1), (1, 3)]) == 0
    # quarter-ish turn that would swing past x < 0: post (1,2), from (0,3) to (0,0), rope 1.58, through 180 deg
    assert _both((0, 3), (0, 0), (1, 2), 3, 4, [(1, 2)]) == 0
    # different distances are never a swing
    assert answers._legal((0, 0), (3, 3), (1, 1), [(1, 1)], 5, 5, 100) == 0
    # rope cap: rope sqrt(26)/2 = 2.55 is allowed under 2.7 and refused under 2.4
    w = _world(6, 4, [(3, 1)], cap=2.7)
    w["start"], w["goal"] = [0, 0], [5, 1]
    sc = {"w": 6, "h": 4, "cap": 2.7, "posts": [{"colour": "red", "shape": "circle", "xy": [3, 1]}], "pits": [],
          "start": [0, 0], "goal": [5, 1]}
    assert sim.solve(w) is not None and abs(answers.best_route(sc)[0] - sim.solve(w)["cost"]) < 1e-9
    assert abs(sim.solve(w)["cost"] - 26 ** 0.5 / 2) < 1e-9
    w["cap"], sc["cap"] = 2.4, 2.4
    assert sim.solve(w) is None and answers.best_route(sc) == (None, None)
    # a pit on the goal square: no route
    w["cap"], w["pits"] = 2.7, [[5, 1]]
    assert sim.solve(w) is None


def test_pinned_seed():
    """Seed 3 (bottom_top), checked by hand against its rendered frames."""
    s = sample(3)
    sc = s["params"]
    assert sc["scenario"] == "bottom_top"
    assert [x["post"] for x in sc["swings"]] == PINNED_ROUTE
    assert [q["answer_text"] for q in s["questions"]] == PINNED_ANSWERS


# Hand check (frames at mid-swing): start (0,6). Pink triangle at corner (1,3): rope sqrt(0.5^2+3.5^2)=3.54,
# lands on (3,5) clockwise. Yellow triangle (7,4): rope 3.81, lands on (3,2) counter-clockwise passing left of
# it. Grey square (2,2): rope 1.58, lands on the goal (1,0) clockwise. Total 3.54+3.81+1.58 = 8.92. With ropes
# capped at 3.2 no post is in reach of the start (nearest is pink at 3.54) -> no route.
PINNED_ROUTE = ["pink", "yellow", "grey"]
PINNED_ANSWERS = ["the pink triangle post", "the yellow triangle post", "the pink triangle post", "4.4 s", "3.81",
                  "2", "2", "no route", "9.89", "4"]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
