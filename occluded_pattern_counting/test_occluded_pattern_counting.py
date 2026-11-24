"""Tests for occluded_pattern_counting: python test_occluded_pattern_counting.py  (or pytest)

Covers determinism (two processes with different PYTHONHASHSEED; identical raw frames on re-render), the 10-question
ladder, scene uniqueness, min_fps on every question, the tamper test, and three pinned seeds worked out by hand
from the drawn occluder rectangles.
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

DS = "occluded_pattern_counting"


def sample(seed):
    return V._roundtrip(V.build_sample(DS, sim, questions, seed))


def _sig(sc):
    return json.dumps({k: v for k, v in sc.items() if k != "build_attempts"}, sort_keys=True)


def test_determinism_across_processes():
    code = ("import sys,json; sys.path[:0]=[r'%s', r'%s']; from common import video5 as V; "
            "import sim, questions, answers; out=[V.build_sample('%s', sim, questions, s) for s in (0, 7, 123, 4999)]; "
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
            render._base.cache_clear()
            render.sprite.cache_clear()
            render._occ_tile.cache_clear()
            sim._track.cache_clear()
            h = hashlib.sha256()
            for f in range(0, render.n_frames(sc), 3):
                h.update(render.frame_at(sc, f).tobytes())
            hs.append(h.hexdigest())
        assert hs[0] == hs[1]


def test_ladder_minfps_and_answers():
    for seed in range(60):
        s = sample(seed)
        sc, qs = s["params"], s["questions"]
        assert not sim.check(sc), sim.check(sc)
        assert L.ladder_errors(qs, "ten") == []
        for q in qs:
            assert q["params"]["min_fps"] >= 1
            assert len(set(q["options"])) == 4
            assert str(answers.answer(q["derivation_key"], q["params"], sc)) == str(q["groundtruth"])


def test_unique_scenes():
    assert len({_sig(sim.build(random.Random(seed))) for seed in range(200)}) == 200


def test_tamper_flags_exactly_the_corrupted():
    s = sample(5)
    sc = s["params"]
    bad = copy.deepcopy(s["questions"])
    idx = (0, len(bad) // 2, len(bad) - 1)
    for i in idx:
        q = bad[i]
        k = (L.LETTERS.index(q["answer"]) + 1) % 4
        q["answer"], q["answer_text"] = L.LETTERS[k], q["options"][k]
        q["groundtruth"] = q["options"][k]
    flagged = [q["qid"] for q in bad
               if str(answers.answer(q["derivation_key"], q["params"], sc)) != str(q["groundtruth"])]
    assert flagged == [bad[i]["qid"] for i in idx]


def _ans(sc, key, **p):
    return answers.answer(key, p, sc)


def _hidden_at(sc, rects):
    """Own rect test: objects whose whole disc lies strictly inside some rectangle."""
    return {i for i, o in enumerate(sc["objects"])
            if any(x0 + o["r"] < o["x"] < x1 - o["r"] and y0 + o["r"] < o["y"] < y1 - o["r"]
                   for x0, y0, x1, y1 in rects)}


def _state0(sc, f):
    """Objects whose disc touches no rectangle on frame f (state 0), by a plain nearest-point test."""
    out = set()
    for i, o in enumerate(sc["objects"]):
        ok = True
        for x0, y0, x1, y1 in sim.track(sc)[f]:
            nx, ny = min(max(o["x"], x0), x1), min(max(o["y"], y0), y1)
            if (nx - o["x"]) ** 2 + (ny - o["y"]) ** 2 < o["r"] ** 2:
                ok = False
        if ok:
            out.add(i)
    return out


def test_pinned_seeds_by_hand():
    # seed 5 (gaps): counts come from the drawn objects; empty slots are not objects
    sc = sample(5)["params"]
    assert sc["scenario"] == "gaps" and sc["gaps"]
    assert _ans(sc, "n_total") == len(sc["objects"])
    assert _ans(sc, "n_gaps") == len(sc["gaps"])
    assert _ans(sc, "n_visible_start") == len(_state0(sc, 0)) == len(sc["objects"]) - len(_hidden_at(sc, sim.track(sc)[0]))
    # seed 4 (two_occluders): start count, and the stay what-if hides at the end what was hidden on frame 0
    sc = sample(4)["params"]
    assert sc["scenario"] == "two_occluders" and len(sc["occluders"]) == 2
    assert _ans(sc, "n_visible_start") == len(_state0(sc, 0))
    h0 = len(_hidden_at(sc, sim.track(sc)[0]))
    assert _ans(sc, "cf_hidden_end", tag="stay", x=None) == h0
    assert _ans(sc, "cf_visible_end", tag="stay", x=None) == len(sc["objects"]) - h0
    assert _ans(sc, "n_occluders") == 2
    # seed 9 (odd_one): n_odd counts the objects not of the majority colour; odd_start agrees with frame 0
    sc = sample(9)["params"]
    assert sc["scenario"] == "odd_one"
    cols = [o["color"] for o in sc["objects"]]
    base = max(sorted(set(cols)), key=cols.count)
    assert _ans(sc, "main_color") == base
    assert _ans(sc, "n_odd") == sum(c != base for c in cols)
    assert _ans(sc, "n_visible_start") == len(_state0(sc, 0))
    odd = [i for i, c in enumerate(cols) if c != base]
    if len(odd) == 1:
        assert _ans(sc, "odd_start", x=odd[0]) == ("yes" if odd[0] in _hidden_at(sc, sim.track(sc)[0]) else "no")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
