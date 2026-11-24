"""Tests for multi_agent_chase: python test_multi_agent_chase.py  (or pytest)

Covers determinism (two processes with different PYTHONHASHSEED; identical raw frames on re-render),
the 10-question ladder, scene uniqueness, min_fps on every question, the tamper test, unique colours and
patterns, the clock text the renderer draws, and three pinned seeds worked out by hand.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import ladder5 as L  # noqa: E402
from common import video5 as V  # noqa: E402
import sim, questions, answers, render  # noqa: E402

DS = "multi_agent_chase"


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
            render._static.cache_clear()
            sim._run.cache_clear()
            h = hashlib.sha256()
            for f in range(0, render.n_frames(sc), 3):
                h.update(render.frame_at(sc, f).tobytes())
            hs.append(h.hexdigest())
        assert hs[0] == hs[1]


def test_ladder_minfps_answers_and_unique_balls():
    for seed in range(60):
        s = sample(seed)
        sc, qs = s["params"], s["questions"]
        assert not sim.check(sc), sim.check(sc)
        assert L.ladder_errors(qs, "ten") == []
        assert len({b["color"] for b in sc["balls"]}) == len(sc["balls"])
        assert len({b["pattern"] for b in sc["balls"]}) == len(sc["balls"])
        for q in qs:
            assert q["params"]["min_fps"] >= 1
            assert len(set(q["options"])) == 4
            assert str(answers.answer(q["derivation_key"], q["params"], sc)) == str(q["groundtruth"])


def test_unique_scenes():
    assert len({_sig(sim.build(__import__("random").Random(seed))) for seed in range(200)}) == 200


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


def test_clock_text_is_what_is_drawn():
    sc = sample(2)["params"]
    x0, y0, x1, y1 = render.CLOCK_BOX
    for f in (0, 1, 5, 24, 100, sc["frames"] - 1):
        ref = render._static(render._static_key(sc)).copy()
        d = ImageDraw.Draw(ref)
        d.rounded_rectangle(render.CLOCK_BOX, radius=10, fill=(24, 24, 30))
        d.text(((x0 + x1) / 2, (y0 + y1) / 2), render.clock_text(f), fill=(255, 255, 255),
               font=V.font(render.CLOCK_PX), anchor="mm")
        got = np.asarray(render.frame_at(sc, f), dtype=np.int16)[y0:y1, x0:x1]
        assert np.abs(got - np.asarray(ref, dtype=np.int16)[y0:y1, x0:x1]).max() == 0
        assert render.clock_text(f) == f"{f * 10 // 24 / 10:.1f} s"
    # a different reading must differ from what is drawn
    other = render._static(render._static_key(sc)).copy()
    render._clock(ImageDraw.Draw(other), 100)
    assert np.abs(np.asarray(render.frame_at(sc, 0), dtype=np.int16)[y0:y1, x0:x1]
                  - np.asarray(other, dtype=np.int16)[y0:y1, x0:x1]).max() > 0


def _ans(sc, key, **p):
    return answers.answer(key, p, sc)


def test_pinned_seeds_by_hand():
    # (seed, scenario, n_start, n_end, first victim, frame of first catch, clock at it, catch count)
    pins = [(0, "bigger", 5, 2, "the purple checkered ball", 54, "2.2 s", 3),
            (2, "open", 5, 2, "the black ringed ball", 34, "1.4 s", 3),
            (5, "safe_zone", 9, 6, "the blue striped ball", 31, "1.2 s", 3)]
    for seed, scen, n0, n1, first, f0, clk, nc in pins:
        sc = sample(seed)["params"]
        assert sc["scenario"] == scen and len(sim.catches(sc)) == nc
        cs = sim.catches(sc)
        assert cs[0][0] == f0 and sim.label(sc["balls"][cs[0][2]]) == first
        assert _ans(sc, "n_start") == n0 and _ans(sc, "n_end") == n1
        assert _ans(sc, "first_caught") == first
        assert _ans(sc, "first_catch_time") == clk
        assert _ans(sc, "catch_time", x=cs[0][2]) == clk
        assert render.clock_text(f0) == clk


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
