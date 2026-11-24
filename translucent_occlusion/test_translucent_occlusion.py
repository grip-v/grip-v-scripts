"""Tests for translucent_occlusion: python test_translucent_occlusion.py  (or pytest)

Covers determinism (two processes with different PYTHONHASHSEED; identical raw frames on re-render), the
10-question ladder, scene uniqueness (looks and scenes), min_fps on every question, the tamper test, the clock
text the renderer draws, and three pinned seeds worked out by hand from the per-frame track.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import ImageDraw

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import ladder5 as L  # noqa: E402
from common import video5 as V  # noqa: E402
import sim, questions, answers, render  # noqa: E402

DS = "translucent_occlusion"


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
            render._floor.cache_clear()
            render.sprite.cache_clear()
            render._panels_cached.cache_clear()
            sim._track.cache_clear()
            h = hashlib.sha256()
            for f in range(0, render.n_frames(sc), 3):
                h.update(render.frame_at(sc, f).tobytes())
            hs.append(h.hexdigest())
        assert hs[0] == hs[1]


def test_ladder_minfps_answers_and_unique_looks():
    for seed in range(60):
        s = sample(seed)
        sc, qs = s["params"], s["questions"]
        assert not sim.check(sc), sim.check(sc)
        assert L.ladder_errors(qs, "ten") == []
        looks = [(a["color"], a["shape"]) for a in sc["actors"]]
        looks += [(a["color"], a["changes"][0][2]) if a["changes"][0][1] == "shape" else (a["changes"][0][2], a["shape"])
                  for a in sc["actors"] if a["changes"] and a["changes"][0][1] in ("color", "shape")]
        assert len(set(looks)) == len(looks)
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


def test_clock_text_is_what_is_drawn():
    sc = sample(2)["params"]
    x0, y0, x1, y1 = render.CLOCK_BOX
    for f in (0, 1, 5, 24, 100, sc["frames"] - 1):
        ref = Image_new_like(render.frame_at(sc, f))
        d = ImageDraw.Draw(ref)
        render._clock(d, f)
        got = np.asarray(render.frame_at(sc, f), dtype=np.int16)[y0:y1, x0:x1]
        assert np.abs(got - np.asarray(ref, dtype=np.int16)[y0:y1, x0:x1]).max() == 0
        assert render.clock_text(f) == f"{f * 10 // 24 / 10:.1f} s"
    other = Image_new_like(render.frame_at(sc, 0))
    render._clock(ImageDraw.Draw(other), 100)
    assert np.abs(np.asarray(render.frame_at(sc, 0), dtype=np.int16)[y0:y1, x0:x1]
                  - np.asarray(other, dtype=np.int16)[y0:y1, x0:x1]).max() > 0


def Image_new_like(img):
    return img.copy()


def _ans(sc, key, **p):
    return answers.answer(key, p, sc)


def _hidden_last(sc):
    """Independent count of discs entirely inside a panel on the last frame, from the drawn track."""
    n = 0
    for x, y, r, _, _ in sim.track(sc)[-1]:
        for p in sc["panels"]:
            if p["kind"] == "rect":
                ok = p["x0"] + r <= x <= p["x1"] - r and p["y0"] + r <= y <= p["y1"] - r
            else:
                ok = math.hypot(x - p["cx"], y - p["cy"]) + r <= p["R"]
            if ok:
                n += 1
                break
    return n


def test_pinned_seeds_by_hand():
    # seed 13 (hide_and_stay): 5 shapes, two end up parked inside a panel; with the panels removed all 5 are seen
    sc = sample(13)["params"]
    assert sc["scenario"] == "hide_and_stay" and len(sc["actors"]) == 5
    assert _ans(sc, "n_total") == len(sc["actors"]) == 5
    assert _hidden_last(sc) == 2 and _ans(sc, "n_hidden_end") == 2
    assert _ans(sc, "cf_n_seen_end", tag="nopanel", x=None) == len(sc["actors"])
    # seed 1 (hidden_change): the changed look is the track's last look, the old look its first
    sc = sample(1)["params"]
    tr = sim.track(sc)
    ch = [i for i, a in enumerate(sc["actors"]) if a["changes"] and a["changes"][0][1] in ("color", "shape")]
    assert ch
    for i in ch:
        assert _ans(sc, "change_look", x=i) == f"a {tr[-1][i][3]} {tr[-1][i][4]}"
        assert _ans(sc, "before_look", x=i) == f"a {tr[0][i][3]} {tr[0][i][4]}"
        assert tr[0][i][3:] != tr[-1][i][3:]
    # seed 7: first hide of each shape found by a plain scan of the track; clock = floor(f * 10 / 24) tenths
    sc = sample(7)["params"]
    tr, ps = sim.track(sc), sc["panels"]

    def inside(a):
        x, y, r = a[:3]
        return any((p["x0"] + r <= x <= p["x1"] - r and p["y0"] + r <= y <= p["y1"] - r) if p["kind"] == "rect"
                   else math.hypot(x - p["cx"], y - p["cy"]) + r <= p["R"] for p in ps)

    checked = 0
    for i in range(len(sc["actors"])):
        f = next((g for g in range(1, len(tr)) if inside(tr[g][i]) and not inside(tr[g - 1][i])), None)
        if f is None or tr[0][i][3:] != tr[-1][i][3:]:
            continue
        assert _ans(sc, "hide_time", x=i) == f"{f * 10 // 24 / 10:.1f} s" == render.clock_text(f)
        checked += 1
    assert checked >= 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
