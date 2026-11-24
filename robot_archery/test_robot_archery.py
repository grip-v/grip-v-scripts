"""Tests for robot_archery: python test_robot_archery.py  (or pytest)

Covers determinism (same seed in two processes with different PYTHONHASHSEED
-> identical JSON; identical raw frames on re-render), the 10-question ladder
counts, scene uniqueness, min_fps on every question, the tamper test, the
ring numbers matching what the renderer draws, drawn arrows sitting at the
stored distances, the rule replays on hand-worked games, and two pinned
seeds whose answers were worked out by hand.
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
from common.video5 import font  # noqa: E402
import sim, questions, answers, render  # noqa: E402

DS = "robot_archery"


def sample(seed):
    return V._roundtrip(V.build_sample(DS, sim, questions, seed))


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
            h = hashlib.sha256()
            for f in range(0, render.n_frames(sc), 3):
                h.update(render.frame_at(sc, f).tobytes())
            hs.append(h.hexdigest())
            render._static.cache_clear()
            render._states.cache_clear()
        assert hs[0] == hs[1]


def test_ladder_uniqueness_minfps_answers():
    sigs = set()
    for seed in range(150):
        s = sample(seed)
        sc, qs = s["params"], s["questions"]
        assert not sim.check(sc), sim.check(sc)
        assert L.ladder_errors(qs, "ten") == []
        counts = [sum(q["difficulty_level"] == lv for q in qs) for lv in range(1, 6)]
        assert all(lo <= k <= hi for k, (lo, hi) in zip(counts, L.TEN_RANGES)) and sum(counts) == 10
        for q in qs:
            assert q["params"]["min_fps"] >= 1
            assert len(set(q["options"])) == 4
            assert str(answers.answer(q["derivation_key"], q["params"], sc)) == str(q["groundtruth"])
        sigs.add(json.dumps({k: v for k, v in sc.items() if k != "build_attempts"}, sort_keys=True))
    assert len(sigs) == 150


def test_tamper_flags_exactly_the_corrupted():
    for seed in (5, 6, 7):
        s = sample(seed)
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


def test_labels_match_the_renderer():
    """The scene's label list is exactly the ring numbers the static layer draws, centred in their rings."""
    import numpy as np
    from PIL import ImageDraw
    for seed in (3, 4, 5):
        sc = sample(seed)["params"]
        tg = sc["target"]
        assert [lab["text"] for lab in sc["labels"]] == ["2", "4", "6", "8", "10"]
        assert sc["labels"] == sim.labels(sc)
        for lab in sc["labels"]:
            r = math.hypot(lab["x"] - tg["cx"], lab["y"] - tg["cy"]) / tg["unit"]
            assert abs(r - (int(lab["text"]) - 0.5)) < 1e-6
        img = np.asarray(render._static(render.static_key(sc)), dtype=np.int16)
        blank = render._static(render.static_key(dict(sc, labels=[])))
        for lab in sc["labels"]:
            b = sim.label_box(lab)
            assert b[3] - b[1] >= 24
            x0, y0, x1, y1 = int(b[0]) - 1, int(b[1]) - 1, int(b[2]) + 1, int(b[3]) + 1
            assert np.abs(img[y0:y1, x0:x1] - np.asarray(blank, dtype=np.int16)[y0:y1, x0:x1]).max() > 0
            ref = blank.copy()
            d = ImageDraw.Draw(ref)
            d.rounded_rectangle([b[0] - 4, b[1] - 3, b[2] + 4, b[3] + 3], radius=5, fill=(255, 255, 255),
                                outline=render.INK, width=1)
            d.text((lab["x"], lab["y"]), lab["text"], fill=render.INK, font=font(lab["size"]), anchor=lab["anchor"])
            assert np.abs(img[y0:y1, x0:x1] - np.asarray(ref, dtype=np.int16)[y0:y1, x0:x1]).max() == 0


def test_drawn_arrows_at_stored_distances():
    """The centre dot of every arrow marker in the last frame is at the stored distance and angle."""
    import numpy as np
    for seed in (20, 21, 22):
        sc = sample(seed)["params"]
        img = np.asarray(render.frame_at(sc, sc["frames"] - 1), dtype=np.int16)
        for d, a in zip(sc["draws"], sc["angles"]):
            x, y = sim.target_xy(sc, d, a)
            px = img[int(round(y)), int(round(x))]
            assert np.abs(px - np.array(render.INK)).sum() < 30, (seed, d, a, px)


def test_rule_replays_by_hand():
    # classic: 0 rec, 1 rec, 2 out, 0 rec, 1 out -> 0 wins
    g = sim.play([0, 1, 2], [5.0, 3.0, 4.0, 2.0, 6.0])
    assert [t["record"] for t in g["turns"]] == [True, True, False, True, False]
    assert g["eliminated"] == [2, 1] and g["winner"] == 0
    # stay_on: 0 rec, 0 rec, 0 out, 1 out -> 2 wins
    g = sim.play([0, 1, 2], [5.0, 3.0, 4.0, 6.0], 1, True)
    assert [t["shooter"] for t in g["turns"]] == [0, 0, 0, 1] and g["winner"] == 2
    # two lives: nobody out after one miss each
    g = sim.play([0, 1], [5.0, 6.0, 7.0, 8.0], 2)
    assert g["eliminated"] == [1] and [t["lives_after"] for t in g["turns"]] == [2, 1, 1, 0]
    for order, draws, lives, stay in (([0, 1, 2], [5.0, 3.0, 4.0, 2.0, 6.0], 1, False),
                                      ([0, 1, 2], [5.0, 3.0, 4.0, 6.0], 1, True), ([0, 1], [5.0, 6.0, 7.0, 8.0], 2, False)):
        a = answers.replay(order, draws, lives, stay)
        g = sim.play(order, draws, lives, stay)
        assert a["out"] == g["eliminated"] and a["winner"] == g["winner"]


def _ans(sc, key, **p):
    return answers.answer(key, p, sc)


def test_pinned_seeds_by_hand():
    sc = sample(3)["params"]                 # two_lives: brown, pink, blue (in that queue order)
    assert sc["scenario"] == "two_lives" and sc["draws"] == [1.91, 1.24, 2.62, 0.46, 4.57, 1.55, 6.62, 4.59]
    assert _ans(sc, "winner") == "the brown robot"
    assert _ans(sc, "elim_order") == "blue → pink"
    assert _ans(sc, "records_by", robot=0) == 2 and _ans(sc, "n_records") == 3 and _ans(sc, "n_arrows") == 8
    assert _ans(sc, "ring_of", robot=2, nth=1) == 3
    assert _ans(sc, "cf_rule_winner", rule="one_life") == "the brown robot"
    assert _ans(sc, "cf_rule_first_out", rule="one_life") == "the blue robot"
    sc = sample(17)["params"]                # big_field: orange, blue, green, pink, brown, red
    assert sc["scenario"] == "big_field"
    assert _ans(sc, "winner") == "the orange robot"
    assert _ans(sc, "elim_order") == "green → brown → red → blue → pink"
    assert _ans(sc, "cf_rule_winner", rule="stay") == "the red robot"
    assert _ans(sc, "cf_rule_first_out", rule="stay") == "the orange robot"
    assert _ans(sc, "record_beaten", robot=0, nth=1) == "the blue robot"     # pink 0.97 beats blue's 3.59


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
