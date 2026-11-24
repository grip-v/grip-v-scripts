"""Tests for vessels_and_clocks: python test_vessels_and_clocks.py  (or pytest)

Covers determinism (same seed in two processes with different PYTHONHASHSEED
-> identical JSON; identical raw frames on re-render), the 10-question ladder
counts, scene uniqueness, min_fps on every question, the tamper test, the
ruler numbers matching what the renderer draws, the drawn water heights
matching the flow laws, a hand-built scene with hand-computed answers, and
pinned seeds.
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
from common.video5 import font  # noqa: E402
import sim, questions, answers, render  # noqa: E402

DS = "vessels_and_clocks"


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
        assert hs[0] == hs[1]


def test_ladder_uniqueness_minfps_answers():
    sigs = set()
    for seed in range(150):
        s = sample(seed)
        sc, qs = s["params"], s["questions"]
        assert L.ladder_errors(qs, "ten") == [], seed
        assert sim.check(sc) == [], seed
        assert sc["scenario"] in sim.SCENARIOS
        sigs.add(json.dumps({k: v for k, v in sc.items() if k != "build_attempts"}, sort_keys=True))
        for q in qs:
            mf = q["params"].get("min_fps")
            assert isinstance(mf, int) and 1 <= mf <= 24
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4
            assert str(answers.answer(q["derivation_key"], q["params"], sc)) == str(q["groundtruth"])
        assert render.n_frames(sc) <= sim.MAX_FRAMES
        tl = sim.t_last(sc)
        for e in sim.events(sc):                       # question events stay inside the clip, 2+ frames apart
            assert sim.EDGE_S <= e["t"] <= tl - sim.EDGE_S
        ts = [e["t"] for e in sim.events(sc)]
        assert all(b - a >= sim.GAP for a, b in zip(ts, ts[1:]))
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
    """The scene's label list is exactly the ruler numbers the static layer draws, at the mark heights."""
    import numpy as np
    from PIL import ImageDraw
    for seed in (3, 4, 5):
        sc = sample(seed)["params"]
        lay = sc["layout"]
        assert [lab["text"] for lab in sc["labels"]] == ["0", "2", "4", "6", "8", "10"]
        assert sc["labels"] == sim.labels(sc)
        assert all(lab["y"] == lay["bottom"] - int(lab["text"]) * lay["M"] for lab in sc["labels"])
        img = np.asarray(render._static(json.dumps(sc, sort_keys=True)), dtype=np.int16)
        blank = render._static(json.dumps(dict(sc, labels=[]), sort_keys=True))
        for lab in sc["labels"]:
            b = sim.label_box(lab)
            assert b[3] - b[1] >= 24                                   # glyphs >= 24 px tall
            ref = blank.copy()
            ImageDraw.Draw(ref).text((lab["x"], lab["y"]), lab["text"], fill=tuple(sc["style"]["ink"]),
                                     font=font(lab["size"]), anchor=lab["anchor"])
            x0, y0, x1, y1 = int(b[0]) - 3, int(b[1]) - 3, int(b[2]) + 3, int(b[3]) + 3
            assert np.abs(img[y0:y1, x0:x1] - np.asarray(ref, dtype=np.int16)[y0:y1, x0:x1]).max() == 0
            assert np.abs(img[y0:y1, x0:x1] - np.asarray(blank, dtype=np.int16)[y0:y1, x0:x1]).max() > 0


def test_drawn_levels_follow_the_laws():
    """Water rows in the rendered frame = level x px-per-mark (rounded), for every vessel, at several frames."""
    import numpy as np
    for seed in (20, 21, 22, 23, 24):
        sc = sample(seed)["params"]
        lay = sc["layout"]
        water = np.array(sc["style"]["water"])
        for f in (0, sc["frames"] // 3, sc["frames"] - 1):
            img = np.asarray(render.frame_at(sc, f), dtype=np.int16)
            for i in range(len(sc["vessels"])):
                x0, x1 = lay["x"][i]
                xm = x0 + (x1 - x0) // 4 if sc["style"]["tick_side"] == "right" else x1 - 1 - (x1 - x0) // 4
                col = img[:, xm]
                rows = [y for y in range(lay["top"], lay["bottom"])
                        if np.abs(col[y] - water).sum() < 60 or np.abs(col[y] - (water - 40).clip(0)).sum() < 60]
                h = sim.level(sc, i, f / 24)
                want = 0 if h <= 0 else max(round(h * lay["M"]), render.FILM)
                assert len(rows) == want, (seed, f, i, len(rows), want)


def _hand_scene():
    """5 vessels, 200 frames (t_last 8.29 s): red tap, green steady drain, purple water clock, orange + black pair."""
    vs = [{"role": "tap", "w": 1.0, "h0": 4.0, "q": 1.25, "colour": "red"},
          {"role": "steady", "w": 1.5, "h0": 6.0, "q": 1.9, "colour": "green"},
          {"role": "leaky", "w": 1.2, "h0": 9.0, "k": 1.14, "colour": "purple"},
          {"role": "pair", "w": 0.8, "h0": 9.0, "valve": 2.02, "K": 0.8889, "colour": "orange"},
          {"role": "pair", "w": 1.6, "h0": 1.5, "valve": 2.02, "K": 0.8889, "colour": "black"}]
    sc = {"scenario": "joined_pair", "frames": 200, "vessels": vs, "pairs": [[3, 4]],
          "style": sim._style(random.Random(1), "light")}
    for k in range(200):
        lay = sim._layout(random.Random(k), sc)
        if lay:
            break
    sc["layout"] = lay
    sc["labels"] = sim.labels(sc)
    return V._roundtrip(sc)


def test_hand_checked_scene():
    sc = _hand_scene()
    a = lambda key, **p: answers.answer(key, p, sc)  # noqa: E731
    # red tap: (10 - 4) * 1.0 / 1.25 = 4.8 s = frame 115.2 -> first shown at frame 116, clock 116*10//24 = 48
    assert a("event_clock", x=0) == "4.8 s"
    # green steady drain: 6 * 1.5 / 1.9 = 4.737 s = frame 113.7 -> frame 114 -> 47 tenths
    assert a("event_clock", x=1) == "4.7 s"
    # purple water clock: 2 * 1.2 * 3 / 1.14 = 6.316 s = frame 151.6 -> frame 152 -> 63 tenths
    assert a("event_clock", x=2) == "6.3 s"
    # valve 2.02 s = frame 48.5 -> frame 49 -> 20 tenths; shared level (0.8*9 + 1.6*1.5) / 2.4 = 4.0
    assert a("valve_clock", x=3) == "2.0 s"
    assert a("shared_level", x=3, y=4) == 4
    # orange 2 marks higher: (0.8*11 + 2.4) / 2.4 = 4.67 -> 5
    assert a("cf_shared_extra", x=3, y=4, dh=2) == 5
    assert a("steady_vs_leaky", x=2, y=1) == "the green vessel"
    assert a("first_event") == "the green vessel"
    assert a("n_overflow") == 1 and a("n_empty_end") == 2 and a("n_taps") == 1
    assert a("n_rising") == 2                                        # the tap and the lower joined vessel
    # end: red full (10 x 1.0), orange / black at 4 (3.2, 6.4) -> red holds the most
    assert a("most_water_end") == "the red vessel"
    # valve never opens: red 10, orange 9 x 0.8 = 7.2, black 1.5 x 1.6 = 2.4 -> red
    assert a("cf_valve_never") == "the red vessel"
    # twice as wide: red would overflow at 9.6 s, after the clip (8.29 s); green would empty at 9.47 s
    assert a("cf_wider_time", x=0) == "it would not overflow before the clip ends"
    assert a("cf_wider_time", x=1) == "it would not run empty before the clip ends"
    # red started 2 marks higher: 4 / 1.25 = 3.2 s = frame 76.8 -> frame 77 -> 32 tenths
    assert a("cf_fuller_time", x=0, dh=2) == "3.2 s"
    # red twice as wide at the end: 4 + 1.25 * 8.29 / 2 = 9.18 -> 9
    assert a("cf_wider_level", x=0) == 9
    # tap twice as fast: red overflows at 2.4 s, before green (4.74 s)
    assert a("cf_faster_tap", x=0) == "the red vessel"
    assert a("level_end", x=4) == 4 and a("drop_marks", x=3) == 5
    assert a("earlier_event", a=["valve", 3], b=["solo", 0]) == \
        "the valve between the orange and black vessels opens"
    assert a("event_order", ev=[["solo", 2], ["valve", 3], ["solo", 1]]) == \
        "orange-black valve opens → green empties → purple empties"
    assert a("time_between", a=["solo", 1], b=["solo", 2]) == "1.6 s"


def test_pinned_seeds():
    got = {}
    for seed in PINNED:
        s = sample(seed)
        got[seed] = [s["params"]["scenario"], [[q["derivation_key"], q["answer_text"]] for q in s["questions"]]]
    assert got == PINNED, json.dumps(got)


PINNED: dict = {
    0: ["joined_pair", [
        ["highest_start", "the brown vessel"],
        ["n_empty_end", "2"],
        ["n_rising", "1"],
        ["higher_end", "the brown vessel"],
        ["valve_flow", "the red vessel"],
        ["event_clock", "4.7 s"],
        ["drop_marks", "7"],
        ["most_lost", "the orange vessel"],
        ["time_between", "2.6 s"],
        ["cf_shared_extra", "6"],
    ]],
    1: ["drain_race", [
        ["widest", "the green vessel"],
        ["earlier_event", "the yellow vessel runs empty"],
        ["steady_vs_leaky", "the green vessel"],
        ["drop_marks", "3"],
        ["event_clock", "5.8 s"],
        ["drop_marks", "2"],
        ["time_between", "2.2 s"],
        ["event_order", "green empties → white empties → purple empties"],
        ["cf_wider_level", "0"],
        ["cf_wider_time", "it would not run empty before the clip ends"],
    ]],
    2: ["taps_drains", [
        ["n_empty_end", "3"],
        ["first_event", "the purple vessel"],
        ["widest", "the yellow vessel"],
        ["higher_end", "the brown vessel"],
        ["earlier_event", "the purple vessel runs empty"],
        ["event_clock", "7.5 s"],
        ["drop_marks", "6"],
        ["level_at_clock", "4"],
        ["event_order", "purple empties → green empties → yellow empties"],
        ["cf_wider_level", "3"],
    ]],
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
                print("FAIL", name_, repr(e)[:400])
    sys.exit(1 if fails else 0)
