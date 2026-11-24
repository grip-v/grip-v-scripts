"""Tests for well_flood_trace: determinism, ladder counts, uniqueness, tamper
detection, min_fps, and hand-checked edge cases.

    python test_well_flood_trace.py          (or: pytest test_well_flood_trace.py)
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import ladder5 as L  # noqa: E402
from common.video5 import build_sample, _roundtrip  # noqa: E402
import sim  # noqa: E402
import questions  # noqa: E402
import answers  # noqa: E402

DS = "well_flood_trace"
SEEDS = range(0, 120)


def _sample(seed):
    return _roundtrip(build_sample(DS, sim, questions, seed))


def test_same_seed_same_json_in_process():
    for seed in (0, 7, 123):
        assert json.dumps(_sample(seed), sort_keys=True) == json.dumps(_sample(seed), sort_keys=True)


def test_same_seed_same_json_across_processes():
    code = ("import sys,json;sys.path[:0]=[{h!r},{p!r}];from common.video5 import build_sample,_roundtrip;"
            "import sim,questions;print(json.dumps([_roundtrip(build_sample('well_flood_trace',sim,questions,s)) "
            "for s in (3,41,977)],sort_keys=True))").format(h=str(HERE), p=str(HERE.parent))
    outs = []
    for hs in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                   check=True).stdout)
    assert outs[0] == outs[1] and len(outs[0]) > 1000


def test_ladder_counts_and_min_fps():
    for seed in SEEDS:
        qs = _sample(seed)["questions"]
        assert L.ladder_errors(qs, "ten") == [], seed
        assert len(qs) == 10
        for q in qs:
            assert isinstance(q["params"].get("min_fps"), int) and 1 <= q["params"]["min_fps"] <= 4, (seed, q)
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4


def test_scenes_unique_and_checked():
    sigs = set()
    for seed in SEEDS:
        sc = _sample(seed)["params"]
        assert sim.check(sc) == [], seed
        sigs.add(hashlib.sha1(json.dumps(sc, sort_keys=True).encode()).hexdigest())
    assert len(sigs) == len(SEEDS)


def test_independent_answers_agree():
    for seed in SEEDS:
        s = _sample(seed)
        for q in s["questions"]:
            assert str(answers.answer(q["derivation_key"], q["params"], s["params"])) == str(q["groundtruth"]), \
                (seed, q["qid"], q["derivation_key"])


def test_tamper_is_caught():
    """Corrupt 3 stored answers in 3 samples: exactly those 3 questions are flagged."""
    samples = [_sample(s) for s in range(30)]
    bad = copy.deepcopy(samples)
    targets = [(2, 0), (11, 4), (25, 9)]
    for si, qi in targets:
        q = bad[si]["questions"][qi]
        q["groundtruth"] = next(o for o in q["options"] if o != q["answer_text"])
    flagged = [(si, qi) for si, s in enumerate(bad) for qi, q in enumerate(s["questions"])
               if str(answers.answer(q["derivation_key"], q["params"], s["params"])) != str(q["groundtruth"])]
    assert flagged == targets


# ------------------------------------------------------------------ hand-checked scenes
def _scene(depth, taps, walls=(), violation=None, nf=190):
    sc = {"scenario": "valid_flow", "R": 3, "C": 3, "terrain": "random", "depth": depth,
          "walls": [list(w) for w in walls], "taps": taps, "violation": violation, "palette": 0,
          "layout": {"bx": 300, "by": 10, "S": 680, "head": 74, "cell": 180, "x0": 370, "y0": 108},
          "n_frames": nf, "duration": nf / 24}
    sc["track"] = sim._track(sim.run_world(sc))
    return sc


#  depth values (cell index in brackets)    entry: cell 0 (depth 9), 3 units/s from frame 12 (t0 = 12.5/24 = 0.5208 s)
#  9[0] 8[1] 7[2]                           cell 0 full at t0 + 9/3 = 3.5208 s; then 1.5/s each into 8 and 4
#  4[3] 5[4] 6[5]                           cell 3 (depth 4) full at 3.5208 + 4/1.5 = 6.1875 s, then 1.5/s into 3
#  3[6] 2[7] 1[8]                           cell 6 (depth 3) full at 8.19 s; cell 1 (depth 8) full at 8.85 s
#  The clip ends at frame 189 = 7.875 s: wet 9, 8, 4, 3; full 9, 4 (8 is 82% full, 3 is 84% full)
BASE = [9, 8, 7, 4, 5, 6, 3, 2, 1]
TAP = [{"cell": 0, "colour": "red", "segs": [[12, None, 3.0]], "change": None}]


def test_hand_checked_basic_flood():
    sc = _scene(BASE, TAP)
    tr = sc["track"]
    assert abs(tr["full"][0] - 3.520833) < 1e-6 and abs(tr["full"][3] - 6.1875) < 1e-6
    assert answers.answer("n_wet_end", {}, sc) == 4
    assert answers.answer("n_full_end", {}, sc) == 2
    assert answers.answer("entry_depth", {"tap": 0}, sc) == "9"
    assert answers.answer("start_clock", {"cell": 3}, sc) == "3.5 s"    # frame 85, clock 850 // 24 = 35
    assert answers.answer("full_clock", {"cell": 3}, sc) == "6.2 s"     # frame 149, clock 62
    assert answers.answer("fed_from", {"cell": 6}, sc) == "from the section above it"
    assert answers.answer("order_three", {"cells": [0, 3, 6]}, sc) == "9, then 4, then 3"
    assert answers.answer("first_depth", {}, sc) is None                # 8 and 4 start together
    assert answers.answer("kth_wet", {"k": 2, "cells": [1, 3, 6, 7]}, sc) is None
    assert answers.answer("kth_wet", {"k": 4, "cells": [1, 3, 6, 7]}, sc) == "the section with depth value 3"
    assert answers.answer("impossible_which", {"cells": [1, 3, 6]}, sc) == answers.NONE_OPT
    assert answers.answer("violation_kind", {}, sc) == answers.KIND_TEXT[None]
    # double rate: 9 full 2.02 s, 4 at 3.35, 8 at 4.69, 3 at 4.35, 2 at 5.02, 1 at 5.35; 8 then feeds 7 and 5
    assert answers.answer("cf_rate", {"mult": 2.0}, sc) == 8
    # depth 4 sealed: everything goes 9 -> 8 (full 6.19 s) -> 7 and 5
    assert answers.answer("cf_block", {"cell": 3}, sc) == 4


def test_hand_checked_wall_and_premature():
    # a wall between 9 and 8: all 3/s go 9 -> 4 (full 4.85 s) -> 3 (5.85 s) -> 2 (6.52 s) -> 1, which soaks away
    sc = _scene(BASE, TAP, walls=[[0, 1]])
    assert answers.answer("n_wet_end", {}, sc) == 5
    assert answers.answer("walls_blocking", {}, sc) == 1                # 9 is full next to the wall
    assert answers.answer("cf_wall", {"wall": [0, 1]}, sc) == 4
    # premature fill: depth 1 starts filling at frame 48 (2.02 s) with nothing feeding it
    sc2 = _scene(BASE, TAP, violation={"kind": "premature_fill", "cell": 8, "frame": 48, "rate": 0.5})
    assert answers.answer("impossible_which", {"cells": [1, 6, 8]}, sc2) == "the section with depth value 1"
    assert answers.answer("violation_kind", {}, sc2) == answers.KIND_TEXT["premature_fill"]
    assert answers.answer("n_wet_end", {}, sc2) == 5
    assert answers.answer("cf_no_violation", {}, sc2) == 4
    assert answers.answer("cf_correction", {"cell": 6}, sc2) == "at the same time as shown"


def test_hand_checked_uphill():
    # depth 4 (full 6.19 s) also spills into depth 5, which is HIGHER: 0.75/s each into 3 and 5
    sc = _scene(BASE, TAP, violation={"kind": "uphill_flow", "src": 3, "dst": 4})
    assert abs(sc["track"]["start"][4] - 6.1875) < 1e-6
    assert answers.answer("impossible_which", {"cells": [3, 4, 6]}, sc) == "the section with depth value 5"
    assert answers.answer("violation_kind", {}, sc) == answers.KIND_TEXT["uphill_flow"]
    assert answers.answer("n_wet_end", {}, sc) == 5                     # 9, 8, 4, then 3 and 5
    assert answers.answer("cf_no_violation", {}, sc) == 4               # depth 5 stays dry
    assert answers.answer("cf_correction", {"cell": 6}, sc) == "at the same time as shown"


def test_clock_edges():
    assert sim.clock_tenths(23) == 9 and sim.clock_tenths(24) == 10 and sim.clock_tenths(239) == 99
    assert sim.frame_of(3.5208333) == 85 and sim.tap_time(12) == 12.5 / 24


def test_rendered_water_appears_on_its_frame():
    import numpy as np
    import render
    sc = _scene(BASE, TAP)
    x0, y0, x1, y1 = render.cell_rect(sc, 3)
    rising = np.array(render.PALETTE[0]["rising"])

    def wet(f):
        a = np.asarray(render.frame(sc, f / 24)).astype(int)[y1 - 6:y1 - 4, x0 + 8:x1 - 8]
        return (np.abs(a - rising).sum(axis=2) < 30).mean() > 0.9
    assert not wet(84) and wet(85)                                      # start at 3.5208 s = frame 84.5


def test_render_is_deterministic():
    import numpy as np
    import render
    sc = _sample(5)["params"]
    h = []
    for _ in range(2):
        render._BASE["scene"] = None
        h.append(hashlib.sha1(b"".join(np.asarray(render.frame(sc, f / 24)).tobytes()
                                       for f in range(0, sc["n_frames"], 9))).hexdigest())
    assert h[0] == h[1]


def test_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for seed in range(200, 260):
            _sample(seed)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print("ok", n)
    print(f"all {len(tests)} tests passed")
