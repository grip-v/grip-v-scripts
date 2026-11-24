"""Tests for traffic_light_queue: python test_traffic_light_queue.py  (or pytest)

Covers determinism (same seed in two processes with different PYTHONHASHSEED
-> identical JSON; identical raw frames), the 10-question ladder counts,
scene uniqueness, the tamper test, min_fps on every question, the clock being
the only text, and a traffic-specific check that every drawn car lies inside
the frame and built scenes pass sim.check.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import ladder5 as L  # noqa: E402
from common import video5 as V  # noqa: E402
import sim, questions, answers, render  # noqa: E402

DS = "traffic_light_queue"


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
            render._SIM.clear()
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


def test_cars_inside_frame_and_check():
    W, H = render.SIZE
    seen = set()
    for seed in range(40):
        sc = sample(seed)["params"]
        seen.add(sc["scenario"])
        assert sim.check(sc) == [], seed
        for f in range(0, render.n_frames(sc), 6):
            for c in render.state(sc, f)["cars"]:
                x0, y0, x1, y1 = c["box"]
                assert 0 <= x0 <= x1 <= W and 0 <= y0 <= y1 <= H, (seed, f, c)
    assert seen == set(sim.SCENARIOS)


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
