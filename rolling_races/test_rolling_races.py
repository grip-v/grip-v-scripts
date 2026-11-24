"""Tests for rolling_races: python test_rolling_races.py  (or pytest)

Covers determinism (same seed in two processes with different PYTHONHASHSEED
-> identical JSON; identical raw frames), the 10-question ladder counts,
scene uniqueness, the tamper test, min_fps on every question, the on-screen
text (the clock is the only text; its box never meets a ball), the
answer-side integrator agreeing with sim.py's, and hand-checked physics and
edge cases (straight-ramp time in closed form, equal final speeds, the
brachistochrone beating the straight ramp, a narrow gap cleared and a wide
gap missed, the clock rule).
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

DS = "rolling_races"


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
            sim.run.cache_clear()
            sim.polyline.cache_clear()
        assert hs[0] == hs[1]


def test_ladder_uniqueness_minfps_answers():
    sigs = set()
    for seed in range(150):
        s = sample(seed)
        sc, qs = s["params"], s["questions"]
        assert L.ladder_errors(qs, "ten") == [], seed
        assert sim.check(sc) == [], seed
        sigs.add(json.dumps({k: v for k, v in sc.items() if k != "build_attempts"}, sort_keys=True))
        for q in qs:
            mf = q["params"].get("min_fps")
            assert isinstance(mf, int) and 1 <= mf <= 24
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4
            assert str(answers.answer(q["derivation_key"], q["params"], sc)) == str(q["groundtruth"])
        assert render.n_frames(sc) <= sim.MAX_FRAMES
        for t, kind, col in sim.events(sc):           # question events stay inside the clip
            assert sim.EDGE_S <= t <= sim.duration(sc) - sim.EDGE_S, (seed, kind, col)
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


def test_clock_is_the_only_text_and_stays_clear():
    """No label list: the clock is the only text. Its box is in the top corner opposite the start, the
    static layer draws nothing there, and no ball, rack slot or track ever enters it."""
    import numpy as np
    for seed in (3, 4, 30):
        sc = sample(seed)["params"]
        assert "labels" not in sc
        box = sim.clock_box(sc)
        sx = sim.px(sc, 0, 0)[0]
        assert (box[0] > 640) != (sx > 640)                      # opposite corner to the start
        img = np.asarray(render._static(json.dumps(sc, sort_keys=True)), dtype=np.int16)
        crop = img[box[1] - 4:box[3] + 5, box[0] - 4:box[2] + 5]
        bg = np.array(sc["style"]["bg"], dtype=np.int16)
        assert (np.abs(crop - bg).sum(-1) > 40).mean() < 0.02     # only faint background pattern there
        for f in range(render.n_frames(sc)):
            for c, (x, y) in render.state(sc, f)["balls"].items():
                R = sc["style"]["R"]
                assert not (x + R > box[0] and x - R < box[2] and y - R < box[3]), (seed, f, c)


def test_integrators_agree():
    """answers.py's exact piecewise integrator and sim.py's Simpson run give the same finish times."""
    for seed in range(40):
        sc = sample(seed)["params"]
        for b in sc["balls"]:
            r = sim.ball_run(sc, b)
            t, v, fell = answers.race(sc, b, sc["v0"])
            assert (r["finish"] is None) == (t is None) and fell == (r["fell"] is not None)
            if t is not None:
                assert abs(t - r["finish"]) < 1e-4, (seed, b["colour"], t, r["finish"])


def _scene(balls, W=10.0, D=3.5, v0=1.0):
    sc = {"scenario": "plain", "W": W, "D": D, "v0": v0, "floor": 6.0, "slow": 2.0, "t0": 1.0, "frames": 220,
          "balls": [dict(colour=c, pattern="solid", **b) for c, b in balls],
          "rack": [c for c, _ in balls]}
    sc["style"] = {"theme": "light", "bg": [242, 239, 232], "bg_pat": "plain", "bg_step": 50, "ink": [34, 36, 42],
                   "pit": [200, 194, 182], "hole": [24, 24, 28], "rack": [214, 210, 200], "R": 12, "track_w": 4,
                   "track_style": "solid", "sleeper_step": 20, "S": 90.0, "ox": 120.0, "oy": 140.0, "mirror": False,
                   "rack_dx": 70, "rack_y0": 150.0, "rack_step": 36}
    return sc


def test_straight_ramp_closed_form_and_equal_speeds():
    sc = _scene([("red", {"family": "straight"}), ("blue", {"family": "cycloid"}),
                 ("green", {"family": "sag", "a": 0.4}), ("orange", {"family": "slow_start", "a": 1.8})])
    Lr = math.hypot(sc["W"], sc["D"])
    a = sim.G * sc["D"] / Lr
    t_exact = (-sc["v0"] + math.sqrt(sc["v0"] ** 2 + 2 * a * Lr)) / a
    r = sim.ball_run(sc, sc["balls"][0])
    assert abs(r["finish"] - t_exact) < 1e-4
    assert abs(answers.race(sc, sc["balls"][0], sc["v0"])[0] - t_exact) < 1e-9
    v_end = math.sqrt(sc["v0"] ** 2 + 2 * sim.G * sc["D"])                   # energy conservation
    for b in sc["balls"]:
        assert abs(sim.ball_run(sc, b)["v_end"] - v_end) < 1e-6
        assert abs(answers.race(sc, b, sc["v0"])[1] - v_end) < 1e-6
    # the brachistochrone beats the straight ramp, and the gentle start loses
    assert answers.answer("winner", {}, sc) == "the blue ball"
    assert answers.answer("last_finisher", {}, sc) == "the orange ball"
    assert answers.answer("same_speed", {}, sc) == "they are all moving at the same speed"
    assert answers.answer("n_finish", {}, sc) == "4"


def test_narrow_gap_cleared_wide_gap_missed():
    sc = _scene([("red", {"family": "straight", "gaps": [{"x0": 4.0, "x1": 4.6, "lip": 20}]}),
                 ("blue", {"family": "straight", "gaps": [{"x0": 4.0, "x1": 7.5, "lip": 10}]})])
    rr, rb = (sim.ball_run(sc, b) for b in sc["balls"])
    assert rr["fell"] is None and rr["land"] and rr["finish"] is not None
    assert rb["fell"] == 0 and rb["finish"] is None and rb["floor_t"] > rb["leave"][0]
    assert answers.answer("n_fell", {}, sc) == "1"
    assert answers.answer("n_finish", {}, sc) == "1"
    assert answers.answer("total_landings", {}, sc) == "1"
    # a landing keeps only the along-track speed: the gapped red ball is slower at the end than a plain one
    plain = sim.ball_run(sc, sc["balls"][0], no_gap=True)
    assert rr["v_end"] < plain["v_end"] and rr["finish"] > plain["finish"]
    # leave time: the ball's centre crosses the gap's near edge on the first frame at/after the event
    o = answers.Obs(sc, answers._frames(sc))
    te = sim.clock_t(sc, rr["leave"][0])
    assert abs(o.leave_t("red", 0) - te) < 0.01


def test_ball_passes_behind_other_tracks():
    """A ball flying or falling across another colour's track is drawn behind it: inside the ball's disc,
    pixels of the other track show the static layer (the track), not the ball. Rolling balls stay on top."""
    import numpy as np
    checked = 0
    for seed in range(40):
        sc = sample(seed)["params"]
        key = json.dumps(sc, sort_keys=True)
        static = np.asarray(render._static(key))
        others = render.other_tracks(key)
        R = sc["style"]["R"]
        for b in sc["balls"]:
            m = np.asarray(others[b["colour"]]) > 0
            for f in range(0, render.n_frames(sc), 2):
                p = render.state(sc, f)["balls"].get(b["colour"])
                q = sim.ball_centre(sc, b, f)
                if p is None or not q[2]:
                    continue
                if any(np.hypot(p[0] - v[0], p[1] - v[1]) < 2 * R + 2
                       for k, v in render.state(sc, f)["balls"].items() if k != b["colour"]):
                    continue
                x, y = int(round(p[0])), int(round(p[1]))
                yy, xx = np.mgrid[y - R + 3:y + R - 2, x - R + 3:x + R - 2]
                inside = (np.hypot(xx - p[0], yy - p[1]) < R - 3) & m[yy, xx]
                if inside.sum() < 5:
                    continue
                img = np.asarray(render.frame_at(sc, f))
                assert (img[yy[inside], xx[inside]] == static[yy[inside], xx[inside]]).all(), (seed, f, b["colour"])
                checked += 1
                break
        if checked >= 5:
            break
    assert checked >= 5


def test_which_lands_counts_any_landing():
    """A two-gap ball that clears its first gap and falls at the second has landed once: it is never a
    distractor of which_lands, and answers.py rejects an option set with two landers."""
    for seed in range(400):
        s = sample(seed)
        sc = s["params"]
        for q in s["questions"]:
            if q["derivation_key"] == "which_lands":
                landed = [c for c in q["params"]["options"]
                          if sim.ball_run(sc, next(b for b in sc["balls"] if b["colour"] == c))["land"]]
                assert len(landed) == 1, (seed, q["params"]["options"])
    sc = sample(0)["params"]
    assert answers._only(["red", "blue"]) is None and answers._only(["red"]) == "the red ball"


def test_clock_rule():
    assert [sim.clock_tenths(f) for f in (0, 2, 3, 23, 24, 25, 239)] == [0, 0, 1, 9, 10, 10, 99]
    assert answers.event_clock(1.0) == 10 and answers.event_clock(1.001) == 10 and answers.event_clock(0.999) == 10


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} tests passed")
