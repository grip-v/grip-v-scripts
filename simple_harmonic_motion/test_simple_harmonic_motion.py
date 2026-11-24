"""Tests for simple_harmonic_motion: python test_simple_harmonic_motion.py  (or pytest)

Covers determinism (same seed in two processes with different PYTHONHASHSEED
-> identical JSON; identical raw frames), the 10-question ladder counts,
scene uniqueness, the tamper test, min_fps on every question, the clock (the
only text) matching the frame number, and hand-checked edge cases: closed
form vs numeric motion, release frames, damped crossings, the jitter band of
the decoded-track reader, and one pinned seed per scenario.
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

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import ladder5 as L  # noqa: E402
from common import video5 as V  # noqa: E402
import sim, questions, answers, render  # noqa: E402

DS = "simple_harmonic_motion"


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
            for f in range(0, render.n_frames(sc), 5):
                h.update(render.frame_at(sc, f).tobytes())
            hs.append(h.hexdigest())
            render._static.cache_clear()
            render.sprite.cache_clear()
        assert hs[0] == hs[1]


def test_ladder_uniqueness_minfps_answers():
    sigs = set()
    for seed in range(150):
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
                    assert sim.EDGE <= q["params"][k] <= sim.duration(sc) - sim.EDGE
        assert sim.MIN_FRAMES <= render.n_frames(sc) <= sim.MAX_FRAMES
    assert len(sigs) == 150


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


def test_clock_is_the_frame_number():
    # the clock shows floor(10 f / 24) tenths; frame_of_clock inverts it on the 0.5 s grid
    assert [render.clock_text(f) for f in (0, 2, 3, 23, 24, 36, 239)] == \
        ["0.0 s", "0.0 s", "0.1 s", "0.9 s", "1.0 s", "1.5 s", "9.9 s"]
    for k in range(2, 20):
        f = answers.frame_of_clock(k / 2)
        assert sim.clock_tenths(f) == 5 * k and sim.clock_tenths(f - 1) == 5 * k - 1
    s = sample(3)
    for q in s["questions"]:
        assert "Hz" not in q["question"]


def _osc(**kw):
    o = {"mode": "free", "A": 80.0, "omega": 2 * math.pi / 1.6, "phi": 0.3, "s": 1, "t_rel": 0.0, "zeta": 0.0,
         "m": 1.0, "b": 0.0}
    o.update(kw)
    o["k"] = o["m"] * o["omega"] ** 2
    o["b"] = 2 * o["zeta"] * math.sqrt(o["k"] * o["m"])
    return o


def test_closed_form_vs_numeric():
    # free: crossings every T/2 = 0.8 s; RK4 agrees on the period after a mass change (x4 -> period x2)
    o = _osc()
    cr = sim.crossings(o, 0, 5)
    assert all(abs((b - a) - 0.8) < 1e-9 for a, b in zip(cr, cr[1:]))
    assert all(abs(sim.u_at(o, t)) < 1e-6 for t in cr)
    sc = {"g": None}
    assert abs(answers.cf_period(o, sc, "mass_x4") - 3.2) < 1e-3
    assert abs(answers.cf_period(o, sc, "stiff_x4") - 0.8) < 1e-3
    assert abs(answers.cf_period(o, sc, "amp_x2") - 1.6) < 1e-3
    # damped: crossings are still evenly spaced, at the damped period / 2; turns decay by exp(-zeta w T/2)
    d = _osc(mode="damped", phi=0.0, zeta=0.08)
    cr = sim.crossings(d, 0, 6)
    Td = sim.period(d)
    assert abs(Td - 1.6 / math.sqrt(1 - 0.08 ** 2)) < 1e-9
    assert all(abs((b - a) - Td / 2) < 1e-9 for a, b in zip(cr, cr[1:]))
    assert all(abs(sim.u_at(d, t)) < 1e-6 for t in cr)
    tu = sim.turns(d, 0, 6)
    r = abs(sim.u_at(d, tu[1][0]) / sim.u_at(d, tu[0][0]))
    assert abs(r - math.exp(-0.08 * d["omega"] * Td / 2)) < 1e-9
    assert abs(answers.cf_period(d, sc, "no_damping") - 1.6) < 1e-3


def test_release_frame_and_held_motion():
    o = _osc(mode="held", s=-1, phi=0.0, t_rel=37 / 24)
    assert sim.u_at(o, 36 / 24) == -80.0 and sim.u_at(o, 37 / 24) == -80.0 and sim.u_at(o, 38 / 24) > -80.0
    assert sim.turns(o, 0, 37 / 24 + 0.01) == []                      # release is not a turning point
    assert abs(sim.crossings(o, 0, 5)[0] - (37 / 24 + 0.4)) < 1e-9     # first pass a quarter period later


def test_decoded_track_jitter_band():
    # a sine sampled at 24 fps plus +-1 px jitter: the h = 2 band reads the same crossings and turns
    rng = random.Random(0)
    sc = {"tick": 30, "orient": "horizontal", "objects": [{"colour": "red", "rest": [600, 300]}]}
    clean = [600 + 90 * math.cos(2 * math.pi * f / 24 / 1.7 + 0.4) for f in range(200)]
    noisy = [x + rng.uniform(-1, 1) for x in clean]
    T0 = answers.Track(sc, {"xy": {"red": [(x, 300) for x in clean]}, "held": {"red": [False] * 200}}, 0.0)
    T1 = answers.Track(sc, {"xy": {"red": [(x, 300) for x in noisy]}, "held": {"red": [False] * 200}}, 2.0)
    assert len(T0.crossings("red")) == len(T1.crossings("red"))
    assert len(T0.turns("red")) == len(T1.turns("red"))
    assert abs(T0.period("red") - 1.7) < 2e-3 and abs(T1.period("red") - 1.7) < 0.02
    assert round(T1.amplitude("red") / 30) == 3


def test_event_frame_rule():
    fa, ok = questions.event_frame(1.52)          # 36.48 frames -> first frame after is 37 (clock 1.5)
    assert (fa, ok) == (37, True)
    assert questions.event_frame(1.5)[1] is False  # exactly on a frame: ambiguous, rejected


def test_pinned_seeds():
    """One seed per scenario, checked by hand against the closed form and the rendered frames."""
    got = {}
    for seed in PINNED:
        s = sample(seed)
        got[seed] = (s["params"]["scenario"], [(q["derivation_key"], q["answer_text"]) for q in s["questions"]])
    assert got == PINNED, json.dumps(got)


# Hand checks against the closed form: seed 2 green T = 1.601 s -> "1.6 s", mass x4 -> 2T = 3.20 s;
# order green 1.60 < pink 1.96 < yellow 2.80. Seed 0 (damped) yellow T_d = 2.832 s, yellow - orange = 2.832 - 0.939
# = 1.89 -> "1.9 s"; undamped yellow reaches its 1.94-tick pull -> 2. Seed 5 (staggered) brown let go at frame 90:
# clock 90*10//24 = 37 -> "3.7 s"; orange returns after T = 2.533 s; blue (6.08 ticks) is the widest, T = 1.606 s.
# Seed 7 (pendulum) brown string /4 -> T/2 = 0.821 s; red string x4 -> 2 x 1.464 = 2.93 s. Seed 1: purple stiffness x4
# -> T/2 = 0.801 s; the red triangle (2.89 s) is slower than all 4 others.
PINNED: dict = {
    0: ('damped', [
        ('n_objects', '3'),
        ('n_faster_than', '2'),
        ('n_faster_than', '0'),
        ('period_of', '2.8 s'),
        ('peak_ticks', '2'),
        ('period_difference', '1.9 s'),
        ('order_by_period', 'orange → teal → yellow'),
        ('period_of_extreme', '2.8 s'),
        ('cf_period', '2.8 s'),
        ('cf_no_damping', '2'),
    ]),
    1: ('vertical', [
        ('pattern_of', 'solid'),
        ('n_objects', '5'),
        ('farthest_at_t', 'the purple diamond'),
        ('n_faster_than', '4'),
        ('far_end_count', '1'),
        ('crossings_count', '2'),
        ('order_by_period', 'teal → purple → red'),
        ('cf_crossings', '1'),
        ('cf_period', '0.8 s'),
        ('cf_crossings', '2'),
    ]),
    2: ('horizontal', [
        ('n_objects', '3'),
        ('pattern_of', 'solid'),
        ('n_faster_than', '1'),
        ('n_faster_than', '0'),
        ('period_of', '1.6 s'),
        ('far_end_count', '2'),
        ('crossing_time', '7.0 s'),
        ('order_by_period', 'green → pink → yellow'),
        ('period_of_extreme', '1.6 s'),
        ('cf_period', '3.2 s'),
    ]),
    5: ('staggered', [
        ('first_released', 'the red square'),
        ('released_before_count', '1'),
        ('n_faster_than', '0'),
        ('period_of', '2.1 s'),
        ('release_time', '3.7 s'),
        ('amplitude_ticks', '6'),
        ('period_of_extreme', '1.6 s'),
        ('return_time', '2.5 s'),
        ('period_difference', '1.7 s'),
        ('cf_crossings', '2'),
    ]),
    7: ('pendulum', [
        ('n_objects', '3'),
        ('pattern_of', 'solid'),
        ('n_faster_than', '1'),
        ('n_faster_than', '2'),
        ('amplitude_ticks', '2'),
        ('period_of', '1.3 s'),
        ('order_by_period', 'blue → red → brown'),
        ('cf_period', '0.8 s'),
        ('cf_crossings', '3'),
        ('cf_period', '2.9 s'),
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
