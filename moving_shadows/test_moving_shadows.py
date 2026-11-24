"""Tests for moving_shadows: python test_moving_shadows.py  (or pytest)

Covers determinism (same seed in two processes with different PYTHONHASHSEED
-> identical JSON and answers; identical raw frames on re-render), the
10-question ladder counts, scene uniqueness, the tamper test, min_fps on
every question, the marker-number labels matching the renderer, the clock,
hand-worked geometry (vertex projection vs the marker-to-light segment
test), the frame / clock conventions, the zigzag readers, the what-if
variants, and pinned seeds with expected answers.
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

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import ladder5 as L  # noqa: E402
from common import video5 as V  # noqa: E402
import sim, questions, answers, render  # noqa: E402,E401

DS = "moving_shadows"


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
            for f in range(0, render.n_frames(sc), 7):
                h.update(render.frame_at(sc, f).tobytes())
            hs.append(h.hexdigest())
            render._static.cache_clear()
        assert hs[0] == hs[1]


def test_ladder_uniqueness_minfps_answers():
    sigs = set()
    for seed in range(120):
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
            if "t" in q["params"]:
                assert 1.0 <= q["params"]["t"] <= sim.t_end(sc) - 1.0 + 1e-9
        assert sim.MIN_FRAMES <= render.n_frames(sc) <= sim.MAX_FRAMES
    assert len(sigs) == 120


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


def test_clock_and_frames():
    assert [render.clock_text(f) for f in (0, 2, 3, 23, 24, 36, 239)] == \
        ["0.0 s", "0.0 s", "0.1 s", "0.9 s", "1.0 s", "1.5 s", "9.9 s"]
    for k in range(10, 95):
        f = answers.frame_of_clock(k / 10)
        assert sim.clock_tenths(f) == k and sim.clock_tenths(f - 1) == k - 1
    # an event at 36.5 frames is first shown on frame 37; one at 36.05 frames is too close to call
    assert questions.first_frame_after(36.5 / 24) == (37, True)
    assert questions.first_frame_after(36.05 / 24)[1] is False


def test_labels_match_renderer():
    """The marker numbers are exactly scene['labels']: right text, inside their stored boxes, readable."""
    import frame_check as FC
    for seed in (1, 2, 3):
        sc = sample(seed)["params"]
        img = np.asarray(render.frame_at(sc, sim.n_frames(sc) // 2))
        bare = FC._bare(render.static_key(sc))
        diff = np.abs(img.astype(int) - bare).sum(-1) > 0
        diff[: sc["ground"] + sc["style"]["strip"] + 2] = False      # only the ground body below the strip
        boxes = [lb["box"] for lb in sc["labels"]]
        ys, xs = np.nonzero(diff)
        for x, y in zip(xs, ys):
            assert any(b[0] - 1 <= x <= b[2] + 1 and b[1] - 1 <= y <= b[3] + 1 for b in boxes), (seed, x, y)
        for j, lb in enumerate(sc["labels"]):
            assert lb["text"] == str(j + 1) and lb["size"] >= 36
            assert lb["box"][3] - lb["box"][1] >= 24
            assert FC.read_label(img, sc, lb) == lb["text"]


def _toy(**kw):
    sc = {"scenario": "floating", "frames": 169, "ground": 600, "markers": [614, 616, 700, 900],
          "light": {"kind": "lamp", "y": 100, "x0": 640.0, "x1": 640.0},
          "casters": [{"kind": "float", "shape": "square", "color": "red", "pattern": "solid", "ground": 600,
                       "x": 640.0, "y": 400.0, "s": 20, "rot": 0}],
          "style": {"sun_r": 20}}
    sc.update(kw)
    return sc


def test_projection_by_hand():
    # square of radius 20 at (640, 400), light at (640, 100), ground at 600: corners at +-14.142 px.
    # lower corners: factor 500 / 314.142 -> +-22.51; upper corners: 500 / 285.858 -> +-24.74
    sc = _toy()
    lo, hi = sim.shadow(sc, 0, 0.0)
    assert abs(lo - (640 - 24.736)) < 0.01 and abs(hi - (640 + 24.736)) < 0.01
    T = answers.Track(sc, [(640.0, 100.0)] * 3, [np.array([sim.outline(sc["casters"][0], 0.0)] * 3)])
    # marker at 614.5 is outside (615.26), 616.5 inside, 700.5 and 900.5 outside
    assert T.cov[0][0].tolist() == [False, True, False, False]
    # the angle-based shadow centre sits under the light
    assert abs(T.shadow_centre(0)[0] - 640.0) < 1e-6


def test_sun_parallel_rays_by_hand():
    # sun at 45 degrees on the right: every point slides left by its height above the ground (cot 45 = 1)
    sc = _toy(light={"kind": "arc", "cx": 640.0, "cy": 400.0, "Rx": 560.0, "Ry": 300.0, "phi0": 45.0, "phi1": 45.0})
    lo, hi = sim.shadow(sc, 0, 0.0)
    assert abs(lo - (640 - 14.142 - (600 - 385.858))) < 0.01 and abs(hi - (640 + 14.142 - (600 - 414.142))) < 0.01
    # two posts of different heights: shadow lengths are in the ratio of the heights (parallel rays)
    posts = [{"kind": "stand", "shape": "post", "color": c, "pattern": "solid", "ground": 600, "x": x, "w": 20,
              "h": h, "roof": 0} for c, x, h in (("red", 300.0, 100), ("blue", 900.0, 200))]
    sc2 = dict(sc, casters=posts)
    lens = [hi - lo - 20 for lo, hi in sim.shadows(sc2, 0.0)]
    assert abs(lens[0] - 100) < 1e-9 and abs(lens[1] - 200) < 1e-9
    sun = sim.light(sc, 0.0)
    T = answers.Track(sc, [sun] * 3, [np.array([sim.outline(sc["casters"][0], 0.0)] * 3)])
    m = np.array(sc["markers"]) + 0.5
    assert T.cov[0][0].tolist() == [bool(lo <= x <= hi) for x in m]
    assert abs(T.shadow_centre(0)[0] - (lo + hi) / 2) < 1e-6


def test_post_shadow_starts_at_its_foot():
    sc = _toy(casters=[{"kind": "stand", "shape": "post", "color": "red", "pattern": "solid", "ground": 600,
                        "x": 500.0, "w": 20, "h": 100, "roof": 0}],
              light={"kind": "lamp", "y": 100, "x0": 800.0, "x1": 800.0})
    lo, hi = sim.shadow(sc, 0, 0.0)
    # light to the right: shadow runs left from the post's right foot (510) to the projected top-left corner
    assert abs(hi - 510.0) < 1e-9
    assert abs(lo - (800 + (490 - 800) * 500 / 400)) < 1e-9


def test_legs_readers_agree():
    xs = [100 + 3 * k for k in range(50)] + [247 - 2 * k for k in range(60)]
    assert [d for d, _, _ in questions.legs(xs, 40.0)] == [1, -1]
    assert answers._legs(xs, 25.0) == (1, -1)
    assert answers._legs([5.0 * k for k in range(40)], 25.0) == (1,)


def test_whatif_variants():
    s = sample(3)
    sc = s["params"]
    T = answers._cached(render.static_key(sc))
    A = T.variant(remove=0)
    assert len(A.polys) == len(T.polys) - 1
    R = T.variant(reverse=True)
    assert np.allclose(R.light[0], T.light[-1])
    # removing an object can only take shade away
    assert (A.shaded <= T.shaded).all()
    assert sim.reversed_light(sim.reversed_light(sc))["light"] == sc["light"]


def test_pinned_seeds():
    """Pinned answers for one seed per scenario (checked by eye against the rendered frames)."""
    got = {}
    for seed in PINNED:
        s = sample(seed)
        got[seed] = (s["params"]["scenario"], [(q["derivation_key"], q["answer_text"]) for q in s["questions"]])
    got = json.loads(json.dumps(got))
    assert got == json.loads(json.dumps(PINNED)), json.dumps(got)


# Pinned answers for one seed per scenario (seeds 0, 1, 2, 5, 7 are also preview clips, so frame_check.py
# re-derives each of these answers from the decoded pixels of the rendered previews).
PINNED: dict = {
    0: ('rising', [('moving_object', 'the ringed red hexagon'), ('shaded_at_t', 'marker 1'), ('light_dir', 'left to right'), ('last_marker', 'marker 3'), ('caster_of', 'the ringed red hexagon'), ('longest_of', 'marker 1'), ('gap_time', '1.5 s'), ('n_entries', '2'), ('total_entries', '4'), ('cf_remove_first', 'marker 1')]),
    1: ('posts', [('n_shaded_at_t', '0'), ('never', 'marker 1'), ('shaded_at_t', 'marker 2'), ('last_marker', 'marker 4'), ('who_covers_at_t', "no object's shadow"), ('gap_time', '1.5 s'), ('n_entries', '2'), ('shade_dur', '2.5 s'), ('total_entries', '4'), ('cf_remove_first', 'marker 2')]),
    2: ('floating', [('shaded_at_t', 'marker 2'), ('n_shaded_at_t', '0'), ('light_dir', 'right to left'), ('shadow_way', 'to the right'), ('last_marker', 'marker 2'), ('shade_dur', '1.0 s'), ('gap_time', '5.0 s'), ('caster_markers', '2'), ('cf_reverse_first', 'marker 2'), ('cf_remove_never', 'marker 2')]),
    5: ('lamp', [('light_dir', 'left to right'), ('n_casters', '3'), ('last_marker', 'marker 4'), ('earliest_of', 'marker 2'), ('cover_time', '7.5 s'), ('n_entries', '1'), ('n_casters_for', '1'), ('cf_remove_count', '3'), ('cf_remove_count', '2'), ('cf_remove_first', 'marker 2')]),
    7: ('drifting', [('n_shaded_at_t', '0'), ('last_marker', 'marker 1'), ('shadow_way', 'to the left'), ('caster_of', 'the ringed pink octagon'), ('n_entries', '1'), ('cover_time', '2.5 s'), ('max_simul', '1'), ('cf_remove_count', '0'), ('cf_remove_first', 'marker 4'), ('cf_remove_count', '4')]),
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
