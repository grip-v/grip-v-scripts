"""Tests for inertia_race. `python -m pytest -q test_inertia_race.py`"""
from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import answers  # noqa: E402
import questions  # noqa: E402
import render  # noqa: E402
import sim  # noqa: E402
from common import ladder5 as L  # noqa: E402
from common.video5 import build_sample  # noqa: E402

N = 60
SEEDS = list(range(N))


@pytest.fixture(scope="module")
def scenes():
    return [sim.build(random.Random(s)) for s in SEEDS]


@pytest.fixture(scope="module")
def samples():
    return [build_sample("inertia_race", sim, questions, s) for s in SEEDS]


# ---------------------------------------------------------------- determinism


def test_same_seed_same_sample():
    for s in (0, 7, 101):
        a = json.dumps(build_sample("inertia_race", sim, questions, s), sort_keys=True)
        b = json.dumps(build_sample("inertia_race", sim, questions, s), sort_keys=True)
        assert a == b


def test_hash_seed_does_not_change_output():
    code = ("import sys,random,json,hashlib;"
            "sys.path[:0]=[%r,%r];"
            "import sim,questions;from common.video5 import build_sample;"
            "h=hashlib.sha256();"
            "[h.update(json.dumps(build_sample('inertia_race',sim,questions,s),"
            "sort_keys=True).encode()) for s in range(40)];"
            "print(h.hexdigest())" % (str(HERE), str(HERE.parent)))
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={"PYTHONHASHSEED": h, "PATH": "/usr/bin:/bin"}).stdout.strip()
            for h in ("0", "1", "42")}
    assert len(outs) == 1 and outs != {""}


def test_scenes_are_distinct(scenes):
    sigs = {json.dumps({k: v for k, v in s.items() if k != "tries"}, sort_keys=True)
            for s in scenes}
    assert len(sigs) == len(scenes)


# ---------------------------------------------------------------- physics


def test_closed_form_matches_numerical_integration(scenes):
    """Semi-implicit Euler in metres against the closed form the generator uses.

    Tolerances are set by the integrator's own step error, not by the physics: a
    wrong inertia coefficient would move the finish by tens of frames, not one.
    """
    for s in scenes[:12]:
        L_m, ang = s["ramp"]["length_m"], s["ramp"]["angle_deg"]
        for r in s["racers"]:
            a = sim.G * math.sin(math.radians(ang)) / (1.0 + sim.K[r["k_shape"]])
            nsub = 400
            dt, x, v = 1.0 / (sim.FPS * nsub), r["start"] * L_m, 0.0
            hit = None
            for f in range(s["n_frames"]):
                if x >= L_m and hit is None:
                    hit = f
                assert abs(min(1.0, x / L_m) - sim.pos_at(r, s["t0"], f)) < 1e-3
                if f >= r["release_f"]:
                    for _ in range(nsub):
                        if x >= L_m:
                            break
                        v += a * dt
                        x += v * dt
            if r["finish_f"] is None:
                assert hit is None
            else:
                assert hit is not None and abs(hit - r["finish_f"]) <= 1


def test_mass_and_radius_cancel(scenes):
    """Two bodies of the same shape and start must finish together whatever their size."""
    for s in scenes:
        if s["scenario"] != "twin":
            continue
        a, b = (s["racers"][i] for i in s["twin_of"])
        assert a["shape"] == b["shape"] and a["r_px"] != b["r_px"]
        assert a["finish_f"] == b["finish_f"] is not None


def test_tau_ordering_holds_in_fair_start(scenes):
    for s in scenes:
        if s["scenario"] != "fair_start" or s["flawed"]:
            continue
        fin = sorted((r["finish_f"], sim.TAU[r["shape"]]) for r in s["racers"]
                     if r["finish_f"] is not None)
        assert [t for _, t in fin] == sorted(t for _, t in fin)


# ---------------------------------------------------------------- scene checks


def test_every_scene_passes_check(scenes):
    for s in scenes:
        assert sim.check(s) == []


def test_check_rejects_a_tie_outside_twin(scenes):
    s = json.loads(json.dumps(next(x for x in scenes if x["scenario"] != "twin")))
    a, b = s["racers"][0], s["racers"][1]
    b["finish_f"] = a["finish_f"]
    assert any("tie" in e or "clearly ahead" in e for e in sim.check(s))


def test_check_rejects_content_off_canvas(scenes):
    s = json.loads(json.dumps(scenes[0]))
    s["view"]["y0"] -= 400
    assert any("canvas" in e for e in sim.check(s))


def test_content_stays_on_canvas(scenes):
    for s in scenes:
        x0, y0, x1, y1 = sim.content_bbox(s)
        assert x0 >= sim.PAD_X - 1 and x1 <= sim.W - sim.PAD_X + 1
        assert y0 >= sim.TIMER_STRIP - 1 and y1 <= sim.H - sim.PAD_B + 1


# ---------------------------------------------------------------- renderer


def test_renderer_replay_matches_the_generator(scenes):
    for s in scenes[:20]:
        for f in range(0, s["n_frames"], 3):
            lay = render.layout(s, f)
            for r in s["racers"]:
                st = lay["racers"][r["id"]]
                got = render.to_fraction(s, (st["contact_x"], st["contact_y"]))
                assert abs(got - sim.pos_at(r, s["t0"], f)) < 1e-4
                assert abs(st["turns"] - sim.spin_turns(r, s["t0"], f)) < 1e-9


def test_labels_place_the_ruler_where_the_renderer_draws_it(scenes):
    for s in scenes:
        for lab in s["labels"]:
            got = render.to_fraction(s, (lab["x"], lab["y"]))
            # the stored label coordinates are rounded to 0.001 px
            assert abs(got - float(lab["text"])) < 1e-5


def test_rotation_can_be_unwrapped_at_this_frame_rate(scenes):
    for s in scenes:
        for r in s["racers"]:
            prev = 0.0
            for f in range(s["n_frames"]):
                cur = sim.spin_turns(r, s["t0"], f)
                assert cur - prev < 0.5          # under the Nyquist half-turn
                prev = cur


def test_duration_is_what_the_pipeline_will_encode(scenes):
    for s in scenes:
        assert render.duration(s) == s["n_frames"] / sim.FPS


# ---------------------------------------------------------------- the ladder


def test_ladder_structure(samples):
    for s in samples:
        assert L.ladder_errors(s["questions"], questions.LADDER) == []


def test_bodies_are_never_named_by_shape(samples):
    words = set(questions.SHAPE_WORD.values())
    for s in samples:
        for q in s["questions"]:
            if q["derivation_key"] in ("shape_of", "cf_as_shape"):
                continue
            assert not any(w in q["question"] for w in words), q["question"]


def test_named_instants_are_on_the_clock(samples):
    for s in samples:
        for q in s["questions"]:
            for fld in ("f", "f1", "f2"):
                if fld in q["params"]:
                    f = q["params"][fld]
                    assert sim.aligned(f)
                    assert sim.EDGE_F <= f <= s["params"]["n_frames"] - 1 - sim.EDGE_F
                    assert f"{sim.timer_tenths(f) / 10:.1f}" in q["question"]


def test_balancing_signature_partitions_answers_exactly(scenes):
    """`_sig` drives the pool balancer; if it does not split candidates by their
    answer, the balancing it does is meaningless. The signature need not equal the
    groundtruth, but it must group candidates the same way the groundtruth does."""
    for s in scenes[:20]:
        c = questions.Ctx(s)
        for lv, pool in questions.candidates(c).items():
            seen: dict = {}
            for key, p in pool:
                sig = questions._sig(c, key, p)
                if sig is None:
                    continue
                gt = questions.make(c, lv, key, p, random.Random(0))["groundtruth"]
                prev = seen.setdefault((key, str(sig)), gt)
                assert prev == gt, (key, sig, prev, gt)
            byg: dict = {}
            for (key, sig), gt in seen.items():
                prev = byg.setdefault((key, gt), sig)
                assert prev == sig, (key, gt, prev, sig)


def test_counterfactuals_never_target_a_rule_breaking_body(scenes):
    for s in scenes:
        viol = {r["id"] for r in s["racers"] if r["violates"] != "none"}
        c = questions.Ctx(s)
        for key, p in questions.candidates(c)[5]:
            assert p.get("r") not in viol


# ---------------------------------------------------------------- the closed loop


def test_independent_answers_match(samples):
    for s in samples:
        for q in s["questions"]:
            got = answers.answer(q["derivation_key"], q["params"], s["params"])
            assert got is not None
            assert str(got) == str(q["groundtruth"]), (q["derivation_key"], q["params"])


def test_l6_matches_and_is_balanced(samples):
    yes = 0
    for s in samples:
        l6 = s["params"]["l6"]
        got = answers.answer("l6_consistency", l6["params"], s["params"])
        assert str(got) == l6["groundtruth"]
        yes += l6["answer"] == "A"
    assert 0.35 < yes / len(samples) < 0.65


def test_answers_never_reads_the_generators_own_facts():
    """The re-derivation must not name any field only the generator knows."""
    src = (HERE / "answers.py").read_text()
    body = src.split('"""', 2)[2]                 # skip the module docstring
    assert "import sim" not in body
    # reads of the scene itself are what matter; `p["shape"]` is a question
    # parameter, not a generator fact
    for banned in ("t0", "ramp", "shape", "k_shape", "start", "release_f", "finish_f",
                   "violates", "racers", "view", "twin_of", "scenario", "flawed",
                   "radius", "r_px"):
        assert f'scene["{banned}"]' not in body, banned
    # the only scene keys it may touch
    import re
    used = set(re.findall(r'scene\["([a-z_0-9]+)"\]', body))
    assert used <= {"labels", "n_frames"}, used


def test_a_tampered_answer_is_caught(samples):
    s = json.loads(json.dumps(samples[0]))
    q = s["questions"][0]
    q["groundtruth"] = "definitely wrong"
    got = answers.answer(q["derivation_key"], q["params"], s["params"])
    assert str(got) != q["groundtruth"]


def test_a_tampered_scene_is_caught(scenes):
    s = json.loads(json.dumps(scenes[0]))
    r = next(x for x in s["racers"] if x["shape"] != "slider")
    r["violates"] = "slip"
    t = answers._track(json.dumps(s, sort_keys=True))
    assert answers._consistency(t) == "no"