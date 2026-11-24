"""Tests for glasses_fill: determinism, ladder counts, uniqueness, tamper
detection, min_fps, label data vs the renderer, and hand-checked scenes.

    python test_glasses_fill.py          (or: pytest test_glasses_fill.py)
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

DS = "glasses_fill"
SEEDS = range(0, 120)


def _sample(seed):
    return _roundtrip(build_sample(DS, sim, questions, seed))


def test_same_seed_same_json_in_process():
    for seed in (0, 7, 123):
        assert json.dumps(_sample(seed), sort_keys=True) == json.dumps(_sample(seed), sort_keys=True)


def test_same_seed_same_json_across_processes():
    code = ("import sys,json;sys.path[:0]=[{h!r},{p!r}];from common.video5 import build_sample,_roundtrip;"
            "import sim,questions;print(json.dumps([_roundtrip(build_sample('glasses_fill',sim,questions,s)) "
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
        assert sc["n_frames"] <= 240
        sigs.add(hashlib.sha1(json.dumps({k: v for k, v in sc.items() if k != "attempts"}, sort_keys=True)
                              .encode()).hexdigest())
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


def test_labels_match_renderer():
    """The label list in the scene is what the renderer draws: one ruler number per 20 ml at the
    tick's height, one number per glass under its centre, all >= 24 px tall and not overlapping."""
    import numpy as np
    import render
    sc = _sample(4)["params"]
    scale = [lab for lab in sc["labels"] if lab["kind"] == "scale"]
    glass = [lab for lab in sc["labels"] if lab["kind"] == "glass"]
    assert [lab["text"] for lab in scale] == ["0", "20", "40", "60", "80", "100"]
    assert [lab["text"] for lab in glass] == [str(g + 1) for g in range(sc["glasses"]["n"])]
    assert all(lab["x"] == sc["glasses"]["cx"][g] for g, lab in enumerate(glass))
    boxes = [render.label_box(sc, lab) for lab in sc["labels"]]
    assert min(b[3] - b[1] for b in boxes) >= 24
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])
    # every label's pixels are on the frame: the box differs from a label-free background
    img = np.asarray(render.frame(sc, 0.0)).astype(int)
    bg = np.array([int(sc["palette"]["bg"][i:i + 2], 16) for i in (1, 3, 5)])
    for b in boxes:
        crop = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
        assert (np.abs(crop - bg).sum(axis=2) > 200).mean() > 0.1


# ------------------------------------------------------------------ hand-checked scenes
def _scene(jugs, plan, leaks=(), n=5, rate=60.0):
    """plan: (jug, glass, ask); timings 0.4 s moves, 0.2 s tilts, every stream >= 0.5 s."""
    left = [j[2] for j in jugs]
    pours, t = [], 0.6
    for j, g, ask in plan:
        got = min(ask, left[j])
        left[j] -= got
        r = min(rate, got / 0.5) if got else rate
        flow = got / r
        hold = 0.8 if got == 0 else (flow + 0.5 if got < ask else flow)
        p = {"jug": j, "glass": g, "ask_ml": ask, "got_ml": got, "rate_ml_s": r, "t_move": t, "t_tilt": t + 0.4,
             "t_pour": t + 0.6, "t_flow_end": t + 0.6 + flow, "t_untilt": t + 0.6 + hold,
             "t_back": t + 0.8 + hold, "t_end": t + 1.2 + hold}
        pours.append(p)
        t = p["t_end"] + 0.15
    lk = []
    for g, hole, tc in leaks:
        peak = sum(p["got_ml"] for p in pours if p["glass"] == g)
        lk.append({"glass": g, "hole_ml": hole, "rate_ml_s": 30.0, "t_crack": tc, "t_drain_end": tc + (peak - hole) / 30})
    end = max([p["t_end"] for p in pours] + [x["t_drain_end"] for x in lk])
    nf = sim.frame_of(end + 0.8)
    slot = (sim.GLASS_X1 - sim.GLASS_X0) / n
    cx = [int(round(sim.GLASS_X0 + slot * (g + 0.5))) for g in range(n)]
    return {"scenario": "hand", "n_frames": nf, "duration": nf / 24,
            "palette": {"bg": "#F7F5EF", "guide": "#E4DDCF", "table": "#C9B79C", "jug_body": "#FFFFFF"},
            "glasses": {"n": n, "cx": cx, "w": 80},
            "jugs": [{"pattern": a, "colour": b, "start_ml": c} for a, b, c in jugs],
            "pours": pours, "leaks": lk, "labels": [], "label_px": 34, "clock_xy": [1262, 14]}


JUGS = [("striped", "red", 100), ("dotted", "blue", 60), ("wavy", "green", 0)]


def test_hand_checked_layers_and_dry():
    # striped 50 -> glass 1; dotted 30 -> glass 1 (on top); wavy (empty) tilts over glass 3; striped 40 -> glass 2
    sc = _scene(JUGS, [(0, 0, 50), (1, 0, 30), (2, 2, 50), (0, 1, 40)])
    a = answers.answer
    assert a("level_end", {"glass": 0}, sc) == "80 ml"
    assert a("bottom_layer", {"glass": 0}, sc) == "red" and a("top_layer", {"glass": 0}, sc) == "blue"
    assert a("layer_ml", {"glass": 0, "colour": "blue"}, sc) == "30 ml"
    assert a("dry_jug", {}, sc) == "the wavy jug"
    assert a("n_tilts", {}, sc) == 4 and a("n_received", {}, sc) == 2
    assert a("jug_total", {"jug": 0}, sc) == "90 ml"
    assert a("first_jug", {}, sc) == "the striped jug"
    assert a("jug_after", {"jug": 1}, sc) == "the wavy jug"
    assert a("last_glass", {}, sc) == "glass 2"
    assert a("cf_swap", {"glass": 0}, sc) == "blue"
    # skipping the first striped pour leaves 100 in it: glass 2 still gets its 40 (asked 40)
    assert a("cf_skip", {"pour": 0, "glass": 0}, sc) == "30 ml"
    assert a("cf_redirect", {"pour": 1, "glass": 3}, sc) == "30 ml"
    assert a("n_empty_end", {}, sc) == 3 and a("total_end", {}, sc) == "120 ml"


def test_hand_checked_runs_dry():
    # dotted holds 60: asked 50 into glass 1 (gets 50), then asked 40 into glass 2 -> only 10 come out
    sc = _scene(JUGS, [(1, 0, 50), (1, 1, 40), (0, 2, 70)])
    a = answers.answer
    assert a("pour_ml", {"pour": 1}, sc) == "10 ml"
    assert a("ran_dry", {}, sc) == "the dotted jug"
    # without the first dotted pour the jug still holds 60, so glass 2 would get the full 40
    assert a("cf_skip", {"pour": 0, "glass": 1}, sc) == "40 ml"
    assert a("cf_skip", {"pour": 1, "glass": 1}, sc) == "0 ml"


def test_hand_checked_crack():
    # glass 2: 70 red; cracks at the 30 ml mark 0.6 s after its pour ends and drains to 30
    sc = _scene(JUGS, [(0, 1, 70), (1, 3, 20)])
    tc = sc["pours"][0]["t_flow_end"] + 0.6
    sc = _scene(JUGS, [(0, 1, 70), (1, 3, 20)], leaks=[(1, 30, tc)])
    a = answers.answer
    assert a("level_end", {"glass": 1}, sc) == "30 ml"
    assert a("leak_loss", {"glass": 1}, sc) == "40 ml"
    assert a("cf_no_crack", {"glass": 1}, sc) == "70 ml"
    assert a("cf_no_crack_total", {}, sc) == "90 ml"
    assert a("crack_mark", {"glass": 1}, sc) == "30 ml"
    assert a("cracked_one", {}, sc) == "glass 2"


def test_clock_edges():
    assert sim.clock_tenths(23) == 9 and sim.clock_tenths(24) == 10 and sim.clock_tenths(239) == 99
    assert sim.frame_of(1.0) == 24 and sim.frame_of(1.0001) == 25


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
