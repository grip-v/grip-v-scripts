"""Tests for mirror_beam_reflection: python test_mirror_beam_reflection.py  (or pytest)

Covers determinism (same seed in two processes with different PYTHONHASHSEED
-> identical JSON; identical raw frames), the 10-question ladder counts,
scene uniqueness, the tamper test, min_fps on every question, the cell
labels matching what the renderer draws, the two tracers agreeing, and
hand-checked edge cases of the reflection rules plus pinned seeds.
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
from common.video5 import font  # noqa: E402
import sim, questions, answers, render  # noqa: E402

DS = "mirror_beam_reflection"


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
            render._trace.cache_clear()
        assert hs[0] == hs[1]


def test_ladder_uniqueness_minfps_answers():
    sigs = set()
    for seed in range(120):
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
            if "t" in q["params"]:
                assert sim.EDGE_S <= q["params"]["t"] <= sim.duration(sc) - sim.EDGE_S
        assert render.n_frames(sc) <= sim.MAX_FRAMES
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


def test_labels_match_the_renderer():
    """The scene's label list is exactly what the static layer draws: same texts, same spots, nothing else."""
    import numpy as np
    from PIL import Image, ImageDraw
    sc = sample(3)["params"]
    assert [l["text"] for l in sc["labels"]] == [sim.LETTERS[c] for c in range(sc["cols"])] + \
        [str(r + 1) for r in range(sc["rows"])]
    assert sc["labels"] == sim.labels(sc)
    img = np.asarray(render._static(json.dumps(sc, sort_keys=True)), dtype=np.int16)
    for lab in sc["labels"]:
        l, t, r, b = font(sim.LABEL_PX).getbbox(lab["text"], anchor="mm")
        assert b - t >= 24                                   # glyphs >= 24 px tall
        x0, y0 = int(lab["x"] + l) - 3, int(lab["y"] + t) - 3            # the glyph box + 3 px
        x1, y1 = int(lab["x"] + r) + 3, int(lab["y"] + b) + 3
        ref = Image.new("RGB", (x1 - x0, y1 - y0), tuple(sc["style"]["bg"]))
        ImageDraw.Draw(ref).text((lab["x"] - x0, lab["y"] - y0), lab["text"], fill=tuple(sc["style"]["ink"]),
                                 font=font(sim.LABEL_PX), anchor="mm")
        crop = img[y0:y1, x0:x1]
        assert np.abs(crop - np.asarray(ref, dtype=np.int16)).mean() < 1.0


def _scene(cols, rows, mirrors, emitters, absorbers=(), step=0.3):
    return {"cols": cols, "rows": rows, "step": step, "mirrors": mirrors, "absorbers": [list(a) for a in absorbers],
            "emitters": emitters}


def test_hand_checked_rules():
    em = [{"edge": "left", "index": 1, "t0": 0.5, "colour": "red"}]
    # '/' in C2 (col 2, row 1): coming from the left the pulse turns UP and leaves through the top at column C
    sc = _scene(5, 4, [{"cell": [2, 1], "kind": "/", "type": "two"}], em)
    r = answers.run(sc)
    assert r[0]["end"] == ("exit", "top", 2) and r[0]["turns"] == 1
    assert sim.trace(sc)["pulses"][0]["end"][:3] == ("exit", "top", 2)
    assert answers.end_text(r[0]["end"]) == "leaves through the top edge at column C"
    # '\' instead: turns DOWN and leaves through the bottom
    sc["mirrors"][0]["kind"] = "\\"
    assert answers.run(sc)[0]["end"] == ("exit", "bottom", 2)
    # one-sided '\' whose silvered face looks down-left (face (-1, 1)): the pulse from the left meets the back -> stopped
    sc["mirrors"][0].update(type="one", face=[1, -1])
    assert answers.run(sc)[0]["end"] == ("absorbed", (2, 1))
    assert sim.trace(sc)["pulses"][0]["end"][0] == "absorbed"
    sc["mirrors"][0]["face"] = [-1, 1]
    assert answers.run(sc)[0]["end"] == ("exit", "bottom", 2)
    # half-mirror: one copy goes on to the right edge, one turns down
    sc["mirrors"][0] = {"cell": [2, 1], "kind": "\\", "type": "split"}
    assert sorted(o["end"] for o in answers.run(sc)) == [("exit", "bottom", 2), ("exit", "right", 1)]
    assert len(sim.trace(sc)["pulses"]) == 2
    # pivot: '/' turning to '\' at 0.5 s; the pulse reaches C2 at 0.5 + 0.15 + 2 * 0.3 = 1.25 s
    sc["mirrors"][0] = {"cell": [2, 1], "kind": "/", "type": "flip", "flip_t": 2.0}
    assert answers.run(sc)[0]["end"] == ("exit", "top", 2)          # reached before the turn: still '/'
    sc["mirrors"][0]["flip_t"] = 1.0
    assert answers.run(sc)[0]["end"] == ("exit", "bottom", 2)       # after the turn: '\'
    assert answers.run(sc, no_flip=True)[0]["end"] == ("exit", "top", 2)
    # timing: exit through the top edge at 1.25 + 0.3 + 0.15 = 1.70 s (two more cells up, half a cell to the edge)
    sc["mirrors"][0]["flip_t"] = 2.0
    assert abs(answers.run(sc)[0]["t_end"] - 1.70) < 1e-9
    assert abs(sim.trace(sc)["pulses"][0]["end"][-1] - 1.70) < 1e-9
    # an absorber in D2 stops a pulse going right
    sc2 = _scene(5, 4, [], em, absorbers=[(3, 1)])
    assert answers.run(sc2)[0]["end"] == ("absorbed", (3, 1))


def test_tracers_agree():
    for seed in range(60):
        sc = sample(seed)["params"]
        a = sorted((o["colour"], o["end"], round(o["t_end"], 9)) for o in answers.run(sc))
        b = sorted((p["colour"], (p["end"][0],) + tuple(tuple(x) if isinstance(x, list) else x
                                                       for x in p["end"][1:-1]), round(p["end"][-1], 9))
                   for p in sim.trace(sim.world_of(sc))["pulses"])
        assert a == b, seed


def test_pinned_seeds():
    got = {}
    for seed in PINNED:
        s = sample(seed)
        got[seed] = (s["params"]["scenario"], [(q["derivation_key"], q["answer_text"]) for q in s["questions"]])
    assert got == PINNED, json.dumps(got)


# Hand check of seed 0 (flip): magenta enters F1 at 0.735 s, 0.392 s per cell. F3 '/' turns it left at 1.715 s,
# D3 '\\' up at 2.499 s (0.784 s and 2 cells later -> "0.8 s", 2), D2 '/' right at 2.891 s, pivot E2 at 3.283 s:
# 4 turns by 4.0 s. At 2.0 s the head is 0.73 cell left of F3's centre -> E3. With the pivot frozen as '/',
# E2 sends it up: E1 '\\' left, D1 '/' down, D2 '/' left, then out through the left edge at row 2.
PINNED: dict = {
    0: ('flip', [
        ('entry_edge', 'top'),
        ('n_slash', '5'),
        ('n_turns_by_t', '4'),
        ('head_cell_at_t', 'E3'),
        ('cells_between', '2'),
        ('time_between_turns', '0.8 s'),
        ('time_after_end', '1.3 s'),
        ('cf_exit_point', 'is stopped in cell C3'),
        ('cf_exit_point', 'leaves through the left edge at row 2'),
        ('cf_exit_point', 'leaves through the right edge at row 5'),
    ]),
    1: ('one_sided', [
        ('n_special', '2'),
        ('n_slash', '1'),
        ('head_cell_at_t', 'D2'),
        ('n_turns_by_t', '0'),
        ('n_heads_at_t', '1'),
        ('cells_between', '1'),
        ('n_lit_cells', '11'),
        ('exit_time', '6.4 s'),
        ('n_turns_total', '6'),
        ('cf_exit_point', 'leaves through the bottom edge at column D'),
    ]),
    2: ('classic', [
        ('n_slash', '6'),
        ('turn_dir_first', 'left'),
        ('n_turns_by_t', '4'),
        ('speed', '3.6 cells per second'),
        ('time_between_mirrors', '0.8 s'),
        ('exit_time', '8.0 s'),
        ('time_after_end', '1.4 s'),
        ('cf_exit_point', 'leaves through the bottom edge at column D'),
        ('cf_exit_point', 'leaves through the top edge at column F'),
        ('cf_exit_point', 'leaves through the bottom edge at column E'),
    ]),
    5: ('two_beams', [
        ('n_mirrors', '12'),
        ('n_slash', '4'),
        ('head_cell_at_t', 'A4'),
        ('n_turns_by_t', '4'),
        ('cells_between', '1'),
        ('n_lit_cells', '17'),
        ('speed', '2.6 cells per second'),
        ('time_after_end', '2.1 s'),
        ('exit_point', 'leaves through the right edge at row 4'),
        ('cf_exit_point', 'leaves through the right edge at row 4'),
    ]),
    7: ('splitter', [
        ('n_special', '2'),
        ('n_turns_by_t', '0'),
        ('n_heads_at_t', '1'),
        ('head_cell_at_t', 'G5'),
        ('speed', '2.9 cells per second'),
        ('time_between_mirrors', '0.7 s'),
        ('time_turn_at', '2.2 s'),
        ('n_exits', '3'),
        ('cf_n_exits', '3'),
        ('cf_n_exits', '1'),
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
