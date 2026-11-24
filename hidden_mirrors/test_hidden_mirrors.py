"""Tests for hidden_mirrors: python test_hidden_mirrors.py  (or pytest)

Covers determinism (same seed in two processes with different PYTHONHASHSEED
-> identical JSON; identical raw frames), the 10-question ladder counts,
scene uniqueness, the tamper test, min_fps on every question, the cell
labels matching what the renderer draws, the answer-side tracer agreeing
with sim.py's, the mirror map inferred from the drawn tracks agreeing with
the hidden truth, hand-checked reflection rules and pinned seeds.
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

DS = "hidden_mirrors"


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
            render._runs.cache_clear()
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
        for r in sim.runs(sc):                         # question events stay inside the clip
            assert r["t0"] >= sim.EDGE_S and sim.end_time(r) <= sim.duration(sc) - sim.EDGE_S
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
    """The scene's label list is exactly what the static layer draws: same texts, same spots, nothing else."""
    import numpy as np
    from PIL import Image, ImageDraw
    for seed in (3, 4):
        sc = sample(seed)["params"]
        assert [lab["text"] for lab in sc["labels"]] == [sim.LETTERS[c] for c in range(sc["cols"])] + \
            [str(r + 1) for r in range(sc["rows"])]
        assert sc["labels"] == sim.labels(sc)
        img = np.asarray(render._static(json.dumps(sc, sort_keys=True)), dtype=np.int16)
        for lab in sc["labels"]:
            l, t, r, b = font(sim.LABEL_PX).getbbox(lab["text"], anchor="mm")
            assert b - t >= 24                                   # glyphs >= 24 px tall
            x0, y0 = int(lab["x"] + l) - 3, int(lab["y"] + t) - 3
            x1, y1 = int(lab["x"] + r) + 3, int(lab["y"] + b) + 3
            ref = Image.new("RGB", (x1 - x0, y1 - y0), tuple(sc["style"]["bg"]))
            ImageDraw.Draw(ref).text((lab["x"] - x0, lab["y"] - y0), lab["text"], fill=tuple(sc["style"]["ink"]),
                                     font=font(sim.LABEL_PX), anchor="mm")
            assert np.abs(img[y0:y1, x0:x1] - np.asarray(ref, dtype=np.int16)).mean() < 1.0


def _scene(cols, rows, mirrors, pulses, blocks=(), step=0.2):
    sc = {"scenario": "classic", "cols": cols, "rows": rows, "step": step,
          "mirrors": [{"cell": list(c), "kind": k, "shown": False} for c, k in mirrors],
          "blocks": [list(b) for b in blocks],
          "pulses": [{"colour": col, "edge": e, "index": i, "t0": t0} for col, e, i, t0 in pulses], "frames": 120}
    sc["style"] = {"theme": "dark", "bg": [16, 18, 26], "board": [38, 42, 54], "board_pat": "plain",
                   "grid": [78, 84, 100], "grid_style": "solid", "ink": [235, 235, 240], "block": [8, 8, 10],
                   "mirror": [150, 220, 255], "mirror_w": 6, "trail_w": 6, "head_px": 24, "head_shape": "disc",
                   "marker": "wedge", "x0": 400, "y0": 150, "cell": 80}
    sc["labels"] = sim.labels(sc)
    return sc


def test_hand_checked_rules():
    # '/' in C2 (column C = 2, row 2 = 1): fired in at the left end of row 2 at 0.625 s, the pulse crosses A2, B2,
    # turns UP in C2 and leaves at the top of column C after 4 cells: 0.625 + 4 * 0.2 = 1.425 s
    sc = _scene(5, 4, [((2, 1), "/")], [("gold", "left", 1, 0.625)])
    r = sim.runs(sc)[0]
    assert r["end"][:3] == ("exit", "top", 2) and abs(r["end"][3] - 1.425) < 1e-9
    assert [tuple(t[1]) for t in r["turns"]] == [(2, 1)] and len(r["cells"]) == 4
    assert answers.answer("exit_port", {"colour": "gold"}, sc) == "leaves at the top of column C"
    assert answers.answer("n_cells", {"colour": "gold"}, sc) == "4"
    assert answers.answer("time_on_board", {"colour": "gold"}, sc) == "0.8 s"
    assert answers.answer("first_turn_dir", {"colour": "gold"}, sc) == "up"
    assert answers.answer("locate_mirror", {"colour": "gold", "k": 1}, sc) == "C2, /"
    # exit at 1.425 s: frame 34.2 -> first frame after it is 35, clock 35 * 10 // 24 = 14 tenths
    assert answers.answer("exit_clock", {"colour": "gold"}, sc) == "1.4 s"
    # reversed: fired in at the top of column C it comes out at the left end of row 2
    assert answers.answer("cf_reverse", {"colour": "gold"}, sc) == "leaves at the left end of row 2"
    # flipping C2 to '\' sends it DOWN: C3, C4, out at the bottom of column C - but C3 / C4 were never seen
    assert answers.answer("cf_change", {"colour": "gold", "cell": [2, 1], "change": "flip"}, sc) is None
    # removing it: straight on through D2, E2 - unseen too
    assert answers.answer("cf_change", {"colour": "gold", "cell": [2, 1], "change": "remove"}, sc) is None
    sc = _scene(5, 4, [((2, 1), "/")], [("gold", "left", 1, 0.625), ("blue", "top", 3, 2.0),
                                        ("pink", "top", 4, 3.5)])
    # with D2 and E2 seen (blue down column D, pink down column E), removing C2 is now knowable
    assert answers.answer("cf_change", {"colour": "gold", "cell": [2, 1], "change": "remove"}, sc) == \
        "leaves at the right end of row 2"
    # seen: A2, B2, C2, C1 (gold), column D (blue), column E (pink) -> 12 of 20 cells, 8 never entered
    assert answers.answer("n_unvisited", {}, sc) == "8"
    # '\' instead of '/': turns DOWN
    sc = _scene(5, 4, [((2, 1), "\\")], [("gold", "left", 1, 0.625)])
    assert sim.runs(sc)[0]["end"][:3] == ("exit", "bottom", 2)
    assert answers.answer("exit_port", {"colour": "gold"}, sc) == "leaves at the bottom of column C"
    # a block in D2 stops a pulse going right along row 2 at the centre of D2: 0.625 + 3.5 * 0.2 = 1.325 s
    sc = _scene(5, 4, [], [("gold", "left", 1, 0.625)], blocks=[(3, 1)])
    r = sim.runs(sc)[0]
    assert r["end"][0] == "stopped" and tuple(r["end"][1]) == (3, 1) and abs(r["end"][2] - 1.325) < 1e-9
    assert answers.answer("exit_port", {"colour": "gold"}, sc) == "is stopped by the block in D2"
    assert answers.answer("cf_change", {"colour": "gold", "cell": [3, 1], "change": "unblock"}, sc) is None


def test_tracers_agree():
    """answers.walk on the full hidden map gives the same ends and turn counts as sim.trace_one."""
    for seed in range(80):
        sc = sample(seed)["params"]
        full = {(c, r): "empty" for c in range(sc["cols"]) for r in range(sc["rows"])}
        full.update({tuple(m["cell"]): m["kind"] for m in sc["mirrors"]})
        full.update({tuple(b): "block" for b in sc["blocks"]})
        for port in sim.all_ports(sc):
            a = answers.walk(sc["cols"], sc["rows"], full, port)
            b = sim.trace_one(sc, port, 0.0)
            e = ("exit", (b["end"][1], b["end"][2])) if b["end"][0] == "exit" else ("stopped", tuple(b["end"][1]))
            assert a["end"] == e and a["turns"] == len(b["turns"]) and a["cells"] == len(b["cells"]), (seed, port)


def test_inferred_map_matches_truth():
    """What the drawn tracks reveal is never wrong: inferred mirrors are real with the right slant, inferred
    empty cells have no mirror."""
    for seed in range(80):
        sc = sample(seed)["params"]
        tr = answers.Track(sc, answers.drawn(sc))
        truth = {tuple(m["cell"]): m["kind"] for m in sc["mirrors"]}
        blocks = {tuple(b) for b in sc["blocks"]}
        for cell, k in tr.known_map().items():
            want = "block" if cell in blocks else truth.get(cell, "empty")
            assert k == want, (seed, cell, k, want)
        assert tr.visited() == {tuple(c) for r in sim.runs(sc) for c in r["cells"]}, seed


def test_pinned_seeds():
    got = {}
    for seed in PINNED:
        s = sample(seed)
        got[seed] = (s["params"]["scenario"], [(q["derivation_key"], q["answer_text"]) for q in s["questions"]])
    assert got == PINNED, json.dumps(got)


# Hand check of seed 0 (blocks, 7 x 7, 0.182 s per cell, blocks A3 G3 A6): orange is fired in at the right end of
# row 1, crosses G1, turns down at F1 ('/'), left at F2 ('/'), crosses E2..B2, turns up at A2 ('\\') and left at A1
# ('\\'), leaving at the left end of row 1; fired back in there it retraces to the right end of row 1. Red (right
# end of row 6) turns first at G6. Turns: orange 4 + red 1 + gold 1 + blue 1 = 7. Gold is fired at 3.083 s, blue at
# 3.958 s: 0.875 s apart -> "0.9 s" (0.025 s from the boundary, the safe1 limit). Speed 1 / 0.182 = 5.49 -> "5.5".
# A pulse fired in at the right end of row 3 enters G3, a block, and is stopped there.
PINNED: dict = {
    0: ('blocks', [
        ('n_pulses', '4'),
        ('n_straight', '0'),
        ('exit_port', 'leaves at the left end of row 1'),
        ('first_turn_dir', 'down'),
        ('first_turn_cell', 'G6'),
        ('launch_gap', '0.9 s'),
        ('speed', '5.5 cells per second'),
        ('total_turns', '7'),
        ('cf_new_port', 'is stopped by the block in G3'),
        ('cf_reverse', 'leaves at the right end of row 1'),
    ]),
    1: ('rect', [
        ('n_straight', '0'),
        ('n_pulses', '3'),
        ('exit_order', 'red, blue, pink'),
        ('exit_port', 'leaves at the bottom of column B'),
        ('first_turn_dir', 'down'),
        ('turn_clock', '2.0 s'),
        ('launch_gap', '1.6 s'),
        ('locate_mirror', 'C1, /'),
        ('must_mirror', 'C1'),
        ('cf_change', 'leaves at the top of column B'),
    ]),
    2: ('classic', [
        ('entry_port', 'right end of row 1'),
        ('first_turn_cell', 'B1'),
        ('exit_port', 'leaves at the bottom of column C'),
        ('launch_order', 'green, red, white'),
        ('launch_gap', '1.0 s'),
        ('speed', '5.0 cells per second'),
        ('time_on_board', '0.4 s'),
        ('total_turns', '5'),
        ('n_unvisited', '2'),
        ('cf_change_turns', '0'),
    ]),
    3: ('rect', [
        ('n_pulses', '5'),
        ('entry_port', 'top of column D'),
        ('n_straight', '1'),
        ('launch_order', 'orange, black, red'),
        ('first_turn_cell', 'C2'),
        ('exit_order', 'black, purple, red'),
        ('turn_clock', '2.2 s'),
        ('speed', '4.5 cells per second'),
        ('total_turns', '4'),
        ('cf_change_turns', '0'),
    ]),
    4: ('rect', [
        ('n_pulses', '5'),
        ('exit_order', 'white, gold, green'),
        ('first_turn_dir', 'right'),
        ('launch_gap', '2.6 s'),
        ('exit_clock', '2.3 s'),
        ('speed', '4.5 cells per second'),
        ('certainly_empty', 'G3'),
        ('cf_reverse', 'leaves at the bottom of column C'),
        ('cf_new_port', 'leaves at the right end of row 2'),
        ('cf_change', 'leaves at the right end of row 2'),
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
                print("FAIL", name_, repr(e)[:400])
    sys.exit(1 if fails else 0)
