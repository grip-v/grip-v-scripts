"""Frame-level check of rendered hidden_mirrors clips against the stored annotations.

    python frame_check.py --dataset-dir <dir> [--report frame_check_report.txt]

Per clip it decodes the MP4 (OpenCV) and checks:
  * clock: every frame's clock box matches the rendered text for that frame
    better than any other reading 0.0-10.0 s (template match = OCR);
  * labels: at the first, middle and last frame each cell label's crop
    matches its own text better than any other label text; glyphs are at
    least 24 px tall and never overlap each other, the board, a port marker
    or the clock;
  * heads: pulse heads located in each frame (pulse-colour pixels eroded
    past the trail width, only inside the board, filled on the diagonals so a
    trail crossing is not taken for a head) agree with render.state within
    TOL px, same set of colours;
  * mirrors stay hidden: no mirror-coloured line in any hidden mirror's cell;
    every drawn mirror (partial) shows as a line on its stated diagonal;
    blocks: the dark-block pixels form exactly the stored blocks;
  * ports: 2 frames after a pulse is fired its entry marker shows its
    colour, and 2 frames after it leaves its exit marker does;
then re-derives EVERY answer from the decoded heads with
answers.answer(..., frames=decoded, jitter=JITTER) (counts, ports, orders,
times, mirror locations, and the what-ifs on the mirror map inferred from
the decoded tracks). Questions whose answer is a clock reading get the clock
checked at that frame of the decoded clip.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import cv2  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import sim  # noqa: E402
import render  # noqa: E402
import answers  # noqa: E402
from common.video5 import font  # noqa: E402

TOL = 3.0
JITTER = 2.0
SCENE_KEYS = {"n_blocks", "n_shown"}          # read from the drawing; checked by the pixel tests below


def decode(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


_CLOCKS: dict = {}


def clock_img(tenths):
    if tenths not in _CLOCKS:
        img = Image.new("RGB", render.SIZE, (0, 0, 0))
        render._clock(ImageDraw.Draw(img), (tenths * 24 + 9) // 10)
        x0, y0, x1, y1 = render.CLOCK_BOX
        _CLOCKS[tenths] = np.asarray(img, dtype=np.int16)[y0 + 4:y1 - 3, x0 + 4:x1 - 3]
    return _CLOCKS[tenths]


def read_clock(frame):
    x0, y0, x1, y1 = render.CLOCK_BOX
    crop = frame[y0 + 4:y1 - 3, x0 + 4:x1 - 3].astype(np.int16)
    diffs = {k: float(np.abs(crop - clock_img(k)).mean()) for k in range(0, 101)}
    return min(diffs, key=diffs.get)


def label_box(lab):
    l, t, r, b = font(sim.LABEL_PX).getbbox(lab["text"], anchor="mm")
    return (lab["x"] + l, lab["y"] + t, lab["x"] + r, lab["y"] + b)


def marker_box(scene, port):
    m = render.marker_poly(scene, port)
    if isinstance(m, tuple):
        return (m[0] - m[2], m[1] - m[2], m[0] + m[2], m[1] + m[2])
    xs, ys = [p[0] for p in m], [p[1] for p in m]
    return (min(xs), min(ys), max(xs), max(ys))


def label_errors(scene, frames):
    errs = []
    st = scene["style"]
    texts = sorted({lab["text"] for lab in scene["labels"]} | set(sim.LETTERS[:10]) | {str(k) for k in range(1, 11)})
    boxes = [label_box(lab) for lab in scene["labels"]]
    x0, y0, c = st["x0"], st["y0"], st["cell"]
    board = (x0, y0, x0 + scene["cols"] * c, y0 + scene["rows"] * c)
    markers = [marker_box(scene, p) for p in sim.all_ports(scene)]
    cb = render.CLOCK_BOX
    hit = lambda a, b, g=0: not (a[2] + g <= b[0] or b[2] + g <= a[0] or a[3] + g <= b[1] or b[3] + g <= a[1])
    for i, b in enumerate(boxes):
        name = scene["labels"][i]["text"]
        if b[3] - b[1] < 24:
            errs.append(f"label {name} glyphs under 24 px tall")
        if hit(b, board, 2) or hit(b, cb, 4) or any(hit(b, m, 2) for m in markers):
            errs.append(f"label {name} touches the board, a port marker or the clock")
        for j in range(i + 1, len(boxes)):
            if hit(b, boxes[j], 2):
                errs.append(f"labels {name} and {scene['labels'][j]['text']} overlap")
    for f in (0, len(frames) // 2, len(frames) - 1):
        fr = frames[f].astype(np.int16)
        for lab in scene["labels"]:
            x, y = int(lab["x"]), int(lab["y"])
            box = (x - 22, y - 20, x + 22, y + 20)
            crop = fr[box[1]:box[3], box[0]:box[2]]
            scores = {}
            for t in texts:
                im = Image.new("RGB", (44, 40), tuple(st["bg"]))
                ImageDraw.Draw(im).text((lab["x"] - box[0], lab["y"] - box[1]), t, fill=tuple(st["ink"]),
                                        font=font(sim.LABEL_PX), anchor="mm")
                scores[t] = float(np.abs(crop - np.asarray(im, dtype=np.int16)).mean())
            best = min(scores, key=scores.get)
            if best != lab["text"]:
                errs.append(f"frame {f}: label at ({x},{y}) reads {best!r}, scene says {lab['text']!r}")
    return errs


def colour_mask(frame, rgb, tol=75):
    d = np.abs(frame.astype(np.int16) - np.array(rgb, dtype=np.int16)).sum(-1)
    return (d < tol).astype(np.uint8)


def board_rect(scene, pad=0.0):
    st = scene["style"]
    x0, y0, c = st["x0"], st["y0"], st["cell"]
    return (x0 - pad, y0 - pad, x0 + scene["cols"] * c + pad, y0 + scene["rows"] * c + pad)


def near_edge(scene, x, y):
    """A head straddling the board edge is cut by the blanked port marker: not located reliably."""
    b = board_rect(scene)
    return min(abs(x - b[0]), abs(x - b[2]), abs(y - b[1]), abs(y - b[3])) < 0.5 * scene["style"]["head_px"] + 3


def find_heads(frame, scene):
    st = scene["style"]
    k = st["trail_w"] + 3
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    b = board_rect(scene)
    boxes = [marker_box(scene, p) for p in sim.all_ports(scene)]
    out = []
    for p in scene["pulses"]:
        name = p["colour"]
        full = colour_mask(frame, render.colour(scene, name))
        for bx in boxes:                            # lit port markers share the pulse colour: blank them
            full[int(bx[1]) - 1:int(math.ceil(bx[3])) + 2, int(bx[0]) - 1:int(math.ceil(bx[2])) + 2] = 0
        m = cv2.erode(full, kern)
        n, _, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 4:
                continue
            x, y = float(cent[i][0]), float(cent[i][1])
            if not (b[0] <= x <= b[2] and b[1] <= y <= b[3]):
                continue                            # lit port markers sit outside the board
            # a trail crossing (a plus sign of width ~trail_w + 2) survives the erosion; a head is filled on
            # the diagonals too, past where the plus's arms reach
            rd = 0.71 * (st["trail_w"] + 2) + 1.0
            pts = [(x + rd * dx, y + rd * dy) for dx, dy in ((0.7071, 0.7071), (-0.7071, 0.7071),
                                                             (0.7071, -0.7071), (-0.7071, -0.7071))]
            inside = [p for p in pts if b[0] < p[0] < b[2] and b[1] < p[1] < b[3]]    # a head on the edge is cut
            if sum(int(full[int(round(p[1])), int(round(p[0]))]) for p in inside) < 2:     # a plus scores 0
                continue
            out.append((name, x, y))
    return out


def mirror_errors(scene, frame):
    """Hidden mirrors leave no mirror-coloured line; drawn ones lie on their diagonal."""
    errs = []
    st = scene["style"]
    c = st["cell"]
    mask = colour_mask(frame, st["mirror"], 60)
    for m in scene["mirrors"]:
        cx, cy = render.px(scene, m["cell"][0] + 0.5, m["cell"][1] + 0.5)
        score = {}
        for name, sgn in (("/", -1), ("\\", 1)):
            hits = 0
            for u in np.linspace(-0.3, 0.3, 13):
                x, y = int(round(cx + u * c)), int(round(cy + sgn * u * c))
                hits += int(mask[y, x])
            score[name] = hits
        if m["shown"]:
            if score[m["kind"]] < 10 or score[m["kind"]] <= score[sim.flipped(m["kind"])]:
                errs.append(f"drawn mirror {answers.cell_name(m['cell'])} does not read as {m['kind']!r} ({score})")
        elif max(score.values()) > 1:
            errs.append(f"hidden mirror {answers.cell_name(m['cell'])} is visible ({score})")
    return errs


def block_errors(scene, frame):
    """The dark-block pixels inside the board form exactly the stored blocks."""
    st = scene["style"]
    c = st["cell"]
    x0, y0, x1, y1 = (int(v) for v in board_rect(scene))
    mask = colour_mask(frame[y0:y1, x0:x1], st["block"], 40)
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=4)
    found = sorted((math.floor(cent[i][0] / c), math.floor(cent[i][1] / c))
                   for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.4 * c * c)
    want = sorted(tuple(x) for x in scene["blocks"])
    return [] if found == want else [f"dark blocks found at {found}, scene has {want}"]


def port_errors(scene, frames):
    errs = []
    for r in render.traced(scene):
        rgb = render.colour(scene, r["colour"])
        events = [(r["t0"], tuple(r["port"]), "entry")]
        if r["end"][0] == "exit":
            events.append((r["end"][-1], (r["end"][1], r["end"][2]), "exit"))
        for t, port, what in events:
            f = math.ceil(t * 24 - 1e-9) + 2
            if f >= len(frames):
                continue
            bx = marker_box(scene, port)
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            if scene["style"]["marker"] == "wedge":                  # centroid of the triangle
                poly = render.marker_poly(scene, port)
                cx, cy = sum(p[0] for p in poly) / 3, sum(p[1] for p in poly) / 3
            patch = frames[f][int(cy) - 2:int(cy) + 3, int(cx) - 2:int(cx) + 3].reshape(-1, 3).mean(0)
            pal = sim.PULSES[scene["style"]["theme"]]
            best = min(pal, key=lambda k: float(np.abs(patch - np.array(pal[k])).sum()))
            if best != r["colour"] or float(np.abs(patch - np.array(rgb)).sum()) > 60:
                errs.append(f"{r['colour']} {what} marker {port} at frame {f} reads {best} ({patch.round()})")
    return errs


def check_clip(s, root):
    scene = s["params"]
    errs, info = [], {"frames": 0, "clock_ok": 0, "q_checked": 0, "head_err": 0.0, "labels": 0, "skipped": 0}
    frames = decode(root / s["media_path"])
    N = render.n_frames(scene)
    if len(frames) != N:
        return [f"decoded {len(frames)} frames, expected {N}"], info
    info["frames"] = N
    decoded = []
    for f, fr in enumerate(frames):
        if read_clock(fr) == sim.clock_tenths(f):
            info["clock_ok"] += 1
        else:
            errs.append(f"frame {f}: clock misread")
        S = render.state(scene, f)
        got = find_heads(fr, scene)
        want = [h for h in S["heads"] if not near_edge(scene, h[1], h[2])]
        if len(want) != len(S["heads"]):              # launch / exit frames: compared only when found
            info["skipped"] += 1
            edge_cols = {h[0] for h in S["heads"]} - {h[0] for h in want}
            cmp_got = [g for g in got if g[0] not in edge_cols]
        else:
            cmp_got = got
        if sorted(h[0] for h in cmp_got) != sorted(h[0] for h in want):
            errs.append(f"frame {f}: heads found {sorted(h[0] for h in got)}, drawn {sorted(h[0] for h in want)}")
        for h in want:
            near = [g for g in got if g[0] == h[0]]
            if near:
                e = min(math.dist(g[1:], h[1:]) for g in near)
                info["head_err"] = max(info["head_err"], e)
                if e > TOL:
                    errs.append(f"frame {f}: {h[0]} head off by {e:.1f} px")
        decoded.append({"heads": got})
    errs += label_errors(scene, frames)
    info["labels"] = len(scene["labels"])
    errs += mirror_errors(scene, frames[0]) + mirror_errors(scene, frames[-1])
    errs += block_errors(scene, frames[0])
    errs += port_errors(scene, frames)
    for q in s["questions"]:
        key = q["derivation_key"]
        if key in SCENE_KEYS:
            continue
        info["q_checked"] += 1
        try:
            got = answers.answer(key, q["params"], scene, frames=decoded, jitter=JITTER)
        except Exception as ex:  # noqa: BLE001
            got = f"error {ex!r}"
        if str(got) != str(q["groundtruth"]):
            errs.append(f"{q['qid']} {key}: from decoded frames {got!r}, stored {q['groundtruth']!r}")
        if key in ("exit_clock", "turn_clock"):
            tenths = round(float(q["groundtruth"].split()[0]) * 10)
            f = next((k for k in range(N) if sim.clock_tenths(k) == tenths), None)
            if f is None or read_clock(frames[f]) != tenths:
                errs.append(f"{q['qid']}: the decoded clock never shows {q['groundtruth']}")
    return errs, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--report", default="frame_check_report.txt")
    args = ap.parse_args(argv)
    root = Path(args.dataset_dir).resolve()
    lines, total = [], 0
    tot = {"frames": 0, "clock_ok": 0, "q_checked": 0, "head_err": 0.0, "labels": 0, "skipped": 0}
    with open(root / "annotations.jsonl", encoding="utf8") as fh:
        rows = fh.readlines()
    for line in rows:
        s = json.loads(line)
        errs, info = check_clip(s, root)
        for k in ("frames", "clock_ok", "q_checked", "labels", "skipped"):
            tot[k] += info[k]
        tot["head_err"] = max(tot["head_err"], info["head_err"])
        total += len(errs)
        lines.append(f"{s['sample_id']} ({s['params']['scenario']}): {len(errs)} problems, max head error "
                     f"{info['head_err']:.2f} px")
        lines += [f"  - {e}" for e in errs[:20]]
        print(lines[-1 - min(20, len(errs))], flush=True)
        for e in errs[:5]:
            print("   ", e)
    summary = (f"RESULT: {sum(1 for x in lines if not x.startswith('  '))} clips, {tot['frames']} frames, clock "
               f"template match {tot['clock_ok']}/{tot['frames']}, {tot['labels']} labels template-matched at 3 frames "
               f"each, {tot['q_checked']} questions re-derived from decoded frames, max head error "
               f"{tot['head_err']:.2f} px, "
               f"{total} problems")
    (root / args.report).write_text("\n".join(lines + ["", summary]) + "\n", encoding="utf8")
    print(summary)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
