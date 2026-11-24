"""Frame-level check of rendered glasses_fill clips against the scene data.

    python frame_check.py --dataset-dir <dir> [--report frame_check.txt]

For every clip it decodes the MP4 (OpenCV) and measures, on the pixels:
  * clock: the timer crop is template-matched against every reading 0.0-9.9 s
    on every 6th frame and on every event frame (stream starts, cracks, the
    frames a question names by clock reading); the best match must be the
    reading the clock should show;
  * labels: every ruler number and glass number crop equals the static
    background render on every 6th frame (nothing covers or moves them), the
    label text heights are >= 24 px and no two label boxes overlap;
  * levels and layers: on the last frame each glass's centre column is
    classified pixel by pixel into the liquid colours; the level and the
    bottom-up layer list (to 3 ml) must equal answers.py's end state;
  * pours: each pour's delivered amount, measured as the level change of its
    glass across the tilt, equals the stored amount; a wet pour shows a
    stream of the jug's colour above the glass mid-flow, a dry pour none;
  * shelf: the order in which jugs leave the shelf equals the pour order;
  * cracks: no hole before the crack, a stream of the right colour leaves the hole
    mid-drain, and a dark hole is on the glass wall after the drain;
  * answers: every level_end / pour_ml / level_at_clock / pour_clock / n_tilts /
    first_jug / total_end / n_empty_end question's stored answer equals the
    value measured from the decoded frames.
Prints a summary and exits 1 on any failure.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import answers  # noqa: E402
import render  # noqa: E402
import sim  # noqa: E402
from common.video5 import font, hex_rgb  # noqa: E402

FPS = sim.FPS
LIQ = {c: np.array(hex_rgb(h), float) for c, h in sim.LIQUIDS.items()}
COL_TOL = 55.0


def decode(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr[:, :, ::-1].astype(np.int16))
    cap.release()
    return out


def classify(px):
    """Liquid colour name of an RGB pixel, or None."""
    best, bd = None, COL_TOL
    for c, v in LIQ.items():
        d = float(np.abs(px - v).sum()) / 3 * 1.0
        if d < bd:
            best, bd = c, d
    return best


def column_layers(fr, scene, g):
    """Bottom-up [[colour, ml]] read off glass g's centre column (3 columns averaged)."""
    x = scene["glasses"]["cx"][g]
    col = fr[int(sim.Y_TOP) + 1:int(sim.Y_BOT), x - 1:x + 2].mean(axis=1)
    labs = [classify(px) for px in col[::-1]]                         # bottom first
    out, run, n = [], None, 0
    for c in labs:
        if c is None:
            break
        if c == run:
            n += 1
        else:
            if run:
                out.append([run, n])
            run, n = c, 1
    if run:
        out.append([run, n])
    merged = []
    for c, k in out:                                                  # drop 1-2 px blends at boundaries
        if k <= 2 and merged:
            merged[-1][1] += k
        elif merged and merged[-1][0] == c:
            merged[-1][1] += k
        else:
            merged.append([c, k])
    return [[c, k / sim.PX_PER_ML] for c, k in merged]


def level(fr, scene, g):
    return sum(m for _, m in column_layers(fr, scene, g))


def clock_templates(scene):
    x0, y0, x1, y1 = (int(v) for v in render.clock_box(scene))
    x0, y0, x1, y1 = x0 - 2, y0 - 2, x1 + 2, y1 + 2
    out = {}
    for tc in range(0, 100):
        im = Image.new("RGB", (sim.W, sim.H), scene["palette"]["bg"])
        d = ImageDraw.Draw(im)
        x, y = scene["clock_xy"]
        d.text((x, y), f"{tc / 10:.1f} s", font=font(scene["label_px"]), fill=render.CLOCK_HEX, anchor="ra")
        out[tc] = np.asarray(im, np.int16)[y0:y1, x0:x1]
    return (x0, y0, x1, y1), out


def check_clip(s, root):
    errs, stats = [], {"clock": 0, "labels": 0, "levels": 0, "pours": 0, "answers": 0}
    sc = s["params"]
    frames = decode(root / s["media_path"])
    nf = sc["n_frames"]
    if len(frames) != nf:
        return [f"decoded {len(frames)} frames, expected {nf}"], stats
    tr = answers._track(answers._key(sc))

    # ---- clock
    (x0, y0, x1, y1), tmpl = clock_templates(sc)
    events = set(range(0, nf, 6))
    for e in tr["eps"]:
        events.update(f for f in (e["f0"], e["f1"], e["flow0"]) if f is not None)
    for lk in sc["leaks"]:
        events.add(min(nf - 1, sim.frame_of(lk["t_crack"])))
    for q in s["questions"]:
        if "clock" in q["params"]:
            events.update(f for f in range(nf) if sim.clock_tenths(f) == q["params"]["clock"])
    for f in sorted(events):
        crop = frames[f][y0:y1, x0:x1]
        best = min(tmpl, key=lambda tc: np.abs(crop - tmpl[tc]).mean())
        stats["clock"] += 1
        if best != sim.clock_tenths(f):
            errs.append(f"frame {f}: clock reads {best / 10:.1f}, should be {sim.clock_tenths(f) / 10:.1f}")

    # ---- labels
    static = np.asarray(render._static_layer(sc), np.int16)
    boxes = [tuple(int(v) for v in render.label_box(sc, lab)) for lab in sc["labels"]]
    for i, b in enumerate(boxes):
        if b[3] - b[1] < 24 and sc["labels"][i]["text"] not in ("",):
            errs.append(f"label {sc['labels'][i]['text']!r} is only {b[3] - b[1]} px tall")
        for b2 in boxes[i + 1:]:
            if b[0] < b2[2] and b2[0] < b[2] and b[1] < b2[3] and b2[1] < b[3]:
                errs.append(f"labels overlap: {b} {b2}")
    for f in range(0, nf, 6):
        for lab, b in zip(sc["labels"], boxes):
            diff = np.abs(frames[f][b[1] - 2:b[3] + 2, b[0] - 2:b[2] + 2] - static[b[1] - 2:b[3] + 2, b[0] - 2:b[2] + 2])
            stats["labels"] += 1
            if diff.mean() > 6:
                errs.append(f"frame {f}: label {lab['text']!r} differs from the static render (covered?)")

    # ---- end levels and layers
    last = frames[-1]
    for g in range(sc["glasses"]["n"]):
        lay = column_layers(last, sc, g)
        want = tr["layers"][g]
        stats["levels"] += 1
        if [c for c, _ in lay] != [c for c, _ in want] or any(abs(a[1] - b[1]) > 3 for a, b in zip(lay, want)):
            errs.append(f"glass {g + 1} at the end: frame shows {[[c, round(m)] for c, m in lay]}, expected {want}")

    # ---- pours: delivered amount, stream, shelf order
    for i, (p, e) in enumerate(zip(sc["pours"], tr["eps"])):
        g = p["glass"]
        before = level(frames[max(0, e["f0"] - 1)], sc, g)
        after = level(frames[min(nf - 1, e["f1"] + 3)], sc, g)
        stats["pours"] += 1
        if abs(after - before - p["got_ml"]) > 3:
            errs.append(f"pour {i}: frames show {after - before:.0f} ml delivered, expected {p['got_ml']}")
        fm = sim.frame_of((p["t_pour"] + (p["t_flow_end"] if p["got_ml"] else p["t_untilt"])) / 2)
        x = sc["glasses"]["cx"][g]
        ys = slice(int(sim.Y_TOP) + 4, int(sim.Y_TOP) + 12)             # inside the glass, above the liquid
        colname = classify(frames[fm][ys, x - 1:x + 2].mean(axis=(0, 1)))
        want = sc["jugs"][p["jug"]]["colour"] if p["got_ml"] else None
        if colname != want:
            errs.append(f"pour {i}: stream above glass {g + 1} at frame {fm} is {colname}, expected {want}")
    gone = []
    bg = np.array(hex_rgb(sc["palette"]["bg"]), float)
    for f in range(0, nf, 2):
        for j in range(len(sc["jugs"])):
            px, py = render.park_pos(j, len(sc["jugs"]))
            patch = frames[f][int(py) - 20:int(py) + 20, int(px) - 20:int(px) + 20]
            empty = np.abs(patch - bg).sum(axis=2).mean() < 12
            if empty and (not gone or gone[-1][0] != j or f - gone[-1][1] > 2):
                gone.append([j, f])
            elif empty:
                gone[-1][1] = f
    order = [j for j, _ in gone]
    if order != [p["jug"] for p in sc["pours"]]:
        errs.append(f"jugs leave the shelf in order {order}, pours are {[p['jug'] for p in sc['pours']]}")

    # ---- cracks
    for lk in sc["leaks"]:
        _, x1g = render.glass_x(sc, lk["glass"])
        hx, hy = int(x1g + 2), int(sim.ml_to_y(lk["hole_ml"]) - (5 if lk["hole_ml"] == 0 else 0))
        fc = sim.frame_of(lk["t_crack"])
        dark = lambda f: frames[f][hy - 2:hy + 3, hx - 2:hx + 3].mean() < 70  # noqa: E731
        if fc - 2 >= 0 and dark(fc - 2):
            errs.append(f"glass {lk['glass'] + 1}: hole visible before the crack")
        fd = sim.frame_of(lk["t_drain_end"])
        if not dark(min(nf - 1, fd + 2)) or not dark(nf - 1):
            errs.append(f"glass {lk['glass'] + 1}: no hole after the drain")
        fm = sim.frame_of((lk["t_crack"] + lk["t_drain_end"]) / 2)
        want = sim.leak_flowing(sc, lk["glass"], fm / FPS)
        x0g, x1g2 = render.glass_x(sc, lk["glass"])
        gap = (sim.GLASS_X1 - sim.GLASS_X0) / sc["glasses"]["n"] - (x1g2 - x0g)
        land_x = hx + max(12.0, min(30.0, gap * 0.4))
        sx, sy = int(hx + (land_x - hx) * 0.6), int(hy + (sim.TABLE_Y - hy) * 0.36)   # on the stream's curve
        if classify(frames[fm][sy - 1:sy + 2, sx - 1:sx + 2].mean(axis=(0, 1))) != want:
            errs.append(f"glass {lk['glass'] + 1}: no {want} stream at the hole mid-drain")

    # ---- answers measured on the frames
    ends = [level(last, sc, g) for g in range(sc["glasses"]["n"])]
    for q in s["questions"]:
        k, pp = q["derivation_key"], q["params"]
        got = None
        if k == "level_end":
            got = f"{int(round(ends[pp['glass']]))} ml"
        elif k == "total_end":
            got = f"{int(round(sum(ends)))} ml"
        elif k == "n_empty_end":
            got = sum(1 for v in ends if v < 3)
        elif k == "level_at_clock":
            fr = [f for f in range(nf) if sim.clock_tenths(f) == pp["clock"]]
            got = f"{int(round(level(frames[fr[len(fr) // 2]], sc, pp['glass'])))} ml"
        elif k == "pour_ml":
            e = tr["eps"][pp["pour"]]
            got = f"{int(round(level(frames[min(nf - 1, e['f1'] + 3)], sc, e['glass']) - level(frames[e['f0'] - 1], sc, e['glass'])))} ml"
        elif k == "pour_clock":
            p = next(p for p in sc["pours"] if p["jug"] == pp["jug"] and p["got_ml"])
            x = sc["glasses"]["cx"][p["glass"]]
            fs = next(f for f in range(nf) if classify(frames[f][int(sim.Y_TOP) + 4:int(sim.Y_TOP) + 12, x - 1:x + 2]
                                                        .mean(axis=(0, 1))) == sc["jugs"][p["jug"]]["colour"])
            crop = frames[fs][y0:y1, x0:x1]
            best = min(tmpl, key=lambda tc: np.abs(crop - tmpl[tc]).mean())
            got = f"{best / 10:.1f} s"
        elif k == "n_tilts":
            got = len(order)
        elif k == "first_jug":
            got = f"the {sc['jugs'][order[0]]['pattern']} jug"
        else:
            continue
        stats["answers"] += 1
        if str(got) != str(q["groundtruth"]):
            errs.append(f"{q['qid']} {k}: frames give {got}, stored {q['groundtruth']}")
    return errs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--report", default="frame_check.txt")
    a = ap.parse_args()
    root = Path(a.dataset_dir)
    lines, total, tot_stats = [], 0, {}
    for line in open(root / "annotations.jsonl", encoding="utf8"):
        s = json.loads(line)
        errs, st = check_clip(s, root)
        for k, v in st.items():
            tot_stats[k] = tot_stats.get(k, 0) + v
        total += len(errs)
        lines.append(f"{s['sample_id']} ({s['params']['scenario']}): {'OK' if not errs else f'{len(errs)} FAIL'}")
        lines += [f"  - {e}" for e in errs[:20]]
        print(lines[-1 - min(len(errs), 20)], flush=True)
    lines.append(f"checked: {tot_stats}")
    lines.append(f"RESULT: {total} failures")
    (root / a.report).write_text("\n".join(lines) + "\n", encoding="utf8")
    print(lines[-2])
    print(lines[-1])
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
