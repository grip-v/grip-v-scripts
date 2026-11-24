"""Frame-level checks for rendered vessels_and_clocks clips (decodes the MP4s).

    python frame_check.py --dataset-dir ../output_videos/vessels_and_clocks_preview

Per clip:
- clock: every decoded frame is template-matched against all readings 0.0-10.0 s and must show f * 10 // 24;
- ruler numbers: each label region is template-matched against every ruler text ("0".."10") drawn on the
  label-free background and must read as its scene text; glyphs >= 24 px; no water ever enters a label box
  (all frames), and the label boxes don't overlap each other, a vessel, a feature or the clock;
- vessels: the coloured plinths found in the pixels (one blob per vessel colour) sit where the layout says
  (centre and extent within 2 px), the count matches, and the water width at frame 0 equals the layout width;
- per frame, per vessel: the water height measured in the pixels (a column clear of ticks and streams)
  matches render.state within 2 px; overflow, tap stream, drain stream and valve handle are read off the
  pixels and must agree with render.state on every frame (so overflow / empty / valve frames are exact);
- answers: the decoded per-frame track replaces render.state and answers.py re-derives every question
  (what-if clock times may differ by 0.1 s from fitting px-quantised levels; counted separately);
- clock questions: the decoded clock at the event frame shows the stored answer.
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
sys.path[:0] = [str(HERE), str(HERE.parent)]
import sim  # noqa: E402
import render  # noqa: E402
import answers  # noqa: E402
from common.video5 import font  # noqa: E402

TOL = 2.0
CF_TIME = {"cf_wider_time", "cf_fuller_time"}
TEXTS = [str(m) for m in range(0, 11, 2)]


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


def water_mask(img, scene, tol=60):
    st = scene["style"]
    a = img.astype(np.int16)
    d1 = np.abs(a - np.array(st["water"], dtype=np.int16)).sum(-1)
    d2 = np.abs(a - np.array(st["water_dark"], dtype=np.int16)).sum(-1)
    return (d1 < tol) | (d2 < tol)


def near(img, rgb, tol=60):
    return np.abs(img.astype(np.int16) - np.array(rgb, dtype=np.int16)).sum(-1) < tol


# ------------------------------------------------------------------ labels
def label_errors(scene, frames):
    errs = []
    blank = dict(scene, labels=[])
    base = render._static(json.dumps(blank, sort_keys=True))
    ink = tuple(scene["style"]["ink"])
    boxes = [sim.label_box(lab) for lab in scene["labels"]]
    for lab, b in zip(scene["labels"], boxes):
        if b[3] - b[1] < 24:
            errs.append(f"label {lab['text']}: glyphs {b[3] - b[1]} px tall")
        x0, y0, x1, y1 = int(b[0]) - 4, int(b[1]) - 4, int(b[2]) + 5, int(b[3]) + 5
        refs = {}
        for t in TEXTS:
            img = base.copy()
            ImageDraw.Draw(img).text((lab["x"], lab["y"]), t, fill=ink, font=font(lab["size"]), anchor=lab["anchor"])
            refs[t] = np.asarray(img, dtype=np.int16)[y0:y1, x0:x1]
        for f in (0, len(frames) // 2, len(frames) - 1):
            crop = frames[f][y0:y1, x0:x1].astype(np.int16)
            d = {t: float(np.abs(crop - r).mean()) for t, r in refs.items()}
            best = min(d, key=d.get)
            if best != lab["text"]:
                errs.append(f"frame {f}: label {lab['text']!r} reads as {best!r}")
    for k, b in enumerate(boxes):
        for o in boxes[k + 1:]:
            if sim._overlap(b, o, 2):
                errs.append("two labels overlap")
        x0, y0, x1, y1 = (int(v) for v in b)
        for f, fr in enumerate(frames):
            if water_mask(fr[y0:y1 + 1, x0:x1 + 1], scene).any():
                errs.append(f"frame {f}: water inside label {scene['labels'][k]['text']!r}")
                break
    errs += [e for e in sim._layout_errors(scene) if "ruler" in e or "clock" in e]
    return errs


# ------------------------------------------------------------------ vessels
def meas_x(scene, i):
    x0, x1 = scene["layout"]["x"][i]
    q = (x1 - x0) // 4
    return x0 + q if scene["style"]["tick_side"] == "right" else x1 - 1 - q


def layout_errors(scene, frame):
    errs = []
    st, lay = scene["style"], scene["layout"]
    y = lay["bottom"] + st["wall"] + 1
    row = frame[y]
    found = 0
    for i, v in enumerate(scene["vessels"]):
        rgb = sim.PALETTE[st["theme"]][v["colour"]]
        m = near(row[None], rgb, 70)[0]
        x0, x1 = lay["x"][i]
        c = (x0 + x1) // 2
        if not m[c]:
            errs.append(f"{v['colour']} plinth not found")
            continue
        l, r = c, c                                  # the colour run through the expected centre
        while l > 0 and m[l - 1]:
            l -= 1
        while r < 1279 and m[r + 1]:
            r += 1
        xs = np.array([l, r])
        found += 1
        want = (x0 - st["wall"] - 4, x1 + st["wall"] + 3)
        if abs(xs.min() - want[0]) > 2 or abs(xs.max() - want[1]) > 2:
            errs.append(f"{v['colour']} plinth at {xs.min()}-{xs.max()}, layout {want}")
        wm = water_mask(frame[lay["bottom"] - 6][None], scene)[0]
        xm = meas_x(scene, i)
        if wm[xm]:
            l = r = xm
            while l > 0 and wm[l - 1]:
                l -= 1
            while r < 1279 and wm[r + 1]:
                r += 1
            if abs((r - l + 1) - (x1 - x0)) > 2:
                errs.append(f"{v['colour']} water {r - l + 1} px wide, layout {x1 - x0}")
    if found != len(scene["vessels"]):
        errs.append(f"found {found} vessels, scene has {len(scene['vessels'])}")
    return errs


def read_state(scene, frame):
    """The per-frame track entries read off the pixels."""
    st, lay = scene["style"], scene["layout"]
    wm = water_mask(frame, scene)
    bot, top, w = lay["bottom"], lay["top"], st["wall"]
    btop = render.bench_top(scene)
    out = []
    for i, v in enumerate(scene["vessels"]):
        x0, x1 = lay["x"][i]
        xm = meas_x(scene, i)
        y = bot - 3                                  # the bottom rows bleed into the glass in yuv420
        if wm[y, xm]:
            while y >= top - 1 and wm[y, xm]:
                y -= 1
            px = float(bot - 1 - y)
        else:
            px = 0.0
        spill = wm[(top + bot) // 2, x1 + w + 3:x1 + w + sim.SPILL_W - 2].mean() > 0.6
        pour = 0.0
        if v["role"] == "tap":
            cx, _, _, nb = render.faucet_geom(scene, i)
            pour = float(wm[nb + 4:nb + 12, int(cx)].mean() > 0.5)
        stream = 0.0
        if lay["spout"][i]:
            sx0, sx1, *_ = render.spout_geom(scene, i)
            yr = (bot + w + btop) // 2
            seg = wm[yr, sx1 - 2:sx1 + 21] if lay["spout"][i] == "right" else wm[yr, sx0 - 20:sx0 + 3]
            stream = float(seg.sum())
        out.append({"water_px": px, "spill": bool(spill), "pour": pour, "stream": stream})
    pipes = []
    hand = (40, 40, 46) if st["theme"] == "light" else (235, 235, 240)
    for a, b in scene["pairs"]:
        *_, cx, cy = render.pipe_geom(scene, a, b)
        cx, cy = int(round(cx)), int(cy)
        vert = near(frame[cy - 16:cy - 8, cx - 2:cx + 3], hand).sum()
        hor = near(frame[cy - 2:cy + 3, cx + 9:cx + 17], hand).sum()
        pipes.append({"a": a, "b": b, "open": bool(hor > vert)})
    return {"vessels": out, "pipes": pipes}


def check_clip(s, root):
    scene = s["params"]
    info = {"frames": 0, "clock_ok": 0, "q_checked": 0, "q_exact": 0, "water_err": 0.0, "labels": 0}
    errs = []
    frames = decode(root / s["media_path"])
    N = render.n_frames(scene)
    if len(frames) != N:
        return [f"decoded {len(frames)} frames, expected {N}"], info
    info["frames"] = N
    n = len(scene["vessels"])
    tr = {"water": [[] for _ in range(n)], "spill": [[] for _ in range(n)], "pour": [[] for _ in range(n)],
          "stream": [[] for _ in range(n)], "open": [[] for _ in scene["pairs"]]}
    for f, fr in enumerate(frames):
        if read_clock(fr) == sim.clock_tenths(f):
            info["clock_ok"] += 1
        else:
            errs.append(f"frame {f}: clock misread")
        S, G = render.state(scene, f), read_state(scene, fr)
        for i, (sv, gv) in enumerate(zip(S["vessels"], G["vessels"])):
            want = float(render.scene_px(scene, sv["water_px"]))
            e = abs(gv["water_px"] - want)
            info["water_err"] = max(info["water_err"], e)
            if e > TOL:
                errs.append(f"frame {f}: {scene['vessels'][i]['colour']} water {gv['water_px']:.0f} px, drawn {want:.0f}")
            if gv["spill"] != sv["spill"]:
                errs.append(f"frame {f}: {scene['vessels'][i]['colour']} overflow {gv['spill']} != {sv['spill']}")
            if (gv["pour"] > 0) != (sv["pour"] > 0):
                errs.append(f"frame {f}: {scene['vessels'][i]['colour']} tap stream mismatch")
            if (gv["stream"] > 0) != (sv["stream"] > 0) or (sv["stream"] and abs(gv["stream"] - sv["stream"]) > 3):
                errs.append(f"frame {f}: {scene['vessels'][i]['colour']} drain stream {gv['stream']} != "
                            f"{sv['stream']:.1f}")
            tr["water"][i].append(gv["water_px"])
            tr["spill"][i].append(gv["spill"])
            tr["pour"][i].append(gv["pour"] > 0)
            tr["stream"][i].append(gv["stream"])
        for k, (sp, gp) in enumerate(zip(S["pipes"], G["pipes"])):
            if sp["open"] != gp["open"]:
                errs.append(f"frame {f}: valve {k} open {gp['open']} != {sp['open']}")
            tr["open"][k].append(gp["open"])
    errs += label_errors(scene, frames)
    info["labels"] = len(scene["labels"])
    errs += layout_errors(scene, frames[0])
    o = answers.Obs(scene, tr)
    for q in s["questions"]:
        key = q["derivation_key"]
        info["q_checked"] += 1
        try:
            got = answers.derive(key, q["params"], o)
        except Exception as ex:  # noqa: BLE001
            got = f"error {ex!r}"
        gt = str(q["groundtruth"])
        if str(got) == gt:
            info["q_exact"] += 1
        elif key in CF_TIME and gt.endswith(" s") and str(got).endswith(" s") and \
                abs(float(str(got).split()[0]) - float(gt.split()[0])) <= 0.1 + 1e-9:
            pass                                       # px-quantised fit: within one tenth
        else:
            errs.append(f"{q['qid']} {key}: from decoded frames {got!r}, stored {gt!r}")
        if key in ("event_clock", "valve_clock"):
            f = o.ev_frame(q["params"]["x"]) if key == "event_clock" else o.valve_frame(q["params"]["x"])
            if f is None or f"{read_clock(frames[f]) / 10:.1f} s" != gt:
                errs.append(f"{q['qid']}: decoded clock at the event frame isn't {gt}")
    return errs, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--report", default="frame_check_report.txt")
    args = ap.parse_args(argv)
    root = Path(args.dataset_dir).resolve()
    lines, total = [], 0
    tot = {"frames": 0, "clock_ok": 0, "q_checked": 0, "q_exact": 0, "water_err": 0.0, "labels": 0}
    with open(root / "annotations.jsonl", encoding="utf8") as fh:
        rows = fh.readlines()
    for line in rows:
        s = json.loads(line)
        errs, info = check_clip(s, root)
        for k in ("frames", "clock_ok", "q_checked", "q_exact", "labels"):
            tot[k] += info[k]
        tot["water_err"] = max(tot["water_err"], info["water_err"])
        total += len(errs)
        lines.append(f"{s['sample_id']} ({s['params']['scenario']}): {len(errs)} problems, max water error "
                     f"{info['water_err']:.2f} px")
        lines += [f"  - {e}" for e in errs[:20]]
        print(lines[-1 - min(20, len(errs))], flush=True)
        for e in errs[:5]:
            print("   ", e)
    summary = (f"RESULT: {len(rows)} clips, {tot['frames']} frames, clock template match "
               f"{tot['clock_ok']}/{tot['frames']}, {tot['labels']} ruler numbers template-matched at 3 frames each, "
               f"{tot['q_checked']} questions re-derived from decoded frames ({tot['q_exact']} exact, the rest "
               f"what-if clock times within 0.1 s), max water-height error {tot['water_err']:.2f} px, "
               f"{total} problems")
    (root / args.report).write_text("\n".join(lines + ["", summary]) + "\n", encoding="utf8")
    print(summary)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
