"""Frame-level check of rendered clips against the scene data.

    python frame_check.py --dataset-dir <dir> [--report frame_check.txt] [--jobs 4]

For every clip it decodes the MP4 (OpenCV) and checks:
  * clock: template-matches the timer panel against all readings 0.0-9.9 s
    on every 6th frame, on every event frame and on every frame a question
    names by its clock reading;
  * ruler numbers: on every 6th frame each label's box is template-matched
    against all numbers 0-12 drawn in the ruler's style and must read as its
    own text; the box must look like the empty ruler (nothing moving ever
    covers a number);
  * chain: on every 6th frame, in columns every 6 px along the chain, the
    middle of the drawn chain (pixels that differ from the empty background)
    lies where the track puts it (within its local slope + 5 px);
  * pulses: on calm frames (only whole, separate pulses) the bumps found in
    the decoded chain match the traced pulses in number, sign and position
    (within 0.12 ruler marks);
  * answers: clock answers read on the decoded event frame; counts at a clock
    reading counted as bumps on the decoded frames showing it; meeting marks
    from the two decoded bumps just before they meet.
Prints a summary and exits 1 on any failure.
"""
from __future__ import annotations
import argparse
import json
import sys
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import render  # noqa: E402
import sim  # noqa: E402
import questions  # noqa: E402
from common.video5 import font  # noqa: E402


def frames_of(path: Path) -> list:
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr[:, :, ::-1].astype(np.int16))
    cap.release()
    return out


_TIMER = {}


def _timer_templates():
    if not _TIMER:
        tb = sim.TIMER_BOX
        for k in range(100):
            img = Image.new("RGB", (sim.W, sim.H), (0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle(tb, radius=10, fill=render.TIMER_BG, outline=render.TIMER_EDGE, width=2)
            d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), sim.clock_text(k), fill=render.TIMER_FG,
                   font=font(sim.TIMER_FONT), anchor="mm")
            _TIMER[k] = np.asarray(img, dtype=np.int16)[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
    return _TIMER


def read_timer(fr) -> int:
    tb = sim.TIMER_BOX
    crop = fr[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
    tm = _timer_templates()
    return min(tm, key=lambda k: np.abs(crop - tm[k]).mean())


def _static(scene) -> np.ndarray:
    img = render._static(json.dumps(scene, sort_keys=True)).resize((sim.W, sim.H), Image.LANCZOS)
    return np.asarray(img, dtype=np.int16)


def _label_box(lb):
    x0, y0, x1, y1 = font(lb["size"]).getbbox(lb["text"], anchor="mt")
    return (int(lb["x"] + x0) - 3, int(lb["y"] + y0) - 3, int(lb["x"] + x1) + 4, int(lb["y"] + y1) + 4)


def _label_templates(scene, st_img):
    """Every number 0..12 drawn in the ruler's style on this clip's background, at each label's spot."""
    ink = tuple(scene["style"]["ink"])
    out = {}
    for lb in scene["labels"]:
        x0, y0, x1, y1 = _label_box(lb)
        base = Image.fromarray(st_img.astype(np.uint8))
        cands = {}
        for v in range(13):
            img = base.copy()
            d = ImageDraw.Draw(img)
            d.rectangle([x0, y0, x1, y1], fill=tuple(int(c) for c in st_img[y0 - 1, x0 - 1]))
            d.text((lb["x"], lb["y"]), str(v), fill=ink, font=font(lb["size"]), anchor="mt")
            cands[str(v)] = np.asarray(img, dtype=np.int16)[y0:y1, x0:x1]
        out[lb["text"]] = ((x0, y0, x1, y1), cands)
    return out


def chain_y(fr, st_img, scene, cols):
    """Middle row of the drawn chain in each pixel column (None where nothing is drawn)."""
    up, down = scene["reach"]
    top = int(scene["y0"] - scene["amp_px"] * up - 30)
    bot = int(scene["y0"] + scene["amp_px"] * down + 30)
    diff = np.abs(fr[top:bot] - st_img[top:bot]).sum(axis=2) > 60
    out = []
    for c in cols:
        rows = np.nonzero(diff[:, c])[0]
        out.append(None if len(rows) == 0 else top + float(np.median(rows)))
    return out


def decoded_bumps(fr, st_img, scene):
    """(x in ruler marks, height in amplitude units) of each separate bump on the decoded frame."""
    x0, sc = scene["x0"], scene["scale"]
    cols = np.arange(int(x0) + 10, int(x0 + scene["L"] * sc) - 10, 3)
    ys = chain_y(fr, st_img, scene, cols)
    h = np.array([0.0 if y is None else (scene["y0"] - y) / scene["amp_px"] for y in ys])
    on = np.abs(h) > 0.15
    out, i = [], 0
    while i < len(h):
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(h) and on[j + 1] and np.sign(h[j + 1]) == np.sign(h[i]):
            j += 1
        k = i + int(np.argmax(np.abs(h[i:j + 1])))
        if abs(h[k]) < 0.22:                      # flank noise (a bead on a slope), not a pulse (>= 0.3)
            i = j + 1
            continue
        w = np.clip(np.abs(h[i:j + 1]) - 0.15, 0, None)
        cx = float((w * cols[i:j + 1]).sum() / w.sum()) if w.sum() > 0 else float(cols[k])
        out.append(((cx - x0) / sc, float(h[k])))
        i = j + 1
    return out


def _clock_event(key, p, F):
    """Frame of the real event a clock answer refers to (None for what-ifs)."""
    S = F.segs
    if key == "first_far_time":
        return sim.frame_of(S[F.arrivals(F.fx)[0]]["te"])
    if key == "return_time":
        return sim.frame_of(S[questions._first_home(F)]["te"])
    if key == "time_kth_far":
        return sim.frame_of(S[F.arrivals(F.fx)[p["k"] - 1]]["te"])
    if key == "junction_return":
        return sim.frame_of(S[questions._first_split(F)[1]]["te"])
    if key == "cross_time":
        return sim.frame_of(S[questions._arrive_other(F, questions._first_of(F, p["side"]))]["te"])
    return None


def check_clip(scene, qs, frames) -> list[str]:
    errs = []
    nf = scene["n_frames"]
    if len(frames) != nf:
        return [f"decoded {len(frames)} frames, expected {nf}"]
    F = questions.Facts(scene)
    st_img = _static(scene)
    tr = render.track(scene)
    # clock
    ev = sorted({sim.frame_of(t) for s in F.segs for t in (s["tb"], s["te"]) if 0 <= sim.frame_of(t) < nf})
    named = [f for q in qs if "k" in q["params"] for f in range(*(lambda a: (a[0], a[1] + 1))(sim.frames_showing(q["params"]["k"])))]
    for f in sorted(set(range(0, nf, 6)) | set(ev) | set(named)):
        if read_timer(frames[f]) != sim.tenths(f):
            errs.append(f"clock on frame {f} reads {read_timer(frames[f])}, expected {sim.tenths(f)}")
    # ruler numbers
    tm = _label_templates(scene, st_img)
    for f in range(0, nf, 6):
        fr = frames[f]
        for lb in scene["labels"]:
            (x0, y0, x1, y1), cands = tm[lb["text"]]
            crop = fr[y0:y1, x0:x1]
            best = min(cands, key=lambda v: np.abs(crop - cands[v]).mean())
            if best != lb["text"]:
                errs.append(f"frame {f}: ruler label {lb['text']} reads as {best}")
            if np.abs(crop - st_img[y0:y1, x0:x1]).mean() > 6:
                errs.append(f"frame {f}: something covers ruler label {lb['text']}")
    # chain position
    x0, sc = scene["x0"], scene["scale"]
    cols = list(range(int(x0) + 8, int(x0 + scene["L"] * sc) - 8, 6))
    px = tr["px"]
    for f in range(0, nf, 6):
        ys = chain_y(frames[f], st_img, scene, cols)
        ty = scene["y0"] + tr["dy"][f]
        bad = 0
        for c, y in zip(cols, ys):
            near = ty[(px >= c - 4) & (px <= c + 4)]
            if y is None or not (near.min() - 5 <= y <= near.max() + 5):
                bad += 1
        if bad > 0.03 * len(cols):
            errs.append(f"frame {f}: chain off its track in {bad}/{len(cols)} columns")
    # pulses on calm frames
    cm = F.calm()
    for f in range(0, nf, 3):
        if not cm[f]:
            continue
        act = sim.active(F.segs, f)
        want = sorted((float(sim.pos_at(F.segs[i], Fr(f, sim.FPS))), float(F.segs[i]["amp"])) for i in act)
        got = sorted(decoded_bumps(frames[f], st_img, scene))
        if len(got) != len(want):
            errs.append(f"frame {f}: {len(got)} bumps decoded, {len(want)} traced")
            continue
        for (gx, gh), (wx, wh) in zip(got, want):
            if abs(gx - wx) > 0.12 or (gh > 0) != (wh > 0) or abs(gh - wh) > 0.2:
                errs.append(f"frame {f}: bump at {gx:.2f} ({gh:+.2f}) vs traced {wx:.2f} ({wh:+.2f})")
    # answers read off the frames
    for q in qs:
        key, p = q["derivation_key"], q["params"]
        fe = _clock_event(key, p, F)
        if fe is not None:
            got = sim.clock_text(read_timer(frames[fe]))
            if got != q["groundtruth"]:
                errs.append(f"{q['qid']} {key}: clock on event frame {fe} reads {got}, answer {q['groundtruth']}")
        if key in ("count_at", "end_count"):
            fs = range(sim.frames_showing(p["k"])[0], sim.frames_showing(p["k"])[1] + 1) if key == "count_at" \
                else range(nf - 12, nf)
            for f in fs:
                n = len(decoded_bumps(frames[f], st_img, scene))
                if n != q["groundtruth"]:
                    errs.append(f"{q['qid']} {key}: {n} bumps decoded on frame {f}, answer {q['groundtruth']}")
        if key in ("meet_mark", "second_meet"):
            ms = questions._meets(F, questions._first_of(F, "left"), questions._first_of(F, "right"))
            t, x, i, j = ms[1 if key == "second_meet" else 0]
            ok = False
            for back in range(3, 30):
                f = sim.frame_of(t) - back
                b = decoded_bumps(frames[f], st_img, scene)
                ea, eb = (float(sim.pos_at(F.segs[n], Fr(f, sim.FPS))) for n in (i, j))
                ga = [g for g in b if abs(g[0] - ea) < 0.15]
                gb = [g for g in b if abs(g[0] - eb) < 0.15]
                if len(ga) == 1 and len(gb) == 1 and ga != gb:
                    mid = (ga[0][0] + gb[0][0]) / 2
                    ok = str(round(mid)) == q["groundtruth"] and abs(mid - float(x)) < 0.1
                    break
            if not ok:
                errs.append(f"{q['qid']} {key}: meeting mark not read back as {q['groundtruth']}")
    return errs


def _one(arg):
    root, line = arg
    s = json.loads(line)
    frames = frames_of(Path(root) / s["media_path"])
    try:
        errs = check_clip(s["params"], s["questions"], frames)
    except Exception as e:  # report, don't crash the pool
        errs = [f"checker error {e!r}"]
    return s["sample_id"], s["params"]["scenario"], errs


def main():
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--report", default="frame_check.txt")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    rows = open(root / "annotations.jsonl", encoding="utf8").read().splitlines()
    lines, total = [], 0
    with Pool(args.jobs) as pool:
        for sid, sc, errs in pool.imap(_one, [(str(root), r) for r in rows]):
            total += len(errs)
            lines += [f"{sid}: {e}" for e in errs]
            print(f"{sid} ({sc}): {len(errs)} failures", flush=True)
    lines.append(f"RESULT: {len(rows)} clips, {total} frame-check failures")
    (root / args.report).write_text("\n".join(lines) + "\n")
    print(lines[-1])
    for e in lines[:-1][:20]:
        print("  -", e)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
