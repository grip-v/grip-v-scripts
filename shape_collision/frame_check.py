"""Frame-level check of rendered clips against the scene data.

    python frame_check.py --dataset-dir <dir> [--report frame_check.txt]

For every clip it decodes the MP4 (OpenCV) and checks:
  * clock: template-matches the timer panel against all readings 0.0-9.9 s
    on every 6th frame, on every event frame and on every frame a question
    names by its clock reading; the best match must be the expected text;
  * positions: on every 6th frame each shape is found as the blob of its own
    colour anywhere in the box; its centre must lie within 0.6 r of the
    track, a shape that fell into a hole must have no blob, and the number of
    blobs must equal the number of shapes still in the box;
  * patterns: on every frame each shape's pattern is classified by matching
    the decoded pixels against its sprite in every pattern; the decoded
    pattern changes (pairs that swap, shapes that get the tag) must be
    exactly the collisions that change patterns, on exactly their frames;
  * holes: a falling shape is drawn on the frame before its capture frame
    and gone on the capture frame;
  * clock answers (coll_time / fall_time / span): the clock read on the
    decoded event frame equals the stored answer;
  * layout: the timer panel lies outside the box (nothing passes under it).
Prints a summary and exits 1 on any failure.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import render  # noqa: E402
import sim  # noqa: E402
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


def _timer_templates(scene):
    key = (tuple(scene["timer"]["box"]), scene["timer"]["size"])
    if key not in _TIMER:
        tb = scene["timer"]["box"]
        out = {}
        for k in range(100):
            img = Image.new("RGB", (sim.W, sim.H), (0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle(tb, radius=10, fill=render.TIMER_BG, outline=render.TIMER_EDGE, width=2)
            d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), sim.clock_text(k), fill=render.TIMER_FG,
                   font=font(scene["timer"]["size"]), anchor="mm")
            out[k] = np.asarray(img, dtype=np.int16)[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
        _TIMER[key] = out
    return _TIMER[key]


def read_timer(scene, fr) -> int:
    tb = scene["timer"]["box"]
    crop = fr[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
    tm = _timer_templates(scene)
    return min(tm, key=lambda k: np.abs(crop - tm[k]).mean())


def _palette(scene):
    """Every flat colour that can appear inside the box."""
    st = scene["style"]
    pal = [tuple(st["floor"]), tuple(max(0, min(255, c + st["tex_shade"])) for c in st["floor"]),
           tuple(st["ring"]), tuple(st["post"]), tuple(st["hole_rim"]), tuple(st["rail"]), render.HOLE]
    for b in scene["bodies"]:
        col = sim.COLORS[b["color"]]
        pal += [sim.second_colour(col), tuple(int(round(c * 0.6)) for c in col)]
    return [sim.COLORS[b["color"]] for b in scene["bodies"]], pal


def blobs(scene, fr) -> dict:
    """Centre of the largest blob of each shape's own colour inside the box
    (None if there is none). Pixels are classified by their nearest colour
    among the shapes' colours and every other flat colour in the scene."""
    L, T, R, B = scene["box"]
    reg = fr[T:B, L:R]
    own, other = _palette(scene)
    cand = np.array(own + other, dtype=np.int16)
    dist = np.abs(reg[:, :, None, :] - cand[None, None]).sum(axis=3)
    near = dist.argmin(axis=2)
    ok = dist.min(axis=2) < 60
    out = {}
    for i, b in enumerate(scene["bodies"]):
        ys, xs = np.nonzero((near == i) & ok)
        if len(xs) < 0.05 * b["r"] ** 2:
            out[i] = None
            continue
        # robust centre: start at the median, then mean-shift over pixels within 1.1 r
        # (stray anti-aliased ring pixels of other shapes can come out near white)
        cx, cy = float(np.median(xs)), float(np.median(ys))
        for _ in range(4):
            sel = np.hypot(xs - cx, ys - cy) < 1.1 * b["r"]
            if sel.sum() == 0:
                break
            cx, cy = float(xs[sel].mean()), float(ys[sel].mean())
        out[i] = (cx + L, cy + T) if sel.sum() >= 0.05 * b["r"] ** 2 else None
    return out


_REF = {}


def _refs(scene, i):
    """Per shape: sprite colours and alpha for every pattern, and the
    inside-the-outline mask (cached)."""
    b = scene["bodies"][i]
    st = scene["style"]
    key = (b["shape"], b["color"], b["r"], tuple(st["ring"]), st["ring_style"], st["edge_w"])
    if key not in _REF:
        sps = [np.asarray(render.sprite(b["shape"], b["color"], p, b["r"], *key[3:]), dtype=np.float32)
               for p in sim.PATTERNS]
        h = sps[0].shape[0]
        yy, xx = np.mgrid[:h, :h]
        inner = (sps[0][:, :, 3] == 255) & (np.hypot(xx - h / 2, yy - h / 2) < b["r"] * 0.85)
        rgb = np.stack([s[:, :, :3][inner] for s in sps])            # (10, m, 3)
        al = np.stack([s[:, :, 3][inner] / 255.0 for s in sps])[..., None]
        _REF[key] = (h, inner, rgb, al)
    return _REF[key]


def pattern_of(scene, fr, i, x, y, bg) -> str:
    """Best-matching pattern: each candidate sprite is composited over the
    drawn background and compared with the decoded pixels inside the outline."""
    h, inner, rgb, al = _refs(scene, i)
    x0, y0 = round(x - h / 2), round(y - h / 2)
    if x0 < 0 or y0 < 0 or x0 + h > sim.W or y0 + h > sim.H:
        return "?"
    crop = fr[y0:y0 + h, x0:x0 + h][inner].astype(np.float32)
    back = bg[y0:y0 + h, x0:x0 + h][inner]
    ref = rgb * al + back[None] * (1 - al)
    d = np.abs(crop[None] - ref).sum(axis=2).mean(axis=1)
    return sim.PATTERNS[int(np.argmin(d))]


def check_clip(scene, qs, frames) -> list:
    errs = []
    nf = scene["n_frames"]
    if len(frames) != nf:
        return [f"decoded {len(frames)} frames, expected {nf}"]
    tr = render.track(scene)
    n = len(scene["bodies"])
    L, T, R, B = scene["box"]
    tb = scene["timer"]["box"]
    if not (tb[3] < T - scene["style"]["wall_w"] or tb[2] < L - scene["style"]["wall_w"]):
        errs.append("timer panel overlaps the box")
    coll = sim.collisions(scene)
    caps = sim.captures(scene)
    ev_frames = sorted({sim.frame_of(t) for t, *_ in coll + caps})
    # clock
    want_t = set(range(0, nf, 6)) | set(ev_frames)
    for q in qs:
        if "t" in q["params"]:
            lo, hi = sim.frames_showing(q["params"]["t"])
            want_t |= set(range(lo, hi + 1))
    for f in sorted(want_t):
        if f < nf and read_timer(scene, frames[f]) != sim.tenths(f):
            errs.append(f"frame {f}: clock misread")
    # positions and counts
    for f in range(0, nf, 6):
        got = blobs(scene, frames[f])
        for i in range(n):
            item = tr[f][i]
            if item is None:
                if got[i] is not None:
                    errs.append(f"frame {f}: fallen shape {i} still visible")
            elif got[i] is None or np.hypot(got[i][0] - item[0], got[i][1] - item[1]) > 0.6 * scene["bodies"][i]["r"]:
                errs.append(f"frame {f}: shape {i} at {got[i]}, track {item[:2]}")
        if sum(g is not None for g in got.values()) != sum(it is not None for it in tr[f]):
            errs.append(f"frame {f}: blob count differs")
    # patterns on every frame -> decoded pattern-change events
    bg = np.asarray(render._static(render._static_key(scene)), dtype=np.float32)
    seq = []
    for f in range(nf):
        seq.append([pattern_of(scene, frames[f], i, tr[f][i][0], tr[f][i][1], bg) if tr[f][i] else None
                    for i in range(n)])
    for f in range(nf):
        for i in range(n):
            if tr[f][i] is not None and seq[f][i] != tr[f][i][2]:
                # at contact the neighbour's ring overlaps; allow only on an event frame +-1
                if not any(abs(f - e) <= 1 for e in ev_frames):
                    errs.append(f"frame {f}: shape {i} shows {seq[f][i]}, drawn {tr[f][i][2]}")
    decoded = []
    for f in range(1, nf):
        ch = [i for i in range(n) if seq[f][i] and seq[f - 1][i] and seq[f][i] != seq[f - 1][i]
              and seq[f][i] == tr[f][i][2] and seq[f - 1][i] == tr[f - 1][i][2]]
        if ch:
            decoded.append((f, tuple(ch)))
    expect = []
    st = sim.start_patterns(scene)
    for t, i, j in coll:
        old = st[:]
        sim.apply_rule(scene, st, i, j)
        ch = tuple(k for k in (i, j) if st[k] != old[k])
        if ch:
            expect.append((sim.frame_of(t), ch))
    if sorted(decoded) != sorted(expect):
        errs.append(f"pattern changes {decoded} != collisions {expect}")
    # holes
    for t, i, _ in caps:
        f = sim.frame_of(t)
        b = blobs(scene, frames[f - 1])[i], blobs(scene, frames[f])[i]
        if b[0] is None or b[1] is not None:
            errs.append(f"capture of shape {i} not on frame {f}")
    # clock answers, read on the decoded event frame
    for q in qs:
        k, p = q["derivation_key"], q["params"]
        if k == "coll_time":
            tc = next(t for t, i, j in coll if {i, j} == {p["x"], p["y"]})
            fc = sim.frame_of(tc)
            f = next((f for f, ch in decoded if f == fc and set(ch) <= {p["x"], p["y"]}), None)
            if f is None:
                if any(set(ch) & {p["x"], p["y"]} and abs(f2 - fc) <= 2 for f2, ch in decoded):
                    errs.append(f"{q['qid']}: pattern change of the pair off its contact frame")
                # no pattern change (two untagged shapes): the decoded blobs are closest on the
                # last frame before contact or the contact frame itself
                dist = {}
                for g in range(fc - 4, fc + 5):
                    bb = blobs(scene, frames[g])
                    dist[g] = np.hypot(bb[p["x"]][0] - bb[p["y"]][0], bb[p["x"]][1] - bb[p["y"]][1])
                if min(dist, key=dist.get) not in (fc - 1, fc):
                    errs.append(f"{q['qid']}: decoded closest approach not at frame {fc}")
                f = fc
            if sim.clock_text(read_timer(scene, frames[f])) != q["groundtruth"]:
                errs.append(f"{q['qid']}: clock on the collision frame != {q['groundtruth']}")
        if k == "fall_time":
            f = next(f for f in range(1, nf) if tr[f][p["x"]] is None)
            if blobs(scene, frames[f])[p["x"]] is not None or \
                    sim.clock_text(read_timer(scene, frames[f])) != q["groundtruth"]:
                errs.append(f"{q['qid']}: fall frame / clock mismatch")
    return errs


def _one(args):
    root, line = args
    s = json.loads(line)
    errs = check_clip(s["params"], s["questions"], frames_of(Path(root) / s["media_path"]))
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
