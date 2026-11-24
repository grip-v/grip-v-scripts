"""Frame-level check of rendered rolling_races clips against the stored annotations.

    python frame_check.py --dataset-dir <dir> [--report frame_check_report.txt]

Per clip it decodes the MP4 (OpenCV) and checks:
  * clock: every frame's clock box matches the rendered text for that frame
    better than any other reading 0.0-10.0 s (template match = OCR);
  * no other text: nothing but the clock, the scene objects and the balls
    differs from the re-rendered static layer outside the moving objects;
  * start gate: present on exactly the frames before the release frame;
  * balls: every blob that differs from the (exactly re-rendered) static
    layer is located (centre of its outline box) and named by the palette
    colour most of its pixels match; isolated balls must agree with
    render.state within TOL px, same set of colours;
  * finish rack: a slot reads as filled (its ball's colour) exactly from the
    frame the ball reaches the finish; fallen balls rest on the pit floor;
then re-derives EVERY answer from the decoded state (balls, rack, gate) with
answers.answer(..., frames=decoded, jitter=JITTER) and checks the decoded
clock at the frame of every clock-reading answer.
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

TOL = 3.0
JITTER = 2.0
MATCH_FOUND, MATCH_OK = 12000, 6000   # mean squared colour error of a sprite match (isolated balls: p99 3750)
STATIC_SAME = 1500                   # mean squared difference below which a spot is just the static layer
ISO = 12                  # a ball counts as isolated when no other ball's edge is within this many px
CLOCK_KEYS = {"finish_clock": "fin", "floor_clock": "rest", "release_clock": "release"}


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


def clock_img(scene, tenths):
    box = tuple(sim.clock_box(scene))
    if (box, tenths) not in _CLOCKS:
        img = Image.new("RGB", render.SIZE, (0, 0, 0))
        render._clock(ImageDraw.Draw(img), scene, (tenths * 24 + 9) // 10)
        x0, y0, x1, y1 = box
        _CLOCKS[(box, tenths)] = np.asarray(img, dtype=np.int16)[y0 + 4:y1 - 3, x0 + 4:x1 - 3]
    return _CLOCKS[(box, tenths)]


def read_clock(scene, frame):
    x0, y0, x1, y1 = sim.clock_box(scene)
    crop = frame[y0 + 4:y1 - 3, x0 + 4:x1 - 3].astype(np.int16)
    diffs = {k: float(np.abs(crop - clock_img(scene, k)).mean()) for k in range(0, 101)}
    return min(diffs, key=diffs.get)


def static_layer(scene):
    return np.asarray(render._static(json.dumps(scene, sort_keys=True)), dtype=np.int16)


def masks(scene):
    """Regions excluded from the ball search: clock, rack and the finish hole (a dark ball over the dark hole
    is not separable from it)."""
    m = np.zeros((render.SIZE[1], render.SIZE[0]), dtype=bool)
    x0, y0, x1, y1 = sim.clock_box(scene)
    m[y0 - 3:y1 + 4, x0 - 3:x1 + 4] = True
    R = scene["style"]["R"]
    for c in scene["rack"]:
        x, y = sim.slot_px(scene, c)
        m[int(y - R - 6):int(y + R + 7), int(x - R - 8):int(x + R + 9)] = True
    hx, hy = sim.hole_px(scene)
    m[int(hy - 2 * R - 14):int(hy + 16), int(hx - R - 28):int(hx + R + 29)] = True
    return m


_SPRITES: dict = {}


def sprite(scene, b):
    """The ball exactly as render._ball draws it (centre at (h, h)), and a mask of its disc."""
    st = scene["style"]
    key = (b["colour"], b["pattern"], st["R"], st["theme"], tuple(st["ink"]))
    if key not in _SPRITES:
        R = st["R"]
        h = R + 4
        img = Image.new("RGB", (2 * h + 1, 2 * h + 1), (0, 0, 0))
        render._ball(ImageDraw.Draw(img), scene, b, h, h)
        yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
        mask = ((xx * xx + yy * yy) <= (R - 0.5) ** 2).astype(np.float32)
        _SPRITES[key] = (np.asarray(img, dtype=np.float32), mask, h)
    return _SPRITES[key]


def other_masks(scene):
    """colour -> bool mask of the other colours' tracks (drawn in front of that ball, render.other_tracks)."""
    return {c: np.asarray(m) > 0 for c, m in render.other_tracks(json.dumps(scene, sort_keys=True)).items()}


def _behind_sqdiff(win, swin, spr, mask, cover):
    """Mean squared colour error at every offset of `win` for a ball drawn BEHIND the other colours' tracks:
    inside its disc the expected image is the sprite, except where another track (`cover`) passes in front,
    where it is the static layer (`swin`). Every disc pixel is compared."""
    from numpy.lib.stride_tricks import sliding_window_view as sw
    th, tw = spr.shape[:2]
    W_ = sw(win, (th, tw), axis=(0, 1))                       # (oh, ow, 3, th, tw)
    S_ = sw(swin, (th, tw), axis=(0, 1))
    C_ = sw(cover, (th, tw), axis=(0, 1))[:, :, None]         # (oh, ow, 1, th, tw)
    exp = np.where(C_, S_, np.moveaxis(spr, -1, 0)[None, None])
    return (((W_ - exp) ** 2) * mask[None, None, None]).sum(axis=(2, 3, 4)) / (3 * mask.sum())


def find_balls(frame, scene, static, excl, others=None, skip=()):
    """[(colour, cx, cy, ok)] per located ball. Candidates: each palette colour's pixel mask averaged over
    a square of side 1.4 R (a ball fills most of it at its centre, a track line of the same colour at most
    about a third). Each candidate is refined by matching the ball's own sprite (colour, pattern, outline)
    over a +-6 px window, masked to its disc: the ball is drawn on top of the tracks, so its disc pixels
    match the sprite wherever it is. ok = the match is close (a whole ball, not one partly covered by
    another ball)."""
    st = scene["style"]
    R = st["R"]
    side = int(1.4 * R) | 1
    out = []
    fr = frame.astype(np.int16)
    frf = frame.astype(np.float32)
    H, Wd = frame.shape[:2]
    byc = {b["colour"]: b for b in scene["balls"]}
    for b in scene["balls"]:
        c = b["colour"]
        if c in skip:                                    # read as already in the finish rack
            continue
        rgb = np.array(render.colour(scene, c), dtype=np.int16)
        hit = np.abs(fr - rgb).sum(-1) < 90
        spr, mask, h = sprite(scene, b)

        def search(resp, thr, budget, early):
            best = None
            for _ in range(budget):
                _, mx, _, (px_, py_) = cv2.minMaxLoc(resp)
                if mx < thr:
                    break
                q = max(2, R // 2)                    # small: a track peak next to the ball must not hide it
                resp[max(0, py_ - q):py_ + q + 1, max(0, px_ - q):px_ + q + 1] = 0
                for _ in range(4):                   # re-centre while the best match sits on the window edge
                    x0, y0 = px_ - h - 6, py_ - h - 6
                    if x0 < 0 or y0 < 0 or x0 + 2 * h + 13 > Wd or y0 + 2 * h + 13 > H:
                        sq = None
                        break
                    win = frf[y0:y0 + 2 * h + 13, x0:x0 + 2 * h + 13]
                    sq = cv2.matchTemplate(win, spr, cv2.TM_SQDIFF, mask=np.dstack([mask] * 3))
                    sq = sq / (3 * mask.sum())
                    cover = others[c][y0:y0 + 2 * h + 13, x0:x0 + 2 * h + 13] if others else None
                    if cover is not None and cover.any():       # or a ball in the air, behind other tracks
                        swin = static[y0:y0 + 2 * h + 13, x0:x0 + 2 * h + 13].astype(np.float32)
                        sq = np.minimum(sq, _behind_sqdiff(win, swin, spr, mask, cover))
                    _, _, (ix, iy), _ = cv2.minMaxLoc(sq)
                    if 0 < ix < sq.shape[1] - 1 and 0 < iy < sq.shape[0] - 1:
                        break
                    px_, py_ = px_ + ix - 6, py_ + iy - 6
                if sq is None:
                    continue
                e = float(sq[iy, ix])
                dx = dy = 0.0
                if 0 < ix < sq.shape[1] - 1:
                    a_, b_, c_ = sq[iy, ix - 1], sq[iy, ix], sq[iy, ix + 1]
                    dx = 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_) if a_ - 2 * b_ + c_ > 0 else 0.0
                if 0 < iy < sq.shape[0] - 1:
                    a_, b_, c_ = sq[iy - 1, ix], sq[iy, ix], sq[iy + 1, ix]
                    dy = 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_) if a_ - 2 * b_ + c_ > 0 else 0.0
                # a ball differs from the static layer under it; a dark band of tracks or the ledge does not
                bx, by = x0 + ix, y0 + iy
                under = static[by:by + 2 * h + 1, bx:bx + 2 * h + 1].astype(np.float32)
                if float(((frf[by:by + 2 * h + 1, bx:bx + 2 * h + 1] - under) ** 2 * mask[..., None]).sum()
                         / (3 * mask.sum())) < STATIC_SAME:
                    continue
                # of all the balls' sprites, this ball's own must fit the disc (less any other track over it) best:
                # a similar-coloured ball crossed by this ball's track is not this ball
                sub = frf[by:by + 2 * h + 1, bx:bx + 2 * h + 1]

                def fit(k):                                     # best of: ball k on top / ball k behind other tracks
                    sk = sprite(scene, byc[k])[0]
                    e1 = float((((sub - sk) ** 2).sum(-1) * mask).sum() / (3 * mask.sum()))
                    if not others:
                        return e1
                    cv = others[k][by:by + 2 * h + 1, bx:bx + 2 * h + 1][..., None]
                    exp = np.where(cv, under, sk)
                    return min(e1, float((((sub - exp) ** 2).sum(-1) * mask).sum() / (3 * mask.sum())))
                err = {k: fit(k) for k in byc}
                if min(err, key=err.get) != c:
                    continue
                cand = (e, x0 + ix + dx + h, y0 + iy + dy + h)
                if best is None or cand[0] < best[0]:
                    best = cand
                if early and best[0] < MATCH_OK:
                    break
            return best

        # stage 1: a ball on top of the tracks; strict colour response, first good match
        m = hit.astype(np.float32)
        m[excl] = 0
        best = search(cv2.boxFilter(m, -1, (side, side), normalize=True, borderType=cv2.BORDER_CONSTANT),
                      0.4, 12, True)
        if (best is None or best[0] >= MATCH_OK) and others:
            # stage 2: a ball in the air behind other tracks; pixels another track hides next to this colour
            # count as hits, every candidate is tried and the best match is kept
            near = cv2.dilate(hit.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            m = (hit | (others[c] & near)).astype(np.float32)
            m[excl] = 0
            b2 = search(cv2.boxFilter(m, -1, (side, side), normalize=True, borderType=cv2.BORDER_CONSTANT),
                        0.25, 100, False)
            if b2 is not None and (best is None or b2[0] < best[0]):
                best = b2
        if best is not None and best[0] < MATCH_FOUND:
            out.append((c, best[1], best[2], best[0] < MATCH_OK))
    return out


def rack_filled(frame, scene, static):
    out = []
    R = scene["style"]["R"]
    for c in scene["rack"]:
        x, y = (int(round(v)) for v in sim.slot_px(scene, c))
        patch = frame[y - R // 2:y + R // 2 + 1, x - R // 2:x + R // 2 + 1].astype(np.int16).reshape(-1, 3)
        rgb = np.array(render.colour(scene, c), dtype=np.int16)
        share = float((np.abs(patch - rgb).sum(-1) < 90).mean())
        if share > 0.35:
            out.append(c)
    return sorted(out)


def gate_on(frame, scene, static):
    """The gate is a 6 px ink bar: its box reads as ink while the strip just downhill of it does not (a ball
    passing over the spot after the release darkens both)."""
    g = [int(round(v)) for v in render.gate_box(scene)]
    ink = np.array(scene["style"]["ink"], dtype=np.int16)
    share = lambda p: float((np.abs(p.astype(np.int16).reshape(-1, 3) - ink).sum(-1) < 90).mean())
    bar = frame[g[1] + 2:g[3] - 1, g[0] + 1:g[2]]
    d = -1 if scene["style"]["mirror"] else 1
    sx0 = g[2] + 3 if d > 0 else g[0] - 7
    side = frame[g[1] + 2:g[3] - 1, sx0:sx0 + 4]
    return share(bar) > 0.6 and share(side) < 0.3


def interpolate(decoded, cols, max_gap=12):
    """Fill short runs of frames where a ball was not located (merged with another ball) by linear
    interpolation between the frames around them; never extrapolate."""
    for c in cols:
        have = [f for f, d in enumerate(decoded) if c in d["balls"]]
        for a, b in zip(have, have[1:]):
            if 1 < b - a <= max_gap:
                pa, pb = decoded[a]["balls"][c], decoded[b]["balls"][c]
                for f in range(a + 1, b):
                    u = (f - a) / (b - a)
                    if c not in decoded[f]["rack"]:
                        decoded[f]["balls"][c] = (pa[0] + u * (pb[0] - pa[0]), pa[1] + u * (pb[1] - pa[1]))
    return decoded


def check_clip(s, root):
    scene = s["params"]
    errs = []
    info = {"frames": 0, "clock_ok": 0, "q_checked": 0, "ball_err": 0.0, "ball_cmp": 0, "rack_ok": 0, "gate_ok": 0,
            "occluded": 0}
    frames = decode(root / s["media_path"])
    N = render.n_frames(scene)
    if len(frames) != N:
        return [f"decoded {len(frames)} frames, expected {N}"], info
    info["frames"] = N
    static = static_layer(scene)
    excl = masks(scene)
    others = other_masks(scene)
    R = scene["style"]["R"]
    decoded = []
    for f, fr in enumerate(frames):
        if read_clock(scene, fr) == sim.clock_tenths(f):
            info["clock_ok"] += 1
        else:
            errs.append(f"frame {f}: clock misread")
        S = render.state(scene, f)
        gate = gate_on(fr, scene, static)
        if gate == S["gate"]:
            info["gate_ok"] += 1
        else:
            errs.append(f"frame {f}: gate reads {gate}, drawn {S['gate']}")
        rack = rack_filled(fr, scene, static)
        if rack == S["rack"]:
            info["rack_ok"] += 1
        else:
            errs.append(f"frame {f}: rack reads {rack}, drawn {S['rack']}")
        found = find_balls(fr, scene, static, excl, others, skip=set(rack))
        whole = {}
        for c, x, y, ok in found:
            if ok:
                whole.setdefault(c, []).append((x, y))
        # isolated drawn balls (nothing else within 2R + ISO px) must be found where they are drawn
        for c, p in S["balls"].items():
            if any(math.dist(p, q) < 2 * R + ISO for k, q in S["balls"].items() if k != c):
                continue
            x0, y0 = int(p[0] - R - 2), int(p[1] - R - 2)
            if excl[max(0, y0):y0 + 2 * R + 5, max(0, x0):x0 + 2 * R + 5].any():
                continue
            if c in S["air"]:                       # behind other tracks: skip if they hide over 35 % of it
                yy, xx = np.mgrid[max(0, y0):y0 + 2 * R + 5, max(0, x0):x0 + 2 * R + 5]
                disc = np.hypot(xx - p[0], yy - p[1]) <= R
                if (others[c][yy, xx] & disc).sum() > 0.35 * disc.sum():
                    info["occluded"] += 1
                    continue
            got = whole.get(c, [])
            if not got:
                errs.append(f"frame {f}: {c} ball not found (drawn at {p[0]:.0f},{p[1]:.0f})")
                continue
            e = min(math.dist(p, q) for q in got)
            info["ball_err"] = max(info["ball_err"], e)
            info["ball_cmp"] += 1
            if e > TOL:
                errs.append(f"frame {f}: {c} ball off by {e:.1f} px")
        extra = sorted(set(whole) - set(S["balls"]))
        if extra:
            errs.append(f"frame {f}: blobs of {extra} where no such ball is drawn")
        decoded.append({"balls": {c: v[0] for c, v in whole.items() if len(v) == 1}, "rack": rack, "gate": gate})
    decoded = interpolate(decoded, [b["colour"] for b in scene["balls"]])
    o = answers.Obs(scene, decoded, JITTER)
    for q in s["questions"]:
        key = q["derivation_key"]
        info["q_checked"] += 1
        try:
            got = answers.answer(key, q["params"], scene, frames=decoded, jitter=JITTER)
        except Exception as ex:  # noqa: BLE001
            got = f"error {ex!r}"
        if str(got) != str(q["groundtruth"]):
            errs.append(f"{q['qid']} {key}: from decoded frames {got!r}, stored {q['groundtruth']!r}")
        if key in CLOCK_KEYS or key == "leave_clock":
            if key == "release_clock":
                f = o.release
            elif key == "leave_clock":
                f = math.ceil(o.leave_t(q["params"]["x"], q["params"]["gap"]) * 24 - 1e-9)
            else:
                f = getattr(o, CLOCK_KEYS[key])[q["params"]["x"]]
            tenths = round(float(q["groundtruth"].split()[0]) * 10)
            if f is None or read_clock(scene, frames[f]) != tenths:
                errs.append(f"{q['qid']}: the decoded clock at the event frame does not show {q['groundtruth']}")
    return errs, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--report", default="frame_check_report.txt")
    ap.add_argument("--shard", default="0/1", help="i/n: check every n-th clip starting at i")
    args = ap.parse_args(argv)
    si, sn = (int(v) for v in args.shard.split("/"))
    root = Path(args.dataset_dir).resolve()
    lines, total = [], 0
    keys = ("frames", "clock_ok", "q_checked", "ball_cmp", "rack_ok", "gate_ok", "occluded")
    tot = {k: 0 for k in keys} | {"ball_err": 0.0}
    with open(root / "annotations.jsonl", encoding="utf8") as fh:
        rows = fh.readlines()[si::sn]
    for line in rows:
        s = json.loads(line)
        errs, info = check_clip(s, root)
        for k in keys:
            tot[k] += info[k]
        tot["ball_err"] = max(tot["ball_err"], info["ball_err"])
        total += len(errs)
        lines.append(f"{s['sample_id']} ({s['params']['scenario']}): {len(errs)} problems, max ball error "
                     f"{info['ball_err']:.2f} px over {info['ball_cmp']} isolated balls")
        lines += [f"  - {e}" for e in errs[:20]]
        print(lines[-1 - min(20, len(errs))], flush=True)
        for e in errs[:5]:
            print("   ", e)
    summary = (f"RESULT: {len(rows)} clips, {tot['frames']} frames, clock template match {tot['clock_ok']}/{tot['frames']}, "
               f"gate {tot['gate_ok']}/{tot['frames']}, rack {tot['rack_ok']}/{tot['frames']}, {tot['ball_cmp']} isolated "
               f"ball positions compared (max error {tot['ball_err']:.2f} px; {tot['occluded']} airborne balls mostly behind other tracks not compared), {tot['q_checked']} questions re-derived "
               f"from decoded frames, {total} problems")
    (root / args.report).write_text("\n".join(lines + ["", summary]) + "\n", encoding="utf8")
    print(summary)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
