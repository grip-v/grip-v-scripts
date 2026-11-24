"""Extra checks for turntable_voxels, driven from run.py:

  python run.py stats --n 5000 [--seed 0] [--workers 8]   diversity + ladder stats over dry-run scenes
  python run.py tamper --dataset-dir DIR [--k 3]           corrupt k answers, count what the validator flags
  python run.py framecheck --dataset-dir DIR               read the rendered MP4s back and compare with the scene
  python run.py framehash --seed S                         sha256 of the raw rendered frames of one seed
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import answers
import questions
import render
import sim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ladder5 as L5  # noqa: E402
from common.video5 import FPS, build_sample, _roundtrip  # noqa: E402

DATASET = "turntable_voxels"


# ----------------------------------------------------------------- stats
def _one(seed):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = _roundtrip(build_sample(DATASET, sim, questions, seed))
        sc = s["params"]
        mism = sum(str(answers.answer(q["derivation_key"], q["params"], sc)) != str(q["groundtruth"])
                   for q in s["questions"])
        h0 = sc["heights"]
        n0 = sum(map(sum, h0))
        return {
            "sig": sim.sig_hash(sc), "scenario": sc["scenario"], "tries": sc["tries"],
            "ladder": bool(L5.ladder_errors(s["questions"], "ten")), "check": bool(sim.check(sc)), "mism": mism,
            "knobs": {"grid": f"{sc['gx']}x{sc['gy']}", "cubes": n0, "tallest": max(map(max, h0)),
                      "hidden_start": n0 - len(sim.visible(sc, 0.0, h0)), "S_px": round(sc["S"]),
                      "elev": round(sc["elev"]), "duration": round(sc["duration"], 1),
                      "peak_deg_s": round(sim.peak_speed(sc)), "quarters": sum(round(abs(lg["a1"] - lg["a0"]) / 90)
                                                                              for lg in sc["legs"]),
                      "marked": len(sc["marked"]), "events": len(sc["events"])},
            "cats": {"layout": sc["layout"], "base": sc["theme"]["base"], "bg": sc["theme"]["bg"],
                     "bg_style": sc["theme"]["bg_style"], "disc": sc["theme"]["disc"],
                     "marked_colours": ",".join(sorted(m["colour"] for m in sc["marked"]))},
            "qs": [(q["difficulty_level"], q["derivation_key"], q["question"], q["answer_text"], q["answer"],
                    q["params"]["min_fps"], q["requires_motion"]) for q in s["questions"]],
        }


def stats(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    seeds = list(range(a.seed, a.seed + a.n))
    if a.workers > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:
            rows = pool.map(_one, seeds, chunksize=20)
    else:
        rows = [_one(s) for s in seeds]
    sigs = Counter(r["sig"] for r in rows)
    scen = Counter(r["scenario"] for r in rows)
    pairs, letters, levels, fps, keys, motion = set(), Counter(), Counter(), [], Counter(), Counter()
    for r in rows:
        lc = Counter(q[0] for q in r["qs"])
        levels[tuple(lc[k] for k in range(1, 6))] += 1
        for lv, key, text, ans, let, mf, mot in r["qs"]:
            pairs.add((text, ans))
            letters[let] += 1
            fps.append(mf)
            keys[(lv, key)] += 1
            if lv >= 2:
                motion[mot] += 1
    nq = len(fps)
    tries = [r["tries"] for r in rows]
    print(f"scenes: {a.n}  unique signatures: {len(sigs)} ({100 * len(sigs) / a.n:.2f}%)")
    print("scenario mix:", dict(sorted(scen.items())))
    for k in rows[0]["knobs"]:
        v = [r["knobs"][k] for r in rows]
        if isinstance(v[0], str):
            print(f"  {k:13s}", dict(sorted(Counter(v).items())))
        else:
            print(f"  {k:13s} min {min(v)}  median {statistics.median(v)}  max {max(v)}  distinct {len(set(v))}")
    for k in rows[0]["cats"]:
        c = Counter(r["cats"][k] for r in rows)
        print(f"  {k}: {len(c)} values", dict(sorted(c.items())) if len(c) <= 12 else "")
    print(f"questions: {nq}  distinct (question, answer) pairs: {len(pairs)}")
    print("answer letters:", {k: f"{100 * v / nq:.1f}%" for k, v in sorted(letters.items())})
    print("level counts (L1..L5):", {k: v for k, v in levels.most_common()})
    print(f"L2-L5 requires_motion: {100 * motion[True] / max(1, sum(motion.values())):.1f}%")
    print(f"min_fps: <=1: {100 * sum(f <= 1 for f in fps) / nq:.1f}%  <=2: {100 * sum(f <= 2 for f in fps) / nq:.1f}%"
          f"  <=4: {100 * sum(f <= 4 for f in fps) / nq:.1f}%  max {max(fps)}")
    print(f"retries: mean tries/scene {statistics.mean(tries):.2f}, max {max(tries)}, "
          f"rejected attempts {sum(tries) - a.n} ({100 * (sum(tries) - a.n) / sum(tries):.1f}% of attempts)")
    by = Counter()
    for r in rows:
        by[r["scenario"]] += r["tries"]
    print("  mean tries by scenario:", {k: round(by[k] / scen[k], 2) for k in sorted(scen)})
    print("templates:", {f"L{k[0]}:{k[1]}": v for k, v in sorted(keys.items())})
    bad = (sum(r["ladder"] for r in rows), sum(r["check"] for r in rows), sum(r["mism"] for r in rows))
    print(f"ladder errors: {bad[0]}  check failures: {bad[1]}  answer mismatches: {bad[2]}")
    return sum(bad) + (a.n - len(sigs))


# ---------------------------------------------------------------- tamper
def tamper(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    root = Path(a.dataset_dir)
    rows = [json.loads(x) for x in open(root / "annotations.jsonl", encoding="utf8")]
    rng = random.Random(a.seed)
    targets = rng.sample([(i, j) for i, r in enumerate(rows) for j in range(len(r["questions"]))], a.k)
    for i, j in targets:
        q = rows[i]["questions"][j]
        wrong = next(x for x in "ABCD"[:len(q["options"])] if x != q["answer"])
        q["answer"] = wrong
        q["answer_text"] = q["options"]["ABCD".index(wrong)]
        q["groundtruth"] = q["answer_text"]
    flagged = 0
    for r in rows:
        for q in r["questions"]:
            if str(answers.answer(q["derivation_key"], q["params"], r["params"])) != str(q["groundtruth"]):
                flagged += 1
    print(f"tamper: corrupted {a.k} answers -> {flagged} flagged by the independent re-derivation")
    (root / "annotations_tampered.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
    return flagged


# ------------------------------------------------------------ frame check
def _read_frames(path):
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


def _timer_crop(t):
    img = Image.new("RGB", (sim.W, sim.H), (0, 0, 0))
    render.draw_timer(img, t)
    x0, y0, x1, y1 = sim.TIMER_BOX
    return np.asarray(img, dtype=np.int16)[y0:y1, x0:x1]


def timer_ok(frame, t):
    """The clock box shows t (to 0.1 s): closest of the readings t - 0.1, t, t + 0.1."""
    x0, y0, x1, y1 = sim.TIMER_BOX
    got = frame[y0:y1, x0:x1]
    want = f"{t:.1f}"
    others = {f"{t + dt:.1f}" for dt in (-0.1, 0.1) if t + dt >= 0} - {want}
    e0 = np.abs(got - _timer_crop(t)).mean()
    eo = min(np.abs(got - _timer_crop(float(o))).mean() for o in others)
    return e0 < 8 and e0 < eo, e0


class Notch:
    """Reads the turntable angle off a frame from the notch (marker colour inside the rim annulus)."""

    def __init__(self, sc):
        R = sim.disc_radius(sc)
        ce, se = sim.cs(sc["elev"])
        S, cx, cy = sc["S"], sc["cx"], sc["cy"]
        yy, xx = np.mgrid[0:sim.H, 0:sim.W].astype(float)
        self.xr = (xx + 0.5 - cx) / S
        self.yr = (cy - yy - 0.5) / (S * se)
        r = np.hypot(self.xr, self.yr)
        self.ann = (r > R - render.NOTCH[0] - 0.04) & (r < R - render.NOTCH[1] + 0.04)
        ys, xs = np.nonzero(self.ann)
        self.box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
        y0, y1, x0, x1 = self.box
        self.ann, self.xr, self.yr = self.ann[y0:y1, x0:x1], self.xr[y0:y1, x0:x1], self.yr[y0:y1, x0:x1]
        self.col = np.array(sc["theme"]["marker_rgb"])
        rm = R - (render.NOTCH[0] + render.NOTCH[1]) / 2
        self.expect = (render.NOTCH[0] - render.NOTCH[1]) * rm * math.radians(2 * render.NOTCH[2]) * S * S * se

    def read(self, fr, frac=0.8):
        import cv2
        y0, y1, x0, x1 = self.box
        m = self.ann & (np.abs(fr[y0:y1, x0:x1] - self.col).sum(axis=2) < 45)
        # opening with a 5x5 square drops 3 px cube outlines that happen to be marker-dark / -light
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
        if m.sum() < frac * self.expect:          # (partly) hidden behind a cube: no read
            return None
        self.last_xy = (self.xr[m].mean(), self.yr[m].mean())
        return math.degrees(math.atan2(self.yr[m].mean(), self.xr[m].mean())) + 90.0


def _adiff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def _rot_point(sc, k, xp, yp, z):
    """Screen point of a point given in the square-on layout after k quarter turns."""
    gx, gy = (sc["gx"], sc["gy"]) if k % 2 == 0 else (sc["gy"], sc["gx"])
    ce, se = sim.cs(sc["elev"])
    return sc["cx"] + sc["S"] * (xp - gx / 2), sc["cy"] - sc["S"] * ((yp - gy / 2) * se + z * ce)


def _escape_points(h, elev):
    """For each visible cube of a square-on heightmap: a screen-layout point in the middle of its visible
    band, with the face it lies on. Independent of answers.py's sampling (dense band search)."""
    ce, se = math.cos(math.radians(elev)), math.sin(math.radians(elev))
    gy, gx = len(h), len(h[0])

    def clear(x, y0, z0, row):
        for yy in range(row, -1, -1):
            u = (y0 - (yy + 1)) / ce
            if h[yy][x] > z0 + se * max(u, 0.0) + 1e-9:
                return False
        return True

    out = {}
    for y in range(gy):
        for x in range(gx):
            for z in range(h[y][x]):
                best = None
                for face in ("top", "front"):
                    if face == "top" and z != h[y][x] - 1:
                        continue
                    fs = [(i + 0.5) / 64 for i in range(64)]
                    ok = [f for f in fs if (clear(x, y + f, z + 1, y - 1) if face == "top" else
                                            clear(x, y, z + f, y - 1))]
                    if ok and (best is None or len(ok) > len(best[1])):
                        best = (face, ok)
                if best:
                    face, ok = best
                    f = ok[len(ok) // 2]
                    pt = (x + 0.5, y + f, z + 1) if face == "top" else (x + 0.5, y, z + f)
                    out[(x, y, z)] = (face, pt, len(ok) / 64)
    return out


def _unrot(sc, k, cell):
    """Turned-layout cell -> start-layout cell."""
    x, y, z = cell
    gx = sc["gx"] if k % 2 == 0 else sc["gy"]
    gy = sc["gy"] if k % 2 == 0 else sc["gx"]
    for _ in range(k):
        x, y = y, gx - 1 - x
        gx, gy = gy, gx
    return (x, y, z)


def colour_mask(px, rgb, rivals, tol=40.0):
    """Pixels that are a shade (x0.55-1.02) of rgb, closer to that shade line than to any rival's."""
    def resid(c):
        c = np.asarray(c, dtype=float)
        f = np.clip((px * c).sum(axis=-1) / (c * c).sum(), 0.0, 2.0)
        return np.abs(px - f[..., None] * c).sum(axis=-1), f
    r, f = resid(rgb)
    m = (r < tol) & (f > 0.55) & (f < 1.02)
    for c in rivals:
        rr, _ = resid(c)
        m &= r < rr
    return m


def _rivals(sc, colour):
    th = sc["theme"]
    dc, rim, grid = render.disc_colours(sc)
    out = [th["base_rgb"], th["bg_rgb"], dc, rim, grid, th["marker_rgb"], sim.OUTLINE]
    out += [sim.COLOURS[m["colour"]] for m in sc["marked"] if m["colour"] != colour]
    return [c for c in out if list(c) != list(colour)]


def _colour_of(sc, cell, mk):
    c = mk.get(cell, "base")
    return sc["theme"]["base_rgb"] if c == "base" else sim.COLOURS[c]


def framecheck_sample(s, video_path):
    sc = s["params"]
    frames = _read_frames(video_path)
    errs, n = [], Counter()
    nfr, last = len(frames), len(frames) - 1
    if nfr != int(round(sc["duration"] * FPS)):
        errs.append(f"video has {nfr} frames for {sc['duration']} s")
    # 1. turntable angle from the notch, every frame; compared with the turning programme
    notch = Notch(sc)
    reads, xy, loose = [], [], []
    for fr in frames:
        reads.append(notch.read(fr))
        xy.append(notch.last_xy if reads[-1] is not None else None)
        loose.append(reads[-1] if reads[-1] is not None else notch.read(fr, 0.25))   # for unwrapping only
    # which way the near edge moves on screen while the notch is in front: from screen x of the notch
    lg = sc["legs"][0]
    dx = [xy[f + 1][0] - xy[f][0] for f in range(int(lg["t0"] * FPS) + 1, min(last, int(lg["t1"] * FPS)))
          if xy[f] and xy[f + 1] and xy[f][1] < 0 and xy[f + 1][1] < 0]
    if dx:
        n["near_edge_reads"] += 1
        exp = 1 if lg["a1"] > lg["a0"] else -1
        if (1 if sum(dx) > 0 else -1) != exp:
            errs.append("near edge moves the wrong way on screen in the first turn")
    ok_reads = 0
    for f in range(0, nfr, 6):                       # 4 fps
        n["notch_samples"] += 1
        if reads[f] is None:
            continue
        ok_reads += 1
        if abs(_adiff(reads[f], sim.angle(sc, f / FPS))) > 8:
            errs.append(f"notch at frame {f} reads {reads[f]:.0f} deg, expected {sim.angle(sc, f / FPS) % 360:.0f}")
    n["notch_reads"] += ok_reads
    # unwrapped track from the readable frames -> turning runs, quarters, final side
    track, prev = [], None
    gap_max, last_f = 0.0, 0
    for f, r in enumerate(loose):
        if r is None:
            continue
        # how far the scene says the turntable turns while the notch is out of sight (unwrapping limit)
        gap_max = max(gap_max, abs(sim.angle(sc, f / FPS) - sim.angle(sc, last_f / FPS)))
        last_f = f
        if prev is None:
            track.append((f, r))
        else:
            track.append((f, track[-1][1] + _adiff(r, prev)))
        prev = r
    n["notch_track_frames"] += len(track)
    n["notch_frames"] += nfr
    if len(track) < 0.5 * nfr or gap_max >= 170:
        errs.append(f"notch track too sparse ({len(track)}/{nfr} frames, {gap_max:.0f} deg turned unseen)")
    else:
        runs, cur = [], None
        for (f0, a0), (f1, a1) in zip(track, track[1:]):
            d = (a1 - a0) / (f1 - f0)
            st = 0 if abs(d) < 0.4 else (1 if d > 0 else -1)
            if cur and cur[0] == st:
                cur[2], cur[3] = f1, cur[3] + (a1 - a0)
            else:
                cur = [st, f0, f1, a1 - a0]
                runs.append(cur)
        # merge slow ends of a turn (ease in / out) into the neighbouring turn of the same sign
        moves = []
        for r in runs:
            if r[0] == 0 and abs(r[3]) > 3:
                r[0] = 1 if r[3] > 0 else -1
            if r[0] and moves and moves[-1][0] == r[0] and r[1] - moves[-1][2] <= 3:
                moves[-1][2], moves[-1][3] = r[2], moves[-1][3] + r[3]
            elif r[0]:
                moves.append(list(r))
        moves = [m for m in moves if abs(m[3]) > 20]
        got_q = sum(int(round(abs(m[3]) / 90)) for m in moves)
        for q in s["questions"]:
            key, gt = q["derivation_key"], q["groundtruth"]
            got = None
            if key == "total_quarters":
                got = got_q
            elif key == "turn_dirs":
                sg = [m[0] for m in moves]
                got = questions.dirs_text(sg) if sg else None
            elif key == "first_leg" and moves:
                got = questions.leg_text(int(round(abs(moves[0][3]) / 90)), moves[0][0])
            elif key == "facing_end":
                got = questions.FACING[int(round(track[-1][1] / 90)) % 4]
            elif key == "facing_stop":
                f = int(round((sim.holds(sc)[1][0] + sim.holds(sc)[1][1]) / 2 * FPS))
                near = [a for ff, a in track if abs(ff - f) <= 6]
                got = questions.FACING[int(round(near[0] / 90)) % 4] if near else None
            tail = abs(sim.angle(sc, last / FPS) - sim.angle(sc, track[-1][0] / FPS))
            if key in ("total_quarters", "facing_end") and tail > 20:
                n["rotation_answers_unreadable(notch hidden at the end)"] += 1      # the notch sits behind the cubes
                continue
            if key in ("total_quarters", "turn_dirs", "first_leg", "facing_end", "facing_stop"):
                n["rotation_answers"] += 1
                if str(got) != str(gt):
                    errs.append(f"{q['qid']} {key}: notch track gives {got!r}, answer {gt!r}")
    # 2. still views: marked-cube colour blobs and the colour in the middle of every visible cube's band
    mk = sim.marked_map(sc)
    holds = sim.holds(sc)
    for hi, (t0, t1, a) in enumerate(holds):
        f = min(last, int(round((t0 + t1) / 2 * FPS)))
        fr = frames[f]
        hgt = sim.heights_after(sc, t0 + 1e-6)
        k = sim.quarter(a)
        rh = sim.rotate_heights(hgt, k)
        pts = _escape_points(rh, sc["elev"])
        vis_world = {_unrot(sc, k, c) for c in pts}
        indep = answers._vis_world(sc, hgt, a)
        if vis_world != indep:
            errs.append(f"hold {hi}: dense band search and answers.py disagree on visibility")
        for cell, (face, pt, frac) in pts.items():
            wc = _unrot(sc, k, cell)
            base = np.array(_colour_of(sc, wc, mk), dtype=float)
            want = base * (1.0 if face == "top" else render.shade((0, -1, 0), (0.0, -1.0)))
            x, y = _rot_point(sc, k, *pt)
            patch = fr[int(y) - 1:int(y) + 2, int(x) - 1:int(x) + 2].reshape(-1, 3)
            n["visible_cube_samples"] += 1
            if np.abs(patch.mean(axis=0) - want).sum() > 45:
                errs.append(f"hold {hi}: cube {wc} should show its {face} face at ({x:.0f}, {y:.0f})")
        lifted = {(e["x"], e["y"], e["z"]) for e in sc["events"] if e["kind"] == "lift" and e["t1"] <= t0 + 1e-6}
        for m in sc["marked"]:
            cell = (m["x"], m["y"], m["z"])
            if cell in lifted:
                continue
            col = sim.COLOURS[m["colour"]]
            corners = [sim.project(sc, a, (m["x"] + i, m["y"] + j, m["z"] + kk))[:2]
                       for i in (0, 1) for j in (0, 1) for kk in (0, 1)]
            bx0, by0 = int(min(p[0] for p in corners)), int(min(p[1] for p in corners))
            bx1, by1 = int(max(p[0] for p in corners)) + 1, int(max(p[1] for p in corners)) + 1
            cnt = int(colour_mask(fr[by0:by1, bx0:bx1].astype(float), col, _rivals(sc, col)).sum())
            want = cell in indep
            n["marked_blob_reads"] += 1
            if want and cnt < 0.5 * sim.vis_thr(sc) or not want and cnt > 40:
                errs.append(f"hold {hi}: {m['colour']} cube {'visible' if want else 'hidden'} by ray casting, "
                            f"{cnt} px of its colour in the frame")
        ok, mae = timer_ok(fr, f / FPS)
        n["timer_reads"] += 1
        if not ok:
            errs.append(f"timer at frame {f} doesn't read {f / FPS:.1f} s (mae {mae:.1f})")
    # 3. drops / lifts: where the scene puts the moving cube, the frame shows that cube's colour
    for i, ev in enumerate(sc["events"]):
        seen = 0
        for t in sim.event_samples(ev):
            f = int(round(t * FPS))
            cubes = sim.cubes_at(sc, f / FPS)
            idx = [j for j, c in enumerate(cubes) if c[5]]
            n["event_frames"] += 1
            if not idx:
                continue
            ids, (ox, oy) = sim.id_image(sc, sim.angle(sc, f / FPS), cubes)
            own = ids == idx[0] + 1
            if own.sum() < 0.5 * sim.vis_thr(sc):
                continue
            import cv2
            own = cv2.erode(own.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            # expected colour of each of the moving cube's faces (shaded as the renderer shades a face)
            exp = Image.new("RGB", (ids.shape[1], ids.shape[0]), (0, 0, 0))
            dr = ImageDraw.Draw(exp)
            for k, nn, nr, poly in sim.faces(sc, sim.angle(sc, f / FPS), cubes):
                if k == idx[0]:
                    dr.polygon([(px_ - ox, py_ - oy) for px_, py_ in poly], fill=render.face_rgb(sc, cubes[k][4], nn, nr))
            ys, xs = np.nonzero(own)
            got = frames[f][ys + oy, xs + ox].astype(float)
            want = np.asarray(exp, dtype=float)[ys, xs]
            frac = (np.abs(got - want).sum(axis=1) < 36).mean() if len(ys) else 0.0
            seen += frac > 0.6
        n["event_reads"] += 1
        if seen < 3:
            errs.append(f"event {i}: moving cube shows in only {seen}/5 sampled frames")
    for name, t in sim.key_times(sc):
        if name == "stop_len":
            continue
        for f in (int(math.floor(t * FPS)), int(math.ceil(t * FPS))):
            ok, mae = timer_ok(frames[min(last, f)], min(last, f) / FPS)
            n["timer_reads"] += 1
            if not ok:
                errs.append(f"timer at frame {f} doesn't read {f / FPS:.1f} s (mae {mae:.1f})")
    # 4. nothing drawn in the clock's surroundings (only the clock itself)
    x0, y0, x1, y1 = sim.TIMER_BOX
    bg = np.asarray(render._background(json.dumps(sc["theme"], sort_keys=True)), dtype=np.int16)
    for f in range(0, nfr, 24):
        ring = np.abs(frames[f][max(0, y0 - 10):y1 + 10, x0 - 10:x1 + 10] - bg[max(0, y0 - 10):y1 + 10, x0 - 10:x1 + 10])
        mask = np.ones(ring.shape[:2], bool)
        oy = y0 - max(0, y0 - 10)
        mask[max(0, oy - 2):oy + (y1 - y0) + 3, 8:10 + (x1 - x0) + 3] = False
        n["clock_clear_reads"] += 1
        if ring.sum(axis=2)[mask].mean() > 12:
            errs.append(f"something drawn next to the clock at frame {f}")
    return errs, n


def framecheck(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    a = ap.parse_args(argv)
    root = Path(a.dataset_dir)
    tot, bad = Counter(), []
    for line in open(root / "annotations.jsonl", encoding="utf8"):
        s = json.loads(line)
        errs, n = framecheck_sample(s, root / s["media_path"])
        tot.update(n)
        bad += [f"{s['sample_id']}: {e}" for e in errs]
    print("frame check:", dict(tot), f"errors: {len(bad)}")
    for e in bad[:40]:
        print("  -", e)
    return len(bad)


def framehash(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    s = _roundtrip(build_sample(DATASET, sim, questions, a.seed))
    h = hashlib.sha256()
    n = int(round(render.duration(s["params"]) * FPS))
    for i in range(n):
        h.update(np.asarray(render.frame(s["params"], i / FPS).convert("RGB"), dtype=np.uint8).tobytes())
    print(h.hexdigest())
    return h.hexdigest()


COMMANDS = {"stats": stats, "tamper": tamper, "framecheck": framecheck, "framehash": framehash}
