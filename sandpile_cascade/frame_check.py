"""Frame-level check of rendered clips against the scene data.

    python frame_check.py --dataset-dir <dir> [--report frame_check.txt] [--jobs 4]

For every clip it decodes the MP4 (OpenCV) and checks:
  * clock: template-matches the timer panel against every reading 0.0-9.9 s
    on every 6th frame, on every event frame and on every frame a question
    names by its clock reading;
  * no other text: the clip's label list is empty;
  * grains at rest, read from the pixels: blobs of the grain colour whose
    area is that of a resting grain (+-25 %, calibrated on the first decoded
    frame) and whose centre sits on a slot (3 px);
    on every frame, every cell that no moving grain overlaps holds exactly
    as many resting grains as the track says;
  * events, frame-accurate: on a landing frame the grain is found resting
    on its slot and one frame earlier it is not; on a topple frame the
    grains that leave were found resting one frame earlier and are gone
    from their slots on that frame; on an arrival frame the grain is found
    resting and one frame earlier it is not;
  * every topple read off the frames: for each cell and frame, the top
    grains vanish from their slots on exactly the topple frames;
  * answers read off the frames: grain counts at the frames the questions
    use (start, end, a named clock reading, after a cascade) and the clock
    on the event frame of every clock question;
  * the grain size and the outline width meet the size rules.
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
import questions  # noqa: E402
from common.video5 import font  # noqa: E402

COL_TOL = 70          # RGB distance for "the grain colour"
POS_TOL = 3.0         # px between a blob centre and a slot centre


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
    patch = fr[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
    T = _timer_templates()
    return min(T, key=lambda k: np.abs(T[k] - patch).mean())


def rest_area(scene) -> float:
    """Pixels of the grain colour in one resting grain, as drawn."""
    st = scene["style"]
    tr = render.track(scene)
    sp = render.grain_sprite(tuple(tr["grain"]["rgb"]), st["shape"], round(tr["grain"]["size"], 2), st["outline"],
                             render.edge(st))
    a = np.asarray(sp.reduce(render.SS).convert("RGBA"), dtype=np.int16)
    rgb = np.array(tr["grain"]["rgb"])
    return float(((np.abs(a[:, :, :3] - rgb).sum(axis=2) < COL_TOL) & (a[:, :, 3] > 200)).sum())


class Reader:
    """Resting grains read from the pixels."""

    def __init__(self, scene):
        import cv2
        self.cv2 = cv2
        self.sc = scene
        self.tr = render.track(scene)
        self.rgb = np.array(self.tr["grain"]["rgb"], dtype=np.int16)
        self.area = rest_area(scene)
        self.tol = 0.2
        self.slots = [(c, k, s[0], s[1]) for c, cell in enumerate(self.tr["cells"]) for k, s in enumerate(cell["slots"])]
        self.sx = np.array([s[2] for s in self.slots])
        self.sy = np.array([s[3] for s in self.slots])

    def calibrate(self, fr):
        """Area of a resting grain as decoded (the codec softens edges): median over the first frame."""
        got = self.blobs(fr)
        if got:
            self.area = float(np.median([a for a, _ in got]))
            self.tol = 0.25

    def blobs(self, fr):
        mask = (np.abs(fr - self.rgb).sum(axis=2) < COL_TOL).astype(np.uint8)
        n, lab, stats, cent = self.cv2.connectedComponentsWithStats(mask, connectivity=4)
        out = []
        for i in range(1, n):
            a = stats[i, self.cv2.CC_STAT_AREA]
            if abs(a - self.area) > 0.3 * self.area:
                continue
            d = np.hypot(self.sx - cent[i][0], self.sy - cent[i][1])
            if d.min() <= POS_TOL:
                out.append((a, int(np.argmin(d))))
        return out

    def resting(self, fr):
        """{(cell, slot)} with a resting grain on this frame."""
        mask = (np.abs(fr - self.rgb).sum(axis=2) < COL_TOL).astype(np.uint8)
        n, lab, stats, cent = self.cv2.connectedComponentsWithStats(mask, connectivity=4)
        got = set()
        for i in range(1, n):
            a = stats[i, self.cv2.CC_STAT_AREA]
            if abs(a - self.area) > self.tol * self.area:
                continue
            x, y = cent[i]
            d = np.hypot(self.sx - x, self.sy - y)
            j = int(np.argmin(d))
            if d[j] <= POS_TOL:
                got.add(self.slots[j][:2])
        return got


def movers(tr, f):
    """Boxes of the grains not resting on a slot on frame f."""
    P = tr["poses"][f]
    g = tr["grain"]["size"]
    slots = np.array([s for c in tr["cells"] for s in c["slots"]])
    out = []
    for x, y, s in P:
        if x != x:
            continue
        if abs(s - 1) < 1e-9 and np.min(np.abs(slots[:, 0] - x) + np.abs(slots[:, 1] - y)) < 0.05:
            continue
        r = g * s / 2 + 3
        out.append((x - r, y - r, x + r, y + r))
    return out


def cell_box(tr, c):
    cell = tr["cells"][c]
    g = tr["grain"]["size"]
    if not cell["slots"]:
        return None
    xs = [p[0] for p in cell["slots"]]
    ys = [p[1] for p in cell["slots"]]
    return min(xs) - g / 2 - 2, min(ys) - g / 2 - 2, max(xs) + g / 2 + 2, max(ys) + g / 2 + 2


def _hit(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def check_clip(scene, qs, frames) -> list[str]:
    errs = []
    n = scene["n_frames"]
    tr = render.track(scene)
    if len(frames) != n:
        return [f"decoded {len(frames)} frames, expected {n}"]
    if scene["labels"] or tr["labels"]:
        errs.append("labels present")
    if tr["grain"]["size"] < 20 or scene["style"]["outline"] < 3:
        errs.append("grain under 20 px or outline under 3 px")
    tl = sim.timeline(scene)
    ev = tl["events"]
    C = sim.counts(scene)
    R = Reader(scene)
    R.calibrate(frames[0])
    rest = [R.resting(fr) for fr in frames]
    occ = np.zeros_like(C)
    for f, got in enumerate(rest):
        for c, _ in got:
            occ[f, c] += 1
    # clock
    fs = set(range(0, n, 6)) | {e["f"] for e in ev}
    for q in qs:
        for k in ("t", "t1", "t2"):
            if k in q["params"]:
                fs |= set(range(*[sim.frames_showing(q["params"][k])[0], sim.frames_showing(q["params"][k])[1] + 1]))
    for f in sorted(fs):
        if f < n and read_timer(frames[f]) != sim.tenths(f):
            errs.append(f"clock wrong on frame {f}")
            break
    # counts on every frame where no moving grain overlaps the cell
    boxes = [cell_box(tr, c) for c in range(len(tr["cells"]))]
    bad = 0
    for f in range(n):
        mv = movers(tr, f)
        for c, b in enumerate(boxes):
            if b is None or any(_hit(b, m) for m in mv):
                continue
            if occ[f, c] != C[f, c]:
                bad += 1
                if bad <= 3:
                    errs.append(f"count: frame {f} cell {c} reads {occ[f, c]} grains, track {C[f, c]}")
    # events, frame-accurate
    T, M = scene["T"], scene["timing"]["M"]
    stacks = [list(range(h)) for h in scene["init"]]          # slot occupancy (indices only)
    hs = list(scene["init"])
    kinds, out = sim.topology(scene)
    chg = []
    for e in ev:
        if e["kind"] == "land":
            chg.append((e["f"], 0, "land", e["cell"]))
        else:
            chg.append((e["f"], 1, "top", e["cell"], e["to"]))
    chg.sort(key=lambda x: (x[0], x[1]))
    arrivals = {}
    for x in chg:
        f = x[0]
        for af in sorted(k for k in list(arrivals) if k <= f):
            for c in arrivals.pop(af):
                hs[c] += 1
        if x[2] == "land":
            k = hs[x[3]]
            hs[x[3]] += 1
            if (x[3], k) not in rest[f] or (x[3], k) in rest[f - 1]:
                errs.append(f"landing on frame {f} not seen there (cell {x[3]} slot {k})")
        else:
            c = x[3]
            top = [(c, k) for k in range(hs[c] - T, hs[c])]
            if not all(s in rest[f - 1] for s in top) or any(s in rest[f] for s in top):
                errs.append(f"topple of cell {c} on frame {f} not seen there")
            hs[c] -= T
            for t in x[4]:
                if t >= 0 and kinds[t] == "cell":
                    arrivals.setdefault(f + M, []).append(t)
    # arrivals: the slot filled on the arrival frame, not one frame earlier
    for e in ev:
        if e["kind"] != "topple":
            continue
        f = e["f"] + M
        for t in set(x for x in e["to"] if x >= 0 and kinds[x] == "cell"):
            k = int(C[f, t]) - 1
            if (t, k) not in rest[f] or (t, k) in rest[f - 1]:
                errs.append(f"arrival in cell {t} on frame {f} not seen there")
    # every topple read off the frames: the top T grains of a cell vanish from their slots
    seen = set()
    for f in range(1, n):
        for c in range(len(tr["cells"])):
            h0 = C[f - 1, c]
            if h0 < T:
                continue
            top = [(c, k) for k in range(h0 - T, h0)]
            if all(s in rest[f - 1] for s in top) and not any(s in rest[f] for s in top):
                seen.add((f, c))
    want = {(e["f"], e["cell"]) for e in ev if e["kind"] == "topple"}
    if seen != want:
        errs.append(f"topples read off the frames {sorted(seen ^ want)[:4]} differ from the track")
    # answers read off the frames
    for q in qs:
        k, p, gt = q["derivation_key"], q["params"], q["groundtruth"]
        cell = None
        if "cell" in p:
            cell = next(c for c in range(len(tr["cells"])) if tr["cells"][c]["kind"] == "cell"
                        and sim.cell_desc(scene, c) == p["cell"])
        val = None
        if k == "start_count":
            val = occ[0, cell]
        elif k == "end_count":
            val = occ[-1, cell]
        elif k == "start_total":
            val = occ[0].sum()
        elif k == "end_total":
            val = occ[-1].sum()
        elif k == "count_at":
            a, b = sim.frames_showing(p["t"])
            vs = {int(occ[f, cell]) for f in range(a, b + 1)}
            val = vs.pop() if len(vs) == 1 else None
        elif k == "count_after":
            i = p["i"]
            lands = [e for e in ev if e["kind"] == "land"]
            g0 = sum(scene["init"])
            if i + 1 < len(lands):
                f = int(np.nonzero(~np.isnan(tr["poses"][:, g0 + i + 1, 0]))[0][0]) - 1
            else:
                f = n - 1
            val = occ[f, cell]
        elif k in ("topple_time", "land_time"):
            if k == "land_time":
                f = [e for e in ev if e["kind"] == "land"][p["i"]]["f"]
            else:
                f = min(e["f"] for e in ev if e["kind"] == "topple" and e["cell"] == cell)
            val = sim.clock_text(read_timer(frames[f]))
        else:
            continue
        if val is None or str(int(val) if not isinstance(val, str) else val) != str(gt):
            errs.append(f"{q['qid']} {k}: read {val} off the frames, answer {gt}")
    return errs


def _one(arg):
    root, line = arg
    s = json.loads(line)
    frames = frames_of(root / s["media_path"])
    try:
        errs = check_clip(s["params"], s["questions"], frames)
    except Exception as ex:  # noqa: BLE001
        errs = [f"exception {ex!r}"]
    return s["sample_id"], s["params"]["scenario"], errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--report", default="frame_check.txt")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.dataset_dir).resolve()
    lines = (root / "annotations.jsonl").read_text().splitlines()
    from multiprocessing import Pool
    with Pool(args.jobs) as pool:
        res = pool.map(_one, [(root, ln) for ln in lines])
    out = []
    n_bad = 0
    for sid, scen, errs in res:
        out.append(f"{sid} [{scen}]: {'ok' if not errs else '; '.join(errs[:6])}")
        n_bad += bool(errs)
    out.append(f"RESULT: {len(res)} clips, {n_bad} with frame-check failures")
    (root / args.report).write_text("\n".join(out) + "\n")
    print("\n".join(out[-12:]))
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
