"""Render the preview set: the first PER seeds of every scenario (seed order), 1280x720 @ 24 fps.

    python preview.py [--per 10] [--output-dir ../output_videos/object_tracking_mot_v2_preview]

Writes annotations.jsonl in the same format as `run.py generate`, so `run.py validate` and
`frame_check.py` run on it unchanged (validation rebuilds each clip from its stored seed).
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import video5 as V  # noqa: E402
import sim, questions, render  # noqa: E402

DS = "object_tracking_mot_v2"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=10)
    ap.add_argument("--total", type=int, default=50, help="clips in all, spread evenly over the scenarios")
    ap.add_argument("--output-dir", default=str(HERE.parent / "output_videos" / f"{DS}_preview"))
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir).resolve()
    (out / "videos").mkdir(parents=True, exist_ok=True)
    k = len(sim.SCENARIOS)
    want = {sc: max(args.per, args.total // k + (1 if i < args.total % k else 0)) for i, sc in enumerate(sim.SCENARIOS)}
    picked, seed = {sc: [] for sc in sim.SCENARIOS}, 0
    while any(len(picked[sc]) < want[sc] for sc in picked):
        sc = sim.build(random.Random(seed))["scenario"]
        if len(picked[sc]) < want[sc]:
            picked[sc].append(seed)
        seed += 1
    seeds = sorted(s for v in picked.values() for s in v)
    with open(out / "annotations.jsonl", "w", encoding="utf8") as f:
        for i, seed in enumerate(seeds):
            s = V._roundtrip(V.build_sample(DS, sim, questions, seed))
            s["media_path"] = f"videos/{s['sample_id']}.mp4"
            s["media_meta"] = {"fps": V.FPS, "resolution": list(V.SIZE),
                               "duration_s": round(render.duration(s["params"]), 3)}
            if not args.no_render:
                s["render_meta"] = V.encode(render, s["params"], out / s["media_path"])
            f.write(json.dumps(s) + "\n")
            print(f"[{i + 1}/{len(seeds)}] {s['sample_id']} {s['params']['scenario']}", flush=True)
