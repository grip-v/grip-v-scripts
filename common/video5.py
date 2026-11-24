"""Shared generate/validate pipeline for the PIL-rendered 5-rung datasets.

A dataset supplies four small modules and hands them to `run_generate` /
`run_validate`:

    sim.build(rng) -> scene            JSON-serializable dict, pure function of the rng
    questions.build(scene, rng) -> [5 question dicts]   (common/ladder5.py;
                                   10 if questions.LADDER == "ten")
    answers.answer(key, params, scene) -> value         independent re-derivation
    render.frame(scene, t) -> PIL.Image, render.duration(scene) -> seconds

Sample `i` uses random.Random(seed + i); the scene and the questions come
from the same rng stream in sequence, so the validator can rebuild both
from the seed and diff them against the stored annotation.
"""
from __future__ import annotations
import argparse
import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import ladder5 as L5
from .fonts import resolve_font_path

FPS = 24
SIZE = (1280, 720)
_FONT_CACHE: dict = {}


def font(size: int):
    from PIL import ImageFont
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(resolve_font_path(), size)
    return _FONT_CACHE[size]


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def encode(render_mod, scene: dict, out_path: Path) -> dict:
    """Render every frame and pipe raw RGB straight into ffmpeg (H.264)."""
    duration = render_mod.duration(scene)
    n = max(1, int(round(duration * FPS)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out_path)]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        img = render_mod.frame(scene, i / FPS).convert("RGB")
        if img.size != SIZE:
            img = img.resize(SIZE)
        proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")
    return {"renderer": "pil+ffmpeg", "n_frames": n, "render_time_s": round(time.time() - t0, 2),
            "bytes": out_path.stat().st_size}


def build_sample(dataset: str, sim, questions, seed: int) -> dict:
    rng = random.Random(seed)
    scene = sim.build(rng)
    qs = questions.build(scene, rng)
    return {"sample_id": f"{dataset}_{seed:06d}", "dataset": dataset, "modality": "video",
            "seed": seed, "params": scene, "questions": qs}


def _roundtrip(x):
    return json.loads(json.dumps(x))


def run_generate(dataset: str, here: Path, sim, questions, render_mod, argv=None):
    ap = argparse.ArgumentParser(description=f"generate {dataset}")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default=str(here.parent / "output_videos" / dataset))
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.output_dir).resolve()
    (out / "videos").mkdir(parents=True, exist_ok=True)
    with open(out / "annotations.jsonl", "w", encoding="utf8") as f:
        for i in range(args.n):
            seed = args.seed + i
            s = _roundtrip(build_sample(dataset, sim, questions, seed))
            s["media_path"] = f"videos/{s['sample_id']}.mp4"
            s["media_meta"] = {"fps": FPS, "resolution": list(SIZE),
                               "duration_s": round(render_mod.duration(s["params"]), 3)}
            if not args.no_render:
                s["render_meta"] = encode(render_mod, s["params"], out / s["media_path"])
            f.write(json.dumps(s) + "\n")
            print(f"[{i + 1}/{args.n}] {s['sample_id']} ({s['media_meta']['duration_s']}s)", flush=True)
    print(f"done -> {out / 'annotations.jsonl'}")


def _video_errors(path: Path, meta: dict) -> list[str]:
    try:
        import cv2
    except ImportError:
        return []
    if not path.exists():
        return [f"missing video {path.name}"]
    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, moving, prev = 0, False, None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames += 1
        if prev is not None and not moving and np.abs(fr.astype(int) - prev).mean() > 0.05:
            moving = True
        prev = fr.astype(int)
    cap.release()
    errs = []
    if [w, h] != meta["resolution"]:
        errs.append(f"resolution {w}x{h} != {meta['resolution']}")
    if abs(frames / meta["fps"] - meta["duration_s"]) > 0.2:
        errs.append(f"duration {frames / meta['fps']:.2f}s != planned {meta['duration_s']}s")
    if not moving:
        errs.append("video never changes (no motion)")
    return errs


def run_validate(dataset: str, here: Path, sim, questions, answers, render_mod, argv=None):
    ap = argparse.ArgumentParser(description=f"validate {dataset}")
    ap.add_argument("--dataset-dir", default=str(here.parent / "output_videos" / dataset))
    ap.add_argument("--ann", default="annotations.jsonl")
    ap.add_argument("--report", default="validation_report.txt")
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.dataset_dir).resolve()
    errors, n_s, n_q = [], 0, 0
    kinds = Counter()
    with open(root / args.ann, encoding="utf8") as ann:
        lines = ann.readlines()
    for line in lines:
        s = json.loads(line)
        n_s += 1
        sid, scene = s["sample_id"], s["params"]
        rebuilt = _roundtrip(build_sample(dataset, sim, questions, s["seed"]))
        if rebuilt["params"] != scene:
            errors.append(f"{sid}: rebuilding from the seed gives a different scene")
        if rebuilt["questions"] != s["questions"]:
            errors.append(f"{sid}: rebuilding from the seed gives different questions")
        for e in getattr(sim, "check", lambda sc: [])(scene):
            errors.append(f"{sid}: {e}")
        errors.extend(f"{sid}: {e}" for e in L5.ladder_errors(s["questions"], getattr(questions, "LADDER", "five")))
        for q in s["questions"]:
            n_q += 1
            kinds[(q["qid"], q["derivation_key"])] += 1
            indep = answers.answer(q["derivation_key"], q["params"], scene)
            if indep is None or str(indep) != str(q["groundtruth"]):
                errors.append(f"{sid}/{q['qid']} ({q['derivation_key']}): independent answer "
                              f"{indep!r} != stored {q['groundtruth']!r}")
            if str(q["groundtruth"]) != q["answer_text"]:
                errors.append(f"{sid}/{q['qid']}: groundtruth {q['groundtruth']!r} isn't the answer option")
        if abs(render_mod.duration(scene) - s["media_meta"]["duration_s"]) > 1e-3:
            errors.append(f"{sid}: stored duration doesn't match the renderer's")
        if not args.skip_video:
            errors.extend(f"{sid}: {e}" for e in _video_errors(root / s["media_path"], s["media_meta"]))
    lines = [f"{dataset} validation report", "=" * 40, f"samples: {n_s}", f"questions: {n_q}",
             "question variants:"] + [f"  {k[0]} {k[1]}: {v}" for k, v in sorted(kinds.items())]
    lines += [f"errors: {len(errors)}"] + [f"  - {e}" for e in errors[:300]]
    lines += ["", f"RESULT: {n_s} samples, {n_q} questions, {len(errors)} mismatches"]
    (root / args.report).write_text("\n".join(lines) + "\n", encoding="utf8")
    print(f"validation done: {n_s} samples, {n_q} questions, {len(errors)} mismatches -> {args.report}")
    if errors:
        for e in errors[:15]:
            print("  -", e)
        sys.exit(1)


def dataset_cli(dataset: str, here: Path, sim, questions, answers, render_mod):
    """`python run.py generate ...` / `python run.py validate ...`"""
    if len(sys.argv) < 2 or sys.argv[1] not in ("generate", "validate"):
        print("usage: run.py {generate|validate} [options]")
        sys.exit(2)
    if sys.argv[1] == "generate":
        run_generate(dataset, here, sim, questions, render_mod, sys.argv[2:])
    else:
        run_validate(dataset, here, sim, questions, answers, render_mod, sys.argv[2:])
