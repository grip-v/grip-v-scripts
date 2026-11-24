"""Shared minimal render helper for concept_demos/ -- PIL + ffmpeg, matching
the rest of this repo's non-Manim rendering convention. Not the full
SHARED_CONTRACT pipeline (no annotations.jsonl/validator yet) -- these are
quick visual proofs-of-concept to look at before investing in a real
dataset build."""
from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def render_clip(render_frame_fn, duration_s: float, fps: int, out_path: Path):
    n_frames = max(1, int(round(duration_s * fps)))
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        frames = []
        for i in range(n_frames):
            t = i / fps
            img = render_frame_fn(t)
            img.save(tdp / f"frame_{i:05d}.png")
            frames.append(i)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(tdp / "frame_%05d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return n_frames


def save_key_frames(render_frame_fn, times, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in times:
        img = render_frame_fn(t)
        p = out_dir / f"{prefix}_t{t:.2f}.png"
        img.save(p)
        paths.append(p)
    return paths
