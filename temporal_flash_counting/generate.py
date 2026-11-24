"""Kept for generate_all_datasets.sbatch: same as `python run.py generate`.

    python generate.py --n 150 --output-dir <dir> [--seed 0]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common.video5 import run_generate  # noqa: E402
import sim, questions, render  # noqa: E402

if __name__ == "__main__":
    run_generate("temporal_flash_counting", HERE, sim, questions, render)
