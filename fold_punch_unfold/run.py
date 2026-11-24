"""python run.py generate --n 20 --seed 0   |   python run.py validate --dataset-dir <dir>"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common.video5 import dataset_cli  # noqa: E402
import sim, questions, answers, render  # noqa: E402,F401

if __name__ == "__main__":
    dataset_cli("fold_punch_unfold", HERE, sim, questions, answers, render)
