"""python run.py generate --n 20 --seed 0 [--output-dir DIR] [--no-render]
python run.py validate --dataset-dir DIR [--skip-video]
python run.py stats --n 5000            (diversity / ladder / min_fps stats, no rendering)
python run.py tamper --dataset-dir DIR  (corrupt 3 answers, count flagged)
python run.py framecheck --dataset-dir DIR   (read rendered MP4s back, compare with the scene)
python run.py framehash --seed S        (hash of raw rendered frames, for determinism checks)"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common.video5 import dataset_cli  # noqa: E402
import sim, questions, answers, render  # noqa: E402,F401

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("stats", "tamper", "framecheck", "framehash"):
        import checks
        res = checks.COMMANDS[sys.argv[1]](sys.argv[2:])
        sys.exit(1 if sys.argv[1] == "framecheck" and res else 0)
    dataset_cli("gear_train", HERE, sim, questions, answers, render)
