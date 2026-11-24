"""Independent check of the number rule (see CHANGES.md).

Rule: the die always stands on a cell whose number equals its top face. It
rolls into the one side neighbour (up/down/left/right) where rolling that way
brings that cell's number to the top. If no direction works, the die stops
(that is the end of the path). On the one broken roll (if any) the die shows
a wrong face; that cell's number is the physically correct top.

Uses only the saved grid, path and shown tops -- nothing from puzzle.py's
path planner -- so a planner bug cannot hide itself.

usage:
    python verify_number_rule.py                     # 3000 seeds, built in memory
    python verify_number_rule.py --seeds 500
    python verify_number_rule.py --dataset <dir>     # check a generated annotations.jsonl
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STEP = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}


def check(p):
    """p: a sample's params dict. Returns a list of problems (empty = rule holds)."""
    grid, N = p["grid_values"], p["N"]
    cells = [tuple(c) for c in p["path_cells"]]
    seq = p["displayed_roll_sequence"]

    # which slot's value comes to the top when rolling each way (written out
    # from the rolling definition, independent of die.py)
    comes_up = {"north": "south", "south": "north", "east": "west", "west": "east"}

    def working(cell, o, came_from):
        r, c = cell
        return [(d, (r + dr, c + dc)) for d, (dr, dc) in STEP.items()
                if 0 <= r + dr < N and 0 <= c + dc < N and grid[r + dr][c + dc] == o[comes_up[d]]
                and (r + dr, c + dc) != came_from]   # never straight back

    errs = []
    brk = p["broken_step_index"]
    orients = [p["initial_orientation"]] + [s["resulting_orientation"] for s in seq]
    if grid[cells[0][0]][cells[0][1]] != orients[0]["top"]:
        errs.append("start cell number is not the start top")
    for i, st in enumerate(seq):
        m = working(cells[i], orients[i], cells[i - 1] if i else None)
        if len(m) != 1:
            errs.append(f"roll {i + 1}: {len(m)} directions bring up their cell number (need exactly 1)")
        elif m[0] != (st["direction"], cells[i + 1]):
            errs.append(f"roll {i + 1}: went {st['direction']}, rule says {m[0][0]}")
        shown = st["resulting_orientation"]["top"]
        number = grid[cells[i + 1][0]][cells[i + 1][1]]
        if i == brk:
            if number != p["correct_orientation_at_break"]["top"]:
                errs.append(f"broken roll {i + 1}: cell number is not the physically correct top")
        elif shown != number:
            errs.append(f"roll {i + 1}: die shows {shown} on a cell numbered {number}")
    if working(cells[-1], orients[-1], cells[-2]):
        errs.append("end: a direction still works, so the die should not have stopped")
    states = [(cells[0], tuple(sorted(p["initial_orientation"].items())))]
    states += [(cells[i + 1], tuple(sorted(s["resulting_orientation"].items()))) for i, s in enumerate(seq)]
    if len(states) != len(set(states)):
        errs.append("loop: the die returns to a cell with the same orientation (repeats the same moves)")
    return errs


ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3000)
ap.add_argument("--dataset")
args = ap.parse_args()

if args.dataset:
    samples = [json.loads(l)["params"] | {"_id": json.loads(l)["sample_id"]}
               for l in open(Path(args.dataset) / "annotations.jsonl", encoding="utf8")]
else:
    import puzzle as P
    from dataclasses import asdict
    samples, t0 = [], time.time()
    for seed in range(args.seeds):
        sc = asdict(P.build_scene(random.Random(seed)))
        sc["_id"] = f"seed {seed}"
        samples.append(sc)
    print(f"built {args.seeds} scenes in {time.time() - t0:.1f}s")

bad = 0
for p in samples:
    errs = check(p)
    if errs:
        bad += 1
        if bad <= 10:
            print("  FAIL", p["_id"], errs[:3])
rolls = [len(p["displayed_roll_sequence"]) for p in samples]
broken = sum(1 for p in samples if p["broken_step_index"] is not None)
print(f"samples: {len(samples)}; rolls min/mean/max: {min(rolls)}/{sum(rolls) / len(rolls):.1f}/{max(rolls)}; "
      f"broken_rolls: {broken} ({broken / len(samples):.0%})")
print(f"RESULT: {'PASS' if not bad else f'{bad} FAILING SAMPLES'}")
sys.exit(1 if bad else 0)
