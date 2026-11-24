"""Scene generation for mirror_beam_reflection_v2: a beam crosses a grid of diagonal mirrors.
Tiers (easy/medium/hard) set grid size and mirror count; the beam may enter at any non-corner
edge cell. The clip shows a long prefix of the true path (>= 4 bounces, <= ~75% of the path)."""
from __future__ import annotations
import random

SEGMENT_S = 0.3
HOLD_S = 1.2
MAX_STEPS = 400
TIERS = ("easy", "medium", "hard")
GRID = {"easy": 6, "medium": 7, "hard": 8}
N_MIRRORS = {"easy": (8, 10), "medium": (10, 13), "hard": (13, 16)}
MIN_PATH = {"easy": 42, "medium": 50, "hard": 58}     # preferred full-path length (cells), relaxed on retries
MIN_SHOWN_BOUNCES = 4
MAX_FRAC = 0.75
EDGES = ["top", "bottom", "left", "right"]


def entries(n):
    """Every non-corner edge cell with its inward direction."""
    out = []
    for i in range(1, n - 1):
        out += [((i, 0), (0, 1)), ((i, n - 1), (0, -1)), ((0, i), (1, 0)), ((n - 1, i), (-1, 0))]
    return out


def trace(mirrors: dict, start, d, n):
    """-> (path of cells, bounce cells, exit edge) or None if the beam loops."""
    path, bounces, seen = [tuple(start)], [], set()
    cell, (dr, dc) = tuple(start), tuple(d)
    for _ in range(MAX_STEPS):
        if (cell, dr, dc) in seen:
            return None
        seen.add((cell, dr, dc))
        r, c = cell[0] + dr, cell[1] + dc
        if not (0 <= r < n and 0 <= c < n):
            edge = "top" if r < 0 else "bottom" if r >= n else "left" if c < 0 else "right"
            return path, bounces, edge
        cell = (r, c)
        path.append(cell)
        m = mirrors.get(f"{r}-{c}")
        if m == "/":
            dr, dc = -dc, -dr
            bounces.append(cell)
        elif m == "\\":
            dr, dc = dc, dr
            bounces.append(cell)
    return None


def show_candidates(path, bounces, mirrors):
    """Indices j where the clip may stop: empty cell, >=4 bounces behind, >=1 ahead, <=75% of path."""
    out = []
    for j in range(1, len(path) - 1):
        if f"{path[j][0]}-{path[j][1]}" in mirrors or j > MAX_FRAC * (len(path) - 1):
            continue
        behind = _bounces_upto(path, mirrors, j)
        ahead = _bounces_upto(path, mirrors, len(path) - 1) - behind
        if behind >= MIN_SHOWN_BOUNCES and ahead >= 1:
            out.append(j)
    return out


def _bounces_upto(path, mirrors, j):
    return sum(1 for cell in path[1:j + 1] if f"{cell[0]}-{cell[1]}" in mirrors)


def build(rng: random.Random) -> dict:
    tier = TIERS[rng.randrange(3)]
    n = GRID[tier]
    ent = entries(n)
    best = None
    for attempt in range(600):
        start, d = rng.choice(ent)
        n_m = rng.randint(*N_MIRRORS[tier])
        cells = [(r, c) for r in range(n) for c in range(n) if (r, c) != start]
        rng.shuffle(cells)
        mirrors = {f"{r}-{c}": rng.choice(["/", "\\"]) for r, c in cells[:n_m]}
        out = trace(mirrors, start, d, n)
        if not out:
            continue
        path, bounces, _ = out
        cands = show_candidates(path, bounces, mirrors)
        if not cands:
            continue
        goal = MIN_PATH[tier] - attempt // 150 * 3          # relax slowly if hard to satisfy
        if len(path) >= goal:
            best = (start, d, mirrors, cands, len(path))
            break
        if best is None or len(path) > best[4]:
            best = (start, d, mirrors, cands, len(path))
    if best is None:
        return build(rng)
    start, d, mirrors, cands, _ = best
    top = sorted(cands)[-max(1, len(cands) // 4):]          # long prefixes: the last third of the candidates
    show = rng.choice(top)
    return {"tier": tier, "grid_n": n, "start": list(start), "dir": list(d), "mirrors": mirrors,
            "truncate_index": show, "duration": round((show + 1) * SEGMENT_S + HOLD_S, 2)}


def check(scene: dict) -> list[str]:
    n = scene["grid_n"]
    out = trace(scene["mirrors"], scene["start"], scene["dir"], n)
    if out is None:
        return ["beam loops"]
    path, bounces, _ = out
    m, k = scene["mirrors"], scene["truncate_index"]
    errs = []
    if scene.get("tier") not in TIERS or GRID[scene["tier"]] != n:
        errs.append("tier/grid mismatch")
    if not (0 < k < len(path) - 1):
        return errs + ["bad truncate_index"]
    if f"{path[k][0]}-{path[k][1]}" in m:
        errs.append("clip stops on a mirror")
    if _bounces_upto(path, m, k) < MIN_SHOWN_BOUNCES:
        errs.append("fewer than 4 bounces shown")
    if _bounces_upto(path, m, len(path) - 1) <= _bounces_upto(path, m, k):
        errs.append("no bounce in the unseen remainder")
    if k > MAX_FRAC * (len(path) - 1) + 1e-9:
        errs.append("too much of the path shown")
    return errs
