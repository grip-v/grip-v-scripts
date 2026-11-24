"""Builds one die_roll_trace scene: grid, custom die identity, roll path,
and the single injected orientation error for broken_rolls videos.

Zero rendering imports. Everything here is pure Python simulation, built
entirely on top of `die.roll()` / `die.fold_net()` / `die.build_adjacency()`.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field

import config as C
import die as D


@dataclass
class SceneData:
    N: int
    canvas_px: int
    # CHANGE: grid values are no longer decorative. The die always stands on a
    # cell whose number equals its top face: it rolls into the one side
    # neighbour where rolling that way brings that cell's number to the top.
    # The values are built by _plan_number_path() so that rule gives the path.
    grid_values: list            # N x N cell values (1-9)
    grid_color_idx: list         # N x N decorative palette indices

    die_net_layout: list         # [{row,col,label,value}, ...] x6
    die_face_adjacency: dict     # {value: {opposite, four_neighbors}}
    initial_orientation: dict    # {top,bottom,north,south,east,west}

    start_cell: list             # [row, col]
    path_cells: list             # [[row,col], ...] length total_rolls+1

    true_roll_sequence: list
    displayed_roll_sequence: list

    content_mode: str            # "valid_rolls" | "broken_rolls"
    broken_step_index: object    # int | None
    correct_orientation_at_break: object   # dict | None
    displayed_wrong_orientation_at_break: object  # dict | None

    total_rolls: int
    total_cells_visited: int
    unique_cells_visited: int
    top_face_value_history: list



_DIR_DELTA = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}


def build_scene(rng: random.Random) -> SceneData:
    N = rng.randrange(C.N_RANGE[0], C.N_RANGE[1] + 1)
    canvas_px = rng.randrange(C.CANVAS_PX_RANGE[0], C.CANVAS_PX_RANGE[1] + 1)

    # CHANGE: grid_values are now built together with the path (see below).
    grid_color_idx = [[rng.randrange(len(C.CELL_COLOR_PALETTE)) for _ in range(N)] for _ in range(N)]

    # --- custom die identity from a randomized cross-net ---
    net_values = rng.sample(C.VALUE_POOL, 6)
    labels = ["U", "L", "F", "R", "B", "D"]
    values_by_label = dict(zip(labels, net_values))
    die_net_layout = [
        {"row": r, "col": c, "label": lbl, "value": values_by_label[lbl]}
        for lbl, (r, c) in D.NET_POSITIONS.items()
    ]
    initial_orientation = D.fold_net(values_by_label)
    die_face_adjacency = D.build_adjacency(initial_orientation)

    # CHANGE: the path is no longer a random walk. The die starts on a cell
    # whose number equals its top face, and each roll goes to the one side
    # neighbour where rolling that way brings that cell's number to the top,
    # so the die always shows the number of the cell it stands on. The grid is
    # built along with the path: exactly one direction works at every step,
    # and the last cell is a dead end (no direction works), where it stops.
    # On a broken roll the die lands showing a wrong face (it does not match
    # the cell); it then carries on from that wrong state.
    # Whether/where a roll is broken is decided first, because the wrong top
    # after a broken roll decides where the die goes next.
    path_len = rng.randrange(C.PATH_LEN_RANGE[0], C.PATH_LEN_RANGE[1] + 1)
    content_mode = "broken_rolls" if rng.random() < C.BROKEN_PROBABILITY else "valid_rolls"
    broken_step_index = rng.randrange(path_len) if content_mode == "broken_rolls" else None
    plan = None
    for _ in range(C.MAX_PATH_ATTEMPTS):
        plan = _plan_number_path(rng, N, initial_orientation, path_len, broken_step_index)
        if plan is not None:
            break
    if plan is None:
        raise RuntimeError("could not build a number-rule path; raise MAX_PATH_ATTEMPTS")
    grid_values, cells, directions, planned_wrong_orientation = plan
    start_row, start_col = cells[0]
    total_rolls = len(directions)

    # --- the fully correct sequence (validation-only for broken videos) ---
    true_seq = []
    o = dict(initial_orientation)
    for i, d in enumerate(directions):
        o = D.roll(o, d)
        true_seq.append({
            "step_index": i, "direction": d,
            "resulting_orientation": dict(o),
            "landed_cell": [cells[i + 1][0], cells[i + 1][1]],
        })

    # CHANGE: content_mode and broken_step_index are now chosen before the path.
    correct_orientation_at_break = None
    displayed_wrong_orientation_at_break = None
    displayed_seq = [
        {**s, "resulting_orientation": dict(s["resulting_orientation"]), "landed_cell": list(s["landed_cell"])}
        for s in true_seq
    ]

    if content_mode == "broken_rolls":
        # CHANGE: the wrong orientation is now picked inside _plan_number_path()
        # (same decoy-direction method as before), because it steers the path.
        correct_orientation = true_seq[broken_step_index]["resulting_orientation"]
        wrong_orientation = planned_wrong_orientation

        correct_orientation_at_break = dict(correct_orientation)
        displayed_wrong_orientation_at_break = dict(wrong_orientation)

        o2 = dict(wrong_orientation)
        displayed_seq[broken_step_index]["resulting_orientation"] = dict(o2)
        for i in range(broken_step_index + 1, total_rolls):
            o2 = D.roll(o2, directions[i])
            displayed_seq[i]["resulting_orientation"] = dict(o2)

    top_face_value_history = [s["resulting_orientation"]["top"] for s in displayed_seq]
    total_cells_visited = len(cells)
    unique_cells_visited = len(set(cells))

    return SceneData(
        N=N, canvas_px=canvas_px, grid_values=grid_values, grid_color_idx=grid_color_idx,
        die_net_layout=die_net_layout, die_face_adjacency=die_face_adjacency,
        initial_orientation=initial_orientation,
        start_cell=[start_row, start_col], path_cells=[[cr, cc] for cr, cc in cells],
        true_roll_sequence=true_seq, displayed_roll_sequence=displayed_seq,
        content_mode=content_mode, broken_step_index=broken_step_index,
        correct_orientation_at_break=correct_orientation_at_break,
        displayed_wrong_orientation_at_break=displayed_wrong_orientation_at_break,
        total_rolls=total_rolls, total_cells_visited=total_cells_visited,
        unique_cells_visited=unique_cells_visited,
        top_face_value_history=top_face_value_history,
    )


# CHANGE: new function. Builds the path and the grid values together so the
# number rule holds. Returns None if this attempt gets stuck (the caller
# retries with the same rng, so results stay deterministic per seed).
def _plan_number_path(rng, N, initial_orientation, path_len, broken_step_index):
    def neighbours(cell):
        return [(d, (cell[0] + dr, cell[1] + dc)) for d, (dr, dc) in _DIR_DELTA.items()
                if 0 <= cell[0] + dr < N and 0 <= cell[1] + dc < N]

    value = {}      # cell -> grid value, once fixed
    banned = {}     # cell -> values it must NOT get (else a second direction would work)
    cell = (rng.randrange(N), rng.randrange(N))
    cells, directions = [cell], []
    o = dict(initial_orientation)
    value[cell] = o["top"]                       # CHANGE: start cell shows the start top
    wrong_orientation = None
    # CHANGE: no loops. Cell + orientation fully decides every later move, so
    # returning to a state already seen would repeat the same moves (e.g. the
    # die bouncing between two cells). Such attempts are rejected.
    seen = {(cell, D.orientation_key(o))}

    for i in range(path_len):
        # CHANGE (rule B): for each direction, the top the die would land with.
        # The cell the die just came from never counts: rolling straight back
        # would always bring that cell's number up again (it undoes the roll),
        # so without this there would always be two choices and no dead end.
        prev = cells[-2] if len(cells) > 1 else None
        nbrs = [(d, n, D.roll(o, d)["top"]) for d, n in neighbours(cell) if n != prev]
        already = [(d, n, t) for d, n, t in nbrs if value.get(n) == t]
        if len(already) > 1:
            return None                          # two directions work: not allowed
        if already:
            d, nxt, t = already[0]               # the rule forces this move
        else:
            free = [(d, n, t) for d, n, t in nbrs if n not in value and t not in banned.get(n, ())]
            if not free:
                return None                      # no direction can be made to work
            d, nxt, t = rng.choice(free)
            value[nxt] = t
        for _, n, t2 in nbrs:                    # no other direction may work
            if n != nxt:
                banned.setdefault(n, set()).add(t2)

        correct = D.roll(o, d)
        if i == broken_step_index:
            # same method as before: the result of rolling a different direction
            decoys = [x for x in D.DIRECTIONS if x != d]
            rng.shuffle(decoys)
            for dec in decoys:
                cand = D.roll(o, dec)
                if cand != correct:
                    wrong_orientation = cand
                    break
            if wrong_orientation is None:
                for cand in D.enumerate_orientations(initial_orientation):
                    if cand != correct:
                        wrong_orientation = cand
                        break
            o = dict(wrong_orientation)
        else:
            o = correct
        cell = nxt
        if (cell, D.orientation_key(o)) in seen:
            return None                          # CHANGE: would start a loop
        seen.add((cell, D.orientation_key(o)))
        cells.append(cell)
        directions.append(d)

    # dead end: no direction may work from the last cell (CHANGE: rule B;
    # the cell it just came from does not count)
    for d, n in neighbours(cell):
        if n == cells[-2]:
            continue
        t = D.roll(o, d)["top"]
        if value.get(n) == t:
            return None
        banned.setdefault(n, set()).add(t)

    grid_values = [[0] * N for _ in range(N)]
    for r in range(N):
        for c in range(N):
            if (r, c) in value:
                grid_values[r][c] = value[(r, c)]
            else:
                allowed = [v for v in range(1, 10) if v not in banned.get((r, c), ())]
                if not allowed:
                    return None
                grid_values[r][c] = rng.choice(allowed)
    return grid_values, cells, directions, wrong_orientation
