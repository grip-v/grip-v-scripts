"""Hook-grid construction + fill-sequence logic for hooks_grid_trace.

Pure Python, zero rendering imports -- the entire puzzle (hook geometry,
row/column targets, fill order, error injection, every derived
statistic) is fully determined here before Manim ever runs.

HOOK GEOMETRY: hook k (1-indexed) is every cell at Chebyshev distance
k-1 from the fixed top-left corner (0, 0) -- i.e. every (r, c) with
max(r, c) == k-1. Hook k has exactly 2k-1 cells, and a fully correct
fill assigns every cell in hook k the digit k. This is the single source
of truth used identically by generation and validation: `build_hooks`
and `correct_grid` are re-derivable from `grid_size` alone.

A mathematical consequence of this construction (not an independent
design choice): row r's target and column r's target are ALWAYS equal,
since max(r, c) is symmetric in r and c. Row r's cells are also NEVER
touched by hooks 1..r -- only by hooks (r+1)..N -- which is exactly what
makes the Level-5c "when did this row's sum lock in" question
non-trivial for inward fills (see README section 4).
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field

import config as C


def build_hooks(grid_size: int) -> dict:
    """hook_index (1..grid_size) -> [(r, c), ...] cell list."""
    hooks = {k: [] for k in range(1, grid_size + 1)}
    for r in range(grid_size):
        for c in range(grid_size):
            hooks[max(r, c) + 1].append((r, c))
    return hooks


def correct_grid(grid_size: int) -> list:
    return [[max(r, c) + 1 for c in range(grid_size)] for r in range(grid_size)]


def row_column_targets(grid_size: int):
    g = correct_grid(grid_size)
    row_t = [sum(g[r]) for r in range(grid_size)]
    col_t = [sum(g[r][c] for r in range(grid_size)) for c in range(grid_size)]
    return row_t, col_t


def fill_order_for(grid_size: int, direction: str) -> list:
    if direction == "outward":
        return list(range(1, grid_size + 1))
    return list(range(grid_size, 0, -1))


def row_contributing_hooks(grid_size: int, row: int) -> list:
    """Hooks that ever place a digit into `row` -- exactly {row+1, ..., grid_size}."""
    return list(range(row + 1, grid_size + 1))


def row_lock_in_step(grid_size: int, fill_order: list, row: int) -> int:
    """0-indexed step (into fill_order) at which row's running sum first
    reaches its final value and never changes again."""
    contributing = set(row_contributing_hooks(grid_size, row))
    return max(fill_order.index(h) for h in contributing)


def cumulative_cells_through(grid_size: int, hooks: dict, fill_order: list, target_hook: int) -> int:
    idx = fill_order.index(target_hook)
    return sum(len(hooks[h]) for h in fill_order[:idx + 1])


@dataclass
class SceneData:
    grid_size: int
    fill_direction: str
    running_sums_displayed: bool
    hooks: dict                      # hook_index -> [(r, c), ...]
    fill_order: list                 # hook indices, animation order
    row_targets: list
    column_targets: list
    shaded_cells: list               # [(r, c), ...]
    shaded_cells_sum: int
    is_broken: bool
    error_cell_position: tuple | None
    error_hook_index: int | None
    correct_digit: int | None
    placed_digit: int | None
    affected_row_index: int | None
    affected_column_index: int | None
    error_fill_step: int | None
    final_grid: list
    final_row_sums: list
    final_column_sums: list
    row_sum_matches: list
    column_sum_matches: list


def build_scene(rng: random.Random) -> SceneData:
    grid_size = rng.randint(*C.GRID_SIZE_RANGE)
    fill_direction = rng.choice(("inward", "outward"))
    running_sums_displayed = rng.random() < C.RUNNING_SUMS_PROBABILITY

    hooks = build_hooks(grid_size)
    fill_order = fill_order_for(grid_size, fill_direction)
    row_targets, column_targets = row_column_targets(grid_size)
    final_grid = correct_grid(grid_size)

    is_broken = rng.random() < C.BROKEN_PROBABILITY
    error_cell_position = error_hook_index = correct_digit = placed_digit = None
    affected_row_index = affected_column_index = error_fill_step = None
    if is_broken:
        error_hook_index = rng.randint(2, grid_size)   # never hook 1 -- the single corner cell
        error_cell_position = rng.choice(hooks[error_hook_index])
        correct_digit = error_hook_index
        placed_digit = correct_digit + rng.choice((-1, 1))
        r, c = error_cell_position
        final_grid[r][c] = placed_digit
        affected_row_index = r
        affected_column_index = c
        error_fill_step = fill_order.index(error_hook_index)

    final_row_sums = [sum(row) for row in final_grid]
    final_column_sums = [sum(final_grid[r][c] for r in range(grid_size)) for c in range(grid_size)]
    row_sum_matches = [final_row_sums[i] == row_targets[i] for i in range(grid_size)]
    column_sum_matches = [final_column_sums[i] == column_targets[i] for i in range(grid_size)]

    shaded_cells = [(r, c) for r in range(grid_size) for c in range(grid_size)
                    if rng.random() < C.SHADE_PROBABILITY]
    shaded_cells_sum = sum(final_grid[r][c] for r, c in shaded_cells)

    return SceneData(
        grid_size=grid_size, fill_direction=fill_direction, running_sums_displayed=running_sums_displayed,
        hooks=hooks, fill_order=fill_order, row_targets=row_targets, column_targets=column_targets,
        shaded_cells=shaded_cells, shaded_cells_sum=shaded_cells_sum, is_broken=is_broken,
        error_cell_position=error_cell_position, error_hook_index=error_hook_index, correct_digit=correct_digit,
        placed_digit=placed_digit, affected_row_index=affected_row_index, affected_column_index=affected_column_index,
        error_fill_step=error_fill_step, final_grid=final_grid, final_row_sums=final_row_sums,
        final_column_sums=final_column_sums, row_sum_matches=row_sum_matches, column_sum_matches=column_sum_matches,
    )
