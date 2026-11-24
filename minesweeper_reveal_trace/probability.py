"""Exact minesweeper constraint-satisfaction probability engine.

`compute_mine_probability` (and its sibling `is_satisfiable`) are the
single most correctness-critical piece of this dataset: they take ONLY
the currently-revealed displayed values (never true mine positions) and
compute the exact P(mine) for an unrevealed target cell, or detect that
no valid mine layout is consistent with what's been revealed at all.

Method: build the constraint graph over unrevealed cells (two unrevealed
cells are linked if some revealed numbered cell has both as neighbors),
split it into independent connected components, brute-force enumerate
every mine/no-mine assignment within each component that satisfies all
of that component's constraints (component sizes stay small at this
grid-size/density range -- see config.MAX_COMPONENT_BRUTE_FORCE_CELLS),
then combine per-component mine-count distributions with the pool of
totally unconstrained cells via integer convolution against the known
GLOBAL total mine count. This is exact combinatorics throughout, never
sampling or approximation, per spec.
"""
from __future__ import annotations
from itertools import product
from math import comb

MAX_COMPONENT_BRUTE_FORCE_CELLS = 45   # enumeration ceiling per constraint component


def neighbors(r, c, n):
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0) and 0 <= r + dr < n and 0 <= c + dc < n]


def _constraints(n, board_state):
    """board_state: {(r,c): displayed_value} for every REVEALED cell (0-8 or
    "mine"). An uncovered mine is a known mine: it is not an unknown cell,
    and each neighbouring number needs one fewer mine among its covered
    neighbours. Returns (unrevealed_cells: set, constraints: list of
    (frozenset_of_unrevealed_neighbors, required_count), impossible: bool),
    where `impossible` flags a number that can't be met at all (e.g. a
    nonzero number with no covered neighbours left)."""
    revealed = set(board_state)
    known_mines = {cell for cell, v in board_state.items() if v == "mine"}
    unrevealed = {(r, c) for r in range(n) for c in range(n)} - revealed
    constraints, impossible = [], False
    for (r, c), v in board_state.items():
        if v == "mine":
            continue
        nbs = neighbors(r, c, n)
        unrev_neighbors = frozenset(nb for nb in nbs if nb in unrevealed)
        need = v - sum(1 for nb in nbs if nb in known_mines)
        if need < 0 or need > len(unrev_neighbors):
            impossible = True
        elif unrev_neighbors:
            constraints.append((unrev_neighbors, need))
    return unrevealed, constraints, impossible


def _connected_components(unrevealed, constraints):
    """Union-find over unrevealed cells touched by at least one constraint."""
    parent = {cell: cell for group, _ in constraints for cell in group}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for group, _ in constraints:
        cells = list(group)
        for i in range(1, len(cells)):
            union(cells[0], cells[i])

    comps = {}
    for cell in parent:
        comps.setdefault(find(cell), set()).add(cell)
    constrained_cells = set(parent)
    unconstrained_cells = unrevealed - constrained_cells
    return list(comps.values()), unconstrained_cells


def _component_assignments(component, constraints):
    """All mine/no-mine assignments over `component`'s cells that satisfy every
    constraint fully contained in it. Exact exhaustive enumeration via pruned
    backtracking (not a raw 2**|component| product): assigns cells one at a
    time and immediately rejects a partial assignment once any constraint's
    running sum exceeds its target, or once a constraint's cells are all
    assigned but its sum doesn't exactly match -- this is what keeps
    real-world minesweeper-style sparse constraint graphs fast up to the
    documented cell ceiling, unlike the worst-case exponential product."""
    cells = sorted(component)
    if len(cells) > MAX_COMPONENT_BRUTE_FORCE_CELLS:
        raise RuntimeError(f"component of {len(cells)} cells exceeds the enumeration ceiling "
                            f"({MAX_COMPONENT_BRUTE_FORCE_CELLS}) -- see README known limitations")
    local_constraints = [(group, req) for group, req in constraints if group <= set(cells)]
    cell_constraints = {cell: [] for cell in cells}
    for ci, (group, _req) in enumerate(local_constraints):
        for cell in group:
            cell_constraints[cell].append(ci)
    total_in_constraint = [len(group) for group, _req in local_constraints]
    req = [r for _group, r in local_constraints]
    assigned_count = [0] * len(local_constraints)
    running_sum = [0] * len(local_constraints)

    results = []
    assignment = {}

    def backtrack(i):
        if i == len(cells):
            mine_set = frozenset(c for c in cells if assignment[c])
            results.append((mine_set, len(mine_set)))
            return
        cell = cells[i]
        for val in (0, 1):
            assignment[cell] = val
            touched = cell_constraints[cell]
            ok = True
            for ci in touched:
                assigned_count[ci] += 1
                running_sum[ci] += val
                if running_sum[ci] > req[ci] or (assigned_count[ci] == total_in_constraint[ci]
                                                  and running_sum[ci] != req[ci]):
                    ok = False
            if ok:
                backtrack(i + 1)
            for ci in touched:
                assigned_count[ci] -= 1
                running_sum[ci] -= val
        del assignment[cell]

    backtrack(0)
    return results


def _count_dist(assignments):
    dist = {}
    for _, m in assignments:
        dist[m] = dist.get(m, 0) + 1
    return dist


def _convolve(dists):
    result = {0: 1}
    for d in dists:
        new = {}
        for k1, v1 in result.items():
            for k2, v2 in d.items():
                new[k1 + k2] = new.get(k1 + k2, 0) + v1 * v2
        result = new
    return result


def _unconstrained_dist(u):
    return {k: comb(u, k) for k in range(u + 1)}


def _solve(n, board_state, total_mine_count):
    """Returns (comps, comp_assignments_by_id, unconstrained_cells, remaining_mines,
    total_ways) -- the full solved state, reused by both is_satisfiable and
    compute_mine_probability so they can never disagree with each other."""
    n_revealed_mines = sum(1 for v in board_state.values() if v == "mine")
    remaining_mines = total_mine_count - n_revealed_mines

    unrevealed, constraints, impossible = _constraints(n, board_state)
    if impossible:
        return [], [], set(unrevealed), remaining_mines, 0
    comps, unconstrained = _connected_components(unrevealed, constraints)
    comp_assignments = [_component_assignments(comp, constraints) for comp in comps]
    comp_dists = [_count_dist(a) for a in comp_assignments]

    total_dist = _convolve(comp_dists + [_unconstrained_dist(len(unconstrained))])
    total_ways = total_dist.get(remaining_mines, 0) if 0 <= remaining_mines <= (n * n) else 0
    return comps, comp_assignments, unconstrained, remaining_mines, total_ways


def is_satisfiable(n, board_state, total_mine_count) -> bool:
    """True iff at least one mine layout is consistent with every revealed
    displayed value AND the known global mine total."""
    *_, total_ways = _solve(n, board_state, total_mine_count)
    return total_ways > 0


def compute_mine_probability(n, board_state, target_cell, total_mine_count):
    """Exact P(target_cell is a mine) given only `board_state`'s revealed
    displayed values and the known total mine count. Returns a
    fractions.Fraction, or None if the revealed values are jointly
    UNSATISFIABLE (no valid layout exists at all -- P is undefined, not 0)."""
    from fractions import Fraction

    comps, comp_assignments, unconstrained, remaining_mines, total_ways = _solve(n, board_state, total_mine_count)
    if total_ways == 0:
        return None
    if not (0 <= target_cell[0] < n and 0 <= target_cell[1] < n) or target_cell in board_state:
        raise ValueError(f"target_cell {target_cell} is not a valid unrevealed cell")

    # find which component (if any) target_cell belongs to
    owner_idx = None
    for i, comp in enumerate(comps):
        if target_cell in comp:
            owner_idx = i
            break

    if owner_idx is None:
        # target is an unconstrained cell: fix it as a mine and remove it from the pool
        u = len(unconstrained)
        other_dists = [_count_dist(a) for a in comp_assignments] + [_unconstrained_dist(u - 1)]
        ways_target_mine = _convolve(other_dists).get(remaining_mines - 1, 0)
    else:
        owner_assignments = comp_assignments[owner_idx]
        other_dists = [_count_dist(comp_assignments[j]) for j in range(len(comps)) if j != owner_idx]
        other_dists.append(_unconstrained_dist(len(unconstrained)))
        other_ways = _convolve(other_dists)
        ways_target_mine = sum(
            other_ways.get(remaining_mines - m, 0)
            for mine_set, m in owner_assignments if target_cell in mine_set
        )

    return Fraction(ways_target_mine, total_ways)
