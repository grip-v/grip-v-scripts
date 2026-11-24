"""Pure-Python ray-tracing engine for laser_mirror_trace -- zero rendering
imports. `simulate()` is the single source of truth for beam physics,
reused identically for the true path, the displayed (possibly broken)
path, and every Level 5 counterfactual re-simulation, so there is only
ever one place beam physics can go wrong.

IMPORTANT finding from development (see README known limitations): a
beam that ENTERS from outside the grid boundary can never loop forever
under this exact reflection rule (single "/" or "\\" per cell, standard
45-degree redirect). Reflection here is reversible -- if entering a cell
in direction d leaves in direction d', then entering in the reverse of
d' always leaves in the reverse of d -- which makes the full transition
function (over cell+direction states, plus the boundary "outside"
states every entry/exit connects to) a bijection built from disjoint
orbits. Boundary entries and exits sit in short entry<->exit orbits;
genuine closed cycles (confirmed to exist -- e.g. a 2x2 "pinwheel" does
form one) are necessarily a SEPARATE orbit consisting only of interior
states, structurally unreachable from any boundary-entry orbit. This
was verified empirically before being trusted: an exhaustive search over
all 512 mirror configurations of a 3x3 grid, every boundary entry point,
found zero loops; 200,000 further random trials up to 7x7 at up to 100%
density found zero more. `simulate()` still fully supports detecting a
loop -- confirmed directly by seeding it from an interior (non-boundary)
starting state, e.g. the 2x2 pinwheel {(0,0):"/",(0,1):"\\",(1,0):"\\",
(1,1):"/"} started at (0,0) facing "left" closes a length-4 cycle
immediately -- should some future variant start a beam from inside the
grid. But `outcome == "infinite_loop"` should never occur from this
dataset's actual (boundary-entry) generation, and is documented as
structurally unreachable rather than silently assumed absent.
"""
from __future__ import annotations

DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def reflect(direction: str, orientation: str) -> str:
    """Standard 45-degree mirror-maze reflection."""
    if orientation == "/":
        return {"right": "up", "up": "right", "left": "down", "down": "left"}[direction]
    if orientation == "\\":
        return {"right": "down", "down": "right", "left": "up", "up": "left"}[direction]
    raise ValueError(orientation)


def in_bounds(cell, n):
    r, c = cell
    return 0 <= r < n and 0 <= c < n


def step(cell, direction):
    dr, dc = DELTA[direction]
    return (cell[0] + dr, cell[1] + dc)


def simulate(mirror_layout: dict, n: int, entry_cell, entry_direction, error_at_reflection_index=None):
    """Traces one beam. `error_at_reflection_index`, if given, forces the
    reflection at that 0-indexed reflection encounter (counting only
    mirror cells) to use the OTHER mirror orientation instead of the true
    one -- every other reflection (before and after) still follows
    correct physics relative to whatever direction the beam actually has
    at that point. Returns (path, result) where result is one of:
      {"outcome": "exited", "exit_cell": .., "exit_direction": ..}
      {"outcome": "infinite_loop", "loop_start_index": .., "loop_cycle_length": ..}
    """
    path = []
    visited = {}
    cell, direction = entry_cell, entry_direction
    reflection_count = 0
    max_steps = n * n * 4 + 8   # finite state space (cell, direction) guarantees termination well before this

    for _ in range(max_steps):
        key = (cell, direction)
        if key in visited:
            loop_start = visited[key]
            return path, {"outcome": "infinite_loop", "loop_start_index": loop_start,
                           "loop_cycle_length": len(path) - loop_start}
        visited[key] = len(path)

        is_mirror = cell in mirror_layout
        reflection_occurred = False
        new_direction = direction
        if is_mirror:
            true_new_direction = reflect(direction, mirror_layout[cell])
            if error_at_reflection_index is not None and reflection_count == error_at_reflection_index:
                wrong_orientation = "\\" if mirror_layout[cell] == "/" else "/"
                new_direction = reflect(direction, wrong_orientation)
            else:
                new_direction = true_new_direction
            reflection_occurred = True
            reflection_count += 1

        path.append({"cell_position": list(cell), "direction_of_travel": direction,
                     "is_mirror_cell": is_mirror, "reflection_occurred": reflection_occurred})

        next_cell = step(cell, new_direction)
        if not in_bounds(next_cell, n):
            return path, {"outcome": "exited", "exit_cell": list(cell), "exit_direction": new_direction}
        cell, direction = next_cell, new_direction

    raise RuntimeError("simulate() exceeded max_steps -- state space bound violated, investigate")


def goal_for_exit(goals, exit_cell, exit_direction):
    for g in goals:
        if tuple(g["cell_position"]) == tuple(exit_cell) and g["direction"] == exit_direction:
            return g
    return None


def classify_result(path, result, goals):
    """Turns a raw simulate() result into (outcome_type, reached_goal, total_reflections)."""
    total_reflections = sum(1 for p in path if p["reflection_occurred"])
    if result["outcome"] == "infinite_loop":
        return "infinite_loop", None, total_reflections
    goal = goal_for_exit(goals, result["exit_cell"], result["exit_direction"])
    if goal is not None:
        return "reached_goal", goal, total_reflections
    return "exited_non_goal", None, total_reflections


def random_entry(rng, n):
    edge = rng.choice(["top", "bottom", "left", "right"])
    idx = rng.randrange(n)
    if edge == "top":
        return (0, idx), "down", edge
    if edge == "bottom":
        return (n - 1, idx), "up", edge
    if edge == "left":
        return (idx, 0), "right", edge
    return (idx, n - 1), "left", edge


def random_goals(rng, n, num_goals):
    edges = ["top", "bottom", "left", "right"]
    goals = []
    seen = set()
    attempts = 0
    while len(goals) < num_goals and attempts < num_goals * 20:
        attempts += 1
        edge = rng.choice(edges)
        idx = rng.randrange(n)
        if edge == "top":
            cell, direction = (0, idx), "up"
        elif edge == "bottom":
            cell, direction = (n - 1, idx), "down"
        elif edge == "left":
            cell, direction = (idx, 0), "left"
        else:
            cell, direction = (idx, n - 1), "right"
        key = (cell, direction)
        if key in seen:
            continue
        seen.add(key)
        goals.append({"cell_position": list(cell), "direction": direction, "edge": edge})
    return goals


def random_mirror_layout(rng, n, density):
    layout = {}
    for r in range(n):
        for c in range(n):
            if rng.random() < density:
                layout[(r, c)] = "/" if rng.random() < 0.5 else "\\"
    return layout
