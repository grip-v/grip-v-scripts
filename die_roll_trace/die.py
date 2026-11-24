"""Die identity (net-folding) and rolling physics -- zero rendering imports.

This module is the ONE place every other piece of ground truth in this
dataset depends on. It has exactly two responsibilities:

1. `fold_net()` -- turn a randomized cross-shaped ("plus hexomino") net into
   a fixed die IDENTITY: a starting {top,bottom,north,south,east,west} state,
   from which the die's permanent opposite/adjacency structure is derived.
2. `roll()` -- the single, exactly-correct physical rotation rule for
   tipping the die one step in a cardinal direction. Every downstream
   sequence (true and displayed) is built by repeated application of this
   one function.

KEYS fixes a canonical ordering used whenever an orientation needs to be
hashed/compared/serialized deterministically.
"""
from __future__ import annotations

KEYS = ["top", "bottom", "north", "south", "east", "west"]

DIRECTIONS = ["north", "south", "east", "west"]

# Fixed net shape: a standard cross/plus hexomino.
#     .  U  .  .
#     L  F  R  B
#     .  D  .  .
# (row, col) position of each of the 6 labeled squares. Only the VALUES
# assigned to these positions are randomized per video -- the shape itself
# is fixed (documented as a deliberate scope reduction in the README: real
# cube nets have 11 distinct valid hexomino foldings, this dataset only uses
# this one canonical cross).
NET_POSITIONS = {
    "U": (0, 1),
    "L": (1, 0), "F": (1, 1), "R": (1, 2), "B": (1, 3),
    "D": (2, 1),
}


def fold_net(values_by_label: dict) -> dict:
    """Fold the cross net into a starting orientation.

    Folding convention (deterministic, matches the standard "wrap the strip
    L-F-R-B around the four sides, U up, D down" cube-net fold): F becomes
    the face pointing north, B (the far end of the strip, hence opposite F
    once folded) becomes south, L becomes west, R becomes east, U becomes
    top, D becomes bottom.
    """
    v = values_by_label
    return {
        "top": v["U"], "bottom": v["D"],
        "north": v["F"], "south": v["B"],
        "east": v["R"], "west": v["L"],
    }


def build_adjacency(initial_orientation: dict) -> dict:
    """Derive the die's fixed identity: {value: {opposite, four_neighbors}}.

    Opposite-face pairs are exactly (top,bottom), (north,south), (east,west)
    in ANY consistent orientation of a folded cube -- so reading them off
    `initial_orientation` is sufficient; this relationship never changes as
    the die rolls. Every other pair of the 6 values is mutually adjacent
    (shares a physical edge), since a cube face touches exactly the 4 faces
    it isn't opposite to.
    """
    pairs = [
        (initial_orientation["top"], initial_orientation["bottom"]),
        (initial_orientation["north"], initial_orientation["south"]),
        (initial_orientation["east"], initial_orientation["west"]),
    ]
    values = [v for pair in pairs for v in pair]
    opposite_of = {}
    for a, b in pairs:
        opposite_of[a] = b
        opposite_of[b] = a
    adjacency = {}
    for val in values:
        opp = opposite_of[val]
        neighbors = sorted(x for x in values if x != val and x != opp)
        adjacency[val] = {"opposite": opp, "four_neighbors": neighbors}
    return adjacency


def roll(orientation: dict, direction: str) -> dict:
    """Physically exact single-step roll. See module docstring.

    Rolling EAST (tipping over the east edge, moving right):
        new_top = old_west, new_bottom = old_east,
        new_east = old_top, new_west = old_bottom, north/south unchanged.
    Rolling WEST is the exact inverse. Rolling NORTH/SOUTH follow the
    analogous cyclic rule with north/south playing the role top/bottom
    played for east/west, and east/west left unchanged.
    """
    t, b = orientation["top"], orientation["bottom"]
    n, s = orientation["north"], orientation["south"]
    e, w = orientation["east"], orientation["west"]

    if direction == "east":
        return {"top": w, "bottom": e, "east": t, "west": b, "north": n, "south": s}
    if direction == "west":
        return {"top": e, "bottom": w, "west": t, "east": b, "north": n, "south": s}
    if direction == "north":
        return {"top": s, "bottom": n, "north": t, "south": b, "east": e, "west": w}
    if direction == "south":
        return {"top": n, "bottom": s, "south": t, "north": b, "east": e, "west": w}
    raise ValueError(f"unknown direction {direction!r}")


def orientation_key(orientation: dict) -> tuple:
    return tuple(orientation[k] for k in KEYS)


def enumerate_orientations(initial_orientation: dict) -> list:
    """BFS over all reachable rotations of the die (exactly 24 for a cube),
    in a fixed, deterministic visiting order (same initial state + same
    direction-iteration order always produces the same list). Used to pick a
    canonical orientation for "what if the die started with value Y on top"
    counterfactual questions (Level 5b).
    """
    from collections import deque

    start_key = orientation_key(initial_orientation)
    seen = {start_key}
    order = [dict(initial_orientation)]
    queue = deque([initial_orientation])
    while queue:
        cur = queue.popleft()
        for d in DIRECTIONS:
            nxt = roll(cur, d)
            key = orientation_key(nxt)
            if key not in seen:
                seen.add(key)
                order.append(nxt)
                queue.append(nxt)
    return order


def orientation_with_top(initial_orientation: dict, top_value) -> dict:
    """First (canonical, deterministic) orientation reachable from
    initial_orientation whose top face equals top_value."""
    for o in enumerate_orientations(initial_orientation):
        if o["top"] == top_value:
            return dict(o)
    raise ValueError(f"value {top_value!r} not one of this die's faces")
