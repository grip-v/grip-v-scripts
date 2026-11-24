"""Multi-ant scenario simulation -- the two-pass "everything happens in
plain Python first" substrate for random_walk_home, exactly like a
single-agent version's original simulate.py: zero rendering imports, fully
deterministic from `seed` alone, scene.py only ever plays back the
resulting per-tick history.

Each ant starts AWAY from its own home (not on it) and takes a uniformly
random walk among its current cell's walkable neighbors each tick, all
ants moving simultaneously per tick, until it either lands on its own
home (done) or shares a cell with a predator after a tick (dead,
vanishes -- no reset). Four independent, per-clip-randomized layers can
also be active: a home relocating once mid-walk, the board's black/white
tiles swapping once or more, grey hazard "hole" cells that bounce an ant
back to its OWN start (not home, and not a death), and roaming predators.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field

import networkx as nx

import arena as A
import config as C


@dataclass
class AntResult:
    symbol: str
    start: tuple
    home_initial: tuple
    home_final: tuple
    outcome: str            # "home" | "dead"
    steps_taken: int        # tick at which it first became resolved
    reset_count: int        # number of times it fell into a hole and was bounced back to `start`
    died_tick: int | None
    died_by_predator: int | None   # predator index, if outcome == "dead"


@dataclass
class ScenarioResult:
    seed: int
    n_ants: int
    ants: list                  # [AntResult, ...]
    n_predators: int
    n_holes: int
    home_moved: bool
    home_moved_ant_symbol: str | None
    tile_swap_count: int
    hole_reset_count: int
    death_count: int
    total_ticks: int
    total_events: int           # home-move + tile-swap + hole-reset + death + wind-shift (CHANGE), each counted once
    positions_history: list     # [{symbol: (q, r) or None (vanished/not yet started concept -- always present)} per tick]
    predator_history: list      # [[(q, r), ...] per tick]
    home_history: list          # [{symbol: (q, r)} per tick -- current home position, changes once if home_moved]
    blocked_history: list       # [{cell: bool} per tick] -- board state at each tick
    board: list                 # all (q, r) on the board
    holes: list                 # fixed hole cells for the whole clip
    wind_strength: str | None = None   # CHANGE: None (calm) | "light" | "strong"
    wind_history: list = field(default_factory=list)   # CHANGE: [dir index 0..5 or None per tick]; [k] = wind
                                                       # during move k ([0] = wind shown in the first frame)
    wind_shift_tick: int | None = None                 # CHANGE: tick at whose start the wind changed direction
    gust_history: list = field(default_factory=list)   # CHANGE: [bool per tick]; [k] = move k is a gust ([0] False)
    hole_hit_history: list = field(default_factory=list)   # CHANGE: [{symbol: hole cell stepped into, or None}
                                                           # per tick] -- lets the scene SHOW the fall (render only)


def _attempt_tile_swap(rng, board_set, blocked, required_cells):
    candidates_to_white = [c for c in blocked if blocked[c]
                            and any(not blocked[n] for n in A.neighbors_on_board(c, board_set))]
    if not candidates_to_white:
        return None
    to_white = rng.choice(candidates_to_white)
    exclude = set(required_cells) | {to_white}
    candidates_to_black = [c for c in blocked if not blocked[c] and c not in exclude]
    rng.shuffle(candidates_to_black)
    for to_black in candidates_to_black[:12]:
        trial = dict(blocked)
        trial[to_white] = False
        trial[to_black] = True
        g = A.white_subgraph(board_set, trial)
        comp = set(nx.node_connected_component(g, next(iter(required_cells)))) if required_cells else set()
        if all(c in comp for c in required_cells):
            blocked[to_white] = False
            blocked[to_black] = True
            return (to_white, to_black)
    return None


def _attempt_home_move(rng, white_cells, old_home, taken_cells):
    lo, hi = C.HOME_MOVE_DIST_RANGE
    candidates = [c for c in white_cells if lo <= A.cube_distance(old_home, c) <= hi and c not in taken_cells]
    if not candidates:
        return None
    return rng.choice(candidates)


def _build_and_run(rng: random.Random):
    n_ants = rng.randint(*C.N_ANTS_RANGE)
    board = A.board_cells(C.ARENA_RADIUS)
    board_set = set(board)
    blocked = A.initial_blocked_layout(rng, board)
    component = A.largest_component_nodes(board_set, blocked)

    reserved_needed = n_ants * 2 + 4
    retries = 0
    while len(component) < reserved_needed and retries < 20:
        blocked = A.initial_blocked_layout(rng, board)
        component = A.largest_component_nodes(board_set, blocked)
        retries += 1
    if len(component) < reserved_needed:
        return None

    pool = list(component)
    rng.shuffle(pool)
    symbol_order = rng.sample(range(len(C.SYMBOLS)), n_ants)
    symbols = [C.SYMBOLS[i] for i in symbol_order]

    homes = pool[:n_ants]
    rest = pool[n_ants:]
    starts = rest[:n_ants]
    rest = rest[n_ants:]

    holes = []
    if rng.random() < C.HOLE_PROBABILITY:
        n_holes = rng.randint(*C.N_HOLES_RANGE)
        holes = rest[:n_holes]
        rest = rest[n_holes:]

    predator_positions = []
    if rng.random() < C.PREDATOR_PROBABILITY:
        n_pred = rng.randint(*C.N_PREDATORS_RANGE)
        predator_positions = rest[:n_pred]
        rest = rest[n_pred:]

    home_move_active = rng.random() < C.HOME_MOVE_PROBABILITY
    home_move_ant_idx = rng.randrange(n_ants) if home_move_active else None
    tile_swap_active = rng.random() < C.TILE_SWAP_PROBABILITY

    # CHANGE: wind, drawn after every pre-existing layer
    wind_strength, wind_dir, wind_shift_at, wind_dir_after = None, None, None, None
    if rng.random() < C.WIND_PROBABILITY:
        wind_strength = rng.choice(sorted(C.WIND_GUST))
        wind_dir = rng.randrange(6)
        if rng.random() < C.WIND_SHIFT_PROBABILITY:
            wind_shift_at = rng.randint(*C.WIND_SHIFT_TICK_RANGE)
            wind_dir_after = rng.choice([d for d in range(6) if d != wind_dir])
    wind_history = [wind_dir]
    wind_shift_tick = None
    gust_history = [False]

    ants = [{"symbol": symbols[i], "pos": starts[i], "start": starts[i], "home": homes[i],
             "home_initial": homes[i], "status": "walking", "reset_count": 0,
             "died_tick": None, "died_by": None, "resolved_tick": None} for i in range(n_ants)]
    holes_set = set(holes)

    positions_history = [{a["symbol"]: a["pos"] for a in ants}]
    predator_history = [list(predator_positions)]
    home_history = [{a["symbol"]: a["home"] for a in ants}]
    blocked_history = [dict(blocked)]
    hole_hit_history = [{a["symbol"]: None for a in ants}]   # CHANGE

    home_moved = False
    home_moved_symbol = None
    tile_swap_count = 0
    hole_reset_count = 0
    death_count = 0

    tick = 0
    while any(a["status"] == "walking" for a in ants) and tick < C.MAX_TICKS:
        tick += 1

        active_positions = [a["pos"] for a in ants if a["status"] != "dead"]
        active_homes = [a["home"] for a in ants if a["status"] == "walking"]
        required = set(active_positions) | set(active_homes) | set(predator_positions)
        if holes_set:
            # CHANGE: a hole sends a walking ant back to its start, so that start must stay open (before, a swap
            # could blacken it and the bounce then put the ant on a black cell -- and could crash the swap check)
            required |= {a["start"] for a in ants if a["status"] == "walking"}

        if tile_swap_active and tick in C.TILE_SWAP_TICKS and required:
            result = _attempt_tile_swap(rng, board_set, blocked, required)
            if result is not None:
                tile_swap_count += 1

        if (home_move_active and tick == C.HOME_MOVE_TICK
                and ants[home_move_ant_idx]["status"] == "walking"):
            white_cells = [c for c in board if not blocked[c]]
            taken = {a["home"] for a in ants} | {a["pos"] for a in ants if a["status"] != "dead"} | holes_set
            # CHANGE: never onto the moving ant's own start -- a hole bounces it back there, and it would then
            # stand on its own home without counting as home (seen at seed 1575)
            taken.add(ants[home_move_ant_idx]["start"])
            new_home = _attempt_home_move(rng, white_cells, ants[home_move_ant_idx]["home"], taken)
            if new_home is not None:
                ants[home_move_ant_idx]["home"] = new_home
                home_moved = True
                home_moved_symbol = ants[home_move_ant_idx]["symbol"]

        if wind_shift_at is not None and tick == wind_shift_at:   # CHANGE: wind shift (counted as an event)
            wind_dir = wind_dir_after
            wind_shift_tick = tick
        wind_history.append(wind_dir)
        gust = False
        if wind_dir is not None:      # CHANGE: gust / lull Markov chain; never a gust on the shift move
            p_on, p_stay = C.WIND_GUST[wind_strength]
            gust = rng.random() < (p_stay if gust_history[-1] else p_on) and tick != wind_shift_tick
        gust_history.append(gust)

        hole_hits = {a["symbol"]: None for a in ants}   # CHANGE
        for a in ants:
            if a["status"] != "walking":
                continue
            nbrs = [c for c in A.neighbors_on_board(a["pos"], board_set) if not blocked[c]]
            new_pos = rng.choice(nbrs) if nbrs else a["pos"]
            if gust:
                # CHANGE: gust -- the random step above is always drawn; it is replaced by one cell downwind for
                # every ant whose downwind cell is on the board and open (all ants pushed together)
                dq, dr = A.HEX_NEIGHBOR_OFFSETS[wind_dir]
                down = (a["pos"][0] + dq, a["pos"][1] + dr)
                if down in board_set and not blocked[down]:
                    new_pos = down
            if new_pos == a["home"]:
                a["pos"] = new_pos
                a["status"] = "home"
                a["resolved_tick"] = tick
            elif new_pos in holes_set:
                hole_hits[a["symbol"]] = new_pos   # CHANGE: recorded only; no extra rng draw
                hole_reset_count += 1
                a["reset_count"] += 1
                a["pos"] = a["start"]
            else:
                a["pos"] = new_pos

        new_predator_positions = []
        for p in predator_positions:
            nbrs = [c for c in A.neighbors_on_board(p, board_set) if not blocked[c]]
            new_predator_positions.append(rng.choice(nbrs) if nbrs else p)
        predator_positions = new_predator_positions

        for pi, ppos in enumerate(predator_positions):
            for a in ants:
                if a["status"] == "walking" and a["pos"] == ppos:
                    a["status"] = "dead"
                    a["died_tick"] = tick
                    a["died_by"] = pi
                    a["resolved_tick"] = tick
                    death_count += 1

        positions_history.append({a["symbol"]: a["pos"] for a in ants})
        predator_history.append(list(predator_positions))
        home_history.append({a["symbol"]: a["home"] for a in ants})
        blocked_history.append(dict(blocked))
        hole_hit_history.append(hole_hits)   # CHANGE

    if any(a["status"] == "walking" for a in ants):
        return None   # hit MAX_TICKS unresolved -- rejected, caller re-draws
    if tick < C.MIN_TICKS:
        return None   # CHANGE: finished in under MIN_TICKS moves (e.g. one ant starting next to its home) --
                      # too short for the question ladder; rejected and re-drawn exactly like MAX_TICKS

    ant_results = [AntResult(symbol=a["symbol"], start=a["start"], home_initial=a["home_initial"],
                              home_final=a["home"], outcome=a["status"], steps_taken=a["resolved_tick"],
                              reset_count=a["reset_count"], died_tick=a["died_tick"], died_by_predator=a["died_by"])
                   for a in ants]

    total_events = ((1 if home_moved else 0) + tile_swap_count + hole_reset_count + death_count
                    + (1 if wind_shift_tick is not None else 0))   # CHANGE: a wind shift is an event too

    return ScenarioResult(
        seed=0, n_ants=n_ants, ants=ant_results, n_predators=len(predator_history[0]), n_holes=len(holes),
        home_moved=home_moved, home_moved_ant_symbol=home_moved_symbol, tile_swap_count=tile_swap_count,
        hole_reset_count=hole_reset_count, death_count=death_count, total_ticks=tick, total_events=total_events,
        positions_history=positions_history, predator_history=predator_history, home_history=home_history,
        blocked_history=blocked_history, board=board, holes=holes,
        wind_strength=wind_strength, wind_history=wind_history, wind_shift_tick=wind_shift_tick,
        gust_history=gust_history,
        hole_hit_history=hole_hit_history,
    )


def simulate_scenario(seed: int) -> ScenarioResult:
    """Fully deterministic from `seed` alone. A candidate scenario that
    doesn't fully resolve (every ant home or dead) within MAX_TICKS is
    REJECTED and a fresh one is drawn from the same rng stream -- the
    same rejection-sampling convention documented for the original single-agent
    simulate.simulate_walk (never a mid-walk truncation)."""
    rng = random.Random(seed)
    result = None
    while result is None:
        result = _build_and_run(rng)
    result.seed = seed
    return result
