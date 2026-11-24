"""Builds the 5-question ladder for one random_walk_home scenario.

CHANGE: rewritten as one closed-loop ladder, one question per level, in order:
  q01 Direct Observation        what is plainly on screen
  q02 Relational Reasoning      which ant relates to which home / predator / event / the wind
  q03 Measurement               counting events or moves over the clip
  q04 Multi-step Integration    combining several events or several ants
  q05 Counterfactual Reasoning  a stated change, everything else held fixed
Every level has wind variants (see config.WIND_*), so wind / calm is asked about at
every rung. Each level picks one of its variants per video. Every answer is
closed-form from the simulated scenario, has exactly one correct option, and a
variant is only offered when its answer is unique (ties are never asked).
"""
from __future__ import annotations
import random

import arena as A
import config as C

# CHANGE: counterfactuals hold every ant's own moves fixed; this sentence is part of every such L5 prompt
SAME_MOVES = ("Assume every ant would have made exactly the same moves as in the video, up to the point "
              "where the change below would have ended its walk. ")

# CHANGE: how the wind looks on screen; prefixed to every wind question
WIND_NOTE = ("The gauge in the corner shows the wind: its arrow points the way the wind blows, one chevron means "
             "a light wind and two chevrons a strong one, and CALM means no wind. The wind comes in gusts: while "
             "the GUST tag is lit, every ant whose next cell downwind is open is blown one cell that way, all at "
             "once; between gusts the ants walk on their own. The red predators are never blown. ")

DIRS = C.WIND_DIR_NAMES


def _mc(rng, correct, distractors, formatter=str):
    pool = list(dict.fromkeys([correct] + list(distractors)))[:4]
    rng.shuffle(pool)
    letters = "ABCD"[:len(pool)]
    options = {letters[i]: formatter(pool[i]) for i in range(len(pool))}
    return options, letters[pool.index(correct)]


def _numeric_distractors(rng, correct, lo=0, hi=None, spreads=(1, 2, 3, 4, 5)):
    out = set()
    for s in spreads:
        for v in (correct - s, correct + s):
            if v != correct and v >= lo and (hi is None or v <= hi):
                out.add(v)
        if len(out) >= 6:
            break
    out = sorted(out)
    rng.shuffle(out)
    return out[:3]


def _name(symbol: str) -> str:
    """CHANGE: how an ant is named in prompts and options, e.g. 'heart (♥)'."""
    return f"{symbol} ({C.SYMBOL_GLYPH[symbol]})"


def _yes_no(rng, yes: bool):
    correct = "yes" if yes else "no"
    return _mc(rng, correct, ["no" if yes else "yes"]) + (correct,)


def _balanced_pick(rng, items, key):
    """CHANGE: pick an item so that key(item) is True or False with equal chance whenever both kinds exist
    (keeps yes/no answers balanced instead of mirroring how common each outcome is)."""
    groups = [[x for x in items if key(x)], [x for x in items if not key(x)]]
    groups = [g for g in groups if g]
    return rng.choice(rng.choice(groups))


def _path(scenario, ant):
    """Cells the ant stood on after each tick 0..steps_taken (0 = its start)."""
    return [scenario.positions_history[t][ant.symbol] for t in range(ant.steps_taken + 1)]


def _step(cell, d):
    dq, dr = A.HEX_NEIGHBOR_OFFSETS[d]
    return (cell[0] + dq, cell[1] + dr)


def _wind_effect(ant, d):
    """CHANGE: -1 if one step toward wind direction d takes the ant (from its start) closer to its starting home,
    +1 if farther, 0 if the same hex distance. Pure geometry, black/grey cells ignored."""
    now = A.cube_distance(ant.start, ant.home_initial)
    nxt = A.cube_distance(_step(ant.start, d), ant.home_initial)
    return (nxt > now) - (nxt < now)


def _downwind_moves(scenario, ant):
    """CHANGE: how many of the ant's moves went exactly one cell the way the wind arrow pointed during that move
    (a step into a grey hole counts by the hole cell it stepped onto)."""
    n = 0
    for t in range(1, ant.steps_taken + 1):
        d = scenario.wind_history[t]
        if d is None:
            continue
        prev = scenario.positions_history[t - 1][ant.symbol]
        landed = scenario.hole_hit_history[t][ant.symbol] or scenario.positions_history[t][ant.symbol]
        if tuple(landed) == _step(prev, d):
            n += 1
    return n


def _home_side(ant, d):
    """CHANGE: cosine between the screen vector start->home and the wind direction d (+1 = home straight
    downwind, -1 = straight upwind)."""
    hx, hy = A.axial_to_pixel(*ant.home_initial, 1.0)
    sx, sy = A.axial_to_pixel(*ant.start, 1.0)
    wx, wy = A.axial_to_pixel(*A.HEX_NEIGHBOR_OFFSETS[d], 1.0)
    vx, vy = hx - sx, hy - sy
    return (vx * wx + vy * wy) / ((vx * vx + vy * vy) ** 0.5 * (wx * wx + wy * wy) ** 0.5)


def _gust_pushes(scenario):
    """CHANGE: (ant, move) pairs where the move was a gust and the ant went one cell downwind (counting the hole
    cell it stepped onto, if it fell in)."""
    out = []
    for a in scenario.ants:
        for t in range(1, a.steps_taken + 1):
            if not scenario.gust_history[t]:
                continue
            prev = scenario.positions_history[t - 1][a.symbol]
            landed = scenario.hole_hit_history[t][a.symbol] or scenario.positions_history[t][a.symbol]
            if tuple(landed) == _step(prev, scenario.wind_history[t]):
                out.append((a.symbol, t))
    return out


def _n_gusts(scenario):
    g = scenario.gust_history
    return sum(1 for k in range(1, len(g)) if g[k] and not g[k - 1])


def _gale(scenario, ant, d):
    """CHANGE: blown downwind on every move from its start on the first-frame board: returns (moves made,
    reached_home). It stops on reaching its starting home, or when the next cell downwind is black, grey or off
    the board."""
    blocked, holes, board = scenario.blocked_history[0], set(scenario.holes), set(scenario.board)
    cell, n = ant.start, 0
    while True:
        nxt = _step(cell, d)
        if nxt not in board or blocked[nxt] or nxt in holes:
            return n, False
        cell, n = nxt, n + 1
        if cell == ant.home_initial:
            return n, True


def _shortest_open_route(scenario, ant, holes_open=False):
    """CHANGE: fewest moves from the ant's start to its starting home over cells that are white in the
    first frame (black tiles and grey holes avoided). None if there is no such route."""
    blocked = scenario.blocked_history[0]
    holes = set() if holes_open else set(scenario.holes)
    board = set(scenario.board)
    frontier, seen, d = [ant.start], {ant.start}, 0
    while frontier:
        if ant.home_initial in frontier:
            return d
        nxt = []
        for c in frontier:
            for nb in A.neighbors_on_board(c, board):
                if nb not in seen and not blocked[nb] and nb not in holes:
                    seen.add(nb)
                    nxt.append(nb)
        frontier, d = nxt, d + 1
    return None


def build_questions(scenario, rng: random.Random) -> list:
    questions = []
    ants = scenario.ants
    n = scenario.n_ants
    home = [a for a in ants if a.outcome == "home"]
    dead = [a for a in ants if a.outcome == "dead"]
    d0 = scenario.wind_history[0]                      # wind in the first frame (None = calm)
    windy = d0 is not None
    shifted = scenario.wind_shift_tick is not None

    def add(qid, level, variant, prompt, options, ans, gt):
        questions.append({"qid": qid, "difficulty_level": level, "task": C.TASK_NAME[level], "variant": variant,
                          "prompt": prompt, "options": options, "answer": ans, "groundtruth": gt})

    def pick(plain, wind):
        """CHANGE: half the time a level asks one of its wind variants (when one is valid for this clip)."""
        if wind and (not plain or rng.random() < 0.5):
            return rng.choice(wind)
        return rng.choice(plain)

    def dir_options(correct):
        others = [x for x in DIRS if x != correct]
        rng.shuffle(others)
        return _mc(rng, correct, others[:3])

    # ---------------- q01  Level 1: Direct Observation ----------------
    # wind_any is asked at different rates in calm and windy clips so its yes/no answers come out ~50/50
    # (windy clips are ~59% of all clips: 0.3 * 41 / 59 ~= 0.21)
    r = rng.random()
    if not windy:
        v = "wind_any" if r < 0.3 else rng.choice(["n_ants", "n_home"])
    elif r < 0.21:
        v = "wind_any"
    elif r < 0.59:
        v = rng.choice(["wind_dir_start", "wind_strength"] + (["wind_dir_end"] if shifted else []))
    else:
        v = rng.choice(["n_ants", "n_home"])
    if v == "n_ants":
        options, ans = _mc(rng, n, _numeric_distractors(rng, n, lo=1, spreads=(1, 2, 3)))
        add("q01", 1, v, "How many ants were on the board at the start of the video?", options, ans, n)
    elif v == "n_home":
        options, ans = _mc(rng, len(home), _numeric_distractors(rng, len(home), lo=0, spreads=(1, 2, 3)))
        add("q01", 1, v, "How many ants reached their own home by the end of the video?", options, ans, len(home))
    elif v == "wind_any":
        options, ans, gt = _yes_no(rng, windy)
        add("q01", 1, v, WIND_NOTE + "Was there any wind in this video? Answer yes or no.", options, ans, gt)
    elif v == "wind_dir_start":
        options, ans = dir_options(DIRS[d0])
        add("q01", 1, v, WIND_NOTE + "At the start of the video, which way was the wind blowing?", options, ans,
            DIRS[d0])
    elif v == "wind_dir_end":
        dE = scenario.wind_history[-1]
        options, ans = dir_options(DIRS[dE])
        add("q01", 1, v, WIND_NOTE + "The wind changed direction once during the video. Which way was it blowing "
            "at the end?", options, ans, DIRS[dE])
    else:
        gt = scenario.wind_strength
        options, ans = _mc(rng, gt, ["light", "strong"])
        add("q01", 1, v, WIND_NOTE + "Was the wind light or strong?", options, ans, gt)

    # ---------------- q02  Level 2: Relational Reasoning ----------------
    # own_home_reached only when the clip has BOTH an ant that got home and one that didn't -- otherwise the
    # answer is always "yes" (most clips end with every ant home) and the model learns a yes-prior
    plain = []
    if home and dead:
        plain.append("own_home_reached")
    if dead and len(dead) < n:
        plain.append("caught_which")
    if scenario.home_moved and n >= 2:
        plain.append("home_moved_whose")
    wind = []
    helped = [a for a in ants if windy and _wind_effect(a, d0) < 0]
    if windy and helped and len(helped) < n:   # both a helped and an unhelped ant: yes/no picked 50/50
        wind += ["helped_at_start", "helped_which"]
    # downwind / upwind only when the home is clearly to one side (angle to the wind < ~70 deg or > ~110 deg)
    side_ok = [a for a in ants if windy and abs(_home_side(a, d0)) > 0.35]
    if side_ok:
        wind.append("home_downwind")
    # where each ant's home sits on screen relative to the ant itself, in the first frame (always available
    # unless every ant starts level with its home)
    def home_higher(a):
        return A.axial_to_pixel(*a.home_initial, 1.0)[1] > A.axial_to_pixel(*a.start, 1.0)[1]
    level_ok = [a for a in ants if abs(A.axial_to_pixel(*a.home_initial, 1.0)[1]
                                       - A.axial_to_pixel(*a.start, 1.0)[1]) > 0.1]
    if level_ok:
        plain.append("home_higher")
    if not plain and not wind:
        plain.append("own_home_reached")   # practically unreachable: answer would be "yes"
    v = pick(plain, wind)
    if v == "own_home_reached":
        a = _balanced_pick(rng, ants, lambda x: x.outcome == "home")
        options, ans, gt = _yes_no(rng, a.outcome == "home")
        add("q02", 2, v, f"Each home is the ring drawn in the colour of the ant it belongs to. Did the "
            f"{_name(a.symbol)} ant reach its own home? Answer yes or no.", options, ans, gt)
    elif v == "caught_which":
        caught = rng.choice(dead)
        others = [a.symbol for a in ants if a.outcome != "dead"]
        rng.shuffle(others)
        options, ans = _mc(rng, caught.symbol, others[:3], formatter=_name)
        add("q02", 2, v, "Which of these ants was caught by a predator?", options, ans, caught.symbol)
    elif v == "home_moved_whose":
        whose = scenario.home_moved_ant_symbol
        others = [a.symbol for a in ants if a.symbol != whose]
        rng.shuffle(others)
        options, ans = _mc(rng, whose, others[:3], formatter=_name)
        add("q02", 2, v, "One ant's home jumped to a new cell during the video. Whose home was it?",
            options, ans, whose)
    elif v == "home_higher":
        a = _balanced_pick(rng, level_ok, home_higher)
        gt = "higher" if home_higher(a) else "lower"
        options, ans = _mc(rng, gt, ["higher", "lower"])
        add("q02", 2, v, f"In the first frame, was the {_name(a.symbol)} ant's own home (the ring in its colour) "
            "higher or lower on the screen than the ant itself?", options, ans, gt)
    elif v == "home_downwind":
        a = _balanced_pick(rng, side_ok, lambda x: _home_side(x, d0) > 0)
        gt = "downwind" if _home_side(a, d0) > 0 else "upwind"
        options, ans = _mc(rng, gt, ["downwind", "upwind"])
        add("q02", 2, v, WIND_NOTE + f"In the first frame, did the {_name(a.symbol)} ant's own home lie downwind of "
            "it (roughly the way the wind blows) or upwind (roughly the way the wind comes from)?",
            options, ans, gt)
    elif v == "helped_at_start":
        a = _balanced_pick(rng, ants, lambda x: _wind_effect(x, d0) < 0)
        options, ans, gt = _yes_no(rng, _wind_effect(a, d0) < 0)
        add("q02", 2, v, WIND_NOTE + f"In the first frame, would one step in the direction the wind was blowing "
            f"have taken the {_name(a.symbol)} ant closer to its own home (counting hex steps, ignoring black "
            "and grey cells)? Answer yes or no.", options, ans, gt)
    else:
        a = rng.choice(helped)
        others = [x.symbol for x in ants if _wind_effect(x, d0) >= 0]
        rng.shuffle(others)
        options, ans = _mc(rng, a.symbol, others[:3], formatter=_name)
        add("q02", 2, v, WIND_NOTE + "In the first frame, for which of these ants would one step in the direction "
            "the wind was blowing have led closer to its own home (counting hex steps, ignoring black and grey "
            "cells)?", options, ans, a.symbol)

    # ---------------- q03  Level 3: Measurement ----------------
    # not for an ant still walking on the move where the wind shifts: the arrow is turning during that move, so
    # "the direction the arrow was pointing" is ambiguous for it
    blow_cands = [a for a in ants if 4 <= a.steps_taken <= 18
                  and (not shifted or a.steps_taken < scenario.wind_shift_tick)] if windy else []
    # count questions whose answer would almost always be 0 are thinned out, so "0" is not a free guess:
    # hole falls only when the board has grey holes; tile swaps in every swap clip and 1 in 4 of the others
    plain = (["n_caught"] if scenario.n_predators > 0 else []) + (["hole_falls"] if scenario.n_holes > 0 else [])
    plain.append("start_distance")
    if scenario.tile_swap_count > 0 or rng.random() < 0.25:
        plain.append("tile_swaps")
    v = pick(plain,
             (["downwind_moves"] if blow_cands else []) + (["wind_shift_move"] if shifted else [])
             + (["n_gusts"] if windy else []))
    if v == "start_distance":
        a = rng.choice(ants)
        c = A.cube_distance(a.start, a.home_initial)
        options, ans = _mc(rng, c, _numeric_distractors(rng, c, lo=1, spreads=(1, 2, 3)))
        add("q03", 3, v, f"In the first frame, how many cells away from its own home was the {_name(a.symbol)} ant, "
            "i.e. the fewest hex steps from its cell to its home if every cell in between were open?",
            options, ans, {"ant": a.symbol, "distance": c})
    elif v == "n_gusts":
        c = _n_gusts(scenario)
        options, ans = _mc(rng, c, _numeric_distractors(rng, c, lo=0, spreads=(1, 2, 3)))
        add("q03", 3, v, WIND_NOTE + "How many separate gusts were there before the last ant's walk ended? (One "
            "gust can last several moves; count each time the GUST tag lights up. 0 if there were none.)",
            options, ans, c)
    elif v == "wind_shift_move":
        c = scenario.wind_shift_tick
        options, ans = _mc(rng, c, _numeric_distractors(rng, c, lo=1, spreads=(1, 2, 3)))
        add("q03", 3, v, WIND_NOTE + "The ants all move at the same time, one move per step of the clock. The "
            "wind changed direction once. Counting the ants' moves from the start, on which move was the new "
            "wind direction first in effect? (1 means the very first move.)", options, ans, c)
    elif v == "downwind_moves":
        a = rng.choice(blow_cands)
        c = _downwind_moves(scenario, a)
        options, ans = _mc(rng, c, _numeric_distractors(rng, c, lo=0, hi=a.steps_taken, spreads=(1, 2, 3)))
        add("q03", 3, v, WIND_NOTE + f"Count the moves the {_name(a.symbol)} ant made that went exactly one cell "
            "in the direction the wind arrow was pointing at that moment, whether the wind pushed it or it "
            "stepped that way on its own. How many such moves did it make before its walk ended?",
            options, ans, {"ant": a.symbol, "moves": c})
    else:
        if v == "hole_falls":
            c = scenario.hole_reset_count
            prompt = ("How many times in total did an ant step into a grey hole and get sent back to its own "
                      "starting cell? Answer with a number (0 if it never happened).")
        elif v == "tile_swaps":
            c = scenario.tile_swap_count
            prompt = ("How many times did a black (blocked) tile and a white (open) tile swap places? Answer "
                      "with a number (0 if it never happened).")
        else:
            c = len(dead)
            prompt = "How many ants were caught by a predator? Answer with a number (0 if none)."
        options, ans = _mc(rng, c, _numeric_distractors(rng, c, lo=0, spreads=(1, 2, 3)))
        add("q03", 3, v, prompt, options, ans, c)

    # ---------------- q04  Level 4: Multi-step Integration ----------------
    plain = ["total_events"]
    if scenario.hole_reset_count > 0:
        plain.append("setback_then_home")
    if len(home) >= 2 and sorted(a.steps_taken for a in home)[0] < sorted(a.steps_taken for a in home)[1]:
        plain.append("first_home")
    if n >= 2:
        longest = sorted((a.steps_taken for a in ants), reverse=True)
        if longest[0] > longest[1]:
            plain.append("longest_walk")
    wind = []
    if windy and any(_wind_effect(a, d0) < 0 for a in ants):
        wind.append("helped_and_home")
    if windy and any(_wind_effect(a, d0) > 0 for a in ants):
        wind.append("hindered_and_home")
    if windy and len(_gust_pushes(scenario)) <= 12:
        wind.append("gust_pushes")
    v = pick(plain, wind)
    if v == "total_events":
        c = scenario.total_events
        options, ans = _mc(rng, c, _numeric_distractors(rng, c, lo=0, spreads=(1, 2, 3)))
        add("q04", 4, v, "Count each of these as ONE event: a home jumping to a new cell, a black tile and a "
            "white tile swapping places, an ant stepping into a grey hole, an ant being caught by a predator, "
            "and the wind changing direction. How many events happened in total?", options, ans, c)
    elif v == "setback_then_home":
        c = sum(1 for a in home if a.reset_count > 0)
        options, ans = _mc(rng, c, _numeric_distractors(rng, c, lo=0, hi=n, spreads=(1, 2, 3)))
        add("q04", 4, v, "Stepping into a grey hole sends an ant back to its start, but it keeps walking. How "
            "many ants stepped into a hole at least once and STILL reached their own home? Answer with a "
            "number.", options, ans, c)
    elif v == "first_home":
        first = min(home, key=lambda a: a.steps_taken)
        others = [a.symbol for a in ants if a.symbol != first.symbol]
        rng.shuffle(others)
        options, ans = _mc(rng, first.symbol, others[:3], formatter=_name)
        add("q04", 4, v, "Which ant was the first to reach its own home?", options, ans, first.symbol)
    elif v == "gust_pushes":
        c = len(_gust_pushes(scenario))
        options, ans = _mc(rng, c, _numeric_distractors(rng, c, lo=0, spreads=(1, 2, 3)))
        add("q04", 4, v, WIND_NOTE + "Add up over every gust and every ant: how many times in total was an ant "
            "blown one cell downwind during a gust? (An ant blown during a gust that lasts two moves counts "
            "twice. 0 if it never happened.)", options, ans, c)
    elif v == "longest_walk":
        lng = max(ants, key=lambda a: a.steps_taken)
        others = [a.symbol for a in ants if a.symbol != lng.symbol]
        rng.shuffle(others)
        options, ans = _mc(rng, lng.symbol, others[:3], formatter=_name)
        add("q04", 4, v, "Which ant kept walking the longest before its walk ended (by reaching home or by "
            "being caught)?", options, ans, lng.symbol)
    else:
        sign = -1 if v == "helped_and_home" else 1
        c = sum(1 for a in home if _wind_effect(a, d0) == sign)
        options, ans = _mc(rng, c, _numeric_distractors(rng, c, lo=0, spreads=(1, 2, 3)))
        word = "closer to" if sign < 0 else "farther from"
        add("q04", 4, v, WIND_NOTE + "Look at where each ant and its home were in the first frame. Count the ants "
            f"for which one step in the direction the wind was blowing would have taken it {word} its own home "
            "(counting hex steps, ignoring black and grey cells) AND that still reached their own home. "
            "How many are there? Answer with a number (0 if none).", options, ans, c)

    # ---------------- q05  Level 5: Counterfactual Reasoning ----------------
    plain = []
    cands = []
    for a in ants:
        p = _path(scenario, a)
        for N in range(2, min(len(p) - 1, 13)):          # move numbers up to 12 only, so they can be counted
            # every move up to N was a real one-cell hop (no stay-put, no hole bounce back to start)
            if all(A.cube_distance(p[t - 1], p[t]) == 1 for t in range(1, N + 1)):
                cands.append((a, N))
    if cands:
        plain.append("earlier_home")
    if scenario.hole_reset_count > 0:
        plain.append("hole_is_loss")
    moved = next((x for x in ants if x.symbol == scenario.home_moved_ant_symbol), None)
    # the original home cell must stay white all clip, else a tile swap there would make the "what if" ill-posed
    if moved is not None and not any(bh[moved.home_initial] for bh in scenario.blocked_history):
        plain.append("home_not_moved")
    route_cands = [(a, r) for a in ants for r in [_shortest_open_route(scenario, a)] if r is not None and r <= 8]
    if route_cands:
        plain.append("shortest_route")
    if not plain:                       # last resort (rare): every open route runs through a grey hole
        route_cands = [(a, r) for a in ants for r in [_shortest_open_route(scenario, a, holes_open=True)]
                       if r is not None]
        plain.append("shortest_route_holes_open")
    wind = []
    if windy:
        wind.append("gale_moves")
        if any(_gale(scenario, a, d0)[1] for a in ants):     # only ask when "yes" is possible, then 50/50
            wind.append("gale_reaches_home")
    v = pick(plain, wind)
    gale_rule = (WIND_NOTE + "Suppose the wind shown in the first frame never changed and were so strong that "
                 "an ant were blown one cell downwind on EVERY move, on the board exactly as in the first frame "
                 "(no tile swaps, no home jump, no predators). It stops for good when it reaches its own home, or "
                 "when the next cell downwind is black, grey or off the board. ")
    if v == "hole_is_loss":
        c = sum(1 for a in home if a.reset_count == 0)
        options, ans = _mc(rng, c, _numeric_distractors(rng, c, lo=0, hi=n, spreads=(1, 2, 3)))
        add("q05", 5, v, SAME_MOVES + "Suppose stepping into a grey hole had removed an ant for good, like "
            "being caught by a predator, instead of sending it back to its start. How many ants would have "
            "reached their own home?", options, ans, c)
    elif v == "home_not_moved":
        p = _path(scenario, moved)
        yes = any(p[t] == moved.home_initial for t in range(C.HOME_MOVE_TICK, len(p)))
        options, ans, gt = _yes_no(rng, yes)
        add("q05", 5, v, SAME_MOVES + f"Suppose the {_name(moved.symbol)} ant's home had stayed on the cell where "
            "it was at the start instead of jumping. Would the ant have stepped onto that original home cell "
            "before its walk actually ended? Answer yes or no.", options, ans, gt)
    elif v == "earlier_home":
        # the ant's home is moved to the cell it stood on after its N-th move: when would it arrive?
        a, N = rng.choice(cands)
        p = _path(scenario, a)
        c = p.index(p[N])                                          # first time it stood there
        options, ans = _mc(rng, c, _numeric_distractors(rng, c, lo=0, hi=N, spreads=(1, 2, 3)))
        add("q05", 5, v, SAME_MOVES + f"Suppose the {_name(a.symbol)} ant's home had been the cell it stood on "
            f"right after its move number {N}. After how many moves would it have reached home? (0 if it "
            "would already have been home at the start.)", options, ans,
            {"ant": a.symbol, "after_move": N, "moves": c})
    elif v in ("shortest_route", "shortest_route_holes_open"):
        a, c = rng.choice(route_cands)
        options, ans = _mc(rng, c, _numeric_distractors(rng, c, lo=1, spreads=(1, 2, 3)))
        holes_txt = ", and suppose the grey holes were ordinary open cells" if v.endswith("holes_open") else ""
        avoid = "black ones" if v.endswith("holes_open") else "black or grey ones"
        add("q05", 5, v, f"Suppose the {_name(a.symbol)} ant had walked straight home along a shortest route, and "
            "suppose the board stayed exactly as it is in the first frame (no tile swaps, no home jump, no "
            f"predators){holes_txt}. Moving only through open cells, never through {avoid} (a cell tinted as "
            "another ant's home counts as open), how many moves would it have needed to reach its home?",
            options, ans, {"ant": a.symbol, "moves": c})
    elif v == "gale_reaches_home":
        a = _balanced_pick(rng, ants, lambda x: _gale(scenario, x, d0)[1])
        yes = _gale(scenario, a, d0)[1]
        options, ans, gt = _yes_no(rng, yes)
        add("q05", 5, v, gale_rule + f"Starting from its own starting cell, would the {_name(a.symbol)} ant reach "
            "its own home? Answer yes or no.", options, ans, gt)
    else:
        a = rng.choice(ants)
        c, _ = _gale(scenario, a, d0)
        options, ans = _mc(rng, c, _numeric_distractors(rng, c, lo=0, spreads=(1, 2, 3)))
        add("q05", 5, v, gale_rule + f"Starting from its own starting cell, how many moves would the "
            f"{_name(a.symbol)} ant make before it stops? (0 if it could not move at all.)", options, ans,
            {"ant": a.symbol, "moves": c})

    return questions
