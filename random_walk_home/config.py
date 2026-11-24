"""Sampling ranges/constants for random_walk_home.

Self-contained preset (own config/arena/simulate/questions/scene/generate,
same convention as bouncing_ball_restitution's presets): 1-5 ants, each a
distinct card-suit-style shape, each starting AWAY from its own home and
having to find its way back, on a hex board that can grow hazards, moving
homes, mutating walkable tiles, and roaming predators -- each an
independent per-clip randomized layer, not all-or-nothing.
"""

FPS = 30
RESOLUTION = (1280, 720)
BACKGROUND_HEX = "#F1EFE8"

ARENA_RADIUS = 4                # axial hex-distance radius of the whole board (tuned so the whole board,
                                 # not just a walk-bounded sub-patch, fits the frame)
CELL_SIZE = 0.5
BLACK_FRACTION = 0.25           # fraction of board cells blocked (impassable) at t=0

N_ANTS_RANGE = (1, 5)
SYMBOLS = ("spade", "heart", "diamond", "club", "star")
SYMBOL_GLYPH = {"spade": "♠", "heart": "♥", "diamond": "♦", "club": "♣", "star": "★"}
HOME_COLOR_PALETTE = ("#3FA79A", "#C4607A", "#D9A441", "#4F82C4", "#8A63C4")   # one per SYMBOLS index

MIN_TICKS = 10                  # CHANGE: a scenario that resolves in fewer ticks is rejected and re-drawn the
                                 # same way as MAX_TICKS (was 3): a walk this short cannot fill the 10 s minimum
                                 # clip length below without crawling (10 ticks -> ~0.78 s per move at most)
MAX_TICKS = 70                  # a scenario that hasn't resolved (every ant home or dead) by this tick is
                                 # rejected and re-drawn from the same rng stream (see simulate.py)

# --- independent per-clip optional layers ---
HOME_MOVE_PROBABILITY = 0.25
HOME_MOVE_TICK = 22             # absolute tick at which one still-active ant's home relocates (no-op if
                                 # every ant has already finished by then)
HOME_MOVE_DIST_RANGE = (2, 4)   # hex-hop distance of the new home from the old one

TILE_SWAP_PROBABILITY = 0.3
TILE_SWAP_TICKS = (14, 30, 46)  # absolute ticks at which one black<->white pair may flip (each independently
                                 # skipped if no safe candidate pair exists at that tick)

HOLE_PROBABILITY = 0.3
N_HOLES_RANGE = (1, 2)          # grey "hole" cells, fixed for the whole clip: an ant stepping onto one is
                                 # teleported back to its OWN start cell (not home) and continues from there

PREDATOR_PROBABILITY = 0.3
N_PREDATORS_RANGE = (1, 1)      # red-ball predator(s) wandering the board; an ant sharing a cell with one
                                 # after a tick dies (vanishes) permanently -- no reset, unlike a hole. On this
                                 # board size even a single predator is quite lethal over a long enough walk
                                 # (measured ~55-60% per-ant death rate when a predator is present) -- see README

# CHANGE: wind layer (replaces the earlier homing-pull idea). Per clip: calm, or a light / strong wind blowing
# toward one of the 6 hex directions. The wind comes in GUSTS: each move is either a gust or a lull (a 2-state
# Markov chain per clip). During a gust EVERY walking ant whose next cell downwind is on the board and open is
# blown one cell downwind, all at the same moment (a grey hole there still swallows it); an ant whose downwind
# cell is blocked, and every ant during a lull, takes its usual random step. Predators are never blown. A strong
# wind gusts more often and for longer. In some windy clips the wind shifts once to a new direction at the start
# of a random move (never a gust on that move). On screen: a gauge (arrow + 1/2 chevrons, or CALM), a GUST tag
# while a gust blows, and streaks that drift slowly in a lull and race in a gust.
WIND_PROBABILITY = 0.62                   # after re-draws of walks that never finish, ~40% of clips are calm
WIND_GUST = {"light": (0.20, 0.50),       # (P(gust starts on a lull move), P(gust continues)) -> ~29% of moves
             "strong": (0.30, 0.65)}      # -> ~46% of moves, gusts ~3 moves long
WIND_SHIFT_PROBABILITY = 0.4              # given wind
WIND_SHIFT_TICK_RANGE = (8, 20)           # tick at whose start the wind changes direction
# names of the 6 directions on screen, in arena.HEX_NEIGHBOR_OFFSETS order (where the wind blows TOWARD)
WIND_DIR_NAMES = ("up-right", "down-right", "down", "down-left", "up-left", "up")

TICK_DURATION_RANGE = (0.35, 0.55)
MAX_VIDEO_DURATION_S = 15.0     # CHANGE: was 14.0; every clip is now 10-15 s long
MIN_VIDEO_DURATION_S = 10.0     # CHANGE: short walks are played at a slower steady pace to reach this
SPEEDUP_AFTER_S = 9.0          # CHANGE: no longer used -- a long walk now plays at ONE steady (faster) pace
                               # chosen so it fits, instead of switching to ~6x speed after 9 s
HOME_FLASH_DURATION_S = 0.4
DEATH_FLASH_DURATION_S = 0.35
FINAL_HOLD_FRAC = 0.12

# CHANGE: level names follow the 5-step closed-loop ladder (see questions.py)
TASK_NAME = {1: "Direct Observation", 2: "Relational Reasoning", 3: "Measurement",
             4: "Multi-step Integration", 5: "Counterfactual Reasoning"}
