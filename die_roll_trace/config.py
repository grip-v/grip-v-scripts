"""Sampling ranges/constants for die_roll_trace."""

VALUE_POOL = list(range(1, 10))   # 1-9, six distinct values drawn per die

N_RANGE = (8, 11)                  # grid is N x N
CANVAS_PX_RANGE = (450, 550)

# CHANGE: shorter videos, about 10-13 s (was (15, 40) rolls, 18-54 s).
PATH_LEN_RANGE = (8, 10)           # total_rolls

FPS_RANGE = (30, 30)   # default output fps bumped to 30
# CHANGE: slower rolls (was (0.30, 0.40)). Each roll = REST_S still + tipping.
STEP_DURATION_S_RANGE = (1.00, 1.10)   # seconds per roll (rest + tip); tipping takes 0.5-0.6 s
# CHANGE: new setting. Before every roll the die sits still this long, showing
# its top face: 0.5 s at the start and after every landing.
REST_S = 0.5
FINAL_HOLD_FRAC = 0.18

BROKEN_PROBABILITY = 0.45
# CHANGE: new. Retries allowed when building a path that follows the number rule.
MAX_PATH_ATTEMPTS = 2000

# purely decorative cell palette -- color carries NO mechanical meaning in this
# dataset (see README §8); it only gives the grid visual density matching the
# reference puzzle's look.
CELL_COLOR_PALETTE = [
    "#F2D5C4", "#C9DFC8", "#C7DCEB", "#F4E3B2", "#E4C9E8", "#F0C6C6", "#D8E8D0",
]

BACKGROUND_HEX = "#F7F5EF"
GRID_LINE_HEX = "#B9B2A0"
CELL_TEXT_HEX = "#5A5346"
TRAIL_HEX = "#3E6E8E"
TRAIL_MARK_HEX = "#2C4E66"
# CHANGE: die is now drawn as a real 3D cube with blue faces. Replaces the old
# DIE_BODY_HEX / DIE_TOP_HEX / DIE_SIDE_HEX / DIE_LABEL_HEX cream-coloured icon.
DIE_FACE_HEX = "#23407A"    # deep blue die body (outer bevel of each face)
DIE_INSET_HEX = "#2F5596"   # raised inner panel of each face
DIE_EDGE_HEX = "#0F1D3A"

# CHANGE: new. Right-hand panel: a large digit showing the number on the die's
# top face: the current top while the die sits still, and already the new top
# as soon as it starts tipping (shown before the die lands on that cell).
TOP_DIGIT_HEX = "#23407A"    # same blue as the die
TOP_DIGIT_PX = 300           # font size of the digit, px

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from common.fonts import resolve_font_path as _resolve_font_path

FONT_PATH = _resolve_font_path()  # cross-platform: macOS Arial Bold, or Linux DejaVu/Liberation Bold

# CHANGE: level names now follow the lead's 5-step ladder (were Image Description,
# Basic Relational, Comparative, Compound, Extended Temporal).
TASK_NAME = {1: "Direct Observation", 2: "Relational Reasoning", 3: "Measurement",
             4: "Multi-step Integration", 5: "Counterfactual Reasoning"}
