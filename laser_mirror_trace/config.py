"""Sampling ranges/constants for laser_mirror_trace."""

FPS_RANGE = (30, 30)
CANVAS_PX_RANGE = (720, 720)
GRID_SIZE_RANGE = (8, 14)

# Final output video is always exactly OUTPUT_W x OUTPUT_H (720p). The square
# puzzle content (CANVAS_PX_RANGE, capped at OUTPUT_H) is rendered at its own
# size and pillarboxed (centered, background-filled) onto the output frame so
# the grid itself is never stretched/distorted.
OUTPUT_W = 1280
OUTPUT_H = 720

MIRROR_DENSITY_RANGE = (0.22, 0.32)
NUM_GOALS_RANGE = (3, 6)

STEP_DURATION_S_RANGE = (0.15, 0.25)
FINAL_HOLD_FRAC = 0.18
LOOP_CYCLE_REPEATS = 2.5   # how many full repetitions of the cycle to animate before
                            # fading out on a loop outcome -- caps rendered loop length

MODE_WEIGHTS = {
    "valid_path_reaches_goal": 0.55,
    "valid_path_no_goal_or_loop": 0.25,
    "broken_path": 0.20,
}
MAX_GENERATION_ATTEMPTS = 150

BACKGROUND_HEX = "#F7F5EF"
GRID_LINE_HEX = "#D8D3C4"
MIRROR_HEX = "#3A3A3A"
BEAM_HEX = "#E8A33D"
BEAM_TRAIL_HEX = "#D9B77C"
REFLECTION_FLASH_HEX = "#FFFFFF"
GOAL_ARROW_HEX = "#5B7DB1"
GOAL_SUCCESS_HEX = "#3E8E56"

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from common.fonts import resolve_font_path as _resolve_font_path

FONT_PATH = _resolve_font_path()  # cross-platform: macOS Arial Bold, or Linux DejaVu/Liberation Bold

TASK_NAME = {1: "Image Description", 2: "Basic Relational Reasoning", 3: "Comparative Reasoning",
             4: "Compound Reasoning", 5: "Extended Temporal Reasoning"}
