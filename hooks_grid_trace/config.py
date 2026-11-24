"""Sampling ranges/constants for hooks_grid_trace."""

FPS = 30
BACKGROUND_HEX = "#F7F3EA"
GRID_LINE_HEX = "#B8B0A0"            # faint, so bold hook boundaries stand out clearly against it
HOOK_LINE_HEX = "#1A1A1A"
TARGET_TEXT_HEX = "#C4423A"          # row/column target labels, per reference
DIGIT_TEXT_HEX = "#1A1A1A"
SHADE_TINT_HEX = "#D9C9A8"           # tan tint for shaded cells
ERROR_FLASH_HEX = "#C4423A"

GRID_SIZE_RANGE = (6, 9)
CANVAS_TARGET_PX = 480               # informational only -- Manim renders in its own units, scaled to fit

LABEL_GAP_FACTOR = 0.62              # label distance from the grid edge, as a multiple of cell_size
MAX_CELL_SIZE = 0.85
GRID_SAFE_HALF_HEIGHT = 3.5          # target half-height BUDGET FOR GRID + LABELS (Manim's default frame
                                      # is 8 units tall, so this leaves real margin -- cell_size_for solves
                                      # for the cell size that keeps column labels safely inside the frame
                                      # at every grid_size; a fixed 7.0/n formula that ignored the label
                                      # margin let 9x9 column labels clip off the top edge -- caught visually)


def cell_size_for(grid_size: int) -> float:
    # the column-label center sits at half_span + cell*LABEL_GAP_FACTOR above the grid; add a further
    # +0.3-cell buffer for the label glyph's own height so the text itself, not just its center, stays in frame
    denom = grid_size / 2.0 + LABEL_GAP_FACTOR + 0.3
    return min(MAX_CELL_SIZE, GRID_SAFE_HALF_HEIGHT / denom)

DURATION_S_RANGE = (3.0, 6.0)        # nominal target; may be gently stretched if grid_size needs more hook steps
FINAL_HOLD_FRAC = 0.20

RUNNING_SUMS_PROBABILITY = 0.5
BROKEN_PROBABILITY = 0.5
SHADE_PROBABILITY = 0.5              # per-cell independent shading, unrelated to hook membership

TASK_NAME = {1: "Image Description", 2: "Basic Relational Reasoning", 3: "Comparative Reasoning",
             4: "Compound Reasoning", 5: "Extended Temporal Reasoning"}
