# CHANGE: this file was rewritten. Grid, trail, frame timing and final frame
# work as before; the die drawing and the frame layout are new. Each new part
# is marked with a "CHANGE:" comment below.
"""PIL + OpenCV raster rendering for one die_roll_trace clip.

Plain per-frame raster pipeline (no Manim). Each 1280x720 frame has two parts:

- LEFT (720x720, drawn directly at output size): the N x N board of
  decorative colored/numbered cells, a small 3D die that physically tips
  from cell to cell, and a persistent trail over visited cells.
- RIGHT (560x720): a large digit showing the number on the die's top face.
  It shows the current top while the die sits still, and switches to the new
  top as soon as the die starts tipping, before it lands.

The board die is drawn by a cube renderer from the die's full 3D rotation,
derived from the orientation dict (a cube rotation is fully determined by
where its faces are). So each face keeps its in-plane twist across rolls,
exactly like a real die -- nothing re-orients or "snaps" on landing.

All puzzle logic (die identity, roll sequence, content_mode) is already
finished before this module is ever called.

Speed: the grid is painted once per clip into a static background; each
frame only copies it, draws the trail, and pastes the two die sprites,
which are supersampled only inside their own small boxes and cached by
pose (landed poses repeat constantly).
"""
from __future__ import annotations
import math

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import config as C
import die as D
import puzzle as P

OUTPUT_W, OUTPUT_H = 1280, 720
# CHANGE: the frame is split: board on the left, big die panel on the right.
# Before, the board was centred with empty bars on both sides.
BOARD_PX = OUTPUT_H                      # board occupies the left 720x720
PANEL_W = OUTPUT_W - BOARD_PX            # digit panel occupies the right 560x720
# CHANGE: new settings for drawing the 3D dice.
SS = 2                                   # die supersampling (anti-aliasing)
BOARD_DIE_EDGE_FRAC = 0.5                # small die edge length, in cells
# CHANGE: new. Grid number size and position, as fractions of a cell.
NUMBER_FONT_FRAC = 0.40
NUMBER_OFFSET = (0.07, 0.03)
# CHANGE: new. The board die sits this far (in cells) right of the cell centre,
# so the number in the cell's top-left corner stays visible next to it (about a
# 3 px gap). The die then spans x 0.33-1.01 of its cell; the number ends at
# x 0.30, and the next cell's number starts at x 1.07.
BOARD_DIE_OFFSET = (0.17, 0.0)


def _font(size):
    return ImageFont.truetype(C.FONT_PATH, size)


def _cell_px(canvas_px, N):
    return canvas_px / N


def _cell_center(row, col, cell_px):
    return (col * cell_px + cell_px / 2, row * cell_px + cell_px / 2)


def _hex_rgb(h):
    return tuple(int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))


def _draw_grid(d: ImageDraw.ImageDraw, scene: P.SceneData, cell_px, small_font):
    N = scene.N
    for row in range(N):
        for col in range(N):
            x0, y0 = col * cell_px, row * cell_px
            x1, y1 = x0 + cell_px, y0 + cell_px
            color = C.CELL_COLOR_PALETTE[scene.grid_color_idx[row][col]]
            d.rectangle([x0, y0, x1, y1], fill=color, outline=C.GRID_LINE_HEX, width=1)
            # CHANGE: numbers are bigger and sit closer to the corner (was
            # offset 0.14/0.12 and font 0.16 of a cell). They now decide where
            # the die goes, so a model must be able to read them even on
            # downscaled frames. They stay clear of the trail through the centre.
            d.text((x0 + cell_px * NUMBER_OFFSET[0], y0 + cell_px * NUMBER_OFFSET[1]), str(scene.grid_values[row][col]),
                   font=small_font, fill=C.CELL_TEXT_HEX)


def _draw_trail(d: ImageDraw.ImageDraw, cells_so_far, cell_px, radius):
    if len(cells_so_far) < 2:
        return
    pts = [_cell_center(r, c, cell_px) for r, c in cells_so_far]
    d.line(pts, fill=C.TRAIL_HEX, width=max(2, int(radius * 0.22)), joint="curve")
    for (r, c) in cells_so_far[:-1]:
        x, y = _cell_center(r, c, cell_px)
        m = radius * 0.14
        d.ellipse([x - m, y - m, x + m, y + m], fill=C.TRAIL_MARK_HEX)


def _lerp(p0, p1, t):
    return (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)


# CHANGE: everything from here to _Painter is new. It replaces the old flat
# die icon, which slid between cells and showed its top value as a digit.
# ---------------------------------------------------------------------------
# 3D cube renderer (used for both the board die and the big panel die)
# ---------------------------------------------------------------------------
# World axes: +x = east, +y = north, +z = up. Unit cube centred at origin.
# Each orientation slot -> (face centre, u axis, v axis) at rest; a point
# (a, b) in [0,1]^2 on the face maps to centre + (a-.5)u + (b-.5)v.
_FACE_FRAMES = {
    "top":    ((0, 0, .5),  (1, 0, 0),  (0, -1, 0)),
    "bottom": ((0, 0, -.5), (1, 0, 0),  (0, 1, 0)),
    "south":  ((0, -.5, 0), (1, 0, 0),  (0, 0, -1)),
    "north":  ((0, .5, 0),  (-1, 0, 0), (0, 0, -1)),
    "east":   ((.5, 0, 0),  (0, 1, 0),  (0, 0, -1)),
    "west":   ((-.5, 0, 0), (0, -1, 0), (0, 0, -1)),
}
_SLOT_NORMAL = {s: 2 * np.asarray(c, float) for s, (c, u, v) in _FACE_FRAMES.items()}

# Tipping over an edge: rotation axis and the bottom edge acting as pivot.
# Checked against die.roll(): e.g. east at 90 deg carries top->east, west->top.
_ROLL_AXIS = {
    "east":  ((0, 1, 0),  (.5, 0, -.5)),
    "west":  ((0, -1, 0), (-.5, 0, -.5)),
    "north": ((-1, 0, 0), (0, .5, -.5)),
    "south": ((1, 0, 0),  (0, -.5, -.5)),
}
_DIR_VEC = {"east": (1, 0, 0), "west": (-1, 0, 0), "north": (0, 1, 0), "south": (0, -1, 0)}

# Camera: looking from above-south-east, so top, south and east faces show.
_CAM_DIR = np.array([0.55, -1.0, 0.85]) / np.linalg.norm([0.55, -1.0, 0.85])
_CAM_RIGHT = np.cross([0, 0, 1], _CAM_DIR)
_CAM_RIGHT /= np.linalg.norm(_CAM_RIGHT)
_CAM_UP = np.cross(_CAM_DIR, _CAM_RIGHT)
_LIGHT = np.array([0.3, -0.5, 1.0]) / np.linalg.norm([0.3, -0.5, 1.0])

# CHANGE: new. The die now tips over its bottom edge instead of sliding.
def _roll_pose(direction, t):
    """(3x3 rotation, translation) placing the unit cube tipped t*90 deg over
    its leading bottom edge, shifted back so it rolls in place."""
    if direction is None or t <= 0:
        return np.eye(3), np.zeros(3)
    axis, pivot = _ROLL_AXIS[direction]
    k = np.asarray(axis, float)
    pivot = np.asarray(pivot, float)
    a = math.radians(90 * min(t, 1.0))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    Rm = math.cos(a) * np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * np.outer(k, k)
    trans = pivot - Rm @ pivot - np.asarray(_DIR_VEC[direction], float) * min(t, 1.0)
    return Rm, trans


# CHANGE: new. Carries the die's full 3D rotation from roll to roll, so faces
# keep their pattern and angle and do not jump when the die lands.
def _state_rotation(orientation, initial):
    """Rotation taking the die from `initial` to `orientation`: body axis
    x/y/z (the faces that started east/north/top) -> wherever those faces'
    values sit now. Unique, because a cube rotation is fixed by its faces."""
    where = {val: slot for slot, val in orientation.items()}
    return np.column_stack([_SLOT_NORMAL[where[initial[s]]] for s in ("east", "north", "top")])


def _unit_box(shadow):
    """Projected extent, in units of cube edge length relative to the resting
    cube centre, of every rolling pose (and the floor shadow, if drawn)."""
    pts = [np.array([x, y, -.5]) for x in (-.6, .6) for y in (-.6, .6)] if shadow else []
    corners = [np.array([x, y, z], float) for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]
    pts += corners
    for d in _ROLL_AXIS:
        for t in np.linspace(0, 1, 46):
            Rm, tr = _roll_pose(d, t)
            pts += [Rm @ c + tr for c in corners]
    xs = [float(p @ _CAM_RIGHT) for p in pts]
    ys = [-float(p @ _CAM_UP) for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


_UNIT_BOX = {True: _unit_box(True), False: _unit_box(False)}


# CHANGE: new. Draws one die (used for both the small and the big die).
class _CubeSprite:
    """Renders the die at one size as a transparent RGBA sprite, cached by
    pose. `anchor` = sprite pixel that sits on the resting cube's centre."""

    # CHANGE: the board die is drawn plain blue (no digits, no shapes).
    def __init__(self, initial, edge_px, shadow=False):
        self.initial = initial
        self.scale = edge_px
        self.shadow = shadow
        x0, y0, x1, y1 = _UNIT_BOX[shadow]
        pad = 3
        self.anchor = (int(math.ceil(-x0 * edge_px)) + pad, int(math.ceil(-y0 * edge_px)) + pad)
        self.size = (self.anchor[0] + int(math.ceil(x1 * edge_px)) + pad,
                     self.anchor[1] + int(math.ceil(y1 * edge_px)) + pad)
        # each value's face frame in the die's own (body) coordinates
        self.body = {initial[s]: tuple(np.asarray(a, float) for a in f) for s, f in _FACE_FRAMES.items()}
        self._cache = {}

    def get(self, orientation, direction=None, t=1.0):
        if direction is None or t >= 1.0:
            direction, t = None, 1.0
        key = (D.orientation_key(orientation), direction, round(t, 5))
        img = self._cache.get(key)
        if img is None:
            img = self._cache[key] = self._draw(orientation, direction, t)
        return img

    def _draw(self, orientation, direction, t):
        W, H = self.size[0] * SS, self.size[1] * SS
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        scale = self.scale * SS
        ox, oy = self.anchor[0] * SS, self.anchor[1] * SS
        Rroll, trans = _roll_pose(direction, t)
        M = Rroll @ _state_rotation(orientation, self.initial)

        def project(p):
            return (ox + scale * float(p @ _CAM_RIGHT), oy - scale * float(p @ _CAM_UP))

        face_rgb = np.array(_hex_rgb(C.DIE_FACE_HEX), float)
        inset_rgb = np.array(_hex_rgb(C.DIE_INSET_HEX), float)
        edge_w = max(1, round(self.scale * SS * 0.012))
        for value, (centre, u, v) in self.body.items():
            c3 = M @ centre + trans
            u3, v3 = M @ u, M @ v
            normal = M @ (2 * centre)
            if float(normal @ _CAM_DIR) <= 1e-6:
                continue                                 # back face

            # face-local (a, b) in [0,1]^2 -> screen, as one affine map
            o2 = np.array(project(c3 - .5 * u3 - .5 * v3))
            du = np.array(project(c3 + .5 * u3 - .5 * v3)) - o2
            dv = np.array(project(c3 - .5 * u3 + .5 * v3)) - o2

            def face_pt(a, b):
                q = o2 + a * du + b * dv
                return (float(q[0]), float(q[1]))

            shade = 0.62 + 0.38 * max(0.0, float(normal @ _LIGHT))
            d.polygon([face_pt(0, 0), face_pt(1, 0), face_pt(1, 1), face_pt(0, 1)],
                      fill=tuple(int(min(255, x * shade)) for x in face_rgb),
                      outline=C.DIE_EDGE_HEX, width=edge_w)
            e = 0.07   # raised inner panel -> bevelled-edge look
            d.polygon([face_pt(e, e), face_pt(1 - e, e), face_pt(1 - e, 1 - e), face_pt(e, 1 - e)],
                      fill=tuple(int(min(255, x * shade)) for x in inset_rgb))

        return img.reduce(SS)                            # box-filter downsample: cheap AA


# CHANGE: new. Paints the grid once per clip, then each frame only adds the
# trail and the two dice (faster than redrawing everything every frame).
class _Painter:
    """Per-clip drawing state: static background (grid painted once), the
    board die sprite, and the top-number digit in the side panel."""

    def __init__(self, scene: P.SceneData):
        self.cell_px = _cell_px(BOARD_PX, scene.N)
        self.radius = self.cell_px * 0.28
        small_font = _font(max(6, int(self.cell_px * NUMBER_FONT_FRAC)))   # CHANGE: was 0.16
        self.board_die = _CubeSprite(scene.initial_orientation,
                                     self.cell_px * BOARD_DIE_EDGE_FRAC)   # CHANGE: plain blue board die
        self.panel_centre = (BOARD_PX + PANEL_W / 2, OUTPUT_H / 2)
        self.digit_font = _font(C.TOP_DIGIT_PX)          # CHANGE: top-number digit

        bg = Image.new("RGB", (OUTPUT_W, OUTPUT_H), C.BACKGROUND_HEX)
        d = ImageDraw.Draw(bg)
        _draw_grid(d, scene, self.cell_px, small_font)
        d.line([(BOARD_PX, 0), (BOARD_PX, OUTPUT_H)], fill=C.GRID_LINE_HEX, width=2)
        self.background = bg

    @staticmethod
    def _paste(img, sprite, sprite_img, centre):
        x = int(round(centre[0] - sprite.anchor[0]))
        y = int(round(centre[1] - sprite.anchor[1]))
        img.paste(sprite_img, (x, y), sprite_img)

    def frame(self, trail_cells, cell_pos, orientation, direction=None, t=1.0, panel=None):
        """cell_pos: (row, col), fractional mid-roll. `orientation` is the
        die's state at the start of the motion (or its resting state).
        Returns a BGR frame."""
        img = self.background.copy()
        d = ImageDraw.Draw(img)
        _draw_trail(d, trail_cells, self.cell_px, self.radius)
        board_centre = _cell_center(cell_pos[0], cell_pos[1], self.cell_px)
        board_centre = (board_centre[0] + BOARD_DIE_OFFSET[0] * self.cell_px,   # CHANGE: off-centre
                        board_centre[1] + BOARD_DIE_OFFSET[1] * self.cell_px)
        self._paste(img, self.board_die, self.board_die.get(orientation, direction, t), board_centre)
        # CHANGE: right panel shows the top number as a digit: the die's current
        # top while it sits still, and already the NEW top while it tips (so the
        # number is shown before the die lands on it). `panel` overrides it.
        d.text(self.panel_centre, str(orientation["top"] if panel is None else panel),
               font=self.digit_font, anchor="mm", fill=C.TOP_DIGIT_HEX)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def render_clip(scene: P.SceneData, fps, step_duration_s, out_path):
    pt = _Painter(scene)

    step_frames = max(2, round(step_duration_s * fps))
    n_steps = scene.total_rolls
    move_frames = step_frames * max(n_steps, 1)
    hold_frames = max(1, round(move_frames * C.FINAL_HOLD_FRAC / (1 - C.FINAL_HOLD_FRAC)))
    total_frames = move_frames + hold_frames

    path = [tuple(rc) for rc in scene.path_cells]

    # CHANGE: helper for the new tip-then-rest animation.
    def orientation_before(step_idx):
        return scene.initial_orientation if step_idx == 0 else \
            scene.displayed_roll_sequence[step_idx - 1]["resulting_orientation"]

    def frame_for_step(step_idx, t):
        """step_idx: which roll is currently animating (0-indexed); t in
        [0,1) is interpolation fraction toward the landing cell; t==1 is
        used for the fully-landed still frame."""
        trail = path[:step_idx + 1] if t < 1.0 else path[:step_idx + 2]
        pos = _lerp(path[step_idx], path[step_idx + 1], min(t, 1.0))
        step = scene.displayed_roll_sequence[step_idx]
        if t <= 0.0:                                     # sitting still before the roll
            return pt.frame(trail, pos, orientation_before(step_idx), step["direction"], t)
        if t < 0.999:                                    # CHANGE: tipping: panel already shows the new top
            return pt.frame(trail, pos, orientation_before(step_idx), step["direction"], t,
                            panel=step["resulting_orientation"]["top"])
        return pt.frame(trail, pos, step["resulting_orientation"])

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUTPUT_W, OUTPUT_H))

    if n_steps == 0:
        f = pt.frame(path[:1], path[0], scene.initial_orientation)
        for _ in range(total_frames):
            writer.write(f)
        writer.release()
        return {"duration_seconds": round(total_frames / fps, 3), "fps": fps, "frame_count": total_frames}

    # CHANGE: each step = the die sitting still for REST_S (showing its current
    # top: at the start, and right after every landing), then tipping over.
    # The last tipping frame is the landed pose. Before, the die moved for the
    # whole step with no pause.
    rest_frames = min(step_frames - 1, round(C.REST_S * fps))
    move_frames = step_frames - rest_frames
    for step_idx in range(n_steps):
        still = frame_for_step(step_idx, 0.0)
        for _ in range(rest_frames):
            writer.write(still)
        for fi in range(move_frames):
            writer.write(frame_for_step(step_idx, (fi + 1) / move_frames))
    last = frame_for_step(n_steps - 1, 1.0)
    for _ in range(hold_frames):
        writer.write(last)
    writer.release()
    return {"duration_seconds": round(total_frames / fps, 3), "fps": fps, "frame_count": total_frames}


def extract_final_frame(scene: P.SceneData, out_png):
    pt = _Painter(scene)
    path = [tuple(rc) for rc in scene.path_cells]
    final_orientation = scene.displayed_roll_sequence[-1]["resulting_orientation"] \
        if scene.total_rolls else scene.initial_orientation
    cv2.imwrite(str(out_png), pt.frame(path, path[-1], final_orientation))
