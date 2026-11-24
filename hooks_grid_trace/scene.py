"""Manim scene for one hooks_grid_trace clip.

Two-pass, same convention as the rest of this suite: hooks.py builds the
ENTIRE puzzle (hook geometry, fill order, error injection, every derived
statistic) in pure Python first; this scene only draws the static grid
structure once (including the nested hook-boundary squares, visible from
frame 1) and animates each hook's cells fading in together, in the
precomputed fill order. No puzzle logic happens inside Manim.

Reads its parameters from the JSON file named by the HGT_PARAMS_FILE env
var (same env-var-driven convention as every other scene in this suite).
"""
from __future__ import annotations
import json
import os

from manim import Scene, Square, Text, VGroup, AnimationGroup, FadeIn, FadeOut

import config as C


class HooksGridScene(Scene):
    def construct(self):
        params_path = os.environ["HGT_PARAMS_FILE"]
        with open(params_path, encoding="utf8") as f:
            p = json.load(f)

        self.camera.background_color = p["background_hex"]
        n = p["grid_size"]
        cell = p["cell_size"]
        half_span = n * cell / 2.0
        top_left_x, top_left_y = -half_span, half_span

        def cell_xy(r, c):
            return (top_left_x + (c + 0.5) * cell, top_left_y - (r + 0.5) * cell)

        # --- static: cell backgrounds (shaded tint or plain), thin grid lines ---
        for r in range(n):
            for c in range(n):
                x, y = cell_xy(r, c)
                shaded = [r, c] in p["shaded_cells"]
                sq = Square(side_length=cell, stroke_color=p["grid_line_hex"], stroke_width=0.8,
                            fill_color=p["shade_hex"] if shaded else p["background_hex"], fill_opacity=1.0)
                sq.move_to((x, y, 0))
                self.add(sq)

        # --- static: bold nested hook-boundary squares (hook k's outer edge ==
        # the boundary of the k x k sub-block from the corner), all visible from frame 1 ---
        for k in range(1, n + 1):
            side = k * cell
            outline = Square(side_length=side, stroke_color=p["hook_line_hex"], stroke_width=5.0, fill_opacity=0)
            cx = top_left_x + side / 2.0
            cy = top_left_y - side / 2.0
            outline.move_to((cx, cy, 0))
            self.add(outline)

        # --- static: row/column target labels ---
        label_gap = cell * C.LABEL_GAP_FACTOR
        for r in range(n):
            _, y = cell_xy(r, 0)
            t = Text(str(p["row_targets"][r]), color=p["target_hex"], font_size=int(cell * 24))
            t.move_to((top_left_x - label_gap, y, 0))
            self.add(t)
        for c in range(n):
            x, _ = cell_xy(0, c)
            t = Text(str(p["column_targets"][c]), color=p["target_hex"], font_size=int(cell * 24))
            t.move_to((x, top_left_y + label_gap, 0))
            self.add(t)

        # --- digits, grouped by hook, added in fill_order ---
        digit_mobs_by_hook = {}
        for k, cells in p["hooks"].items():
            k = int(k)
            group = VGroup()
            for (r, c) in cells:
                x, y = cell_xy(r, c)
                digit = p["final_grid"][r][c]
                # deliberately the SAME color as every other digit, including any error cell --
                # the error must be spotted because its VALUE clashes with its hook-mates, never
                # from a color cue (a color-coded error would just be "find the red pixel")
                txt = Text(str(digit), color=p["digit_hex"], font_size=int(cell * 30))
                txt.move_to((x, y, 0))
                group.add(txt)
            digit_mobs_by_hook[k] = group

        step_s = p["step_duration_s"]
        elapsed = 0.0   # tracks actual play()/wait() time spent so far, so the final hold below can
                         # compensate exactly for the running-sums overlay's extra time when it's shown
        for k in p["fill_order"]:
            hook_group = digit_mobs_by_hook[k]
            self.play(AnimationGroup(*[FadeIn(m, scale=1.3) for m in hook_group]), run_time=step_s)
            elapsed += step_s

            if p["running_sums_displayed"]:
                touched_rows = sorted({r for (r, c) in p["hooks"][str(k)]})
                touched_cols = sorted({c for (r, c) in p["hooks"][str(k)]})
                overlay = VGroup()
                for r in touched_rows:
                    x, y = cell_xy(r, n - 1)
                    running = p["row_running_sums"][str(k)][str(r)]
                    lbl = Text(str(running), color=p["target_hex"], font_size=int(cell * 18))
                    lbl.move_to((x + cell * 0.9, y, 0))
                    overlay.add(lbl)
                for c in touched_cols:
                    x, y = cell_xy(n - 1, c)
                    running = p["col_running_sums"][str(k)][str(c)]
                    lbl = Text(str(running), color=p["target_hex"], font_size=int(cell * 18))
                    lbl.move_to((x, y - cell * 0.9, 0))
                    overlay.add(lbl)
                self.play(FadeIn(overlay), run_time=0.18)
                self.play(FadeOut(overlay), run_time=0.32)
                elapsed += 0.5

        # --- final hold: the completed grid and its ORIGINAL target labels only -- no computed
        # sums are ever displayed. Revealing the actual row/column totals here would let sum-
        # related questions be answered by reading two printed numbers instead of actually adding
        # up the visible digits, which defeats the point of testing running-sum arithmetic ---
        hold_s = max(0.0, p["total_duration_s"] - elapsed)
        if hold_s > 0:
            self.wait(hold_s)
