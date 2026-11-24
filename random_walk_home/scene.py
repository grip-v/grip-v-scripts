"""Manim scene for one random_walk_home clip.

Two-pass, same convention as the rest of this suite: simulate.py runs
the ENTIRE multi-ant scenario (every tick, every event) in pure Python
first; generate.py converts it to pixel positions and a per-tick timing
schedule; this scene only ever reads that back through simple
deterministic functions of an animation clock.

Reads its parameters from the JSON file named by the MAG_PARAMS_FILE env
var (same env-var-driven convention as every other scene in this suite).
"""
from __future__ import annotations
import bisect
import json
import math
import os

import random

from manim import (Scene, RegularPolygon, Dot, Text, ValueTracker, Flash, AnimationGroup, linear, smooth,
                   Circle, Line, Arrow, VGroup, VMobject, RoundedRectangle, always_redraw)

BLACK_FILL = "#1A1A1A"
WHITE_FILL = "#FFFFFF"
GREY_FILL = "#9A9A9A"
PREDATOR_COLOR = "#C4423A"
CELL_STROKE = "#333333"
DEATH_FLASH_COLOR = "#B03A3A"
HOME_FLASH_DURATION_S = 0.4
DEATH_FLASH_DURATION_S = 0.35
# CHANGE: wind visuals
WIND_COLOR = "#24476B"            # gauge arrow + chevrons
STREAK_COLOR = "#9ED2E6"          # air streaks over the board (kept far from every ant colour)
GAUGE_CENTER = (-5.3, 2.4)
GAUGE_RADIUS = 0.95
STREAK_SPEED = {"light": 0.45, "strong": 0.8}  # scene units / s, air drift in a lull
STREAK_GUST_SPEED = 3.2                         # ... and in a gust
GUST_TAG_COLOR = "#24476B"
N_STREAKS = 34
STREAK_BOX = (-3.5, 3.5, -3.95, 3.95)          # xmin, xmax, ymin, ymax (the board area)


class MultiAgentWalkScene(Scene):
    def construct(self):
        params_path = os.environ["MAG_PARAMS_FILE"]
        with open(params_path, encoding="utf8") as f:
            p = json.load(f)

        self.camera.background_color = p["background_hex"]
        cum_times = p["cum_times"]
        move_duration = cum_times[-1] if cum_times else 0.0
        total_duration = p["total_duration_s"]
        clock = ValueTracker(0.0)

        # CHANGE: re-timed so move k plays over [cum[k-2], cum[k-1]) (cum[-1] := 0). Before, move k played over
        # [cum[k-1], cum[k]): the first slot was idle and the clip's LAST move had no slot at all, so it (and a
        # catch on the last tick) snapped instantly at the end. Total video length is unchanged.
        n_moves = len(cum_times)

        def tick_at(t):
            """The tick whose move is playing at time t (1..n_moves); n_moves once the walk is over. Tick k's
            board state (tile swap, home jump) is shown from the start of move k -- the simulator applies those
            before the ants step."""
            if n_moves == 0:
                return 0
            return min(n_moves, bisect.bisect_right(cum_times, t) + 1)

        def seg(tick_idx):
            return (cum_times[tick_idx - 2] if tick_idx >= 2 else 0.0), cum_times[tick_idx - 1]

        def walk_end_time(resolved_tick):
            """CHANGE: time at which an ant resolved on `resolved_tick` finishes that move."""
            return seg(resolved_tick)[1]

        def frac_within_tick(t, tick_idx):
            if tick_idx == 0:
                return 1.0
            seg_start, seg_end = seg(tick_idx)
            if seg_end <= seg_start:
                return 1.0
            return smooth(min(1.0, max(0.0, (t - seg_start) / (seg_end - seg_start))))

        # --- static-position board cells, color changes over time (tile swaps) ---
        blocked_by_tick = p["blocked_by_tick"]     # [tick][cell_idx] -> bool
        is_hole = p["is_hole"]                     # [cell_idx] -> bool (fixed for the whole clip)
        n_ticks_total = len(blocked_by_tick) - 1

        for i, cell in enumerate(p["cells"]):
            hexagon = RegularPolygon(n=6, radius=cell["size"], color=CELL_STROKE,
                                      fill_color=WHITE_FILL, fill_opacity=1.0, stroke_width=1.0)
            hexagon.move_to((cell["pos"][0], cell["pos"][1], 0))

            def make_cell_updater(idx=i):
                def updater(mobj):
                    tick = min(n_ticks_total, tick_at(clock.get_value()))
                    if blocked_by_tick[tick][idx]:
                        mobj.set_fill(BLACK_FILL)
                    elif is_hole[idx]:
                        mobj.set_fill(GREY_FILL)
                    else:
                        mobj.set_fill(WHITE_FILL)
                return updater
            hexagon.add_updater(make_cell_updater())
            self.add(hexagon)

        # --- CHANGE: wind -- streaks drifting over the board + a gauge (arrow, 1/2 chevrons, or CALM) ---
        wind_dirs = p["wind_dirs"]                 # [tick] -> direction index or None
        wind_strength = p["wind_strength"]
        unit = []
        for vx, vy in p["wind_vectors"]:
            m = math.hypot(vx, vy)
            unit.append((vx / m, vy / m))

        def wind_dir_at(t):
            return wind_dirs[min(len(wind_dirs) - 1, tick_at(t))] if t < move_duration else wind_dirs[-1]

        def wind_angle_at(t):
            """arrow angle; on a shift it turns over the first 0.3 s (or 40% of the move) of the new move"""
            d = wind_dir_at(t)
            ang = math.atan2(unit[d][1], unit[d][0])
            k = min(len(wind_dirs) - 1, tick_at(t))
            if t < move_duration and k >= 1 and wind_dirs[k - 1] is not None and wind_dirs[k - 1] != d:
                s0, s1 = seg(k)
                f = smooth(min(1.0, (t - s0) / max(1e-6, min(0.3, 0.4 * (s1 - s0)))))
                a0 = math.atan2(unit[wind_dirs[k - 1]][1], unit[wind_dirs[k - 1]][0])
                diff = (ang - a0 + math.pi) % (2 * math.pi) - math.pi
                ang = a0 + diff * f
            return ang

        gusts = p["gusts"]                          # CHANGE: [tick] -> bool, move k is a gust

        def gust_level(t):
            """0 in a lull, 1 in a gust; eases in over the first 20% of a gust's first move and out over the last
            20% of its last move, so gusts swell and die away instead of switching on and off"""
            if t >= move_duration or not gusts:
                return 0.0
            k = min(len(gusts) - 1, tick_at(t))
            if not gusts[k]:
                return 0.0
            s0, s1 = seg(k)
            u = (t - s0) / max(1e-6, s1 - s0)
            lvl = 1.0
            if not gusts[k - 1]:
                lvl = min(lvl, smooth(min(1.0, u / 0.2)))
            if k + 1 >= len(gusts) or not gusts[k + 1]:
                lvl = min(lvl, smooth(min(1.0, (1.0 - u) / 0.2)))
            return lvl

        if wind_strength is not None:
            lull_speed, gust_speed = STREAK_SPEED[wind_strength], STREAK_GUST_SPEED

            def air_speed(t):
                return lull_speed + (gust_speed - lull_speed) * gust_level(t)

            # air displacement = integral of speed along the (possibly turning) wind direction, sampled finely
            DT = 1.0 / 120
            samples = [(0.0, 0.0)]
            n_s = int((move_duration + 30.0) / DT) + 2
            x = y = 0.0
            for i in range(1, n_s):
                t = i * DT
                ang = wind_angle_at(min(t, move_duration))
                v = air_speed(t)
                x += math.cos(ang) * v * DT
                y += math.sin(ang) * v * DT
                samples.append((x, y))

            def air_disp(t):
                i = min(len(samples) - 1, int(t / DT))
                return samples[i]

            xmin, xmax, ymin, ymax = STREAK_BOX
            srng = random.Random(7)
            bases = [(srng.uniform(xmin, xmax), srng.uniform(ymin, ymax)) for _ in range(N_STREAKS)]

            def build_streaks():
                t = clock.get_value()
                dx, dy = air_disp(t)
                ang = wind_angle_at(min(t, move_duration))
                ux, uy = math.cos(ang), math.sin(ang)
                g = gust_level(t)
                half = 0.14 + 0.24 * g                 # streaks stretch in a gust
                base_op = 0.45 + 0.5 * g               # and brighten
                group = VGroup()
                for bx, by in bases:
                    x = xmin + (bx + dx - xmin) % (xmax - xmin)
                    y = ymin + (by + dy - ymin) % (ymax - ymin)
                    edge = min(x - xmin, xmax - x, y - ymin, ymax - y)
                    op = base_op * min(1.0, edge / 0.5)  # fade in/out at the box edge instead of popping
                    if op <= 0.02:
                        continue
                    group.add(Line((x - ux * half, y - uy * half, 0), (x + ux * half, y + uy * half, 0),
                                   color=STREAK_COLOR, stroke_width=3.0, stroke_opacity=op))
                return group
            self.add(always_redraw(build_streaks))

        gx, gy = GAUGE_CENTER
        dial = Circle(radius=GAUGE_RADIUS, color="#555555", stroke_width=3.0)
        dial.set_fill("#FFFFFF", opacity=1.0)
        dial.move_to((gx, gy, 0))
        label = Text("WIND", font_size=24, color="#333333", weight="BOLD").move_to((gx, gy - GAUGE_RADIUS - 0.3, 0))
        self.add(dial, label)
        if wind_strength is not None:      # CHANGE: GUST tag, fades with the gust
            tag_box = RoundedRectangle(width=1.45, height=0.5, corner_radius=0.12, color=GUST_TAG_COLOR,
                                       fill_color=GUST_TAG_COLOR, fill_opacity=1.0, stroke_width=0)
            tag_txt = Text("GUST", font_size=26, color="#FFFFFF", weight="BOLD")
            tag = VGroup(tag_box, tag_txt).move_to((gx, gy - GAUGE_RADIUS - 0.95, 0))

            def tag_updater(mobj):
                lvl = gust_level(clock.get_value())
                mobj[0].set_fill(opacity=lvl)
                mobj[1].set_opacity(lvl)
            tag.add_updater(tag_updater)
            self.add(tag)
        if wind_strength is None:
            self.add(Text("CALM", font_size=26, color="#333333", weight="BOLD").move_to((gx, gy, 0)))
        else:
            n_chev = 1 if wind_strength == "light" else 2

            def build_arrow():
                ang = wind_angle_at(clock.get_value())
                ux, uy = math.cos(ang), math.sin(ang)
                L = GAUGE_RADIUS * 0.78
                arrow = Arrow((gx - ux * L, gy - uy * L, 0), (gx + ux * L, gy + uy * L, 0), buff=0,
                              color=WIND_COLOR, stroke_width=9, max_tip_length_to_length_ratio=0.32,
                              max_stroke_width_to_length_ratio=12)
                group = VGroup(arrow)
                px_, py_ = -uy, ux                       # perpendicular
                for i in range(n_chev):
                    c = -0.5 * L + i * 0.32 * L          # chevrons along the tail half, pointing downwind
                    tipx, tipy = gx + ux * (c + 0.16), gy + uy * (c + 0.16)
                    w = 0.3
                    chev = VMobject(color=WIND_COLOR, stroke_width=7)
                    chev.set_points_as_corners([(tipx - ux * 0.22 + px_ * w, tipy - uy * 0.22 + py_ * w, 0),
                                                (tipx, tipy, 0),
                                                (tipx - ux * 0.22 - px_ * w, tipy - uy * 0.22 - py_ * w, 0)])
                    group.add(chev)
                return group
            self.add(always_redraw(build_arrow))

        # --- per-ant home marker: colored ring + glyph, repositions if the home moves ---
        home_mobs = []
        for ant in p["ants"]:
            homes = [tuple(h) for h in ant["homes"]]

            def home_pos_at(t, homes=homes):
                tick = min(len(homes) - 1, tick_at(t))
                return homes[tick]

            ring = RegularPolygon(n=6, radius=p["cells"][0]["size"] * 1.05, color=ant["color"],
                                   stroke_width=4.0, fill_color=ant["color"], fill_opacity=0.22)
            x0, y0 = home_pos_at(0.0)
            ring.move_to((x0, y0, 0))

            def make_home_updater(home_pos_at=home_pos_at):
                def updater(mobj):
                    x, y = home_pos_at(clock.get_value())
                    mobj.move_to((x, y, 0))
                return updater
            ring.add_updater(make_home_updater())
            home_mobs.append(ring)
            self.add(ring)

        # --- ants: glyph icons that hop tick-to-tick, vanish if caught, freeze once home ---
        ant_mobs = []
        ant_pos_fns = []
        for ant in p["ants"]:
            positions = [tuple(pt) for pt in ant["positions"]]
            resolved_tick = ant["resolved_tick"]
            outcome = ant["outcome"]

            hole_hits = ant["hole_hits"]

            def hole_phase(t, hole_hits=hole_hits):
                """CHANGE: (hole_xy, u) if the ant is on a tick where it fell into a hole, u = raw 0..1 progress
                through that tick; else None. The fall is shown as: walk onto the hole (u < 0.35), shrink into
                it (0.35 <= u < 0.6), stay gone (0.6 <= u < 0.75), reappear on its start cell (u >= 0.75)."""
                if t >= move_duration:
                    return None
                tick = min(tick_at(t), len(hole_hits) - 1)
                if tick == 0 or hole_hits[tick] is None:
                    return None
                seg0, seg1 = seg(tick)
                return hole_hits[tick], min(1.0, max(0.0, (t - seg0) / (seg1 - seg0)))

            def pos_at(t, positions=positions, hole_phase=hole_phase):
                if t >= move_duration:
                    return positions[-1]
                tick = tick_at(t)
                tick = min(tick, len(positions) - 1)
                p0 = positions[max(0, tick - 1)]
                p1 = positions[tick]
                hp = hole_phase(t)
                if hp is not None:                       # CHANGE: into the hole, then back on its start (= p1)
                    (hx, hy), u = hp
                    if u < 0.35:
                        f = smooth(u / 0.35)
                        return (p0[0] + (hx - p0[0]) * f, p0[1] + (hy - p0[1]) * f)
                    return (hx, hy) if u < 0.75 else p1
                frac = frac_within_tick(t, tick)
                return (p0[0] + (p1[0] - p0[0]) * frac, p0[1] + (p1[1] - p0[1]) * frac)

            glyph = Text(ant["glyph"], color=ant["color"], font_size=28)
            base_h = [glyph.height]
            x0, y0 = pos_at(0.0)
            glyph.move_to((x0, y0, 0))

            def make_ant_updater(pos_at=pos_at, resolved_tick=resolved_tick, outcome=outcome,
                                 hole_phase=hole_phase, base_h=base_h):
                def updater(mobj):
                    t = clock.get_value()
                    tick = tick_at(t)
                    # CHANGE: was `tick > resolved_tick`, which never fires for an ant caught on the clip's
                    # last tick (tick_at tops out at that tick), so it stayed visible on top of the predator
                    if outcome == "dead" and t >= walk_end_time(resolved_tick):
                        mobj.set_opacity(0.0)
                        return
                    hp = hole_phase(t)
                    # CHANGE: falling into a hole: shrink on the hole cell, hidden, then back at full size on
                    # its start cell
                    scale = 1.0
                    if hp is not None and 0.35 <= hp[1] < 0.75:
                        scale = max(0.0, 1.0 - (hp[1] - 0.35) / 0.25)
                    mobj.set_opacity(1.0 if scale > 0.02 else 0.0)
                    if scale > 0.02:
                        mobj.scale_to_fit_height(base_h[0] * scale)
                    elif mobj.height != base_h[0]:
                        mobj.scale_to_fit_height(base_h[0])
                    x, y = pos_at(t)
                    mobj.move_to((x, y, 0))
                return updater
            glyph.add_updater(make_ant_updater())
            ant_mobs.append(glyph)
            ant_pos_fns.append(pos_at)
            self.add(glyph)


        # --- predators: roaming red dots ---
        predator_positions = p["predator_positions"]   # [tick][predator_idx] -> [x, y]
        predator_mobs = []
        for pi in range(p["n_predators"]):
            dot = Dot(radius=p["cells"][0]["size"] * 0.32, color=PREDATOR_COLOR)
            x0, y0 = predator_positions[0][pi]
            dot.move_to((x0, y0, 0))

            def make_predator_updater(pi=pi):
                # CHANGE: was a snap to the next cell at the start of each tick (so a predator could sit on an ant
                # without catching it). Now it hops cell to cell over the same time slot as the ants, so a catch
                # shows the predator landing on the ant exactly as the ant vanishes.
                base_w = None

                def updater(mobj):
                    nonlocal base_w
                    if base_w is None:
                        base_w = mobj.width
                    t = clock.get_value()
                    tick = min(len(predator_positions) - 1, tick_at(t))
                    if t >= move_duration:
                        x, y = predator_positions[-1][pi]
                        f = 1.0
                    else:
                        (x0, y0), (x1, y1) = predator_positions[max(0, tick - 1)][pi], predator_positions[tick][pi]
                        f = frac_within_tick(t, tick)
                        x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
                    hop = 1.0 + 0.35 * math.sin(math.pi * f) if t < move_duration else 1.0   # "jump" pulse
                    mobj.scale_to_fit_width(base_w * hop)
                    mobj.move_to((x, y, 0))
                return updater
            dot.add_updater(make_predator_updater())
            predator_mobs.append(dot)
            self.add(dot)

        # --- play the walk ---
        if move_duration > 0:
            self.play(clock.animate.set_value(move_duration), run_time=move_duration, rate_func=linear)

        # --- flashes for ants that made it home, all played together ---
        flashes = [Flash(ring, color=ant["color"], flash_radius=p["cells"][0]["size"] * 1.6,
                          line_length=p["cells"][0]["size"] * 0.4)
                   for ant, ring in zip(p["ants"], home_mobs) if ant["outcome"] == "home"]
        if flashes:
            self.play(AnimationGroup(*flashes, lag_ratio=0.0), run_time=HOME_FLASH_DURATION_S)
        else:
            self.wait(HOME_FLASH_DURATION_S)

        hold_s = max(0.0, total_duration - move_duration - HOME_FLASH_DURATION_S)
        if hold_s > 0:
            self.wait(hold_s)
