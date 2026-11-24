"""Independent answers for turntable_voxels (used only by the validator).

Nothing here calls sim.py or the questions' Ctx. The answer is rebuilt from
what the renderer draws:

  * the turntable angle is replayed frame by frame (24 fps) from the drawn
    turning programme, and the motion start / stops / reversal and the
    quarter-turn counts are read off that per-frame track;
  * the structure at any frame is rebuilt from the stack heights plus the
    drawn drops / lifts (a cube counts once its drop has landed, and is
    gone once its lift has finished), and event moments are read off the
    per-frame cube positions;
  * visibility in a still view is decided by RAY CASTING, not by the
    generator's ID-buffer render. Still views are square on (multiples of
    90 degrees), so a ray from a cube face towards the camera keeps its
    column and only climbs over the rows in front of it. Each camera-facing
    face (top and front) is sampled at 16 points; a cube is visible if any
    ray clears every stack it passes over.
"""
from __future__ import annotations
import math

FPS = 24
FACING = ("the side that faced you at the start", "the side that was on your left at the start",
          "the side that was at the back at the start", "the side that was on your right at the start")
AMOUNT = {1: "a quarter turn", 2: "half a turn", 3: "three quarters of a turn", 4: "one full turn",
          5: "one and a quarter turns", 6: "one and a half turns", 7: "one and three quarter turns",
          8: "two full turns"}
VISTOP = ("visible, and it is the top cube of its stack", "visible, but it is not the top cube of its stack",
          "hidden, and it is the top cube of its stack", "hidden, and it is not the top cube of its stack")
NSAMP = 16


# ------------------------------------------------------------ motion replay
def _smooth(u):
    u = 0.0 if u < 0 else 1.0 if u > 1 else u
    return u * u * (3 - 2 * u)


def _angle(scene, t):
    a = 0.0
    for lg in scene["legs"]:
        if t >= lg["t1"]:
            a = lg["a1"]
        elif t > lg["t0"]:
            return lg["a0"] + (lg["a1"] - lg["a0"]) * _smooth((t - lg["t0"]) / (lg["t1"] - lg["t0"]))
    return float(a)


def _track(scene):
    n = int(round(scene["duration"] * FPS))
    return [_angle(scene, i / FPS) for i in range(n)]


def _segments(scene):
    """Per-frame track -> runs of (still/moving, first frame, last frame, signed travel)."""
    tr = _track(scene)
    runs = []
    for i in range(1, len(tr)):
        d = tr[i] - tr[i - 1]
        state = 0 if abs(d) < 1e-9 else (1 if d > 0 else -1)
        if runs and runs[-1][0] == state:
            runs[-1][2] = i
            runs[-1][3] += d
        else:
            runs.append([state, i - 1, i, d])
    return runs, tr


def _moves(scene):
    """Turning runs (sign, travel in quarter turns, t_start, t_end) from the per-frame track; a reversal
    (sign change without a still frame pair) splits a run."""
    runs, _ = _segments(scene)
    return [(r[0], int(round(abs(r[3]) / 90)), r[1] / FPS, r[2] / FPS) for r in runs if r[0] != 0]


def _stills(scene):
    runs, tr = _segments(scene)
    out = [(r[1] / FPS, r[2] / FPS, tr[r[1]]) for r in runs if r[0] == 0]
    if runs and runs[0][0] != 0:
        out.insert(0, (0.0, 0.0, tr[0]))
    return out


def _quarter(a):
    return int(round(a / 90.0)) % 4


# --------------------------------------------------------------- structure
def _heights_at(scene, t):
    h = [list(r) for r in scene["heights"]]
    for ev in scene["events"]:
        if t >= ev["t1"]:
            h[ev["y"]][ev["x"]] += 1 if ev["kind"] == "drop" else -1
    return h


def _end_heights(scene):
    return _heights_at(scene, 1e9)


def _rot(h, k):
    """Heights seen square on after k quarter turns counter-clockwise from above: cell (x, y) of the
    turned layout is the cell that then sits x-th from the left, y-th from the front."""
    for _ in range(k % 4):
        gy, gx = len(h), len(h[0])
        h = [[h[gy - 1 - xp][yp] for xp in range(gy)] for yp in range(gx)]
    return h


def _visible(h, elev):
    """Visible cubes (x, y, z) of a heightmap seen square on from the front at elevation elev."""
    r = math.radians(elev)
    c, s = math.cos(r), math.sin(r)
    gy, gx = len(h), len(h[0])
    out = set()

    def clear(x, y0, z0, row_from):
        # ray (y, z) = (y0 - c u, z0 + s u); over row yy (< y0) it is lowest where it enters (y = yy + 1)
        for yy in range(row_from, -1, -1):
            u = (y0 - (yy + 1)) / c
            if h[yy][x] > z0 + s * max(u, 0.0) + 1e-9:
                return False
        return True

    for y in range(gy):
        for x in range(gx):
            for z in range(h[y][x]):
                seen = False
                if z == h[y][x] - 1:                     # top face: points (y + f, z + 1)
                    for i in range(NSAMP):
                        if clear(x, y + (i + 0.5) / NSAMP, z + 1, y - 1):
                            seen = True
                            break
                if not seen:                             # front face: points (y, z + f)
                    for i in range(NSAMP):
                        if clear(x, y, z + (i + 0.5) / NSAMP, y - 1):
                            seen = True
                            break
                if seen:
                    out.add((x, y, z))
    return out


def _vis_world(scene, h, a):
    """Visible cubes in world (start-layout) coordinates for a still view at angle a."""
    k = _quarter(a)
    rh = _rot(h, k)
    vis = _visible(rh, scene["elev"])
    # map turned-layout cells back: invert one quarter turn k times
    gy, gx = len(h), len(h[0])
    out = set()
    for (xp, yp, z) in vis:
        x, y = xp, yp
        cy_, cx_ = len(rh), len(rh[0])
        for _ in range(k):
            # (x', y') on the turned grid came from (x, y) = (y', cx - 1 - x') one quarter turn earlier
            x, y = y, cx_ - 1 - x
            cy_, cx_ = cx_, cy_
        out.add((x, y, z))
    del gy, gx
    return out


def _fv(h):
    return ", ".join(str(max(h[y][x] for y in range(len(h)))) for x in range(len(h[0])))


def _marked(scene, col):
    return next(m for m in scene["marked"] if m["colour"] == col)


def _set_text(cols):
    cols = sorted(cols)
    if not cols:
        return "none of them"
    if len(cols) == 1:
        return f"only the {cols[0]} cube"
    return "the " + ", ".join(cols[:-1]) + f" and {cols[-1]} cubes"


def _event_moments(scene):
    """Landing frames (drop) / lift-off frames (lift), read from per-frame cube heights."""
    out = []
    n = int(round(scene["duration"] * FPS))
    for ev in scene["events"]:
        if ev["kind"] == "drop":
            f = next(i for i in range(n) if i / FPS >= ev["t1"])          # first frame the cube rests
            out.append(f / FPS - 0.5 / FPS)
        else:
            f = next(i for i in range(n) if i / FPS > ev["t0"])           # first frame the cube has risen
            out.append(f / FPS - 0.5 / FPS)
    return out


def answer(key: str, p: dict, scene: dict):
    h0 = scene["heights"]
    he = _end_heights(scene)
    n0 = sum(map(sum, h0))
    moves = _moves(scene)
    stills = _stills(scene)
    a_end = stills[-1][2]
    vis0 = _vis_world(scene, h0, 0.0)
    if key == "grid_squares":
        return len(h0) * len(h0[0])
    if key == "tallest_start":
        return max(map(max, h0))
    if key == "colour_present":
        return p["c"]
    if key == "turn_dirs":
        signs = [m[0] for m in moves]
        if len(set(signs)) == 1:
            return "always to the right" if signs[0] > 0 else "always to the left"
        return "first to the right, later to the left" if signs[0] > 0 else "first to the left, later to the right"
    if key == "n_events":
        return len(scene["events"])
    if key == "marked_vis_top":
        m = _marked(scene, p["c"])
        v = (m["x"], m["y"], m["z"]) in vis0
        top = h0[m["y"]][m["x"]] == m["z"] + 1
        return VISTOP[(0 if v else 2) + (0 if top else 1)]
    if key == "above_marked":
        m = _marked(scene, p["c"])
        return h0[m["y"]][m["x"]] - m["z"] - 1
    if key == "below_marked":
        return _marked(scene, p["c"])["z"]
    if key == "marked_set_start":
        return _set_text([m["colour"] for m in scene["marked"] if (m["x"], m["y"], m["z"]) in vis0])
    if key == "marked_set_end":
        ve = _vis_world(scene, he, a_end)
        lifted = {(e["x"], e["y"], e["z"]) for e in scene["events"] if e["kind"] == "lift"}
        return _set_text([m["colour"] for m in scene["marked"]
                          if (m["x"], m["y"], m["z"]) in ve and (m["x"], m["y"], m["z"]) not in lifted])
    if key == "highest_marked":
        return max(scene["marked"], key=lambda m: m["z"])["colour"]
    if key == "facing_stop":
        return FACING[_quarter(stills[1][2])]
    if key == "n_stacks":
        return sum(1 for r in h0 for v in r if v)
    if key == "visible_start":
        return len(vis0)
    if key == "visible_end":
        return len(_vis_world(scene, he, a_end))
    if key == "total_start":
        return n0
    if key == "t_start":
        return int(round(moves[0][2]))
    if key == "first_leg":
        sg, q = moves[0][0], moves[0][1]
        return f"{AMOUNT[q]}, with its near edge moving {'right' if sg > 0 else 'left'}"
    if key == "total_quarters":
        return sum(m[1] for m in moves)
    if key == "layer_diff":
        return _marked(scene, p["a"])["z"] - _marked(scene, p["b"])["z"]
    if key == "t_stop":
        return int(round(stills[1][0]))
    if key == "stop_len":
        return int(round(stills[1][1] - stills[1][0]))
    if key == "visible_stop":
        return len(_vis_world(scene, _heights_at(scene, stills[1][0]), stills[1][2]))
    if key == "t_reverse":
        return int(round(moves[0][3]))
    if key == "t_event":
        return int(round(_event_moments(scene)[p["i"]]))
    if key == "hidden_start":
        return n0 - len(vis0)
    if key == "facing_end":
        return FACING[_quarter(a_end)]
    if key == "front_view_start":
        return _fv(h0)
    if key == "front_view_end":
        return _fv(_rot(he, _quarter(a_end)))
    if key == "hidden_end":
        return sum(map(sum, he)) - len(_vis_world(scene, he, a_end))
    if key == "total_end":
        return sum(map(sum, he))
    if key == "tallest_end":
        return max(map(max, he))
    if key == "cf_add_top":
        return n0 + sum(1 for r in h0 for v in r if v)
    if key in ("cf_remove_fv", "cf_remove_hidden"):
        m = _marked(scene, p["c"])
        h2 = [list(r) for r in h0]
        h2[m["y"]][m["x"]] = 0
        if key == "cf_remove_fv":
            return _fv(h2)
        return sum(map(sum, h2)) - len(_vis_world(scene, h2, 0.0))
    if key == "cf_other_dir":
        return FACING[_quarter(-a_end)]
    if key == "cf_stop_early":
        return len(_vis_world(scene, he, a_end - moves[-1][0] * 90))
    if key == "cf_no_events":
        return len(_vis_world(scene, h0, a_end))
    return None
