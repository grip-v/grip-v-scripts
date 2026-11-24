"""mirror_beam_reflection scene sampler and closed-form tracer.

A light pulse leaves an emitter on the edge of a board of square cells and
moves one cell per `step` seconds, leaving a lit trail. Mirrors sit on cell
diagonals ('/' or '\\'); a pulse reaching a mirror cell's centre turns 90
degrees. The clip stops before the beam has left the board, so the rest of
the path has to be predicted from the reflection rule.

Scenarios (scene["scenario"]):
  classic    two-sided mirrors, one beam
  one_sided  some mirrors are silvered on one face only (dark backing on the
             other); a pulse that hits the backing is absorbed
  splitter   1-2 half-mirrors (drawn dashed): the pulse goes on straight AND
             a copy is reflected
  flip       1-2 pivoting mirrors (drawn with a pivot hub) turn to the other
             diagonal at a fixed moment; a pulse uses the orientation at the
             time it reaches the cell
  two_beams  two emitters with different beam colours, launched at different times
Any scenario may add 0-3 absorber blocks (dark squares that stop a pulse).

Coordinates: cell (c, r), c = column from the left, r = row from the top;
directions (dc, dr) with dr > 0 = down. Cells are named on screen by a
column letter above the board and a row number left of it ("C4" = column C,
row 4). Times are seconds; frame f is shown at t = f / 24 and the clock
shows f * 10 // 24 tenths. Pure function of the rng passed in.
"""
from __future__ import annotations
import math
import random

FPS = 24
MAX_FRAMES = 240
EDGE_S = 0.5
AREA = (250, 20, 1262, 704)
CLOCK_BOX = (18, 16, 214, 84)
SCENARIOS = ("classic", "one_sided", "splitter", "flip", "two_beams")
LETTERS = "ABCDEFGHIJ"
MAX_STEPS = 200
LABEL_PX = 34                       # label font size (glyphs >= 24 px tall)

BEAMS = {"red": (235, 50, 50), "green": (40, 200, 70), "orange": (250, 140, 20), "magenta": (225, 60, 220),
         "blue": (50, 110, 250)}
THEMES = {
    "dark": {"bg": [(18, 20, 26), (26, 22, 30), (16, 26, 28), (28, 28, 30)],
             "board": [(36, 40, 50), (44, 40, 52), (34, 46, 48)], "grid": [(70, 76, 90), (84, 78, 96), (66, 88, 86)],
             "mirror": [(150, 220, 255), (220, 225, 235), (170, 240, 220)], "ink": (235, 235, 240),
             "absorber": (6, 6, 8), "back": (165, 165, 172)},
    "light": {"bg": [(232, 230, 224), (220, 226, 234), (236, 232, 240)],
              "board": [(250, 250, 246), (244, 247, 250), (248, 246, 240)], "grid": [(180, 176, 168), (164, 172, 186)],
              "mirror": [(20, 90, 160), (60, 60, 70), (20, 120, 110)], "ink": (20, 20, 24),
              "absorber": (40, 40, 44), "back": (70, 70, 78)},
}


# ------------------------------------------------------------------ the reflection rule
def reflect(kind, d):
    dc, dr = d
    return (-dr, -dc) if kind == "/" else (dr, dc)


def flipped(kind):
    return "\\" if kind == "/" else "/"


def kind_at(m, t, step):
    """Mirror orientation seen by a pulse reaching the cell centre at time t."""
    if m["type"] == "flip" and t >= m["flip_t"]:
        return flipped(m["kind"])
    return m["kind"]


def face_ok(m, d):
    """One-sided mirror: does a pulse moving along d hit the silvered face?"""
    return d[0] * m["face"][0] + d[1] * m["face"][1] < 0


def edge_entry(scene_like, em):
    """(first cell, direction) of an emitter."""
    W, H = scene_like["cols"], scene_like["rows"]
    e, i = em["edge"], em["index"]
    return {"left": ((0, i), (1, 0)), "right": ((W - 1, i), (-1, 0)),
            "top": ((i, 0), (0, 1)), "bottom": ((i, H - 1), (0, -1))}[e]


def trace(world, max_steps=MAX_STEPS):
    """Simulate every pulse. world: cols, rows, step, mirrors [{cell, kind, type, face?, flip_t?}],
    absorbers [cells], emitters [{edge, index, t0, colour}].
    Returns {"pulses": [...], "loop": bool}. Each pulse: colour, parent (index or None),
    pts [(t, x, y)] waypoints in grid units (cell centre = c + 0.5, r + 0.5), events
    [(t, kind, cell)] with kind in bounce / split / absorb / back / flip_bounce, and end
    ("exit", edge, index, t) | ("absorbed", cell, t) | ("open",) when max_steps ran out."""
    W, H, s = world["cols"], world["rows"], world["step"]
    mir = {tuple(m["cell"]): m for m in world["mirrors"]}
    absb = {tuple(a) for a in world["absorbers"]}
    pulses, queue, loop = [], [], False
    for em in world["emitters"]:
        cell, d = edge_entry(world, em)
        bx, by = cell[0] + 0.5 - 0.5 * d[0], cell[1] + 0.5 - 0.5 * d[1]
        p = {"colour": em["colour"], "parent": None, "pts": [(em["t0"], bx, by)], "events": [], "end": ("open",)}
        pulses.append(p)
        queue.append((len(pulses) - 1, cell, d, em["t0"] + s / 2, 0))
    seen = set()
    while queue:
        pi, cell, d, t, n = queue.pop(0)
        p = pulses[pi]
        while True:
            if n > max_steps:
                p["end"] = ("open",)
                break
            c, r = cell
            if not (0 <= c < W and 0 <= r < H):
                bx, by = c + 0.5 - 0.5 * d[0], r + 0.5 - 0.5 * d[1]
                t_exit = t - s / 2
                p["pts"].append((t_exit, bx, by))
                edge = "left" if c < 0 else "right" if c >= W else "top" if r < 0 else "bottom"
                p["end"] = ("exit", edge, r if edge in ("left", "right") else c, t_exit)
                break
            p["pts"].append((t, c + 0.5, r + 0.5))
            key = (cell, d)
            if key in seen:
                loop = True
                p["end"] = ("open",)
                break
            seen.add(key)
            if cell in absb:
                p["events"].append((t, "absorb", cell))
                p["end"] = ("absorbed", cell, t)
                break
            m = mir.get(cell)
            if m is not None:
                k = kind_at(m, t, s)
                if m["type"] == "one" and not face_ok(m, d):
                    p["events"].append((t, "back", cell))
                    p["end"] = ("absorbed", cell, t)
                    break
                nd = reflect(k, d)
                if m["type"] == "split":
                    p["events"].append((t, "split", cell))
                    child = {"colour": p["colour"], "parent": pi, "pts": [(t, c + 0.5, r + 0.5)], "events": [],
                             "end": ("open",)}
                    pulses.append(child)
                    queue.append((len(pulses) - 1, (c + nd[0], r + nd[1]), nd, t + s, n + 1))
                else:
                    p["events"].append((t, "bounce", cell))
                    d = nd
            cell = (c + d[0], r + d[1])
            t += s
            n += 1
    return {"pulses": pulses, "loop": loop}


def world_of(scene, **over):
    w = {k: scene[k] for k in ("cols", "rows", "step", "mirrors", "absorbers", "emitters")}
    w.update(over)
    return w


def head_at(p, t):
    """(x, y) of a pulse's head at time t in grid units, or None if not started / finished."""
    pts = p["pts"]
    if t < pts[0][0] or len(pts) < 2 and p["end"][0] != "open":
        return None
    if t > pts[-1][0]:
        return None if p["end"][0] in ("exit", "absorbed") else (pts[-1][1], pts[-1][2])
    for (ta, xa, ya), (tb, xb, yb) in zip(pts, pts[1:]):
        if ta <= t <= tb:
            u = 0 if tb == ta else (t - ta) / (tb - ta)
            return xa + (xb - xa) * u, ya + (yb - ya) * u
    return pts[-1][1], pts[-1][2]


def n_frames(scene):
    return scene["frames"]


def duration(scene):
    return scene["frames"] / FPS


def clock_tenths(frame: int) -> int:
    return frame * 10 // FPS


def cell_name(cell):
    return f"{LETTERS[cell[0]]}{cell[1] + 1}"


# ------------------------------------------------------------------ geometry on screen
def layout(rng, cols, rows):
    """Board origin and cell size (px), leaving room for the labels above and left."""
    lab = LABEL_PX + 18
    x_lo, y_lo = AREA[0] + lab, AREA[1] + lab
    cmax = min(96.0, (AREA[2] - 30 - x_lo) / cols, (AREA[3] - 30 - y_lo) / rows)
    cell = int(rng.uniform(max(60.0, 0.8 * cmax), cmax))
    x0 = x_lo + rng.randint(0, int(AREA[2] - 30 - x_lo - cols * cell))
    y0 = y_lo + rng.randint(0, int(AREA[3] - 30 - y_lo - rows * cell))
    return x0, y0, cell


def labels(scene):
    """Every text label on screen except the clock: [{text, x, y}] (centre, px). Fixed for the clip."""
    st = scene["style"]
    x0, y0, cell = st["x0"], st["y0"], st["cell"]
    out = [{"text": LETTERS[c], "x": x0 + (c + 0.5) * cell, "y": y0 - 8 - LABEL_PX / 2} for c in range(scene["cols"])]
    out += [{"text": str(r + 1), "x": x0 - 10 - LABEL_PX / 2, "y": y0 + (r + 0.5) * cell} for r in range(scene["rows"])]
    return out


# ------------------------------------------------------------------ generation
def _style(rng, cols, rows, n_beams):
    theme = rng.choice(("dark", "dark", "light"))
    T = THEMES[theme]
    x0, y0, cell = layout(rng, cols, rows)
    beams = rng.sample(sorted(BEAMS), n_beams)
    return {"theme": theme, "bg": list(rng.choice(T["bg"])), "board": list(rng.choice(T["board"])),
            "board_pat": rng.choice(("plain", "checker", "dots")), "grid": list(rng.choice(T["grid"])),
            "grid_style": rng.choice(("solid", "dashed", "dotted")), "mirror": list(rng.choice(T["mirror"])),
            "mirror_w": rng.randint(5, 8), "mirror_len": round(rng.uniform(0.62, 0.82), 3),
            "trail_w": rng.randint(5, 9), "head_px": rng.randint(22, 30), "ink": list(T["ink"]),
            "absorber": list(T["absorber"]), "back": list(T["back"]), "emitter_shape": rng.choice(("box", "wedge")),
            "x0": x0, "y0": y0, "cell": cell}, beams


def _edges(rng, cols, rows):
    edge = rng.choice(("left", "right", "top", "bottom"))
    n = rows if edge in ("left", "right") else cols
    return edge, rng.randrange(n)


def _room(cols, rows, cell, d):
    n, c = 0, cell
    while True:
        c = (c[0] + d[0], c[1] + d[1])
        if not (0 <= c[0] < cols and 0 <= c[1] < rows):
            return n
        n += 1


def _route(rng, cols, rows, cell, d, mirrors, used):
    """Walk a random route from `cell` along d, placing a turning mirror every 1-4 cells (path-first).
    Every cell of the route is marked used, so decoy mirrors stay off it."""
    for _ in range(rng.randint(3, 7)):
        room = _room(cols, rows, cell, d)
        if room < 1:
            return
        run = rng.randint(1, min(4, room))
        for _ in range(run):
            nxt = (cell[0] + d[0], cell[1] + d[1])
            if nxt in used:
                return
            cell = nxt
            used.add(cell)
        sides = [(d[1], d[0]), (-d[1], -d[0])]
        sides.sort(key=lambda nd: -_room(cols, rows, cell, nd))
        nd = sides[0] if rng.random() < 0.75 else sides[1]
        kind = "/" if reflect("/", d) == nd else "\\"
        mirrors.append({"cell": list(cell), "kind": kind, "type": "two", "_din": list(d)})
        d = nd


def _world(rng, scenario):
    cols, rows = rng.randint(5, 10), rng.randint(4, 8)
    n_beams = 2 if scenario == "two_beams" else 1
    emitters, spots = [], set()
    for k in range(n_beams):
        for _ in range(20):
            e = _edges(rng, cols, rows)
            if e not in spots:
                break
        spots.add(e)
        t0 = round(rng.uniform(0.5, 1.0) if k == 0 else rng.uniform(1.4, 3.0), 3)
        emitters.append({"edge": e[0], "index": e[1], "t0": t0})
    world = {"cols": cols, "rows": rows}
    mirrors, used = [], set()
    for em in emitters:
        cell, d = edge_entry(world, em)
        used.add(tuple(cell))
        _route(rng, cols, rows, tuple(cell), d, mirrors, used)
    free = [(c, r) for c in range(cols) for r in range(rows) if (c, r) not in used]
    rng.shuffle(free)
    n_decoy = rng.randint(1, max(2, cols * rows // 8))
    for cell in free[:n_decoy]:
        mirrors.append({"cell": list(cell), "kind": rng.choice(("/", "\\")), "type": "two"})
    n_abs = rng.randint(0, 3)
    absorbers = [list(c) for c in free[n_decoy:n_decoy + n_abs]]
    step = round(rng.uniform(0.22, 0.45), 3)
    if scenario == "one_sided":
        for m in rng.sample(mirrors, min(len(mirrors), rng.randint(2, max(2, len(mirrors) // 2)))):
            faces = ((-1, -1), (1, 1)) if m["kind"] == "/" else ((1, -1), (-1, 1))
            if "_din" in m and rng.random() < 0.8:      # route mirrors mostly face the oncoming beam
                f = next(x for x in faces if x[0] * m["_din"][0] + x[1] * m["_din"][1] < 0)
            else:
                f = rng.choice(faces)
            m["type"], m["face"] = "one", list(f)
    if scenario == "splitter":
        for m in rng.sample(mirrors, min(len(mirrors), rng.randint(1, 2))):
            m["type"] = "split"
    for m in mirrors:
        m.pop("_din", None)
    mirrors.sort(key=lambda m: (m["cell"][1], m["cell"][0]))
    return {"cols": cols, "rows": rows, "step": step, "mirrors": mirrors, "absorbers": absorbers,
            "emitters": emitters}


def _end_time(res):
    ends = []
    for p in res["pulses"]:
        e = p["end"]
        ends.append(e[-1] if e[0] in ("exit", "absorbed") else math.inf)
    return max(ends)


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    for attempt in range(1, 2000):
        world = _world(rng, scenario)
        st, beams = _style(rng, world["cols"], world["rows"], len(world["emitters"]))
        for em, b in zip(world["emitters"], beams):
            em["colour"] = b
        res = trace(world)
        if res["loop"] or any(p["end"][0] == "open" for p in res["pulses"]):
            continue
        if scenario == "flip":
            # pivots are mirrors the beam really meets; each turns at a frame instant >= 2 steps from any hit
            hit = {}
            for p in res["pulses"]:
                for t, kind, cell in p["events"]:
                    if kind == "bounce" and 1.0 <= t <= 8.5:
                        hit.setdefault(tuple(cell), []).append(t)
            cand = [m for m in world["mirrors"] if tuple(m["cell"]) in hit]
            if not cand:
                continue
            for m in rng.sample(cand, min(len(cand), rng.randint(1, 2))):
                first = min(hit[tuple(m["cell"])])
                lo, hi = (0.8, first - 2 * world["step"]) if rng.random() < 0.5 else \
                    (first + 2 * world["step"], first + 2 * world["step"] + 1.5)
                if hi <= lo:
                    continue
                m["type"] = "flip"
                m["flip_t"] = round(rng.uniform(lo, hi) * FPS) / FPS      # a frame instant
            if not any(m["type"] == "flip" for m in world["mirrors"]):
                continue
            res = trace(world)
            if res["loop"] or any(p["end"][0] == "open" for p in res["pulses"]):
                continue
        bounces = sum(1 for p in res["pulses"] for e in p["events"] if e[1] in ("bounce", "split"))
        if bounces < 3:
            continue
        t_end_full = _end_time(res)
        # the clip stops while at least one pulse is still on the board, after >= 2 bounces were seen,
        # with at least one bounce / split / absorb still to come
        evs = sorted(e[0] for p in res["pulses"] for e in p["events"])
        if len(evs) < 4:
            continue
        lo = max(evs[2] + 0.6, 4.5)                     # >= 3 events and >= 4.5 s on screen
        hi = min(t_end_full - 0.6, evs[-1] - 0.3, MAX_FRAMES / FPS)
        if hi < lo:
            continue
        t_clip = lo + (hi - lo) * math.sqrt(rng.random())      # lean towards longer clips
        frames = int(t_clip * FPS)
        scene = {"scenario": scenario, **world, "frames": frames, "style": st, "build_attempts": attempt}
        scene["labels"] = labels(scene)
        if check(scene) or not _ladder_ok(scene):
            continue
        return scene
    raise RuntimeError("no valid board")


def _ladder_ok(scene):
    """Can this board fill the 10-question ladder? (A trial build with its own fixed rng.)"""
    import questions                                   # late import: questions.py imports this module
    try:
        questions.build(scene, random.Random(0))
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------ consistency check
def all_events(res):
    return sorted((e[0], e[1], tuple(e[2]), i) for i, p in enumerate(res["pulses"]) for e in p["events"])


def check(scene: dict) -> list[str]:
    errs = []
    res = trace(world_of(scene))
    if res["loop"]:
        return ["a pulse loops"]
    if any(p["end"][0] == "open" for p in res["pulses"]):
        return ["a pulse never finishes"]
    dur = duration(scene)
    if not 108 <= scene["frames"] <= MAX_FRAMES:
        errs.append("clip length out of range")
    s = scene["step"]
    # two heads never closer than one cell at the same moment
    tt = 0.0
    while tt <= dur:
        hs = [head_at(p, tt) for p in res["pulses"]]
        hs = [h for h in hs if h is not None]
        for i in range(len(hs)):
            for j in range(i + 1, len(hs)):
                if math.dist(hs[i], hs[j]) < 1.0 - 1e-9 and not _siblings_at_split(res, hs[i], tt):
                    errs.append("two pulse heads overlap")
                    break
        tt += 1 / FPS
    # pulses must not visit the same cell twice (crossing trails are fine, but no beam re-uses a cell's diagonal)
    for m in scene["mirrors"]:
        if m["type"] == "flip":
            hits = [ev for ev in all_events(res) if ev[2] == tuple(m["cell"])]
            near = [abs(ev[0] - m["flip_t"]) for ev in hits]
            if any(x < 2 * s for x in near):
                errs.append("a pulse reaches a pivoting mirror while it turns")
            if not EDGE_S <= m["flip_t"] <= dur - EDGE_S:
                errs.append("a mirror turns outside the clip")
    cells = [tuple(m["cell"]) for m in scene["mirrors"]] + [tuple(a) for a in scene["absorbers"]]
    if len(set(cells)) != len(cells):
        errs.append("two objects share a cell")
    for em in scene["emitters"]:
        if tuple(edge_entry(scene, em)[0]) in set(cells):
            errs.append("an emitter fires straight into an object")
    if len({(e["edge"], e["index"]) for e in scene["emitters"]}) != len(scene["emitters"]):
        errs.append("two emitters share a spot")
    st = scene["style"]
    if st["cell"] < 60 or st["head_px"] < 20 or st["trail_w"] < 3:
        errs.append("objects too small")
    x1, y1 = st["x0"] + scene["cols"] * st["cell"], st["y0"] + scene["rows"] * st["cell"]
    if st["x0"] - LABEL_PX - 18 < AREA[0] or st["y0"] - LABEL_PX - 18 < AREA[1] or x1 > AREA[2] or y1 > AREA[3]:
        errs.append("board leaves its area")
    evs = all_events(res)
    special = {tuple(m["cell"]) for m in scene["mirrors"] if m["type"] in ("one", "split", "flip")}
    if special and not any(e[2] in special and e[0] <= dur - EDGE_S for e in evs):
        errs.append("no pulse meets the scenario's special mirror during the clip")
    shown = [e for e in evs if e[0] <= dur]
    if len(shown) < 3 or len(shown) == len(evs):
        errs.append("needs >= 3 events in the clip and >= 1 after it")
    return errs


def _siblings_at_split(res, h, t):
    """Right at a split the two halves start from the same point: not an overlap."""
    for p in res["pulses"]:
        if p["parent"] is not None and abs(p["pts"][0][0] - t) < res_step(res) + 1e-9:
            if math.dist((p["pts"][0][1], p["pts"][0][2]), h) <= 1.0:
                return True
    return False


def res_step(res):
    for p in res["pulses"]:
        if len(p["pts"]) > 2:
            return p["pts"][2][0] - p["pts"][1][0]
    return 0.3
