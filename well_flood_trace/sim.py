"""well_flood_trace: scene builder and flood engine (no rendering).

The world (the original well_flood_trace convention, kept)
----------------------------------------------------------
An N x N grid of sections (N = 5..8). Each section has a distinct integer
depth, a shuffle of 1..N*N, printed in the cell; a section of depth d holds d
units of water. Water is poured into an entry section (outlined in red). A
section fills at the rate water reaches it; the moment it is full, its whole
inflow is split evenly among ALL its neighbouring sections (up / down / left /
right, not across a wall) with a LOWER depth value. A full section with no
lower neighbour stays full and the extra water soaks away.

Scenarios (scene["scenario"]):
    valid_flow     one entry, constant pour, everything physically correct
    two_entries    a second entry (outlined in purple) starts pouring later
    pour_change    the pour stops (outline turns grey) or doubles (outline thickens)
    walled         thick walls between some neighbours block the spill there
    broken_flow    ONE physically impossible event (the original two kinds):
                   premature_fill - a section starts filling with no full
                                    higher neighbour feeding it, before the
                                    water could really get there
                   uphill_flow    - a section receives water from a full
                                    neighbour with a LOWER depth value

Timing
------
Pour events sit at frame + 1/2 (t = (k + 0.5) / 24), and every event time is
kept >= 0.03 frame away from a frame boundary, so "is the section wet at frame
f" (start <= f / 24) never depends on float rounding. Distinct event instants
are >= 3 frames apart and none falls in the first or last 0.5 s.

Everything random comes from the `rng` passed to build().
"""
from __future__ import annotations
import heapq
import math

FPS = 24
SCENARIOS = ("valid_flow", "two_entries", "pour_change", "walled", "broken_flow")
FILL_SECONDS = (1.2, 1.5, 1.8, 2.1, 2.4, 2.8, 3.2)        # the entry section fills in this long
EPS = 1e-12
MIN_GAP_FRAMES = 3
EDGE_MARGIN_S = 0.5
FRAME_SAFE = 0.03                                    # event must be >= this fraction of a frame from a boundary
ENTRY_COLOURS = ("red", "purple")                   # first and second entry outlines
PALETTES = 3                                         # subtle variants of the original colours (render.py)


# ------------------------------------------------------------------ grid helpers
def nbrs(R: int, C: int, i: int) -> list[int]:
    r, c = divmod(i, C)
    out = []
    if r > 0:
        out.append(i - C)
    if r < R - 1:
        out.append(i + C)
    if c > 0:
        out.append(i - 1)
    if c < C - 1:
        out.append(i + 1)
    return sorted(out)


def wall_set(walls) -> set:
    return {(min(a, b), max(a, b)) for a, b in walls}


GUTTER = -1      # "soaks away": where a full section with no lower neighbour sends its water


def children(R, C, depth, walls, extra=(), drop=()) -> list[list[int]]:
    """down[i]: the sections i spills into once full (lower depth, not walled);
    [GUTTER] (soaks away) for a section with no lower neighbour."""
    ws = wall_set(walls)
    dropped = {tuple(e) for e in drop}
    down = []
    for i in range(R * C):
        kids = [j for j in nbrs(R, C, i) if depth[j] < depth[i] and (min(i, j), max(i, j)) not in ws
                and (i, j) not in dropped]
        down.append(kids)
    for a, b in extra:
        if b not in down[a]:
            down[a] = sorted(down[a] + [b])
    for i in range(R * C):
        if not down[i]:
            down[i] = [GUTTER]
    return down


def topo(n: int, down) -> list[int] | None:
    indeg = [0] * n
    for i in range(n):
        for j in down[i]:
            if j != GUTTER:
                indeg[j] += 1
    heap = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        i = heapq.heappop(heap)
        order.append(i)
        for j in down[i]:
            if j == GUTTER:
                continue
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(heap, j)
    return order if len(order) == n else None


def tap_time(k: int) -> float:
    return (k + 0.5) / FPS


def sources_of(scene: dict, rate_mult: float = 1.0, move=None, drop_tap=None, unchanged=False,
               violation=True) -> list:
    """[(cell, [(t0, t1|None, rate), ...]), ...] for the flood engine.
    move=(tap index, new cell), drop_tap=tap index, unchanged: ignore the tap's mid-clip change."""
    out = []
    for k, tp in enumerate(scene["taps"]):
        if k == drop_tap:
            continue
        cell = move[1] if move and move[0] == k else tp["cell"]
        segs = []
        for f0, f1, rate in tp["segs"]:
            if unchanged:
                f1 = None
                rate = tp["segs"][0][2]
            segs.append((tap_time(f0), None if f1 is None else tap_time(f1), rate * rate_mult))
            if unchanged:
                break
        out.append((cell, segs))
    v = scene.get("violation")
    if violation and v and v["kind"] == "premature_fill":
        out.append((v["cell"], [(tap_time(v["frame"]), None, v["rate"])]))
    return out


def graph_mods(scene: dict, violation=True):
    v = scene.get("violation")
    if violation and v and v["kind"] == "uphill_flow":
        return [(v["src"], v["dst"])], [(v["dst"], v["src"])]
    return [], []


# ------------------------------------------------------------------ flood engine
def flood(R, C, depth, walls, sources, extra=(), drop=(), horizon=11.0, blocked=()) -> dict | None:
    """Event-stepped simulation. At every event instant all inflow rates are
    recomputed from scratch (topological order), then time jumps to the next
    tap change or tank completion. Returns None if the spill graph has a cycle."""
    n = R * C
    down = children(R, C, depth, walls, extra, drop)
    blocked = set(blocked)
    if blocked:
        down = [[j for j in d if j not in blocked] for d in down]
        down = [d if d else [GUTTER] for d in down]
    order = topo(n, down)
    if order is None:
        return None
    bounds = sorted({t for _, segs in sources for s in segs for t in (s[0], s[1]) if t is not None})
    vol, full = [0.0] * n, [False] * n
    start, tfull = [None] * n, [None] * n
    first_src = [None] * n
    level = [[] for _ in range(n)]
    open_edges: dict = {}
    flows = []
    steps = []
    t = 0.0
    while True:
        rin = [0.0] * n
        contrib = [[] for _ in range(n)]
        for k, (cell, segs) in enumerate(sources):
            if cell in blocked:
                continue
            for t0, t1, rate in segs:
                # same EPS as the next-boundary test below, so a boundary that lands within
                # EPS of a completion is applied at that step instead of being skipped
                if t0 <= t + EPS and (t1 is None or t + EPS < t1) and rate > 0:
                    rin[cell] += rate
                    contrib[cell].append(-1 - k)
        for i in order:
            if full[i] and down[i] and rin[i] > EPS:
                share = rin[i] / len(down[i])
                for j in down[i]:
                    if j == GUTTER:
                        continue
                    rin[j] += share
                    contrib[j].append(i)
        for j in range(n):
            if start[j] is None and rin[j] > EPS:
                start[j], first_src[j] = t, sorted(contrib[j])
            if start[j] is not None:
                slope = 0.0 if full[j] else rin[j]
                if not level[j] or level[j][-1][2] != slope:
                    level[j].append([t, vol[j], slope])
        live = {(i, j) for i in range(n) if full[i] and rin[i] > EPS for j in down[i]}
        for e in sorted(set(open_edges) - live):
            flows.append([e[0], e[1], open_edges.pop(e), t])
        for e in sorted(live - set(open_edges)):
            open_edges[e] = t
        cands = [b for b in bounds if b > t + EPS]
        for i in range(n):
            if not full[i] and rin[i] > EPS:
                cands.append(t + (depth[i] - vol[i]) / rin[i])
        t_next = min(cands) if cands else math.inf
        if t_next > horizon:
            break
        for i in range(n):
            if not full[i]:
                vol[i] += rin[i] * (t_next - t)
        t = t_next
        for i in range(n):
            if not full[i] and rin[i] > EPS and vol[i] >= depth[i] - 1e-9:
                full[i], vol[i], tfull[i] = True, float(depth[i]), t
        steps.append(t)
    for e in sorted(open_edges):
        flows.append([e[0], e[1], open_edges[e], None])
    flows.sort(key=lambda f: (f[2], f[0], f[1]))
    return {"start": start, "full": tfull, "level": level, "flows": flows, "first_src": first_src,
            "steps": steps, "down": down}


def end_time(scene: dict) -> float:
    return (scene["n_frames"] - 1) / FPS


def frame_of(t: float) -> int:
    """First frame that shows an event at time t (frames show time f / 24)."""
    return math.ceil(t * FPS - 1e-9)


def clock_tenths(frame: int) -> int:
    return frame * 10 // FPS


def run_world(scene: dict, horizon=None, **kw) -> dict | None:
    """Re-simulate the scene, optionally changed: rate_mult, move, drop_tap, unchanged,
    violation (False = no impossible flow), walls (override), blocked (sealed cells)."""
    violation = kw.pop("violation", True)
    walls = kw.pop("walls", scene["walls"])
    blocked = kw.pop("blocked", ())
    extra, drop = graph_mods(scene, violation)
    srcs = sources_of(scene, violation=violation, **kw)
    return flood(scene["R"], scene["C"], scene["depth"], walls, srcs, extra, drop,
                 horizon=end_time(scene) + 1.0 if horizon is None else horizon, blocked=blocked)


# ------------------------------------------------------------------ building
TERRAINS = ("random", "cone", "tilt")


def _depths(rng, N, terrain):
    """A shuffle of 1..N*N (all distinct, as in the original), optionally shaped:
    `random` (plain shuffle), `cone` (high around a peak), `tilt` (high along one
    side), each with noise, then ranked into 1..N*N. Returns (depths, peak)."""
    n = N * N
    peak = rng.randrange(n)
    pr, pc = divmod(peak, N)
    if terrain == "cone":
        target = [-(abs(i // N - pr) + abs(i % N - pc)) + rng.uniform(-1.6, 1.6) for i in range(n)]
    elif terrain == "tilt":
        ang = rng.uniform(0, 2 * math.pi)
        dx, dy = math.cos(ang), math.sin(ang)
        target = [(i % N) * dx + (i // N) * dy + rng.uniform(-1.4, 1.4) for i in range(n)]
        peak = max(range(n), key=lambda i: (target[i], -i))
    else:
        target = [rng.random() for _ in range(n)]
    order = sorted(range(n), key=lambda i: (target[i], i))
    d = [0] * n
    for rank, i in enumerate(order):
        d[i] = rank + 1
    return d, peak


def _layout(rng, N):
    """The original board: a header strip (the clock) above an N x N grid, on a
    square canvas placed in the 1280 x 720 frame (centred, or shifted sideways)."""
    S = rng.randint(640, 700)
    head = int(0.11 * S)
    m = int(0.035 * S)
    cell = (S - head - 2 * m) // N
    grid = cell * N
    bx = int((1280 - S) / 2 + rng.uniform(-0.6, 0.6) * (1280 - S) / 2)
    by = (720 - S) // 2
    return {"bx": bx, "by": by, "S": S, "head": head, "cell": cell,
            "x0": bx + (S - grid) // 2, "y0": by + head + m}


def _gaps_ok(times, end) -> bool:
    ts = sorted(times)
    for a, b in zip(ts, ts[1:]):
        if b - a < MIN_GAP_FRAMES / FPS - 1e-9:
            return False
    for t in ts:
        if t < EDGE_MARGIN_S or t > end - EDGE_MARGIN_S:
            return False
        fr = t * FPS - math.floor(t * FPS)
        if fr < FRAME_SAFE or fr > 1 - FRAME_SAFE:
            return False
    return True


def _pick_frames(rng, w) -> int | None:
    """Clip length in frames: the longest <= a random cap (7-10 s) with no event in
    the last 0.5 s."""
    cap = int(rng.uniform(7.0, 10.0) * FPS)
    steps = w["steps"]
    for nf in range(cap, int(5.5 * FPS) - 1, -1):
        end = (nf - 1) / FPS
        if any(end - EDGE_MARGIN_S < s <= end + 0.05 for s in steps):
            continue
        return nf
    return None


def reach(down, i) -> int:
    """How many sections water poured into i can ever spill to (i included)."""
    seen, stack = set(), [i]
    while stack:
        j = stack.pop()
        if j == GUTTER or j in seen:
            continue
        seen.add(j)
        stack.extend(down[j])
    return len(seen)


def _rate_for(rng, depth_e):
    """A pour rate (units/s, to 0.25) that fills the entry section in about 1.2-3.2 s."""
    return max(0.25, round(depth_e / rng.choice(FILL_SECONDS) * 4) / 4)


def _try(rng, scenario, variant):
    N = rng.randint(5, 8)
    n = N * N
    terrain = rng.choice(TERRAINS)
    depth, peak = _depths(rng, N, terrain)
    scene = {"scenario": scenario, "R": N, "C": N, "terrain": terrain, "depth": depth, "walls": [], "taps": [],
             "violation": None, "layout": _layout(rng, N), "palette": rng.randrange(PALETTES)}
    base0 = children(N, N, depth, [])
    outlets = [i for i in range(n) if GUTTER not in base0[i]]
    big = [i for i in outlets if reach(base0, i) >= 6]
    if not big:
        return None
    f_on = rng.randint(12, 30)
    if terrain == "random":
        e1 = rng.choice(big)
    else:
        near_peak = [i for i in big if abs(i // N - peak // N) + abs(i % N - peak % N) <= 1]
        e1 = peak if peak in big and rng.random() < 0.6 else rng.choice(near_peak or big)
    rate = _rate_for(rng, depth[e1])
    scene["taps"].append({"cell": e1, "colour": ENTRY_COLOURS[0], "segs": [[f_on, None, rate]], "change": None})
    if scenario == "two_entries":
        far = [i for i in big if i != e1 and i not in nbrs(N, N, e1)]
        if not far:
            return None
        e2 = rng.choice(far)
        f2 = f_on + rng.randint(24, 96)
        scene["taps"].append({"cell": e2, "colour": ENTRY_COLOURS[1],
                              "segs": [[f2, None, _rate_for(rng, depth[e2])]], "change": None})
    elif scenario == "pour_change":
        if variant == "stop":
            # stop only once the flood is under way: >= 5 wet and >= 2 full sections 0.3 s before
            w0 = run_world(scene, horizon=10.5)
            ok = [k for k in range(f_on + 36, f_on + 171) if tap_time(k) <= 7.6
                  and sum(1 for t in w0["start"] if t is not None and t <= tap_time(k) - 0.3) >= 5
                  and sum(1 for t in w0["full"] if t is not None and t <= tap_time(k) - 0.3) >= 2]
            if not ok:
                return None
            fc = rng.choice(ok)
        else:
            fc = f_on + rng.randint(36, 150)
        scene["taps"][0]["segs"] = [[f_on, fc, rate]] + ([[fc, None, rate * 2]] if variant == "double" else [])
        scene["taps"][0]["change"] = variant
    elif scenario == "walled":
        w0 = run_world(scene, horizon=10.5)
        used = sorted({(min(a, b), max(a, b)) for a, b, t0, _ in w0["flows"] if b != GUTTER and t0 <= 9.0})
        pairs = sorted({(min(i, j), max(i, j)) for i in range(n) for j in nbrs(N, N, i)})
        for _ in range(12):
            k_hot = min(len(used), rng.randint(1, 2))
            walls = rng.sample(used, k_hot) if used else []
            rest = [p for p in pairs if p not in walls]
            walls += rng.sample(rest, rng.randint(1, 3))
            if reach(children(N, N, depth, walls), e1) >= 6:
                break
        else:
            return None
        scene["walls"] = [list(p) for p in sorted(walls)]
    elif scenario == "broken_flow":
        w0 = run_world(scene, horizon=10.5)
        if variant == "uphill_flow":
            cands = []
            for s in range(n):
                ts = w0["full"][s]
                if ts is None or not 1.0 <= ts <= 7.0:
                    continue
                for v in nbrs(N, N, s):
                    st_v = w0["start"][v]
                    if depth[v] > depth[s] and (st_v is None or st_v > ts + 0.3):
                        cands.append((s, v))
            if not cands:
                return None
            s, v = rng.choice(cands)
            scene["violation"] = {"kind": "uphill_flow", "src": s, "dst": v}
        else:
            fv = rng.randint(40, 150)
            tv = tap_time(fv)
            # a section the water really does reach, but only later: shown filling too early
            cands = [i for i in range(n) if i != e1 and w0["start"][i] is not None and w0["start"][i] > tv + 0.5]
            if not cands:
                return None
            v = rng.choice(cands)
            scene["violation"] = {"kind": "premature_fill", "cell": v, "frame": fv,
                                  "rate": round(rng.uniform(0.25, 0.6) * depth[v], 2)}
    w = run_world(scene, horizon=10.5)
    if w is None:
        return None
    nf = _pick_frames(rng, w)
    if nf is None:
        return None
    scene["n_frames"] = nf
    scene["duration"] = round(nf / FPS, 6)
    w = run_world(scene)
    scene["track"] = _track(w)
    return scene


def _r6(x):
    return None if x is None else round(x, 6)


def _track(w) -> dict:
    return {"start": [_r6(t) for t in w["start"]], "full": [_r6(t) for t in w["full"]],
            "level": [[[_r6(a), round(v, 6), round(s, 9)] for a, v, s in lv] for lv in w["level"]],
            "flows": [[a, b, _r6(t0), _r6(t1)] for a, b, t0, t1 in w["flows"]],
            "first_src": w["first_src"], "steps": [_r6(t) for t in w["steps"]]}


def _ladder_ok(scene, rng) -> bool:
    """Dry-run the question builder on a copy of the rng state: the real call that
    follows build() consumes the same stream, so it is then guaranteed to succeed."""
    import random
    import questions
    probe = random.Random()
    probe.setstate(rng.getstate())
    try:
        questions.build(scene, probe)
    except RuntimeError:
        return False
    return True


VARIANTS = {"pour_change": ("stop", "double"), "broken_flow": ("premature_fill", "uphill_flow")}


def build(rng) -> dict:
    """Scenario and its variant are drawn once per seed (before any retry), so retries
    can't skew the mix; everything else is re-drawn until the scene passes check()
    and can carry a full 10-question ladder."""
    scenario = rng.choice(SCENARIOS)
    variant = rng.choice(VARIANTS[scenario]) if scenario in VARIANTS else None
    for attempt in range(1, 400):
        scene = _try(rng, scenario, variant)
        if scene is None or check(scene):
            continue
        scene["attempts"] = attempt
        if _ladder_ok(scene, rng):
            return scene
    raise RuntimeError("no valid scene in 400 attempts")


# ------------------------------------------------------------------ checks
def check(scene: dict) -> list[str]:
    """Reasons to reject a scene; [] means it is clean."""
    errs = []
    n = scene["R"] * scene["C"]
    depth, tr = scene["depth"], scene["track"]
    end = end_time(scene)
    if sorted(depth) != list(range(1, n + 1)):
        errs.append("depths are not a shuffle of 1..N*N")
    w = run_world(scene)
    if w is None:
        return errs + ["spill graph has a cycle"]
    if _track(w) != tr:
        errs.append("stored track differs from a fresh simulation")
    events = sorted({t for t in w["steps"] if t <= end + 0.05})
    if not _gaps_ok(events, end):
        errs.append("events too close together, to a frame boundary or to the clip edges")
    wet = [i for i in range(n) if tr["start"][i] is not None and tr["start"][i] <= end]
    if len(wet) < 5:
        errs.append("fewer than 5 sections get water")
    if sum(1 for t in tr["full"] if t is not None and t <= end) < 2:
        errs.append("fewer than 2 sections fill up")
    if len({t["cell"] for t in scene["taps"]}) != len(scene["taps"]):
        errs.append("two entries share a section")
    if scene["layout"]["cell"] < 60:
        errs.append("cells too small")
    v = scene["violation"]
    if scene["scenario"] == "broken_flow":
        if not v:
            errs.append("broken_flow without a violation")
        elif v["kind"] == "uphill_flow":
            s, d = v["src"], v["dst"]
            on = [f for f in tr["flows"] if f[0] == s and f[1] == d]
            if not on or on[0][2] > end - 0.5 or tr["first_src"][d] != [s]:
                errs.append("uphill flow not shown clearly")
        else:
            c = v["cell"]
            st = tr["start"][c]
            if st is None or st > end - 0.5:
                errs.append("premature fill happens too late")
            else:
                feeds = [f[2] for f in tr["flows"] if f[1] == c]
                if feeds and min(feeds) < st + 0.5:
                    errs.append("premature fill is covered by a real inflow within 0.5 s")
    elif v:
        errs.append("violation outside broken_flow")
    if scene["scenario"] == "pour_change":
        fc = scene["taps"][0]["segs"][0][1]
        if tap_time(fc) > end - 1.0:
            errs.append("pour change too late to see")
    if scene["scenario"] == "two_entries":
        if tap_time(scene["taps"][1]["segs"][0][0]) > end - 1.0:
            errs.append("second entry starts too late")
    return errs
