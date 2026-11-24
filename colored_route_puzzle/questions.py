"""10-question ladder for colored_route_puzzle (common/ladder5.py, LADDER = "ten").

Routes are named by colour ("the teal route"; every colour is used once per
clip), stops by shape ("the star stop"; every shape at most once), the ring
as "the black ring", and moments by the on-screen clock ("when the clock
shows 3.4 s"). Every question has 4 options and stores derivation_key +
params (route / stop ids, clock tenths, option sets) + min_fps for
answers.py.

Margins: compared start / finish times differ by >= 8 frames (0.33 s);
clock moments lie >= 2 frames from every event on every frame that shows
that reading (and, for "growing" questions, past the first quarter of the
growth); clock answers are offered in 0.5 s steps; the longest route beats
the runner-up by >= 15 % and the slowest-drawn route by >= 8 frames.
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
FPS = sim.FPS
FPS_STEPS = (1, 2, 3, 4, 6, 8, 12, 24)
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
NEVER = "it would never become possible"
NO_WAY = "it could not get there"
DEAD = "its dead end (the crossbar)"
TRAVEL = "(changing routes only at stops)"


def fps_for(gap_frames: float) -> int:
    gap = gap_frames / FPS
    for f in FPS_STEPS:
        if f * gap >= 1 - 1e-9:
            return f
    return 24


def lab(r) -> str:
    return f"the {r['color']} route"


def slab(scene, sid) -> str:
    return f"the {sim.shape(scene, sid)} stop"


def pair_text(scene, a, b) -> str:
    a, b = sorted((a, b), key=lambda s: sim.SHAPES.index(sim.shape(scene, s)))
    return f"{slab(scene, a)} and {slab(scene, b)}"


def clock_opts(tenths: int, nf: int) -> list:
    top = (10 * (nf - 1)) // FPS
    out = []
    for k in (1, -1, 2, -2, 3, -3, 4, -4):
        v = tenths + 5 * k
        if 1 <= v <= top:
            out.append(sim.clock_text(v))
    return out


def _moments(scene, lo, hi):
    """Clock tenths whose frames all lie inside [lo, hi] and >= 0.5 s from both clip ends."""
    nf = scene["n_frames"]
    lo, hi = max(lo, sim.EDGE_F), min(hi, nf - 1 - sim.EDGE_F)
    if hi < lo:
        return []
    out = []
    for t in range(sim.timer_tenths(lo), sim.timer_tenths(hi) + 1):
        a, b = sim.frames_showing(t)
        if lo <= a and b <= hi:
            out.append(t)
    return out


def _stable_windows(scene):
    """(a, b) between consecutive events: frames a+2..b-2 see no change and no event within 2 frames."""
    fr = sorted(set(sim.event_frames(scene)))
    out = []
    for a, b in zip(fr, fr[1:]):
        if b - a >= 2 * sim.GAP_F + 3:
            out.append((a, b))
    return out


def reach_count(scene, rids, sid) -> int:
    return len(sim.reach(scene, rids, sid)) - 1


def first_connect(scene, pair, skip=None):
    """Route whose completion first lets you travel pair[0] -> pair[1]; (route id, frame) or (None, None)."""
    a, b = pair
    for f, kind, rid in sim.events(scene):
        if kind != "finish" or rid == skip:
            continue
        ids = [x for x in sim.complete_ids(scene, f) if x != skip]
        if b in sim.reach(scene, ids, a):
            return rid, f
    return None, None


def connect_in_order(scene, order, pair):
    a, b = pair
    for k in range(len(order)):
        if b in sim.reach(scene, order[:k + 1], a):
            return order[k]
    return None


def shortest_rides(scene, rids, a, b):
    adj = {s["id"]: set() for s in scene["stops"]}
    for rid in rids:
        r = scene["routes"][rid]
        if sim.is_normal(r):
            adj[r["a"]].add(r["b"])
            adj[r["b"]].add(r["a"])
    dist, todo = {a: 0}, [a]
    while todo:
        nxt = []
        for x in todo:
            for y in sorted(adj[x]):
                if y not in dist:
                    dist[y] = dist[x] + 1
                    nxt.append(y)
        todo = nxt
    return dist.get(b)


def candidates(scene, rng):
    """Every askable question, grouped by level: (key, params, text, gt, distractors, motion, min_fps)."""
    R, S, sc, nf = scene["routes"], scene["stops"], scene["scenario"], scene["n_frames"]
    n, ns = len(R), len(S)
    ev = sim.event_frames(scene)
    hold = nf - 1 - ev[-1]
    order = [r["id"] for r in sim.start_order(scene)]
    st = {r["id"]: r["draw"][0] for r in R}
    fin = {r["id"]: r["draw"][1] for r in R}
    erased = [r for r in R if r["erase"]]
    final = sim.complete_ids(scene, nf - 1)
    allr = [r["id"] for r in R]
    normal = [r for r in R if sim.is_normal(r)]
    styles = sorted({r["style"] for r in R})
    tk = scene["token"]
    C = {lv: [] for lv in range(1, 6)}
    seen = set()

    def add(lv, key, params, text, gt, dis, motion, gap):
        sig = (key, json.dumps(params, sort_keys=True))
        if gt is None or sig in seen:
            return
        ds = list(dict.fromkeys(str(d) for d in dis if str(d) != str(gt)))
        if len(ds) < 3:
            return
        seen.add(sig)
        C[lv].append((key, dict(params), text, gt, ds, motion, fps_for(gap)))

    def others(ex, pref=()):
        ex = set(ex)
        out = [lab(R[j]) for j in pref if j not in ex]
        rest = [j for j in allr if j not in ex and j not in pref]
        rng.shuffle(rest)
        return out + [lab(R[j]) for j in rest]

    def stop_others(ex, pref=()):
        ex = set(ex)
        out = [slab(scene, j) for j in pref if j not in ex]
        rest = [s["id"] for s in S if s["id"] not in ex and s["id"] not in pref]
        rng.shuffle(rest)
        return out + [slab(scene, j) for j in rest]

    def visible(r):
        """frames for which a route can be seen complete (gap for static-looking questions)."""
        if r["erase"]:
            return r["erase"][0] - r["draw"][1]
        return nf - 1 - r["draw"][1]

    def start_gap(ids):
        s = sorted(st[j] for j in ids)
        return min(b - a for a, b in zip(s, s[1:]))

    def fin_gap(ids):
        s = sorted(fin[j] for j in ids)
        return min(b - a for a, b in zip(s, s[1:]))

    in_erase = sc == "erase"
    during = " during the video" if in_erase else ""
    end_pref = "At the end of the video, h" if in_erase else "H"

    # ------------------------------------------------------------ L1 observation
    add(1, "n_routes", {}, f"How many coloured routes are drawn{during}?", n,
        ([len(final)] if in_erase else []) + L.near_ints(n, 1, 12), in_erase,
        min(visible(r) for r in erased) if in_erase else hold)
    add(1, "n_stops", {}, "How many stops (dark shape markers) are on the map?", ns, L.near_ints(ns, 2, 9), False, nf)
    if in_erase:
        add(1, "n_end", {}, "How many coloured routes are still on screen at the end of the video?", len(final),
            [n] + L.near_ints(len(final), 1, 12), False, hold)
    for style in styles:
        k = sum(1 for r in R if r["style"] == style)
        if k == n:
            continue
        mot = in_erase and any(r["style"] == style for r in erased)
        add(1, "n_style", {"style": style}, f"How many routes are drawn as {style} lines{during}?", k,
            L.near_ints(k, 0, 9), mot, min([visible(r) for r in erased] + [hold]) if mot else hold)
    add(1, "first_drawn", {}, "Which route is the first to start being drawn?", lab(R[order[0]]),
        others([order[0]], order[1:4]), True, st[order[1]] - st[order[0]])
    add(1, "last_started", {}, "Which route is the last to start being drawn?", lab(R[order[-1]]),
        others([order[-1]], order[-2:-5:-1]), True, st[order[-1]] - st[order[-2]])
    for r in erased:
        keep = [j for j in allr if not R[j]["erase"]]
        opts = sorted([r["id"]] + rng.sample(keep, 3))
        add(1, "erased_which", {"route": r["id"], "options": opts}, "Which of these routes disappears during the video?",
            lab(r), [lab(R[j]) for j in opts if j != r["id"]], True,
            min(r["erase"][0] - r["draw"][0], nf - 1 - r["erase"][1]))
    if sc == "dead_ends":
        k = sum(1 for r in R if not sim.is_normal(r))
        add(1, "n_dead", {}, "How many routes stop at a dead end (a short crossbar) instead of reaching a second stop?",
            k, L.near_ints(k, 0, 6), False, hold)
    if tk:
        rides = tk["rides"]
        visited = [tk["start"]] + [rd["to"] for rd in rides]
        add(1, "token_start", {}, "On which stop does the black ring sit at the start of the video?",
            slab(scene, tk["start"]), stop_others([tk["start"]], visited[1:]), False, rides[0]["t"][0])
        add(1, "token_end", {}, "On which stop is the black ring at the end of the video?",
            slab(scene, rides[-1]["to"]), stop_others([rides[-1]["to"]], visited[:-1]), False, nf - 1 - rides[-1]["t"][1])

    # ------------------------------------------------------------ L2 relational
    for r in normal:
        a, b = r["a"], r["b"]
        prs = [(x, y) for x in range(ns) for y in range(x + 1, ns) if {x, y} != {a, b}]
        share = [p for p in prs if a in p or b in p]
        rest = [p for p in prs if p not in share]
        rng.shuffle(share)
        rng.shuffle(rest)
        mot = bool(r["erase"])
        add(2, "route_ends", {"route": r["id"]}, f"Which two stops does {lab(r)} connect?", pair_text(scene, a, b),
            [pair_text(scene, *p) for p in share + rest], mot, visible(r) if mot else hold)
    for r in R:
        anc = r["a"] if r["grow"] == "a" else r["b"]
        other = r["b"] if r["grow"] == "a" else r["a"]
        gt = DEAD if anc is None else slab(scene, anc)
        dis = [DEAD if other is None else slab(scene, other)] + stop_others([x for x in (anc, other) if x is not None])
        add(2, "grow_from", {"route": r["id"]}, f"From which end does {lab(r)} start growing?", gt, dis, True,
            r["draw"][1] - r["draw"][0])
    pool = order[1:]
    for _ in range(3):
        if len(pool) < 4:
            break
        opts = sorted(rng.sample(pool, 4))
        if start_gap(opts) < sim.CMP_F:
            continue
        g = min(opts, key=lambda j: st[j])
        add(2, "earlier_of", {"options": opts}, "Which of these routes starts being drawn first?", lab(R[g]),
            [lab(R[j]) for j in opts if j != g], True, start_gap(opts))
    for k, x in enumerate(order[:-1]):
        nx = order[k + 1]
        gaps = [st[nx] - st[x]] + ([st[order[k + 2]] - st[nx]] if k + 2 < n else [])
        if min(gaps) < sim.CMP_F:
            continue
        add(2, "next_start", {"route": x}, f"Which route is the next one to start being drawn after {lab(R[x])} starts?",
            lab(R[nx]), others([x, nx], order[k + 2:k + 3] + order[max(0, k - 1):k]), True, min(gaps))
    if sc == "overlap":
        for _ in range(40):
            opts = sorted(rng.sample(allr, 4))
            if fin_gap(opts) < sim.CMP_F:
                continue
            g = min(opts, key=lambda j: fin[j])
            if g == min(opts, key=lambda j: st[j]):
                continue
            add(2, "first_complete", {"options": opts}, "Which of these routes is the first to be completely drawn?",
                lab(R[g]), [lab(R[j]) for j in opts if j != g], True, fin_gap(opts))
    for s in S:
        k = sum(1 for j in (final if in_erase else allr) if sim.touches(R[j], s["id"]))
        alt = sum(1 for j in allr if sim.touches(R[j], s["id"]))
        add(2, "degree", {"stop": s["id"]}, f"{end_pref}ow many routes end at {slab(scene, s['id'])}?", k,
            ([alt] if alt != k else []) + L.near_ints(k, 0, 4), False, hold)
    for a in range(ns):
        for b in range(a + 1, ns):
            k = sum(1 for j in (final if in_erase else allr) if sim.is_normal(R[j]) and {R[j]["a"], R[j]["b"]} == {a, b})
            if k == 0 and rng.random() < 0.7:
                continue
            add(2, "pair_count", {"pair": [a, b]},
                f"{end_pref}ow many routes directly connect {pair_text(scene, a, b)}?", k, L.near_ints(k, 0, 4), False,
                hold)
    if tk:
        rides = tk["rides"]
        prev = max(fin.values())
        for k, rd in enumerate(rides):
            after = (rides[k + 1]["t"][0] if k + 1 < len(rides) else nf - 1) - rd["t"][1]
            gap = min(rd["t"][1] - rd["t"][0], rd["t"][0] - prev, after)
            prev = rd["t"][1]
            ridden = [x["route"] for x in rides if x["route"] != rd["route"]]
            near = [j for j in allr if sim.touches(R[j], rd["from"]) or sim.touches(R[j], rd["to"])]
            add(2, "token_ride", {"k": k + 1}, f"Which route does the black ring ride along on its {ORD[k]} trip?",
                lab(R[rd["route"]]), others([rd["route"]], ridden + near), True, gap)
            if k + 1 < len(rides):
                add(2, "token_dest", {"k": k + 1},
                    f"At which stop does the black ring arrive at the end of its {ORD[k]} trip?", slab(scene, rd["to"]),
                    stop_others([rd["to"]], [x["to"] for x in rides] + [tk["start"]]), True, gap)

    # ------------------------------------------------------------ L3 measurement
    ranked = sorted(R, key=lambda r: -sim.length_steps(r))
    if sim.length_steps(ranked[0]) >= 1.15 * sim.length_steps(ranked[1]):
        mot = any(r["erase"] for r in ranked[:4])
        add(3, "longest", {}, f"Measured along the line, which route drawn{during} is the longest?"
            if in_erase else "Measured along the line, which route is the longest?",
            lab(ranked[0]), [lab(r) for r in ranked[1:]], mot,
            min([visible(r) for r in ranked[:4] if r["erase"]] + [hold]))
    for r in R:
        k = sim.n_corners(r)
        add(3, "corners", {"route": r["id"]}, f"How many corners (turns) does {lab(r)} make?", k,
            L.near_ints(k, 0, 16), bool(r["erase"]), visible(r) if r["erase"] else hold)
        t = sim.timer_tenths(r["draw"][1])
        add(3, "finish_clock", {"route": r["id"]},
            f"What does the clock show at the moment {lab(r)} becomes completely drawn?", sim.clock_text(t),
            clock_opts(t, nf), True, 6)
        lo, hi = sim.visible_growth(r)
        ts = _moments(scene, lo, hi)
        if ts:
            t = rng.choice(ts)
            a, b = sim.frames_showing(t)
            idle = [o["id"] for o in R if o is not r and all(
                not (o["draw"][0] - sim.GAP_F <= f <= o["draw"][1] + sim.GAP_F) for f in range(a, b + 1))]
            if len(idle) >= 3:
                opts = sorted([r["id"]] + rng.sample(idle, 3))
                add(3, "drawing_at", {"route": r["id"], "t": t, "options": opts},
                    f"Which of these routes is growing (being drawn) when the clock shows {sim.clock_text(t)}?",
                    lab(r), [lab(R[j]) for j in opts if j != r["id"]], True, r["draw"][1] - r["draw"][0])
    wins = _stable_windows(scene)
    picks = rng.sample(wins, min(3, len(wins)))
    for a, b in picks:
        ts = _moments(scene, a + sim.GAP_F, b - sim.GAP_F)
        if not ts:
            continue
        t = rng.choice(ts)
        f0 = sim.frames_showing(t)[0]
        k = len(sim.complete_ids(scene, f0))
        grow = sum(1 for r in R if sim.growing_at(r, f0))
        add(3, "n_complete_at", {"t": t},
            f"How many routes are completely drawn and on screen when the clock shows {sim.clock_text(t)}?", k,
            ([k + grow] if grow else []) + L.near_ints(k, 0, 9), True, b - a)
        if sc == "overlap" and grow:
            lo_ok = all(not sim.growing_at(r, f0) or sim.visible_growth(r)[0] <= f0 for r in R)
            if lo_ok:
                add(3, "n_growing_at", {"t": t},
                    f"How many routes are growing (being drawn) at the same time when the clock shows "
                    f"{sim.clock_text(t)}?", grow, L.near_ints(grow, 0, 6), True, b - a)
    dur = sorted(R, key=lambda r: -(r["draw"][1] - r["draw"][0]))
    d0, d1 = (dur[0]["draw"][1] - dur[0]["draw"][0]), (dur[1]["draw"][1] - dur[1]["draw"][0])
    if d0 - d1 >= sim.CMP_F:
        add(3, "slowest", {}, "Which route takes the longest time to draw, from first appearing to complete?",
            lab(dur[0]), [lab(r) for r in dur[1:]], True, d0 - d1)
    for r in erased:
        t = sim.timer_tenths(r["erase"][1])
        add(3, "erase_clock", {"route": r["id"]},
            f"What does the clock show at the moment {lab(r)} has completely disappeared?", sim.clock_text(t),
            clock_opts(t, nf), True, 6)
    if tk:
        dests = [rd["to"] for rd in tk["rides"]]
        for s in sorted(set(dests)):
            if dests.count(s) != 1:
                continue
            rd = tk["rides"][dests.index(s)]
            t = sim.timer_tenths(rd["t"][1])
            add(3, "arrive_clock", {"stop": s}, f"What does the clock show when the black ring arrives at "
                f"{slab(scene, s)}?", sim.clock_text(t), clock_opts(t, nf), True, 6)

    # ------------------------------------------------------------ L4 integration
    comp_ev = sorted([r["draw"][1] for r in R] + [r["erase"][0] for r in erased])
    pairs = [(a, b) for a in range(ns) for b in range(a + 1, ns)]
    for p in pairs:
        x, f = first_connect(scene, p)
        if x is None:
            continue
        k = comp_ev.index(f)
        nb = ([f - comp_ev[k - 1]] if k else []) + ([comp_ev[k + 1] - f] if k + 1 < len(comp_ev) else [])
        gap = min(nb) if nb else f
        if gap < sim.CMP_F:
            continue
        onpath = [j for j in allr if sim.is_normal(R[j]) and (sim.touches(R[j], p[0]) or sim.touches(R[j], p[1]))]
        add(4, "connect_by", {"pair": list(p)},
            f"Which route, once completely drawn, first makes it possible to travel from {slab(scene, p[0])} to "
            f"{slab(scene, p[1])} along the routes {TRAVEL}?", lab(R[x]), others([x], onpath), True, gap)
    for style in styles:
        k_all = sum(1 for r in R if r["style"] == style)
        if k_all == n:
            continue
        x = next(j for j in order if R[j]["style"] == style)
        k = len(sim.complete_ids(scene, st[x]))
        near = min(abs(st[x] - f) for f in comp_ev)
        if near < sim.CMP_F:
            continue
        add(4, "n_before_style", {"style": style},
            f"How many routes are already completely drawn at the moment the first {style} route starts to appear?",
            k, [order.index(x)] + L.near_ints(k, 0, 9), True, near)
    for s in S:
        touch = [j for j in order if sim.touches(R[j], s["id"])]
        if len(touch) < 2 or st[touch[1]] - st[touch[0]] < sim.CMP_F:
            continue
        add(4, "first_at_stop", {"stop": s["id"]},
            f"Of the routes that end at {slab(scene, s['id'])}, which one starts being drawn first?",
            lab(R[touch[0]]), others([touch[0]], touch[1:]), True, st[touch[1]] - st[touch[0]])
    for k in range(2, n + 1):
        x = order[k - 1]
        gaps = [st[x] - st[order[k - 2]]] + ([st[order[k]] - st[x]] if k < n else [])
        if min(gaps) < sim.CMP_F:
            continue
        add(4, "kth_start", {"k": k}, f"Which route is the {ORD[k - 1]} one to start being drawn?", lab(R[x]),
            others([x], order[max(0, k - 3):k + 1]), True, min(gaps))
    for a, b in rng.sample(wins, min(3, len(wins))):
        ts = _moments(scene, a + sim.GAP_F, b - sim.GAP_F)
        if not ts:
            continue
        t = rng.choice(ts)
        f0 = sim.frames_showing(t)[0]
        ids = sim.complete_ids(scene, f0)
        s = rng.choice(S)["id"]
        k = reach_count(scene, ids, s)
        add(4, "reach_at", {"stop": s, "t": t},
            f"When the clock shows {sim.clock_text(t)}, how many other stops can be reached from "
            f"{slab(scene, s)} along the completely drawn routes {TRAVEL}?", k,
            [reach_count(scene, final, s)] + L.near_ints(k, 0, ns - 1), True, b - a)
    if tk:
        rides = tk["rides"]
        prev, gaps = max(fin.values()), []
        for rd in rides:
            gaps += [rd["t"][0] - prev, rd["t"][1] - rd["t"][0]]
            prev = rd["t"][1]
        vis = len({tk["start"]} | {rd["to"] for rd in rides})
        add(4, "token_n_stops", {}, "How many different stops does the black ring visit, counting the one it starts on?",
            vis, [len(rides) + 1] + L.near_ints(vis, 1, 6), True, min(gaps))
        tc = sum(sim.n_corners(R[rd["route"]]) for rd in rides)
        add(4, "token_corners", {}, "Adding up the corners (turns) of every route the black ring rides along, how many "
            "corners are there in total?", tc, L.near_ints(tc, 0, 40), True, min(gaps))
    if in_erase:
        best = max(len(sim.complete_ids(scene, i)) for i in range(nf))
        runs, cur = [], 0
        for i in range(nf):
            if len(sim.complete_ids(scene, i)) == best:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        add(4, "max_complete", {}, "What is the largest number of completely drawn routes on screen at the same moment?",
            best, [n, len(final)] + L.near_ints(best, 1, 9), True, max(runs))

    # ------------------------------------------------------------ L5 counterfactual
    for p in pairs:
        x, f = first_connect(scene, p)
        if x is None:
            continue
        y, fy = first_connect(scene, p, skip=x)
        gt = NEVER if y is None else lab(R[y])
        key_f = [f] + ([fy] if y is not None else [])
        sep = min(abs(a - b) for a in key_f for b in comp_ev if b != a)
        if sep < sim.CMP_F:
            continue
        add(5, "cf_skip_connect", {"route": x, "pair": list(p)},
            f"Suppose {lab(R[x])} had never been drawn (everything else the same). Which route, once completely "
            f"drawn, would first make it possible to travel from {slab(scene, p[0])} to {slab(scene, p[1])} {TRAVEL}?",
            gt, [lab(R[x])] + ([NEVER] if y is not None else []) + others([x] + ([y] if y is not None else [])),
            True, sep)
        if not in_erase:
            z = connect_in_order(scene, order[::-1], p)
            if z is not None and z != x:
                add(5, "cf_reverse", {"pair": list(p)},
                    f"Suppose the routes had been drawn one at a time in the reverse order (the last one first). Which "
                    f"route, once completely drawn, would first make it possible to travel from {slab(scene, p[0])} to "
                    f"{slab(scene, p[1])} {TRAVEL}?", lab(R[z]), [lab(R[x])] + others([x, z]), True, start_gap(allr))
    base = final
    for r in normal:
        if in_erase and r["id"] not in final:
            continue
        for s in (r["a"], r["b"]):
            k = reach_count(scene, [j for j in base if j != r["id"]], s)
            k0 = reach_count(scene, base, s)
            if k == k0 and rng.random() < 0.7:
                continue
            add(5, "cf_remove_reach", {"route": r["id"], "stop": s},
                f"Suppose {lab(r)} had never been drawn. How many other stops could be reached from {slab(scene, s)} "
                f"{'at the end of the video ' if in_erase else ''}along the routes {TRAVEL}?", k,
                [k0] + L.near_ints(k, 0, ns - 1), in_erase, hold)
    for x in order[1:]:
        new = [x] + [j for j in order if j != x]
        ks = [k for k in range(2, n + 1) if new[k - 1] != order[k - 1]]
        if not ks:
            continue
        k = rng.choice(ks)
        add(5, "cf_first", {"route": x, "k": k},
            f"Suppose {lab(R[x])} had been the first route drawn and the others had kept their order. Which route "
            f"would then be the {ORD[k - 1]} one to start being drawn?", lab(R[new[k - 1]]),
            [lab(R[order[k - 1]])] + others([new[k - 1], order[k - 1]], new[k - 2:k + 1]), True, start_gap(allr))
    for r in erased:
        for s in S:
            k = reach_count(scene, final + [r["id"]], s["id"])
            k0 = reach_count(scene, final, s["id"])
            if k == k0:
                continue
            add(5, "cf_no_erase", {"route": r["id"], "stop": s["id"]},
                f"Suppose {lab(r)} had never been erased. How many other stops could be reached from "
                f"{slab(scene, s['id'])} at the end of the video {TRAVEL}?", k, [k0] + L.near_ints(k, 0, ns - 1), True,
                visible(r))
    if tk:
        a, b = tk["start"], tk["rides"][-1]["to"]
        if a != b:
            nr = [j for j in allr if sim.is_normal(R[j])]
            for rd in tk["rides"]:
                k = shortest_rides(scene, [j for j in nr if j != rd["route"]], a, b)
                gt = NO_WAY if k is None else str(k)
                dis = [str(len(tk["rides"])), str(shortest_rides(scene, nr, a, b))] + \
                    ([NO_WAY] if k is not None else []) + [str(v) for v in L.near_ints(k or 2, 1, 6)]
                add(5, "cf_detour", {"route": rd["route"]},
                    f"Suppose {lab(R[rd['route']])} had never been drawn. What is the smallest number of routes the "
                    "black ring would have to ride to get from the stop where it starts to the stop where it ends?",
                    gt, dis, True, min(x["t"][1] - x["t"][0] for x in tk["rides"]))
    if not in_erase:
        for style in styles:
            if sum(1 for r in R if r["style"] == style) == n:
                continue
            rest = [r for r in R if r["style"] != style]
            k = sum(1 for s in S if not any(sim.touches(r, s["id"]) for r in rest))
            add(5, "cf_style_gone", {"style": style},
                f"Suppose every {style} route were removed. How many stops would have no route left at all?", k,
                L.near_ints(k, 0, ns), False, hold)
    return C


FEATURED = {"sequential": ("grow_from", "connect_by", "cf_reverse"),
            "overlap": ("first_complete", "n_growing_at", "earlier_of"),
            "dead_ends": ("n_dead", "grow_from"),
            "erase": ("erased_which", "n_end", "erase_clock", "max_complete", "cf_no_erase"),
            "traveller": ("token_start", "token_end", "token_ride", "token_dest", "arrive_clock", "token_n_stops",
                          "token_corners", "cf_detour")}


def _pick(rng, items, k, featured, level):
    """k items over as many derivation keys as possible; the scenario's own
    keys usually first; on L2-L5 at most one question that a still frame answers."""
    by = {}
    for it in items:
        by.setdefault(it[0], []).append(it)
    keys = sorted(by)
    rng.shuffle(keys)
    front = [key for key in keys if key in featured and rng.random() < 0.7]
    keys = front + [key for key in keys if key not in front]
    out, statics = [], 0
    for limit_static in (True, False):
        while len(out) < k:
            added = False
            for key in keys:
                pool = [it for it in by[key] if it not in out]
                if not pool or len(out) >= k:
                    continue
                if limit_static and level >= 2:
                    pool = [it for it in pool if it[5] or statics < 1]
                    if not pool:
                        continue
                it = rng.choice(pool)
                out.append(it)
                statics += 0 if it[5] else 1
                added = True
            if not added:
                break
    return out


def build(scene: dict, rng: random.Random) -> list:
    C = candidates(scene, rng)
    counts = L.ten_counts(rng)
    qs = []
    for lv in range(1, 6):
        picked = _pick(rng, C[lv], counts[lv - 1], FEATURED[scene["scenario"]], lv)
        if len(picked) != counts[lv - 1]:
            raise ValueError(f"level {lv}: only {len(picked)} questions")
        for key, params, text, gt, dis, motion, mf in picked:
            opts, let = L.mc(rng, str(gt), dis)
            if len(opts) != 4:
                raise ValueError(f"{key}: fewer than 4 options")
            qs.append(L.question(lv, key, text, gt, opts, let, motion, {**params, "min_fps": mf}))
    return L.number(qs)
