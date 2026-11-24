"""10-question ladder for bouncing_ball_restitution (common/ladder5.py, LADDER = "ten").

Objects are named by colour + pattern + shape ("the striped red ball"); pads
by colour ("the green pad"). Times are what the on-screen clock shows at the
frame nearest the event. Every question has 4 options and stores
derivation_key + params (object ids, times, ...) + min_fps for answers.py.

Margins (no near ties): compared times >= 0.3 s apart, compared heights
>= 15 px and >= 10 %, compared bounciness e^2 >= 0.1 apart, sizes >= 15 %,
counting windows >= 0.3 s from every hit, clock answers only for events
whose two bracketing frames show the same reading (sim.clean_time).
"""
from __future__ import annotations
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
FPS_STEPS = (1, 2, 3, 4, 6, 8, 12, 24)
GAP = 0.3            # min separation of compared / boundary times (s)
TIME_STEP = 0.5      # spacing of clock-reading options (s)
BUCKETS = (20, 35, 50, 65, 80)   # rebound height as % of drop height
BUCKET_TOL = 0.035


def fps_for(need: float) -> int:
    for f in FPS_STEPS:
        if f >= need - 1e-9:
            return f
    return 24


def _surface(scene):
    return "its shelf" if scene["scenario"] == "throw" else "the floor"


def _info(scene):
    d = scene["duration"]
    out = []
    for o in scene["objects"]:
        h = sim.history(scene, o)
        hits = [t for t in h["impacts"] if t < d]
        rest = h["rest"] if h["rest"] is not None and h["rest"] < d else None
        walls = [(t, s) for t, s in sim.wall_hits(scene, o) if t < d]
        seq = [o["release"]] + hits
        gaps = [b - a for a, b in zip(seq, seq[1:])]
        out.append({"o": o, "hist": h, "hits": hits, "rest": rest, "walls": walls,
                    "min_gap": min(gaps) if gaps else d, "e": sim.restitution(scene, o),
                    "apex1": h["apexes"][0] if h["apexes"] and h["impacts"][0] < d else None,
                    "rebounds": h["ups"][0] > 0 if h["ups"] else False})
    return out


def _decoys(rng, scene, k=4):
    used = {o["color"] for o in scene["objects"]} | {p["color"] for p in scene.get("pads", [])}
    free = [c for c, _ in sim.COLORS if c not in used]
    rng.shuffle(free)
    out = []
    for c in free[:k]:
        out.append(sim.name_of({"color": c, "pattern": rng.choice(sim.PATTERNS), "shape": rng.choice(sim.SHAPES)}))
    # too few unused colours: absent pattern/shape variants of the colours on screen
    real = {(o["color"], o["pattern"], o["shape"]) for o in scene["objects"]}
    for o in scene["objects"]:
        if len(out) >= k:
            break
        alt = [(o["color"], p, sh) for p in sim.PATTERNS for sh in sim.SHAPES
               if p != o["pattern"] and sh != o["shape"] and (o["color"], p, sh) not in real]
        c, p, sh = rng.choice(alt)
        out.append(sim.name_of({"color": c, "pattern": p, "shape": sh}))
    return out, free[k:] or free


def _clean_count_window(hits, a, b):
    return all(abs(t - a) >= GAP and abs(t - b) >= GAP for t in hits)


def _time_opts(gt_t: float, d: float):
    base = sim.timer_tenths(sim.frame_of(gt_t))
    cands = []
    for k in (1, -1, 2, -2, 3, -3, 4):
        v = base + 5 * k
        if 1 <= v <= int(d * 10):
            cands.append(f"{v // 10}.{v % 10} s")
    return cands


def candidates(scene, rng):
    """Every askable question for this scene, grouped by level (pure apart
    from the decoy names drawn from rng)."""
    d, sc = scene["duration"], scene["scenario"]
    objs = scene["objects"]
    inf = _info(scene)
    surf = _surface(scene)
    decoys, free_cols = _decoys(rng, scene)
    names = [sim.name_of(o) for o in objs]
    C = {lv: [] for lv in range(1, 6)}

    def add(lv, key, params, text, gt, dis, motion, mf):
        C[lv].append((key, params, text, gt, list(dis), motion, fps_for(mf)))

    def others(i, extra=()):
        return [names[j] for j in range(len(objs)) if j != i and j not in extra] + decoys

    # ---------------- L1 observation
    n = len(objs)
    add(1, "n_objects", {}, "How many bouncing objects are there in the video?", n, L.near_ints(n, 1), False, 1)
    for i, f in enumerate(inf):
        k = len(f["hits"])
        if 1 <= k <= 14:
            verb = "land on its shelf" if sc == "throw" else "hit the floor"
            add(1, "hits", {"obj": i}, f"How many times does {names[i]} {verb} during the video?", k,
                L.near_ints(k, 1), True, 2.0 / f["min_gap"])
    for which, sgn in (("largest", 1), ("smallest", -1)):
        rs = sorted(range(n), key=lambda i: -sgn * objs[i]["r"])
        a, b = objs[rs[0]]["r"], objs[rs[1]]["r"]
        if max(a, b) / min(a, b) >= 1.15:
            add(1, which, {}, f"Which object is the {which}?", names[rs[0]], others(rs[0]), False, 1)
    if sc != "throw":
        hs = sorted(range(n), key=lambda i: -objs[i]["h0"])
        top = lambda i: objs[i]["h0"] + 2 * objs[i]["r"]
        if objs[hs[0]]["h0"] - objs[hs[1]]["h0"] >= 25 and all(top(hs[0]) - top(j) >= 25 for j in hs[1:]):
            add(1, "highest_start", {}, "Which object starts highest above the floor?", names[hs[0]],
                others(hs[0]), False, 1)
    k_rest = sum(1 for f in inf if f["rest"] is not None)
    if sc != "throw":
        add(1, "n_resting_end", {}, "How many objects are lying still on the floor at the end of the video?", k_rest,
            L.near_ints(k_rest, 0, n), True, 2)
    if sc == "staggered":
        order = sorted(range(n), key=lambda i: objs[i]["release"])
        gap = objs[order[1]]["release"] - objs[order[0]]["release"]
        add(1, "first_released", {}, "Which object is released first?", names[order[0]], others(order[0]),
            True, 1.0 / gap)
    if sc == "pads":
        pads = scene["pads"]
        for i in range(n):
            pc = pads[objs[i]["lane"]]["color"]
            dis = [p["color"] for p in pads if p["color"] != pc] + free_cols
            add(1, "pad_under", {"obj": i}, f"What colour is the floor pad under {names[i]}?", pc, dis, False, 1)
    if sc == "throw":
        for i, f in enumerate(inf):
            k = len(f["walls"])
            if k <= 10:
                gaps = [b[0] - a[0] for a, b in zip(f["walls"], f["walls"][1:])] or [d]
                add(1, "wall_hits", {"obj": i}, f"How many times does {names[i]} bounce off a side wall?", k,
                    L.near_ints(k, 0), True, 1.0 / min(gaps))

    # ---------------- L2 relational
    firsts = sorted((f["hits"][0], i) for i, f in enumerate(inf) if f["hits"])
    for pos in range(min(2, len(firsts) - 1)):
        t0, w = firsts[pos]
        opp = [i for t, i in firsts[pos + 1:] if t - t0 >= GAP][:3]
        if opp:
            gap = min(inf[j]["hits"][0] for j in opp) - t0
            add(2, "first_hit", {"among": sorted([w] + opp)}, f"Which of these objects is the first to "
                f"{'land on its shelf' if sc == 'throw' else 'hit the floor'}?", names[w],
                [names[j] for j in opp] + decoys, True, 1.0 / gap)
    reb = [(f["apex1"] if f["rebounds"] else 0.0, i) for i, f in enumerate(inf) if f["apex1"] is not None]
    if len(reb) >= 2 and all(f["apex1"] is not None for f in inf):
        reb.sort(reverse=True)
        (a, w), (b, _) = reb[0], reb[1]
        hist = inf[w]["hist"]
        if a - b >= 15 and a >= 1.1 * b and len(hist["impacts"]) >= 2 and hist["impacts"][1] < d:
            add(2, "highest_rebound", {}, f"After its first bounce, which object rises highest above {'its own shelf' if sc == 'throw' else 'the floor'}?", names[w],
                others(w), True, 2.0 / (hist["impacts"][1] - hist["impacts"][0]))
    rests = sorted((f["rest"], i) for i, f in enumerate(inf) if f["rest"] is not None)
    if rests:
        t0, w = rests[0]
        later = [f["rest"] for f in inf if f["rest"] is not None and f["o"]["id"] != w]
        if all(t - t0 >= GAP for t in later):
            gap = min([t - t0 for t in later] + [inf[w]["min_gap"]])
            add(2, "first_rest", {}, "Which object is the first to stop bouncing?", names[w], others(w), True,
                max(1.0 / gap, 2.0 / inf[w]["min_gap"]))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in pairs:
        fi, fj = inf[i], inf[j]
        if fi["rebounds"] and fj["rebounds"] and abs(fi["e"] ** 2 - fj["e"] ** 2) >= 0.1:
            w = i if fi["e"] > fj["e"] else j
            mg = min(fi["min_gap"], fj["min_gap"])
            add(2, "bouncier_pair", {"a": i, "b": j},
                f"Which keeps more of its height from one bounce to the next: {names[i]} or {names[j]}?",
                names[w], [names[j if w == i else i], "they keep the same share", "neither of them bounces"],
                True, 2.0 / mg)
        ki, kj = len(fi["hits"]), len(fj["hits"])
        if ki and kj and ki <= 14 and kj <= 14:
            gt = names[i] if ki > kj else names[j] if kj > ki else "they land equally often"
            dis = [names[i], names[j], "they land equally often", "neither of them lands"]
            add(2, "more_hits_pair", {"a": i, "b": j},
                f"Which lands on {'its shelf' if sc == 'throw' else 'the floor'} more times during the video: "
                f"{names[i]} or {names[j]}?",
                gt, [x for x in dis if x != gt], True, 2.0 / min(fi["min_gap"], fj["min_gap"]))
    if sc == "staggered":
        order = sorted(range(n), key=lambda i: objs[i]["release"])
        for k in range(n - 1):
            a, b = order[k], order[k + 1]
            add(2, "released_after", {"obj": a}, f"Which object is released right after {names[a]}?", names[b],
                others(b, (a,)), True, 1.0 / (objs[b]["release"] - objs[a]["release"]))
    if sc == "throw":
        ws = sorted((f["walls"][0][0], i) for i, f in enumerate(inf) if f["walls"])
        for pos in range(min(2, len(ws) - 1)):
            t0, w = ws[pos]
            opp = [i for t, i in ws[pos + 1:] if t - t0 >= GAP][:3]
            if opp:
                gap = min(inf[j]["walls"][0][0] for j in opp) - t0
                add(2, "first_wall", {"among": sorted([w] + opp)}, "Which of these objects is the first to bounce "
                    "off a side wall?", names[w], [names[j] for j in opp] + decoys, True, 1.0 / gap)

    # ---------------- L3 measurement
    for i, f in enumerate(inf):
        if f["hits"] and sim.clean_time(f["hits"][0]):
            gt = sim.clock_text(f["hits"][0])
            add(3, "first_hit_time", {"obj": i},
                f"What does the clock show when {names[i]} first {'lands on its shelf' if sc == 'throw' else 'hits the floor'}?",
                gt, _time_opts(f["hits"][0], d), True, 2.0 / TIME_STEP)
        if f["rest"] is not None and sim.clean_time(f["rest"]) and sc != "throw":
            add(3, "rest_time", {"obj": i}, f"What does the clock show when {names[i]} lands for the last time and "
                f"stays on the floor?", sim.clock_text(f["rest"]), _time_opts(f["rest"], d), True,
                max(2.0 / TIME_STEP, 2.0 / f["min_gap"]))
        if sc == "staggered" and sim.clean_time(f["o"]["release"]):
            add(3, "release_time", {"obj": i}, f"What does the clock show when {names[i]} is released?",
                sim.clock_text(f["o"]["release"]), _time_opts(f["o"]["release"], d), True, 2.0 / TIME_STEP)
        if len(f["hits"]) >= 2 and f["hist"]["ups"][0] > 0:
            dt = f["hits"][1] - f["hits"][0]
            v = dt * 10
            if abs(v - round(v)) <= 0.3:
                k = int(round(v))
                gt = f"{k // 10}.{k % 10} s"
                dis = [f"{(k + s) // 10}.{(k + s) % 10} s" for s in (5, -5, 10, -10, 15) if k + s >= 1]
                add(3, "gap_first_second", {"obj": i}, f"How much time passes between the first and the second "
                    f"time {names[i]} {'lands on its shelf' if sc == 'throw' else 'hits the floor'}?", gt, dis, True,
                    2.0 / TIME_STEP)
        if f["apex1"] is not None and f["rebounds"] and len(f["hits"]) >= 2:
            frac = f["e"] ** 2 * 100
            bk = min(BUCKETS, key=lambda b: abs(b - frac))
            if abs(frac - bk) <= BUCKET_TOL * 100:
                gt = f"about {bk}%"
                add(3, "rebound_fraction", {"obj": i}, f"After its first bounce, {names[i]} rises to about what "
                    f"share of the height it fell from?", gt, [f"about {b}%" for b in BUCKETS if b != bk], True,
                    2.0 / (f["hits"][1] - f["hits"][0]))
        hits = f["hits"]
        verb = "land on its shelf" if sc == "throw" else "hit the floor"
        for a in range(1, int(d)):
            for b in range(a + 2, min(a + 5, int(d) + 1)):
                if b > d - 0.3:
                    continue
                inside = [t for t in hits if a < t < b]
                if len(inside) >= 1 and _clean_count_window(hits, a, b):
                    mg = min([y - x for x, y in zip(inside, inside[1:])] + [f["min_gap"] * 2])
                    add(3, "hits_between", {"obj": i, "t0": a, "t1": b},
                        f"Between {a}.0 s and {b}.0 s on the clock, how many times does {names[i]} {verb}?",
                        len(inside), L.near_ints(len(inside), 0), True, max(2.0 / mg, 1.0 / GAP))
        for T in range(1, int(d)):
            if T > d - 0.5 or any(abs(t - T) < GAP for t in hits):
                continue
            k = sum(1 for t in hits if t < T)
            if 1 <= k < len(hits):
                add(3, "hits_by", {"obj": i, "t": T}, f"By the time the clock shows {T}.0 s, how many times has "
                    f"{names[i]} {'landed on its shelf' if sc == 'throw' else 'hit the floor'}?", k,
                    L.near_ints(k, 0), True, max(2.0 / f["min_gap"], 1.0 / GAP))
        if sc == "throw" and f["walls"] and sim.clean_time(f["walls"][0][0]):
            add(3, "first_wall_time", {"obj": i}, f"What does the clock show when {names[i]} first bounces off a "
                f"side wall?", sim.clock_text(f["walls"][0][0]), _time_opts(f["walls"][0][0], d), True,
                2.0 / TIME_STEP)

    # ---------------- L4 integration
    all_hits = sorted(t for f in inf for t in f["hits"])
    if 2 <= len(all_hits) <= 30:
        mg = min(f["min_gap"] for f in inf if f["hits"])
        add(4, "total_hits", {}, f"Counting every object, how many times in total does an object "
            f"{'land on its shelf' if sc == 'throw' else 'hit the floor'} during the video?", len(all_hits),
            L.near_ints(len(all_hits), 0), True, 2.0 / mg)
    bounc = [i for i, f in enumerate(inf) if f["rebounds"] and len(f["hits"]) >= 2]
    if len(bounc) == n:
        es = sorted(range(n), key=lambda i: -inf[i]["e"])
        for which, w, nxt in (("most", es[0], es[1]), ("least", es[-1], es[-2])):
            if abs(inf[w]["e"] ** 2 - inf[nxt]["e"] ** 2) >= 0.1:
                mg = min(f["min_gap"] for f in inf)
                if sc == "pads":
                    pads = scene["pads"]
                    pname = lambda i: f"the {pads[objs[i]['lane']]['color']} pad"
                    add(4, f"{which}_bouncy_pad", {}, f"Each object bounces on its own floor pad. Which pad is the "
                        f"{which} bouncy (an object on it keeps the {'most' if which == 'most' else 'least'} height "
                        f"from bounce to bounce)?", pname(w), [pname(j) for j in es if j != w] +
                        [f"the {c} pad" for c in free_cols], True, 2.0 / mg)
                else:
                    add(4, f"{which}_bouncy", {}, f"Which object is the {which} bouncy (keeps the "
                        f"{'most' if which == 'most' else 'least'} height from bounce to bounce)?", names[w],
                        others(w), True, 2.0 / mg)
    ks = sorted(((len(f["hits"]), i) for i, f in enumerate(inf)), reverse=True)
    if ks[0][0] > ks[1][0] and ks[0][0] <= 14:
        mg = min(f["min_gap"] for f in inf if f["hits"])
        add(4, "most_hits", {}, f"Which object {'lands on its shelf' if sc == 'throw' else 'hits the floor'} the most "
            f"times during the video?", names[ks[0][1]], others(ks[0][1]), True, 2.0 / mg)
    for i, f in enumerate(inf):
        if not f["hits"]:
            continue
        t0 = f["hits"][0]
        oth = [t for j, g in enumerate(inf) if j != i for t in g["hits"]]
        if all(abs(t - t0) >= GAP for t in oth):
            k = sum(1 for t in oth if t < t0)
            if k >= 1 or rng.random() < 0.3:
                close = min([abs(t - t0) for t in oth] + [d])
                mg = min(g["min_gap"] for g in inf)
                add(4, "hits_before", {"obj": i}, f"Before {names[i]} first "
                    f"{'lands on its shelf' if sc == 'throw' else 'hits the floor'}, how many times in total do the "
                    f"other objects {'land on their shelves' if sc == 'throw' else 'hit the floor'}?", k,
                    L.near_ints(k, 0), True, max(1.0 / close, 2.0 / mg))
    if sc != "throw":
        changes = sorted([f["rest"] for f in inf if f["rest"] is not None] +
                         ([f["o"]["release"] for f in inf] if sc == "staggered" else []))
        for T in range(2, int(d)):
            if T > d - 0.5 or any(abs(c - T) < GAP for c in changes):
                continue
            moving = sum(1 for f in inf if f["o"]["release"] < T and (f["rest"] is None or f["rest"] > T))
            if 0 < moving:
                add(4, "moving_at", {"t": T}, f"When the clock shows {T}.0 s, how many objects are still moving "
                    f"(released and not yet lying still)?", moving, L.near_ints(moving, 0, n), True,
                    max(2.0, 1.0 / GAP))
    if sc == "throw":
        tot = sum(len(f["walls"]) for f in inf)
        if 1 <= tot <= 20:
            add(4, "total_wall_hits", {}, "Counting every object, how many times in total does an object bounce "
                "off a side wall?", tot, L.near_ints(tot, 0), True, 2.0)
    restall = [f["rest"] for f in inf]
    if sc != "throw" and all(r is not None for r in restall):
        rs = sorted((r, i) for i, r in enumerate(restall))
        if rs[-1][0] - rs[-2][0] >= GAP:
            w = rs[-1][1]
            add(4, "last_rest", {}, "Which object is the last to come to rest?", names[w], others(w), True,
                max(1.0 / (rs[-1][0] - rs[-2][0]), 2.0 / inf[w]["min_gap"]))

    # ---------------- L5 counterfactual (re-simulated)
    tot_hits = len(all_hits)
    land = "land on its shelf" if sc == "throw" else "hit the floor"
    for i, f in enumerate(inf):
        o = f["o"]
        allh = f["hist"]["impacts"]
        for N in (2, 3, 4):
            end = d + N
            if all(abs(t - end) >= GAP for t in allh) and end < d + sim.EXTRA_S:
                k = sum(1 for t in allh if d <= t < end)
                if k >= 1 or N == 2:
                    add(5, "cf_more_time", {"obj": i, "extra": N}, f"If the video kept going for {N} more seconds, "
                        f"how many more times would {names[i]} {land}?", k, L.near_ints(k, 0), True,
                        2.0 / f["min_gap"])
        for mult, word in ((2.0, "twice"), (0.5, "half")):
            h = sim.history(scene, o, h0=o["h0"] * mult)
            if sim.borderline(h["apexes"]) or any(abs(t - d) < GAP for t in h["impacts"]):
                continue
            k = sum(1 for t in h["impacts"] if t < d)
            if k != len(f["hits"]) or mult == 2.0:
                add(5, "cf_height", {"obj": i, "mult": mult}, f"If {names[i]} had started {word} as high above "
                    f"{surf} (everything else the same), how many times would it {land} during the video?", k,
                    [len(f["hits"])] + L.near_ints(k, 0), True, 2.0 / f["min_gap"])
        if sc == "pads":
            for p_i, p in enumerate(scene["pads"]):
                if p_i == o["lane"]:
                    continue
                h = sim.history(scene, o, e=p["e"])
                if sim.borderline(h["apexes"]) or any(abs(t - d) < GAP for t in h["impacts"]):
                    continue
                k = sum(1 for t in h["impacts"] if t < d)
                add(5, "cf_pad", {"obj": i, "pad": p_i}, f"If {names[i]} had fallen onto the {p['color']} pad "
                    f"instead (same height, same moment), how many times would it hit the floor during the video?",
                    k, [len(f["hits"])] + L.near_ints(k, 0), True, 2.0 / f["min_gap"])
        else:
            for j, g in enumerate(inf):
                if j == i or not g["rebounds"] or abs(g["e"] ** 2 - f["e"] ** 2) < 0.1:
                    continue
                h = sim.history(scene, o, e=g["e"])
                if sim.borderline(h["apexes"]) or any(abs(t - d) < GAP for t in h["impacts"]):
                    continue
                k = sum(1 for t in h["impacts"] if t < d)
                add(5, "cf_bounciness", {"obj": i, "like": j}, f"If {names[i]} were exactly as bouncy as "
                    f"{names[j]} (same starting height and moment), how many times would it {land} during the "
                    f"video?", k, [len(f["hits"])] + L.near_ints(k, 0), True,
                    2.0 / min(f["min_gap"], g["min_gap"]))
        if len(f["hits"]) >= 2 and tot_hits <= 30:
            k = tot_hits - (len(f["hits"]) - 1)
            add(5, "cf_stick", {"obj": i}, f"If {names[i]} had stuck to {surf} the first time it landed, how many "
                f"landings would there be in total, counting every object?", k, [tot_hits] + L.near_ints(k, 0),
                True, 2.0 / min(g["min_gap"] for g in inf if g["hits"]))
    if sc == "staggered":
        fall = sorted((math.sqrt(2 * o["h0"] / scene["g"]), i) for i, o in enumerate(objs))
        for pos in range(min(2, len(fall) - 1)):
            t0, w = fall[pos]
            opp = [i for t, i in fall[pos + 1:] if t - t0 >= GAP][:3]
            if opp:
                add(5, "cf_same_release", {"among": sorted([w] + opp)}, "If every object had been released at "
                    "the same moment, which of these would hit the floor first?", names[w],
                    [names[j] for j in opp] + decoys, True, 2.0)
    if sc == "throw":
        for i, f in enumerate(inf):
            o = f["o"]
            for mult, word in ((2.0, "twice as fast"), (0.5, "half as fast")):
                hs = sim.wall_hits(scene, o, vx=o["vx"] * mult)
                if any(abs(t - d) < GAP for t, _ in hs) or any(t < sim.EDGE_S for t, _ in hs):
                    continue
                k = sum(1 for t, _ in hs if t < d)
                add(5, "cf_speed", {"obj": i, "mult": mult}, f"If {names[i]} slid sideways {word} (same start), "
                    f"how many times would it bounce off a side wall during the video?", k,
                    [len(f["walls"])] + L.near_ints(k, 0), True, 2.0)
    return C


def _pick(rng, pool, k):
    """k items, as many different derivation keys as possible."""
    by_key = {}
    for c in pool:
        by_key.setdefault(c[0], []).append(c)
    keys = sorted(by_key)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(by_key[key])
    out = []
    while len(out) < k:
        for key in keys:
            if by_key[key] and len(out) < k:
                out.append(by_key[key].pop())
    return out


def build(scene: dict, rng: random.Random) -> list[dict]:
    C = candidates(scene, rng)
    sizes = [len(C[lv]) for lv in range(1, 6)]
    counts = L.ten_counts(rng)
    if any(c > s for c, s in zip(counts, sizes)):
        feas = [c for c in L._count_combos() if sum(c) == L.TEN_TOTAL and all(a <= s for a, s in zip(c, sizes))]
        if not feas:
            raise ValueError(f"not enough questions: {sizes}")
        counts = list(rng.choice(feas))
    qs = []
    for lv in range(1, 6):
        for key, params, text, gt, dis, motion, mf in _pick(rng, C[lv], counts[lv - 1]):
            opts, a = L.mc(rng, gt, dis)
            qs.append(L.question(lv, key, text, gt, opts, a, motion, {**params, "min_fps": mf}))
    return L.number(qs)
