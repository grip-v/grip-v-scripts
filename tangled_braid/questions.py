"""10-question ladder for tangled_braid (common/ladder5.py, LADDER = "ten").

Strands are named by colour (unique within a clip), places by counting from
the left (from the top when the braid runs sideways), crossings by their
order ("the third crossing"; not in `parallel`, where several happen at
once) or by the two strands and which time they cross, moments by the
on-screen clock. Every question has 4 options and stores derivation_key +
params + min_fps; answers.py re-derives every answer from the renderer's
per-frame track.
"""
from __future__ import annotations
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
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
       "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
       "eighteenth", "nineteenth", "twentieth", "21st", "22nd", "23rd", "24th", "25th", "26th", "27th",
       "28th", "29th", "30th"]
NTH = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]
TIMES = ["once", "twice", "three times", "four times"]
BAR = "at the loom bar (where the braid is being woven)"


def fps_for(gap_s: float) -> int:
    for f in FPS_STEPS:
        if f * gap_s >= 1 - 1e-9:
            return f
    return 24


# ---------------------------------------------------------------- wording

def cn(scene, name) -> str:
    return sim.strand(scene, name)["color_name"]


def side_words(scene):
    """(counting phrase, low-end word, high-end word)"""
    if scene["layout"]["orient"] == "h":
        return "counting from the top", "at the top", "at the bottom"
    return "counting from the left", "leftmost", "rightmost"


def place_text(scene, p: int) -> str:
    return f"{NTH[p]} {side_words(scene)[0].replace('counting ', '')}"


def pair_text(scene, a, b) -> str:
    x, y = sorted([cn(scene, a), cn(scene, b)])
    return f"{x} and {y}"


def sg(z: int) -> str:
    """Signed total: +2, -1, 0."""
    return f"{z:+d}" if z else "0"


# ---------------------------------------------------------------- generator-side derived data

def derived(scene, steps=None, start=None):
    steps = scene["steps"] if steps is None else steps
    start = scene["start"] if start is None else start
    orders, ev = sim.replay(start, steps)
    return orders, ev


def _variant(scene, skip=None, flip=None, mirror=False, swap=None):
    """Re-simulate with crossing number `skip` left out, crossing `flip`
    going the other way over, every crossing flipped (mirror), or two
    strands' starting places swapped. Crossings keep their places."""
    start = list(scene["start"])
    if swap:
        i, j = start.index(swap[0]), start.index(swap[1])
        start[i], start[j] = start[j], start[i]
    steps, e = [], 0
    for st in scene["steps"]:
        mv2 = []
        for mv in st["moves"]:
            if e != skip:
                o = mv["over"]
                if e == flip or mirror:
                    o = "high" if o == "low" else "low"
                mv2.append({"i": mv["i"], "over": o})
            e += 1
        if mv2:
            steps.append({"moves": mv2})
    return sim.replay(start, steps)


def _net(ev, x, y):
    return sum(1 for c in ev if (c["over"], c["under"]) == (x, y)) - \
        sum(1 for c in ev if (c["over"], c["under"]) == (y, x))


def _pair_events(ev, a, b):
    return [k for k, c in enumerate(ev) if {c["low"], c["high"]} == {a, b}]


# ---------------------------------------------------------------- timing gaps (seconds)

def _vis(scene, steps) -> float:
    steps = list(steps) or range(len(scene["steps"]))
    return min(sim.vis_frames(scene, k) for k in steps) / sim.FPS


def _quiet_moments(scene):
    """(tenths, span index, span length in frames) for clock readings that
    lie well inside a still stretch after at least one crossing."""
    out = []
    spans = sim.quiet_spans(scene)
    for j, (lo, hi) in enumerate(spans):
        if j == 0:
            continue
        for t in range(sim.timer_tenths(lo) + 1, sim.timer_tenths(hi) + 1):
            a, b = sim.frames_showing(t)
            if a >= lo + 2 and b <= hi - 2 and b <= scene["n_frames"] - 1 - sim.EDGE:
                out.append((t, j, hi - lo + 1))
    return out


def _edge_span(scene, place: int, at_end: bool) -> float:
    """How long the strand at `place` stays put at the start (or the end)."""
    st = scene["steps"]
    touch = [k for k, s in enumerate(st) if any(mv["i"] in (place, place - 1) for mv in s["moves"])]
    if at_end:
        k = touch[-1] if touch else None
        f0 = st[k]["a"] + st[k]["c"] if k is not None else 0
        return (scene["n_frames"] - f0) / sim.FPS
    k = touch[0] if touch else None
    return ((st[k]["a"] if k is not None else scene["n_frames"]) + 1) / sim.FPS


# ---------------------------------------------------------------- candidates

def candidates(scene, rng):
    sc = scene["scenario"]
    names = list(scene["start"])
    n = len(names)
    orders, ev = derived(scene)
    final = orders[-1]
    steps = scene["steps"]
    par = sc == "parallel"
    count_w, lo_w, hi_w = side_words(scene)
    C = lambda x: cn(scene, x)  # noqa: E731
    out = {lv: [] for lv in range(1, 6)}
    seen = set()
    allvis = _vis(scene, [])
    dur = scene["n_frames"] / sim.FPS
    involved = {x: [k for k, c in enumerate(ev) if x in (c["low"], c["high"])] for x in names}
    step_of = [c["step"] for c in ev]

    def add(lv, key, params, text, gt, dis, gap, motion=True):
        gts = str(gt)
        ds = list(dict.fromkeys(str(d) for d in dis if str(d) != gts and str(d)))
        sig = (key, tuple(sorted((k, str(v)) for k, v in params.items())))
        if len(ds) < 3 or sig in seen:
            return
        seen.add(sig)
        out[lv].append((key, dict(params), text, gts, ds, motion, fps_for(gap)))

    def strands_near(x_list, excl):
        """Other strands, the ones placed near the given ones first."""
        pos = {m: names.index(m) for m in names}
        rest = [m for m in names if m not in excl]
        rng.shuffle(rest)
        rest.sort(key=lambda m: min(abs(pos[m] - pos[x]) for x in x_list) if x_list else 0)
        return [C(m) for m in rest]

    def vis_ev(ks):
        return _vis(scene, {step_of[k] for k in ks})

    # ------------------------------------------------ L1 observation
    add(1, "n_strands", {}, "How many strands are there in the braid?", n, L.near_ints(n, 2), dur, False)
    add(1, "n_crossings", {}, "How many crossings (two neighbouring strands swapping places, one passing over "
        "the other) happen in the whole video?", len(ev), L.near_ints(len(ev), 1), allvis)
    xs = rng.sample(names, min(3, n))
    for x in xs:
        k = len(involved[x])
        add(1, "strand_crossings", {"x": x}, f"How many crossings does the {C(x)} strand take part in?", k,
            L.near_ints(k, 0), vis_ev(involved[x]) if involved[x] else allvis)
    for end, w, p in (("low", lo_w, 0), ("high", hi_w, n - 1)):
        add(1, "start_edge", {"end": end}, f"At the start of the video, which strand is {w} {BAR}?",
            C(scene["start"][p]), strands_near([scene["start"][p]], [scene["start"][p]]), _edge_span(scene, p, False),
            False)
        add(1, "end_edge", {"end": end}, f"At the end of the video, which strand is {w} {BAR}?",
            C(final[p]), [C(scene["start"][p])] + strands_near([final[p]], [final[p]]), _edge_span(scene, p, True),
            False)
    if len(steps[0]["moves"]) == 1:
        c0 = ev[0]
        dis = []
        for c in ev[1:]:
            dis.append(pair_text(scene, c["low"], c["high"]))
        for p in range(n - 1):
            dis.append(pair_text(scene, names[p], names[p + 1]))
        add(1, "first_pair", {}, "Which two strands cross first?", pair_text(scene, c0["low"], c0["high"]),
            dis, vis_ev([0]))
    never = [x for x in names if not involved[x]]
    if len(never) == 1 and n >= 4:
        x = never[0]
        dis = sorted(names, key=lambda m: len(involved[m]))
        add(1, "never_cross", {}, "Which strand never crosses another strand?", C(x),
            [C(m) for m in dis if m != x], allvis)

    # ------------------------------------------------ L2 relational
    if not par:
        for k in rng.sample(range(len(ev)), min(3, len(ev))):
            c = ev[k]
            dis = [C(c["under"])]
            for j in (k - 1, k + 1, k - 2, k + 2):
                if 0 <= j < len(ev):
                    dis += [C(ev[j]["over"]), C(ev[j]["under"])]
            dis += strands_near([c["low"], c["high"]], [c["over"]])
            add(2, "over_at", {"k": k}, f"At the {ORD[k]} crossing of the video, which strand passes over the "
                "other one?", C(c["over"]), dis, _vis(scene, range(0, k + 1)))
    for x in rng.sample(names, n):
        if not involved[x]:
            continue
        k = involved[x][0]
        c = ev[k]
        y = c["high"] if c["low"] == x else c["low"]
        others = [m for m in names if m not in (x, y)]
        z_pref = [ev[j]["low"] if ev[j]["low"] not in (x, y) else ev[j]["high"] for j in involved[x][1:2]]
        z_pref += [m for m in others if abs(names.index(m) - names.index(x)) == 1]
        z = next((m for m in z_pref if m not in (x, y)), rng.choice(others))
        w = "over" if c["over"] == x else "under"
        opts = [f"it passes {a} the {C(b)} strand" for b in (y, z) for a in ("over", "under")]
        gt = f"it passes {w} the {C(y)} strand"
        add(2, "first_cross_of", {"x": x}, f"What happens the first time the {C(x)} strand crosses another strand?",
            gt, [o for o in opts if o != gt], vis_ev([k]))
        if len(out[2]) >= 6:
            break
    crossers = [x for x in names if involved[x]]
    for _ in range(3):
        if len(crossers) < 2:
            break
        pool = rng.sample(names, min(4, n))
        firsts = {x: step_of[involved[x][0]] for x in pool if involved[x]}
        if not firsts:
            continue
        best = min(firsts.values())
        if sum(1 for v in firsts.values() if v == best) != 1 or len(pool) < 4:
            continue
        x = next(m for m, v in firsts.items() if v == best)
        add(2, "first_of", {"opts": sorted(pool)}, "Which of these strands is the first to cross another strand: "
            + ", ".join(C(m) for m in sorted(pool, key=C)) + "?", C(x), [C(m) for m in pool if m != x],
            _vis(scene, set(firsts.values())))
    qm = _quiet_moments(scene)
    for t, j, ln in rng.sample(qm, min(4, len(qm))):
        o = orders[j]
        x = rng.choice(o)
        p = o.index(x)
        side = rng.choice([s for s, ok in (("low", p > 0), ("high", p < n - 1)) if ok])
        nb = o[p - 1] if side == "low" else o[p + 1]
        wd = ("above" if side == "low" else "below") if scene["layout"]["orient"] == "h" else \
            ("to the left of" if side == "low" else "to the right of")
        dis = []
        for q in (p - 1, p + 1, p - 2, p + 2):
            if 0 <= q < n and o[q] != nb:
                dis.append(C(o[q]))
        prev = orders[j - 1]
        q = prev.index(x) + (-1 if side == "low" else 1)
        if 0 <= q < n:
            dis.insert(0, C(prev[q]))
        dis += strands_near([x], [x, nb])
        add(2, "neighbor_at", {"x": x, "t": t, "side": side}, f"When the clock reads {sim.clock_text(t)}, which "
            f"strand is directly {wd} the {C(x)} strand {BAR}?", C(nb), [d for d in dis if d != C(x)], ln / sim.FPS)
    for x in rng.sample(names, n):
        ks = involved[x]
        if len(ks) < 2:
            continue
        c0 = ev[ks[0]]
        y = c0["high"] if c0["low"] == x else c0["low"]
        c1 = ev[ks[1]]
        z = c1["high"] if c1["low"] == x else c1["low"]
        dis = [C(y)] + [C(ev[k]["low"] if ev[k]["low"] != x else ev[k]["high"]) for k in ks[2:]]
        dis += strands_near([x], [x, z])
        add(2, "next_partner", {"x": x, "y": y}, f"After the {C(x)} strand first crosses the {C(y)} strand, "
            f"which strand does the {C(x)} strand cross next?", C(z), dis, vis_ev(ks[:2]))
        add(2, "partner_first", {"x": x}, f"Which strand does the {C(x)} strand cross first?", C(y),
            [C(z)] + dis, vis_ev(ks[:1]))
        break
    if par:
        for k, st in enumerate(steps):
            if len(st["moves"]) != 2:
                continue
            e1, e2 = [j for j, c in enumerate(ev) if c["step"] == k]
            a, b = ev[e1]["low"], ev[e1]["high"]
            if _pair_events(ev, a, b)[0] != e1:
                continue
            gt = pair_text(scene, ev[e2]["low"], ev[e2]["high"])
            dis = [pair_text(scene, c["low"], c["high"]) for c in ev if c["step"] != k]
            dis += [pair_text(scene, orders[k][p], orders[k][p + 1]) for p in range(n - 1)]
            dis = [d for d in dis if d != pair_text(scene, a, b)]
            add(2, "same_time", {"a": a, "b": b}, f"Which two strands cross at the same moment as the "
                f"{C(a)} and {C(b)} strands cross for the first time?", gt, dis, vis_ev([e1, e2]))

    # ------------------------------------------------ L3 measurement
    pairs = sorted({tuple(sorted((c["low"], c["high"]))) for c in ev})
    rng.shuffle(pairs)
    for a, b in pairs[:3]:
        ks = _pair_events(ev, a, b)
        add(3, "pair_count", {"a": a, "b": b}, f"How many times do the {C(a)} and {C(b)} strands cross each other?",
            len(ks), L.near_ints(len(ks), 0), vis_ev(ks))
    for a, b in pairs[:4]:
        k = _pair_events(ev, a, b)[0]
        m = steps[step_of[k]]["m"]
        tt = sim.timer_tenths(m)
        others = sorted({steps[j]["m"] for j in range(len(steps)) if j != step_of[k]}, key=lambda f: abs(f - m))
        cand, frames = [], []
        for f in others:
            t2 = sim.timer_tenths(f)
            if all(abs(t2 - x) >= 4 for x in cand + [tt]):
                cand.append(t2)
                frames.append(f)
        for dt in (6, -6, 10, -10, 14, -14):
            t2 = tt + dt
            if 3 <= t2 <= int(dur * 10) - 3 and all(abs(t2 - x) >= 4 for x in cand + [tt]):
                cand.append(t2)
                frames.append(sim.frames_showing(t2)[0])
        cand, frames = cand[:3], frames[:3]
        gap = min(abs(f - m) for f in frames) / 2 / sim.FPS if frames else 1.0
        occ = "the first time" if len(_pair_events(ev, a, b)) > 1 else "when"
        add(3, "cross_time", {"a": a, "b": b}, f"What does the clock read at the moment the {C(a)} and {C(b)} "
            f"strands meet, {occ} they cross{' each other' if occ == 'when' else ''}?",
            sim.clock_text(tt), [sim.clock_text(x) for x in cand], min(gap, vis_ev([k])))
    if len(qm) >= 2:
        for _ in range(4):
            (t0, j0, l0), (t1, j1, l1) = sorted(rng.sample(qm, 2))
            if j0 == j1:
                continue
            f0, f1 = sim.frames_showing(t0)[1], sim.frames_showing(t1)[0]
            ks = [k for k, c in enumerate(ev) if f0 < steps[c["step"]]["m"] < f1]
            gap = min(l0, l1) / sim.FPS
            add(3, "between", {"t0": t0, "t1": t1}, f"How many crossings happen between the moment the clock reads "
                f"{sim.clock_text(t0)} and the moment it reads {sim.clock_text(t1)}?", len(ks),
                L.near_ints(len(ks), 0), min(gap, vis_ev(ks) if ks else gap))
    for x in rng.sample(names, min(3, n)):
        dv = abs(final.index(x) - names.index(x))
        add(3, "displacement", {"x": x}, f"By how many places has the {C(x)} strand moved {BAR} between the start "
            "and the end of the video (0 = it ends where it started)?", dv, L.near_ints(dv, 0, n - 1),
            min(_edge_span(scene, names.index(x), False), _edge_span(scene, final.index(x), True)))
    for t, j, ln in rng.sample(qm, min(3, len(qm))):
        o = orders[j]
        x = rng.choice([m for m in o if o.index(m) != names.index(m)] or o)
        p = o.index(x)
        dis = [place_text(scene, q) for q in (names.index(x), final.index(x), orders[j - 1].index(x), p - 1, p + 1,
                                              p - 2, p + 2) if 0 <= q < n]
        add(3, "place_at", {"x": x, "t": t}, f"When the clock reads {sim.clock_text(t)}, in which place is the "
            f"{C(x)} strand {BAR}, {count_w}?", place_text(scene, p), dis, ln / sim.FPS)
        f_t = sim.frames_showing(t)[0]
        ks = [k for k in involved[x] if steps[step_of[k]]["m"] < f_t]
        add(3, "count_by", {"x": x, "t": t}, f"By the time the clock reads {sim.clock_text(t)}, how many crossings "
            f"has the {C(x)} strand taken part in?", len(ks), L.near_ints(len(ks), 0),
            min(ln / sim.FPS, vis_ev(ks) if ks else 1.0))

    # ------------------------------------------------ L4 integration
    for a, b in pairs[:3]:
        if rng.random() < 0.5:
            a, b = b, a
        v = _net(ev, a, b)
        add(4, "net_over", {"a": a, "b": b}, f"Go through every crossing between the {C(a)} and {C(b)} strands: "
            f"count +1 each time the {C(a)} strand passes over, -1 each time it passes under. What is the total?",
            sg(v), [sg(z) for z in (-v, v + 1, v - 1, v + 2, v - 2) if z != v], vis_ev(_pair_events(ev, a, b)))
    for x in [m for m in rng.sample(names, n) if involved[m]][:3]:
        k = sum(1 for c in ev if c["over"] == x)
        und = len(involved[x]) - k
        add(4, "over_count", {"x": x}, f"How many times does the {C(x)} strand pass over another strand?", k,
            [und, len(involved[x])] + L.near_ints(k, 0), vis_ev(involved[x]))
    cnt = {x: len(involved[x]) for x in names}
    top = sorted(names, key=lambda m: -cnt[m])
    if n >= 3 and cnt[top[0]] > cnt[top[1]]:
        add(4, "most_crossings", {}, "Which strand takes part in the most crossings?", C(top[0]),
            [C(m) for m in top[1:]], allvis)
    ov = {x: sum(1 for c in ev if c["over"] == x) for x in names}
    topo = sorted(names, key=lambda m: -ov[m])
    if n >= 3 and ov[topo[0]] > ov[topo[1]]:
        add(4, "over_most", {}, "Which strand passes over other strands the most times?", C(topo[0]),
            [C(m) for m in topo[1:]], allvis)
    for x in rng.sample(names, n):
        ks = involved[x]
        seq = [ev[k]["high"] if ev[k]["low"] == x else ev[k]["low"] for k in ks]
        if not 2 <= len(seq) <= 5 or len(set(seq)) < 2:
            continue
        gt = ", then ".join(C(m) for m in seq)
        dis = [", then ".join(C(m) for m in seq[::-1])]
        for i in range(len(seq) - 1):
            s2 = list(seq)
            s2[i], s2[i + 1] = s2[i + 1], s2[i]
            dis.append(", then ".join(C(m) for m in s2))
        for m in strands_near(seq, [x] + seq)[:2]:
            dis.append(", then ".join(C(z) for z in seq[:-1] + [next(y for y in names if C(y) == m)]))
        add(4, "partner_seq", {"x": x}, f"Which strands does the {C(x)} strand cross, in order?", gt, dis, vis_ev(ks))
        break
    home = sum(1 for x in names if final.index(x) == names.index(x))
    if sc != "pure":
        add(4, "home_count", {}, f"How many strands end {BAR} in the same place they started?", home,
            L.near_ints(home, 0, n), min(_edge_span(scene, 0, False), _edge_span(scene, n - 1, True)))
    for x in [m for m in rng.sample(names, n) if final[names.index(m)] != m][:3]:
        y = final[names.index(x)]
        dis = [C(x), C(names[final.index(x)])] + strands_near([y], [y])
        add(4, "where_started", {"x": x}, f"Which strand ends {BAR} in the place where the {C(x)} strand started?",
            C(y), dis, min(_edge_span(scene, names.index(x), False), _edge_span(scene, names.index(x), True)))

    # ------------------------------------------------ L5 counterfactual (re-simulated)
    def cond_skip(k):
        if not par:
            return f"Suppose the {ORD[k]} crossing had not happened"
        c = ev[k]
        ks = _pair_events(ev, c["low"], c["high"])
        nth = "" if len(ks) == 1 else f" the {ORD[ks.index(k)]} time they meet"
        return f"Suppose the {pair_text(scene, c['low'], c['high'])} strands had not crossed{nth}"
    KEEP = ("and every other crossing had still swapped the strands in the same two places, with the strand "
            "coming from the same side passing over")
    for k in rng.sample(range(len(ev)), min(4, len(ev))):
        o2, ev2 = _variant(scene, skip=k)
        f2 = o2[-1]
        diff = [p for p in range(n) if f2[p] != final[p]] or list(range(n))
        p = rng.choice(diff)
        end_w = place_text(scene, p)
        dis = [C(final[p])] + [C(f2[q]) for q in (p - 1, p + 1) if 0 <= q < n] + strands_near([f2[p]], [f2[p]])
        add(5, "cf_skip_end", {"k": k, "p": p}, f"{cond_skip(k)}, {KEEP}. Which strand would end up {end_w} {BAR}?",
            C(f2[p]), dis, allvis)
        prs = sorted({tuple(sorted((c["low"], c["high"]))) for c in ev2} | set(pairs))
        rng.shuffle(prs)
        prs.sort(key=lambda ab: len(_pair_events(ev2, *ab)) == len(_pair_events(ev, *ab)))
        a, b = prs[0]
        v2 = len(_pair_events(ev2, a, b))
        add(5, "cf_skip_pair", {"k": k, "a": a, "b": b}, f"{cond_skip(k)}, {KEEP}. How many times would the {C(a)} "
            f"and {C(b)} strands cross each other?", v2, [len(_pair_events(ev, a, b))] + L.near_ints(v2, 0), allvis)
    for k in rng.sample(range(len(ev)), min(2, len(ev))):
        c = ev[k]
        a, b = (c["over"], c["under"])
        _, ev3 = _variant(scene, flip=k)
        x, y = (a, b) if rng.random() < 0.5 else (b, a)
        v = _net(ev3, x, y)
        which = f"the {ORD[k]} crossing" if not par else (
            f"the {ORD[_pair_events(ev, a, b).index(k)]} time the {C(a)} and {C(b)} strands cross")
        add(5, "cf_flip_net", {"k": k, "x": x, "y": y}, f"Suppose that at {which} the {C(b)} strand had passed over "
            f"the {C(a)} strand instead (everything else the same). Going through every crossing between the {C(x)} "
            f"and {C(y)} strands, count +1 each time the {C(x)} strand passes over, -1 each time it passes under. "
            "What would the total be?", sg(v), [sg(z) for z in (_net(ev, x, y), -v, v + 1, v - 1, v + 2)], allvis)
    for x in [m for m in rng.sample(names, n) if involved[m]][:2]:
        _, ev4 = _variant(scene, mirror=True)
        v = sum(1 for c in ev4 if c["over"] == x)
        add(5, "cf_mirror", {"x": x}, "Suppose that at every crossing the other strand had passed over (same "
            f"crossings, same places, same order). How many times would the {C(x)} strand pass over another strand?",
            v, [sum(1 for c in ev if c["over"] == x), len(involved[x])] + L.near_ints(v, 0), vis_ev(involved[x]))
    for _ in range(3):
        x, y = rng.sample(names, 2)
        o5, _ = _variant(scene, swap=(x, y))
        f5 = o5[-1]
        diff = [p for p in range(n) if f5[p] != final[p]] or list(range(n))
        p = rng.choice(diff)
        dis = [C(final[p])] + [C(f5[q]) for q in (p - 1, p + 1) if 0 <= q < n] + strands_near([f5[p]], [f5[p]])
        add(5, "cf_swap_start", {"x": x, "y": y, "p": p}, f"Suppose the {C(x)} and {C(y)} strands had swapped "
            f"starting places, and every crossing had still happened in the same places and the same order. Which "
            f"strand would end up {place_text(scene, p)} {BAR}?", C(f5[p]), dis, allvis)
    return out


FEATURED = {"classic": ("over_at", "partner_seq", "cf_skip_end"),
            "parallel": ("same_time", "between", "cf_skip_pair"),
            "sideways": ("neighbor_at", "place_at", "cf_swap_start"),
            "pauses": ("cross_time", "count_by", "between"),
            "pure": ("net_over", "pair_count", "cf_flip_net")}


def _pick(rng, items, k, featured=()):
    """k items spread over as many derivation keys as possible; the
    scenario's own question types usually come first."""
    by = {}
    for it in items:
        by.setdefault(it[0], []).append(it)
    keys = sorted(by)
    rng.shuffle(keys)
    front = [key for key in keys if key in featured and rng.random() < 0.7]
    keys = front + [key for key in keys if key not in front]
    out = []
    while len(out) < k:
        added = False
        for key in keys:
            pool = [it for it in by[key] if it not in out]
            if pool and len(out) < k:
                out.append(rng.choice(pool))
                added = True
        if not added:
            break
    return out


def build(scene: dict, rng: random.Random) -> list:
    C = candidates(scene, rng)
    counts = L.ten_counts(rng)
    qs = []
    for lv in range(1, 6):
        picked = _pick(rng, C[lv], counts[lv - 1], FEATURED[scene["scenario"]])
        if len(picked) != counts[lv - 1]:
            raise ValueError(f"level {lv}: only {len(picked)} askable questions")
        for key, params, text, gt, dis, motion, mf in picked:
            opts, let = L.mc(rng, gt, dis[:3] if len(dis) <= 3 else _choose(rng, dis))
            if len(opts) != 4:
                raise ValueError(f"{key}: fewer than 4 options")
            qs.append(L.question(lv, key, text, gt, opts, let, motion, {**params, "min_fps": mf}))
    return L.number(qs)


def _choose(rng, dis):
    """The two most plausible distractors plus one more at random."""
    return dis[:2] + [rng.choice(dis[2:])]
