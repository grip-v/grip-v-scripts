"""10-question ladder for rolling_races (common/ladder5.py, mode "ten").

Balls are named by colour ("the red ball"); a ball's track is drawn in its
colour. Times are what the clock shows on the first frame an event is drawn
(a ball's rack slot fills, it touches the pit floor, its centre passes a
gap's near edge, the start gate is gone). Answers come from sim.py's runs;
answers.py re-derives them from the drawn frames and re-simulates what-ifs
with its own integrator.

Templates per level (derivation_key):
  L1 n_balls, n_gapped, n_fell, n_finish, winner, lowest_track, straight_track, two_gap_track
  L2 earlier, fell_first, which_falls, which_lands, leader_at, same_speed, before_x, after_x
  L3 finish_clock, leave_clock, floor_clock, release_clock, race_time, finish_gap
  L4 finish_order, place_of, last_finisher, next_after, n_after_x, n_racing_at_finish, fell_before_first,
     n_done_at_rest, total_landings, lead_changes
  L5 cf_no_gap, cf_straight (place), cf_no_gap_clock, cf_straight_clock, cf_push_clock, cf_push_falls,
     cf_push_winner, cf_no_gaps_winner, cf_push_lands

Margins: anything compared in time is at least M = 0.3 s apart on the clock;
clock readings sit 0.3-0.7 of a frame interval away from the frame instant
and both frames around the event show the same tenth; "ahead" comparisons
are at least 15 px apart; track comparisons at least 0.25 m (>= 20 px).
"""
from __future__ import annotations
import itertools
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ladder5 as L  # noqa: E402
import sim  # noqa: E402
import answers as A  # noqa: E402
from answers import ball, fmt_s, ORD  # noqa: E402

LADDER = "ten"
TEN_SPLITS = [c for c in itertools.product(*(range(lo, hi + 1) for lo, hi in L.TEN_RANGES)) if sum(c) == L.TEN_TOTAL]
FPS = sim.FPS
M = 0.3                         # time margin (s, clock)
PX = 15.0                       # distance margin (px)
CLOCK_FPS = 4                   # clock readings: options 0.5 s apart -> 4 fps sampling resolves them
RARE = {"release_clock"}        # picked only when a level has nothing else left


def fps_for(gaps, floor=1):
    g = min(gaps) if gaps else 99.0
    return int(min(24, max(floor, math.ceil(1.0 / g - 1e-9))))


def mc4(rng, correct, distractors, fmt=str):
    seen, opts = {fmt(correct)}, [correct]
    for d in distractors:
        if len(opts) == 4:
            break
        if fmt(d) not in seen:
            opts.append(d)
            seen.add(fmt(d))
    if len(opts) < 4:
        return None
    rng.shuffle(opts)
    texts = [fmt(o) for o in opts]
    return texts, L.LETTERS[texts.index(fmt(correct))]


def int_opts(rng, gt, lo, hi=99):
    near = [v for v in (gt + 1, gt - 1, gt + 2, gt - 2, gt + 3, gt - 3, gt + 4, gt + 5) if lo <= v <= hi]
    rng.shuffle(near)
    near.sort(key=lambda v: abs(v - gt))
    return mc4(rng, str(gt), [str(v) for v in near])


def clock_opts(rng, tenths, extra=(), lo=0):
    """Options at least 0.5 s apart; `extra` are plausible wrong readings (other events), tried first."""
    cands = [round(e) for e in extra]
    rng.shuffle(cands)
    near = [tenths + d for d in (-5, 5, -10, 10, -15, 15, 20, -20)]
    rng.shuffle(near)
    keep = []
    for c in cands + near:
        if c >= lo and all(abs(c - u) >= 5 for u in keep + [tenths]):
            keep.append(c)
    return mc4(rng, tenths, keep, fmt_s)


def ord_opts(rng, gt_i, n):
    """Place options among 1st..max(4, n)."""
    near = [i for i in range(max(4, n)) if i != gt_i]
    rng.shuffle(near)
    near.sort(key=lambda i: abs(i - gt_i))
    return mc4(rng, ORD[gt_i], [ORD[i] for i in near])


def colour_opts(rng, gt, preferred, others, pads=()):
    """A colour answer: plausible colours first, then the rest, then non-colour pads (only when the race has
    fewer than four balls to choose from)."""
    pre = list(preferred)
    rest = [c for c in others if c not in pre]
    rng.shuffle(rest)
    return mc4(rng, ball(gt), [ball(c) for c in pre + rest] + list(pads))


def event_ok(te):
    """An event read off the clock: 0.3-0.7 of the way through a frame interval, and both frames around it
    show the same tenth."""
    x = te * FPS
    fa = math.ceil(x - 1e-9)
    return 0.3 <= x - math.floor(x) <= 0.7 and A.clock_tenths(fa - 1) == A.clock_tenths(fa)


def say(rng, *variants):
    """One of several phrasings of the same question."""
    return rng.choice(variants)


def Q(key, params, text, gt, opts, motion, fps):
    if opts is None:
        return None
    assert opts[0][L.LETTERS.index(opts[1])] == gt, (key, gt, opts)
    return (key, dict(params, min_fps=fps), text, gt, opts[0], opts[1], motion)


# ------------------------------------------------------------------ context (generator side)
class Ctx:
    def __init__(self, scene):
        self.s = scene
        self.bs = scene["balls"]
        self.by = {b["colour"]: b for b in self.bs}
        self.cols = [b["colour"] for b in self.bs]
        self.n = len(self.bs)
        self.run = {b["colour"]: sim.ball_run(scene, b) for b in self.bs}
        ct = lambda tau: sim.clock_t(scene, tau)
        self.fin = {c: ct(r["finish"]) for c, r in self.run.items() if r["finish"] is not None}
        self.order = sorted(self.fin, key=self.fin.get)
        self.fell = sorted(c for c, r in self.run.items() if r["fell"] is not None)
        self.leave = {c: [ct(t) for t in r["leave"]] for c, r in self.run.items()}
        self.fall_leave = {c: self.leave[c][self.run[c]["fell"]] for c in self.fell}
        self.floor = {c: ct(self.run[c]["floor_t"]) for c in self.fell}
        self.gapped = [c for c in self.cols if self.by[c].get("gaps")]
        self.landers = [c for c in self.gapped if self.run[c]["land"]]      # landed across at least one gap
        self.t0 = scene["t0"]
        self.rf = sim.release_frame(scene)
        self.N = scene["frames"]
        self.has_gaps = bool(self.gapped)

    def clear_of(self, t, others, m=M):
        return all(abs(t - u) >= m for u in others)

    def fin_times_except(self, c):
        return [t for k, t in self.fin.items() if k != c]


# ------------------------------------------------------------------ L1
def level1(c: Ctx, rng):
    out = [Q("n_balls", {}, say(rng, "How many balls take part in the race?", "How many balls are racing?",
                                   "How many different balls roll down the tracks?"), str(c.n), int_opts(rng, c.n, 1, 12), False, 1)]
    if c.has_gaps:
        k = len(c.gapped)
        out.append(Q("n_gapped", {}, say(rng, "How many of the tracks have at least one gap in them?",
                                                     "How many tracks are broken by a gap?"), str(k),
                     int_opts(rng, k, 0, c.n), False, 1))
        k = len(c.fell)
        out.append(Q("n_fell", {}, say(rng, "How many balls fall through a gap and end up on the pit floor?",
                                                    "How many balls drop out of the race through a gap?",
                                                    "How many balls miss a jump and fall to the floor?"), str(k),
                     int_opts(rng, k, 0, c.n), False, 1))
    k = len(c.fin)
    out.append(Q("n_finish", {}, say(rng, "How many balls reach the finish hole?",
                                             "How many balls make it all the way to the finish?",
                                             "By the end of the clip, how many balls are in the finish rack?"), str(k), int_opts(rng, k, 0, c.n), False, 1))
    o = c.order
    if len(o) == 1 or c.fin[o[1]] - c.fin[o[0]] >= M:
        gap = [c.fin[o[1]] - c.fin[o[0]]] if len(o) > 1 else []
        out.append(Q("winner", {}, say(rng, "Which ball reaches the finish first?", "Which ball wins the race?",
                                                "Which ball drops into the finish hole first?"), ball(o[0]),
                     colour_opts(rng, o[0], o[1:3], c.cols, ["two balls tie for first", "no ball reaches the finish"]),
                     True, fps_for(gap)))
    low = {b["colour"]: max(y for x, y in sim.polyline(sim.shape(c.s, b))
                            if not any(g["x0"] < x < g["x1"] for g in b.get("gaps", ()))) for b in c.bs}
    srt = sorted(low, key=low.get, reverse=True)
    if c.n >= 4 and low[srt[0]] - low[srt[1]] >= 0.25:
        out.append(Q("lowest_track", {}, say(rng, "Whose track dips lowest on the screen?",
                                                          "Which ball's track reaches the lowest point?"), ball(srt[0]),
                     colour_opts(rng, srt[0], srt[1:3], c.cols), False, 1))
    st = [b["colour"] for b in c.bs if b["family"] == "straight"]
    if c.n >= 4 and len(st) == 1:
        near = sorted((b for b in c.bs if b["colour"] != st[0]),
                      key=lambda b: max(abs(y - x * c.s["D"] / c.s["W"]) for x, y in sim.polyline(sim.shape(c.s, b))[::50]))
        if max(abs(y - x * c.s["D"] / c.s["W"]) for x, y in sim.polyline(sim.shape(c.s, near[0]))[::50]) >= 0.25:
            out.append(Q("straight_track", {}, say(rng, "Which ball rides a perfectly straight track?",
                                                              "Which ball's track is a straight line?"), ball(st[0]),
                         colour_opts(rng, st[0], [b["colour"] for b in near[:2]], c.cols), False, 1))
    two = [b["colour"] for b in c.bs if len(b.get("gaps", ())) == 2]
    if c.n >= 4 and len(two) == 1:
        one = [x for x in c.gapped if x != two[0]]
        out.append(Q("two_gap_track", {}, "Whose track has two gaps in it?", ball(two[0]),
                     colour_opts(rng, two[0], one, c.cols), False, 1))
    return out


# ------------------------------------------------------------------ L2
def _leader_moments(c: Ctx):
    """Clock seconds T at which the horizontal leader is the same, at least PX ahead, over [T - 0.5, T + 0.5],
    with the same set of balls in play (none finishing, landing on the floor or leaving the screen) and no ball
    falling through a gap anywhere near the window."""
    out = []
    dirn = -1 if c.s["style"]["mirror"] else 1
    sx = sim.px(c.s, 0, 0)[0]
    floor_y = sim.floor_px(c.s) - c.s["style"]["R"]
    falling = [(c.fall_leave[x], c.floor[x]) for x in c.fell]
    for T in range(math.ceil(c.t0 + 0.8), int(sim.duration(c.s) - 0.8) + 1):
        if any(a < T + 0.5 + M and b > T - 0.5 - M for a, b in falling):
            continue                                   # a ball dropping out of the race is not "in the lead"
        lead, cands, ok = None, None, True
        for f in range(math.floor((T - 0.5) * FPS), math.ceil((T + 0.5) * FPS) + 1):
            live = []
            for b in c.bs:
                p = sim.ball_centre(c.s, b, f)
                if p is not None and p[1] < floor_y - 1.0:
                    live.append(((p[0] - sx) * dirn, b["colour"]))
            live.sort(reverse=True)
            names = sorted(x[1] for x in live)
            if len(live) < 2 or live[0][0] - live[1][0] < PX or (lead and live[0][1] != lead) or \
                    (cands and names != cands):
                ok = False
                break
            lead, cands = live[0][1], names
        if ok:
            out.append((T, lead, cands))
    return out


def level2(c: Ctx, rng):
    out = []
    o = c.order
    pads_fin = ["they arrive at the same moment", "neither of them reaches the finish"]
    pairs = [(a, b) for a, b in itertools.combinations(o, 2) if c.fin[b] - c.fin[a] >= M]
    rng.shuffle(pairs)
    for a, b in pairs[:3]:
        pr = [a, b]
        rng.shuffle(pr)
        out.append(Q("earlier", {"pair": pr}, say(rng, f"Which reaches the finish first: {ball(pr[0])} or {ball(pr[1])}?",
                                                     f"Which gets to the finish hole sooner, {ball(pr[0])} or "
                                                     f"{ball(pr[1])}?"),
                     ball(a), mc4(rng, ball(a), [ball(b)] + pads_fin), True, fps_for([c.fin[b] - c.fin[a]])))
    fp = [(a, b) for a, b in itertools.combinations(c.fell, 2) if abs(c.fall_leave[a] - c.fall_leave[b]) >= M]
    rng.shuffle(fp)
    for a, b in fp[:2]:
        pr = [a, b]
        rng.shuffle(pr)
        first = min(pr, key=c.fall_leave.get)
        out.append(Q("fell_first", {"pair": pr},
                     f"Both {ball(pr[0])} and {ball(pr[1])} fall through a gap. Which of them leaves its track there "
                     "first?", ball(first), mc4(rng, ball(first), [ball(x) for x in pr] +
                                                ["they leave at the same moment", "neither of them leaves its track"]),
                     True, fps_for([abs(c.fall_leave[a] - c.fall_leave[b])])))
    non = [x for x in c.cols if x not in c.fell]
    if c.fell and len(non) >= 3:
        x = rng.choice(c.fell)
        dis = [y for y in non if y in c.gapped] + [y for y in non if y not in c.gapped]
        dis = dis[:3]
        opts = [x] + dis
        rng.shuffle(opts)
        out.append(Q("which_falls", {"options": opts}, "Which of these balls falls through a gap in its track?",
                     ball(x), mc4(rng, ball(x), [ball(y) for y in dis]), False, 1))
    if c.landers and len([y for y in c.cols if y not in c.landers]) >= 3:
        x = rng.choice(c.landers)
        dis = [y for y in c.fell if y not in c.landers]           # fell at their first gap: never landed
        rng.shuffle(dis)
        dis = (dis + [y for y in c.cols if y not in c.gapped])[:3]
        if len(dis) == 3:
            opts = [x] + dis
            rng.shuffle(opts)
            out.append(Q("which_lands", {"options": opts},
                         "Which of these balls jumps across a gap and lands on the far side?", ball(x),
                         mc4(rng, ball(x), [ball(y) for y in dis]), True, 1))
    lm = _leader_moments(c)
    rng.shuffle(lm)
    for T, lead, cands in lm[:2]:
        rest = [y for y in cands if y != lead]
        out.append(Q("leader_at", {"tenths": 10 * T},
                     f"When the clock shows {T}.0 s, which ball is farthest along toward the finish (horizontally)?",
                     ball(lead), colour_opts(rng, lead, rest, [], ["two balls are level", "no ball is still racing"]),
                     True, 1))
    if not c.has_gaps:
        gt = "they are all moving at the same speed"
        opts = mc4(rng, gt, [ball(o[0]), ball(o[-1]), ball(o[len(o) // 2]), "the one whose track dips lowest"])
        out.append(Q("same_speed", {}, "The tracks are frictionless and every ball gets the same push. Which ball is "
                     "moving fastest at the moment it reaches the finish?", gt, opts, False, 1))
    for x in o:
        others = [y for y in o if y != x]
        early = [y for y in others if c.fin[y] <= c.fin[x] - M]
        late = [y for y in others if c.fin[y] >= c.fin[x] + M]
        bad = [y for y in others if abs(c.fin[y] - c.fin[x]) < M]
        if bad:
            continue
        if early and len(late) + len(c.fell) >= 3:
            g = rng.choice(early)
            dis = late + c.fell
            rng.shuffle(dis)
            dis.sort(key=lambda y: c.fin.get(y, 99))
            dis = dis[:3]
            opts = [g] + dis
            rng.shuffle(opts)
            out.append(Q("before_x", {"x": x, "options": opts},
                         f"Which of these balls reaches the finish before {ball(x)} does?", ball(g),
                         mc4(rng, ball(g), [ball(y) for y in dis]), True,
                         fps_for([abs(c.fin[y] - c.fin[x]) for y in [g] + [d for d in dis if d in c.fin]])))
        if late and len(early) >= 3:
            g = rng.choice(late)
            dis = sorted(early, key=lambda y: -c.fin[y])[:3]
            opts = [g] + dis
            rng.shuffle(opts)
            out.append(Q("after_x", {"x": x, "options": opts},
                         f"Which of these balls reaches the finish after {ball(x)} does?", ball(g),
                         mc4(rng, ball(g), [ball(y) for y in dis]), True,
                         fps_for([abs(c.fin[y] - c.fin[x]) for y in [g] + dis])))
    return out


# ------------------------------------------------------------------ L3
def level3(c: Ctx, rng):
    out = []
    rel = A.clock_tenths(c.rf)
    fin_t = {x: A.event_clock(t) for x, t in c.fin.items()}
    for x in c.order:
        if not event_ok(c.fin[x]):
            continue
        tn = fin_t[x]
        out.append(Q("finish_clock", {"x": x}, say(rng, f"What does the clock show when {ball(x)} reaches the finish?",
                                               f"At what clock time does {ball(x)} drop into the finish hole?"),
                     fmt_s(tn), clock_opts(rng, tn, [fin_t[y] for y in c.order if y != x]), True, CLOCK_FPS))
        out.append(Q("race_time", {"x": x},
                     say(rng, f"By the clock, how long after the start gate opens does {ball(x)} reach the finish?",
                         f"How many seconds (by the clock) does {ball(x)} take from the release to the finish?"),
                     fmt_s(tn - rel), clock_opts(rng, tn - rel, [tn, fin_t[c.order[0]] - rel], lo=1), True, CLOCK_FPS))
    for x in c.gapped:
        b = c.by[x]
        for gi, te in enumerate(c.leave[x]):
            if not event_ok(te):
                continue
            # the centre moves >= 2.5 px per frame horizontally there (the crossing frame is unambiguous)
            f = te * FPS
            p0 = sim.ball_centre(c.s, b, math.floor(f) - 1)
            p1 = sim.ball_centre(c.s, b, math.ceil(f) + 1)
            if p0 is None or p1 is None or abs(p1[0] - p0[0]) < 2.5 * (math.ceil(f) - math.floor(f) + 2):
                continue
            which = "" if len(b["gaps"]) == 1 else (" first" if gi == 0 else " second")
            tn = A.event_clock(te)
            extra = [A.event_clock(u) for u in c.leave[x] if u != te] + \
                [A.event_clock(c.floor[x])] if x in c.floor else [A.event_clock(u) for u in c.leave[x] if u != te]
            out.append(Q("leave_clock", {"x": x, "gap": gi},
                         f"What does the clock show when {ball(x)} reaches the edge of the{which} gap in its track and "
                         "leaves the track?", fmt_s(tn), clock_opts(rng, tn, extra), True, CLOCK_FPS))
    for x in c.fell:
        if event_ok(c.floor[x]):
            tn = A.event_clock(c.floor[x])
            out.append(Q("floor_clock", {"x": x}, f"What does the clock show when {ball(x)} lands on the pit floor?",
                         fmt_s(tn), clock_opts(rng, tn, [A.event_clock(c.fall_leave[x])]), True, CLOCK_FPS))
    if True:                                   # offered always; _pick takes it last (RARE)
        out.append(Q("release_clock", {}, "What does the clock show when the start gate opens and the balls set off?",
                     fmt_s(rel), clock_opts(rng, rel), True, CLOCK_FPS))
    ok = [x for x in c.order if event_ok(c.fin[x])]
    pairs = [(a, b) for a, b in itertools.combinations(ok, 2) if c.fin[b] - c.fin[a] >= 0.5]
    rng.shuffle(pairs)
    for a, b in pairs[:2]:
        d = fin_t[b] - fin_t[a]
        out.append(Q("finish_gap", {"pair": [a, b]},
                     f"By the clock, how much later than {ball(a)} does {ball(b)} reach the finish?", fmt_s(d),
                     clock_opts(rng, d, [], lo=1), True, CLOCK_FPS))
    return out


# ------------------------------------------------------------------ L4
def _lead_change_q(c: Ctx):
    """Leader changes between the release and the first finish, counted on frames with a clear lead
    (>= PX); every stretch without a clear lead is short (<= 0.5 s) and changes are >= 0.5 s apart."""
    if not c.order:
        return None
    frames = [{"balls": {}, "rack": [], "gate": f < c.rf} for f in range(c.N)]
    f_end = sim.first_frame(c.fin[c.order[0]])
    for f in range(c.rf, f_end):
        for b in c.bs:
            p = sim.ball_centre(c.s, b, f)
            if p is not None:
                frames[f]["balls"][b["colour"]] = (p[0], p[1])
    o = A.Obs(c.s, frames)
    clear, run, longest = [], 0, 0
    for f in range(c.rf, f_end):
        lead, m = o.leader(f)
        if m >= PX:
            clear.append(f)
            run = 0
        else:
            run += 1
            if f - c.rf > 12:                          # the start, where all balls overlap, is exempt
                longest = max(longest, run)
            if f >= f_end - 1:
                return None
    if not clear or clear[0] - c.rf > 36 or longest > 12:
        return None
    k, ch = A.lead_changes(o, f_end, PX)
    if any(b - a < 12 for a, b in zip(ch, ch[1:])):
        return None
    return k, ch


def level4(c: Ctx, rng):
    out = []
    o = c.order
    gaps = [b - a for a, b in zip([c.fin[x] for x in o], [c.fin[x] for x in o][1:])]
    if 2 <= len(o) <= 5 and all(g >= M for g in gaps):
        gt = ", ".join(o)
        wrong = []
        for i in range(len(o) - 1):
            alt = list(o)
            alt[i], alt[i + 1] = alt[i + 1], alt[i]
            wrong.append(", ".join(alt))
        rng.shuffle(wrong)
        wrong.append(", ".join(o[::-1]))
        for x in c.fell:
            wrong.append(", ".join(o + [x]))
            wrong.append(", ".join([x] + o[1:]))
        if len(o) >= 3:
            wrong.append(", ".join([o[1], o[2], o[0]] + o[3:]))
        out.append(Q("finish_order", {}, say(rng, "In what order do the balls that make it reach the finish?",
                                                         "What is the finishing order (first to last) of the balls "
                                                         "that reach the finish?"), gt,
                     mc4(rng, gt, wrong), True, fps_for(gaps)))
    for i, x in enumerate(o):
        if not c.clear_of(c.fin[x], c.fin_times_except(x)):
            continue
        nb = [abs(c.fin[x] - c.fin[y]) for y in o if y != x]
        if c.n >= 4:
            out.append(Q("place_of", {"x": x}, f"Counting only the balls that reach the finish, in what place does "
                         f"{ball(x)} finish?", ORD[i], ord_opts(rng, i, c.n), True, fps_for(nb)))
        if i + 1 < len(o) and c.n >= 4:
            nxt = o[i + 1]
            if c.clear_of(c.fin[nxt], c.fin_times_except(nxt)):
                pref = [o[i + 2]] if i + 2 < len(o) else []
                pref += [o[i - 1]] if i >= 1 else []
                out.append(Q("next_after", {"x": x}, f"Which ball reaches the finish right after {ball(x)}?",
                             ball(nxt), colour_opts(rng, nxt, pref, [y for y in c.cols if y != x]), True,
                             fps_for([c.fin[nxt] - c.fin[x]] + ([c.fin[o[i + 2]] - c.fin[nxt]] if i + 2 < len(o)
                                                               else []))))
    if len(o) >= 2 and c.fin[o[-1]] - c.fin[o[-2]] >= M and c.n >= 4:
        out.append(Q("last_finisher", {}, "Of the balls that reach the finish, which one gets there last?", ball(o[-1]),
                     colour_opts(rng, o[-1], [o[-2]] + c.fell, c.cols), True, fps_for([c.fin[o[-1]] - c.fin[o[-2]]])))
    if o and c.fell and c.clear_of(c.fin[o[0]], list(c.fall_leave.values())):
        k = sum(1 for x in c.fell if c.fall_leave[x] < c.fin[o[0]])
        out.append(Q("fell_before_first", {}, "When the first ball reaches the finish, how many balls have already "
                     "left their track at the gap they fall through?", str(k), int_opts(rng, k, 0, c.n), True,
                     fps_for([abs(c.fin[o[0]] - t) for t in c.fall_leave.values()])))
    for x in c.fell:
        t = c.floor[x]
        if c.clear_of(t, list(c.fin.values())):
            k = sum(1 for u in c.fin.values() if u < t)
            out.append(Q("n_done_at_rest", {"x": x}, f"When {ball(x)} lands on the pit floor, how many balls are "
                         "already in the finish rack?", str(k), int_opts(rng, k, 0, c.n), True,
                         fps_for([abs(t - u) for u in c.fin.values()])))
    for x in o:
        if not c.clear_of(c.fin[x], c.fin_times_except(x)):
            continue
        k = sum(1 for y in o if c.fin[y] > c.fin[x])
        out.append(Q("n_after_x", {"x": x}, say(rng, f"How many balls reach the finish after {ball(x)}?",
                                                  f"After {ball(x)} reaches the finish, how many more balls get there?"),
                     str(k), int_opts(rng, k, 0, c.n), True, fps_for([abs(c.fin[x] - u) for u in c.fin_times_except(x)])))
        ends = c.fin_times_except(x) + list(c.floor.values())
        if c.clear_of(c.fin[x], ends):
            k = sum(1 for y in c.cols if y != x and c.fin.get(y, c.floor.get(y)) > c.fin[x])
            out.append(Q("n_racing_at_finish", {"x": x}, f"At the moment {ball(x)} reaches the finish, how many other "
                         "balls are still rolling or flying (not yet in the finish rack or on the pit floor)?", str(k),
                         int_opts(rng, k, 0, c.n), True, fps_for([abs(c.fin[x] - u) for u in ends])))
    if c.has_gaps:
        k = sum(len(c.run[x]["land"]) for x in c.cols)
        out.append(Q("total_landings", {}, "Counting every gap, how many times in the race does a ball jump a gap and "
                     "land safely on the far side?", str(k), int_opts(rng, k, 0, 20), True, 1))
    lc = _lead_change_q(c)
    if lc is not None and c.n >= 2:
        k, ch = lc
        out.append(Q("lead_changes", {"clear_px": PX}, "From the start until the first ball reaches the finish, how many "
                     "times does a different ball take the lead (get farthest along toward the finish, "
                     "horizontally)?", str(k), int_opts(rng, k, 0, 9), True,
                     fps_for([(b - a) / FPS for a, b in zip(ch, ch[1:])], 2)))
    return out


# ------------------------------------------------------------------ L5
def level5(c: Ctx, rng):
    out = []
    s, v0 = c.s, c.s["v0"]
    ct = lambda tau: sim.clock_t(s, tau)
    for x in c.cols:
        b = c.by[x]
        others = c.fin_times_except(x)
        variants = []
        if b.get("gaps"):
            variants.append(("cf_no_gap", {"no_gap": True},
                             f"If {ball(x)}'s track had no gap{'s' if len(b['gaps']) > 1 else ''} (everything "
                             "else the same)"))
        if b["family"] != "straight" or b.get("gaps"):
            variants.append(("cf_straight", {"straight": True},
                             f"If {ball(x)}'s track were a straight ramp with no gap from the start to the finish "
                             "(everything else the same)"))
        for key, kw, lead in variants:
            r = sim.ball_run(s, b, **kw)
            if r["finish"] is None:
                continue
            t = ct(r["finish"])
            if c.clear_of(t, others) and c.n >= 4:
                i = sum(1 for u in others if u < t)
                out.append(Q(key, {"x": x}, f"{lead}, in what place would it finish, counting only the balls that "
                             "reach the finish?", ORD[i], ord_opts(rng, i, c.n), True,
                             fps_for([abs(t - u) for u in others])))
            if event_ok(t) and abs(t - c.fin.get(x, -9)) >= 0.5:
                tn = A.event_clock(t)
                ex = [A.event_clock(c.fin[x])] if x in c.fin else []
                out.append(Q(key + "_clock", {"x": x}, f"{lead}, what would the clock show when it reaches the finish?",
                             fmt_s(tn), clock_opts(rng, tn, ex), True, CLOCK_FPS))
    push = {b["colour"]: sim.ball_run(s, b, sim.PUSH_CF * v0) for b in c.bs}
    pt = {x: ct(r["finish"]) for x, r in push.items() if r["finish"] is not None}
    for x in c.order:
        if x in pt and event_ok(pt[x]) and abs(pt[x] - c.fin[x]) >= 0.5:
            tn = A.event_clock(pt[x])
            out.append(Q("cf_push_clock", {"x": x}, f"If every ball started with a push three times as strong, what "
                         f"would the clock show when {ball(x)} reaches the finish?", fmt_s(tn),
                         clock_opts(rng, tn, [A.event_clock(c.fin[x])]), True, CLOCK_FPS))
    if c.has_gaps:
        k = sum(1 for r in push.values() if r["fell"] is not None)
        out.append(Q("cf_push_falls", {}, "If every ball started with a push three times as strong, how many balls "
                     "would fall through a gap?", str(k), mc4(rng, str(k), [str(len(c.fell))] +
                                                            [str(v) for v in (k + 1, k - 1, k + 2, k - 2, k + 3)
                                                             if 0 <= v <= c.n]), True, 1))
        lands = [x for x in c.gapped if push[x]["fell"] is None]
        falls = [x for x in c.gapped if push[x]["fell"] is not None]
        if lands and len(falls) >= 3:
            x = rng.choice(lands)
            dis = falls[:]
            rng.shuffle(dis)
            opts = [x] + dis[:3]
            rng.shuffle(opts)
            out.append(Q("cf_push_lands", {"options": opts}, "If every ball started with a push three times as strong, "
                         "which of these balls would make it across all the gaps in its track?", ball(x),
                         mc4(rng, ball(x), [ball(y) for y in dis[:3]]), True, 1))
        ng = {b["colour"]: ct(sim.ball_run(s, b, no_gap=True)["finish"]) for b in c.bs}
        so = sorted(ng, key=ng.get)
        if ng[so[1]] - ng[so[0]] >= M:
            out.append(Q("cf_no_gaps_winner", {}, "If none of the tracks had any gaps (everything else the same), which "
                         "ball would reach the finish first?", ball(so[0]),
                         colour_opts(rng, so[0], ([o for o in c.order[:1] if o != so[0]] + so[1:3]), c.cols,
                                     ["two balls would tie for first", "no ball would reach the finish"]), True, 1))
    po = sorted(pt, key=pt.get)
    if po and (len(po) == 1 or pt[po[1]] - pt[po[0]] >= M):
        out.append(Q("cf_push_winner", {}, "If every ball started with a push three times as strong, which ball would "
                     "reach the finish first?", ball(po[0]),
                     colour_opts(rng, po[0], [o for o in c.order[:1] if o != po[0]] + po[1:3], c.cols,
                                 ["two balls would tie for first", "no ball would reach the finish"]), True, 1))
    return out


# ------------------------------------------------------------------ assembly
def _pick(rng, cands, n):
    """Distinct keys first; templates that need the motion are preferred (70 %); RARE keys come last."""
    by_key = {}
    for cand in cands:
        by_key.setdefault(cand[0], []).append(cand)
    keys = list(by_key)
    rng.shuffle(keys)
    if rng.random() < 0.7:
        keys.sort(key=lambda k: not any(q[6] for q in by_key[k]))
    keys.sort(key=lambda k: k in RARE)
    chosen, used = [], set()
    while len(chosen) < n:
        avail = [k for k in keys if by_key[k]]
        if not avail:
            break
        k = ([k for k in avail if k not in used] or avail)[0]
        chosen.append(by_key[k].pop(rng.randrange(len(by_key[k]))))
        used.add(k)
        keys.remove(k)
        keys.append(k)
    return chosen


def build(scene: dict, rng: random.Random) -> list[dict]:
    c = Ctx(scene)
    makers = {1: level1, 2: level2, 3: level3, 4: level4, 5: level5}
    made = {}
    for lv in range(1, 6):
        seen, ok = set(), []
        for q in makers[lv](c, random.Random(rng.random())):
            if q is None:
                continue
            sig = (q[0], json.dumps(q[1], sort_keys=True))
            if sig not in seen:
                seen.add(sig)
                ok.append(q)
        made[lv] = ok
    feasible = [cmb for cmb in TEN_SPLITS if all(len(made[lv]) >= cmb[lv - 1] for lv in range(1, 6))]
    if not feasible:
        raise ValueError("not enough question candidates for this scene")
    counts = rng.choice(feasible)
    qs = []
    for lv in range(1, 6):
        for key, params, text, gt, opts, letter, motion in _pick(rng, made[lv], counts[lv - 1]):
            qs.append(L.question(lv, key, text, gt, opts, letter, motion, params, idx=len(qs) + 1))
    return qs
