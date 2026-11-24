"""10-question ladder for shell_game_tracking (common/ladder5.py, LADDER = "ten").

Positions are named by where they are ("the second position from the left",
"the top-right position"), tokens by colour and shape ("the red star"),
moves by order ("the third swap"), moments by the clock; the cups all look
alike and are never named. Ground truth comes from sim.py (sim.play for who
is where, the scene's move and peek frames); answers.py re-derives every
answer by following the drawn cups. Every question stores derivation_key +
params (+ min_fps).
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
ORD = sim.ORD
MARGIN = 8          # frames between events a question compares (1/3 s)
WIN = 7             # frames of calm either side of a clock reading a question names
# correct-letter weights per option count, so A-D come out even over the dataset
WEIGHTS = {2: (1, 1), 3: (1, 1, 1), 4: (0.161, 0.161, 0.322, 0.356)}


def fps_for(gap_frames) -> int:
    gap = float(gap_frames) / sim.FPS
    for f in FPS_STEPS:
        if f * gap >= 1 - 1e-9:
            return f
    return 24


def clock(f) -> str:
    return sim.clock_text(sim.tenths(int(f)))


def _place(rng, gt, distractors):
    opts = [gt] + list(distractors)
    if len(set(opts)) != len(opts) or not 2 <= len(opts) <= 4:
        return None
    rest = opts[1:]
    rng.shuffle(rest)
    k = rng.choices(range(len(opts)), weights=WEIGHTS[len(opts)])[0]
    opts = rest[:k] + [gt] + rest[k:]
    return opts, L.LETTERS[k]


def _pick_d(rng, gt, pool, k=3):
    pool = [p for p in dict.fromkeys(pool) if p != gt]
    if len(pool) < k:
        return None
    return rng.sample(pool, k)


def _q(level, key, text, gt, opts, motion, params):
    if opts is None or params["min_fps"] > 4:          # must stay answerable from ~4 fps samples
        return None
    o, letter = opts
    return L.question(level, key, text, gt, [str(x) for x in o], letter, motion, params)


def _ints(rng, gt, lo=0, hi=999):
    pool = [v for v in (gt + 1, gt - 1, gt + 2, gt - 2, gt + 3, gt - 3, gt + 4) if lo <= v <= hi]
    if len(pool) < 3:
        return None
    d = pool[:2] + [rng.choice(pool[2:])]
    return _place(rng, gt, d)


def _clock_opts(rng, f):
    k = sim.tenths(int(f))
    pool = [sim.clock_text(k + dk) for dk in (5, -5, 10, -10, 15) if k + dk >= 0]
    d = _pick_d(rng, sim.clock_text(k), pool[:4] if len(pool) >= 4 else pool)
    return d and _place(rng, sim.clock_text(k), d)


def _yn(rng, gt):
    return _place(rng, gt, ["no" if gt == "yes" else "yes"])


class Facts:
    def __init__(self, scene):
        self.s = scene
        self.sc = scene["scenario"]
        self.n = len(scene["slots"])
        self.row = self.sc != "grid"
        self.mw = sim.move_word(scene)
        self.moves = scene["moves"]
        self.nm = len(self.moves)
        self.peeks = scene["peeks"]
        pl = sim.play(scene)
        self.hist = pl["hist"]                           # cup at each slot after k moves
        self.toks = scene["tokens"]
        self.tn = [sim.token_name(t) for t in self.toks]
        self.tcup = [t["slot"] for t in self.toks]       # cup id = start slot
        self.nframes = scene["n_frames"]
        self.names = [sim.slot_name(scene, s) for s in range(self.n)]
        evs = sorted([(m["start"], m["start"] + m["dur"] - 1) for m in self.moves] +
                     [(p["start"], p["start"] + p["up"] + p["hold"] + p["down"] - 1) for p in self.peeks])
        gaps = [b[0] - a[1] - 1 for a, b in zip(evs, evs[1:])]
        self.fps_count = max(fps_for(min(m["dur"] for m in self.moves)), fps_for(min(gaps)) if gaps else 1)
        # every move followed (which cup went where): two samples inside the shortest move
        self.fps_all = max(self.fps_count, fps_for(math.ceil(min(m["dur"] for m in self.moves) / 2)))
        shown = sum(1 for f in range(self.moves[0]["start"]) if sim.token_visible(scene, f))
        self.fps_reveal = fps_for(shown)

    def slot_of(self, cup, k):
        return self.hist[k].index(cup)

    def cups_in(self, k):
        """Cups that change place in move k."""
        return sorted({self.hist[k][a] for a, _ in self.moves[k]["perm"]})

    def fps_track(self, cup, upto=None):
        """Following one cup: two samples inside each of its moves, and every move seen."""
        upto = self.nm if upto is None else upto
        need = [fps_for(math.ceil(self.moves[k]["dur"] / 2)) for k in range(upto) if cup in self.cups_in(k)]
        return max([self.fps_count] + need)

    def tok_slots(self, i):
        return [self.slot_of(self.tcup[i], k) for k in range(self.nm + 1)]

    def slot_opts(self, rng, gt, prefer=()):
        pool = [s for s in prefer if s != gt] + [s for s in range(self.n) if s != gt]
        pool = list(dict.fromkeys(pool))
        k = min(3, len(pool))
        if k < 1:
            return None
        d = pool[:max(k, 3)]
        d = rng.sample(d, k) if len(d) > k else d
        return _place(rng, self.names[gt], [self.names[x] for x in d])

    def tok_name(self, i):
        return self.tn[i]

    def which_tok(self, rng):
        return rng.randrange(len(self.toks))

    def move_name(self, k):
        return f"the {ORD[k]} {self.mw}"


def _ptok(F, i):
    return F.tn[i]


def _who(F, i):
    return "the token" if len(F.toks) == 1 else F.tn[i]


# ------------------------------------------------------------------ L1 observation

def q_n_cups(F, rng):
    return _q(1, "n_cups", "How many cups are on the table?", F.n, _ints(rng, F.n, 2), False, {"min_fps": 1})


def q_n_moves(F, rng):
    gt = F.nm
    what = "moves (a swap, three cups turning, or two swaps at once, each counts as one move)" \
        if F.sc == "rotate" else "swaps"
    return _q(1, "n_moves", f"How many {what} happen in the video?", gt, _ints(rng, gt, 1), True,
              {"min_fps": F.fps_count})


def q_n_tokens(F, rng):
    if F.sc != "multi":
        return None
    gt = len(F.toks)
    return _q(1, "n_tokens", "How many tokens are shown under the cups at the start?", gt, _ints(rng, gt, 1), False,
              {"min_fps": F.fps_reveal})


def q_start_slot(F, rng):
    i = F.which_tok(rng)
    s = F.tcup[i]
    return _q(1, "start_slot", f"At the start, under the cup in which position is {F.tn[i]}?", F.names[s],
              F.slot_opts(rng, s), False, {"tok": _ptok(F, i), "min_fps": F.fps_reveal})


def q_first_swap(F, rng):
    m = F.moves[0]
    if len(m["perm"]) != 2:
        return None
    a, b = sorted(x for x, _ in m["perm"])
    gt = f"{F.names[a]} and {F.names[b]}"
    pool = [f"{F.names[x]} and {F.names[y]}" for x in range(F.n) for y in range(x + 1, F.n)]
    d = _pick_d(rng, gt, pool)
    return _q(1, "first_swap", f"In the first {F.mw}, the cups in which two positions trade places?", gt,
              d and _place(rng, gt, d), True, {"min_fps": F.fps_count})


def q_n_peeks(F, rng):
    if not F.peeks:
        return None
    gt = len(F.peeks)
    return _q(1, "n_peeks", "After the start, how many times is a single cup lifted to show what is under it?", gt,
              _ints(rng, gt, 0), True, {"min_fps": F.fps_count})


def _peek_text(F, p):
    return clock(p["start"] + p["up"])


def q_peek_has(F, rng):
    if not F.peeks:
        return None
    i = rng.randrange(len(F.peeks))
    p = F.peeks[i]
    k = sum(1 for m in F.moves if m["start"] + m["dur"] <= p["start"])
    gt = "yes" if F.slot_of(F.tcup[0], k) == p["slot"] else "no"
    return _q(1, "peek_has", f"When a single cup is lifted at {_peek_text(F, p)}, is {_who(F, 0)} under it?", gt,
              _yn(rng, gt), False, {"p": i, "min_fps": fps_for(p["hold"])})


def q_peek_slot(F, rng):
    if not F.peeks:
        return None
    i = rng.randrange(len(F.peeks))
    p = F.peeks[i]
    return _q(1, "peek_slot", f"In which position is the cup that is lifted at {_peek_text(F, p)}?",
              F.names[p["slot"]], F.slot_opts(rng, p["slot"]), False, {"p": i, "min_fps": fps_for(p["hold"])})


# ------------------------------------------------------------------ L2 relational

def q_in_move(F, rng):
    i = F.which_tok(rng)
    k = rng.randrange(F.nm)
    gt = "yes" if F.tcup[i] in F.cups_in(k) else "no"
    return _q(2, "in_move", f"Does the cup hiding {F.tn[i]} take part in {F.move_name(k)}?", gt, _yn(rng, gt), True,
              {"tok": _ptok(F, i), "k": k, "min_fps": F.fps_track(F.tcup[i], k + 1)})


def q_end_side(F, rng):
    if not F.row:
        return None
    i = F.which_tok(rng)
    ts = F.tok_slots(i)
    gt = "further left" if ts[-1] < ts[0] else "further right" if ts[-1] > ts[0] else "in the same position"
    return _q(2, "end_side", f"At the end, is {F.tn[i]} further left, further right, or in the same position "
              f"compared with where it started?", gt,
              _place(rng, gt, [x for x in ("further left", "further right", "in the same position") if x != gt]),
              True, {"tok": _ptok(F, i), "min_fps": F.fps_track(F.tcup[i])})


def q_first_mover(F, rng):
    if len(F.toks) < 2:
        return None
    a, b = rng.sample(range(len(F.toks)), 2)
    fa = next((m["start"] for k, m in enumerate(F.moves) if F.tcup[a] in F.cups_in(k)), None)
    fb = next((m["start"] for k, m in enumerate(F.moves) if F.tcup[b] in F.cups_in(k)), None)
    if fa is None or fb is None or abs(fa - fb) < MARGIN:
        return None
    gt = F.tn[a] + "'s cup" if fa < fb else F.tn[b] + "'s cup"
    other = F.tn[b] + "'s cup" if fa < fb else F.tn[a] + "'s cup"
    return _q(2, "first_mover", f"Which cup moves first: the one hiding {F.tn[a]} or the one hiding {F.tn[b]}?", gt,
              _place(rng, gt, [other]), True,
              {"a": F.tn[a], "b": F.tn[b], "min_fps": max(F.fps_count, fps_for(abs(fa - fb)))})


def q_left_end(F, rng):
    if len(F.toks) < 2 or not F.row:
        return None
    a, b = rng.sample(range(len(F.toks)), 2)
    sa, sb = F.tok_slots(a)[-1], F.tok_slots(b)[-1]
    gt = F.tn[a] if sa < sb else F.tn[b]
    return _q(2, "left_end", f"At the end, which is further left: {F.tn[a]} or {F.tn[b]}?", gt,
              _place(rng, gt, [F.tn[b] if gt == F.tn[a] else F.tn[a]]), True,
              {"a": F.tn[a], "b": F.tn[b], "min_fps": max(F.fps_track(F.tcup[a]), F.fps_track(F.tcup[b]))})


def q_first_dest(F, rng):
    i = F.which_tok(rng)
    ks = [k for k in range(F.nm) if F.tcup[i] in F.cups_in(k)]
    if not ks:
        return None
    k = ks[0]
    gt = F.slot_of(F.tcup[i], k + 1)
    src = F.slot_of(F.tcup[i], k)
    return _q(2, "first_dest", f"The first time the cup hiding {F.tn[i]} moves, to which position does it go?",
              F.names[gt], F.slot_opts(rng, gt, [src] + [b for _, b in F.moves[k]["perm"]]), True,
              {"tok": _ptok(F, i), "min_fps": F.fps_track(F.tcup[i], k + 1)})


def q_moved_more(F, rng):
    a, b = rng.sample(range(F.n), 2)
    na = sum(1 for k in range(F.nm) if a in F.cups_in(k))
    nb = sum(1 for k in range(F.nm) if b in F.cups_in(k))
    if na == nb:
        return None
    ga, gb = f"the cup that starts in {F.names[a]}", f"the cup that starts in {F.names[b]}"
    gt = ga if na > nb else gb
    return _q(2, "moved_more", f"Which cup changes position more often over the video: {ga} or {gb}?", gt,
              _place(rng, gt, [gb if gt == ga else ga]), True,
              {"a": F.names[a], "b": F.names[b], "min_fps": max(F.fps_track(a), F.fps_track(b))})


def q_leave_first(F, rng):
    a, b = rng.sample(range(F.n), 2)
    fa = next((m["start"] for m in F.moves if any(x == a for x, _ in m["perm"])), None)
    fb = next((m["start"] for m in F.moves if any(x == b for x, _ in m["perm"])), None)
    if fa is None or fb is None or abs(fa - fb) < MARGIN:
        return None
    gt = F.names[a] if fa < fb else F.names[b]
    return _q(2, "leave_first", f"Which position does a cup leave first: {F.names[a]} or {F.names[b]}?", gt,
              _place(rng, gt, [F.names[b] if gt == F.names[a] else F.names[a]]), True,
              {"a": F.names[a], "b": F.names[b], "min_fps": max(F.fps_count, fps_for(abs(fa - fb)))})


# ------------------------------------------------------------------ L3 measurement

def q_tok_moves(F, rng):
    i = F.which_tok(rng)
    gt = sum(1 for k in range(F.nm) if F.tcup[i] in F.cups_in(k))
    return _q(3, "tok_moves", f"How many times does the cup hiding {F.tn[i]} change position?", gt,
              _ints(rng, gt, 0, F.nm), True, {"tok": _ptok(F, i), "min_fps": F.fps_track(F.tcup[i])})


def q_slot_moves(F, rng):
    s = rng.randrange(F.n)
    gt = sum(1 for m in F.moves if any(a == s for a, _ in m["perm"]))
    return _q(3, "slot_moves", f"How many times does a cup leave {F.names[s]}?", gt, _ints(rng, gt, 0, F.nm), True,
              {"slot": F.names[s], "min_fps": F.fps_count})


def q_first_move_time(F, rng):
    i = F.which_tok(rng)
    ks = [k for k in range(F.nm) if F.tcup[i] in F.cups_in(k)]
    if not ks:
        return None
    f = F.moves[ks[0]]["start"]
    return _q(3, "first_move_time", f"At what time on the clock does the cup hiding {F.tn[i]} start to move for the "
              f"first time?", clock(f), _clock_opts(rng, f), True, {"tok": _ptok(F, i), "min_fps": 4})


def q_move_time(F, rng):
    k = rng.randrange(F.nm)
    f = F.moves[k]["start"]
    return _q(3, "move_time", f"At what time on the clock does {F.move_name(k)} start?", clock(f),
              _clock_opts(rng, f), True, {"k": k, "min_fps": 4})


def _calm(F, f):
    return all(not (m["start"] - WIN <= f <= m["start"] + m["dur"] - 1 + WIN) for m in F.moves)


def q_moves_between(F, rng):
    ks = []
    for k in range(sim.tenths(sim.EDGE), sim.tenths(F.nframes - 1 - sim.EDGE)):
        a, b = sim.frames_showing(k)
        if _calm(F, a) and _calm(F, b):
            ks.append(k)
    if len(ks) < 2:
        return None
    k1, k2 = sorted(rng.sample(ks, 2))
    if k2 - k1 < 10:
        return None
    f1, f2 = sim.frames_showing(k1)[1], sim.frames_showing(k2)[0]
    gt = sum(1 for m in F.moves if f1 < m["start"] < f2)
    if gt == 0:
        return None
    return _q(3, "moves_between", f"How many {F.mw}s start between {sim.clock_text(k1)} and {sim.clock_text(k2)} on "
              f"the clock?", gt, _ints(rng, gt, 0), True, {"t1": k1, "t2": k2, "min_fps": F.fps_count})


def q_tok_visits(F, rng):
    i = F.which_tok(rng)
    gt = len(set(F.tok_slots(i)))
    return _q(3, "tok_visits", f"Counting where it starts, how many different positions does {F.tn[i]} occupy over "
              f"the video?", gt, _ints(rng, gt, 1, F.n), True, {"tok": _ptok(F, i), "min_fps": F.fps_track(F.tcup[i])})


# ------------------------------------------------------------------ L4 integration

def q_final_slot(F, rng):
    i = F.which_tok(rng)
    ts = F.tok_slots(i)
    gt = ts[-1]
    return _q(4, "final_slot", f"At the end of the video, under the cup in which position is {F.tn[i]}?", F.names[gt],
              F.slot_opts(rng, gt, [ts[0]] + ts[-3:-1]), True, {"tok": _ptok(F, i), "min_fps": F.fps_track(F.tcup[i])})


def q_slot_at(F, rng):
    i = F.which_tok(rng)
    cup = F.tcup[i]
    mv = [F.moves[k] for k in range(F.nm) if cup in F.cups_in(k)]
    ks = []
    for k in range(sim.tenths(sim.EDGE), sim.tenths(F.nframes - 1 - sim.EDGE)):
        a, b = sim.frames_showing(k)
        if a < F.moves[0]["start"]:
            continue
        if all(not (m["start"] - WIN <= b and a <= m["start"] + m["dur"] - 1 + WIN) for m in mv):
            ks.append(k)
    if not ks:
        return None
    k = rng.choice(ks)
    a, _ = sim.frames_showing(k)
    done = sum(1 for m in F.moves if m["start"] + m["dur"] - 1 < a)
    gt = F.slot_of(cup, done)
    if gt == F.tcup[i] and rng.random() < 0.5:
        return None
    return _q(4, "slot_at", f"When the clock shows {sim.clock_text(k)}, under the cup in which position is "
              f"{F.tn[i]}?", F.names[gt], F.slot_opts(rng, gt, [F.tcup[i]]), True,
              {"tok": _ptok(F, i), "t": k, "min_fps": F.fps_track(cup, done)})


def q_returned(F, rng):
    i = F.which_tok(rng)
    ts = F.tok_slots(i)
    away, back = False, False
    for s in ts[1:-1]:
        if s != ts[0]:
            away = True
        elif away:
            back = True
    gt = "yes" if back else "no"
    return _q(4, "returned", f"Before the last {F.mw}, does the cup hiding {F.tn[i]} ever come back to the position "
              f"it started in after having left it?", gt, _yn(rng, gt), True,
              {"tok": _ptok(F, i), "min_fps": F.fps_track(F.tcup[i])})


def q_cup_end(F, rng):
    empties = [s for s in range(F.n) if s not in F.tcup]
    s = rng.choice(empties if empties and rng.random() < 0.8 else list(range(F.n)))
    gt = F.slot_of(s, F.nm)
    return _q(4, "cup_end", f"Follow the cup that starts in {F.names[s]}. Where is it at the end of the video?",
              F.names[gt], F.slot_opts(rng, gt, [s]), True, {"slot": F.names[s], "min_fps": F.fps_track(s)})


def q_token_at_end(F, rng):
    if len(F.toks) < 2:
        return None
    s = rng.randrange(F.n)
    hit = [i for i in range(len(F.toks)) if F.tok_slots(i)[-1] == s]
    gt = F.tn[hit[0]] if hit else "no token"
    pool = [t for t in F.tn if t != gt] + (["no token"] if gt != "no token" else [])
    d = pool[:3]
    return _q(4, "token_at_end", f"At the end of the video, what is under the cup in {F.names[s]}?", gt,
              _place(rng, gt, d), True,
              {"slot": F.names[s], "min_fps": max(F.fps_track(c) for c in F.tcup)})


def q_peek_origin(F, rng):
    if not F.peeks:
        return None
    i = rng.randrange(len(F.peeks))
    p = F.peeks[i]
    k = sum(1 for m in F.moves if m["start"] + m["dur"] <= p["start"])
    cup = F.hist[k][p["slot"]]
    return _q(4, "peek_origin", f"Where did the cup that is lifted at {_peek_text(F, p)} stand at the very start?",
              F.names[cup], F.slot_opts(rng, cup, [p["slot"]]), True, {"p": i, "min_fps": F.fps_track(cup, k)})


# ------------------------------------------------------------------ L5 counterfactual (re-played)

def _run(perms, s):
    for perm in perms:
        for a, b in perm:
            if a == s:
                s = b
                break
    return s


def _perms(F):
    return [m["perm"] for m in F.moves]


def q_cf_skip(F, rng):
    i = F.which_tok(rng)
    k = rng.randrange(F.nm)
    pr = _perms(F)
    gt = _run(pr[:k] + pr[k + 1:], F.tcup[i])
    return _q(5, "cf_skip", f"Suppose {F.move_name(k)} had never happened and every other {F.mw} had happened just "
              f"the same. Under the cup in which position would {F.tn[i]} be at the end?", F.names[gt],
              F.slot_opts(rng, gt, [F.tok_slots(i)[-1]]), True,
              {"tok": _ptok(F, i), "k": k, "min_fps": F.fps_all})


def q_cf_extra(F, rng):
    i = F.which_tok(rng)
    x, y = rng.sample(range(F.n), 2)
    gt = _run(_perms(F) + [[[x, y], [y, x]]], F.tcup[i])
    return _q(5, "cf_extra", f"Suppose that after the video ends, the cups in {F.names[x]} and {F.names[y]} trade "
              f"places once more. Under the cup in which position would {F.tn[i]} then be?", F.names[gt],
              F.slot_opts(rng, gt, [x, y]), True,
              {"tok": _ptok(F, i), "x": F.names[x], "y": F.names[y], "min_fps": F.fps_track(F.tcup[i])})


def q_cf_start(F, rng):
    if len(F.toks) != 1:
        return None
    s = rng.choice([x for x in range(F.n) if x != F.tcup[0]])
    gt = _run(_perms(F), s)
    return _q(5, "cf_start", f"Suppose {F.tn[0]} had started under the cup in {F.names[s]} instead, and the cups had "
              f"moved exactly the same way. Where would it be at the end?", F.names[gt],
              F.slot_opts(rng, gt, [s, F.tok_slots(0)[-1]]), True,
              {"slot": F.names[s], "min_fps": F.fps_all})


def q_cf_order(F, rng):
    if F.nm < 2:
        return None
    i = F.which_tok(rng)
    k = rng.randrange(F.nm - 1)
    pr = _perms(F)
    pr[k], pr[k + 1] = pr[k + 1], pr[k]
    gt = _run(pr, F.tcup[i])
    return _q(5, "cf_order", f"Suppose {F.move_name(k)} and {F.move_name(k + 1)} had happened in the opposite order, "
              f"with everything else the same. Under the cup in which position would {F.tn[i]} be at the end?",
              F.names[gt], F.slot_opts(rng, gt, [F.tok_slots(i)[-1]]), True,
              {"tok": _ptok(F, i), "k": k, "min_fps": F.fps_all})


TEMPLATES = {
    1: [q_n_cups, q_n_moves, q_n_tokens, q_start_slot, q_first_swap, q_n_peeks, q_peek_has, q_peek_slot],
    2: [q_in_move, q_end_side, q_first_mover, q_left_end, q_first_dest, q_moved_more, q_leave_first],
    3: [q_tok_moves, q_slot_moves, q_first_move_time, q_move_time, q_moves_between, q_tok_visits],
    4: [q_final_slot, q_slot_at, q_returned, q_cup_end, q_token_at_end, q_peek_origin],
    5: [q_cf_skip, q_cf_extra, q_cf_start, q_cf_order],
}


def _sig(q):
    return q["derivation_key"], tuple(sorted((k, str(v)) for k, v in q["params"].items() if k != "min_fps"))


def pools(F, rng) -> dict:
    out = {}
    for lv in range(1, 6):
        got = {}
        tpl = TEMPLATES[lv][:]
        rng.shuffle(tpl)
        for _ in range(8):
            for fn in tpl:
                q = fn(F, rng)
                if q and _sig(q) not in got:
                    got[_sig(q)] = q
        out[lv] = list(got.values())
    return out


def _pick(pool, want, rng, motion_first=True, static_ok=False):
    chosen, keys = [], {}
    rest = pool[:]
    rng.shuffle(rest)
    while len(chosen) < want:
        static_used = sum(1 for q in chosen if not q["requires_motion"])
        rest.sort(key=lambda q: (keys.get(q["derivation_key"], 0),
                                 motion_first and (not q["requires_motion"]) and static_used >= 1,
                                 motion_first and not static_ok and not q["requires_motion"]))
        q = rest.pop(0)
        chosen.append(q)
        keys[q["derivation_key"]] = keys.get(q["derivation_key"], 0) + 1
    return chosen


def build(scene: dict, rng: random.Random) -> list[dict]:
    F = Facts(scene)
    P = pools(F, rng)
    avail = [len(P[lv]) for lv in range(1, 6)]
    combos = [c for c in L._count_combos() if sum(c) == L.TEN_TOTAL and all(a >= k for a, k in zip(avail, c))]
    if not combos:
        raise RuntimeError(f"not enough questions per level {avail} for {scene['scenario']}")
    counts = rng.choice(combos)
    out = []
    for lv, want in zip(range(1, 6), counts):
        out += _pick(P[lv], want, rng, motion_first=lv > 1, static_ok=rng.random() < 0.3)
    return L.number(out)
