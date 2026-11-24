"""10-question ladder for cipher_card_trace (common/ladder5.py, LADDER = "ten").

Cards are named by rank and suit ("the 9 of diamonds"; no two dealt cards
share both), jokers by dealing order ("the second joker"), inner-ring
symbols by colour and rank ("the red queen"), piles by side, moments by the
on-screen clock. Every question has 4 options; card options are always
cards that really are dealt (so "the one I saw" is never a giveaway), code
options are the decodes a viewer gets from plausible mistakes (wrong key,
a missed or misjudged turn), never random strings. Every question stores
derivation_key + params + min_fps for answers.py.
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
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
RULE = ("(Each card stands for the letter on the outer ring that its colour and rank on the inner ring sit under "
        "at the moment the card lands.)")
PILE = ["left", "right"]


def fps_for(gap_s: float) -> int:
    for f in FPS_STEPS:
        if f * gap_s >= 1 - 1e-9:
            return f
    return 24


def turn_text(t: int, directed: bool):
    if not directed:
        return abs(t)
    if t == 0:
        return "no net turn"
    n = abs(t)
    return f"{n} position{'s' if n != 1 else ''} {'clockwise' if t > 0 else 'counter-clockwise'}"


def directed(scene) -> bool:
    return scene["scenario"] in ("both_ways", "letter_ring")


def ring_name(scene) -> str:
    return "the outer letter ring" if scene["moving"] == "outer" else "the inner ring"


# ---------------------------------------------------------------- timing gaps (seconds)

def _vis(scene, i) -> float:
    a, b = sim.top_span(scene, i)
    return (b - a + 1) / sim.FPS


def _decode_gap(scene, i) -> float:
    """To decode card i a viewer must see the card (any time it lies on top)
    and the wheel during the ring rest that contains its landing."""
    land = scene["deal"][i]["land"]
    a, b, _ = next(r for r in sim.ring_rests(scene) if r[0] <= land <= r[1])
    return min(_vis(scene, i), (b - a + 1) / sim.FPS)


def _rest_gap(scene, turn_items=None) -> float:
    """Shortest ring rest next to the given turns (all rests if None)."""
    rests = sim.ring_rests(scene)
    T = sim.turners(scene)
    if turn_items is None:
        use = rests
    else:
        use = []
        for j in turn_items:
            k = T.index(j)
            use += [rests[k], rests[k + 1]]
    return min((b - a + 1) for a, b, _ in use) / sim.FPS


# ---------------------------------------------------------------- distractor helpers

def _shift(s: str, k: int) -> str:
    return "".join(sim.ABC[(sim.ABC.index(c) + k) % 26] for c in s)


def code_variants(scene) -> list:
    """Codes produced by plausible mistakes, most plausible first."""
    T = sim.turners(scene)
    turns = [scene["deal"][i]["turn"] for i in T]
    R = sim.rels(scene)
    out = [scene["code"], sim.decode_const(scene, sim.final_rel(scene)), sim.decode_const(scene, R[0])]
    for k in range(len(T)):
        out.append(sim.decode(scene, turns[:k] + [0] + turns[k + 1:]))
        out.append(sim.decode(scene, turns[:k] + [-turns[k]] + turns[k + 1:]))
        for e in (1, -1):
            t2 = turns[k] + e * (1 if turns[k] > 0 else -1)
            if t2:
                out.append(sim.decode(scene, turns[:k] + [t2] + turns[k + 1:]))
    out.append(sim.decode(scene, [0] * len(T)))
    out.append(sim.decode(scene, [t + (1 if t > 0 else -1) for t in turns]))
    return list(dict.fromkeys(out))


def _letters_near(gt: str, extra) -> list:
    out = [x for x in extra if x != gt]
    for k in (1, -1, 2, -2):
        out.append(sim.ABC[(sim.ABC.index(gt) + k) % 26])
    return list(dict.fromkeys(x for x in out if x != gt))


def _clock_of(i):
    return sim.clock_text(sim.timer_tenths(i))


# ---------------------------------------------------------------- candidates

def candidates(scene, rng):
    """Every askable question, by level: (key, params, text, gt, distractors, motion, min_fps)."""
    sc, deal = scene["scenario"], scene["deal"]
    C, J, T = sim.cards(scene), sim.jokers(scene), sim.turners(scene)
    R = sim.rels(scene)
    nm = lambda i: sim.card_name(deal[i])  # noqa: E731
    letter = {i: sim.letter_of(scene, i, R[i]) for i in C}
    turns = [deal[i]["turn"] for i in T]
    dirn = directed(scene)
    RN = ring_name(scene)
    trig = sc == "trigger"
    allvis = min(_vis(scene, i) for i in range(len(deal)))
    out = {lv: [] for lv in range(1, 6)}
    seen = set()

    def add(lv, key, params, text, gt, dis, gap, motion=True):
        gts = str(gt)
        ds = list(dict.fromkeys(str(d) for d in dis if str(d) != gts and str(d)))
        sig = (key, tuple(sorted((k, str(v)) for k, v in params.items())))
        if len(ds) < 3 or sig in seen:
            return
        seen.add(sig)
        out[lv].append((key, dict(params), text, gt, ds, motion, fps_for(gap)))

    def others(excl, prefer=()):
        rest = [i for i in C if i not in excl and i not in prefer]
        rng.shuffle(rest)
        return [nm(i) for i in prefer if i not in excl] + [nm(i) for i in rest]

    def tref(k):
        return f"the {ORD[k]} turn of the inner ring" if trig else f"the {ORD[k]} joker"

    # ------------------------------------------------ L1 observation
    n = len(C)
    add(1, "n_cards", {}, "How many cards are dealt in total?" if trig else
        "How many playing cards are dealt in total, not counting jokers?", n, L.near_ints(n, 2, 9), allvis)
    if J:
        add(1, "n_jokers", {}, "How many jokers are dealt?", len(J), L.near_ints(len(J), 0, 6), allvis)
    for color, txt in (("red", "red cards (hearts or diamonds)"), ("black", "black cards (spades or clubs)")):
        k = sum(1 for i in C if deal[i]["color"] == color)
        add(1, "n_color", {"color": color}, f"How many {txt} are dealt?", k, L.near_ints(k, 0, 6), allvis)
    for suit in ("hearts", "diamonds", "spades", "clubs"):
        k = sum(1 for i in C if deal[i]["suit"] == suit)
        if k:
            add(1, "n_suit", {"suit": suit}, f"How many {suit} are dealt?", k, L.near_ints(k, 0, 6), allvis)
    add(1, "first_card", {}, "Which card is dealt first?", nm(C[0]), others([C[0]], [C[1], C[-1]]),
        _vis(scene, C[0]))
    if trig:
        add(1, "n_turns", {}, "How many times does the inner ring turn during the video?", len(T),
            L.near_ints(len(T), 0, 6), _rest_gap(scene))
    if sc == "two_piles":
        for p in (0, 1):
            k = sum(1 for d in deal if d["pile"] == p)
            add(1, "pile_count", {"pile": p}, f"How many cards land on the {PILE[p]} pile in total, jokers included?",
                k, L.near_ints(k, 1, 9), min(_vis(scene, i) for i in range(len(deal)) if deal[i]["pile"] == p))

    # ------------------------------------------------ L2 relational
    for _ in range(3):
        if len(C) < 4:
            break
        opts = sorted(rng.sample(C, 4))
        add(2, "earliest", {"options": opts}, "Which of these cards is dealt first?", nm(opts[0]),
            [nm(i) for i in opts[1:]], min(_vis(scene, i) for i in opts))
    pool = C[:-1]
    if len(pool) >= 4:
        opts = sorted(rng.sample(pool, 4))
        add(2, "latest", {"options": opts}, "Which of these cards is dealt last?", nm(opts[-1]),
            [nm(i) for i in opts[:-1]], min(_vis(scene, i) for i in opts))
    for x in C:
        if x + 1 < len(deal) and deal[x + 1]["kind"] == "card":
            nx = x + 1
            prev = [x - 1] if x - 1 >= 0 and deal[x - 1]["kind"] == "card" else []
            add(2, "next_card", {"x": x}, f"Which card is dealt right after {nm(x)}?", nm(nx),
                others([x, nx], prev + ([nx + 1] if nx + 1 < len(deal) and deal[nx + 1]["kind"] == "card" else [])),
                min(_vis(scene, x), _vis(scene, nx)))
    for x in C:
        if x - 1 >= 0 and deal[x - 1]["kind"] == "card":
            pv = x - 1
            pref = [x + 1] if x + 1 < len(deal) and deal[x + 1]["kind"] == "card" else []
            add(2, "prev_card", {"x": x}, f"Which card is dealt right before {nm(x)}?", nm(pv),
                others([x, pv], pref + ([pv - 1] if pv - 1 >= 0 and deal[pv - 1]["kind"] == "card" else [])),
                min(_vis(scene, x), _vis(scene, pv)))
    for x in C:
        same = [i for i in C if i != x and R[i] == R[x]]
        diff = [i for i in C if R[i] != R[x]]
        if same and len(diff) >= 3:
            y = rng.choice(same)
            opts = sorted([y] + rng.sample(diff, 3))
            add(2, "same_key", {"x": x, "options": opts}, f"Which of these cards lands while the wheel is in the same "
                f"position as when {nm(x)} lands?", nm(y), [nm(i) for i in opts if i != y],
                min(_decode_gap(scene, i) for i in opts + [x]))
    for k, j in enumerate(T):
        if trig:
            add(2, "before_turn", {"k": k}, f"Which card is on top of the pile when {tref(k)} starts?", nm(j),
                others([j], [j + 1] + ([j - 1] if j > 0 else [])), _vis(scene, j))
        else:
            add(2, "after_joker", {"k": k}, f"Which card is dealt right after {tref(k)}?", nm(j + 1),
                others([j + 1], [j - 1] + ([j + 2] if j + 2 < len(deal) and deal[j + 2]["kind"] == "card" else [])),
                min(_vis(scene, j), _vis(scene, j + 1)))
            add(2, "before_joker", {"k": k}, f"Which card is dealt right before {tref(k)}?", nm(j - 1),
                others([j - 1], [j + 1]), min(_vis(scene, j), _vis(scene, j - 1)))
    if sc == "two_piles":
        for x in C:
            nx = sim.next_same_pile(scene, x)
            if nx is not None and deal[nx]["kind"] == "card":
                pref = [i for i in C if i > x and i != nx][:2]
                add(2, "covers", {"x": x}, f"Which card is the next one to land on {nm(x)}, covering it?",
                    nm(nx), others([x, nx], pref), min(_vis(scene, x), _vis(scene, nx)))
        for p in (0, 1):
            mine = [i for i in C if deal[i]["pile"] == p]
            theirs = [i for i in C if deal[i]["pile"] != p]
            if mine and len(theirs) >= 3:
                x = rng.choice(mine)
                opts = sorted([x] + rng.sample(theirs, 3))
                add(2, "on_pile", {"pile": p, "options": opts}, f"Which of these cards lands on the {PILE[p]} pile?",
                    nm(x), [nm(i) for i in opts if i != x], min(_vis(scene, i) for i in opts))

    # ------------------------------------------------ L3 measurement
    for k, j in enumerate(T):
        t = turns[k]
        if dirn:
            gt = turn_text(t, True)
            dis = [turn_text(-t, True)] + [turn_text(v, True) for v in (t + 1, t - 1, -(t + 1), -(t - 1)) if v]
            text = f"After {tref(k)} lands, which way and by how many positions does {RN} turn?"
        else:
            gt = abs(t)
            dis = L.near_ints(abs(t), 1, 8)
            text = (f"By how many positions does the inner ring turn the {ORD[k]} time it turns?" if trig else
                    f"After {tref(k)} lands, by how many positions does the inner ring turn?")
        add(3, "turn_size", {"k": k}, text, gt, dis, min(_rest_gap(scene, [j]), _vis(scene, j)))
    rest_rels = [r for _, _, r in sim.ring_rests(scene)]
    for x in C:
        gt = letter[x]
        extra = [sim.letter_of(scene, x, r) for r in rest_rels]
        add(3, "decode_one", {"x": x}, f"Which letter does {nm(x)} stand for? {RULE}", gt,
            _letters_near(gt, extra), _decode_gap(scene, x))
    lands = {i: deal[i]["land"] for i in C}
    for x in C:
        gt = _clock_of(lands[x])
        near = sorted((i for i in C if i != x), key=lambda i: abs(lands[i] - lands[x]))
        dis = [_clock_of(lands[i]) for i in near]
        spacing = min(abs(lands[i] - lands[x]) for i in near[:3]) / sim.FPS
        add(3, "land_time", {"x": x}, f"What does the clock show when {nm(x)} lands on its pile?", gt, dis,
            spacing / 2)
    tops = []
    for x in C:
        a, b = sim.top_span(scene, x)
        ts = [t for t in range(sim.timer_tenths(a + 2), sim.timer_tenths(b - 2) + 1)
              if sim.frames_showing(t)[0] >= a + 2 and sim.frames_showing(t)[1] <= b - 2]
        if ts:
            tops.append((x, rng.choice(ts)))
    for x, t in rng.sample(tops, min(3, len(tops))):
        p = deal[x]["pile"]
        where = f"the {PILE[p]} pile" if sc == "two_piles" else "the pile"
        same = [i for i in C if deal[i]["pile"] == p and i != x]
        pref = sorted(same, key=lambda i: abs(i - x))[:3]
        add(3, "top_at", {"t": t, "pile": p}, f"Which card is on top of {where} when the clock shows "
            f"{sim.clock_text(t)}?", nm(x), others([x], pref), _vis(scene, x))
    rests = sim.ring_rests(scene)
    picks = []
    for ri, (a, b, r) in enumerate(rests):
        ts = [t for t in range(sim.timer_tenths(a + 2), sim.timer_tenths(b - 2) + 1)
              if sim.frames_showing(t)[0] >= a + 2 and sim.frames_showing(t)[1] <= b - 2]
        if ts:
            picks.append((ri, rng.choice(ts)))
    for ri, t in rng.sample(picks, min(2, len(picks))):
        a, b, r = rests[ri]
        s = rng.choice(scene["ring"])
        j = scene["ring"].index(s)
        gt = sim.ABC[(j + r) % 26]
        extra = [sim.ABC[(j + r2) % 26] for _, _, r2 in rests]
        add(3, "under_at", {"t": t, "sym": s}, f"When the clock shows {sim.clock_text(t)}, which letter of the outer "
            f"ring is {sim.sym_name(s)} of the inner ring under?", gt, _letters_near(gt, extra), (b - a + 1) / sim.FPS)

    # ------------------------------------------------ L4 integration
    dec_all = min(_decode_gap(scene, i) for i in C)
    variants = code_variants(scene)
    add(4, "code", {}, f"Reading every card in dealing order, what code do the cards spell? {RULE}",
        scene["code"], variants[1:] + [_shift(scene["code"], 1)], min(dec_all, _rest_gap(scene)))
    fr = sim.final_rel(scene)
    for x in C:
        lt = letter[x]
        decoy = [i for i in C if i != x and lt in (sim.letter_of(scene, i, fr), sim.letter_of(scene, i, R[0]))]
        near = sorted((i for i in C if i != x and i not in decoy), key=lambda i: abs(i - x))
        add(4, "card_for_letter", {"letter": lt}, f"Which card stands for the letter {lt}? {RULE}", nm(x),
            [nm(i) for i in decoy + near], dec_all)
    net = sum(turns)
    if dirn:
        gt = turn_text(net, True)
        absum = sum(abs(t) for t in turns)
        dis = [turn_text(v, True) for v in (-net, absum, -absum, net + 1, net - 1, net + 2)]
        add(4, "net_turn", {}, f"By the end of the video, how far is {RN} from where it started (net, which way)?",
            gt, dis, _rest_gap(scene))
    else:
        add(4, "net_turn", {}, "Adding up every turn, by how many positions does the inner ring turn in total?",
            net, [max(turns), net + 1, net - 1, net + 2, net - 2], _rest_gap(scene))
    still = sum(1 for i in C if R[i] == fr)
    add(4, "n_still", {}, "If every card were decoded with the wheel as it stands at the end of the video, how many "
        f"cards would still stand for the same letter as they really do? {RULE}", still,
        L.near_ints(still, 0, len(C)), min(dec_all, _rest_gap(scene)))
    if len(T) >= 2:
        mags = [abs(t) for t in turns]
        top = max(mags)
        if mags.count(top) == 1:
            k = mags.index(top)
            j = T[k]
            if trig:
                add(4, "after_biggest", {}, "Which card is on top of the pile when the inner ring makes its biggest "
                    "turn?", nm(j), others([j], [T[m] for m in range(len(T)) if m != k] + [j + 1]),
                    min(_rest_gap(scene, T), allvis))
            else:
                add(4, "after_biggest", {}, f"Which card is dealt right after the joker that turns {RN} the "
                    "farthest?", nm(j + 1), others([j + 1], [T[m] + 1 for m in range(len(T)) if m != k] + [j - 1]),
                    min(_rest_gap(scene, T), allvis))
    if sc == "two_piles":
        for p in (0, 1):
            mine = [i for i in C if deal[i]["pile"] == p]
            if len(mine) >= 2:
                gt = "".join(letter[i] for i in mine)
                dis = ["".join(sim.letter_of(scene, i, r) for i in mine) for r in (fr, R[0])]
                theirs = [i for i in C if deal[i]["pile"] != p]
                dis.append("".join(letter[i] for i in theirs))
                dis += [_shift(gt, 1), _shift(gt, -1)]
                add(4, "pile_code", {"pile": p}, f"Reading only the cards that land on the {PILE[p]} pile, in the order "
                    f"they land, which letters do they stand for? {RULE}", gt, dis, dec_all)

    # ------------------------------------------------ L5 counterfactual (re-simulated)
    nt = sim.decode(scene, [0] * len(T))
    for k, j in enumerate(T):
        gt = sim.decode(scene, turns[:k] + [0] + turns[k + 1:])
        cond = (f"Suppose {tref(k)} had not happened" if trig else f"Suppose {tref(k)} had not turned {RN}")
        add(5, "cf_skip", {"k": k}, f"{cond} (every other turn unchanged). What would the cards spell? {RULE}", gt,
            [v for v in variants if v != gt] + [_shift(gt, 1)], min(dec_all, _rest_gap(scene)))
        after = [x for x in C if x > j]
        x = rng.choice(after)
        g2 = sim.letter_of(scene, x, (R[x] - sim.rel_shift(scene, turns[k])) % 26)
        add(5, "cf_skip_letter", {"k": k, "x": x}, f"{cond} (every other turn unchanged). Which letter would {nm(x)} "
            f"stand for? {RULE}", g2, _letters_near(g2, [letter[x]] + [sim.letter_of(scene, x, r) for r in rest_rels]),
            min(_decode_gap(scene, x), _rest_gap(scene, [j])))
        if dirn:
            g3 = sim.decode(scene, turns[:k] + [-turns[k]] + turns[k + 1:])
            add(5, "cf_reverse", {"k": k}, f"Suppose {tref(k)} had turned {RN} the same number of positions but the "
                f"other way (every other turn unchanged). What would the cards spell? {RULE}", g3,
                [v for v in variants if v != g3] + [_shift(g3, 1)], min(dec_all, _rest_gap(scene)))
    add(5, "cf_no_turns", {}, f"Suppose {RN} had never turned. What would the same cards spell? {RULE}", nt,
        [v for v in variants if v != nt] + [_shift(nt, 1)], min(dec_all, _rest_gap(scene)))
    last = C[-1]
    li = sim.ABC.index(letter[last])
    need = scene["ring"][(li - R[0]) % 26]
    have = sim.symbol(deal[last])
    dis = [have] + [scene["ring"][(li - R[0] + e) % 26] for e in (1, -1, 2)]
    add(5, "cf_needed", {}, f"Suppose {RN} had never turned. Which colour and rank of the inner ring would the last "
        f"card dealt need to have to stand for the same letter it really stands for? {RULE}", sim.sym_name(need),
        [sim.sym_name(s) for s in dis if s != need], min(_decode_gap(scene, last), _rest_gap(scene)))
    pairs = [(x, y) for x in C for y in C if x < y and R[x] != R[y] and not deal[x]["turn"] and not deal[y]["turn"]]
    for x, y in rng.sample(pairs, min(2, len(pairs))):
        a, b = (x, y) if rng.random() < 0.5 else (y, x)
        g4 = sim.letter_of(scene, a, R[b])
        add(5, "cf_swap", {"a": a, "b": b}, f"Suppose {nm(a)} and {nm(b)} had swapped places in the deal (everything "
            f"else the same). Which letter would {nm(a)} stand for? {RULE}", g4,
            _letters_near(g4, [letter[a], letter[b]] + [sim.letter_of(scene, a, r) for r in rest_rels]),
            min(_decode_gap(scene, a), _decode_gap(scene, b)))
    return out


FEATURED = {"clockwise": ("turn_size", "cf_skip", "after_joker"),
            "both_ways": ("turn_size", "cf_reverse", "net_turn"),
            "letter_ring": ("turn_size", "cf_reverse", "under_at"),
            "trigger": ("n_turns", "before_turn", "cf_skip_letter"),
            "two_piles": ("pile_count", "on_pile", "covers", "pile_code")}


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
            opts, let = L.mc(rng, str(gt), dis)
            if len(opts) != 4:
                raise ValueError(f"{key}: fewer than 4 options")
            qs.append(L.question(lv, key, text, gt, opts, let, motion, {**params, "min_fps": mf}))
    return L.number(qs)
