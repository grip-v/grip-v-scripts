"""Independent answers for cipher_card_trace.

Nothing here reads the generator's offset bookkeeping (sim.rels, the stored
code, turn flags) or any sim helper. The clip is decoded from what the
renderer draws: `render.track(scene, i)` gives, for frame i, the rotation of
both rings in degrees and every card drawn with its centre. From that
per-frame track this module recovers:
  * landings: the first frame a card sits exactly on a pile centre, and
    which pile (left/right by its x);
  * the top card of each pile in every frame;
  * ring turns: runs of frames where a ring's rotation changes; the turn is
    the rotation after the run minus before it, in sectors (sign = way);
    a turn belongs to the card or joker that landed last before it started;
  * decoding: at a card's landing frame the angle of its symbol on the
    inner ring is compared with the angle of every letter on the outer
    ring; the nearest letter (must be within 1 degree) is the card's letter;
  * clock readings from frame numbers.
Counterfactuals are replayed on these measurements (relative ring angle at
each landing with one turn removed / reversed, angles at frame 0, swapped
landing frames).
"""
from __future__ import annotations
import functools
import json

import render

FPS = 24
ABC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
STEP = 360 / 26
NAMES = {"A": "ace", "J": "jack", "Q": "queen", "K": "king"}


def _wrap(a: float) -> float:
    return (a + 180) % 360 - 180


def _clock(i: int) -> str:
    k = (10 * i) // FPS
    return f"{k // 10}.{k % 10} s"


def _card(c) -> str:
    return f"the {NAMES.get(c['rank'], c['rank'])} of {c['suit']}"


def _sym(s) -> str:
    color, rank = s.split(":")
    return f"the {color} {NAMES.get(rank, rank)}"


def _turn_text(t: int, directed: bool):
    if not directed:
        return abs(t)
    if t == 0:
        return "no net turn"
    n = abs(t)
    return f"{n} position{'s' if n != 1 else ''} {'clockwise' if t > 0 else 'counter-clockwise'}"


@functools.lru_cache(maxsize=64)
def observe(key: str) -> dict:
    scene = json.loads(key)
    piles = scene["geometry"]["piles"]
    left_first = sorted(range(len(piles)), key=lambda p: piles[p][0])      # pile ids, left to right
    side = {p: left_first.index(p) for p in range(len(piles))}            # 0 = left, 1 = right
    land, tops, rots = {}, [], []
    for i in range(scene["n_frames"]):
        tr = render.track(scene, i)
        tp = {}
        for k, x, y in tr["cards"]:
            for p, (px, py) in enumerate(piles):
                if abs(x - px) < 1e-6 and abs(y - py) < 1e-6:
                    tp[side[p]] = k
                    land.setdefault(k, (i, side[p]))
        tops.append(tp)
        rots.append((tr["rot"]["inner"], tr["rot"]["outer"]))
    order = sorted(land, key=lambda k: land[k][0])
    # turns
    turns, i, n = [], 1, len(rots)
    while i < n:
        if rots[i] != rots[i - 1]:
            a = i
            while i < n and rots[i] != rots[i - 1]:
                i += 1
            b = i - 1
            di = rots[b][0] - rots[a - 1][0]
            do = rots[b][1] - rots[a - 1][1]
            ring = "inner" if abs(di) > abs(do) else "outer"
            steps = round((di if ring == "inner" else do) / STEP)
            by = max((k for k in order if land[k][0] < a), key=lambda k: land[k][0])
            turns.append({"start": a - 1, "end": b, "ring": ring, "steps": steps, "by": by,
                          "rel_shift": di - do})
        else:
            i += 1
    return {"land": land, "order": order, "tops": tops, "rots": rots, "turns": turns}


def _letter(scene, item, ri, ro):
    """Letter on the outer ring nearest the item's symbol on the inner ring."""
    c = scene["deal"][item]
    j = scene["ring"].index(f"{c['color']}:{c['rank']}")
    best = min(range(26), key=lambda m: abs(_wrap(j * STEP + ri - m * STEP - ro)))
    if abs(_wrap(j * STEP + ri - best * STEP - ro)) > 1.0:
        return None
    return ABC[best]


def answer(key: str, p: dict, scene: dict):
    ob = observe(json.dumps(scene, sort_keys=True))
    deal = scene["deal"]
    land, order, turns, rots = ob["land"], ob["order"], ob["turns"], ob["rots"]
    cards = [k for k in order if deal[k]["kind"] == "card"]
    jok = [k for k in order if deal[k]["kind"] == "joker"]
    directed = scene["scenario"] in ("both_ways", "letter_ring")

    def at_land(k, shift=0.0):
        ri, ro = rots[land[k][0]]
        return _letter(scene, k, ri + shift, ro)

    def letters(skip_shift=lambda k: 0.0):
        return "".join(at_land(k, skip_shift(k)) or "?" for k in cards)

    def turn_of(k):
        """The turn the k-th joker (or k-th turn, in trigger) makes."""
        if scene["scenario"] == "trigger":
            return turns[k]
        return next(t for t in turns if t["by"] == jok[k])

    def removed(t, factor=1.0):
        return lambda k: -factor * t["rel_shift"] if land[k][0] > t["end"] else 0.0

    def nxt(k):
        pos = order.index(k)
        return order[pos + 1] if pos + 1 < len(order) else None

    def prv(k):
        pos = order.index(k)
        return order[pos - 1] if pos > 0 else None

    def name_card(k):
        return _card(deal[k]) if k is not None and deal[k]["kind"] == "card" else None

    def frames_of(tenths):
        return [i for i in range(len(rots)) if (10 * i) // FPS == tenths]

    if key == "n_cards":
        return len(cards)
    if key == "n_jokers":
        return len(jok)
    if key == "n_color":
        return sum(1 for k in cards if deal[k]["color"] == p["color"])
    if key == "n_suit":
        return sum(1 for k in cards if deal[k]["suit"] == p["suit"])
    if key == "first_card":
        return name_card(cards[0])
    if key == "n_turns":
        return len(turns)
    if key == "pile_count":
        return sum(1 for k in order if land[k][1] == p["pile"])
    if key == "earliest":
        return name_card(min(p["options"], key=lambda k: land[k][0]))
    if key == "latest":
        return name_card(max(p["options"], key=lambda k: land[k][0]))
    if key == "next_card":
        return name_card(nxt(p["x"]))
    if key == "prev_card":
        return name_card(prv(p["x"]))
    if key == "same_key":
        rel = lambda k: rots[land[k][0]][0] - rots[land[k][0]][1]  # noqa: E731
        hit = [k for k in p["options"] if abs(_wrap(rel(k) - rel(p["x"]))) < 1.0]
        return name_card(hit[0]) if len(hit) == 1 else None
    if key == "before_turn":
        t = turns[p["k"]]
        return name_card(ob["tops"][t["start"]].get(0))
    if key == "after_joker":
        return name_card(nxt(jok[p["k"]]))
    if key == "before_joker":
        return name_card(prv(jok[p["k"]]))
    if key == "covers":
        x = p["x"]
        later = [k for k in order if land[k][1] == land[x][1] and land[k][0] > land[x][0]]
        return name_card(later[0]) if later else None
    if key == "on_pile":
        hit = [k for k in p["options"] if land[k][1] == p["pile"]]
        return name_card(hit[0]) if len(hit) == 1 else None
    if key == "turn_size":
        return _turn_text(turn_of(p["k"])["steps"], directed)
    if key == "decode_one":
        return at_land(p["x"])
    if key == "land_time":
        return _clock(land[p["x"]][0])
    if key == "top_at":
        seen = {ob["tops"][i].get(p["pile"]) for i in frames_of(p["t"])}
        return name_card(seen.pop()) if len(seen) == 1 else None
    if key == "under_at":
        j = scene["ring"].index(p["sym"])
        got = set()
        for i in frames_of(p["t"]):
            ri, ro = rots[i]
            best = min(range(26), key=lambda m: abs(_wrap(j * STEP + ri - m * STEP - ro)))
            got.add(ABC[best] if abs(_wrap(j * STEP + ri - best * STEP - ro)) <= 1.0 else None)
        return got.pop() if len(got) == 1 else None
    if key == "code":
        return letters()
    if key == "card_for_letter":
        hit = [k for k in cards if at_land(k) == p["letter"]]
        return name_card(hit[0]) if len(hit) == 1 else None
    if key == "net_turn":
        return _turn_text(sum(t["steps"] for t in turns), directed)
    if key == "n_still":
        ri, ro = rots[-1]
        return sum(1 for k in cards if _letter(scene, k, ri, ro) == at_land(k))
    if key == "after_biggest":
        mags = [abs(t["steps"]) for t in turns]
        big = turns[mags.index(max(mags))]
        if mags.count(max(mags)) != 1:
            return None
        if scene["scenario"] == "trigger":
            return name_card(ob["tops"][big["start"]].get(0))
        return name_card(nxt(big["by"]))
    if key == "pile_code":
        return "".join(at_land(k) or "?" for k in cards if land[k][1] == p["pile"])
    if key == "cf_skip":
        return letters(removed(turn_of(p["k"])))
    if key == "cf_skip_letter":
        return at_land(p["x"], removed(turn_of(p["k"]))(p["x"]))
    if key == "cf_reverse":
        return letters(removed(turn_of(p["k"]), 2.0))
    if key == "cf_no_turns":
        ri0, ro0 = rots[0]
        return "".join(_letter(scene, k, ri0, ro0) or "?" for k in cards)
    if key == "cf_needed":
        last = cards[-1]
        target = ABC.index(at_land(last))
        ri0, ro0 = rots[0]
        hit = [s for j, s in enumerate(scene["ring"]) if abs(_wrap(j * STEP + ri0 - target * STEP - ro0)) < 1.0]
        return _sym(hit[0]) if len(hit) == 1 else None
    if key == "cf_swap":
        ri, ro = rots[land[p["b"]][0]]
        return _letter(scene, p["a"], ri, ro)
    return None
