"""10-question ladder for rainbow_bagel (common/ladder5.py, LADDER = "ten").

Templates per level (derivation_key):
  L1 observation   n_pieces, n_rings, tray_count
  L2 relational    tray_more_pieces, tray_more_outer, ring_most, first_lift
  L3 measurement   tray_area, gold_area, ruler_radius, land_time, gold_time, count_at_time
  L4 integration   marked_origin, piece_tray, first_land_origin, gold_already
  L5 counterfact.  cf_move_ring, cf_gold_other, cf_more_slices, cf_count_move, cf_gold_delay, cf_screen_slow

Every question stores params["min_fps"]: the sampling rate needed to see its
answer (1 for end-state questions; from the smallest gap between the events it
relies on otherwise; for following a look-alike, from sim.track_fps).
Counterfactuals are re-simulated: the changed scene is re-planned and re-packed
by sim, and the answer is read off the re-simulated pieces."""
from __future__ import annotations
import copy
import json
import random
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
SEP = 8            # frames (> 0.3 s) between events a question compares
REST_MARGIN = 12   # frames (0.5 s) between a clock reading and any arrival it could change

WEIGHTS = {   # per-key pick weights within a level; motion templates weigh more
    1: {"n_pieces": 1, "n_rings": 1, "tray_count": 1},
    2: {"tray_more_pieces": 1, "tray_more_outer": 1, "ring_most": 1, "first_lift": 5},
    3: {"tray_area": 2, "gold_area": 1, "ruler_radius": 1, "land_time": 3, "gold_time": 3, "count_at_time": 3},
    4: {"marked_origin": 3, "piece_tray": 2, "first_land_origin": 2, "gold_already": 1},
    5: {"cf_move_ring": 1, "cf_gold_other": 1, "cf_more_slices": 1, "cf_count_move": 1, "cf_gold_delay": 3,
        "cf_screen_slow": 3},
}


# ---------------------------------------------------------------- formatting
def area_txt(v: float) -> str:
    return f"{v:.1f}"


def tenths_txt(t: int) -> str:
    return f"{t // 10}.{t % 10} s"


def slice_name(scene, s: int) -> str:
    return sim.SLICES[scene["slices"]][s][1]


def origin_txt(scene, p) -> str:
    """Answer text naming where a piece was cut from."""
    return f"ring {p['ring']}, {slice_name(scene, p['slice'])} {sim.UNIT[scene['slices']]}"


def piece_ref(scene, p) -> str:
    """How a question names a piece (by where it sits in the whole bagel)."""
    return f"the {slice_name(scene, p['slice'])} {sim.UNIT[scene['slices']]} of ring {p['ring']}"


def safe_area(v: float) -> bool:
    """Away from a rounding boundary of the 1-decimal answer."""
    return abs((v * 10) % 1 - 0.5) >= 0.1


def fps_gap(g: int) -> int:
    """Rate that puts a sample between two events g frames apart."""
    return next(f for f in sim.FPS_LADDER if sim.FPS / f < g)


def fps_window(w: int) -> int:
    """Rate that samples within w frames of an event."""
    return next(f for f in sim.FPS_LADDER if sim.FPS / f <= w)


def num_distractors(v: float, cands, rng, gap: float = 0.3) -> list[str] | None:
    """Three 1-decimal distractors, positive, at least `gap` apart from each other and the answer."""
    cands = [c for c in cands if c > 0.05]
    rng.shuffle(cands)
    chosen = [v]
    for c in cands:
        if all(abs(c - o) >= gap for o in chosen):
            chosen.append(c)
        if len(chosen) == 4:
            return [area_txt(c) for c in chosen[1:]]
    return None


def time_distractors(t: int, t_max: int, rng) -> list[str]:
    offs = [-20, -10, 10, 20, 30, -30]
    rng.shuffle(offs)
    out = [t + o for o in offs if 0 <= t + o <= t_max][:3]
    return [tenths_txt(x) for x in out]


# ---------------------------------------------------------------- facts
class Facts:
    def __init__(self, scene):
        self.sc = scene
        self.pcs = scene["pieces"]
        self.P = sim.tracks(scene)
        self.hid = sim.hidden_mask(scene, self.P)
        self.trays = scene["trays"]
        self.nf = scene["n_frames"]
        self.gold = scene["marked"]
        self._tf = {}

    def tray_pcs(self, tray, pcs=None):
        return [p for p in (pcs or self.pcs) if p["tray"] == tray]

    def area(self, p, r0=None, n=None):
        return sim.piece_area(self.sc["hole"] if r0 is None else r0, p["ring"], n or self.sc["slices"])

    def tray_area(self, tray, pcs=None, r0=None):
        return sum(self.area(p, r0) for p in self.tray_pcs(tray, pcs))

    def track(self, i, a, b):
        if (i, a, b) not in self._tf:
            self._tf[(i, a, b)] = sim.track_fps(self.sc, i, a, b, self.P, self.hid)
        return self._tf[(i, a, b)]

    def conc_fps(self, i) -> int:
        """Rate needed to tell this piece's screen entry and exit from a look-alike's, when they
        are told apart by time only."""
        p = self.pcs[i]
        _, h, s, _ = sim.phases(p)
        g = None
        for q in self.pcs:
            if q is p or q["ring"] != p["ring"]:
                continue
            _, h2, s2, _ = sim.phases(q)
            if s < h2 or s2 < h:
                continue
            for d, dx in ((abs(h - h2), abs(p["x_in"] - q["x_in"])), (abs(s - s2), abs(p["x_out"] - q["x_out"]))):
                if dx < sim.SEP_X:
                    g = d if g is None else min(g, d)
        return 1 if g is None else fps_gap(g)

    def landed_before(self, f, tray=None, skip=None):
        return sum(1 for j, p in enumerate(self.pcs)
                   if j != skip and p["land"] < f and (tray is None or p["tray"] == tray))

    def follow_fps(self, i) -> int | None:
        p = self.pcs[i]
        t = self.track(i, p["lift"], p["land"])
        if t is None or t > sim.MAX_TRACK_FPS:
            return None
        return max(t, self.conc_fps(i))


def resim(scene, edit) -> dict:
    """Re-plan a changed copy of the scene: landing frames from the flight plan, trays re-packed."""
    sc = copy.deepcopy(scene)
    edit(sc)
    for p in sc["pieces"]:
        p["land"] = sim.phases(p)[3]
    sc["packed"] = sim.pack(sc)
    return sc


# ---------------------------------------------------------------- candidates
def _cand(level, key, params, text, gt, dist, rm, fps):
    return {"level": level, "key": key, "params": params, "text": text, "gt": gt, "dist": dist, "rm": rm,
            "fps": fps}


def _cmp_two(counts: dict, rng) -> tuple[str, list[str]]:
    (ta, a), (tb, b) = counts.items()
    if a == b:
        opts = [f"{ta}, by 2", f"{tb}, by 2", f"{ta}, by 1", f"{tb}, by 1"]
        rng.shuffle(opts)
        return "same number", opts[:3]
    side, other, d = (ta, tb, a - b) if a > b else (tb, ta, b - a)
    dist = [f"{other}, by {d}", f"{side}, by {d + 1}", f"{side}, by {d - 1}" if d > 1 else "same number",
            f"{side}, by {d + 2}"]
    return f"{side}, by {d}", dist


def _cmp_many(counts: dict) -> tuple[str, list[str]]:
    top = max(counts.values())
    leaders = [t for t, c in counts.items() if c == top]
    tie = "a tie for the most"
    gt = leaders[0] if len(leaders) == 1 else tie
    return gt, [o for o in list(counts) + [tie] if o != gt]


def _origin_distractors(scene, p, rng, extra=None) -> list[str]:
    same = [q for q in scene["pieces"] if q["ring"] == p["ring"] and q is not p]
    rng.shuffle(same)
    other = [q for q in scene["pieces"] if q["ring"] != p["ring"] and q["slice"] == p["slice"]]
    rng.shuffle(other)
    out = [origin_txt(scene, q) for q in ([extra] if extra is not None and extra["ring"] == p["ring"] else [])]
    for q in same + other:
        t = origin_txt(scene, q)
        if t not in out and t != origin_txt(scene, p):
            out.append(t)
        if len(out) == 3:
            break
    return out


def pools(scene: dict, rng: random.Random) -> dict:
    F = Facts(scene)
    pcs, n, k, r0, trays = F.pcs, scene["slices"], scene["rings"], scene["hole"], F.trays
    N = len(pcs)
    unit = sim.UNIT[n]
    out = {lv: [] for lv in range(1, 6)}
    add = lambda c: out[c["level"]].append(c)   # noqa: E731
    ruler = "Using the ruler (its ticks are 1 unit apart), "
    a1 = sim.piece_area(r0, 1, n)

    # ---- L1 observation
    add(_cand(1, "n_pieces", {}, "How many pieces is the bagel cut into?", N,
              [N - n, N + n, N - k, N + k, N + 2, N - 2], False, 1))
    add(_cand(1, "n_rings", {}, "How many rings (bands) does the bagel have?", k, L.near_ints(k, 1), False, 1))
    for t in trays:
        c = len(F.tray_pcs(t))
        add(_cand(1, "tray_count", {"tray": t}, f"How many pieces end up in the {t} tray?", c,
                  L.near_ints(c, 0), False, 1))

    # ---- L2 relational
    counts = {t: len(F.tray_pcs(t)) for t in trays}
    outer = {t: sum(1 for p in F.tray_pcs(t) if p["ring"] == k) for t in trays}
    gt, dist = _cmp_two(counts, rng)
    add(_cand(2, "tray_more_pieces", {}, "Which tray ends up with more pieces, and by how many?", gt, dist,
              False, 1))
    gt, dist = _cmp_two(outer, rng)
    add(_cand(2, "tray_more_outer", {},
              "Which tray ends up with more pieces from the outermost ring, and by how many?", gt, dist, False, 1))
    for t in trays:
        by_ring = {f"ring {i}": sum(1 for p in F.tray_pcs(t) if p["ring"] == i) for i in range(1, k + 1)}
        gt, dist = _cmp_many(by_ring)
        dist = [d for d in dist if d != gt]
        rng.shuffle(dist)
        dist = dist[:3]
        add(_cand(2, "ring_most", {"tray": t},
                  f"Which ring (ring 1 is the innermost) has the most pieces in the {t} tray?",
                  gt, dist, False, 1))
    lifts = sorted({p["lift"] for p in pcs})
    if len(lifts) >= 4:
        for order in ("first", "last"):
            for _ in range(2):
                for _ in range(20):   # prefer lift-offs at least SEP frames apart
                    ws = sorted(rng.sample(lifts, 4))
                    if min(b - a for a, b in zip(ws, ws[1:])) >= SEP:
                        break
                chosen = [rng.choice([p for p in pcs if p["lift"] == w]) for w in ws]
                target = chosen[0] if order == "first" else chosen[-1]
                gap = min(abs(target["lift"] - q["lift"]) for q in chosen if q is not target)
                ids = sorted(pcs.index(q) for q in chosen)
                add(_cand(2, "first_lift", {"order": order, "pieces": ids},
                          f"Which of these pieces is the {order} to lift off the bagel?", piece_ref(scene, target),
                          [piece_ref(scene, q) for q in chosen if q is not target], True, fps_gap(gap)))

    # ---- L3 measurement
    t_max = sim.clock_tenths(F.nf - 1)
    for t in trays:
        v = F.tray_area(t)
        if safe_area(v):
            cands = [F.tray_area(o) for o in trays if o != t] + [v + a1, v - a1, v + 2 * a1, v - 2 * a1,
                                                                    F.tray_area(t, r0=r0 + 1), F.tray_area(t, r0=r0 - 1)]
            dist = num_distractors(v, cands, rng)
            if dist:
                add(_cand(3, "tray_area", {"tray": t},
                          ruler + f"what is the total area of the pieces in the {t} tray, in square units?",
                          area_txt(v), dist, False, 1))
    g = pcs[F.gold]
    v = F.area(g)
    if safe_area(v):
        cands = [sim.piece_area(r0, i, n) for i in range(1, k + 2) if i != g["ring"]]
        cands += [F.area(g, r0=r0 + 1), F.area(g, r0=r0 - 1), F.area(g, n=4 if n != 4 else 6), 2 * v, v / 2]
        dist = num_distractors(v, cands, rng)
        if dist:
            add(_cand(3, "gold_area", {}, ruler + "what is the area of the gold piece, in square units?",
                      area_txt(v), dist, False, 1))
    add(_cand(3, "ruler_radius", {"which": "hole"}, ruler + "what is the radius of the bagel's hole, in units?",
              r0, L.near_ints(r0, 1), False, 1))
    add(_cand(3, "ruler_radius", {"which": "outer"}, ruler + "what is the bagel's outer radius, in units?",
              r0 + k, L.near_ints(r0 + k, 1), False, 1))
    for t in trays:
        lands = sorted(p["land"] for p in F.tray_pcs(t))
        for which, f in (("first", lands[0]), ("last", lands[-1])):
            tt = sim.clock_tenths(f)
            add(_cand(3, "land_time", {"tray": t, "which": which},
                      f"At what time on the clock does the {which} piece come to rest in the {t} tray?",
                      tenths_txt(tt), time_distractors(tt, t_max, rng), True, fps_window(12)))
    # frames the gold piece is seen gold before it is first completely behind the screen: a sample
    # must land in that window to see where (and when) it turned gold
    gwin = int(np.flatnonzero(F.hid[scene["gold_frame"]:, F.gold])[0])
    tt = sim.clock_tenths(scene["gold_frame"])
    add(_cand(3, "gold_time", {}, "At what time on the clock does a piece turn gold?", tenths_txt(tt),
              time_distractors(tt, t_max, rng), True, fps_window(min(12, gwin))))
    for t in trays:
        busy = [(sim.phases(p)[2], p["land"]) for p in F.tray_pcs(t)]
        ok_t = []
        for x in range(10, t_max - 9):
            f0, f1 = -(-24 * x // 10), -(-24 * (x + 1) // 10) - 1     # frames showing x tenths
            if all(b + REST_MARGIN < f0 or f1 + REST_MARGIN < a for a, b in busy):
                c = F.landed_before(f0 + 1, t)
                if 1 <= c < len(busy):
                    ok_t.append((x, c))
        if ok_t:
            x, c = rng.choice(ok_t)
            add(_cand(3, "count_at_time", {"tray": t, "clock": tenths_txt(x)},
                      f"How many pieces are resting in the {t} tray when the clock reads {tenths_txt(x)}?",
                      c, L.near_ints(c, 0), True, fps_window(REST_MARGIN)))

    # ---- L4 integration
    gf = max(F.track(F.gold, g["lift"], scene["gold_frame"]), fps_window(gwin))
    add(_cand(4, "marked_origin", {},
              f"Follow the gold piece back to the bagel. Which ring was it cut from (ring 1 is the innermost), "
              f"and which {unit} of the bagel was it?", origin_txt(scene, g), _origin_distractors(scene, g, rng),
              True, gf))
    lands_all = sorted(p["land"] for p in pcs)
    for i, p in enumerate(pcs):
        if i == F.gold:
            continue
        others = [q["land"] for q in pcs if q is not p]
        if min(abs(p["land"] - x) for x in others) < SEP:
            continue
        fol = F.follow_fps(i)
        if fol is None:
            continue
        c = F.landed_before(p["land"], p["tray"])
        dist = [f"{p['tray']} tray, {c + 1} already there"]
        if c > 0:
            dist.append(f"{p['tray']} tray, {c - 1} already there")
        for o in trays:
            if o != p["tray"]:
                co = F.landed_before(p["land"], o)
                dist += [f"{o} tray, {co} already there", f"{o} tray, {co + 1} already there"]
        rng.shuffle(dist)
        add(_cand(4, "piece_tray", {"piece": i},
                  f"Follow {piece_ref(scene, p)} after it lifts off. Which tray does it come to rest in, and how "
                  f"many pieces are already resting in that tray when it arrives?",
                  f"{p['tray']} tray, {c} already there", dist, True, max(fol, fps_gap(SEP))))
    for t in trays:
        mine = sorted(F.tray_pcs(t), key=lambda q: q["land"])
        first = mine[0]
        if pcs.index(first) == F.gold or mine[1]["land"] - first["land"] < SEP:
            continue
        fol = F.follow_fps(pcs.index(first))
        if fol is None:
            continue
        add(_cand(4, "first_land_origin", {"tray": t},
                  f"Which piece is the first to come to rest in the {t} tray? Give the ring it was cut from "
                  f"(ring 1 is the innermost) and its {unit} of the bagel.", origin_txt(scene, first),
                  _origin_distractors(scene, first, rng, mine[1]), True,
                  max(fol, fps_gap(mine[1]["land"] - first["land"]))))
    gl = g["land"]
    if min(abs(gl - x) for j, x in enumerate(p2["land"] for p2 in pcs) if j != F.gold) >= SEP:
        c = F.landed_before(gl, skip=F.gold)
        add(_cand(4, "gold_already", {},
                  "When the gold piece comes to rest, how many other pieces are already resting in the trays?",
                  c, L.near_ints(c, 0), True, fps_gap(SEP)))

    # ---- L5 counterfactual (re-simulated)
    for j in range(1, k + 1):
        def flip(sc, j=j):
            for p in sc["pieces"]:
                if p["ring"] == j:
                    p["tray"] = "right" if p["tray"] == "left" else "left"
        cf = resim(scene, flip)
        if not cf["packed"]:
            continue
        v = abs(F.tray_area("left", cf["pieces"]) - F.tray_area("right", cf["pieces"]))
        if not safe_area(v):
            continue
        now = abs(F.tray_area("left") - F.tray_area("right"))
        aj = sim.piece_area(r0, j, n)
        dist = num_distractors(v, [now, v + 2 * aj, abs(v - 2 * aj), v + 2 * a1, v + a1 * 4, abs(v - 4 * a1),
                                   v + 1.0, v + 2.0], rng)
        if dist:
            add(_cand(5, "cf_move_ring", {"ring": j},
                      f"If every piece of ring {j} had landed in the other tray instead, how much bigger (in "
                      f"square units) would the larger tray's total area be than the smaller tray's?",
                      area_txt(v), dist, False, 1))
        for dest in trays:
            if all(p["tray"] == dest for p in pcs if p["ring"] == j):
                continue

            def move(sc, j=j, dest=dest):
                for p in sc["pieces"]:
                    if p["ring"] == j:
                        p["tray"] = dest
            cf = resim(scene, move)
            if not cf["packed"]:
                continue
            c = len(F.tray_pcs(dest, cf["pieces"]))
            add(_cand(5, "cf_count_move", {"ring": j, "to": dest},
                      f"If every piece of ring {j} had landed in the {dest} tray, how many pieces would the "
                      f"{dest} tray hold?", c, [c - 1, c + 1, len(F.tray_pcs(dest)), c - 2, c + 2], False, 1))
    for dest in trays:
        if dest == g["tray"]:
            continue

        def gold_to(sc, dest=dest):
            sc["pieces"][sc["marked"]]["tray"] = dest
        cf = resim(scene, gold_to)
        if not cf["packed"]:
            continue
        v = F.tray_area(dest, cf["pieces"])
        if not safe_area(v):
            continue
        dist = num_distractors(v, [F.tray_area(dest), v + a1, v - a1, F.tray_area(g["tray"]), v + F.area(g),
                                   v - 2 * F.area(g) if v > 2 * F.area(g) else v + 2 * a1], rng)
        if dist:
            add(_cand(5, "cf_gold_other", {"to": dest},
                      f"If the gold piece had come to rest in the {dest} tray instead, what would the total area of "
                      f"the pieces in the {dest} tray be, in square units?", area_txt(v), dist, False, 1))
    for m in (2, 3, 4, 6, 8):
        if m == n:
            continue
        v = sim.piece_area(r0, g["ring"], m)
        if not safe_area(v):
            continue
        dist = num_distractors(v, [F.area(g), sim.piece_area(r0, g["ring"] + 1, m), sim.piece_area(r0, max(1, g["ring"] - 1), m),
                                   sim.piece_area(r0 + 1, g["ring"], m), v * 2, v / 2], rng)
        if dist:
            add(_cand(5, "cf_more_slices", {"slices": m},
                      f"If every ring had been cut into {m} equal slices instead, what would the area of one piece "
                      f"from the gold piece's ring be, in square units?", area_txt(v), dist, False, 1))
    for delay in (24, 48, -24):
        if g["lift"] + delay < sim.CUT1 + 2:
            continue

        def shift(sc, delay=delay):
            sc["pieces"][sc["marked"]]["lift"] += delay
        cf = resim(scene, shift)
        new = cf["pieces"][F.gold]["land"]
        others = [p["land"] for j, p in enumerate(pcs) if j != F.gold]
        if min(abs(new - x) for x in others) < SEP:
            continue
        c = sum(1 for x in others if x < new)
        word = f"{abs(delay) // 24} s {'later' if delay > 0 else 'earlier'}"
        add(_cand(5, "cf_gold_delay", {"shift_frames": delay},
                  f"If the gold piece had lifted off {word} (with the same flight), how many other pieces would "
                  f"already be resting in the trays when it comes to rest?", c,
                  [F.landed_before(g["land"], skip=F.gold), c + 1, c - 1, c + 2], True, fps_gap(SEP)))
    for extra in (24, 48):
        def slow(sc, extra=extra):
            for p in sc["pieces"]:
                p["hf"] += extra
        cf = resim(scene, slow)
        last = max(p["land"] for p in cf["pieces"])
        tt = sim.clock_tenths(last)
        now = sim.clock_tenths(max(lands_all))
        offs = [tt + o for o in (10, -10, 20, -20, 30) if tt + o >= 0 and tt + o != now]
        rng.shuffle(offs)
        add(_cand(5, "cf_screen_slow", {"extra_frames": extra},
                  f"If every piece had taken {extra // 24} s longer to pass behind the screen, at what time on the "
                  f"clock would the last piece come to rest?", tenths_txt(tt),
                  [tenths_txt(x) for x in [now] + offs[:2]], True, fps_window(12)))
    return out


# ---------------------------------------------------------------- build
def _materialize(c, rng, idx):
    if isinstance(c["gt"], int) and not isinstance(c["gt"], bool):
        dist = [d for d in c["dist"] if isinstance(d, int) and d >= 0]
    else:
        dist = c["dist"]
    opts, a = L.mc(rng, c["gt"], dist, str)
    params = dict(c["params"], min_fps=c["fps"])
    return L.question(c["level"], c["key"], c["text"], c["gt"], opts, a, c["rm"], params, idx)


def build(scene: dict, rng: random.Random) -> list[dict]:
    P = pools(scene, rng)
    seen = set()
    for lv in P:   # drop repeated key+params; keep candidates that make four distinct options
        keep = []
        for c in P[lv]:
            sig = (c["key"], json.dumps(c["params"], sort_keys=True))
            if sig in seen:
                continue
            if isinstance(c["gt"], int) or len({str(d) for d in c["dist"]} - {str(c["gt"])}) >= 3:
                seen.add(sig)
                keep.append(c)
        P[lv] = keep
    combos = [c for c in L._count_combos() if sum(c) == L.TEN_TOTAL and all(len(P[lv + 1]) >= c[lv]
                                                                               for lv in range(5))]
    counts = rng.choice(combos)
    picked = []
    for lv in range(1, 6):
        pool = list(P[lv])
        for _ in range(counts[lv - 1]):
            keys = sorted({c["key"] for c in pool})
            key = rng.choices(keys, weights=[WEIGHTS[lv][kk] for kk in keys])[0]
            c = rng.choice([c for c in pool if c["key"] == key])
            pool.remove(c)
            picked.append(c)
    qs = [_materialize(c, rng, i + 1) for i, c in enumerate(picked)]
    return L.number(qs)
