"""10-question ladder for chess_dance_trace (common/ladder5.py, LADDER = "ten").

Pieces are named by colour + type ("the teal knight"; every colour is used
once per clip); squares by the board coordinates drawn beside the board
(files A-H under it, ranks 1-8 beside it); moments by the on-screen clock
("when the clock shows 3.4 s"). Every question has 4 options and stores
derivation_key + params (piece ids, move numbers, clock tenths, option sets)
+ min_fps for answers.py.

Margins: clock moments sit >= 2 frames inside a rest (or a glide) on every
frame that shows that reading; compared first moves are whole rounds apart;
clock-reading answers are offered in 0.5 s steps; counts and squares are
exact integers / names.
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
NO_ILLEGAL = "no piece does"
NO_BLOCK = "no piece would be blocked"
FIX = {None: "a legal move", "pattern": "illegal: a path its type can never take",
       "blocked": "illegal: it slides through another piece", "occupied": "illegal: it lands on an occupied square"}
ILLEGAL_DEF = ("Does any piece make a move its type is not allowed to make (a path its type can never take, "
               "such as a rook moving diagonally, or a slide through another piece)? If so, which piece?")


def fps_for(gap_s: float) -> int:
    for f in FPS_STEPS:
        if f * gap_s >= 1 - 1e-9:
            return f
    return 24


def _attack_def(scene):
    holes = " and missing squares" if scene["holes"] else ""
    return (f"(A piece attacks another if it could move onto that piece's square in one move; pieces{holes} "
            "in between block a slide, knights jump.)")


# ---------------------------------------------------------------- timing gaps (seconds)

def _round_rests(scene, rounds) -> float:
    """Shortest still interval before / after any of these rounds."""
    R, nf = scene["rounds"], scene["n_frames"]
    out = []
    for r in rounds:
        s, land = sim.round_frames(scene, r)
        out.append(s - (sim.round_frames(scene, r - 1)[1] if r > 1 else 0))
        out.append((sim.round_frames(scene, r + 1)[0] if r < R else nf - 1) - land)
    return min(out) / sim.FPS if out else nf / sim.FPS


def _piece_rests(scene, xs) -> float:
    out = [b - a for x in xs for a, b, _ in sim.rests(scene, x)]
    return min(out) / sim.FPS


def _intro(scene) -> float:
    return sim.round_frames(scene, 1)[0] / sim.FPS


def _hold(scene) -> float:
    return (scene["n_frames"] - 1 - sim.round_frames(scene, scene["rounds"])[1]) / sim.FPS


# ---------------------------------------------------------------- distractor helpers

def _near_squares(scene, s, rng, avoid=()):
    N = scene["N"]
    holes = {sim.sq(h) for h in scene["holes"]}
    out = [(s[0] + a, s[1] + b) for a in (-2, -1, 0, 1, 2) for b in (-2, -1, 0, 1, 2)
           if (a or b) and 0 <= s[0] + a < N and 0 <= s[1] + b < N]
    out = [q for q in out if q not in holes and q not in set(avoid)]
    rng.shuffle(out)
    return [sim.name(q) for q in out]


def _time_opts(tenths: int, nf: int):
    out = []
    top = (10 * (nf - 1)) // sim.FPS
    for k in (1, -1, 2, -2, 3, -3, 4, -4):
        v = tenths + 5 * k
        if 1 <= v <= top:
            out.append(sim.clock_text(v))
    return out


def candidates(scene, rng):
    """Every askable question, grouped by level: (key, params, text, gt, distractors, motion, min_fps)."""
    n, R, sc, nf = scene["n"], scene["rounds"], scene["scenario"], scene["n_frames"]
    pcs = scene["pieces"]
    P = sim.positions(scene)
    types = [p["type"] for p in pcs]
    mr = {i: sim.move_rounds(scene, i) for i in range(n)}
    cnt = {i: len(mr[i]) for i in range(n)}
    capr = {i: sim.captured_round(scene, i) for i in range(n)}
    lab = lambda i: "the " + sim.label(scene, i)  # noqa: E731
    all_rounds = list(range(1, R + 1))
    v = sim.viable(scene)
    C = {lv: [] for lv in range(1, 6)}

    def add(lv, key, params, text, gt, dis, motion, gap):
        if gt is None or len({str(d) for d in dis} - {str(gt)}) < 3:
            return
        C[lv].append((key, dict(params), text, gt, list(dis), motion, fps_for(gap)))

    def others(xs, pool=None):
        pool = list(range(n)) if pool is None else pool
        out = [lab(j) for j in pool if j not in xs]
        rng.shuffle(out)
        return out

    for key, prm in v[1]:
        if key == "n_pieces_start":
            add(1, key, prm, "How many chess pieces are on the board at the start?", n,
                L.near_ints(n, 1, 12), False, _intro(scene))
        elif key == "n_type":
            k = types.count(prm["type"])
            add(1, key, prm, f"How many {sim.PLURAL[prm['type']]} are on the board at the start?", k,
                L.near_ints(k, 0, 9), False, _intro(scene))
        elif key == "start_square":
            x = prm["x"]
            gt = sim.name(P[0][x])
            ds = [sim.name(s) for s in dict.fromkeys(P[r][x] for r in range(1, R + 1)) if s is not None]
            ds += _near_squares(scene, P[0][x], rng)
            add(1, key, prm, f"On which square does {lab(x)} start?", gt, [d for d in ds if d != gt], False,
                _intro(scene))
        elif key == "end_square":
            x = prm["x"]
            gt = sim.name(P[R][x])
            ds = [sim.name(s) for s in dict.fromkeys(P[r][x] for r in range(0, R)) if s is not None]
            ds += _near_squares(scene, P[R][x], rng)
            add(1, key, prm, f"On which square is {lab(x)} at the end of the video?", gt,
                [d for d in ds if d != gt], False, _hold(scene))
        elif key == "color_of_type":
            i = types.index(prm["type"])
            gt = pcs[i]["color"]
            ds = [pcs[j]["color"] for j in range(n) if j != i]
            rng.shuffle(ds)
            ds += [c for c, _ in sim.COLORS if c not in {p["color"] for p in pcs}]
            add(1, key, prm, f"What colour is the {sim.NAME[prm['type']]}?", gt, ds, False, _intro(scene))
        elif key == "n_holes":
            k = len(scene["holes"])
            add(1, key, prm, "How many squares are cut out of the board (missing squares)?", k,
                L.near_ints(k, 1, 12), False, 1.0)
        elif key == "n_pieces_end":
            k = sum(1 for s in P[R] if s is not None)
            add(1, key, prm, "How many chess pieces are on the board at the end of the video?", k,
                L.near_ints(k, 1, 12), False, _hold(scene))

    first = {i: (mr[i][0] if mr[i] else 99) for i in range(n)}
    seen_fm = set()
    for key, prm in v[2]:
        if key == "first_mover":
            x = prm["x"]
            later = sorted(j for j in range(n) if first[j] > first[x])
            opts = sorted([x] + rng.sample(later, 3))
            rounds = sorted({first[j] for j in opts if first[j] < 99})
            add(2, key, {"x": x, "options": opts}, "Which of these pieces is the first to move?", lab(x),
                [lab(j) for j in opts if j != x], True, _round_rests(scene, rounds))
            seen_fm.add(x)
        elif key == "moved_with":
            x = prm["x"]
            mate = sorted({m["p"] for r in mr[x] for m in scene["moves"][r - 1] if m["p"] != x})[0]
            lonely = sorted(j for j in range(n) if j != x and j != mate and cnt[j] and capr[j] is None
                            and not set(mr[j]) & set(mr[x]))
            opts = sorted([mate] + rng.sample(lonely, 3))
            rounds = sorted(set(mr[x]) | {r for j in opts for r in mr[j]})
            add(2, key, {"x": x, "options": opts},
                f"Which of these pieces moves at the same moment as {lab(x)} at least once?", lab(mate),
                [lab(j) for j in opts if j != mate], True, _round_rests(scene, rounds))
        elif key == "more_moves":
            a, b = prm["pair"]
            gt = lab(a if cnt[a] > cnt[b] else b)
            dis = [lab(b if cnt[a] > cnt[b] else a), "they move equally often", "neither of them moves"]
            add(2, key, prm, f"Which moves more times: {lab(a)} or {lab(b)}?", gt, dis, True,
                _piece_rests(scene, [a, b]))
        elif key == "moved_before":
            a, b = prm["pair"]
            gt = lab(a if first[a] < first[b] else b)
            dis = [lab(b if first[a] < first[b] else a), "they first move at the same moment", "neither of them moves"]
            add(2, key, prm, f"Which starts moving earlier: {lab(a)} or {lab(b)}?", gt, dis, True,
                _round_rests(scene, [first[a], first[b]]))
        elif key == "captured_by":
            x = prm["x"]
            r = capr[x]
            cap = next(m["p"] for m in scene["moves"][r - 1] if m["capture"] == x)
            pool = [j for j in range(n) if j not in (x, cap)]
            add(2, key, prm, f"Which piece captures {lab(x)} (lands on its square, making it disappear)?",
                lab(cap), others([x, cap], pool), True, _round_rests(scene, [r]))

    for key, prm in v[3]:
        x = prm.get("x")
        if key == "slide_len":
            r = mr[x][prm["k"] - 1]
            m = sim.move_of(scene, x, r)
            gt = sim.cheb(sim.sq(m["from"]), sim.sq(m["to"]))
            add(3, key, prm, f"How many squares does {lab(x)} travel in its {ORD[prm['k'] - 1]} move?", gt,
                L.near_ints(gt, 1, scene["N"] - 1), True, _round_rests(scene, [r]))
        elif key == "land_time":
            r = mr[x][prm["k"] - 1]
            tenths = sim.timer_tenths(sim.round_frames(scene, r)[1])
            add(3, key, prm, f"What does the clock show when {lab(x)} comes to rest after its "
                f"{ORD[prm['k'] - 1]} move?", sim.clock_text(tenths), _time_opts(tenths, nf), True, 0.25)
        elif key == "move_count":
            add(3, key, prm, f"How many times does {lab(x)} move during the video?", cnt[x],
                L.near_ints(cnt[x], 0, R), True, _piece_rests(scene, [x]))
        elif key == "square_at_time":
            t = prm["t"]
            lo, _ = sim.frames_showing(t)
            rest = next((a, b, s) for a, b, s in sim.rests(scene, x) if a <= lo <= b)
            gt = sim.name(rest[2])
            ds = [sim.name(s) for s in dict.fromkeys(P[r][x] for r in range(R + 1)) if s is not None]
            ds += _near_squares(scene, rest[2], rng)
            add(3, key, prm, f"On which square is {lab(x)} when the clock shows {sim.clock_text(t)}?", gt,
                [d for d in ds if d != gt], True, (rest[1] - rest[0]) / sim.FPS)
        elif key == "n_moving_at":
            t = prm["t"]
            lo, _ = sim.frames_showing(t)
            r = next(r for r in all_rounds if sim.round_frames(scene, r)[0] < lo < sim.round_frames(scene, r)[1])
            k = len(scene["moves"][r - 1])
            add(3, key, prm, f"How many pieces are moving when the clock shows {sim.clock_text(t)}?", k,
                L.near_ints(k, 0, 6), True, scene["timing"]["move_f"] / sim.FPS)
        elif key == "n_captures":
            k = sum(1 for mv in scene["moves"] for m in mv if m["capture"] is not None)
            rounds = [r for r in all_rounds if any(m["capture"] is not None for m in scene["moves"][r - 1])]
            add(3, key, prm, "How many captures happen during the video (a piece landing on another piece, "
                "which then disappears)?", k, L.near_ints(k, 0, 6), True, _round_rests(scene, rounds))

    ill = scene["illegal"]
    for key, prm in v[4]:
        x = prm.get("x")
        if key == "illegal_piece":
            if ill:
                gt = lab(ill["p"])
                same = [m["p"] for m in scene["moves"][ill["round"] - 1] if m["p"] != ill["p"]]
                rest = [j for j in range(n) if j != ill["p"] and j not in same and cnt[j]]
                rng.shuffle(same)
                rng.shuffle(rest)
                dis = [NO_ILLEGAL] + [lab(j) for j in same + rest]
                gap = min(_round_rests(scene, [ill["round"]]), scene["timing"]["move_f"] / sim.FPS)
            else:
                gt = NO_ILLEGAL
                dis = others([], [j for j in range(n) if cnt[j]]) + others([], [j for j in range(n) if not cnt[j]])
                gap = min(_round_rests(scene, all_rounds), scene["timing"]["move_f"] / sim.FPS)
            add(4, key, prm, ILLEGAL_DEF, gt, dis, True, gap)
        elif key == "illegal_time":
            tenths = sim.timer_tenths(sim.round_frames(scene, ill["round"])[1])
            add(4, key, prm, "One piece makes a move its type is not allowed to make (a path its type can never "
                "take, or a slide through another piece). What does the clock show when that piece comes to rest "
                "after this move?", sim.clock_text(tenths), _time_opts(tenths, nf), True, 0.25)
        elif key == "total_moves":
            k = sum(len(mv) for mv in scene["moves"])
            add(4, key, prm, "How many moves are made in total during the video (count every move of every "
                "piece)?", k, L.near_ints(k, R), True, _round_rests(scene, all_rounds))
        elif key == "most_moves":
            top = max(range(n), key=lambda i: (cnt[i], -i))
            order = sorted(range(n), key=lambda i: (-cnt[i], i))
            add(4, key, prm, "Which piece moves the most times during the video?", lab(top),
                [lab(j) for j in order if j != top], True, _round_rests(scene, all_rounds))
        elif key == "never_on":
            stood = [s for s in dict.fromkeys(P[r][x] for r in range(R + 1)) if s is not None]
            path = set(stood)
            for r in mr[x]:
                m = sim.move_of(scene, x, r)
                path |= set(sim.between(sim.sq(m["from"]), sim.sq(m["to"])))
            near = []
            for s in stood:
                near += [q for q in _near_squares(scene, s, rng) if sim.sq(q) not in path and q not in near]
            if not near:
                continue
            gt = near[0]
            ds = [sim.name(s) for s in rng.sample(stood, 3)]
            opts = sorted([gt] + ds)
            add(4, key, {"x": x, "options": opts}, f"Which of these squares does {lab(x)} never stand on?",
                gt, ds, True, _piece_rests(scene, [x]))
        elif key == "total_slide":
            gt = sum(sim.cheb(sim.sq(sim.move_of(scene, x, r)["from"]), sim.sq(sim.move_of(scene, x, r)["to"]))
                     for r in mr[x])
            add(4, key, prm, f"Adding up the length of each of its moves, how many squares does {lab(x)} travel "
                "in total?", gt, L.near_ints(gt, 1), True, _piece_rests(scene, [x]))
        elif key == "end_distance":
            gt = sim.cheb(P[0][x], P[R][x])
            add(4, key, prm, f"At the end, how many king steps away from its starting square is {lab(x)} "
                "(the larger of the file and rank differences)?", gt, L.near_ints(gt, 0, scene["N"] - 1), True,
                min(_intro(scene), _hold(scene)))
        elif key == "most_captures":
            caps = {i: sum(1 for mv in scene["moves"] for m in mv if m["p"] == i and m["capture"] is not None)
                    for i in range(n)}
            top = max(range(n), key=lambda i: (caps[i], -i))
            order = sorted(range(n), key=lambda i: (-caps[i], i))
            rounds = [r for r in all_rounds if any(m["capture"] is not None for m in scene["moves"][r - 1])]
            add(4, key, prm, "Which piece captures the most pieces?", lab(top), [lab(j) for j in order if j != top],
                True, _round_rests(scene, rounds))
        elif key == "first_captured":
            evs = sorted((r, m["capture"]) for r, mv in enumerate(scene["moves"], 1) for m in mv
                         if m["capture"] is not None)
            vic = evs[0][1]
            dis = [lab(e[1]) for e in evs[1:]] + others([vic] + [e[1] for e in evs])
            add(4, key, prm, "Which piece is captured first?", lab(vic), dis, True,
                _round_rests(scene, [e[0] for e in evs]))

    for key, prm in v[5]:
        x = prm.get("x")
        if key == "cf_skip_attacks":
            r = mr[x][prm["k"] - 1]
            Q = list(P[r])
            Q[x] = P[r - 1][x]
            gt, real = sim.attacks(scene, Q, x), sim.attacks(scene, P[r], x)
            add(5, key, prm, f"Suppose {lab(x)} had skipped its {ORD[prm['k'] - 1]} move and every other piece "
                f"moved exactly as in the video. How many other pieces would {lab(x)} attack right after that "
                f"round of moves? {_attack_def(scene)}", gt, [real] + L.near_ints(gt, 0, max(n - 1, 4)), True,
                _round_rests(scene, [r]))
        elif key == "cf_never_moved_block":
            r0, hit = sim.first_blocked(scene, P[0][x], mr[x][0], skip_piece=x)
            gt = lab(hit[0]) if hit else NO_BLOCK
            later = sorted({m["p"] for q in range(mr[x][0], R + 1) for m in scene["moves"][q - 1] if m["p"] != x})
            pool = [j for j in later if not hit or j != hit[0]] + [j for j in range(n) if j not in later and j != x]
            dis = ([NO_BLOCK] if hit else []) + [lab(j) for j in pool]
            add(5, key, prm, f"Suppose {lab(x)} had never left its starting square, and every other piece tried to "
                f"make the same moves as in the video. Which piece would be the first to find a move blocked "
                f"because it would have to land on {lab(x)} or slide through its square (knights jump over "
                "pieces)?", gt, dis, True, _round_rests(scene, list(range(mr[x][0], R + 1))))
        elif key == "cf_hole":
            s = sim.sq(prm["sq"])
            r0, hit = sim.first_blocked(scene, s, 1)
            gt = lab(hit[0])
            movers = sorted({m["p"] for mv in scene["moves"] for m in mv if m["p"] != hit[0]})
            dis = [NO_BLOCK] + [lab(j) for j in movers] + others([hit[0]] + movers)
            add(5, key, prm, f"Suppose square {prm['sq']} had also been cut out of the board, and every piece "
                "tried to make the same moves as in the video. Which piece would be the first unable to make one "
                "of its moves (no piece may land on a missing square or slide across one; knights jump over "
                "them)?", gt, dis, True, _round_rests(scene, all_rounds))
        elif key == "cf_never_moved_left":
            gt, _ = sim.never_moved_left(scene, x)
            real = sum(1 for s in P[R] if s is not None)
            add(5, key, prm, f"Suppose {lab(x)} had never moved, and every other piece made the same moves as in "
                "the video (a piece that lands on another still captures it). How many pieces would be on the "
                "board at the end?", gt, [real] + L.near_ints(gt, 1, n), True, _round_rests(scene, all_rounds))
        elif key == "cf_fix_next":
            r2, kind = sim.fix_next(scene)
            gt = FIX[kind]
            add(5, key, prm, f"Suppose {lab(ill['p'])} had stayed put instead of making its illegal move, and "
                "later made its next move to the same square it actually moves to next. What would that next "
                "move be?", gt, [FIX[k] for k in FIX if k != kind], True, _round_rests(scene, [ill["round"], r2]))
    return C


FEATURED = {"rogue": ("illegal_piece", "illegal_time", "cf_fix_next"),
            "capture": ("captured_by", "n_captures", "most_captures", "first_captured", "cf_never_moved_left",
                        "n_pieces_end"),
            "holes": ("n_holes", "cf_hole"), "relay": (), "dance": ()}


def _pick(rng, items, k, featured=()):
    """k items, spreading over as many derivation keys as possible; the
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
        for key in keys:
            pool = [it for it in by[key] if it not in out]
            if pool and len(out) < k:
                out.append(rng.choice(pool))
        if all(it in out for key in keys for it in by[key]):
            break
    return out


def build(scene: dict, rng: random.Random) -> list:
    C = candidates(scene, rng)
    counts = L.ten_counts(rng)
    qs = []
    for lv in range(1, 6):
        for key, params, text, gt, dis, motion, mf in _pick(rng, C[lv], counts[lv - 1], FEATURED[scene["scenario"]]):
            gts = str(gt)
            ds = [str(d) for d in dis if str(d) != gts]
            opts, let = L.mc(rng, gts, ds)
            if len(opts) != 4:
                raise ValueError(f"{key}: fewer than 4 options")
            qs.append(L.question(lv, key, text, gt, opts, let, motion, {**params, "min_fps": mf}))
    return L.number(qs)
