"""Independent answers for chess_dance_trace.

Nothing here reads the generator's move list, flags or helpers. The clip is
decoded from what the renderer draws: `render.track(scene, i)` gives every
piece's on-screen centre and visibility in frame i. From that per-frame
track this module recovers each glide (the frames where a piece's centre
changes), the squares it rests on (pixel -> square through the board
geometry), the rounds (glides that start on the same frame), captures (a
piece vanishing on the frame another lands on its square), and the clock
reading of any frame. Legality, attacks and every counterfactual are then
re-derived with the hand-written direction-stepping geometry below (no
python-chess).
"""
from __future__ import annotations
import functools
import json

import render

FILES = "ABCDEFGH"
NAME = {"Q": "queen", "K": "king", "R": "rook", "B": "bishop", "N": "knight"}
ORTH = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
KNIGHT = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
FPS = 24
NO_ILLEGAL = "no piece does"
NO_BLOCK = "no piece would be blocked"
FIX = {None: "a legal move", "pattern": "illegal: a path its type can never take",
       "blocked": "illegal: it slides through another piece", "occupied": "illegal: it lands on an occupied square"}


def sname(s):
    return FILES[s[0]] + str(s[1] + 1)


def parse(n):
    return FILES.index(n[0]), int(n[1:]) - 1


def clock(i: int) -> str:
    k = (10 * i) // FPS
    return f"{k // 10}.{k % 10} s"


def frames_of(tenths: int):
    return [i for i in range(tenths * FPS // 10 - 2, (tenths + 1) * FPS // 10 + 2) if (10 * i) // FPS == tenths]


# ---------------------------------------------------------------- geometry

def _on(s, N):
    return 0 <= s[0] < N and 0 <= s[1] < N


def targets(t, o, occ, N, holes=frozenset()):
    """Squares a `t` on `o` reaches/attacks: slides stop at the first piece
    (included) or hole (excluded); knights and kings step."""
    out = set()
    if t in "KN":
        for df, dr in (ORTH + DIAG if t == "K" else KNIGHT):
            s = (o[0] + df, o[1] + dr)
            if _on(s, N) and s not in holes:
                out.add(s)
        return out
    for df, dr in {"R": ORTH, "B": DIAG, "Q": ORTH + DIAG}[t]:
        s = (o[0] + df, o[1] + dr)
        while _on(s, N) and s not in holes:
            out.add(s)
            if s in occ:
                break
            s = (s[0] + df, s[1] + dr)
    return out


def line(a, b):
    """Squares strictly between a and b on a rank, file or diagonal, else []."""
    df, dr = b[0] - a[0], b[1] - a[1]
    if not (df == 0 or dr == 0 or abs(df) == abs(dr)):
        return []
    n = max(abs(df), abs(dr))
    sf, sr = (df > 0) - (df < 0), (dr > 0) - (dr < 0)
    return [(a[0] + sf * k, a[1] + sr * k) for k in range(1, n)]


def why_illegal(t, a, b, occ, N, holes, capture_ok):
    if b in holes:
        return "hole"
    if b in occ and not capture_ok:
        return "occupied"
    if b not in targets(t, a, set(), N):
        return "pattern"
    if b not in targets(t, a, occ, N, holes):
        return "blocked"
    return None


def dist(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


# ---------------------------------------------------------------- decoding the track

@functools.lru_cache(maxsize=8)
def _decode(key: str):
    scene = json.loads(key)
    st = scene["style"]
    N, c, x0, y0 = scene["N"], st["cell"], st["x0"], st["y0"]
    n = len(scene["pieces"])
    nf = scene["n_frames"]
    T = [render.track(scene, i) for i in range(nf)]

    def square(x, y):
        f, r = (x - x0) / c - 0.5, N - 1 - ((y - y0) / c - 0.5)
        fi, ri = int(round(f)), int(round(r))
        if abs(f - fi) > 1e-6 or abs(r - ri) > 1e-6:
            return None
        return fi, ri

    moves = []
    vis_end = {}
    for p in range(n):
        xs = [(T[i][p][0], T[i][p][1]) for i in range(nf)]
        vis = [T[i][p][2] for i in range(nf)]
        gone = next((i for i in range(nf) if not vis[i]), None)
        vis_end[p] = gone
        last = gone if gone is not None else nf
        i = 1
        while i < last:
            if xs[i] != xs[i - 1]:
                s = i - 1
                j = i
                while j + 1 < last and xs[j + 1] != xs[j]:
                    j += 1
                moves.append({"p": p, "s": s, "land": j, "a": square(*xs[s]), "b": square(*xs[j])})
                i = j + 1
            else:
                i += 1
    moves.sort(key=lambda m: (m["s"], m["p"]))
    starts = sorted({m["s"] for m in moves})
    for m in moves:
        m["round"] = starts.index(m["s"]) + 1

    def board(i):
        """Resting squares of the visible pieces at frame i."""
        return {p: square(T[i][p][0], T[i][p][1]) for p in range(n) if T[i][p][2]}

    for m in moves:
        m["victim"] = None
        for q, g in vis_end.items():
            if g is not None and g == m["land"] and board(m["land"] - 1).get(q) == m["b"]:
                m["victim"] = q
    types = [p["type"] for p in scene["pieces"]]
    holes = frozenset(parse(h) for h in scene["holes"])
    capture_ok = any(m["victim"] is not None for m in moves)
    for m in moves:
        occ = set(board(m["s"]).values())
        m["why"] = why_illegal(types[m["p"]], m["a"], m["b"], occ, N, holes, capture_ok)
    return {"scene": scene, "n": n, "N": N, "nf": nf, "T": T, "moves": moves, "board": board,
            "types": types, "holes": holes, "gone": vis_end, "rounds": len(starts), "starts": starts,
            "capture_ok": capture_ok}


def decode(scene):
    return _decode(json.dumps(scene, sort_keys=True))


def lab(D, i):
    p = D["scene"]["pieces"][i]
    return f"the {p['color']} {NAME[p['type']]}"


def _own(D, x):
    return [m for m in D["moves"] if m["p"] == x]


def _attack_count(D, B, x):
    occ = {s for j, s in B.items() if j != x}
    hit = targets(D["types"][x], B[x], occ, D["N"], D["holes"])
    return sum(1 for j, s in B.items() if j != x and s in hit)


def _first_block(D, sqr, from_start_frame, skip=None):
    """Earliest round (start frame >= from_start_frame) in which a move (not
    by `skip`) lands on `sqr` or slides through it; knights jump."""
    for s in D["starts"]:
        if s < from_start_frame:
            continue
        hit = sorted(m["p"] for m in D["moves"] if m["s"] == s and m["p"] != skip and
                     (m["b"] == sqr or (D["types"][m["p"]] != "N" and sqr in line(m["a"], m["b"]))))
        if hit:
            return hit
    return []


def _resim_frozen(D, x):
    """Replay every decoded move except x's, in round order, with x never
    moving; a landing on a piece captures it. None if a move becomes
    impossible."""
    cur = {p: s for p, s in D["board"](0).items()}
    for s in D["starts"]:
        mv = [m for m in D["moves"] if m["s"] == s and m["p"] != x and m["p"] in cur]
        occ = set(cur.values())
        movers = {m["p"] for m in mv}
        lands = []
        for m in mv:
            if cur[m["p"]] != m["a"]:
                return None
            if why_illegal(D["types"][m["p"]], m["a"], m["b"], occ, D["N"], D["holes"], True):
                return None
            victim = next((q for q, sq in cur.items() if sq == m["b"]), None)
            if victim is not None and victim in movers:
                return None
            lands.append((m["p"], m["b"], victim))
        if len({b for _, b, _ in lands}) != len(lands):
            return None
        for p, b, victim in lands:
            if victim is not None:
                del cur[victim]
            cur[p] = b
    return len(cur)


# ---------------------------------------------------------------- answers

def answer(key: str, p: dict, scene: dict):
    D = decode(scene)
    n, nf, moves = D["n"], D["nf"], D["moves"]
    types = D["types"]
    B0, BE = D["board"](0), D["board"](nf - 1)
    cnt = {i: len(_own(D, i)) for i in range(n)}
    first = {i: (_own(D, i)[0]["s"] if cnt[i] else 10 ** 6) for i in range(n)}
    illegal = [m for m in moves if m["why"]]

    if key == "n_pieces_start":
        return len(B0)
    if key == "n_type":
        return sum(1 for i in B0 if types[i] == p["type"])
    if key == "start_square":
        return sname(B0[p["x"]])
    if key == "end_square":
        return sname(BE[p["x"]])
    if key == "color_of_type":
        hit = [i for i in B0 if types[i] == p["type"]]
        return D["scene"]["pieces"][hit[0]]["color"] if len(hit) == 1 else None
    if key == "n_holes":
        return len(D["holes"])
    if key == "n_pieces_end":
        return len(BE)

    if key == "first_mover":
        best = sorted(p["options"], key=lambda i: first[i])
        return lab(D, best[0]) if first[best[0]] < first[best[1]] else None
    if key == "moved_with":
        mine = {m["s"] for m in _own(D, p["x"])}
        hit = [i for i in p["options"] if mine & {m["s"] for m in _own(D, i)}]
        return lab(D, hit[0]) if len(hit) == 1 else None
    if key == "more_moves":
        a, b = p["pair"]
        if cnt[a] == cnt[b]:
            return "they move equally often"
        return lab(D, a if cnt[a] > cnt[b] else b)
    if key == "moved_before":
        a, b = p["pair"]
        if first[a] == first[b]:
            return "they first move at the same moment"
        return lab(D, a if first[a] < first[b] else b)
    if key == "captured_by":
        hit = [m["p"] for m in moves if m["victim"] == p["x"]]
        return lab(D, hit[0]) if len(hit) == 1 else None

    if key == "slide_len":
        m = _own(D, p["x"])[p["k"] - 1]
        return dist(m["a"], m["b"])
    if key == "land_time":
        return clock(_own(D, p["x"])[p["k"] - 1]["land"])
    if key == "move_count":
        return cnt[p["x"]]
    if key == "square_at_time":
        fr = frames_of(p["t"])
        sqs = {D["board"](i).get(p["x"]) for i in fr}
        moving = any(m["p"] == p["x"] and m["s"] < i < m["land"] for m in moves for i in fr)
        if moving or len(sqs) != 1 or None in sqs:
            return None
        return sname(sqs.pop())
    if key == "n_moving_at":
        fr = frames_of(p["t"])
        counts = {sum(1 for m in moves if m["s"] < i < m["land"]) for i in fr}
        return counts.pop() if len(counts) == 1 else None
    if key == "n_captures":
        return sum(1 for m in moves if m["victim"] is not None)

    if key == "illegal_piece":
        if not illegal:
            return NO_ILLEGAL
        return lab(D, illegal[0]["p"]) if len(illegal) == 1 else None
    if key == "illegal_time":
        return clock(illegal[0]["land"]) if len(illegal) == 1 else None
    if key == "total_moves":
        return len(moves)
    if key == "most_moves":
        top = max(cnt.values())
        hit = [i for i in range(n) if cnt[i] == top]
        return lab(D, hit[0]) if len(hit) == 1 else None
    if key == "never_on":
        stood = {D["board"](i).get(p["x"]) for i in range(nf)}
        miss = [s for s in p["options"] if parse(s) not in stood]
        return miss[0] if len(miss) == 1 else None
    if key == "total_slide":
        return sum(dist(m["a"], m["b"]) for m in _own(D, p["x"]))
    if key == "end_distance":
        return dist(B0[p["x"]], BE[p["x"]])
    if key == "most_captures":
        caps = {i: sum(1 for m in moves if m["p"] == i and m["victim"] is not None) for i in range(n)}
        top = max(caps.values())
        hit = [i for i in range(n) if caps[i] == top]
        return lab(D, hit[0]) if len(hit) == 1 else None
    if key == "first_captured":
        caps = sorted((m["land"], m["victim"]) for m in moves if m["victim"] is not None)
        if not caps or (len(caps) > 1 and caps[0][0] == caps[1][0]):
            return None
        return lab(D, caps[0][1])

    if key == "cf_skip_attacks":
        m = _own(D, p["x"])[p["k"] - 1]
        Bk = dict(D["board"](m["land"]))
        Bk[p["x"]] = m["a"]
        return _attack_count(D, Bk, p["x"])
    if key == "cf_never_moved_block":
        x = p["x"]
        hit = _first_block(D, B0[x], first[x], skip=x)
        if not hit:
            return NO_BLOCK
        return lab(D, hit[0]) if len(hit) == 1 else None
    if key == "cf_hole":
        hit = _first_block(D, parse(p["sq"]), 0)
        if not hit:
            return NO_BLOCK
        return lab(D, hit[0]) if len(hit) == 1 else None
    if key == "cf_never_moved_left":
        return _resim_frozen(D, p["x"])
    if key == "cf_fix_next":
        if len(illegal) != 1:
            return None
        bad = illegal[0]
        x = bad["p"]
        nxt = [m for m in _own(D, x) if m["s"] > bad["s"]]
        if not nxt:
            return None
        m2 = nxt[0]
        a = bad["a"]
        others = {s for q, s in D["board"](m2["s"]).items() if q != x}
        if a in others:
            return None
        kinds = []
        if m2["b"] in others:
            kinds.append("occupied")
        if m2["b"] not in targets(types[x], a, set(), D["N"]):
            kinds.append("pattern")
        elif m2["b"] not in targets(types[x], a, others, D["N"], D["holes"]):
            kinds.append("blocked")
        if len(kinds) > 1:
            return None
        return FIX[kinds[0] if kinds else None]
    return None
