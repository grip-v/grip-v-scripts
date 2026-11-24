"""Independent answers for tile_alignment_shuffle.

Nothing comes from sim.py's bookkeeping (no grids, move counts, first
moves, line rules or timer helper). The renderer's own per-frame geometry
(render.state) is read for EVERY frame of the clip:
  * on frames where nothing moves, each tile's pixel centre is snapped to a
    cell using the board layout -> the resting board;
  * consecutive different resting boards give the moves, as the set of
    tiles that changed cell (with from/to cells), and the frame on which
    each new board first rests;
  * the clock reading is recomputed from the frame number here.
Line scans, counts, paths, clock times and every counterfactual replay are
done in this file, from those recovered boards and moves.
"""
from __future__ import annotations
import functools
import json

import render

FPS = 24
SHAPE_NAMES = {"circle", "square", "triangle", "star", "heart", "hexagon", "diamond", "cross", "crescent",
               "pentagon"}


@functools.lru_cache(maxsize=32)
def _watch(key: str):
    scene = json.loads(key)
    n = scene["n"]
    lay = scene["layout"]
    cell = lay["span"] / n
    boards, rest_frames, moves = [], [], []
    for f in range(scene["n_frames"]):
        st = render.state(scene, f)
        if st["moving"]:
            continue
        g = [[None] * n for _ in range(n)]
        for tid, (x, y) in st["pos"].items():
            r, c = int((y - lay["y0"]) // cell), int((x - lay["x0"]) // cell)
            g[r][c] = tid
        if not boards or g != boards[-1]:
            boards.append(g)
            rest_frames.append(f)
    for a, b in zip(boards, boards[1:]):
        pa, pb = _pos(a), _pos(b)
        moves.append({t: (pa[t], pb[t]) for t in pa if pa[t] != pb[t]})
    return boards, rest_frames, moves


def _pos(g):
    return {t: (r, c) for r, row in enumerate(g) for c, t in enumerate(row)}


def _clock(f: int) -> str:
    tenths = f * 10 // FPS
    return f"{tenths // 10}.{tenths % 10} s"


def _cell(rc) -> str:
    return f"row {rc[0] + 1}, column {rc[1] + 1}"


def _ord(k: int) -> str:
    suf = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suf}"


def _dname(d) -> str:
    return f"{d['color']} {d['kind']}" + ("" if d["kind"] in SHAPE_NAMES else " symbol")


def _all_lines(n):
    ls = [[(r, c) for c in range(n)] for r in range(n)] + [[(r, c) for r in range(n)] for c in range(n)]
    return ls + [[(i, i) for i in range(n)], [(i, n - 1 - i) for i in range(n)]]


def _full_lines(g, look):
    return [cs for cs in _all_lines(len(g)) if len({look(g[r][c]) for r, c in cs}) == 1]


def _line_text(cs, n) -> str:
    rows, cols = {r for r, _ in cs}, {c for _, c in cs}
    if len(rows) == 1:
        return f"row {cs[0][0] + 1}"
    if len(cols) == 1:
        return f"column {cs[0][1] + 1}"
    return "the diagonal from top-left to bottom-right" if (0, 0) in cs else \
        "the diagonal from top-right to bottom-left"


def _replay(start, moves, skip=None):
    """Replay cell-to-cell moves (each a {tid: (from, to)} of the real run) on `start`,
    leaving out move `skip`: whatever tile sits in a move's from-cell goes to its to-cell."""
    g = [row[:] for row in start]
    counts = {}
    for j, m in enumerate(moves):
        if j == skip:
            continue
        new = [row[:] for row in g]
        for fr, to in m.values():
            t = g[fr[0]][fr[1]]
            new[to[0]][to[1]] = t
            counts[t] = counts.get(t, 0) + 1
        g = new
    return g, counts


def answer(key: str, p: dict, scene: dict):
    boards, rest, moves = _watch(json.dumps(scene, sort_keys=True))
    n = scene["n"]
    start, end = boards[0], boards[-1]
    D, T = scene["designs"], scene["tiles"]
    look = lambda t: (D[T[t]]["kind"], D[T[t]]["color"])  # noqa: E731
    name = lambda t: _dname(D[T[t]])  # noqa: E731
    tids = sorted(_pos(start))
    path = {t: [_pos(g)[t] for g in boards] for t in tids}
    nmov = {t: sum(1 for m in moves if t in m) for t in tids}
    first = {t: next((j for j, m in enumerate(moves) if t in m), None) for t in tids}
    at = lambda x: start[x[0]][x[1]]  # noqa: E731
    first_line = next((k for k, g in enumerate(boards) if _full_lines(g, look)), None)
    S = len(moves)

    if key == "count_design":
        d = D[p["design"]]
        return sum(1 for t in tids if look(t) == (d["kind"], d["color"]))
    if key == "n_false":
        looks = [look(t) for t in tids]
        return sum(1 for x in looks if looks.count(x) == 1)
    if key == "n_kinds":
        return len({look(t) for t in tids})
    if key == "n_moves":
        return S
    if key == "tile_at_start":
        return name(at(p["cell"]))
    if key == "tile_at_end":
        return name(end[p["cell"][0]][p["cell"][1]])
    if key == "line_where":
        return _line_text(_full_lines(end, look)[0], n)
    if key == "aligned_design":
        cs = _full_lines(end, look)[0]
        return name(end[cs[0][0]][cs[0][1]])
    if key == "partner":
        t = at(p["cell"])
        k = first[t]
        fr = moves[k][t][0]
        return name(boards[k + 1][fr[0]][fr[1]])
    if key in ("first_mover", "more_moves"):
        a, b = (tuple(x) for x in p["cells"])
        ta, tb = at(a), at(b)
        A, B = f"the one that starts in {_cell(a)}", f"the one that starts in {_cell(b)}"
        if key == "first_mover":
            fa, fb = (first[t] if first[t] is not None else 10 ** 6 for t in (ta, tb))
            return A if fa < fb else B if fb < fa else None
        if nmov[ta] == nmov[tb]:
            return "they move the same number of times" if nmov[ta] else "neither of them ever moves"
        return A if nmov[ta] > nmov[tb] else B
    if key == "moves_of":
        return nmov[at(p["cell"])]
    if key == "end_cell":
        return _cell(path[at(p["cell"])][-1])
    if key == "cell_after":
        return _cell(path[at(p["cell"])][p["k"]])
    if key == "first_rest_time":
        return _clock(rest[first[at(p["cell"])] + 1])
    if key == "line_after":
        return f"the {_ord(first_line)} move"
    if key == "line_time":
        return _clock(rest[first_line])
    if key == "most_moves":
        top = max(nmov.values())
        best = [t for t in tids if nmov[t] == top]
        return _cell(path[best[0]][0]) if len(best) == 1 else None
    if key == "start_of":
        return _cell(path[end[p["cell"][0]][p["cell"][1]]][0])
    if key == "never_moved":
        return sum(1 for t in tids if nmov[t] == 0)
    if key == "back_home":
        return sum(1 for t in tids if nmov[t] and path[t][0] == path[t][-1])
    if key == "cells_visited":
        return len(set(path[at(p["cell"])]))
    if key == "line_origin":
        cs = _full_lines(end, look)[0]
        return sum(1 for r, c in cs if path[end[r][c]][0] in cs)
    if key == "partner_origin":
        t = at(p["cell"])
        m = moves[first[t]]
        fr, to = m[t]
        other = next(o for o, (f2, t2) in m.items() if f2 == to and t2 == fr)
        return _cell(path[other][0])
    if key == "cf_skip":
        g, _ = _replay(start, moves, p["k"])
        return _cell(_pos(g)[at(p["cell"])])
    if key == "cf_skip_moves":
        _, counts = _replay(start, moves, p["k"])
        return counts.get(at(p["cell"]), 0)
    if key == "cf_skip_lines":
        g, _ = _replay(start, moves, p["k"])
        return len(_full_lines(g, look))
    if key == "cf_false_real":
        ft = next(t for t in tids if T[t] == p["design"])
        a = D[p["as"]]
        alt = lambda t: (a["kind"], a["color"]) if t == ft else look(t)  # noqa: E731
        return len(_full_lines(end, alt))
    return None
