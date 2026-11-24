"""Independent answers for mirror_beam_reflection: a fresh vector trace of the beam on the stored grid."""
from __future__ import annotations


def _run(mirrors, start, d, n):
    """-> (cells visited, [(cell, kind)] bounces, exit cell, exit edge) or None if it loops."""
    pos, vel = complex(start[1], start[0]), complex(d[1], d[0])   # x = col, y = row
    cells, hits, seen = [(start[0], start[1])], [], set()
    for _ in range(200):
        st = (pos, vel)
        if st in seen:
            return None
        seen.add(st)
        pos += vel
        r, c = int(pos.imag), int(pos.real)
        if pos.imag < 0 or pos.real < 0 or r >= n or c >= n:
            edge = ("top" if pos.imag < 0 else "bottom" if r >= n else "left" if pos.real < 0 else "right")
            return cells, hits, cells[-1], edge
        cells.append((r, c))
        m = mirrors.get(f"{r}-{c}")
        if m == "/":       # swap and negate components
            vel = complex(-vel.imag, -vel.real)
            hits.append(((r, c), m))
        elif m == "\\":    # swap components
            vel = complex(vel.imag, vel.real)
            hits.append(((r, c), m))
    return None


def _lab(c):
    return f"row {c[0]}, col {c[1]}"


def answer(key: str, p: dict, scene: dict):
    n, ms, k = scene["grid_n"], scene["mirrors"], scene["truncate_index"]
    cells, hits, ex, edge = _run(ms, scene["start"], scene["dir"], n)
    shown_hits = [(cells[i], ms[f"{cells[i][0]}-{cells[i][1]}"]) for i in range(1, k + 1)
                  if f"{cells[i][0]}-{cells[i][1]}" in ms]
    if key == "mirror_count":
        return len(ms)
    if key == "backslash_mirror_count":
        return sum(1 for m in ms.values() if m == "\\")
    if key == "entry_side":
        dr, dc = scene["dir"]
        return {(0, 1): "left", (0, -1): "right", (1, 0): "top", (-1, 0): "bottom"}[(dr, dc)]
    if key == "slash_mirror_count":
        return sum(1 for m in ms.values() if m == "/")
    if key == "bounces_shown":
        return len(shown_hits)
    if key == "first_mirror_hit":
        return _lab(hits[0][0])
    if key == "third_mirror_hit":
        return _lab(hits[2][0])
    if key == "cells_shown":
        return len(set(cells[:k + 1]))
    if key == "backslash_hits_shown":
        return sum(1 for _, m in shown_hits if m == "\\")
    if key == "slash_hits_shown":
        return sum(1 for _, m in shown_hits if m == "/")
    if key == "exit_edge":
        return edge
    if key == "exit_cell":
        return _lab(ex)
    if key in ("cf_insert_exit_edge", "cf_insert_exit_cell"):
        r = _run({**ms, f"{p['cell'][0]}-{p['cell'][1]}": p["kind"]}, scene["start"], scene["dir"], n)
        return r[3] if key.endswith("edge") else _lab(r[2])
    if key == "cf_remove_exit_edge":
        r = _run({a: b for a, b in ms.items() if a != f"{p['cell'][0]}-{p['cell'][1]}"},
                 scene["start"], scene["dir"], n)
        return r[3]
    if key in ("cf_flip_exit_edge", "cf_flip_exit_cell"):
        ck = f"{p['cell'][0]}-{p['cell'][1]}"
        r = _run({**ms, ck: "\\" if ms[ck] == "/" else "/"}, scene["start"], scene["dir"], n)
        return r[3] if key.endswith("edge") else _lab(r[2])
    raise KeyError(key)
