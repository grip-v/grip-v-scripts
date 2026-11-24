"""Hex-board primitives for random_walk_home -- a self-contained,
simplified relative of the earlier single-agent graphs.py: no 3-coloring rule
here (this board's black cells are plain obstacles, not a soccer-ball
analogue), and blocked/walkable status can change over the course of a
clip (see simulate.py's tile-swap layer), so this module only provides
the static geometry -- neighbor lookup, pixel conversion, hex distance --
plus the initial random blocked-cell layout.
"""
from __future__ import annotations
import math
from random import Random

import networkx as nx

import config as C

HEX_NEIGHBOR_OFFSETS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))


def cube_distance(a, b) -> int:
    dq, dr = b[0] - a[0], b[1] - a[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def axial_to_pixel(q, r, size) -> tuple:
    x = size * (1.5 * q)
    y = size * (math.sqrt(3) / 2 * q + math.sqrt(3) * r)
    return (x, y)


def board_cells(radius: int) -> list:
    cells = []
    for q in range(-radius, radius + 1):
        r_lo, r_hi = max(-radius, -q - radius), min(radius, -q + radius)
        for r in range(r_lo, r_hi + 1):
            cells.append((q, r))
    return cells


def neighbors_on_board(cell, board_set) -> list:
    q, r = cell
    return [(q + dq, r + dr) for dq, dr in HEX_NEIGHBOR_OFFSETS if (q + dq, r + dr) in board_set]


def initial_blocked_layout(rng: Random, board: list) -> dict:
    """Random ~BLACK_FRACTION of the board is blocked at t=0; returns
    {cell: bool}. Reserved cells (homes/starts/holes/predators) are chosen
    afterward, ONLY from the largest connected component of the resulting
    white subgraph -- see simulate.py -- so no retry loop is needed here."""
    return {cell: (rng.random() < C.BLACK_FRACTION) for cell in board}


def white_subgraph(board_set, blocked: dict) -> nx.Graph:
    g = nx.Graph()
    white = [c for c in board_set if not blocked.get(c, False)]
    g.add_nodes_from(white)
    white_set = set(white)
    for c in white:
        for n in neighbors_on_board(c, white_set):
            g.add_edge(c, n)
    return g


def largest_component_nodes(board_set, blocked: dict) -> list:
    g = white_subgraph(board_set, blocked)
    if g.number_of_nodes() == 0:
        return []
    return list(max(nx.connected_components(g), key=len))
