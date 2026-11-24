"""Scene generation for fold_punch_unfold (New Problem Types B23).

An 8 x 8 gridded square sheet (columns A-H left to right, rows 1-8 top to
bottom; x right, y down, one unit per cell) is folded 1-4 times. Each fold
is along a grid line (vertical or horizontal) or a diagonal through grid
points, and one side of the crease is folded over on top of the other.
Then the folded stack is punched once or twice at cell centres; every
layer under the punch gets a hole. The clip stops before unfolding, so the
hole pattern on the opened sheet has to be worked out.

The paper is tracked as facets: a piece of the original sheet (polygon in
sheet coordinates) plus the transform that places it now (a composition of
reflections). A fold splits every facet by the crease and reflects the
moving pieces. The holes on the opened sheet are the punch point mapped
back through each facet that covers it.
"""
from __future__ import annotations
import random

from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

N = 8
COLS = "ABCDEFGH"
MARGIN = 0.3                       # punch points stay this far from any paper edge or crease
FOLDS = (1, 4)
DIAG_P = 0.3
MAX_HOLES = 16


def side(f: dict, x: float, y: float) -> float:
    c = f["c"]
    return {"v": x - c, "h": y - c, "d1": x - y - c, "d2": x + y - c}[f["axis"]]


def reflect(f: dict, x: float, y: float):
    c = f["c"]
    return {"v": (2 * c - x, y), "h": (x, 2 * c - y), "d1": (y + c, x - c), "d2": (c - y, c - x)}[f["axis"]]


def half_plane(f: dict, sign: int) -> Polygon:
    """The half plane on the `sign` side of the crease (a large polygon)."""
    c, B = f["c"], 100.0
    p0, d, g = {"v": ((c, 0), (0, 1), (1, 0)), "h": ((0, c), (1, 0), (0, 1)),
                "d1": ((c, 0), (1, 1), (1, -1)), "d2": ((c, 0), (1, -1), (1, 1))}[f["axis"]]
    nx, ny = g[0] * sign, g[1] * sign
    return Polygon([(p0[0] + B * d[0], p0[1] + B * d[1]), (p0[0] + B * d[0] + B * nx, p0[1] + B * d[1] + B * ny),
                    (p0[0] - B * d[0] + B * nx, p0[1] - B * d[1] + B * ny), (p0[0] - B * d[0], p0[1] - B * d[1])])


def _apply(T, x, y):
    for f in T:
        x, y = reflect(f, x, y)
    return x, y


def _place(T, poly: Polygon) -> Polygon:
    return Polygon([_apply(T, x, y) for x, y in poly.exterior.coords])


def facets(folds: list) -> list:
    """Bottom-to-top list of (piece in sheet coords, transform = list of folds applied)."""
    return [(p, T) for p, T, _ in facets_tagged(folds)]


def facets_tagged(folds: list) -> list:
    """Like facets(), plus a flag: True if the piece moved in the LAST fold."""
    out = [(box(0, 0, N, N), [], False)]
    for f in folds:
        mv = half_plane(f, f["move"])
        st = half_plane(f, -f["move"])
        stay, move = [], []
        for piece, T, _ in out:
            cur = _place(T, piece)
            for part, dest, extra, flag in ((cur.intersection(st), stay, [], False),
                                            (cur.intersection(mv), move, [f], True)):
                for g in getattr(part, "geoms", [part]):
                    if g.geom_type == "Polygon" and g.area > 1e-9:
                        # back to sheet coordinates: undo T (reflections are their own inverse, reversed order)
                        orig = Polygon([_apply(T[::-1], x, y) for x, y in g.exterior.coords])
                        dest.append((orig, T + extra, flag))
        out = stay + move[::-1]
    return out


def shape(folds: list):
    return unary_union([_place(T, p) for p, T in facets(folds)])


def holes(folds: list, punches: list) -> list:
    """Sorted list of (x, y) hole centres on the opened sheet, one per layer punched."""
    out = []
    for px, py in punches:
        for piece, T in facets(folds):
            if _place(T, piece).contains(Point(px, py)):
                out.append(tuple(round(v, 6) for v in _apply(T[::-1], px, py)))
    return sorted(out)


def layers(folds: list, p) -> int:
    return sum(1 for piece, T in facets(folds) if _place(T, piece).contains(Point(*p)))


def cell(x: float, y: float) -> str:
    return f"{COLS[int(x)]}{int(y) + 1}"


def clear(folds: list, p) -> bool:
    """p is on the paper and at least MARGIN from every facet edge."""
    pt = Point(*p)
    fs = [_place(T, piece) for piece, T in facets(folds)]
    return any(g.contains(pt) for g in fs) and all(g.exterior.distance(pt) >= MARGIN for g in fs)


def describe(f: dict) -> str:
    return {("v", 1): "the right part is folded over to the left",
            ("v", -1): "the left part is folded over to the right",
            ("h", 1): "the bottom part is folded up over the top",
            ("h", -1): "the top part is folded down over the bottom",
            ("d1", 1): "the upper-right part is folded over to the lower-left",
            ("d1", -1): "the lower-left part is folded over to the upper-right",
            ("d2", 1): "the lower-right part is folded over to the upper-left",
            ("d2", -1): "the upper-left part is folded over to the lower-right"}[(f["axis"], f["move"])]


def valid(folds: list) -> bool:
    sheet = box(0, 0, N, N).buffer(1e-9)
    cur = box(0, 0, N, N)
    for k, f in enumerate(folds):
        mv = cur.intersection(half_plane(f, f["move"]))
        st = cur.intersection(half_plane(f, -f["move"]))
        if mv.area < 3 or st.area < 3:
            return False
        landed = unary_union([Polygon([reflect(f, x, y) for x, y in g.exterior.coords])
                              for g in getattr(mv, "geoms", [mv]) if g.geom_type == "Polygon"])
        if landed.intersection(st).area < 0.5 * mv.area:
            return False
        cur = shape(folds[:k + 1])
        if not sheet.contains(cur):
            return False
    return True


def check(scene: dict) -> list:
    errs = []
    folds, punches = scene["folds"], scene["punches"]
    if not FOLDS[0] <= len(folds) <= FOLDS[1] or not valid(folds):
        errs.append("bad fold sequence")
    for p in punches:
        if not clear(folds, p):
            errs.append(f"punch {p} is off the paper or too near an edge")
    hs = holes(folds, punches)
    if len(set(hs)) != len(hs) or hs != [tuple(h) for h in scene["holes"]]:
        errs.append("holes repeat or don't match the stored list")
    if any(not (0 < x < N and 0 < y < N) or abs(x % 1 - 0.5) > 1e-6 or abs(y % 1 - 0.5) > 1e-6 for x, y in hs):
        errs.append("a hole is not at a cell centre")
    return errs


def _fold(rng: random.Random, folds: list):
    for _ in range(60):
        axis = rng.choice(["d1", "d2"]) if rng.random() < DIAG_P else rng.choice(["v", "h"])
        c = rng.randint(1, N - 1) if axis in ("v", "h") else (rng.randint(-5, 5) if axis == "d1"
                                                              else rng.randint(3, 13))
        f = {"axis": axis, "c": c, "move": rng.choice([1, -1])}
        if valid(folds + [f]):
            return f
    return None


def build(rng: random.Random) -> dict:
    n = rng.randint(*FOLDS)
    while True:
        folds = []
        for _ in range(n):
            f = _fold(rng, folds)
            if f is None:
                break
            folds.append(f)
        if len(folds) < n:
            continue
        k = 2 if rng.random() < 0.4 else 1
        cand = [(x + 0.5, y + 0.5) for x in range(N) for y in range(N) if clear(folds, (x + 0.5, y + 0.5))]
        if len(cand) < k:
            continue
        thick = [p for p in cand if layers(folds, p) >= 2]
        punches = []
        for _ in range(k):
            pool = [p for p in (thick if thick and rng.random() < 0.8 else cand)
                    if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) >= 2 for q in punches)]
            if not pool:
                break
            punches.append(rng.choice(pool))
        if len(punches) < k:
            continue
        hs = holes(folds, punches)
        if len(set(hs)) != len(hs) or len(hs) > MAX_HOLES:
            continue
        return {"folds": folds, "punches": [list(p) for p in punches], "holes": [list(h) for h in hs]}
