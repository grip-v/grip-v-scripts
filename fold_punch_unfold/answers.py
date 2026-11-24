"""Independent answers for fold_punch_unfold. No polygons: a point is
traced back through the folds one at a time. Going back over fold k, a
point q of the folded paper came from q itself if q lies on the side that
stayed put and that spot was paper before the fold, and from the mirror
image of q if the mirror image lies on the side that moved and was paper
before the fold. "Was paper before fold k" is the same test applied
recursively, down to the plain 8 x 8 sheet."""
from __future__ import annotations

N = 8
COLS = "ABCDEFGH"


def _side(f, x, y):
    c = f["c"]
    return {"v": x - c, "h": y - c, "d1": x - y - c, "d2": x + y - c}[f["axis"]]


def _mirror(f, x, y):
    c = f["c"]
    if f["axis"] == "v":
        return 2 * c - x, y
    if f["axis"] == "h":
        return x, 2 * c - y
    if f["axis"] == "d1":
        return y + c, x - c
    return c - y, c - x


def _origins(folds, x, y):
    """Points of the open sheet that end up at (x, y) after all `folds`."""
    if not folds:
        return [(x, y)] if 0 < x < N and 0 < y < N else []
    f, before = folds[-1], folds[:-1]
    out = []
    if _side(f, x, y) * f["move"] < 0:
        out += _origins(before, x, y)
    mx, my = _mirror(f, x, y)
    if _side(f, mx, my) * f["move"] > 0:
        out += _origins(before, mx, my)
    return out


def _holes(folds, punches):
    return sorted((round(x, 6), round(y, 6)) for px, py in punches for x, y in _origins(folds, px, py))


def _cell(x, y):
    return f"{COLS[int(x)]}{int(y) + 1}"


def answer(key: str, p: dict, scene: dict):
    folds, punches = scene["folds"], scene["punches"]
    hs = _holes(folds, punches)
    cells = [_cell(x, y) for x, y in hs]
    if key == "n_folds":
        return len(folds)
    if key == "fold_way":
        f = folds[p["k"] - 1]
        return {("v", 1): "right to left", ("v", -1): "left to right", ("h", 1): "bottom to top",
                ("h", -1): "top to bottom"}[(f["axis"], f["move"])]
    if key == "has_hole":
        return next(c for c in p["cells"] if c in cells)
    if key == "more_layers":
        a, b = (len(_origins(folds, *q)) for q in punches)
        return "the first punch" if a > b else ("the second punch" if b > a else "the same number")
    if key == "n_layers":
        return len(_origins(folds, *punches[p["punch"]]))
    if key == "n_holes":
        return len(hs)
    if key == "hole_set":
        return ", ".join(sorted(cells, key=lambda c: (int(c[1:]), c[0])))
    if key == "row_count":
        return sum(1 for c in cells if int(c[1:]) == p["row"])
    if key == "col_count":
        return sum(1 for c in cells if c[0] == p["col"])
    if key == "cf_flip":
        alt = [dict(f) for f in folds]
        alt[p["k"] - 1]["move"] *= -1
        return len(_holes(alt, punches))
    if key == "cf_skip":
        return len(_holes([f for i, f in enumerate(folds, 1) if i != p["k"]], punches))
    if key == "cf_punch_at":
        return len(_holes(folds, [tuple(p["at"])]))
    if key == "cf_skip_has":
        alt = [_cell(x, y) for x, y in _holes([f for i, f in enumerate(folds, 1) if i != p["k"]], punches)]
        return next(c for c in p["cells"] if c in alt)
    return None
