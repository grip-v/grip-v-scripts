"""Independent answers for minesweeper_reveal_trace. The generator uses
probability.py (component split + counting every layout). This module
never counts: it runs its own backtracking search that only asks "does
at least one mine layout fit these numbers and the mine total?", with an
optional tile forced to mine or safe, and derives every logical answer
(certain mine / certain safe / consistent) from those yes-no searches."""
from __future__ import annotations


def _nb(r, c, n):
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0) and 1 <= r + dr <= n and 1 <= c + dc <= n]


def _state(scene, steps=None, skip=None, change=None):
    seq = scene["reveals"] if steps is None else scene["reveals"][:steps]
    out = {}
    for k, rv in enumerate(seq):
        if k != skip:
            out[(rv["r"], rv["c"])] = change[1] if change and change[0] == k else rv["shown"]
    return out


def fits(scene, st, force=None) -> bool:
    """Is there a mine layout consistent with every shown number, the
    uncovered mines and the total? `force` = ((r, c), is_mine)."""
    n = scene["n"]
    covered = [(r, c) for r in range(1, n + 1) for c in range(1, n + 1) if (r, c) not in st]
    rem = scene["mines"] - sum(1 for v in st.values() if v == "mine")
    cons = [(set(x for x in _nb(r, c, n) if x not in st),
             v - sum(1 for x in _nb(r, c, n) if st.get(x) == "mine")) for (r, c), v in st.items() if v != "mine"]
    if any(v < 0 or len(g) < v for g, v in cons):
        return False
    front = sorted({x for g, _ in cons for x in g})
    fset = set(front)
    loose = [x for x in covered if x not in fset]
    # visit frontier tiles in breadth-first order so constraints close early
    order, seen = [], set()
    for s in front:
        if s in seen:
            continue
        queue = [s]
        seen.add(s)
        while queue:
            x = queue.pop(0)
            order.append(x)
            for y in _nb(*x, n):
                if y in fset and y not in seen:
                    seen.add(y)
                    queue.append(y)
    of = {x: [i for i, (g, _) in enumerate(cons) if x in g] for x in order}
    need = [v for _, v in cons]
    left = [len(g) for g, _ in cons]
    val = {}
    fx, fm = force if force else (None, None)

    def loose_ok(k):
        spare = rem - k
        if fx is not None and fx in loose:
            return (1 <= spare <= len(loose)) if fm else (0 <= spare <= len(loose) - 1)
        return 0 <= spare <= len(loose)

    def go(i, k):
        if k > rem:
            return False
        if i == len(order):
            return loose_ok(k)
        x = order[i]
        choices = (1, 0) if fx != x else ((1,) if fm else (0,))
        for m in choices:
            ok = True
            for ci in of[x]:
                need[ci] -= m
                left[ci] -= 1
                if need[ci] < 0 or need[ci] > left[ci]:
                    ok = False
            if ok and go(i + 1, k + m):
                for ci in of[x]:
                    need[ci] += m
                    left[ci] += 1
                return True
            for ci in of[x]:
                need[ci] += m
                left[ci] += 1
        return False
    return go(0, 0)


def status(scene, st, cell):
    can_mine = fits(scene, st, (cell, True))
    can_safe = fits(scene, st, (cell, False))
    return "mine" if not can_safe else "safe" if not can_mine else "unsure"


def _first_bad(scene, **kw):
    for s in range(1, len(scene["reveals"]) + 1):
        if not fits(scene, _state(scene, s, **kw)):
            return s
    return None


def _cell(p, k="cell"):
    return tuple(p[k])


def answer(key: str, p: dict, scene: dict):
    rv = scene["reveals"]
    final = _state(scene)
    if key == "n_revealed":
        return len(rv)
    if key == "value_at":
        v = final[_cell(p)]
        return "blank" if v == 0 else str(v)
    if key == "mine_uncovered":
        return "yes" if any(x["shown"] == "mine" for x in rv) else "no"
    if key == "earlier":
        a, b = tuple(p["a"]), tuple(p["b"])
        pos = {(x["r"], x["c"]): i for i, x in enumerate(rv)}
        return f"row {a[0]}, column {a[1]}" if pos[a] < pos[b] else f"row {b[0]}, column {b[1]}"
    if key == "larger":
        a, b = tuple(p["a"]), tuple(p["b"])
        w = a if final[a] > final[b] else b
        return f"row {w[0]}, column {w[1]}"
    if key == "count_value":
        return sum(1 for x in rv if x["shown"] == p["v"])
    if key == "reveal_step":
        return next(i for i, x in enumerate(rv) if (x["r"], x["c"]) == _cell(p)) + 1
    if key == "covered_neighbours":
        return sum(1 for y in _nb(*_cell(p), scene["n"]) if y not in final)
    if key in ("certain_mine", "certain_safe"):
        want = "mine" if key == "certain_mine" else "safe"
        hits = [tuple(c) for c in p["options"] if status(scene, final, tuple(c)) == want]
        return f"row {hits[0][0]}, column {hits[0][1]}" if len(hits) == 1 else None
    if key == "contradiction_step":
        return _first_bad(scene)
    if key == "cf_bump":
        k = next(i for i, x in enumerate(rv) if (x["r"], x["c"]) == _cell(p))
        return "yes" if fits(scene, _state(scene, change=(k, rv[k]["shown"] + 1))) else "no"
    if key == "cf_hide":
        k = next(i for i, x in enumerate(rv) if (x["r"], x["c"]) == _cell(p))
        st = _state(scene, skip=k)
        return "yes" if status(scene, st, tuple(p["target"])) == p["was"] else "no"
    if key == "cf_skip":
        return "yes" if _first_bad(scene, skip=p["step"]) is not None else "no"
    return None
