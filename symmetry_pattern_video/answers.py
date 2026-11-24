"""Independent answers for symmetry_pattern_video.

Never reads the timeline, the orbit list or the break record. Everything is
read off what the renderer draws (`render.track`: every shape's outline in
screen pixels, how much of it is drawn, the tip dot, the fill fade, its
style; the disc outline and notch in the turns scenario) plus the centre
point the picture is built around:

  * symmetry is measured on the drawn outlines with a separate matcher: a
    transformed outline matches another when every corner of each lies on
    the other's boundary (within 1.5 px) and the styles agree;
  * appearances, traces, glides and settles are first / last frames of the
    drawn track; turning directions come from the tip dot's path;
  * the odd shape is the one shape that no symmetry of the rest maps onto a
    like shape (or, for tracing, the one turning the other way);
  * disc moves come from fitting a 2x2 map to all outline corners on every
    frame (turn angle summed frame by frame, flips from a negative
    determinant); counterfactual moves and timings are replayed from those.
"""
from __future__ import annotations
import functools
import json
import math

import render

FPS = 24
TOL = 1.5
ORD = ["first", "second", "third", "fourth", "fifth"]
WORDS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
         "twelfth"]
PLURAL = {"triangle": "triangles", "square": "squares", "diamond": "diamonds", "pentagon": "pentagons",
          "hexagon": "hexagons", "star": "stars", "circle": "circles", "kite": "kites", "chevron": "chevrons",
          "bolt": "lightning bolts"}
OCT = ("right", "top-right", "top", "top-left", "left", "bottom-left", "bottom", "bottom-right")
BREAK_NAMES = {None: "every shape matches its partners", "fill": "one shape has a different fill",
               "color": "one shape has a different colour", "shape": "one shape has a different outline",
               "size": "one shape is a different size", "rotation": "one shape is turned differently",
               "position": "one shape is out of place", "direction": "one shape is traced the other way round",
               "incomplete": "one shape's outline is never finished"}


def _clock(f):
    k = (10 * f) // FPS
    return f"{k // 10}.{k % 10} s"


def _first_frame(k):
    return -(-k * FPS // 10)


def _octant(x, y):
    a = math.degrees(math.atan2(y, x)) % 360.0
    return OCT[int(((a + 22.5) % 360.0) // 45)]


def gname(g):
    return f"the {g[0]} {PLURAL[g[1]]}"


# ---------------------------------------------------------------- polygons (y up, about the centre)

def _poly(scene, item):
    cx, cy = scene["center"]
    return [(x - cx, cy - y) for x, y in item["pts"]]


def _seg_dist(p, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    L2 = vx * vx + vy * vy
    u = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L2))
    return math.hypot(p[0] - a[0] - u * vx, p[1] - a[1] - u * vy)


def _on(p, poly):
    return min(_seg_dist(p, poly[k], poly[(k + 1) % len(poly)]) for k in range(len(poly))) <= TOL


def same_outline(a, b):
    ca, cb = _cent(a), _cent(b)
    if math.dist(ca, cb) > 3 * TOL:
        return False
    return all(_on(p, b) for p in a) and all(_on(p, a) for p in b)


def _cent(p):
    return (sum(x for x, _ in p) / len(p), sum(y for _, y in p) / len(p))


def _style(item):
    return (item["color"], item["fill"], item["dash"], round(item["frac"], 3), len(item["pts"]))


def _xf(M, p):
    return (M[0][0] * p[0] + M[0][1] * p[1], M[1][0] * p[0] + M[1][1] * p[1])


def _rot(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, -s], [s, c]]


def _refl(axis_deg):
    c, s = math.cos(math.radians(2 * axis_deg)), math.sin(math.radians(2 * axis_deg))
    return [[c, s], [s, -c]]


def maps_onto(shapes, M, skip=()):
    """Number of shapes (not in `skip`) that M carries onto a like shape."""
    n = 0
    for i, (st, P) in enumerate(shapes):
        if i in skip:
            continue
        Q = [_xf(M, p) for p in P]
        if any(st == st2 and same_outline(Q, P2) for j, (st2, P2) in enumerate(shapes) if j not in skip):
            n += 1
    return n


def is_sym(shapes, M):
    return maps_onto(shapes, M) == len(shapes)


def _axes(shapes):
    cands = set()
    for st, P in shapes:
        for st2, P2 in shapes:
            if st != st2:
                continue
            a, b = _cent(P), _cent(P2)
            aa, ab = math.degrees(math.atan2(a[1], a[0])), math.degrees(math.atan2(b[1], b[0]))
            cands.add(round(((aa + ab) / 2) % 180, 2))
            cands.add(round(((aa + ab) / 2 + 90) % 180, 2))
    return sorted(cands)


def measure(shapes):
    """(rotation order, mirror lines)."""
    if not shapes:
        return 1, 0
    order = next((m for m in range(12, 1, -1) if is_sym(shapes, _rot(360 / m))), 1)
    axes = []
    for ax in _axes(shapes):
        if any(min(abs(ax - o), 180 - abs(ax - o)) < 0.5 for o in axes):
            continue
        if is_sym(shapes, _refl(ax)):
            axes.append(ax)
    return order, len(axes)


def turn_label(order):
    return "no turn short of a full turn" if order == 1 else f"{round(360 / order)}°"


# ---------------------------------------------------------------- decoding the track

@functools.lru_cache(maxsize=8)
def _decode(key: str):
    scene = json.loads(key)
    tr = render.track(scene)
    nf, n = len(tr), len(scene["shapes"])
    sh = lambda f, i: tr[f]["shapes"][i]  # noqa: E731
    first = [next(f for f in range(nf) if sh(f, i) is not None) for i in range(n)]
    ref_f = 0 if "disc" in tr[0] else nf - 1
    ref = [(_style(sh(ref_f, i)), _poly(scene, sh(ref_f, i))) for i in range(n)]
    names = [(sh(nf - 1, i)["color"], sh(nf - 1, i)["type"]) for i in range(n)]
    D = {"nf": nf, "n": n, "first": first, "ref": ref, "names": names, "ref_f": ref_f}
    # tracing: first frame fully traced, turning direction from the tip path
    done, turn = [], []
    for i in range(n):
        done.append(next((f for f in range(nf) if sh(f, i) is not None and sh(f, i)["frac"] >= 1
                          and sh(f, i)["tip"] is None), None))
        tips = [sh(f, i)["tip"] for f in range(nf) if sh(f, i) is not None and sh(f, i)["tip"] is not None]
        # signed area of the loop the tip dot runs around (screen y points down,
        # so a positive shoelace sum is a clockwise turn on screen)
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(tips, tips[1:] + tips[:1]))
        turn.append("clockwise" if area > 0 else "counter_clockwise")
    D["done"], D["turn"] = done, turn
    # gliding: first frame that differs from frame 0, first frame of the final rest
    move0, rest = [], []
    for i in range(n):
        f_in = first[i]
        P0, PL = sh(f_in, i)["pts"], sh(nf - 1, i)["pts"]
        diff = lambda P, Q: max(math.dist(a, b) for a, b in zip(P, Q))  # noqa: E731
        move0.append(next((f for f in range(f_in, nf) if diff(sh(f, i)["pts"], P0) > 1e-6), None))
        r = nf - 1
        while r > f_in and diff(sh(r - 1, i)["pts"], PL) <= 1e-6:
            r -= 1
        rest.append(r)
    D["move0"], D["rest"] = move0, rest
    if "disc" in tr[0]:
        D["moves"] = _moves(scene, tr)
        cx, cy = scene["center"]
        D["notch"] = [(x - cx, cy - y) for x, y in (fr["notch"] for fr in tr)]
    return D


def _fit(P0, P):
    """Least-squares 2x2 map M with M p0 ~ p."""
    sxx = sum(a[0] * a[0] for a in P0)
    syy = sum(a[1] * a[1] for a in P0)
    sxy = sum(a[0] * a[1] for a in P0)
    det = sxx * syy - sxy * sxy
    M = []
    for k in range(2):
        bx = sum(a[0] * b[k] for a, b in zip(P0, P))
        by = sum(a[1] * b[k] for a, b in zip(P0, P))
        M.append([(bx * syy - by * sxy) / det, (by * sxx - bx * sxy) / det])
    return M


def _moves(scene, tr):
    cx, cy = scene["center"]
    pts = lambda f: [(x - cx, cy - y) for it in tr[f]["shapes"] for x, y in it["pts"]]  # noqa: E731
    P0 = pts(0)
    Ms = [_fit(P0, pts(f)) for f in range(len(tr))]
    changing = [f for f in range(1, len(tr)) if max(abs(Ms[f][a][b] - Ms[f - 1][a][b]) for a in range(2)
                                                     for b in range(2)) > 1e-7]
    runs = []
    for f in changing:
        if runs and f == runs[-1][1] + 1:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    out = []
    for f0, f1 in runs:
        A, B = Ms[f0 - 1], Ms[f1]
        detA = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        inv = [[A[1][1] / detA, -A[0][1] / detA], [-A[1][0] / detA, A[0][0] / detA]]
        step = [[sum(B[a][k] * inv[k][b] for k in range(2)) for b in range(2)] for a in range(2)]
        det = step[0][0] * step[1][1] - step[0][1] * step[1][0]
        if det < 0:
            axis = "vertical" if step[0][0] < 0 else "horizontal"
            out.append({"kind": "flip", "axis": axis, "f0": f0, "f1": f1, "M": step})
        else:
            ang = 0.0
            for f in range(f0, f1 + 1):
                Mp, Mc = Ms[f - 1], Ms[f]
                dp = Mp[0][0] * Mp[1][1] - Mp[0][1] * Mp[1][0]
                ip = [[Mp[1][1] / dp, -Mp[0][1] / dp], [-Mp[1][0] / dp, Mp[0][0] / dp]]
                inc = [[sum(Mc[a][k] * ip[k][b] for k in range(2)) for b in range(2)] for a in range(2)]
                ang += math.degrees(math.atan2(inc[1][0], inc[0][0]))
            out.append({"kind": "turn", "angle": round(ang), "f0": f0, "f1": f1, "M": step})
    return out


def _D(scene):
    return _decode(json.dumps(scene, sort_keys=True))


def groups(scene, D):
    """Distinct drawn (colour, outline) names that appear more than once."""
    names = D["names"]
    return sorted({nm for nm in names if names.count(nm) > 1})


def members(D, g):
    return [i for i, nm in enumerate(D["names"]) if nm == tuple(g)]


def odd_shape(scene, D):
    """Index of the one shape that breaks the symmetry, or None."""
    n = D["n"]
    turns = D["turn"]
    if any(sh is not None for sh in D["done"]) and scene["scenario"] == "waves":
        maj = max(set(turns), key=turns.count)
        minority = [i for i in range(n) if turns[i] != maj]
        if len(minority) == 1:
            return minority[0]
    ref = D["ref"]
    elems = [_rot(360 * j / m) for m in range(2, 13) for j in range(1, m)] + [_refl(ax) for ax in _axes(ref)]
    good = []
    for M in elems:
        fails = [i for i, (st, P) in enumerate(ref)
                 if not any(st == st2 and same_outline([_xf(M, p) for p in P], P2) for st2, P2 in ref)]
        if fails and len(fails) <= 2:
            good.append(set(fails))
    if not good:
        return None
    common = set.intersection(*good)
    return common.pop() if len(common) == 1 else None


# ---------------------------------------------------------------- answers

def answer(key: str, p: dict, scene: dict):
    D = _D(scene)
    n, nf = D["n"], D["nf"]
    ref = D["ref"]
    first, done, rest, move0 = D["first"], D["done"], D["rest"], D["move0"]
    G = groups(scene, D)

    def g_first(g):
        return min(first[i] for i in members(D, g))

    def g_last(g):
        return max(first[i] for i in members(D, g))

    if key == "n_shapes":
        return n
    if key == "n_type":
        return sum(1 for nm in D["names"] if nm[1] == p["type"])
    if key == "n_colors":
        return len({nm[0] for nm in D["names"]})
    if key == "n_waves":
        return len(set(first))
    if key == "first_group":
        if scene["scenario"] == "assemble":
            return gname(min(G, key=lambda g: min(move0[i] for i in members(D, g))))
        if scene["scenario"] == "waves":
            return gname(min(G, key=g_first))
        i0 = min(range(n), key=lambda i: first[i])
        return gname(D["names"][i0]) if tuple(D["names"][i0]) in G else None
    if key == "n_moves":
        ms = D["moves"]
        return len(ms) if p["what"] == "moves" else sum(1 for m in ms if m["kind"] == "flip")
    if key == "break_kind":
        return _break_kind(scene, D)
    if key == "group_wave":
        waves = sorted(set(first))
        return ORD[waves.index(g_first(p["g"]))]
    if key == "last_group":
        if scene["scenario"] == "assemble":
            return gname(max(G, key=lambda g: max(rest[i] for i in members(D, g))))
        return gname(max(G, key=g_last))
    if key == "first_done":
        if scene["scenario"] == "assemble":
            return gname(min(G, key=lambda g: max(rest[i] for i in members(D, g))))
        return gname(min(G, key=g_last))
    if key == "earlier_group":
        listed = [g for g in G if gname(g) in p["options"]]
        if scene["scenario"] == "assemble":
            return gname(min(listed, key=lambda g: min(move0[i] for i in members(D, g))))
        return gname(min(listed, key=g_first))
    if key == "kth_move":
        m = D["moves"][p["k"] - 1]
        if m["kind"] == "turn":
            return "a counter-clockwise turn" if m["angle"] > 0 else "a clockwise turn"
        return "a left-right flip" if m["axis"] == "vertical" else "an upside-down flip"
    if key == "odd_place":
        k = odd_shape(scene, D)
        return None if k is None else _octant(*_cent(_poly(scene, render.track(scene)[nf - 1]["shapes"][k])))
    if key == "min_turn":
        return turn_label(measure(ref)[0])
    if key == "mirrors":
        return measure(ref)[1]
    if key == "wave_time":
        return _clock(g_first(p["g"]))
    if key == "done_time":
        sc = scene["scenario"]
        if sc == "waves":
            return _clock(max(d for d in done if d is not None))
        if sc == "reveal":
            return _clock(max(first))
        if sc == "assemble":
            return _clock(max(rest))
        return _clock(D["moves"][-1]["f1"])
    if key == "count_at":
        f = _first_frame(p["t"])
        tr = render.track(scene)[f]["shapes"]
        sc = scene["scenario"]
        if sc == "reveal":
            return sum(1 for it in tr if it is not None)
        if sc == "waves":
            return sum(1 for it in tr if it is not None and it["frac"] >= 1)
        return sum(1 for g in G if max(rest[i] for i in members(D, g)) < f)
    if key == "total_turn":
        return f"{sum(abs(m['angle']) for m in D['moves'] if m['kind'] == 'turn')}°"
    if key == "notch_after":
        x, y = D["notch"][D["moves"][p["k"] - 1]["f1"]]
        return _octant(x, y)
    if key == "size_ratio":
        k = odd_shape(scene, D)
        if k is None:
            return None
        rad = lambda P: max(math.dist(_cent(P), q) for q in P)  # noqa: E731
        partners = [i for i in range(n) if i != k and D["names"][i] == D["names"][k]]
        r = rad(ref[k][1]) / rad(ref[partners[0]][1])
        return f"about {r:.1f} times"
    if key == "odd_wave":
        k = odd_shape(scene, D)
        return None if k is None else ORD[sorted(set(first)).index(first[k])]
    if key == "before_wave":
        f0 = g_first(p["g"])
        return sum(1 for d in done if d is not None and d < f0)
    if key == "dir_count":
        t = D["turn"]
        return max(t.count("clockwise"), t.count("counter_clockwise"))
    if key == "first_sym":
        order = sorted(range(n), key=lambda i: first[i])
        for m in range(2, n + 1):
            o, mr = measure([ref[i] for i in order[:m]])
            if (p["kind"] == "half" and o % 2 == 0) or (p["kind"] == "mirror" and mr >= 1):
                return m
        return None
    if key == "kth_settle":
        order = sorted(G, key=lambda g: max(rest[i] for i in members(D, g)))
        return gname(order[p["k"] - 1])
    if key == "moving_at":
        lo = _first_frame(p["t"])
        hi = -(-(p["t"] + 1) * FPS // 10) - 1
        return sum(1 for g in G if min(move0[i] for i in members(D, g)) <= lo and
                   max(rest[i] for i in members(D, g)) >= hi)
    if key in ("match_after", "n_match"):
        tr = render.track(scene)
        hits = []
        for k, m in enumerate(D["moves"]):
            now = [(_style(it), _poly(scene, it)) for it in tr[m["f1"]]["shapes"]]
            hits.append(all(any(st == st2 and same_outline(P, Q) for st2, Q in now) for st, P in ref))
        if key == "n_match":
            return sum(hits)
        return "never" if not any(hits) else f"after the {ORD[hits.index(True)]} move"
    if key == "odd_after":
        k = odd_shape(scene, D)
        if k is None:
            return None
        it = render.track(scene)[D["moves"][p["k"] - 1]["f1"]]["shapes"][k]
        return _octant(*_cent(_poly(scene, it)))
    if key == "kth_shape":
        order = sorted(range(n), key=lambda i: first[i])
        return gname(D["names"][order[p["k"] - 1]])
    if key == "on_screen_when":
        f = g_last(p["g"])
        return sum(1 for i in range(n) if first[i] <= f)
    if key == "after_wave":
        f = g_first(p["g"])
        return sum(1 for i in range(n) if first[i] > f)
    if key == "turn_until":
        return f"{sum(abs(m['angle']) for m in D['moves'][:p['k']] if m['kind'] == 'turn')}°"
    if key == "cf_rotate":
        names = {"a quarter turn (90°)": _rot(90), "a half turn (180°)": _rot(180), "a 60° turn": _rot(60),
                 "a 72° turn": _rot(72), "a 120° turn": _rot(120), "a 45° turn": _rot(45),
                 "a left-to-right flip": _refl(90), "an upside-down flip": _refl(0)}
        good = [o for o in p["options"] if o in names and is_sym(ref, names[o])]
        if "none of these" in p["options"]:
            return "none of these" if not good else None
        return good[0] if len(good) == 1 else None
    if key == "cf_remove_odd":
        k = odd_shape(scene, D)
        return None if k is None else measure([s for i, s in enumerate(ref) if i != k])[1]
    if key == "cf_more_wave":
        per = n // len(set(first))
        return n + per
    if key == "cf_slow":
        starts = sorted(set(first))
        trace = max(d for d in done if d is not None) - starts[-1] + 1
        last = starts[-1] + (len(starts) - 1) * trace + 2 * trace - 1
        return _clock(last)
    if key == "cf_drop_last":
        last = max(range(n), key=lambda i: first[i])
        o, mr = measure([s for i, s in enumerate(ref) if i != last])
        return mr if p["kind"] == "mirrors" else turn_label(o)
    if key == "cf_delay":
        settle = {tuple(g): max(rest[i] for i in members(D, g)) for g in G}
        settle[tuple(p["g"])] += FPS
        return gname(max(settle, key=settle.get))
    if key == "cf_skip_move":
        M = [[1.0, 0.0], [0.0, 1.0]]
        for j, m in enumerate(D["moves"]):
            if j != p["k"] - 1:
                M = [[sum(m["M"][a][c] * M[c][b] for c in range(2)) for b in range(2)] for a in range(2)]
        return _octant(*_xf(M, D["notch"][0]))
    return None


def _break_kind(scene, D):
    """What kind of break the drawn picture shows (None = no break)."""
    n = D["n"]
    t = D["turn"]
    if scene["scenario"] == "waves" and len(set(t)) == 2 and min(t.count(x) for x in set(t)) == 1:
        return BREAK_NAMES["direction"]
    ref = D["ref"]
    fr = render.track(scene)[D["ref_f"]]["shapes"]
    if any(it["frac"] < 1 for it in fr):
        return BREAK_NAMES["incomplete"]
    # a picture with no break has every shape matched by the group of its symmetries
    order, mirrors = measure(ref)
    names = D["names"]
    counts = {nm: names.count(nm) for nm in names}
    singles = [i for i in range(n) if counts[names[i]] == 1]
    if not singles:
        fills = {}
        for i, it in enumerate(fr):
            fills.setdefault(names[i], set()).add(it["fill"])
        if any(len(v) > 1 for v in fills.values()):
            return BREAK_NAMES["fill"]
        # same names, same fills: size, turn or place
        rad = lambda P: max(math.dist(_cent(P), q) for q in P)  # noqa: E731
        for nm in set(names):
            rs = sorted(rad(ref[i][1]) for i in range(n) if names[i] == nm)
            if rs[-1] - rs[0] > 2:
                return BREAK_NAMES["size"]
        for nm in set(names):
            ds = sorted(math.hypot(*_cent(ref[i][1])) for i in range(n) if names[i] == nm)
            r = max(rad(ref[i][1]) for i in range(n) if names[i] == nm)
            if ds[-1] - ds[0] > 0.2 * r + 3:          # a turn moves the corner-average far less
                return BREAK_NAMES["position"]
        if _is_whole(ref):
            return BREAK_NAMES[None]
        # equal distance from the centre but not symmetric: either a turned shape or a shifted one
        return BREAK_NAMES["rotation"] if _turned(ref, names) else BREAK_NAMES["position"]
    i = singles[0]
    nm = names[i]
    if any(o[0] == nm[0] for o in names if o != nm):
        return BREAK_NAMES["shape"]
    return BREAK_NAMES["color"]


def _is_whole(ref):
    """Some non-trivial symmetry maps every shape onto a like shape."""
    elems = [_rot(360 * j / m) for m in range(2, 13) for j in range(1, m)] + [_refl(ax) for ax in _axes(ref)]
    return any(is_sym(ref, M) for M in elems)


def _turned(ref, names):
    """A symmetry maps every shape's centre onto a like shape's centre (so the
    misfit is a turn, not a shift)."""
    cents = [(names[i], _cent(P), max(math.dist(_cent(P), q) for q in P)) for i, (_, P) in enumerate(ref)]
    elems = [_rot(360 * j / m) for m in range(2, 13) for j in range(1, m)] + [_refl(ax) for ax in _axes(ref)]
    for M in elems:
        # a turn moves the corner-average by at most ~0.1 r for these outlines;
        # an out-of-place shape is shifted by at least 0.6 r + 10 px
        if all(any(nm == nm2 and math.dist(_xf(M, c), c2) <= 0.2 * r + 3 for nm2, c2, _ in cents)
               for nm, c, r in cents):
            return True
    return False
