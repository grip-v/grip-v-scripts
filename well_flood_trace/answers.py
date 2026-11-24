"""Independent answers for well_flood_trace, used only by the validator.

No imports from sim.py / questions.py / render.py, and the stored track
(scene["track"], what the generator logged) is never read. Everything is
rebuilt from the scene's setup: grid, depth values, walls, entry pours, the
impossible-event spec and the clip length.

Method (structurally different from sim.py's event-stepped float engine):
exact rational arithmetic (fractions.Fraction). Sections are visited in a
depth-first topological order of the spill graph. Each section's inflow is a
piecewise-constant function of time, built as the sum of its tap pieces and
1/k of every upstream section's outflow; integrating it gives the exact time
the section is full, and its outflow is its inflow from that moment on, split
evenly over its lower neighbours (or soaking away when it has none). Being
exact, every per-frame fact (is the section wet / full at frame f, what the
clock reads) is decided without tolerances.

Frames show time f / 24 s; the clock at frame f reads floor(10 f / 24) tenths.

Answers that pick one of several listed sections return None (a validator
failure) unless exactly one listed option satisfies the question, and every
answer that depends on a margin (0.3 s between compared events, clock-safe
readings, 90% full-or-not margin) re-checks it and returns None when it fails.
"""
from __future__ import annotations
import math
from fractions import Fraction as Fr

FPS = 24
MARGIN = Fr(3, 10)
FULL_MARGIN = Fr(9, 10)
GUTTER = "gutter"
NONE_OPT = "none of them: every fill was physically correct"
SIDE = {"up": "from the section above it", "down": "from the section below it",
        "left": "from the section to its left", "right": "from the section to its right"}
KIND_TEXT = {"premature_fill": "a section started filling before any full higher neighbour could feed it",
             "uphill_flow": "water flowed into a section from a neighbour with a LOWER depth value",
             None: "nothing: every fill was physically correct"}
CF_TIMING = ("earlier than shown", "later than shown", "at the same time as shown",
             "not at all before the end of the video")


def _q(x) -> Fr:
    return Fr(repr(x)) if isinstance(x, float) else Fr(x)


def _t(frame_k) -> Fr:
    return Fr(2 * frame_k + 1, 2 * FPS)


# ------------------------------------------------------------------ geometry / naming
def _adj(R, C, i):
    r, c = i // C, i % C
    out = {}
    if r > 0:
        out["up"] = i - C
    if r < R - 1:
        out["down"] = i + C
    if c > 0:
        out["left"] = i - 1
    if c < C - 1:
        out["right"] = i + 1
    return out


def _name(scene, i):
    return "the section with depth value %d" % scene["depth"][i]


def _secs(tenths):
    return "%d.%d s" % (tenths // 10, tenths % 10)


# ------------------------------------------------------------------ step functions
# a step function is a sorted list of (t, rate): `rate` holds from t until the next t
def _sum(fs):
    ts = sorted({t for f in fs for t, _ in f})
    out = []
    for t in ts:
        v = sum((_at(f, t) for f in fs), Fr(0))
        if not out or out[-1][1] != v:
            out.append((t, v))
    return out


def _at(f, t):
    v = Fr(0)
    for a, r in f:
        if a <= t:
            v = r
        else:
            break
    return v


def _scale(f, k):
    return [(t, r * k) for t, r in f]


def _from(f, T):
    """f restricted to t >= T (zero before)."""
    if T is None:
        return []
    out = [(T, _at(f, T))]
    out += [(t, r) for t, r in f if t > T]
    return out


def _first_pos(f):
    for t, r in f:
        if r > 0:
            return t
    return None


def _fill_time(f, depth):
    vol = Fr(0)
    for k, (t, r) in enumerate(f):
        nxt = f[k + 1][0] if k + 1 < len(f) else None
        if r > 0:
            if nxt is None or vol + r * (nxt - t) >= depth:
                return t + (depth - vol) / r
            vol += r * (nxt - t)
    return None


def volume(f, depth, t):
    """Water in a tank of `depth` with inflow step function f, at time t (capped at depth)."""
    vol = Fr(0)
    for k, (a, r) in enumerate(f):
        if a >= t:
            break
        b = f[k + 1][0] if k + 1 < len(f) else t
        vol += r * (min(b, t) - a)
    return min(vol, Fr(depth))


def _intervals(f):
    """[(on, off|None)] where f > 0."""
    out, on = [], None
    for t, r in f:
        if r > 0 and on is None:
            on = t
        elif r == 0 and on is not None:
            out.append((on, t))
            on = None
    if on is not None:
        out.append((on, None))
    return out


# ------------------------------------------------------------------ the world
def world(scene, rate_mult=1, move=None, drop_tap=None, unchanged=False, violation=True, walls=None,
          blocked=()):
    R, C, depth = scene["R"], scene["C"], scene["depth"]
    n = R * C
    walls = scene["walls"] if walls is None else walls
    barred = {frozenset(w) for w in walls}
    v = scene["violation"] if violation else None
    blocked = set(blocked)
    spill = {}
    for i in range(n):
        tg = [j for j in _adj(R, C, i).values()
              if depth[j] < depth[i] and frozenset((i, j)) not in barred and j not in blocked]
        if v and v["kind"] == "uphill_flow":
            if i == v["src"] and v["dst"] not in blocked:
                tg.append(v["dst"])
            if i == v["dst"]:
                tg = [j for j in tg if j != v["src"]]
        if not tg:
            tg = [GUTTER]                 # no lower neighbour: once full, extra water soaks away
        spill[i] = sorted(tg, key=str)
    # depth-first topological order (reverse post-order); None on a cycle
    state, post = {}, []

    def visit(i):
        state[i] = 1
        for j in spill[i]:
            if j == GUTTER:
                continue
            if state.get(j) == 1:
                raise ValueError("cycle")
            if j not in state:
                visit(j)
        state[i] = 2
        post.append(i)
    try:
        for i in range(n):
            if i not in state:
                visit(i)
    except ValueError:
        return None
    order = post[::-1]
    pieces = {i: [] for i in range(n)}
    feeders = {i: [] for i in range(n)}          # (who, step function of what it sends)
    for k, tp in enumerate(scene["taps"]):
        if k == drop_tap:
            continue
        cell = move[1] if move and move[0] == k else tp["cell"]
        segs = tp["segs"][:1] if unchanged else tp["segs"]
        f = [(Fr(0), Fr(0))]
        for f0, f1, rate in segs:
            r = _q(rate) * _q(rate_mult)
            f.append((_t(f0), r))
            if f1 is not None and not unchanged:
                f.append((_t(f1), Fr(0)))
        f = _sum([f])
        if cell not in blocked:
            pieces[cell].append(f)
            feeders[cell].append(("tap", f))
    if v and v["kind"] == "premature_fill" and v["cell"] not in blocked:
        f = [(Fr(0), Fr(0)), (_t(v["frame"]), _q(v["rate"]))]
        pieces[v["cell"]].append(f)
        feeders[v["cell"]].append(("nowhere", f))
    out = {"start": {}, "full": {}, "edges": {}, "first": {}, "spill": spill, "inflow": {}}
    for i in order:
        inflow = _sum(pieces[i]) if pieces[i] else []
        out["inflow"][i] = inflow
        st = _first_pos(inflow)
        full = _fill_time(inflow, depth[i]) if st is not None else None
        out["start"][i], out["full"][i] = st, full
        if st is not None:
            out["first"][i] = sorted(str(w) for w, f in feeders[i] if _at(f, st) > 0)
        sent = _from(inflow, full)
        for j in spill[i]:
            share = _scale(sent, Fr(1, len(spill[i])))
            out["edges"][(i, j)] = _intervals(share)
            if j != GUTTER:
                pieces[j].append(share)
                feeders[j].append((i, share))
    return out


# ------------------------------------------------------------------ per-frame reading
class View:
    def __init__(self, scene, w):
        self.s, self.w = scene, w
        self.n = scene["R"] * scene["C"]
        self.nf = scene["n_frames"]
        self.end = Fr(self.nf - 1, FPS)

    def wet_at(self, i, t):
        st = self.w["start"][i]
        return st is not None and st <= t

    def full_at(self, i, t):
        f = self.w["full"][i]
        return f is not None and f <= t

    def frac(self, i, t):
        if self.full_at(i, t):
            return Fr(1)
        if not self.wet_at(i, t):
            return Fr(0)
        return volume(self.w["inflow"][i], self.s["depth"][i], t) / self.s["depth"][i]

    def full_clear(self, t, cells=None):
        cells = range(self.n) if cells is None else cells
        return all(self.full_at(i, t) or self.frac(i, t) <= FULL_MARGIN for i in cells)

    def wet_end(self):
        return [i for i in range(self.n) if self.wet_at(i, self.end)]

    def frame(self, t):
        return math.ceil(t * FPS)

    def clock(self, f):
        return f * 10 // FPS

    def safe_clock(self, t):
        if t is None or t > self.end - Fr(1, 2):
            return None
        f = self.frame(t)
        if f + 1 >= self.nf or self.clock(f) != self.clock(f + 1):
            return None
        return self.clock(f)

    def arrows_at(self, t):
        return [e for e, ivs in self.w["edges"].items() for a, b in ivs if a <= t and (b is None or t < b)]

    def edges_by_end(self):
        return {e: ivs for e, ivs in self.w["edges"].items() if ivs and ivs[0][0] <= self.end}

    def inflow_srcs(self, j):
        return {e[0] for e in self.edges_by_end() if e[1] == j}


def _usable(scene, w):
    """A what-if world counts only if no section starts filling within 0.3 s of the end."""
    if w is None:
        return None
    vw = View(scene, w)
    if any(t is not None and abs(t - vw.end) < MARGIN for t in w["start"].values()):
        return None
    return vw


def _one(items):
    return items[0] if len(items) == 1 else None


# ------------------------------------------------------------------ answers
def answer(key: str, p: dict, scene: dict):
    w = world(scene)
    if w is None:
        return None
    vw = View(scene, w)
    st, fl, end, n = w["start"], w["full"], vw.end, vw.n
    depth = scene["depth"]
    nm = lambda i: _name(scene, i)  # noqa: E731
    wet = vw.wet_end()
    taps = [tp["cell"] for tp in scene["taps"]]

    if key == "n_wet_end":
        return len(wet)
    if key == "n_full_end":
        return sum(1 for i in range(n) if vw.full_at(i, end)) if vw.full_clear(end) else None
    if key == "entry_depth":
        return str(depth[scene["taps"][p["tap"]]["cell"]])
    if key in ("reached_one_of_four", "dry_one_of_four"):
        want = key == "reached_one_of_four"
        h = _one([i for i in p["cells"] if vw.wet_at(i, end) == want])
        return None if h is None else nm(h)

    if key in ("earliest_of_four", "latest_of_four"):
        cs = p["cells"]
        if any(not vw.wet_at(i, end) for i in cs):
            return None
        sgn = 1 if key == "earliest_of_four" else -1
        ranked = sorted(cs, key=lambda i: sgn * st[i])
        if abs(st[ranked[1]] - st[ranked[0]]) < MARGIN:
            return None
        return nm(ranked[0])
    if key == "fed_from":
        x = p["cell"]
        first = w["first"].get(x, [])
        if len(first) != 1 or first[0] in ("tap", "nowhere"):
            return None
        side = [s for s, j in _adj(scene["R"], scene["C"], x).items() if j == int(first[0])]
        return SIDE[side[0]] if side else None
    if key == "full_when_start":
        tb = st[p["cell"]]
        if tb is None:
            return None
        hits = [i for i in p["cells"] if fl[i] is not None and fl[i] <= tb - MARGIN]
        unclear = [i for i in p["cells"] if fl[i] is not None and tb - MARGIN < fl[i] < tb + MARGIN]
        if unclear or not vw.full_clear(tb, p["cells"]):
            return None
        h = _one(hits)
        return None if h is None else nm(h)
    if key == "first_full_of_four":
        cs = p["cells"]
        done = sorted((fl[i], i) for i in cs if fl[i] is not None and fl[i] <= end)
        if not done or any(i in taps for i in cs):
            return None
        rest = [fl[i] for i in cs if i != done[0][1] and fl[i] is not None]
        if rest and min(rest) - done[0][0] < MARGIN:
            return None
        return nm(done[0][1])

    if key in ("start_clock", "full_clock"):
        c = vw.safe_clock((st if key == "start_clock" else fl)[p["cell"]])
        return None if c is None else _secs(c)
    if key in ("count_at_clock", "n_full_at_clock"):
        times = [(st if key == "count_at_clock" else fl)[i] for i in range(n)]
        frs = [f for f in range(vw.nf) if vw.clock(f) == p["tenths"]]
        if not frs:
            return None
        a, b = Fr(frs[0], FPS), Fr(frs[-1], FPS)
        if any(t is not None and a - MARGIN < t < b + MARGIN for t in times):
            return None
        if key == "n_full_at_clock" and not vw.full_clear(b):
            return None
        return sum(1 for t in times if t is not None and t <= a)
    if key == "fill_duration":
        x = p["cell"]
        a, b = vw.safe_clock(st[x]), vw.safe_clock(fl[x])
        return None if a is None or b is None else _secs(b - a)
    if key in ("start_gap", "full_gap"):
        times = st if key == "start_gap" else fl
        a, b = vw.safe_clock(times[p["a"]]), vw.safe_clock(times[p["b"]])
        if a is None or b is None or times[p["b"]] - times[p["a"]] < Fr(3, 5):
            return None
        return _secs(b - a)
    if key == "first_depth":
        order = sorted((st[i], i) for i in wet if i not in taps)
        if len(order) < 2 or order[1][0] - order[0][0] < MARGIN or order[0][0] - st[taps[0]] < MARGIN:
            return None
        return str(depth[order[0][1]])

    if key == "order_three":
        cs = sorted(p["cells"], key=lambda i: st[i])
        if any(st[i] is None or st[i] > end for i in cs):
            return None
        if min(st[cs[1]] - st[cs[0]], st[cs[2]] - st[cs[1]]) < MARGIN:
            return None
        return ", then ".join(str(depth[i]) for i in cs)
    if key == "kth_wet":
        hits = []
        for i in p["cells"]:
            if not vw.wet_at(i, end):
                continue
            if 1 + sum(1 for j in wet if st[j] < st[i]) == p["k"]:
                if any(j != i and abs(st[j] - st[i]) < MARGIN for j in wet):
                    return None
                hits.append(i)
        h = _one(hits)
        return None if h is None else nm(h)
    if key in ("impossible_which", "violation_kind"):
        bad, kinds = [], set()
        for i in wet:
            if "nowhere" in w["first"].get(i, []):
                bad.append(i)
                kinds.add("premature_fill")
            for src in vw.inflow_srcs(i):
                if src != GUTTER and depth[src] < depth[i]:
                    bad.append(i)
                    kinds.add("uphill_flow")
        bad = sorted(set(bad))
        if len(bad) > 1 or len(kinds) > 1:
            return None
        if key == "violation_kind":
            return KIND_TEXT[kinds.pop() if kinds else None]
        if not bad:
            return NONE_OPT
        return nm(bad[0]) if bad[0] in p["cells"] else None
    if key == "wet_before_full":
        tf = fl[p["cell"]]
        if tf is None or tf > end:
            return None
        if any(t is not None and tf - MARGIN <= t < tf for t in st.values()):
            return None
        return sum(1 for t in st.values() if t is not None and t < tf)
    if key == "wet_at_entry2":
        t2 = _t(scene["taps"][1]["segs"][0][0])
        e2 = scene["taps"][1]["cell"]
        for i, t in st.items():
            if t is not None and abs(t - t2) < MARGIN and not (i == e2 and t == t2):
                return None
        return sum(1 for t in st.values() if t is not None and t < t2)
    if key == "new_after_change":
        tc = _t(scene["taps"][0]["segs"][0][1])
        ts = [t for t in st.values() if t is not None and t <= end]
        if any(abs(t - tc) < MARGIN for t in ts):
            return None
        return sum(1 for t in ts if t > tc)
    if key == "walls_blocking":
        highs = [a if depth[a] > depth[b] else b for a, b in scene["walls"]]
        if not vw.full_clear(end, highs):
            return None
        return sum(1 for h in highs if vw.full_at(h, end))

    # counterfactuals: re-simulated under the changed setup
    if key == "cf_x2_which":
        cv = _usable(scene, world(scene, rate_mult=2))
        if cv is None or any(vw.wet_at(i, end) for i in p["cells"]):
            return None
        cst = cv.w["start"]
        if any(cst[i] is not None and end - MARGIN < cst[i] < end + MARGIN for i in p["cells"]):
            return None
        h = _one([i for i in p["cells"] if cst[i] is not None and cst[i] <= end - MARGIN])
        return None if h is None else nm(h)
    if key == "cf_correction":
        cv = _usable(scene, world(scene, violation=False))
        y = p["cell"]
        if cv is None or not vw.wet_at(y, end):
            return None
        shown, true = st[y], cv.w["start"][y]
        if true is None or true > end + MARGIN:
            return CF_TIMING[3]
        if true == shown:
            return CF_TIMING[2]
        if true <= shown - MARGIN:
            return CF_TIMING[0]
        if shown + MARGIN <= true <= end - MARGIN:
            return CF_TIMING[1]
        return None
    kw = {"cf_rate": lambda: {"rate_mult": p["mult"]},
          "cf_move_entry": lambda: {"move": (p["tap"], p["cell"])},
          "cf_block": lambda: {"blocked": (p["cell"],)},
          "cf_no_violation": lambda: {"violation": False},
          "cf_wall": lambda: {"walls": [x for x in scene["walls"] if sorted(x) != sorted(p["wall"])]},
          "cf_unchanged": lambda: {"unchanged": True},
          "cf_drop_entry": lambda: {"drop_tap": p["tap"]}}.get(key)
    if kw is None:
        return None
    cv = _usable(scene, world(scene, **kw()))
    return None if cv is None else len(cv.wet_end())
