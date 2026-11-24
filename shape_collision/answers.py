"""Independent answers for shape_collision.

Never reads the generator's event list. Everything comes from what the
renderer draws (`render.track`: each shape's centre and pattern on every
frame, None once it has fallen in) plus the drawn geometry (box, radii,
posts, holes):

  * per shape, the frame-to-frame displacement is constant between events;
    where it changes, the pre-change motion is extended from the last frame
    before the change and the cause is the wall / post / other shape (that
    changes motion in the same frame step) it touches inside that step; the
    contact time comes from that extension. A shape that stops being drawn
    fell into the hole its extended path enters;
  * patterns are replayed from the frame-0 patterns with the scenario's rule
    (swap, or tag passing) over the decoded collisions;
  * counterfactual worlds (a shape removed, posts taken out, holes covered)
    are re-simulated with a fixed 2 ms time step from the frame-0 positions
    and the first frame step's velocities, rewinding each contact to the
    moment of touch inside its step (not the generator's closed-form solver).
"""
from __future__ import annotations
import functools
import json
import math

import render

FPS = 24
DT = 0.002
TOL = 1e-6


def _lbl(scene, i):
    b = scene["bodies"][i]
    return f"the {b['color']} {b['shape']}"


def _pair(scene, i, j):
    a, b = sorted([_lbl(scene, i), _lbl(scene, j)])
    return f"{a} and {b}"


def _clock(k):
    return f"{k // 10}.{k % 10} s"


def _reading(t):
    """Clock tenths on the first frame that shows an event at time t."""
    f = math.ceil(t * FPS - 1e-7)
    return (10 * f) // FPS


def _solve_contact(px, py, ux, uy, rs):
    """Smallest s with |p + u s| = rs (p relative position, u relative velocity), or None."""
    a = ux * ux + uy * uy
    b = px * ux + py * uy
    c = px * px + py * py - rs * rs
    if a < 1e-12:
        return None
    disc = b * b - a * c
    if disc < 0:
        return None
    return (-b - math.sqrt(disc)) / a


# ---------------------------------------------------------------- decoding the drawn track

@functools.lru_cache(maxsize=16)
def _decode(key: str):
    scene = json.loads(key)
    tr = render.track(scene)
    nf, n = len(tr), len(scene["bodies"])
    r = [b["r"] for b in scene["bodies"]]
    L, T, R, B = scene["box"]
    groups = {}          # step k0 -> [(i, pre position, pre velocity)]
    gone = []
    speed0, start_pat, end_pat = [], [], []
    for i in range(n):
        last = max(f for f in range(nf) if tr[f][i] is not None)
        P = [tr[f][i][:2] for f in range(last + 1)]
        d = [(P[f + 1][0] - P[f][0], P[f + 1][1] - P[f][1]) for f in range(last)]
        speed0.append(math.hypot(*d[0]) * FPS)
        start_pat.append(tr[0][i][2])
        end_pat.append(tr[-1][i][2] if tr[-1][i] else None)
        k = 1
        while k < len(d):
            if abs(d[k][0] - d[k - 1][0]) > TOL or abs(d[k][1] - d[k - 1][1]) > TOL:
                k0 = k
                while k + 1 < len(d) and (abs(d[k + 1][0] - d[k][0]) > TOL or abs(d[k + 1][1] - d[k][1]) > TOL):
                    k += 1
                groups.setdefault(k0, []).append((i, P[k0], (d[k0 - 1][0] * FPS, d[k0 - 1][1] * FPS)))
            k += 1
        if last < nf - 1:
            gone.append((i, last, P[last], (d[last - 1][0] * FPS, d[last - 1][1] * FPS)))
    coll, walls, posts, caps, bad = [], [], [], [], []
    win = 1 / FPS + 1e-6
    for k0, gs in sorted(groups.items()):
        used = set()
        for a in range(len(gs)):
            for b in range(a + 1, len(gs)):
                i, pi, ui = gs[a]
                j, pj, uj = gs[b]
                s = _solve_contact(pj[0] - pi[0], pj[1] - pi[1], uj[0] - ui[0], uj[1] - ui[1], r[i] + r[j])
                if s is not None and -1e-6 <= s <= win and a not in used and b not in used:
                    coll.append((k0 / FPS + s, min(i, j), max(i, j)))
                    used |= {a, b}
        for a, (i, p, u) in enumerate(gs):
            if a in used:
                continue
            hit = None
            for ax, lo, hi in ((0, L + r[i], R - r[i]), (1, T + r[i], B - r[i])):
                for bound in (lo, hi):
                    if abs(u[ax]) > 1e-9:
                        s = (bound - p[ax]) / u[ax]
                        if -1e-6 <= s <= win:
                            hit = ("w", s, None)
            for kk, po in enumerate(scene["posts"]):
                s = _solve_contact(p[0] - po["x"], p[1] - po["y"], u[0], u[1], r[i] + po["r"])
                if s is not None and -1e-6 <= s <= win:
                    hit = ("p", s, kk)
            if hit is None:
                bad.append((k0, i))
            elif hit[0] == "w":
                walls.append((k0 / FPS + hit[1], i))
            else:
                posts.append((k0 / FPS + hit[1], i, hit[2]))
    for i, last, p, u in gone:
        for kk, h in enumerate(scene["holes"]):
            s = _solve_contact(p[0] - h["x"], p[1] - h["y"], u[0], u[1], h["r"])
            if s is not None and -1e-6 <= s <= win:
                caps.append((last / FPS + s, i, kk))
                break
        else:
            bad.append((last, i))
    first_pos = [tr[0][i][:2] for i in range(n)]
    vel0 = [((tr[1][i][0] - tr[0][i][0]) * FPS, (tr[1][i][1] - tr[0][i][1]) * FPS) for i in range(n)]
    return {"coll": sorted(coll), "walls": sorted(walls), "posts": sorted(posts), "caps": sorted(caps),
            "bad": bad, "speed0": speed0, "start": start_pat, "end": end_pat, "n_frames": nf,
            "p0": first_pos, "v0": vel0}


def _D(scene):
    return _decode(json.dumps(scene, sort_keys=True))


# ---------------------------------------------------------------- rule replay

def _apply(scene, st, i, j):
    if scene["rule"] == "swap":
        st[i], st[j] = st[j], st[i]
    elif (st[i] == scene["tag"]) != (st[j] == scene["tag"]):
        st[i] = st[j] = scene["tag"]


def _hist(scene, coll, start, skip=None, upto=None):
    st = list(start)
    out = [st[:]]
    for k, (_, i, j) in enumerate(coll):
        if upto is not None and k >= upto:
            break
        if k != skip:
            _apply(scene, st, i, j)
        out.append(st[:])
    return out


# ---------------------------------------------------------------- fixed-step re-simulation

@functools.lru_cache(maxsize=64)
def _resim(key: str):
    w = json.loads(key)
    n = len(w["p"])
    p = [list(q) for q in w["p"]]
    v = [list(q) for q in w["v"]]
    r = w["r"]
    m = [x * x for x in r]
    alive = [True] * n
    L, T, R, B = w["box"]
    coll, caps = [], []
    for s in range(int(round(w["T"] / DT))):
        t = s * DT
        old = [q[:] for q in p]
        for i in range(n):
            if not alive[i]:
                continue
            for ax, lo, hi in ((0, L + r[i], R - r[i]), (1, T + r[i], B - r[i])):
                q = p[i][ax] + v[i][ax] * DT
                if q < lo or q > hi:
                    edge = lo if q < lo else hi
                    q = 2 * edge - q
                    v[i][ax] = -v[i][ax]
                p[i][ax] = q
        for i in range(n):
            if not alive[i]:
                continue
            for cx, cy, cr in w["posts"]:
                if math.hypot(p[i][0] - cx, p[i][1] - cy) >= r[i] + cr:
                    continue
                tau = _solve_contact(old[i][0] - cx, old[i][1] - cy, v[i][0], v[i][1], r[i] + cr)
                if tau is None:
                    continue
                tau = min(DT, max(0.0, tau))
                ci = (old[i][0] + v[i][0] * tau, old[i][1] + v[i][1] * tau)
                d = math.hypot(ci[0] - cx, ci[1] - cy)
                nx, ny = (ci[0] - cx) / d, (ci[1] - cy) / d
                vn = v[i][0] * nx + v[i][1] * ny
                if vn >= 0:
                    continue
                v[i] = [v[i][0] - 2 * vn * nx, v[i][1] - 2 * vn * ny]
                p[i] = [ci[0] + v[i][0] * (DT - tau), ci[1] + v[i][1] * (DT - tau)]
            for kk, (cx, cy, cr) in enumerate(w["holes"]):
                if math.hypot(p[i][0] - cx, p[i][1] - cy) < cr:
                    tau = _solve_contact(old[i][0] - cx, old[i][1] - cy, v[i][0], v[i][1], cr)
                    caps.append((t + min(DT, max(0.0, tau or 0.0)), i, kk))
                    alive[i] = False
                    break
        for i in range(n):
            for j in range(i + 1, n):
                if not (alive[i] and alive[j]):
                    continue
                rs = r[i] + r[j]
                if math.dist(p[i], p[j]) >= rs:
                    continue
                d0 = (old[j][0] - old[i][0], old[j][1] - old[i][1])
                u = (v[j][0] - v[i][0], v[j][1] - v[i][1])
                tau = _solve_contact(d0[0], d0[1], u[0], u[1], rs)
                if tau is None or d0[0] * u[0] + d0[1] * u[1] >= 0:
                    continue
                tau = min(DT, max(0.0, tau))
                ci = (old[i][0] + v[i][0] * tau, old[i][1] + v[i][1] * tau)
                cj = (old[j][0] + v[j][0] * tau, old[j][1] + v[j][1] * tau)
                dist = math.dist(ci, cj)
                nx, ny = (cj[0] - ci[0]) / dist, (cj[1] - ci[1]) / dist
                close = (v[i][0] - v[j][0]) * nx + (v[i][1] - v[j][1]) * ny
                if close <= 0:
                    continue
                k = 2 * close / (m[i] + m[j])
                v[i] = [v[i][0] - k * m[j] * nx, v[i][1] - k * m[j] * ny]
                v[j] = [v[j][0] + k * m[i] * nx, v[j][1] + k * m[i] * ny]
                rest = DT - tau
                p[i] = [ci[0] + v[i][0] * rest, ci[1] + v[i][1] * rest]
                p[j] = [cj[0] + v[j][0] * rest, cj[1] + v[j][1] * rest]
                coll.append((t + tau, i, j))
    return sorted(coll), sorted(caps)


def _cf(scene, drop=None, no_posts=False, no_holes=False):
    """Collisions / captures (original ids, inside the clip) of a changed world."""
    D = _D(scene)
    keep = [i for i in range(len(scene["bodies"])) if i != drop]
    tl = (D["n_frames"] - 1) / FPS
    w = {"p": [D["p0"][i] for i in keep], "v": [D["v0"][i] for i in keep],
         "r": [scene["bodies"][i]["r"] for i in keep], "box": scene["box"],
         "posts": [] if no_posts else [[p["x"], p["y"], p["r"]] for p in scene["posts"]],
         "holes": [] if no_holes else [[h["x"], h["y"], h["r"]] for h in scene["holes"]],
         "T": round(tl + 0.05, 4)}
    coll, caps = _resim(json.dumps(w, sort_keys=True))
    coll = [(t, min(keep[i], keep[j]), max(keep[i], keep[j])) for t, i, j in coll if t <= tl]
    caps = [(t, keep[i], k) for t, i, k in caps if t <= tl]
    return coll, caps


# ---------------------------------------------------------------- answers

def _only(vals):
    """The single winner of a {id: score} max (None on a tie)."""
    best = max(vals.values())
    win = [k for k, v in vals.items() if v == best]
    return win[0] if len(win) == 1 else None


def answer(key: str, p: dict, scene: dict):
    D = _D(scene)
    if D["bad"]:
        return None
    n = len(scene["bodies"])
    coll, caps = D["coll"], D["caps"]
    lb = lambda i: _lbl(scene, i)  # noqa: E731
    hits = {x: [t for t, i, j in coll if x in (i, j)] for x in range(n)}
    hist = _hist(scene, coll, D["start"])
    final = hist[-1]
    tag = scene.get("tag")
    fall = {i: t for t, i, _ in caps}
    walls = {x: [t for t, i in D["walls"] if i == x] for x in range(n)}
    posts = {x: [t for t, i, _ in D["posts"] if i == x] for x in range(n)}

    def upto_frame(k):
        lo = -(-k * FPS // 10)
        return [c for c in coll if math.ceil(c[0] * FPS - 1e-7) <= lo]

    def tagged_when():
        out = {}
        for k, (t, i, j) in enumerate(coll):
            for x in (i, j):
                if hist[k][x] != tag and hist[k + 1][x] == tag:
                    out[x] = (t, i + j - x)
        return out

    if key == "n_shapes":
        return sum(1 for it in render.track(scene)[0] if it is not None)
    if key == "shape_of":
        return scene["bodies"][p["x"]]["shape"]
    if key == "color_of":
        return scene["bodies"][p["x"]]["color"]
    if key == "n_collisions":
        return len(coll)
    if key == "first_pair":
        return _pair(scene, coll[0][1], coll[0][2])
    if key == "start_pattern":
        return D["start"][p["x"]]
    if key == "end_pattern":
        return D["end"][p["x"]]
    if key == "n_fixtures":
        return len(scene["posts"]) or len(scene["holes"]) or len(scene["lanes"])
    if key == "tag_start":
        return lb(D["start"].index(tag))
    if key == "n_tagged_end":
        return D["end"].count(tag)
    if key == "n_left_end":
        return sum(1 for x in D["end"] if x is not None)
    if key == "partner_of4":
        x = p["x"]
        met = [y for y in p["options"] if any({i, j} == {x, y} for _, i, j in coll)]
        return lb(met[0]) if len(met) == 1 else None
    if key == "first_partner":
        x = p["x"]
        _, i, j = next(c for c in coll if x in c[1:])
        return lb(i + j - x)
    if key == "more_hits4":
        w = _only({k: len(hits[k]) for k in p["options"]})
        return None if w is None else lb(w)
    if key == "earlier_pair":
        firsts = {tuple(pr): min([t for t, i, j in coll if (i, j) == tuple(pr)] or [99]) for pr in p["options"]}
        w = _only({k: -v for k, v in firsts.items()})
        return None if w is None else _pair(scene, *w)
    if key == "first_wall4":
        w = _only({k: -(walls[k][0] if walls[k] else 99) for k in p["options"]})
        return None if w is None else lb(w)
    if key == "fastest4":
        w = _only({k: D["speed0"][k] for k in p["options"]})
        return None if w is None else lb(w)
    if key == "tagger_of":
        return lb(tagged_when()[p["x"]][1])
    if key == "tagged_first4":
        tw = tagged_when()
        w = _only({k: -(tw[k][0] if k in tw else 99) for k in p["options"]})
        return None if w is None else lb(w)
    if key == "fall_first4":
        w = _only({k: -fall.get(k, 99) for k in p["options"]})
        return None if w is None else lb(w)
    if key == "post_first4":
        w = _only({k: -(posts[k][0] if posts[k] else 99) for k in p["options"]})
        return None if w is None else lb(w)
    if key == "hits_of":
        return len(hits[p["x"]])
    if key == "walls_of":
        return len(walls[p["x"]])
    if key == "posts_of":
        return len(posts[p["x"]])
    if key == "coll_time":
        t = next(t for t, i, j in coll if {i, j} == {p["x"], p["y"]})
        return _clock(_reading(t))
    if key == "fall_time":
        return _clock(_reading(fall[p["x"]]))
    if key == "span":
        return _clock(_reading(coll[-1][0]) - _reading(coll[0][0]))
    if key == "n_coll_by":
        return len(upto_frame(p["t"]))
    if key == "pattern_at":
        return _hist(scene, upto_frame(p["t"]), D["start"])[-1][p["x"]]
    if key == "n_tagged_at":
        return _hist(scene, upto_frame(p["t"]), D["start"])[-1].count(tag)
    if key == "n_left_at":
        lo = -(-p["t"] * FPS // 10)
        return n - sum(1 for t in fall.values() if math.ceil(t * FPS - 1e-7) <= lo)
    if key == "most_hits":
        w = _only({k: len(hits[k]) for k in range(n)})
        return None if w is None else lb(w)
    if key == "kth_pair":
        _, i, j = coll[p["k"] - 1]
        return _pair(scene, i, j)
    if key == "n_partners":
        return len({i + j - p["x"] for _, i, j in coll if p["x"] in (i, j)})
    if key == "n_worn":
        return len({h[p["x"]] for h in hist})
    if key == "visits":
        return len({h.index(p["p"]) for h in hist})
    if key == "who_has":
        return lb(final.index(D["start"][p["x"]]))
    if key == "n_kept":
        return sum(1 for k in range(n) if final[k] == D["start"][k] and D["end"][k] is not None)
    if key == "n_passed_by":
        return sum(1 for x, (t, y) in tagged_when().items() if y == p["x"])
    if key == "kth_tagged":
        order = sorted((t, x) for x, (t, _) in tagged_when().items())
        return lb(order[p["k"] - 1][1])
    if key == "last_fall":
        return lb(max(fall, key=fall.get))
    if key == "most_posts":
        w = _only({k: len(posts[k]) for k in range(n)})
        return None if w is None else lb(w)
    if key == "cf_skip_pattern":
        return _hist(scene, coll, D["start"], skip=p["k"] - 1)[-1][p["z"]]
    if key == "cf_skip_count":
        alt = _hist(scene, coll, D["start"], skip=p["k"] - 1)[-1]
        if scene["rule"] == "swap":
            return sum(1 for z in range(n) if alt[z] == D["start"][z] and D["end"][z] is not None)
        return alt.count(tag)
    if key == "cf_stop":
        return _hist(scene, coll, D["start"], upto=len(coll) - 1)[-1][p["z"]]
    if key == "cf_zero":
        st = ["solid"] * n
        st[p["y"]] = tag
        return _hist(scene, coll, st)[-1].count(tag)
    if key == "cf_remove_n":
        return len(_cf(scene, drop=p["x"])[0])
    if key == "cf_remove_first":
        cc = _cf(scene, drop=p["x"])[0]
        return _pair(scene, cc[0][1], cc[0][2])
    if key == "cf_no_fixtures":
        if scene["posts"]:
            return len(_cf(scene, no_posts=True)[0])
        return len(_cf(scene, no_holes=True)[0])
    if key == "cf_remove_falls":
        return len(_cf(scene, drop=p["x"])[1])
    return None
