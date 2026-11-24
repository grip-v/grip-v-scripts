"""Stacking collapse: blocks are stacked (or pulled out of a stack) one at a
time; after every change each resting surface is checked, and when the
combined centre of mass of everything above a surface lies outside the
contact span under it, everything above the LOWEST failing surface tips over
the contact edge as one rigid piece and falls out of the frame. The clip goes
on after a collapse (later blocks land on what is left).

Scenarios (`scene["scenario"]`):
  tower     one pedestal; blocks drop from above and land on top of the stack
  twin      two pedestals (left / right); each block drops onto one of them
  pull_out  a tower stands at the start; blocks are pulled out sideways one
            at a time and the part above drops onto the block below
  plank     a long plank lies on a narrow support; blocks drop onto the
            plank side by side; if the centre of mass of plank + blocks
            leaves the support, the plank tips over with everything on it

World units: blocks are integer widths (1-6 units) and 0.75 / 1 / 1.25 units
tall, mass = area; positions are multiples of 0.5. The background shows an
unnumbered grid of 1-unit squares (the only measuring aid); the only text is
the clock. Every centre of mass that the rule compares sits at least MARGIN
from every contact edge, and touching layers overlap by at least 1 unit.

Motion (px, whole frames): a dropped block starts at rest above the frame
and falls under gravity g px/frame^2 for F frames, landing exactly on frame
`start + F`. A pulled block slides out at constant speed from frame
`start + 1`; once it is clear, the part above drops (gravity) onto the block
below it. A collapse waits `pause` frames, turns about the contact edge
(angular speed w0 + acc * k deg/frame) for t1 frames, then flies off
(outward vx, falling with g) until it has left the frame.
"""
from __future__ import annotations
import functools
import json
import math
import random

W, H, FPS = 1280, 720, 24
MAX_FRAMES = 240
EDGE = 12                 # frames between the clip's start / end and any question-relevant event
TIMER_BOX = (14, 14, 196, 70)
TIMER_FONT = 36
CLOCK_CLEAR = 12          # px between the clock panel and anything else
MARGIN = 0.25             # units between a compared centre of mass and the contact edges
MIN_CONTACT = 1.0         # units of overlap between touching layers
GAP = 8                   # px between things that do not touch
PEN = 1.0                 # px of overlap tolerated between touching things (drawing rounds to 0.5 px)
SEP = 8                   # frames between consecutive landings / pulls
PLANK_H = 0.5
SCENARIOS = ("tower", "twin", "pull_out", "plank")

COLOURS = {"red": (218, 46, 46), "blue": (36, 96, 226), "green": (32, 160, 70), "orange": (245, 140, 18),
           "purple": (140, 64, 204), "teal": (0, 170, 170), "pink": (244, 100, 176), "yellow": (244, 208, 28)}
PATTERNS = ("plain", "striped", "dotted")
HEIGHTS = (0.75, 1.0, 1.25)

BACKGROUNDS = [  # (top, bottom, light?)
    ((238, 234, 224), (220, 214, 200), True), ((228, 234, 240), (204, 214, 226), True),
    ((232, 238, 228), (212, 220, 206), True), ((240, 232, 236), (222, 212, 220), True),
    ((26, 30, 42), (42, 48, 64), False), ((36, 36, 40), (58, 58, 64), False), ((22, 40, 46), (36, 60, 66), False),
]
PILLARS = {True: [(128, 130, 138), (150, 140, 124), (112, 124, 136)],
           False: [(150, 154, 166), (170, 160, 142), (132, 146, 160)]}
PLANKS = [(150, 104, 62), (126, 88, 56), (170, 126, 80)]


def tenths(f: int) -> int:
    return (10 * f) // FPS


def clock_text(k: int) -> str:
    return f"{k // 10}.{k % 10} s"


def frames_showing(k: int) -> tuple[int, int]:
    """First and last frame whose clock reads k tenths."""
    return -(-k * FPS // 10), -(-(k + 1) * FPS // 10) - 1


def name(b) -> str:
    return f"the {b['pattern']} {b['colour']} block"


def q2(v: float) -> float:
    """Round to the 0.5 px grid the renderer draws on."""
    return round(v * 2) / 2


def mass(b) -> float:
    return b["w"] * b["h"]


def base_name(scene, b) -> str:
    if scene["scenario"] == "plank":
        return "the support"
    if scene["scenario"] == "twin":
        return ("the left pedestal", "the right pedestal")[b]
    return "the pedestal"


def item_name(scene, it) -> str:
    """Block index -> its name; 'base0' / 'base1' -> the pedestal; 'plank' -> the plank."""
    if isinstance(it, int):
        return name(scene["blocks"][it])
    if it == "plank":
        return "the plank"
    return base_name(scene, int(it[4:]))


# ---------------------------------------------------------------- the rule (world units)

def _edges(x, w):
    return x - w / 2, x + w / 2


def contact(scene, b, stack, i):
    """Contact span under layer i of the stack on base b (the pedestal for i = 0)."""
    bl = scene["blocks"]
    j, x = stack[i]
    l, r = _edges(x, bl[j]["w"])
    if i == 0:
        B = scene["bases"][b]
        pl, pr = _edges(B["x"], B["w"])
    else:
        pj, px_ = stack[i - 1]
        pl, pr = _edges(px_, bl[pj]["w"])
    return max(l, pl), min(r, pr)


def stack_status(scene, b, stack):
    """(lowest failing surface or None, side, smallest margin, smallest contact) over surfaces up to the failure."""
    bl = scene["blocks"]
    worst, cmin = math.inf, math.inf
    for i in range(len(stack)):
        lo, hi = contact(scene, b, stack, i)
        cmin = min(cmin, hi - lo)
        m = sum(mass(bl[j]) for j, _ in stack[i:])
        c = sum(mass(bl[j]) * x for j, x in stack[i:]) / m
        worst = min(worst, abs(c - lo), abs(c - hi))
        if c < lo or c > hi:
            return i, ("left" if c < lo else "right"), worst, cmin
    return None, None, worst, cmin


def plank_status(scene, items):
    """(falls?, side, margin) for the plank with blocks `items` [(j, x)] on it."""
    P, B, bl = scene["plank"], scene["bases"][0], scene["blocks"]
    lo, hi = _edges(B["x"], B["w"])
    m = P["w"] * PLANK_H + sum(mass(bl[j]) for j, _ in items)
    c = (P["w"] * PLANK_H * P["x"] + sum(mass(bl[j]) * x for j, x in items)) / m
    side = "left" if c < lo else "right" if c > hi else None
    return side is not None, side, min(abs(c - lo), abs(c - hi))


def plank_fits(scene, items, j, x) -> bool:
    """Block j at x lies on the plank (0.5 in from its ends) and 0.5 clear of the other blocks on it."""
    P, bl = scene["plank"], scene["blocks"]
    l, r = _edges(x, bl[j]["w"])
    pl, pr = _edges(P["x"], P["w"])
    if l < pl + 0.5 - 1e-9 or r > pr - 0.5 + 1e-9:
        return False
    for k, xk in items:
        kl, kr = _edges(xk, bl[k]["w"])
        if l < kr + 0.5 - 1e-9 and kl < r + 0.5 - 1e-9:
            return False
    return True


def stack_y(scene, b, stack, i) -> float:
    """World y of the bottom of layer i."""
    return scene["bases"][b]["top"] + sum(scene["blocks"][j]["h"] for j, _ in stack[:i])


def play(scene, actions=None, bases=None) -> dict:
    """Replay the actions with the rule. Returns per-action records, the final
    state, and ok = every contact >= MIN_CONTACT and every compared centre of
    mass >= MARGIN from the edges (else the outcome is not clear-cut)."""
    sc = scene if bases is None else dict(scene, bases=bases)
    actions = scene["actions"] if actions is None else actions
    bl = sc["blocks"]
    stacks = [[] for _ in sc["bases"]]
    if sc["scenario"] == "pull_out":
        stacks[0] = [tuple(t) for t in sc["init"]]
    plank = [] if sc["plank"] else None
    plank_up = sc["plank"] is not None
    ok, worst = True, math.inf
    if sc["scenario"] == "pull_out":
        fi, _, m, c = stack_status(sc, 0, stacks[0])
        ok &= fi is None and c >= MIN_CONTACT - 1e-9
        worst = min(worst, m)
    if plank is not None:
        f, _, m = plank_status(sc, [])
        ok &= not f
        worst = min(worst, m)
    recs = []
    for a in actions:
        rec = {"kind": a["kind"], "block": a["block"], "fall": None}
        if a["kind"] == "drop" and plank is not None:
            j, x = a["block"], a["x"]
            if not plank_up or not plank_fits(sc, plank, j, x):
                ok = False
                recs.append(dict(rec, bad=True))
                break
            rec.update(on="plank", y=PLANK_H + sc["bases"][0]["top"], before=list(plank))
            plank.append((j, x))
            f, side, m = plank_status(sc, plank)
            worst = min(worst, m)
            if f:
                rec["fall"] = {"base": 0, "blocks": [k for k, _ in plank], "plank": True, "side": side,
                               "from": "base0", "surface": 0}
                plank_up = False
        elif a["kind"] == "drop":
            j, x, b = a["block"], a["x"], a["base"]
            st = stacks[b]
            if st:
                tl, tr = _edges(st[-1][1], bl[st[-1][0]]["w"])
            else:
                tl, tr = _edges(sc["bases"][b]["x"], sc["bases"][b]["w"])
            l, r = _edges(x, bl[j]["w"])
            if min(r, tr) - max(l, tl) < MIN_CONTACT - 1e-9:
                ok = False
                recs.append(dict(rec, bad=True))
                break
            rec.update(on=st[-1][0] if st else f"base{b}", y=stack_y(sc, b, st, len(st)), base=b,
                       before=list(st))
            st.append((j, x))
            fi, side, m, c = stack_status(sc, b, st)
            worst = min(worst, m)
            ok &= c >= MIN_CONTACT - 1e-9
            if fi is not None:
                rec["fall"] = {"base": b, "blocks": [k for k, _ in st[fi:]], "plank": False, "side": side,
                               "from": st[fi - 1][0] if fi else f"base{b}", "surface": fi}
                del st[fi:]
        else:                                                   # pull
            st = stacks[0]
            ks = [i for i, (k, _) in enumerate(st) if k == a["block"]]
            if not ks:
                ok = False
                recs.append(dict(rec, bad=True))
                break
            k = ks[0]
            rec.update(before=list(st), index=k, above=[q for q, _ in st[k + 1:]],
                       on=(st[k - 1][0] if k else "base0") if k + 1 < len(st) else None)
            del st[k]
            if st:
                fi, side, m, c = stack_status(sc, 0, st)
                worst = min(worst, m)
                ok &= c >= MIN_CONTACT - 1e-9
                if fi is not None:
                    rec["fall"] = {"base": 0, "blocks": [q for q, _ in st[fi:]], "plank": False, "side": side,
                                   "from": st[fi - 1][0] if fi else "base0", "surface": fi}
                    del st[fi:]
        recs.append(rec)
    ok &= worst >= MARGIN - 1e-9
    return {"recs": recs, "stacks": stacks, "plank": plank, "plank_up": plank_up, "ok": ok, "worst": worst}


def standing_end(res) -> int:
    return sum(len(s) for s in res["stacks"]) + (len(res["plank"]) if res["plank"] is not None and res["plank_up"]
                                                  else 0)


def n_fallen(res) -> int:
    return sum(len(r["fall"]["blocks"]) for r in res["recs"] if r.get("fall"))


# ---------------------------------------------------------------- kinematics (px, frames)

def fall_frames(land_py: float, g: float) -> int:
    """Frames of free fall from rest so that the block starts wholly above the frame."""
    return math.ceil(math.sqrt(2 * (land_py + 1) / g))


def rect(cx, cy, ang, wpx, hpx):
    c, s = math.cos(ang), math.sin(ang)
    out = []
    for dx, dy in ((-wpx / 2, -hpx / 2), (wpx / 2, -hpx / 2), (wpx / 2, hpx / 2), (-wpx / 2, hpx / 2)):
        out.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
    return out


def _outside(polys) -> bool:
    for p in polys:
        xs = [v[0] for v in p]
        ys = [v[1] for v in p]
        if max(xs) > 0 and min(xs) < W and max(ys) > 0 and min(ys) < H:
            return False
    return True


def _tip(pose0, pivot, side, k, tp, g):
    """Pose k frames after the last upright frame of a collapse."""
    t1 = tp["t1"]
    if k <= t1:
        th, dx, dy = tp["w0"] * k + 0.5 * tp["acc"] * k * k, 0.0, 0.0
    else:
        kk = k - t1
        th = tp["w0"] * t1 + 0.5 * tp["acc"] * t1 * t1 + (tp["w0"] + tp["acc"] * t1) * kk
        dx, dy = tp["vx"] * kk, tp["vy"] * max(0, kk - 1) + 0.5 * g * max(0, kk - 1) ** 2
    th = math.radians(th) * (1 if side == "right" else -1)
    dx *= 1 if side == "right" else -1
    cx, cy, a0 = pose0
    rx, ry = cx - pivot[0], cy - pivot[1]
    c, s = math.cos(th), math.sin(th)
    return q2(pivot[0] + rx * c - ry * s + dx), q2(pivot[1] + rx * s + ry * c + dy), a0 + th


@functools.lru_cache(maxsize=16)
def _timeline(key: str) -> dict:
    scene = json.loads(key)
    return timeline_of(scene, scene["n_frames"])


def timeline(scene) -> dict:
    return _timeline(json.dumps(scene, sort_keys=True))


def timeline_of(scene, n: int) -> dict:
    """Per-frame poses (cx, cy, angle) of every block and the plank (None when
    not drawn), the events, and the frames at which touching pairs come into
    contact. Uses play() for the outcomes."""
    S, ox, oy, g = scene["S"], scene["ox"], scene["oy"], scene["g"]
    bl = scene["blocks"]
    nb = len(bl)
    res = play(scene)
    pose = [[None] * nb for _ in range(n)]
    ppose = [None] * n
    contacts = []                          # (frame, item, item)
    events = []
    PX = lambda x: ox + S * x
    PY = lambda y: oy - S * y
    dims = [(b["w"] * S, b["h"] * S) for b in bl]

    def rest(j, x, y):
        return q2(PX(x)), q2(PY(y) - dims[j][1] / 2), 0.0

    def fill(j, f0, fn, f1=None):
        for f in range(max(0, f0), n if f1 is None else min(n, f1)):
            v = fn(f)
            if j == "plank":
                ppose[f] = v
            else:
                pose[f][j] = v

    if scene["plank"]:
        P = scene["plank"]
        pp = (q2(PX(P["x"])), q2(PY(scene["bases"][0]["top"] + PLANK_H / 2)), 0.0)
        fill("plank", 0, lambda f: pp)
        contacts.append((0, "plank", "base0"))
    if scene["scenario"] == "pull_out":
        for i, (j, x) in enumerate(scene["init"]):
            r = rest(j, x, stack_y(scene, 0, scene["init"], i))
            fill(j, 0, lambda f, r=r: r)
            contacts.append((0, j, scene["init"][i - 1][0] if i else "base0"))
    cur = {}                                # block -> current rest pose
    world = {}                              # block -> (x, bottom y) in units while it rests
    for i, (j, x) in enumerate(scene["init"]):
        cur[j] = pose[0][j]
        world[j] = (x, stack_y(scene, 0, scene["init"], i))
    for ai, (a, rec) in enumerate(zip(scene["actions"], res["recs"])):
        if rec.get("bad"):
            break
        j = a["block"]
        if a["kind"] == "drop":
            x = a["x"]
            land_py = PY(rec["y"])
            F = fall_frames(land_py, g)
            s = a["start"]
            L = s + F
            r = rest(j, x, rec["y"])
            hh = dims[j][1] / 2

            def falling(f, s=s, F=F, land_py=land_py, r=r, hh=hh, L=L):
                if f < s:
                    return None
                if f >= L:
                    return r
                return r[0], q2(land_py - 0.5 * g * (F * F - (f - s) ** 2) - hh), 0.0
            fill(j, s, falling)
            cur[j] = r
            world[j] = (x, rec["y"])
            on = rec["on"]
            contacts.append((s, j, on))
            events.append({"kind": "land", "f": L, "start": s, "F": F, "block": j, "a": ai, "on": on,
                           "base": rec.get("base", 0)})
            trig = L
        else:
            s = a["start"]
            v, d = scene["pull_v"], a["dir"]
            r0 = cur[j]
            wj = dims[j][0]
            l0, r0x = r0[0] - wj / 2, r0[0] + wj / 2
            gone = s + (math.floor((W - l0) / v) + 1 if d > 0 else math.floor(r0x / v) + 1)
            fill(j, s + 1, lambda f, s=s, r0=r0: (r0[0] + d * v * (f - s), r0[1], 0.0), gone)
            fill(j, gone, lambda f: None)
            ev = {"kind": "pull", "f": s + 1, "start": s, "block": j, "a": ai, "dir": d, "gone": gone,
                  "above": rec["above"], "on": rec["on"], "clear": None, "settle": None}
            trig = s + 1
            if rec["above"]:
                ul = min(cur[q][0] - dims[q][0] / 2 for q in rec["above"])
                ur = max(cur[q][0] + dims[q][0] / 2 for q in rec["above"])
                k = math.ceil((ur + GAP + 1 - l0) / v) if d > 0 else math.ceil((r0x - ul + GAP + 1) / v)
                c = s + k
                dh = dims[j][1]
                Dd = max(2, math.ceil(math.sqrt(2 * dh / g)))
                for q in rec["above"]:
                    rq = cur[q]
                    world[q] = (world[q][0], world[q][1] - bl[j]["h"])
                    newr = rest(q, *world[q])
                    fill(q, c + 1, lambda f, c=c, rq=rq, newr=newr, Dd=Dd:
                         newr if f >= c + Dd else (rq[0], q2(rq[1] + dh * ((f - c) / Dd) ** 2), 0.0))
                    cur[q] = newr
                contacts.append((c, rec["above"][0], rec["on"]))
                ev.update(clear=c, settle=c + Dd)
                trig = c + Dd
            del cur[j]
            events.append(ev)
        fl = rec["fall"]
        if fl:
            tip = trig + a["pause"]
            tp = scene["tip"]
            side = fl["side"]
            b = fl["base"]
            if fl["plank"]:
                B = scene["bases"][0]
                lo, hi = _edges(B["x"], B["w"])
                pivot = (PX(hi if side == "right" else lo), PY(B["top"]))
                movers = [("plank", (q2(PX(scene["plank"]["x"])), q2(PY(B["top"] + PLANK_H / 2)), 0.0))]
                movers += [(q, cur[q]) for q in fl["blocks"]]
            else:
                before = rec["before"] + ([(j, a["x"])] if a["kind"] == "drop" else [])
                if a["kind"] == "pull":
                    before = [t for t in rec["before"] if t[0] != j]
                i = fl["surface"]
                lo, hi = contact(scene, b, before, i)
                pivot = (PX(hi if side == "right" else lo), PY(stack_y(scene, b, before, i)))
                movers = [(q, cur[q]) for q in fl["blocks"]]
            sizes = {q: (scene["plank"]["w"] * S, PLANK_H * S) if q == "plank" else dims[q] for q, _ in movers}
            k = 0
            while True:
                k += 1
                polys = [rect(*_tip(p0, pivot, side, k, tp, g), *sizes[q]) for q, p0 in movers]
                if _outside(polys) or k > 600:
                    break
            gone = tip + k
            for q, p0 in movers:
                fill(q, tip + 1, lambda f, p0=p0: _tip(p0, pivot, side, f - tip, tp, g), gone)
                fill(q, gone, lambda f: None)
                cur.pop(q, None)
            events.append({"kind": "fall", "f": tip + 1, "tip": tip, "gone": gone, "a": ai, "base": b,
                           "blocks": list(fl["blocks"]), "plank": fl["plank"], "side": side, "from": fl["from"],
                           "cause": j, "trigger": trig, "pivot": list(pivot)})
    events.sort(key=lambda e: (e["f"], e["kind"]))
    return {"pose": pose, "plank": ppose, "events": events, "contacts": contacts, "res": res, "n": n}


def bases_px(scene) -> list:
    """Pillar rectangles (x0, y_top, x1) in px; they run down past the bottom of the frame."""
    S, ox, oy = scene["S"], scene["ox"], scene["oy"]
    return [(ox + S * (B["x"] - B["w"] / 2), oy - S * B["top"], ox + S * (B["x"] + B["w"] / 2)) for B in scene["bases"]]


# ---------------------------------------------------------------- geometry checks

def _sat(A, B) -> float:
    """Largest separating gap over the edge normals (< 0: overlap depth on the best axis)."""
    best = -math.inf
    for P in (A, B):
        for i in range(len(P)):
            x1, y1 = P[i]
            x2, y2 = P[(i + 1) % len(P)]
            nx, ny = y2 - y1, x1 - x2
            L = math.hypot(nx, ny)
            if L == 0:
                continue
            nx, ny = nx / L, ny / L
            pa = [x * nx + y * ny for x, y in A]
            pb = [x * nx + y * ny for x, y in B]
            best = max(best, min(pb) - max(pa), min(pa) - max(pb))
    return best


def _pseg(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def poly_gap(A, B) -> float:
    """Distance between convex polygons (> 0), or minus the overlap depth."""
    s = _sat(A, B)
    if s <= 0:
        return s
    d = math.inf
    for P, Q in ((A, B), (B, A)):
        for p in P:
            for i in range(len(Q)):
                d = min(d, _pseg(p, Q[i], Q[(i + 1) % len(Q)]))
    return d


def _bbox(p):
    xs = [v[0] for v in p]
    ys = [v[1] for v in p]
    return min(xs), min(ys), max(xs), max(ys)


def _bgap(a, b) -> float:
    return max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3])


def polys_at(scene, tl, f) -> dict:
    """item -> polygon of everything drawn at frame f (blocks, plank, pillars)."""
    S = scene["S"]
    out = {}
    for j, b in enumerate(scene["blocks"]):
        p = tl["pose"][f][j]
        if p is not None:
            out[j] = rect(p[0], p[1], p[2], b["w"] * S, b["h"] * S)
    if scene["plank"] and tl["plank"][f] is not None:
        p = tl["plank"][f]
        out["plank"] = rect(p[0], p[1], p[2], scene["plank"]["w"] * S, PLANK_H * S)
    for b, (x0, yt, x1) in enumerate(bases_px(scene)):
        out[f"base{b}"] = [(x0, yt), (x1, yt), (x1, H + 60), (x0, H + 60)]
    return out


def geometry_errors(scene, tl) -> list[str]:
    """Touching pairs never overlap (> PEN px); other pairs stay >= GAP px apart;
    nothing comes within CLOCK_CLEAR px of the clock; resting things stay in the frame."""
    n = tl["n"]
    errs = []
    tb = TIMER_BOX
    clock = [(tb[0], tb[1]), (tb[2], tb[1]), (tb[2], tb[3]), (tb[0], tb[3])]
    touching = {}                          # pair -> has it touched yet (dropped once it separates again)
    starts = {}
    for f, a, b in tl["contacts"]:
        starts.setdefault(f, []).append(frozenset((a, b)))
    prev = None
    for f in range(n):
        P = polys_at(scene, tl, f)
        for pr in starts.get(f, []):
            touching[pr] = False
        moved = set(P) if prev is None else {k for k in P if k not in prev or P[k] != prev[k]}
        if moved:
            bb = {k: _bbox(v) for k, v in P.items()}
            keys = list(P)
            for a in moved:
                if a in ("base0", "base1"):
                    continue
                if _bgap(bb[a], _bbox(clock)) < CLOCK_CLEAR:
                    if poly_gap(P[a], clock) < CLOCK_CLEAR:
                        errs.append(f"frame {f}: {a} near the clock")
                for b in keys:
                    if b == a or (b in moved and str(b) < str(a)):
                        continue
                    pr = frozenset((a, b))
                    if _bgap(bb[a], bb[b]) >= GAP:
                        if touching.get(pr):
                            del touching[pr]
                        continue
                    d = poly_gap(P[a], P[b])
                    if pr in touching:
                        if d < -PEN:
                            errs.append(f"frame {f}: {a} and {b} overlap ({d:.1f} px)")
                        elif d >= GAP and touching[pr]:
                            del touching[pr]
                        elif d < GAP:
                            touching[pr] = True
                    elif d < GAP:
                        errs.append(f"frame {f}: {a} and {b} only {d:.1f} px apart")
                if len(errs) > 5:
                    return errs
        prev = P
    for j, b in enumerate(scene["blocks"]):
        for f in (0, n - 1):
            p = tl["pose"][f][j]
            q = tl["pose"][f - 1 if f else 1][j] if n > 1 else p
            if p is not None and p[2] == 0 and p == q:
                x0, y0, x1, y1 = _bbox(rect(p[0], p[1], 0, b["w"] * scene["S"], b["h"] * scene["S"]))
                if x0 < 6 or x1 > W - 6 or y0 < 6:
                    errs.append(f"block {j} rests outside the frame")
    return errs


# ---------------------------------------------------------------- generation

def _style(rng):
    top, bot, light = rng.choice(BACKGROUNDS)
    if rng.random() < 0.4:
        top, bot = bot, top
    return {"bg": [list(top), list(bot)], "light": light, "bg_style": rng.choice(["flat", "gradient", "vignette"]),
            "grid": rng.choice(["lines", "dots"]), "grid_mix": rng.choice([0.16, 0.22, 0.28]),
            "pillar": list(rng.choice(PILLARS[light])), "plank": list(rng.choice(PLANKS)),
            "outline": rng.choice([3, 4]), "stripe": rng.choice([0.22, 0.3])}


def _half_steps(lo, hi):
    a = math.ceil(lo * 2 - 1e-9)
    b = math.floor(hi * 2 + 1e-9)
    return [k / 2 for k in range(a, b + 1)]


def _xbounds(scene):
    """World x range where a resting block may sit (the frame is fitted around it afterwards)."""
    return (-15.0, 15.0) if scene["scenario"] == "twin" else (-8.5, 8.5)


def _extent(sc):
    """(xmin, xmax, highest top) in world units over everything that ever rests."""
    bl = sc["blocks"]
    xs, top = [], 0.0
    for B in sc["bases"]:
        xs += list(_edges(B["x"], B["w"]))
        top = max(top, B["top"])
    if sc["plank"]:
        xs += list(_edges(sc["plank"]["x"], sc["plank"]["w"]))
        top = max(top, PLANK_H)
    for i, (j, x) in enumerate(sc["init"]):
        xs += list(_edges(x, bl[j]["w"]))
        top = max(top, stack_y(sc, 0, sc["init"], i) + bl[j]["h"])
    for a, rec in zip(sc["actions"], play(sc)["recs"]):
        if a["kind"] == "drop":
            xs += list(_edges(a["x"], bl[a["block"]]["w"]))
            top = max(top, rec["y"] + bl[a["block"]]["h"])
    return min(xs), max(xs), top


def _frame(rng, sc) -> bool:
    """Scale (px per unit) and origin so the whole build fills the frame."""
    xmin, xmax, top = _extent(sc)
    sc["oy"] = rng.randint(400, 600) if sc["plank"] else rng.randint(470, 640)
    s_hi = min(44 if sc["plank"] else 46, (W - 280) / (xmax - xmin), (sc["oy"] - 120) / top)
    if s_hi < 28:
        return False
    sc["S"] = rng.randint(max(28, int(s_hi * 0.75)), int(s_hi))
    lo, hi = math.ceil(240 - sc["S"] * xmin), math.floor(W - 40 - sc["S"] * xmax)
    if lo > hi:
        return False
    sc["ox"] = rng.randint(lo, hi)
    return True


def _new_block(rng, scene, widths, heights):
    b = {"w": rng.choice(widths), "h": rng.choice(heights)}
    scene["blocks"].append(b)
    return len(scene["blocks"]) - 1


def _lean(scene, res, b) -> float:
    """How close the structure on base b is to tipping after the replay (smaller = closer)."""
    if scene["plank"]:
        return plank_status(scene, res["plank"])[2] if res["plank_up"] else math.inf
    st = res["stacks"][b]
    return stack_status(scene, b, st)[2] if st else math.inf


def _gen_drops(rng, scene, n, fall_at, base_of, widths, heights):
    """Drop actions for tower / twin / plank, with a collapse exactly at the steps in fall_at.
    Before a planned collapse the stack is often built to lean (closest-to-tipping candidate)."""
    xlo, xhi = _xbounds(scene)
    for k in range(n):
        want = k in fall_at
        b = base_of[k]
        later = any(k < t <= k + 2 and base_of[t] == b for t in fall_at)     # a collapse here soon
        lean = later and rng.random() < (0.85 if scene["plank"] else 0.5)
        found = False
        for _ in range(30):
            j = _new_block(rng, scene, widths, heights)
            w = scene["blocks"][j]["w"]
            if scene["plank"]:
                P = scene["plank"]
                xs = _half_steps(P["x"] - P["w"] / 2 + w / 2 + 0.5, P["x"] + P["w"] / 2 - w / 2 - 0.5)
            else:
                st = play(scene)["stacks"][b]
                top = (st[-1][1], scene["blocks"][st[-1][0]]["w"]) if st else (scene["bases"][b]["x"],
                                                                               scene["bases"][b]["w"])
                xs = _half_steps(top[0] - top[1] / 2 - w / 2 + MIN_CONTACT, top[0] + top[1] / 2 + w / 2 - MIN_CONTACT)
            xs = [x for x in xs if xlo + w / 2 <= x <= xhi - w / 2]
            rng.shuffle(xs)
            good = []
            for x in xs[:16]:
                act = {"kind": "drop", "block": j, "x": x, "base": b, "start": 0, "pause": rng.randint(3, 8)}
                res = play(scene, scene["actions"] + [act])
                if not res["ok"]:
                    continue
                fl = res["recs"][-1]["fall"]
                if (fl is not None) != want:
                    continue
                if fl and scene["scenario"] == "twin" and fl["side"] != ("left", "right")[b] and rng.random() < 0.85:
                    continue
                good.append((_lean(scene, res, b) if lean else 0, len(good), act))
            if good:
                scene["actions"].append(min(good)[2] if lean else good[0][2])
                found = True
                break
            scene["blocks"].pop()
        if not found:
            return False
    return True


def _gen_pulls(rng, scene, n, fall_at):
    for k in range(n):
        want = k in fall_at
        st = play(scene)["stacks"][0]
        if len(st) < 2:
            return k >= 2
        cands = [q for q, _ in st[:-1]] + ([st[-1][0]] if rng.random() < 0.15 else [])
        rng.shuffle(cands)
        done = False
        for q in cands:
            act = {"kind": "pull", "block": q, "dir": rng.choice([-1, 1]), "start": 0, "pause": rng.randint(3, 8)}
            res = play(scene, scene["actions"] + [act])
            if not res["ok"] or (res["recs"][-1]["fall"] is not None) != want:
                continue
            scene["actions"].append(act)
            done = True
            break
        if not done:
            return k > max(fall_at, default=99)           # stop early once the planned collapse happened
    return True


def _gen_init(rng, sc, n0, widths, heights, weak, zig) -> bool:
    """The standing tower of a pull_out clip (bottom to top). With `weak` = k, the
    block above k is placed so that it tips once k is pulled out."""
    xlo, xhi = _xbounds(sc)
    sign = rng.choice([-1, 1])
    bl = sc["blocks"]

    def below():
        if sc["init"]:
            return sc["init"][-1][1], bl[sc["init"][-1][0]]["w"]
        return 0.0, sc["bases"][0]["w"]

    def fits(x, w):
        bx, bw = below()
        l, r = _edges(x, w)
        return min(r, bx + bw / 2) - max(l, bx - bw / 2) >= MIN_CONTACT and xlo + w / 2 <= x <= xhi - w / 2

    def tips():
        r = play(sc, [{"kind": "pull", "block": sc["init"][weak][0], "dir": 1, "start": 0, "pause": 3}])
        return r["ok"] and r["recs"][0]["fall"] is not None

    i = 0
    while i < n0:
        placed = False
        for _ in range(40):
            if weak is not None and i == weak:
                jb = _new_block(rng, sc, widths, heights)
                xb = below()[0] + rng.choice([-1, 1]) * rng.choice([1.0, 1.5, 2.0])
                if not fits(xb, bl[jb]["w"]):
                    bl.pop()
                    continue
                sc["init"].append((jb, xb))
                jc = _new_block(rng, sc, [w for w in widths if w >= 3] or [3, 4], heights)
                bx, bw = below()
                w = bl[jc]["w"]
                xs = [x for x in _half_steps(bx - bw / 2 - w / 2 + MIN_CONTACT, bx + bw / 2 + w / 2 - MIN_CONTACT)
                      if xlo + w / 2 <= x <= xhi - w / 2]
                rng.shuffle(xs)
                for x in xs:
                    sc["init"].append((jc, x))
                    if play(sc)["ok"] and tips():
                        placed = True
                        break
                    sc["init"].pop()
                if placed:
                    i += 2
                    break
                sc["init"].pop()
                bl.pop()
                bl.pop()
                continue
            j = _new_block(rng, sc, widths, heights)
            if weak is not None and i > weak:
                x = below()[0] + rng.choice([-1.0, -0.5, 0.0, 0.0, 0.5, 1.0])
            elif rng.random() < zig:
                sign = -sign
                x = below()[0] + sign * rng.choice([1.0, 1.5, 2.0])
            else:
                x = below()[0] + rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
            if fits(x, bl[j]["w"]):
                sc["init"].append((j, x))
                if play(sc)["ok"] and (weak is None or i <= weak or tips()):
                    placed = True
                    i += 1
                    break
                sc["init"].pop()
            bl.pop()
        if not placed:
            return False
    return True


def _schedule(rng, scene) -> bool:
    """Start frames: landings / pulls SEP+ frames apart, collapses out of the way first."""
    g, S, oy = scene["g"], scene["S"], scene["oy"]
    res = play(scene)
    tp = scene["tip"]
    nxt = EDGE + rng.randint(0, 10)              # earliest next landing / pull
    tower_free = {}                              # base -> earliest landing after a collapse there
    last_b = None
    for a, rec in zip(scene["actions"], res["recs"]):
        if a["kind"] == "drop":
            F = fall_frames(oy - S * rec["y"], g)
            b = a["base"]
            L = max(nxt, tower_free.get(b, 0), EDGE + F if a is scene["actions"][0] else 0)
            if last_b == b:
                L = max(L, nxt + rng.randint(2, 10))
            a["start"] = L - F
            trig = L
            last_b = b
        else:
            a["start"] = nxt
            tl = timeline_of(scene, MAX_FRAMES * 3)
            ev = [e for e in tl["events"] if e["kind"] == "pull" and e["block"] == a["block"]][0]
            trig = ev["settle"] or ev["f"]
            L = ev["f"]
        nxt = max(L, trig) + SEP + rng.randint(0, 10)
        if rec["fall"]:
            tip = trig + a["pause"]
            b = rec["fall"]["base"]
            tower_free[b] = tip + tp["t1"] + rng.randint(5, 10)
            nxt = max(nxt, tip + 4)
            if a["kind"] == "pull":
                nxt = max(nxt, tower_free[b])
    return True


def _looks(rng, scene) -> bool:
    """Colour + pattern for every block: unique looks, and blocks that ever touch differ in colour."""
    nb = len(scene["blocks"])
    pats = rng.sample(PATTERNS, rng.choice([1, 2, 2, 3, 3]))
    need = max(3, -(-nb // len(pats)))
    if need > len(COLOURS):
        pats = list(PATTERNS)
        need = max(3, -(-nb // 3))
    cols = rng.sample(list(COLOURS), rng.randint(need, len(COLOURS)))
    tl = timeline_of(scene, MAX_FRAMES * 3)
    nbr = {j: set() for j in range(nb)}
    for _, a, b in tl["contacts"]:
        if isinstance(a, int) and isinstance(b, int):
            nbr[a].add(b)
            nbr[b].add(a)
    used = set()
    for j in range(nb):
        opts = [(c, p) for c in cols for p in pats if (c, p) not in used
                and all(scene["blocks"][k].get("colour") != c for k in nbr[j])]
        if not opts:
            return False
        c, p = rng.choice(opts)
        scene["blocks"][j].update(colour=c, pattern=p)
        used.add((c, p))
    return True


REJECTS: dict = {}                          # why attempts were thrown away (for the stats report)


def _reject(why):
    REJECTS[why] = REJECTS.get(why, 0) + 1
    return None


def _attempt(rng, scenario, target, size):
    sc = {"scenario": scenario, "style": _style(rng), "labels": [], "plank": None, "init": [], "actions": [],
          "blocks": [], "g": rng.choice([1.6, 1.8, 2.0, 2.2, 2.4]), "pull_v": rng.randint(18, 30),
          "tip": {"w0": rng.choice([2.0, 2.5, 3.0, 3.5]), "acc": rng.choice([0.6, 0.8, 1.0]),
                  "t1": rng.randint(4, 7), "vx": rng.choice([5.0, 6.0, 7.0, 8.0]), "vy": rng.choice([0.0, 1.0, 2.0])}}
    widths = sorted(rng.sample([2, 3, 4, 5, 6], rng.randint(3, 5)))
    heights = [1.0] if rng.random() < 0.3 else sorted(rng.sample(HEIGHTS, rng.randint(2, 3)))
    if scenario == "tower":
        sc["bases"] = [{"x": 0.0, "w": rng.choice([2, 3, 4, 5]), "top": 0.0}]
        n = size
        c = target
        steps = list(range(2, n))
        fall_at = set()
        for _ in range(c):
            ok_steps = [s for s in steps if all(abs(s - t) >= 2 for t in fall_at)]
            if ok_steps:
                fall_at.add(rng.choice(ok_steps))
        if not _gen_drops(rng, sc, n, fall_at, [0] * n, widths, heights):
            return _reject("tower drops")
    elif scenario == "twin":
        D = rng.choice([5.0, 5.5, 6.0, 6.5, 7.0])
        sc["bases"] = [{"x": -D, "w": rng.choice([2, 3, 4]), "top": rng.choice([0.0, 0.5, 1.0, 1.5])},
                       {"x": D, "w": rng.choice([2, 3, 4]), "top": rng.choice([0.0, 0.5, 1.0, 1.5])}]
        n = size
        base_of = []
        for k in range(n):
            base_of.append(rng.randrange(2) if k < 2 or rng.random() < 0.4 else 1 - base_of[-1])
        if len(set(base_of)) < 2:
            return _reject("twin one tower")
        c = target
        fall_at = set()
        for _ in range(c):
            ok_steps = [s for s in range(2, n) if base_of[:s].count(base_of[s]) >= 1
                        and all(abs(s - t) >= 2 for t in fall_at)]
            if ok_steps:
                fall_at.add(rng.choice(ok_steps))
        if not _gen_drops(rng, sc, n, fall_at, base_of, [w for w in widths if w <= 5] or [3, 4], heights):
            return _reject("twin drops")
    elif scenario == "pull_out":
        sc["bases"] = [{"x": 0.0, "w": rng.choice([2, 3, 4, 5]), "top": 0.0}]
        n0 = size
        n = rng.randint(2, 4)
        c = target
        fall_at = {0 if rng.random() < 0.7 else 1} if c else set()
        zig = rng.choice([0.3, 0.6, 0.8])
        for _ in range(12):
            sc["blocks"], sc["init"], sc["actions"] = [], [], []
            weak = rng.randint(0, n0 - 3) if c else None
            if _gen_init(rng, sc, n0, widths, heights, weak, zig) and _gen_pulls(rng, sc, n, fall_at):
                break
        else:
            return _reject("pull_out tower / pulls")
        if len(sc["actions"]) < 2:
            return _reject("pull_out tower / pulls")
        sc["init"] = [list(t) for t in sc["init"]]
    else:
        ws = rng.choice([1, 2, 3])
        sc["bases"] = [{"x": 0.0, "w": ws, "top": 0.0}]
        pw = rng.choice([10, 12, 14, 16])
        px_ = rng.choice([v for v in (-1.0, -0.5, 0.0, 0.5, 1.0) if abs(v) <= ws / 2 - MARGIN])
        sc["plank"] = {"x": px_, "w": pw}
        n = size
        fall_at = {n - 1} if target else set()
        if not _gen_drops(rng, sc, n, fall_at, [0] * n, [1, 2, 3], heights):
            return _reject("plank drops")
    res = play(sc)
    if not res["ok"]:
        return _reject("rule")
    if sum(1 for r in res["recs"] if r["fall"]) != target:
        return _reject("collapse count")
    if not _frame(rng, sc):
        return _reject("does not fit the frame")
    _schedule(rng, sc)
    tl = timeline_of(sc, MAX_FRAMES * 3)
    last = 0
    for e in tl["events"]:
        last = max(last, e.get("settle") or e["f"])
        if e["kind"] == "fall":
            last = max(last, e["tip"] + sc["tip"]["t1"], e["gone"] - EDGE + 4)
    sc["n_frames"] = last + EDGE + 1 + rng.randint(0, 10)
    if sc["n_frames"] > MAX_FRAMES:
        return _reject("too long")
    if not _looks(rng, sc):
        return _reject("looks")
    return sc


TARGETS = {"tower": ((0, 1, 2), (0.3, 0.45, 0.25)), "twin": ((0, 1, 2), (0.3, 0.45, 0.25)),
           "pull_out": ((0, 1), (0.4, 0.6)), "plank": ((0, 1), (0.45, 0.55))}


def build(rng: random.Random) -> dict:
    scenario = rng.choice(SCENARIOS)
    target = rng.choices(*TARGETS[scenario])[0]
    size = rng.randint(*{"tower": (4 + 2 * max(0, target - 1), 9), "twin": (5 + max(0, target - 1), 10),
                         "pull_out": (5, 8), "plank": (3, 6)}[scenario])
    for attempt in range(1, 2000):
        sc = _attempt(rng, scenario, target, size)
        if sc is None:
            continue
        errs = check(sc)
        if not errs:
            sc["attempts"] = attempt
            return sc
        _reject("check: " + errs[0].split(": ")[-1].split(" only")[0].split(" overlap")[0][:40])
    raise RuntimeError(f"no valid {scenario} scene")


# ---------------------------------------------------------------- checks

def check(scene) -> list[str]:
    errs = []
    n = scene.get("n_frames", 0)
    if scene.get("scenario") not in SCENARIOS:
        errs.append("unknown scenario")
    if not 1 <= n <= MAX_FRAMES:
        errs.append(f"clip has {n} frames")
        return errs
    if scene["labels"]:
        errs.append("labels present")
    bl = scene["blocks"]
    looks = [(b["colour"], b["pattern"]) for b in bl]
    if len(set(looks)) != len(looks):
        errs.append("two blocks look the same")
    if min(min(b["w"], b["h"]) * scene["S"] for b in bl) < 20:
        errs.append("a block is under 20 px")
    res = play(scene)
    if not res["ok"]:
        errs.append(f"rule not clear-cut (margin {res['worst']:.2f}) or a contact under {MIN_CONTACT}")
    tl = timeline(scene)
    for _, a, b in tl["contacts"]:
        if isinstance(a, int) and isinstance(b, int) and bl[a]["colour"] == bl[b]["colour"]:
            errs.append("touching blocks share a colour")
    ev = tl["events"]
    marks = []                                     # (frame, what) of every change a question can name
    for e in ev:
        if e["kind"] == "land":
            marks.append((e["f"], "land"))
            if e["start"] < 0:
                errs.append("a drop starts before the clip")
        elif e["kind"] == "pull":
            marks.append((e["f"], "pull"))
            if e["settle"]:
                marks.append((e["settle"], "settle"))
        else:
            marks.append((e["f"], "fall"))
            if e["gone"] > n - 1 - 4:
                errs.append("a falling piece is still in view at the end")
            if e["tip"] + scene["tip"]["t1"] > n - 1 - EDGE:
                errs.append("a collapse too close to the end")
    marks.sort()
    if not marks:
        errs.append("nothing happens")
        return errs
    if min(f for f, _ in marks) < EDGE or max(f for f, _ in marks) > n - 1 - EDGE:
        errs.append("an event within 0.5 s of the start or end")
    first = [e for e in ev if e["kind"] in ("land", "pull")]
    if first and first[0].get("start", 0) < EDGE and first[0]["kind"] == "land":
        errs.append("first block starts falling within 0.5 s of the start")
    for (f1, k1), (f2, k2) in zip(marks, marks[1:]):
        if f2 - f1 < 2:
            errs.append(f"events {k1}@{f1} and {k2}@{f2} closer than 2 frames")
    steps = [e["f"] for e in ev if e["kind"] in ("land", "pull")]
    if any(b - a < SEP for a, b in zip(steps, steps[1:])):
        errs.append("landings / pulls closer than SEP frames")
    errs += geometry_errors(scene, tl)
    return errs
