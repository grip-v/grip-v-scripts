"""Independent answers for spinning_polyhedra.

Nothing here reads sim's closed-form motion results (segment angles, crossing
solver, the SOLIDS count table or the convex hull). Instead:

  * Motion comes from the renderer's own per-frame geometry: the vertex
    positions render.camera_coords draws at every frame are taken back to
    world space, and the rotation between consecutive frames is recovered by
    a Kabsch (SVD) fit. That gives, per frame step, the turning axis and the
    signed angle; still frames are steps with no rotation. Turns, speeds,
    stops, axis changes and directions are all read off this track.
  * Markers are followed by the drawn position of their corner: a crossing
    in front of the upright axis is a sign change of its x with z > 0; its
    side / depth label is read from the sign of its screen offset from the
    axis line and its depth.
  * Edge directions are read from the screen motion of the drawn vertices on
    the near side.
  * Shape counts are rebuilt by brute force from the drawn vertex
    coordinates: every plane through three vertices with all others on one
    side is a face, an edge is two vertices sharing two faces. The solid's
    name is the unique table entry with the same (corners, edges, faces,
    face sizes) signature.
  * Counterfactuals are re-simulated: the changed motion (sim.scale_body /
    no_stop_body / swap_speed_body) is re-tracked through the renderer, and
    truncated / dual / one-more-side solids are actually constructed and
    brute-force counted.
"""
from __future__ import annotations
import functools
import itertools
import json
import math

import numpy as np

import questions as Q
import render
import sim

STILL = 1e-7  # degrees per frame


# ---- brute-force polyhedron counting --------------------------------------------------
def brute(v: np.ndarray):
    """(faces as frozensets of vertex indices with outward unit normal and offset, edges)."""
    v = v - v.mean(axis=0)
    n = len(v)
    tri = np.array(list(itertools.combinations(range(n), 3)))
    a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    nr = np.cross(b - a, c - a)
    ln = np.linalg.norm(nr, axis=1)
    ok = ln > 1e-9
    nr, a = nr[ok] / ln[ok, None], a[ok]
    d = (nr * a).sum(axis=1)
    s = v @ nr.T - d
    sup = (s <= 1e-7).all(axis=0) | (s >= -1e-7).all(axis=0)
    faces = {}
    for k in np.nonzero(sup)[0]:
        key = frozenset(np.nonzero(np.abs(s[:, k]) < 1e-6)[0].tolist())
        if key not in faces:
            nn, dd = nr[k], d[k]
            if dd < 0:
                nn, dd = -nn, -dd
            faces[key] = (nn, dd)
    edges = {(i, j) for i, j in itertools.combinations(range(n), 2)
             if sum(1 for f in faces if i in f and j in f) == 2}
    return faces, edges


@functools.lru_cache(maxsize=None)
def counts(name):
    """(V, E, F, face sizes sorted, edges, faces) of the solid as drawn."""
    v = sim.geometry(name)[0]
    faces, edges = brute(v)
    return len(v), len(edges), len(faces), tuple(sorted(len(f) for f in faces)), edges, faces


@functools.lru_cache(maxsize=None)
def identify(name):
    """The table name whose signature matches the drawn geometry (must be unique)."""
    V, E, F, sizes = counts(name)[:4]
    hits = [o for o, (v, e, f, sh) in sim.SOLIDS.items()
            if (v, e, f) == (V, E, F) and tuple(sorted(k for k, c in sh.items() for _ in range(c))) == sizes]
    return hits[0] if len(hits) == 1 else None


@functools.lru_cache(maxsize=None)
def truncated_counts(name):
    """Slice off every corner with a plane inside its normal cone; count the new solid by brute force."""
    v = sim.geometry(name)[0]
    v = v - v.mean(axis=0)
    _, _, _, _, edges, faces = counts(name)
    nbr = {i: [] for i in range(len(v))}
    for i, j in edges:
        nbr[i].append(j)
        nbr[j].append(i)
    pts = []
    for i in range(len(v)):
        nvec = sum(nn for f, (nn, _) in faces.items() if i in f)
        nvec = nvec / np.linalg.norm(nvec)
        gaps = {j: float(nvec @ (v[i] - v[j])) for j in nbr[i]}
        delta = 0.25 * min(gaps.values())
        for j in nbr[i]:
            pts.append(v[i] + (v[j] - v[i]) * delta / gaps[j])
    pts = np.array(pts)
    f2, e2 = brute(pts)
    return len(pts), len(e2), len(f2)


@functools.lru_cache(maxsize=None)
def dual_corners(name):
    """Polar dual: one corner per face plane (n / d); count the distinct corners."""
    _, _, _, _, _, faces = counts(name)
    pts = {tuple(np.round(nn / dd, 6)) for nn, dd in faces.values()}
    return len(pts)


@functools.lru_cache(maxsize=None)
def more_sides_edges(name):
    nm = identify(name)
    fam = nm.split()[-1] if nm else None
    if fam not in sim.FAMILY_FN:
        return None
    V, _, _, sizes = counts(name)[:4]
    if fam == "bipyramid":
        n = V - 2
    else:
        want = 1 if fam == "pyramid" else 2
        n = next(k for k in set(sizes) if sizes.count(k) == want)
    _, e = brute(np.array(sim.FAMILY_FN[fam](n + 1), dtype=float))
    return len(e)


# ---- motion track from the renderer -----------------------------------------------------
def _body_key(scene, body):
    return json.dumps({"b": body, "cam": scene["camera"], "n": scene["n_frames"]}, sort_keys=True)


@functools.lru_cache(maxsize=32)
def _track(key):
    d = json.loads(key)
    scene = {"camera": d["cam"], "n_frames": d["n"]}
    body = d["b"]
    n = d["n"]
    ie = sim.rot("x", math.radians(scene["camera"]["elev"]))       # world -> camera; inverse is .T
    cam = np.array([render.camera_coords(scene, 0, f, body) for f in range(n)])
    world = cam @ ie                                                  # undo the camera elevation
    P, Qm = world[:-1], world[1:]
    Hm = np.einsum("fvi,fvj->fij", P, Qm)
    U, S, Vt = np.linalg.svd(Hm)
    dsgn = np.sign(np.linalg.det(np.einsum("fji,fkj->fik", Vt, U)))
    D = np.zeros((n - 1, 3, 3))
    D[:, 0, 0], D[:, 1, 1], D[:, 2, 2] = 1, 1, dsgn
    R = np.einsum("fji,fjk,flk->fil", Vt, D, U)
    w = 0.5 * np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=1)
    cosang = (np.trace(R, axis1=1, axis2=2) - 1) / 2
    ang = np.degrees(np.arctan2(np.linalg.norm(w, axis=1), cosang))
    ax = np.argmax(np.abs(w), axis=1)
    sign = np.sign(w[np.arange(n - 1), ax])
    signed = ang * sign
    axes = ["xyz"[k] if a > STILL else None for k, a in zip(ax, ang)]
    sx, sy = render.to_screen(scene, 0, cam, body)
    return {"world": world, "cam": cam, "sx": sx, "sy": sy, "ang": ang, "signed": signed, "axes": axes}


def track(scene, b, body=None):
    return _track(_body_key(scene, body or scene["bodies"][b]))


def moving(tr):
    return [a is not None for a in tr["axes"]]


def runs_steps(tr):
    """Lists of step indices (step i = frame i -> i + 1) of each stretch of turning."""
    out, cur = [], []
    for i, mv in enumerate(moving(tr)):
        if mv:
            cur.append(i)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def stops_of(tr):
    """First frame of each still pose inside the clip (after a stretch of turning)."""
    return [r[-1] + 1 for r in runs_steps(tr)[:-1]]


def resumes_of(tr):
    return [r[0] + 1 for r in runs_steps(tr)[1:]]


def y_angle(tr, steps):
    return float(sum(tr["signed"][i] for i in steps if tr["axes"][i] == "y"))


def floor_turns(deg):
    return int(math.floor(abs(deg) / 360 + 1e-9))


def run_speed(tr, steps):
    return float(np.mean([tr["ang"][i] for i in steps])) * sim.FPS


def near_dir(scene, b, f, body=None):
    """Which way the near side moves on screen between frames f and f+1 (or None if it does not)."""
    tr = track(scene, b, body)
    z = tr["cam"][f, :, 2]
    wgt = np.clip(z, 0, None)
    if wgt.sum() <= 0 or tr["axes"][f] is None:
        return None
    dx = float(((tr["sx"][f + 1] - tr["sx"][f]) * wgt).sum() / wgt.sum())
    dy = float(((tr["sy"][f + 1] - tr["sy"][f]) * wgt).sum() / wgt.sum())
    if tr["axes"][f] == "z":
        return None
    if abs(dx) >= abs(dy):
        return "to the right" if dx > 0 else "to the left"
    return "downward" if dy > 0 else "upward"


def axis_label(tr, step):
    a = tr["axes"][step]
    return Q.AXIS[a] if a else None


def mark_xz(scene, m, body=None):
    mk = scene["marks"][m]
    tr = track(scene, mk["body"], body)
    p = tr["world"][:, mk["v"]]
    return p[:, 0], p[:, 2], tr


def crossings(scene, m, body=None):
    x, z, _ = mark_xz(scene, m, body)
    return [f for f in range(1, len(x)) if (x[f] > 0) != (x[f - 1] > 0) and z[f] > 0 and z[f - 1] > 0]


def side_label(scene, m, f, body=None):
    mk = scene["marks"][m]
    body = body or scene["bodies"][mk["body"]]
    tr = track(scene, mk["body"], body)
    dx = tr["sx"][f, mk["v"]] - body["cx"]
    z = tr["world"][f, mk["v"], 2]
    return Q.QUAD[("left" if dx < 0 else "right", "near" if z > 0 else "far")]


def trend_label(tr):
    st = runs_steps(tr)
    if len(st) != 1:
        return None
    sp = tr["ang"][st[0]]
    k = len(sp)
    a, mid, c = sp[: k // 4].mean(), sp[int(k * 0.4): int(k * 0.6) + 1].mean(), sp[-(k // 4):].mean()
    if max(a, mid, c) < 1.05 * min(a, mid, c):
        return Q.TREND[2]
    if mid >= 1.3 * a and mid >= 1.3 * c:
        return Q.TREND[3]
    if c >= 1.3 * a:
        return Q.TREND[0]
    if a >= 1.3 * c:
        return Q.TREND[1]
    return None


def halves(tr, n):
    h = (n - 1) / 2
    first = second = 0.0
    for i, a in enumerate(tr["ang"]):
        lo, hi = i, i + 1
        if hi <= h:
            first += a
        elif lo >= h:
            second += a
        else:
            first += a * (h - lo)
            second += a * (hi - h)
    if second < 1e-6:
        return Q.HALVES[3]
    if max(first, second) < 1.1 * min(first, second):
        return Q.HALVES[2]
    return Q.HALVES[0] if first > second else Q.HALVES[1]


def time_text(f):
    return render.sim.timer_text(f)


# ---- answers -------------------------------------------------------------------------
def answer(key: str, p: dict, scene: dict):
    n = scene["n_frames"]
    B = scene["bodies"]
    b = p.get("b")
    m = p.get("m")
    if m is not None:
        b = scene["marks"][m]["body"]
    body = B[b] if b is not None else None
    name = body["solid"] if body else None
    tr = track(scene, b) if b is not None else None

    # L1
    if key == "front_dir":
        return near_dir(scene, b, 0 if p["when"] == "start" else n - 2)
    if key == "n_stops":
        return len(stops_of(tr))
    if key == "spin_axis":
        return axis_label(tr, 0 if p["when"] == "start" else n - 2)
    if key == "mark_side":
        return side_label(scene, m, 0 if p["when"] == "start" else n - 1)
    # geometry
    if key in ("more_fv", "face_kinds", "common_face", "n_faces", "n_corners", "n_edges", "name",
               "cf_truncate_faces", "cf_truncate_corners", "cf_dual_corners", "cf_more_sides"):
        V, E, F, sizes = counts(name)[:4]
        if key == "more_fv":
            return Q.FV[3] if F == 2 * V else Q.FV[0] if F > V else Q.FV[1] if V > F else Q.FV[2]
        if key == "face_kinds":
            return Q.FACE_KINDS[len(set(sizes)) - 1]
        if key == "common_face":
            top = max(sizes.count(k) for k in set(sizes))
            best = [k for k in set(sizes) if sizes.count(k) == top]
            return {3: "triangles", 4: "four-sided faces", 5: "pentagons", 6: "hexagons"}[best[0]] \
                if len(best) == 1 else None
        if key == "n_faces":
            return F
        if key == "n_corners":
            return V
        if key == "n_edges":
            return E
        if key == "name":
            return identify(name)
        if key == "cf_truncate_faces":
            return truncated_counts(name)[2]
        if key == "cf_truncate_corners":
            return truncated_counts(name)[0]
        if key == "cf_dual_corners":
            return dual_corners(name)
        return more_sides_edges(name)
    # L2
    if key == "speed_trend":
        return trend_label(tr)
    if key == "halves":
        return halves(tr, n)
    if key == "start_end_dir":
        a0, a1 = tr["axes"][0], tr["axes"][n - 2]
        if a1 == "x":
            return Q.SED[2]
        if a1 == "z":
            return Q.SED[3]
        return Q.SED[0] if near_dir(scene, b, 0) == near_dir(scene, b, n - 2) else Q.SED[1]
    if key == "first_front":
        firsts = sorted((crossings(scene, i)[0], i) for i, mk in enumerate(scene["marks"])
                        if mk["body"] == b and crossings(scene, i))
        return Q.mn(scene, firsts[0][1]) if firsts else None
    if key == "wider_orbit":
        rs = []
        for i in p["ms"]:
            x, z, _ = mark_xz(scene, i)
            rs.append((float(np.hypot(x, z).mean()), i))
        return Q.mn(scene, max(rs)[1])
    if key == "speed_compare":
        st = runs_steps(tr)
        if len(st) < 2:
            return Q.SPEEDCMP[3]
        a, c = run_speed(tr, st[0][-3:]), run_speed(tr, st[1][:3])
        if abs(a - c) < 0.02 * max(a, c):
            return Q.SPEEDCMP[2]
        return Q.SPEEDCMP[0] if a > c else Q.SPEEDCMP[1]
    if key in ("faster", "dir_pair", "turn_diff", "pair_sum", "vf_pair"):
        trs = [track(scene, i) for i in range(2)]
        sp = [run_speed(t, [i for i in range(n - 1) if t["axes"][i]]) for t in trs]
        if key == "faster":
            return Q.FASTER[0] if sp[0] > sp[1] else Q.FASTER[1]
        if key == "dir_pair":
            d = tuple("left" if near_dir(scene, i, 0) == "to the left" else "right" for i in range(2))
            return Q.DIRPAIR[d]
        if key == "turn_diff":
            ks = [floor_turns(y_angle(t, range(n - 1))) for t in trs]
            fast = 0 if sp[0] > sp[1] else 1
            return ks[fast] - ks[1 - fast]
        c0, c1 = counts(B[0]["solid"]), counts(B[1]["solid"])
        if key == "pair_sum":
            i = {"corners": 0, "edges": 1, "faces": 2}[p["what"]]
            return c0[i] + c1[i]
        if c0[0] == c1[0] or c0[2] == c1[2]:
            return None
        return Q.VFPAIR[(0 if c0[0] > c1[0] else 1, 0 if c0[2] > c1[2] else 1)]
    # L3
    if key == "n_turns":
        return floor_turns(y_angle(tr, range(n - 1)))
    if key == "period":
        return f"{360 / run_speed(tr, range(n - 1)):.1f} s"
    if key == "stop_time":
        st = stops_of(tr)
        return time_text(st[p["k"] - 1]) if len(st) >= p["k"] else None
    if key == "resume_time":
        rs = resumes_of(tr)
        return time_text(rs[p["k"] - 1]) if len(rs) >= p["k"] else None
    if key == "turns_by_time":
        f = next(f for f in range(n) if sim.tenths(f) == p["t"])
        return floor_turns(y_angle(tr, range(f)))
    if key == "turns_before_stop":
        return floor_turns(y_angle(tr, runs_steps(tr)[0]))
    if key == "count_passes":
        return len(crossings(scene, m))
    if key == "first_pass_time":
        cr = crossings(scene, m)
        return time_text(cr[0]) if cr else None
    # L4
    if key == "mark_degree":
        edges = counts(name)[4]
        return sum(1 for e in edges if scene["marks"][m]["v"] in e)
    if key == "pass_time":
        cr = crossings(scene, m)
        return time_text(cr[p["k"] - 1]) if len(cr) >= p["k"] else None
    if key == "passes_before_stop":
        stop = stops_of(tr)[0]
        return sum(1 for f in crossings(scene, m) if f <= stop)
    if key == "extra_turn":
        t = abs(y_angle(tr, range(n - 1))) / 360
        return Q.EXTRA[min(3, int((t - math.floor(t)) * 4))]
    if key == "side_at_time":
        f = next(f for f in range(n) if sim.tenths(f) == p["t"])
        return side_label(scene, m, f)
    if key == "mark_at_stop":
        return side_label(scene, m, stops_of(tr)[0])
    if key == "turns_total":
        return sum(floor_turns(y_angle(tr, r)) for r in runs_steps(tr))
    if key == "net_dir":
        net = y_angle(tr, range(n - 1))
        return Q.NETDIR[0] if net < 0 else Q.NETDIR[1]
    if key == "axis_seq":
        st = runs_steps(tr)
        return Q.AXSEQ.get((tr["axes"][st[0][0]], tr["axes"][st[-1][-1]]))
    # L5 motion counterfactuals: re-track the changed motion through the renderer
    if key == "cf_speed":
        return floor_turns(y_angle(track(scene, b, sim.scale_body(body, p["factor"])), range(n - 1)))
    if key == "cf_speed_first":
        t2 = track(scene, b, sim.scale_body(body, p["factor"]))
        return floor_turns(y_angle(t2, runs_steps(t2)[0]))
    if key == "cf_stop_side":
        alt = sim.scale_body(body, -1)
        return side_label(scene, m, stops_of(track(scene, b, alt))[0], alt)
    if key == "cf_no_switch":
        return floor_turns(y_angle(track(scene, b, sim.upright_body(body)), range(n - 1)))
    if key == "cf_no_stop":
        return floor_turns(y_angle(track(scene, b, sim.no_stop_body(body, n)), range(n - 1)))
    if key == "cf_swap":
        return floor_turns(y_angle(track(scene, b, sim.swap_speed_body(body, B[1 - b])), range(n - 1)))
    if key == "cf_other_way":
        return side_label(scene, m, n - 1, sim.scale_body(body, -1))
    if key == "cf_passes_double":
        return len(crossings(scene, m, sim.scale_body(body, 2)))
    return None
