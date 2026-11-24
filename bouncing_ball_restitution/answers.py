"""Independent answers for bouncing_ball_restitution.

Nothing here reads the generator's bounce lists. Observed quantities come
from the TRACK the renderer draws: `render.object_centers` is sampled at
480 Hz over the clip and floor hits, rests, wall bounces, releases and
rebound heights are detected from that track (local minima of the gap to
the floor, sign flips of the sideways motion, first movement), with hit
times refined by fitting the free-fall parabola through the two samples
before contact. Counterfactuals are re-simulated with a separate
fixed-step integrator (dt = 1/2000 s, crossing solved inside the step).
Clock answers use the frame nearest the detected time.
"""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render  # noqa: E402

FPS = 24
RATE = 480
SETTLE = 30.0
NOUN = {"circle": "ball", "square": "block", "hexagon": "hexagon", "diamond": "diamond"}
BUCKETS = (20, 35, 50, 65, 80)


def _name(o):
    return f"the {o['pattern']} {o['color']} {NOUN[o['shape']]}"


def _floor(scene, o):
    return scene["shelves"][o["lane"]]["floor_y"] if scene["scenario"] == "throw" else scene["floor_y"]


def _clock(t: float) -> str:
    k = (10 * int(round(t * FPS))) // FPS
    return f"{k // 10}.{k % 10} s"


@functools.lru_cache(maxsize=8)
def _tracks(key: str):
    scene = json.loads(key)
    d, g = scene["duration"], scene["g"]
    n = int(round(d * RATE))
    ts = [k / RATE for k in range(n)]
    pts = [render.object_centers(scene, t) for t in ts]
    out = []
    for i, o in enumerate(scene["objects"]):
        fy = _floor(scene, o)
        gap = [fy - (p[i][1] + o["r"]) for p in pts]
        xs = [p[i][0] for p in pts]
        hits = []
        for k in range(2, n - 1):
            if gap[k] < gap[k - 1] - 1e-9 and gap[k + 1] >= gap[k] - 1e-9 and gap[k] <= 4.0:
                y1, y2 = gap[k - 2], gap[k - 1]          # both before contact
                v = (y2 - y1) * RATE - 0.5 * g / RATE     # d(gap)/dt at sample k-1 (negative)
                # gap(t) = y2 + v*s - 0.5*g*s^2 = 0
                s = (v + math.sqrt(max(0.0, v * v + 2 * g * y2))) / g
                hits.append(ts[k - 1] + s)
        rest = None
        if hits:
            k0 = int(hits[-1] * RATE) + 2
            if k0 < n and all(abs(x) <= 0.5 for x in gap[k0:]):
                rest = hits[-1]
        walls = []
        hw = o["r"] * (0.87 if o["shape"] == "hexagon" else 1.0)
        lo, hi = scene["wall_l"] + hw, scene["wall_r"] - hw
        last_s, last_k = 0, 0
        for k in range(1, n):
            step = xs[k] - xs[k - 1]
            s = 0 if abs(step) < 1e-9 else (1 if step > 0 else -1)   # a hit midway between samples gives a 0 step
            if s == 0:
                continue
            if last_s and s != last_s and last_k >= 2:
                # the bounce lies after sample last_k-1 (else the step into last_k would already
                # point the new way); last_k itself may be just past the wall, so fit on k-2, k-1
                v = (xs[last_k - 1] - xs[last_k - 2]) * RATE
                wall = hi if v > 0 else lo
                walls.append((ts[last_k - 1] + (wall - xs[last_k - 1]) / v, "right" if v > 0 else "left"))
            last_s, last_k = s, k
        y0 = pts[0][i][1]
        release = 0.0 if scene["scenario"] != "staggered" else math.inf   # never moves -> never released
        for k in range(1, n):
            dy = pts[k][i][1] - y0
            if dy > 1e-6:
                release = max(0.0, ts[k] - math.sqrt(2 * dy / g))
                break
        h0 = gap[0]
        apex1 = None
        if len(hits) >= 1:
            k1 = int(hits[0] * RATE) + 1
            k2 = int(hits[1] * RATE) if len(hits) >= 2 else n
            apex1 = max(gap[k1:k2]) if k2 > k1 else 0.0
        out.append({"hits": hits, "rest": rest, "walls": walls, "release": release, "h0": h0, "apex1": apex1,
                    "e": math.sqrt(max(apex1 or 0.0, 0.0) / h0) if h0 > 0 else 0.0})
    return out


def resim(h0, e, g, release, t_end, dt=1.0 / 2000):
    """Fixed-step re-simulation of one drop; returns floor-hit times < t_end."""
    t, y, v, hits = release, h0, 0.0, []
    while t < t_end:
        yn = y + v * dt - 0.5 * g * dt * dt
        if yn > 0:
            t, y, v = t + dt, yn, v - g * dt
            continue
        tau = (v + math.sqrt(v * v + 2 * g * y)) / g
        t_hit = t + tau
        if t_hit >= t_end:
            break
        hits.append(t_hit)
        vout = -e * (v - g * tau)
        if vout * vout / (2 * g) < SETTLE:
            break
        t, y, v = t_hit, 0.0, vout
    return hits


def resim_walls(x, vx, lo, hi, t_end, dt=1.0 / 2000):
    t, n = 0.0, 0
    while t < t_end:
        xn = x + vx * dt
        if xn > hi or xn < lo:
            wall = hi if xn > hi else lo
            if t + (wall - x) / vx >= t_end:
                break
            n += 1
            xn = 2 * wall - xn
            vx = -vx
        t, x = t + dt, xn
    return n


def answer(key: str, p: dict, scene: dict):
    tr = _tracks(json.dumps(scene, sort_keys=True))
    objs, d, g = scene["objects"], scene["duration"], scene["g"]
    names = [_name(o) for o in objs]
    n = len(objs)
    i = p.get("obj")

    def unique_max(vals, margin):
        order = sorted(range(len(vals)), key=lambda j: -vals[j])
        return order[0] if vals[order[0]] - vals[order[1]] >= margin else None

    if key == "n_objects":
        return n
    if key == "hits":
        return len(tr[i]["hits"])
    if key in ("largest", "smallest"):
        rs = [o["r"] * (1 if key == "largest" else -1) for o in objs]
        return names[unique_max(rs, 1.0)]
    if key == "highest_start":
        return names[unique_max([t["h0"] for t in tr], 20)]
    if key == "n_resting_end":
        return sum(1 for t in tr if t["rest"] is not None)
    if key == "first_released":
        return names[min(range(n), key=lambda j: tr[j]["release"])]
    if key == "pad_under":
        x = objs[i]["x"]
        return [q["color"] for q in scene["pads"] if q["x0"] <= x <= q["x1"]][0]
    if key == "wall_hits":
        return len(tr[i]["walls"])
    if key == "first_hit":
        return names[min((tr[j]["hits"][0], j) for j in p["among"])[1]]
    if key == "highest_rebound":
        return names[unique_max([t["apex1"] or 0.0 for t in tr], 10)]
    if key == "first_rest":
        return names[min((t["rest"], j) for j, t in enumerate(tr) if t["rest"] is not None)[1]]
    if key == "last_rest":
        return names[max((t["rest"], j) for j, t in enumerate(tr))[1]]
    if key == "bouncier_pair":
        a, b = p["a"], p["b"]
        return names[a] if tr[a]["e"] > tr[b]["e"] else names[b]
    if key == "more_hits_pair":
        ka, kb = len(tr[p["a"]]["hits"]), len(tr[p["b"]]["hits"])
        return names[p["a"]] if ka > kb else names[p["b"]] if kb > ka else "they land equally often"
    if key == "released_after":
        order = sorted(range(n), key=lambda j: tr[j]["release"])
        return names[order[order.index(i) + 1]]
    if key == "first_wall":
        return names[min((tr[j]["walls"][0][0], j) for j in p["among"])[1]]
    if key == "first_hit_time":
        return _clock(tr[i]["hits"][0])
    if key == "rest_time":
        return _clock(tr[i]["rest"])
    if key == "release_time":
        return _clock(tr[i]["release"])
    if key == "first_wall_time":
        return _clock(tr[i]["walls"][0][0])
    if key == "gap_first_second":
        k = int(round((tr[i]["hits"][1] - tr[i]["hits"][0]) * 10))
        return f"{k // 10}.{k % 10} s"
    if key == "rebound_fraction":
        frac = 100 * tr[i]["apex1"] / tr[i]["h0"]
        return f"about {min(BUCKETS, key=lambda b: abs(b - frac))}%"
    if key == "hits_between":
        return sum(1 for t in tr[i]["hits"] if p["t0"] < t < p["t1"])
    if key == "hits_by":
        return sum(1 for t in tr[i]["hits"] if t < p["t"])
    if key == "total_hits":
        return sum(len(t["hits"]) for t in tr)
    if key in ("most_bouncy", "least_bouncy", "most_bouncy_pad", "least_bouncy_pad"):
        es = [t["e"] * (1 if key.startswith("most") else -1) for t in tr]
        w = max(range(n), key=lambda j: es[j])
        return f"the {scene['pads'][objs[w]['lane']]['color']} pad" if key.endswith("pad") else names[w]
    if key == "most_hits":
        return names[unique_max([len(t["hits"]) for t in tr], 1)]
    if key == "hits_before":
        t0 = tr[i]["hits"][0]
        return sum(1 for j, t in enumerate(tr) if j != i for h in t["hits"] if h < t0)
    if key == "moving_at":
        T = p["t"]
        return sum(1 for t in tr if t["release"] < T and (t["rest"] is None or t["rest"] > T))
    if key == "total_wall_hits":
        return sum(len(t["walls"]) for t in tr)
    # ---- counterfactuals: re-simulated from the measured start state
    o = objs[i] if i is not None else None
    if key == "cf_more_time":
        e = _e_of(scene, o)
        return sum(1 for t in resim(tr[i]["h0"], e, g, tr[i]["release"], d + p["extra"]) if t >= d)
    if key == "cf_height":
        return len(resim(tr[i]["h0"] * p["mult"], _e_of(scene, o), g, tr[i]["release"], d))
    if key == "cf_pad":
        return len(resim(tr[i]["h0"], scene["pads"][p["pad"]]["e"], g, tr[i]["release"], d))
    if key == "cf_bounciness":
        return len(resim(tr[i]["h0"], _e_of(scene, objs[p["like"]]), g, tr[i]["release"], d))
    if key == "cf_stick":
        return sum(len(t["hits"]) for t in tr) - (len(tr[i]["hits"]) - 1)
    if key == "cf_same_release":
        return names[min(p["among"], key=lambda j: resim(tr[j]["h0"], 0.5, g, 0.0, 30.0)[0])]
    if key == "cf_speed":
        hw = o["r"] * (0.87 if o["shape"] == "hexagon" else 1.0)
        x0, x1 = render.object_centers(scene, 0.0)[i][0], render.object_centers(scene, 1.0 / RATE)[i][0]
        vx = (x1 - x0) * RATE
        return resim_walls(x0, vx * p["mult"], scene["wall_l"] + hw, scene["wall_r"] - hw, d)
    return None


def _e_of(scene, o):
    return scene["pads"][o["lane"]]["e"] if scene["scenario"] == "pads" else o["e"]
