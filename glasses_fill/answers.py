"""Independent re-derivation of every glasses_fill answer (the closed loop).

Shares no logic with questions.py or sim's state functions. It rebuilds the
clip frame by frame from what the renderer draws:
  * jug poses come from render.jug_pose (the geometry the frames are drawn
    with); a "tilt" is a run of frames with the jug turned >= 70 degrees, and
    the glass it pours into is the one under its spout tip;
  * glass levels come from its own per-frame formula (stream rate x time,
    capped at what the pour delivered; drain rate x time down to the crack);
  * the amount a tilt delivers is the target glass's level change over it;
  * layers are rebuilt by stacking those amounts and cutting at the crack.
Counterfactuals re-run the jug bookkeeping (each pour gets min(asked, left))
under the change. Returns the answer string the question should have, or
None when the question can't be answered from the clip.
"""
from __future__ import annotations
import functools
import json

import render

FPS = 24
CAP = 100
TILTED = 70.0


def _key(scene):
    return json.dumps(scene, sort_keys=True)


@functools.lru_cache(maxsize=8)
def _track(skey):
    scene = json.loads(skey)
    nf, n = scene["n_frames"], scene["glasses"]["n"]
    cx = scene["glasses"]["cx"]

    def level(g, f):
        t = f / FPS
        v = 0.0
        for p in scene["pours"]:
            if p["glass"] == g and p["got_ml"] and t > p["t_pour"]:
                v += min(p["got_ml"], (t - p["t_pour"]) * p["rate_ml_s"])
        for lk in scene["leaks"]:
            if lk["glass"] == g and t > lk["t_crack"]:
                v -= min(v - lk["hole_ml"] if v > lk["hole_ml"] else 0.0, (t - lk["t_crack"]) * lk["rate_ml_s"])
        return v

    levels = [[level(g, f) for f in range(nf)] for g in range(n)]
    tilts = []                                             # [jug, glass, f0, f1]
    for j in range(len(scene["jugs"])):
        run = None
        for f in range(nf + 1):
            on = f < nf and render.jug_pose(scene, j, f / FPS)[2] >= TILTED
            if on and run is None:
                run = f
            if not on and run is not None:
                x, y, deg = render.jug_pose(scene, j, (run + f - 1) / 2 / FPS)
                tip_x = x + render._rot(render.SPOUT_TIP, deg)[0]
                g = min(range(n), key=lambda k: abs(cx[k] - tip_x))
                tilts.append([j, g, run, f - 1])
                run = None
    tilts.sort(key=lambda e: e[2])
    eps = []
    for j, g, f0, f1 in tilts:
        fe = min(nf - 1, f1 + 3)          # the stream ends at most one frame after the jug starts back
        got = int(round(levels[g][fe] - levels[g][max(0, f0 - 1)]))
        flow = [f for f in range(f0, fe + 1) if f and levels[g][f] > levels[g][f - 1] + 1e-9]
        eps.append({"jug": j, "glass": g, "got": got, "f0": f0, "f1": f1,
                    "flow0": flow[0] if flow else None, "flow1": flow[-1] if flow else None})
    final = [int(round(levels[g][nf - 1])) for g in range(n)]
    # layers: stack what each tilt delivered, then cut at the crack
    stacks = [[] for _ in range(n)]
    for e in eps:
        if e["got"]:
            col = scene["jugs"][e["jug"]]["colour"]
            s = stacks[e["glass"]]
            if s and s[-1][0] == col:
                s[-1][1] += e["got"]
            else:
                s.append([col, e["got"]])
    peak = [sum(m for _, m in s) for s in stacks]
    for lk in scene["leaks"]:
        cut, h = [], 0
        for col, m in stacks[lk["glass"]]:
            keep = min(m, max(0, lk["hole_ml"] - h))
            if keep:
                cut.append([col, keep])
            h += m
        stacks[lk["glass"]] = cut
    return {"scene": scene, "levels": levels, "eps": eps, "final": final, "layers": stacks, "peak": peak}


def _G(g):
    return f"glass {g + 1}"


def _J(scene, j):
    return f"the {scene['jugs'][j]['pattern']} jug"


def _ml(v):
    return f"{v} ml"


def _resim(tr, skip=None, redirect=None):
    """Final levels with pour `skip` removed or pour redirect[0] sent to glass redirect[1]."""
    sc, eps = tr["scene"], tr["eps"]
    left = [jg["start_ml"] for jg in sc["jugs"]]
    lev = [0] * sc["glasses"]["n"]
    for i, e in enumerate(eps):
        if i == skip:
            continue
        g = redirect[1] if redirect and redirect[0] == i else e["glass"]
        got = min(sc["pours"][i]["ask_ml"], left[e["jug"]])
        left[e["jug"]] -= got
        lev[g] += got
    for lk in sc["leaks"]:
        lev[lk["glass"]] = min(lev[lk["glass"]], lk["hole_ml"])
    return lev


def answer(key: str, params: dict, scene: dict):
    tr = _track(_key(scene))
    eps, final, layers, peak = tr["eps"], tr["final"], tr["layers"], tr["peak"]
    n = scene["glasses"]["n"]
    jugs = scene["jugs"]
    leaks = {lk["glass"]: lk for lk in scene["leaks"]}
    p = params

    if key == "n_glasses":
        return n
    if key == "n_jugs":
        return len(jugs)
    if key == "jug_colour":
        return jugs[p["jug"]]["colour"] if jugs[p["jug"]]["start_ml"] >= 40 else None
    if key == "jug_of_colour":
        js = [j for j, jg in enumerate(jugs) if jg["colour"] == p["colour"] and jg["start_ml"] >= 40]
        return _J(scene, js[0]) if len(js) == 1 else None
    if key == "first_jug":
        return _J(scene, eps[0]["jug"])
    if key == "dry_jug":
        dry = [e for e in eps if e["got"] == 0]
        return _J(scene, dry[0]["jug"]) if len(dry) == 1 else None
    if key == "cracked_one":
        return _G(next(iter(leaks))) if len(leaks) == 1 else None
    if key == "n_cracked":
        return len(leaks)
    if key == "most_end":
        order = sorted(range(n), key=lambda g: -final[g])
        if n > 1 and final[order[0]] - final[order[1]] < 20:
            return None
        return _G(order[0])
    if key == "first_target":
        return _G(next(e["glass"] for e in eps if e["jug"] == p["jug"]))
    if key == "bottom_layer":
        return layers[p["glass"]][0][0] if len(layers[p["glass"]]) >= 2 else None
    if key == "top_layer":
        return layers[p["glass"]][-1][0] if len(layers[p["glass"]]) >= 2 else None
    if key == "jug_into":
        js = {e["jug"] for e in eps if e["glass"] == p["glass"] and e["got"]}
        return _J(scene, js.pop()) if len(js) == 1 else None
    if key == "jug_after":
        k = next(i for i, e in enumerate(eps) if e["jug"] == p["jug"])
        return _J(scene, eps[k + 1]["jug"]) if k + 1 < len(eps) else None
    if key == "unused_jug":
        idle = [j for j in range(len(jugs)) if not any(e["jug"] == j for e in eps)
                and all(render.jug_pose(scene, j, f / FPS)[:2] == render.park_pos(j, len(jugs))
                        for f in range(0, scene["n_frames"], 6))]
        return _J(scene, idle[0]) if len(idle) == 1 else None
    if key == "jug_more":
        tot = {}
        for e in eps:
            tot[e["jug"]] = tot.get(e["jug"], 0) + e["got"]
        order = sorted(tot, key=lambda j: -tot[j])
        if len(order) > 1 and tot[order[0]] - tot[order[1]] < 20:
            return None
        return _J(scene, order[0])
    if key == "last_glass":
        wet = [e for e in eps if e["got"]]
        return _G(wet[-1]["glass"]) if wet else None
    if key == "first_glass":
        firsts = {}
        for e in eps:
            if e["got"] and e["glass"] not in firsts:
                firsts[e["glass"]] = e["flow0"]
        got = [g for g in p["cands"] if g in firsts]
        return _G(min(got, key=lambda g: firsts[g])) if got else None
    if key == "ran_dry":
        dry = [e for e in eps if e["got"] and e["flow1"] is not None and e["f1"] - e["flow1"] >= 6]
        return _J(scene, dry[0]["jug"]) if len(dry) == 1 else None
    if key == "level_end":
        return _ml(final[p["glass"]])
    if key == "layer_ml":
        vs = [m for c, m in layers[p["glass"]] if c == p["colour"]]
        return _ml(vs[0]) if len(vs) == 1 else None
    if key == "crack_mark":
        return _ml(leaks[p["glass"]]["hole_ml"]) if p["glass"] in leaks else None
    if key == "n_tilts":
        return len(eps)
    if key == "n_empty_end":
        return sum(1 for v in final if v == 0)
    if key == "pour_ml":
        return _ml(eps[p["pour"]]["got"])
    if key == "level_at_clock":
        fr = [f for f in range(scene["n_frames"]) if f * 10 // FPS == p["clock"]]
        vals = {int(round(tr["levels"][p["glass"]][f])) for f in fr}
        return _ml(vals.pop()) if len(vals) == 1 else None
    if key == "pour_clock":
        es = [e for e in eps if e["jug"] == p["jug"] and e["got"]]
        if len(es) != 1:
            return None
        return f"{es[0]['flow0'] * 10 // FPS / 10:.1f} s"
    if key == "total_end":
        return _ml(sum(final))
    if key == "jug_total":
        return _ml(sum(e["got"] for e in eps if e["jug"] == p["jug"]))
    if key == "leak_loss":
        return _ml(peak[p["glass"]] - final[p["glass"]]) if p["glass"] in leaks else None
    if key == "diff_end":
        return _ml(final[p["a"]] - final[p["b"]])
    if key == "n_received":
        return len({e["glass"] for e in eps if e["got"]})
    if key == "before_last":
        es = [e for e in eps if e["glass"] == p["glass"] and e["got"]]
        return _ml(int(round(tr["levels"][p["glass"]][es[-1]["flow0"] - 1])))
    if key == "cf_no_crack":
        return _ml(peak[p["glass"]])
    if key == "cf_no_crack_total":
        return _ml(sum(peak))
    if key == "cf_skip":
        return _ml(_resim(tr, skip=p["pour"])[p["glass"]])
    if key == "cf_redirect":
        return _ml(_resim(tr, redirect=(p["pour"], p["glass"]))[p["glass"]])
    if key == "cf_swap":
        es = [e for e in eps if e["glass"] == p["glass"] and e["got"]]
        return jugs[es[-1]["jug"]]["colour"] if len(es) == 2 else None
    return None
