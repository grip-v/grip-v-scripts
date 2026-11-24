"""Independent answers for overlapping_shapes.

The generator measures everything with exact shapely polygons (areas,
intersections, threshold-crossing times found by bisection). This module
never touches those: it rasterises the outlines the renderer draws into
pixel masks and answers from mask arithmetic and a frame-by-frame replay:

  * overlap = at least SHARED_PX shared pixels between two masks;
  * stack depth = the per-pixel count of covering masks;
  * paper "on top" = an index image painted in the drawing order;
  * presence on the board = the arrival / departure frames replayed frame by
    frame, the same way the renderer fades a shape in or out;
  * a sliding shape = its pose replayed at every frame (24 fps), rasterised,
    and intersected with each still shape's mask; counterfactual trips
    (stopped early, reversed, half speed) are replayed the same way.

It reads only the scene description (shapes, events, paths), never the
generator's derived data."""
from __future__ import annotations
import functools
import json
import math

import numpy as np
from PIL import Image, ImageDraw

import sim

FPS = 24
SHARED_PX = 100
W, H = 1280, 720


def _outline(sh, x, y, rot):
    return sim.outline_at(sh, x, y, rot)


def _mask(pts, box=None):
    """Bool mask of a polygon, full frame or cropped to box (x0, y0, x1, y1)."""
    if box is None:
        box = (0, 0, W, H)
    x0, y0, x1, y1 = box
    im = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(im).polygon([(x - x0, y - y0) for x, y in pts], fill=1)
    return np.asarray(im, dtype=bool)


@functools.lru_cache(maxsize=64)
def _static_masks(key: str):
    shapes = json.loads(key)
    return {i: _mask(_outline(s, s["x"], s["y"], s["rot"])) for i, s in enumerate(shapes) if "path" not in s}


def _masks(scene):
    return _static_masks(json.dumps(scene["shapes"], sort_keys=True))


@functools.lru_cache(maxsize=64)
def _graph_key(key: str):
    m = _static_masks(key)
    idx = sorted(m)
    nb = {i: set() for i in idx}
    for a in idx:
        for b in idx:
            if a < b and np.count_nonzero(m[a] & m[b]) >= SHARED_PX:
                nb[a].add(b)
                nb[b].add(a)
    return nb


def graph(scene, members=None):
    nb = _graph_key(json.dumps(scene["shapes"], sort_keys=True))
    members = set(nb) if members is None else set(members)
    return {i: nb[i] & members for i in sorted(members)}


def n_pairs(nb):
    return sum(len(v) for v in nb.values()) // 2


def _lab(scene, i):
    s = scene["shapes"][i]
    return f"{s['color']} {s['type']}"


# ---------------------------------------------------------- frame replay
def _alpha(scene, i, f):
    """How much of shape i is drawn at frame f (0..1), replaying the event list."""
    kind = scene["scenario"]
    if kind == "glide":
        return 1.0
    A = scene["anim_f"]
    start = None
    for e in scene["events"]:
        if e["shape"] == i:
            start = e["f"]
    if kind == "depart":
        if start is None or f < start:
            return 1.0
        return max(0.0, 1.0 - (f - start + 1) / A)
    if start is None or f < start:
        return 0.0
    return min(1.0, (f - start + 1) / A)


def full_at(scene, f):
    return [i for i in range(len(scene["shapes"])) if _alpha(scene, i, f) >= 1.0]


def _start_frame(scene, i):
    """First frame in which shape i changes (the replay's own scan, not the event list)."""
    a0 = _alpha(scene, i, 0)
    for f in range(1, scene["n_frames"]):
        if _alpha(scene, i, f) != a0:
            return f
    return None


def _changers(scene):
    """Shapes that change during the clip, in the order they start to change."""
    st = [(_start_frame(scene, i), i) for i in range(len(scene["shapes"]))]
    return [i for f, i in sorted(x for x in st if x[0] is not None)]


def _arrivals(nb, order):
    seen, out = set(), []
    for i in order:
        out.append(nb[i] & seen)
        seen.add(i)
    return out


# ---------------------------------------------------------- glide replay
def _pose(sh, f, path=None):
    p = path or sh["path"]
    x0, y0 = p.get("xs", sh["x"]), p.get("ys", sh["y"])
    r0 = p.get("rs", sh["rot"])
    span = p["f1"] - p["f0"]
    u = min(1.0, max(0.0, (f - p["f0"]) / span))
    return x0 + (p["x1"] - x0) * u, y0 + (p["y1"] - y0) * u, r0 + p["spin"] * u * span / FPS


def _bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (max(0, int(math.floor(min(xs))) - 1), max(0, int(math.floor(min(ys))) - 1),
            min(W, int(math.ceil(max(xs))) + 2), min(H, int(math.ceil(max(ys))) + 2))


@functools.lru_cache(maxsize=128)
def _trip(key: str, m: int, variant: str):
    """{static: [frames in which mover m overlaps it]} for a replayed trip.
    variant: 'real', 'reverse', 'half', or 'stop:<frame>'."""
    shapes = json.loads(key)
    sh = shapes[m]
    p = dict(sh["path"])
    n_frames = p.pop("_n")
    if variant == "reverse":
        span = (p["f1"] - p["f0"]) / FPS
        p = {"xs": p["x1"], "ys": p["y1"], "rs": sh["rot"] + p["spin"] * span, "x1": sh["x"], "y1": sh["y"],
             "f0": p["f0"], "f1": p["f1"], "spin": -p["spin"]}
    elif variant == "half":
        p["f1"] = p["f0"] + 2 * (p["f1"] - p["f0"])      # same line, same turn per px, half the speed
        p["spin"] = p["spin"] / 2
    stop = int(variant.split(":")[1]) if variant.startswith("stop:") else None
    smasks = {i: (_bbox(_outline(s, s["x"], s["y"], s["rot"])), s) for i, s in enumerate(shapes) if "path" not in s}
    out = {i: [] for i in smasks}
    for f in range(n_frames):
        ff = min(f, stop) if stop is not None else f
        pts = _outline(sh, *_pose(sh, ff, p))
        mb = _bbox(pts)
        for i, (sb, s) in smasks.items():
            box = (max(mb[0], sb[0]), max(mb[1], sb[1]), min(mb[2], sb[2]), min(mb[3], sb[3]))
            if box[0] >= box[2] or box[1] >= box[3]:
                continue
            a = _mask(pts, box)
            b = _mask(_outline(s, s["x"], s["y"], s["rot"]), box)
            if np.count_nonzero(a & b) >= sim.MIN_OVERLAP:
                out[i].append(f)
    return out


def trip(scene, m, variant="real"):
    shapes = json.loads(json.dumps(scene["shapes"]))
    shapes[m]["path"]["_n"] = scene["n_frames"]
    return _trip(json.dumps(shapes, sort_keys=True), m, variant)


def touched(scene, m, variant="real"):
    """Statics the mover overlaps, ordered by their first overlapping frame."""
    tr = trip(scene, m, variant)
    return sorted((fr[0], i) for i, fr in tr.items() if fr)


# ---------------------------------------------------------------- answers
def answer(key: str, p: dict, scene: dict):
    shapes = scene["shapes"]
    n = len(shapes)
    kind = scene["scenario"]
    st = [i for i, s in enumerate(shapes) if "path" not in s]
    nb = graph(scene)
    order = _changers(scene)

    def pick(pred):
        hits = [i for i in p["cands"] if pred(i)]
        return _lab(scene, hits[0]) if len(hits) == 1 else None

    # ----- L1
    if key == "n_shapes":
        return n
    if key == "n_types":
        return len({s["type"] for s in shapes})
    if key == "n_color":
        return sum(s["color"] == p["color"] for s in shapes)
    if key == "n_style":
        return sum(s["style"] == p["style"] for s in shapes)
    if key == "n_fill":
        return sum(s["fill"] == p["fill"] for s in shapes)
    if key == "n_kind":
        return sum(s["type"] == p["kind"] for s in shapes)
    if key == "first_shape":
        return _lab(scene, order[0])
    if key == "last_shape":
        return _lab(scene, order[-1])
    if key == "first_gone":
        return _lab(scene, order[0])
    if key == "n_left":
        return len(full_at(scene, scene["n_frames"] - 1))
    if key == "mover_which":
        return _lab(scene, p["m"])
    if key == "mover_dir":
        sh = shapes[p["m"]]
        x0, y0, _ = _pose(sh, 0)
        x1, y1, _ = _pose(sh, scene["n_frames"] - 1)
        ang = math.atan2(-(y1 - y0), x1 - x0)                      # maths angle, y up
        names = ["right", "up and to the right", "up", "up and to the left", "left", "down and to the left",
                 "down", "down and to the right"]
        return names[int(round(ang / (math.pi / 4))) % 8]
    # ----- L2
    if key == "overlap_which":
        return pick(lambda i: i in nb[p["x"]])
    if key == "overlap_none":
        return pick(lambda i: not nb[i])
    if key == "earlier4":
        return _lab(scene, min(p["cands"], key=order.index))
    if key == "earlier2":
        return _lab(scene, min(p["pair"], key=order.index))
    if key == "arrival_partner":
        (j,) = _arrivals(nb, order)[order.index(p["x"])]
        return _lab(scene, j)
    if key in ("on_top", "cf_reverse_top"):
        z = order if key == "on_top" else order[::-1]
        return pick(lambda i: _top(scene, z, p["x"], i) == i)
    if key == "gone_after":
        return _lab(scene, order[order.index(p["x"]) + 1])
    if key == "passes_over":
        hit = {i for _, i in touched(scene, p["m"])}
        return pick(lambda i: i in hit)
    if key == "first_touch":
        return _lab(scene, touched(scene, p["m"])[0][1])
    # ----- L3
    if key == "partners":
        return len(nb[p["x"]])
    if key == "n_pairs":
        return n_pairs(nb)
    if key == "max_depth":
        m = _masks(scene)
        stack = np.sum(np.stack([m[i] for i in st]), axis=0)
        return max(d for d in range(1, int(stack.max()) + 1) if np.count_nonzero(stack >= d) >= SHARED_PX)
    if key == "area_ratio":
        m = _masks(scene)
        areas = [np.count_nonzero(m[i]) for i in st]
        return f"{round(max(areas) / min(areas) * 2) / 2:.1f}"
    if key == "count_at_T":
        return len(full_at(scene, p["f"]))
    if key == "time_of":
        return f"{int(round(_start_frame(scene, p['x']) / FPS)):g} s"
    if key == "n_touched":
        return len(touched(scene, p["m"]))
    if key == "overlap_at_T":
        return sum(1 for fr in trip(scene, p["m"]).values() if p["f"] in fr)
    # ----- L4
    if key == "isolated_arrivals":
        return sum(1 for a in _arrivals(nb, order)[1:] if not a)
    if key == "most_new":
        arr = _arrivals(nb, order)
        return _lab(scene, max(order, key=lambda i: len(arr[order.index(i)])))
    if key == "arrival_count":
        return len(_arrivals(nb, order)[order.index(p["x"])])
    if key == "pairs_at_T":
        return n_pairs(graph(scene, full_at(scene, p["f"])))
    if key == "pairs_before_last":
        return n_pairs(graph(scene, full_at(scene, _start_frame(scene, order[-1]) - 1)))
    if key == "partners_when_gone":
        f = _start_frame(scene, p["x"]) - 1
        return len(graph(scene, full_at(scene, f))[p["x"]])
    if key == "lost_pairs":
        return n_pairs(graph(scene, full_at(scene, 0))) - n_pairs(graph(scene, full_at(scene, scene["n_frames"] - 1)))
    if key == "newly_alone":
        end = full_at(scene, scene["n_frames"] - 1)
        g0, g1 = graph(scene, full_at(scene, 0)), graph(scene, end)
        return sum(1 for i in end if g0[i] and not g1[i])
    if key == "last_touch":
        return _lab(scene, touched(scene, p["m"])[-1][1])
    if key == "touch_order":
        seq = [i for _, i in touched(scene, p["m"]) if i in p["ids"]]
        return ", then ".join(f"the {_lab(scene, i)}" for i in seq)
    if key == "max_simul":
        tr = trip(scene, p["m"])
        return max(sum(1 for fr in tr.values() if f in fr) for f in range(scene["n_frames"]))
    if key == "touches_between":
        return sum(1 for f, _ in touched(scene, p["m"]) if p["fa"] < f <= p["fb"])
    if key == "longest_overlap":
        tr = trip(scene, p["m"])
        return _lab(scene, max((len(fr), i) for i, fr in tr.items() if fr)[1])
    if key == "which_at_T":
        now = [i for i, fr in trip(scene, p["m"]).items() if p["f"] in fr]
        return _lab(scene, now[0]) if len(now) == 1 else None
    if key == "more_touched":
        a, b = [i for i, s in enumerate(shapes) if "path" in s]
        ca, cb = len(touched(scene, a)), len(touched(scene, b))
        return _lab(scene, a) if ca > cb else _lab(scene, b) if cb > ca else "they overlap the same number"
    # ----- L5
    if key == "cf_remove_pairs":
        return n_pairs(graph(scene, [i for i in st if i != p["x"]]))
    if key == "cf_remove_isolated":
        g = graph(scene, [i for i in st if i != p["x"]])
        return sum(1 for v in g.values() if not v)
    if key == "cf_reverse":
        return sum(1 for a in _arrivals(nb, order[::-1])[1:] if not a)
    if key == "cf_skip_first":
        return sum(1 for a in _arrivals(nb, order[1:])[1:] if not a)
    if key in ("cf_keep", "cf_keep_alone"):
        keep = set(full_at(scene, scene["n_frames"] - 1)) | {p["x"]}
        g = graph(scene, keep)
        return n_pairs(g) if key == "cf_keep" else sum(1 for v in g.values() if not v)
    if key == "cf_stop":
        return len(touched(scene, p["m"], f"stop:{p['f']}"))
    if key == "cf_reverse_path":
        return _lab(scene, touched(scene, p["m"], "reverse")[0][1])
    if key == "cf_half_speed":
        return len(touched(scene, p["m"], "half"))
    return None


def _top(scene, z, a, b):
    """Paper: which of a, b shows in their shared pixels when the two are painted in z-order."""
    m = _masks(scene)
    idx = np.full((H, W), -1, dtype=np.int16)
    for i in z:
        if i in (a, b):
            idx[m[i]] = i
    shared = m[a] & m[b]
    if np.count_nonzero(shared) < SHARED_PX:
        return None
    va, vb = np.count_nonzero(idx[shared] == a), np.count_nonzero(idx[shared] == b)
    return a if va > vb else b
