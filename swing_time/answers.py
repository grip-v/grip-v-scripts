"""Independent answers for swing_time (used only by the validator and tests).

Nothing here reads the generator's route or timing tables. Swings are read
off the rendered timeline: render.state(scene, f) is what frame f draws (the
particle's position and the post the rope is tied to, if any), so a swing
is a run of frames with a rope, its pivot is the post at the rope's anchor,
its direction and size come from adding up the particle's per-frame turn
around that post, its rope length from the particle's distance to the post,
its landing square from where the particle comes to rest, and its times from
the frame numbers through the on-screen clock.

Counterfactual routes are recomputed by a separate solver: legality with
exact integer cross products in doubled coordinates (sim.py uses float
angles) and Bellman-Ford relaxation (sim.py uses Dijkstra).
"""
from __future__ import annotations
import json
import math

import render

FPS = 24
QUADS = ("up and to the left of it", "up and to the right of it",
         "down and to the left of it", "down and to the right of it")
DIRS = ("clockwise, less than a quarter turn", "clockwise, more than a quarter turn",
        "counter-clockwise, less than a quarter turn", "counter-clockwise, more than a quarter turn")
NO_ROUTE = "no route"
NONE_TWICE = "no post is used twice"
_TRACKS: dict = {}      # caches keyed by the full scene JSON, so nothing carries over between scenes
_ROUTES: dict = {}


# ------------------------------------------------------------------ names / formats (shared with questions.py)
def post_name(scene, colour):
    shape = next(p["shape"] for p in scene["posts"] if p["colour"] == colour)
    return f"the {colour} {shape} post"


def fmt_len(v):
    return NO_ROUTE if v is None or v == math.inf else f"{v:.2f}"


def fmt_secs(v):
    return f"{v:.1f} s"


def fmt_seq(colours):
    return " → ".join(colours)


# ------------------------------------------------------------------ the rendered timeline
def track(scene):
    """[{pivot colour, start, end, rope, turn (signed rad), land square, land pos}] from the frames."""
    key = json.dumps(scene, sort_keys=True)
    if key not in _TRACKS:
        if len(_TRACKS) > 64:
            _TRACKS.clear()
        _TRACKS[key] = _track(scene)
    return _TRACKS[key]


def _track(scene):
    at = {tuple(p["xy"]): p["colour"] for p in scene["posts"]}
    out, cur = [], None
    for f in range(render.n_frames(scene)):
        s = render.state(scene, f)
        pos = s["pos"]
        if s["pivot"] is not None:
            px, py = s["pivot"]
            if cur is None or (px, py) != cur["xy"]:
                if cur is not None:
                    raise ValueError("rope moved to another post without a landing")
                cur = {"xy": (px, py), "pivot": at[(px, py)], "start": f,
                       "rope": math.hypot(pos[0] - px, pos[1] - py), "turn": 0.0, "last": pos}
            else:
                cur["turn"] += _turn(cur["xy"], cur["last"], pos)
                cur["last"] = pos
        elif cur is not None:
            cur["turn"] += _turn(cur["xy"], cur["last"], pos)
            cur.update(end=f, land_pos=pos, land=(math.floor(pos[0]), math.floor(pos[1])))
            del cur["last"]
            out.append(cur)
            cur = None
    return out


def _turn(c, a, b):
    t = math.atan2(b[1] - c[1], b[0] - c[0]) - math.atan2(a[1] - c[1], a[0] - c[0])
    return (t + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------ exact solver (doubled integer coordinates)
def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _in_wedge(a, b, w):
    """w (non-zero) inside the closed wedge swept counter-clockwise from a to b (|angle| <= pi)."""
    if _cross(a, b) > 0:
        return _cross(a, w) >= 0 and _cross(w, b) >= 0
    return _cross(a, w) >= 0                     # half turn: the closed left half-plane of a


def _legal(src, dst, p, posts, W, H, cap2x4):
    P = (2 * p[0], 2 * p[1])
    u = (2 * src[0] + 1 - P[0], 2 * src[1] + 1 - P[1])
    v = (2 * dst[0] + 1 - P[0], 2 * dst[1] + 1 - P[1])
    R2 = u[0] ** 2 + u[1] ** 2
    if R2 != v[0] ** 2 + v[1] ** 2 or R2 > cap2x4:
        return 0
    c = _cross(u, v)
    for d in ((1, -1) if c == 0 else ((1,) if c > 0 else (-1,))):
        a, b = (u, v) if d == 1 else (v, u)
        ok = True
        for e, room in (((1, 0), 2 * W - P[0]), ((-1, 0), P[0]), ((0, 1), 2 * H - P[1]), ((0, -1), P[1])):
            if _in_wedge(a, b, e) and R2 > room * room:
                ok = False
        for o in posts:
            if o != p:
                w = (2 * o[0] - P[0], 2 * o[1] - P[1])
                if w[0] ** 2 + w[1] ** 2 <= R2 and _in_wedge(a, b, w):
                    ok = False
        if ok:
            return d
    return 0


def best_route(scene, drop=None, cap=None, pits=True):
    """(cost, [(post colour, landing square)]) of the cheapest route, or (None, None)."""
    key = (json.dumps(scene, sort_keys=True), drop, cap, pits)
    if key not in _ROUTES:
        if len(_ROUTES) > 256:
            _ROUTES.clear()
        _ROUTES[key] = _best_route(scene, drop, cap, pits)
    return _ROUTES[key]


def _best_route(scene, drop, cap, pits):
    W, H = scene["w"], scene["h"]
    cap = scene["cap"] if cap is None else cap
    cap2x4 = math.floor(4 * cap * cap + 1e-9)
    posts = {tuple(p["xy"]): p["colour"] for p in scene["posts"] if p["colour"] != drop}
    holes = {tuple(x) for x in scene["pits"]} if pits else set()
    squares = [(x, y) for x in range(W) for y in range(H) if (x, y) not in holes]
    edges = []
    for p in posts:
        rings = {}
        for a in squares:
            rings.setdefault((2 * a[0] + 1 - 2 * p[0]) ** 2 + (2 * a[1] + 1 - 2 * p[1]) ** 2, []).append(a)
        for R2, ring in rings.items():
            if R2 > cap2x4:
                continue
            for a in ring:
                for b in ring:
                    if a != b and _legal(a, b, p, list(posts), W, H, cap2x4):
                        edges.append((a, b, p, math.sqrt(R2) / 2))
    s, g = tuple(scene["start"]), tuple(scene["goal"])
    dist, pred = {q: math.inf for q in squares}, {}
    dist[s] = 0.0
    for _ in range(len(squares)):
        changed = False
        for a, b, p, r in edges:
            if dist[a] + r < dist[b] - 1e-9:
                dist[b] = dist[a] + r
                pred[b] = (a, p)
                changed = True
        if not changed:
            break
    if dist.get(g, math.inf) == math.inf:
        return None, None
    route, q = [], g
    while q != s:
        a, p = pred[q]
        route.append((posts[p], q))
        q = a
    return dist[g], route[::-1]


# ------------------------------------------------------------------ answers
def answer(key: str, p: dict, scene: dict):
    tr = track(scene)
    piv = [s["pivot"] for s in tr]

    def one(colour):
        hits = [s for s in tr if s["pivot"] == colour]
        return hits[0] if len(hits) == 1 else None

    if key == "n_swings":
        return len(tr)
    if key == "n_posts_used":
        return len(set(piv))
    if key == "n_posts_total":
        return len(scene["posts"])
    if key == "n_pits":
        return len(scene["pits"])
    if key == "first_pivot":
        return post_name(scene, piv[0])
    if key == "last_pivot":
        return post_name(scene, piv[-1])
    if key == "pivot_of_swing":
        return post_name(scene, piv[p["swing"] - 1])
    if key == "swing_direction":
        s = one(p["post"])
        big = abs(s["turn"]) > math.pi / 2
        return DIRS[(0 if s["turn"] < 0 else 2) + (1 if big else 0)]
    if key == "next_pivot":
        i = piv.index(p["post"])
        return post_name(scene, piv[i + 1])
    if key == "land_quadrant":
        s = one(p["post"])
        dx, dy = s["land_pos"][0] - s["xy"][0], s["land_pos"][1] - s["xy"][1]
        return QUADS[(0 if dy > 0 else 2) + (1 if dx > 0 else 0)]
    if key == "earliest_of":
        first = {c: next((i for i, x in enumerate(piv) if x == c), math.inf) for c in p["posts"]}
        return post_name(scene, min(p["posts"], key=lambda c: first[c]))
    if key == "longest_rope":
        return fmt_len(max(s["rope"] for s in tr))
    if key == "rope_of_post":
        return fmt_len(one(p["post"])["rope"])
    if key == "total_rope":
        return fmt_len(sum(s["rope"] for s in tr))
    if key == "swing_duration":
        s = one(p["post"])
        return fmt_secs(round((s["end"] - s["start"]) / FPS, 1))
    if key == "land_time":
        return render.clock_text(tr[-1]["end"])
    if key == "first_attach_time":
        return render.clock_text(tr[0]["start"])
    if key == "pivot_sequence":
        return fmt_seq(piv)
    if key == "pivot_reused":
        twice = sorted({c for c in piv if piv.count(c) > 1})
        return post_name(scene, twice[0]) if twice else NONE_TWICE
    if key == "n_clockwise":
        return sum(1 for s in tr if s["turn"] < 0)
    if key == "n_long_ropes":
        return sum(1 for s in tr if s["rope"] > p["over"])
    if key == "longest_swing_post":
        return post_name(scene, max(tr, key=lambda s: s["end"] - s["start"])["pivot"])
    if key == "cf_remove_post":
        return fmt_len(best_route(scene, drop=p["post"])[0])
    if key == "cf_remove_n_swings":
        cost, route = best_route(scene, drop=p["post"])
        return NO_ROUTE if route is None else len(route)
    if key == "cf_remove_first_pivot":
        return post_name(scene, best_route(scene, drop=p["post"])[1][0][0])
    if key == "cf_short_rope":
        return fmt_len(best_route(scene, cap=p["cap"])[0])
    if key == "cf_no_pits":
        return fmt_len(best_route(scene, pits=False)[0])
    return None


# ------------------------------------------------------------------ wrong-option labels
# Every wrong option is tagged with the mistake that produces it. One rule set
# (label_options) is applied twice: by the generator to its own route data and by
# the validator to `facts_from_track` (what the frames show), so a tag that doesn't
# match the video is caught like a wrong answer.
MISTAKES = {
    "off_by_one": "a count that is 1 too high or too low",
    "off_by_two": "a count that is 2 too high or too low",
    "off_by_three_or_more": "a count that is 3 or more off",
    "counted_swings_not_posts": "the number of swings instead of different posts",
    "counted_only_used_posts": "only the posts the rope touched",
    "counted_counter_clockwise": "the counter-clockwise swings instead",
    "counted_short_ropes": "the swings at or under the length instead",
    "original_route_swing_count": "the swing count before the post was removed",
    "claims_no_route": "says there is no route when there is one",
    "claims_a_route": "gives a route when none exists",
    "pivot_of_adjacent_swing": "the post of the swing just before or after",
    "pivot_of_another_swing": "a post used on a different swing",
    "post_never_used": "a post the rope never touches",
    "pivot_before_not_after": "the post of the swing before, not after",
    "attached_later": "a post the rope reaches later",
    "shorter_swing": "a post whose swing is shorter",
    "original_first_pivot": "the first post of the route before the change",
    "post_from_original_route": "a post from the route before the change",
    "other_post": "another post",
    "claims_no_post_is_reused": "says no post is used twice",
    "post_used_only_once": "a post the rope touches only once",
    "wrong_direction": "right size of turn, wrong direction",
    "wrong_turn_size": "right direction, wrong size of turn",
    "wrong_direction_and_size": "wrong direction and wrong size of turn",
    "mirrored_left_right": "left and right swapped",
    "mirrored_up_down": "up and down swapped",
    "opposite_side": "both left/right and up/down swapped",
    "shortest_rope": "the shortest rope instead of the longest",
    "rope_of_adjacent_swing": "the rope of the swing just before or after",
    "rope_of_another_swing": "the rope of a different swing",
    "length_no_swing_uses": "a possible rope length that no swing uses",
    "left_out_one_swing": "the total with one swing left out",
    "counted_one_swing_twice": "the total with one swing counted twice",
    "misread_one_rope": "the total with one rope misread as another possible length",
    "other_total": "another total",
    "cost_before_the_change": "the cost before the change",
    "time_of_another_event": "the clock at a different swing's start or landing",
    "duration_of_another_swing": "how long a different swing lasts",
    "nearby_time": "a nearby clock reading",
    "one_swing_left_out": "the sequence with one swing left out",
    "one_extra_swing": "the sequence with an extra swing",
    "reversed_order": "the right posts in reverse order",
    "two_neighbours_swapped": "two neighbouring swings swapped",
    "one_post_wrong": "one post replaced by another",
    "other_sequence": "another sequence",
}


def _ropes_possible(limit=6.0):
    out = set()
    for a in range(1, 14, 2):
        for b in range(a, 14, 2):
            if math.hypot(a, b) / 2 <= limit:
                out.add(math.hypot(a, b) / 2)
    return sorted(out)


def facts_from_track(scene):
    """The route as the frames show it (see track)."""
    tr = track(scene)
    return {"piv": [s["pivot"] for s in tr], "rope": [s["rope"] for s in tr], "turn": [s["turn"] for s in tr],
            "frames": [(s["start"], s["end"]) for s in tr]}


def _same(a, b):
    return abs(a - b) < 0.005


def _post_of(scene, text):
    return next((p["colour"] for p in scene["posts"] if post_name(scene, p["colour"]) == text), None)


def _route_post(i, cp, piv):
    """Tag for post option cp when the right answer is the pivot of swing i (0-based)."""
    if cp in (piv[i - 1] if i > 0 else None, piv[i + 1] if i + 1 < len(piv) else None):
        return "pivot_of_adjacent_swing"
    return "pivot_of_another_swing" if cp in piv else "post_never_used"


def _count(v, gt):
    d = abs(v - gt)
    return "off_by_one" if d == 1 else "off_by_two" if d == 2 else "off_by_three_or_more"


def _classify(key, p, o, gt, F, scene):
    piv, rope, frames = F["piv"], F["rope"], F["frames"]
    n = len(piv)
    cost = sum(rope)
    if key in ("n_swings", "n_posts_used", "n_posts_total", "n_pits", "n_clockwise", "n_long_ropes"):
        v, g = int(o), int(gt)
        if key == "n_posts_used" and v == n:
            return "counted_swings_not_posts"
        if key == "n_posts_total" and v == len(set(piv)):
            return "counted_only_used_posts"
        if key == "n_clockwise" and v == n - g:
            return "counted_counter_clockwise"
        if key == "n_long_ropes" and v == n - g:
            return "counted_short_ropes"
        return _count(v, g)
    if key == "cf_remove_n_swings":
        if o == NO_ROUTE:
            return "claims_no_route"
        if int(o) == n:
            return "original_route_swing_count"
        return "claims_a_route" if gt == NO_ROUTE else _count(int(o), int(gt))
    if key in ("cf_remove_post", "cf_short_rope", "cf_no_pits"):
        if o == NO_ROUTE:
            return "claims_no_route"
        if _same(float(o), cost):
            return "cost_before_the_change"
        return "claims_a_route" if gt == NO_ROUTE else "other_total"
    if key in ("first_pivot", "last_pivot", "pivot_of_swing", "next_pivot", "earliest_of", "longest_swing_post",
               "cf_remove_first_pivot", "pivot_reused"):
        if key == "pivot_reused" and o == NONE_TWICE:
            return "claims_no_post_is_reused"
        cp = _post_of(scene, o)
        if key == "first_pivot":
            return _route_post(0, cp, piv)
        if key == "last_pivot":
            return _route_post(n - 1, cp, piv)
        if key == "pivot_of_swing":
            return _route_post(p["swing"] - 1, cp, piv)
        if key == "next_pivot":
            i = piv.index(p["post"])
            if i > 0 and cp == piv[i - 1]:
                return "pivot_before_not_after"
            return _route_post(i + 1, cp, piv)
        if key == "earliest_of":
            return "attached_later" if cp in piv else "post_never_used"
        if key == "longest_swing_post":
            return "shorter_swing" if cp in piv else "post_never_used"
        if key == "cf_remove_first_pivot":
            return "original_first_pivot" if cp == piv[0] else \
                "post_from_original_route" if cp in piv else "other_post"
        return "post_used_only_once" if piv.count(cp) == 1 else "post_never_used"
    if key == "swing_direction":
        cw_o, cw_g = o.startswith("clockwise"), gt.startswith("clockwise")
        big_o, big_g = "more than" in o, "more than" in gt
        return "wrong_turn_size" if cw_o == cw_g else "wrong_direction" if big_o == big_g else \
            "wrong_direction_and_size"
    if key == "land_quadrant":
        up_o, up_g = o.startswith("up"), gt.startswith("up")
        return "mirrored_left_right" if up_o == up_g else \
            "mirrored_up_down" if ("left" in o) == ("left" in gt) else "opposite_side"
    if key in ("longest_rope", "rope_of_post"):
        v = float(o)
        if key == "longest_rope":
            if _same(v, min(rope)):
                return "shortest_rope"
            return "rope_of_another_swing" if any(_same(v, r) for r in rope) else "length_no_swing_uses"
        i = piv.index(p["post"])
        near = [rope[j] for j in (i - 1, i + 1) if 0 <= j < n]
        if any(_same(v, r) for r in near):
            return "rope_of_adjacent_swing"
        return "rope_of_another_swing" if any(_same(v, r) for r in rope) else "length_no_swing_uses"
    if key == "total_rope":
        v = float(o)
        if any(_same(v, cost - r) for r in rope):
            return "left_out_one_swing"
        if any(_same(v, cost + r) for r in rope):
            return "counted_one_swing_twice"
        if any(_same(v, cost - r + q) for r in rope for q in _ropes_possible()):
            return "misread_one_rope"
        return "other_total"
    if key in ("land_time", "first_attach_time"):
        own = frames[-1][1] if key == "land_time" else frames[0][0]
        others = [f for se in frames for f in se if f != own]
        return "time_of_another_event" if o in {render.clock_text(f) for f in others} else "nearby_time"
    if key == "swing_duration":
        i = piv.index(p["post"])
        others = {fmt_secs(round((e - s) / FPS, 1)) for j, (s, e) in enumerate(frames) if j != i}
        return "duration_of_another_swing" if o in others else "nearby_time"
    if key == "pivot_sequence":
        alt = o.split(" → ")
        if len(alt) == n - 1 and any(alt == piv[:j] + piv[j + 1:] for j in range(n)):
            return "one_swing_left_out"
        if len(alt) == n + 1 and any(alt[:j] + alt[j + 1:] == piv for j in range(n + 1)):
            return "one_extra_swing"
        if alt == piv[::-1]:
            return "reversed_order"
        if len(alt) == n:
            diff = [j for j in range(n) if alt[j] != piv[j]]
            if len(diff) == 2 and diff[1] == diff[0] + 1 and alt[diff[0]] == piv[diff[1]] \
                    and alt[diff[1]] == piv[diff[0]]:
                return "two_neighbours_swapped"
            if len(diff) == 1:
                return "one_post_wrong"
        return "other_sequence"
    raise KeyError(key)


def label_options(key, params, options, answer_text, facts, scene):
    """[None for the correct option, else the MISTAKES tag of that wrong option], aligned with options."""
    return [None if o == answer_text else _classify(key, params, o, answer_text, facts, scene) for o in options]


def option_errors(q, scene):
    """Independent labels for a stored question, from the frames."""
    return label_options(q["derivation_key"], q["params"], q["options"], q["answer_text"],
                         facts_from_track(scene), scene)
