"""theory_of_mind_belief: a watcher (eye icon) looks away for one or two
windows while tracked shapes change (a counter ticks, a shape morphs or a
colour swaps). The camera always shows the truth; the watcher's BELIEF about
a shape is frozen at whatever was true the last time it was watching.

Scene = JSON only. Each actor has an attribute (number/shape/color), a start
value and a list of changes {t, delta} (number) or {t, to} (shape/color).
`cf_shift_s` is the shift used by the window-shift counterfactual."""
from __future__ import annotations
import random

SHAPES = ["circle", "square", "triangle", "pentagon", "hexagon", "star", "ellipse"]
PALETTE = {"red": "#E4553F", "blue": "#3F8FE4", "green": "#4FBF6B", "yellow": "#E4C43F",
           "purple": "#B85FE4", "teal": "#3FD4C4", "orange": "#E4823F", "pink": "#E45FA8"}
MARGIN = 0.2


def _away(t, windows):
    return any(w["start_s"] <= t <= w["end_s"] for w in windows)


def value_at(actor, time):
    v = actor["start"]
    for c in actor["changes"]:
        if c["t"] < time:
            v = max(0, v + c["delta"]) if actor["attr"] == "number" else c["to"]
    return v


def belief_at(actor, windows, t):
    for w in windows:
        if w["start_s"] <= t <= w["end_s"]:
            return value_at(actor, w["start_s"])
    return value_at(actor, t)


def _shifted(windows, d):
    return [{"start_s": round(w["start_s"] + d, 2), "end_s": round(w["end_s"] + d, 2)} for w in windows]


def _far_from_edges(t, windows):
    return all(abs(t - w[k]) >= MARGIN for w in windows for k in ("start_s", "end_s"))


def _windows(rng, dur):
    if rng.random() < 0.5:
        s = round(rng.uniform(0.25, 0.5) * dur, 2)
        return [{"start_s": s, "end_s": round(s + rng.uniform(1.5, 3.0), 2)}]
    s1 = round(rng.uniform(0.12, 0.28) * dur, 2)
    e1 = round(s1 + rng.uniform(1.2, 2.2), 2)
    s2 = round(rng.uniform(0.5, 0.65) * dur, 2)
    e2 = round(s2 + rng.uniform(1.5, 2.5), 2)
    return [{"start_s": s1, "end_s": e1}, {"start_s": s2, "end_s": e2}]


def _actor(rng, i, dur):
    attr = rng.choice(["number", "number", "shape", "color"])
    a = {"index": i, "attr": attr, "shape": rng.choice(SHAPES), "color": rng.choice(list(PALETTE)),
         "changes": []}
    if attr == "number":
        a["start"] = rng.randint(1, 4)
    else:
        a["start"] = a[attr]
    n = rng.randint(1, 3) if attr == "number" else rng.randint(1, 2)
    times = []
    while len(times) < n:
        t = round(rng.uniform(0.7, dur - 0.7), 2)
        if all(abs(t - u) >= 0.6 for u in times):
            times.append(t)
    cur = a["start"]
    for t in sorted(times):
        if attr == "number":
            d = rng.choice([-1, 1])
            if cur + d < 0:
                d = 1
            cur += d
            a["changes"].append({"t": t, "delta": d})
        else:
            pool = SHAPES if attr == "shape" else list(PALETTE)
            cur = rng.choice([x for x in pool if x != cur])
            a["changes"].append({"t": t, "to": cur})
    return a


def build(rng: random.Random) -> dict:
    for _ in range(500):
        dur = round(rng.uniform(8.0, 12.0), 1)
        windows = _windows(rng, dur)
        if windows[-1]["end_s"] > dur - 0.8 or (len(windows) == 2 and windows[1]["start_s"] - windows[0]["end_s"] < 0.8):
            continue
        shift = rng.choice([-1, 1]) * round(rng.uniform(0.8, 1.6), 1)
        sw = _shifted(windows, shift)
        if sw[0]["start_s"] < 0.3 or sw[-1]["end_s"] > dur - 0.3:
            continue
        actors = [_actor(rng, i, dur) for i in range(rng.choice([2, 3]))]
        query = round(rng.uniform(0.55 * dur, dur - 0.4), 2)
        times = [c["t"] for a in actors for c in a["changes"]] + [query]
        if any(abs(query - c["t"]) < 0.4 for a in actors for c in a["changes"]):
            continue
        if not all(_far_from_edges(t, windows) and _far_from_edges(t, sw) for t in times):
            continue
        unseen = sum(1 for a in actors for c in a["changes"] if _away(c["t"], windows))
        stale = [a for a in actors if belief_at(a, windows, query) != value_at(a, query)]
        if unseen >= 1 and stale:
            return {"duration_s": dur, "windows": windows, "actors": actors, "query_t": query,
                    "cf_shift_s": shift, "palette": PALETTE}
    raise RuntimeError("could not build a valid scene")


def check(scene: dict) -> list[str]:
    errs = []
    ws = scene["windows"]
    if any(w["end_s"] <= w["start_s"] for w in ws) or ws[-1]["end_s"] > scene["duration_s"]:
        errs.append("bad windows")
    for a in scene["actors"]:
        for c in a["changes"]:
            if not _far_from_edges(c["t"], ws):
                errs.append("change too close to a window edge")
    return errs
