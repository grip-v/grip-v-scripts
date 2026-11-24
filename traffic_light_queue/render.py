"""PIL renderer for traffic_light_queue (1280x720, 24 fps).

Roads with lane markings, a white stop line and a lamp (red / green disc on a
pole) for every light, and the cars: rounded boxes in their colour whose
length shows the body (car / van / truck / bus), with a dark windscreen at
the front. Cars appear at the start of their lane (the left edge, or the top
for the vertical road) and leave at the far end. The clock (top-left box) is
the only text. Pure function of (scene, frame).
"""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.video5 import font  # noqa: E402
import sim  # noqa: E402

FPS = sim.FPS
SIZE = (1280, 720)
CLOCK_BOX = sim.CLOCK_BOX
_SIM: dict = {}


def duration(scene) -> float:
    return sim.duration(scene)


def n_frames(scene) -> int:
    return sim.n_frames(scene)


def clock_text(frame: int) -> str:
    return f"{sim.clock_tenths(frame) / 10:.1f} s"


def simulated(scene):
    key = json.dumps(scene, sort_keys=True)
    if key not in _SIM:
        if len(_SIM) > 16:
            _SIM.clear()
        _SIM[key] = sim.simulate(scene)
    return _SIM[key]


def car_box(scene, c, s, clip=True):
    """(x0, y0, x1, y1) of a car whose front is s along its lane (clipped to the road unless clip=False)."""
    lane = scene["lanes"][c["lane"]]
    w = scene["style"]["lane_w"] * 0.62
    fx, fy = sim.lane_xy(lane, s)
    bx, by = sim.lane_xy(lane, s - c["len"])
    (ax, ay), (ex, ey) = lane["a"], lane["b"]          # clipped to the road: cars come out of the gate
    if not clip:
        return (bx, fy - w / 2, fx, fy + w / 2) if lane["dir"] == "h" else (fx - w / 2, by, fx + w / 2, fy)
    if lane["dir"] == "h":
        return (max(bx, ax), fy - w / 2, min(fx, ex), fy + w / 2)
    return (fx - w / 2, max(by, ay), fx + w / 2, min(fy, ey))


def lamp_xy(scene, li):
    """Centre of light li's lamp (beside its stop line, off the road)."""
    L = scene["lights"][li]
    lane = next(l for l in scene["lanes"] if li in l["lights"])
    x, y = sim.lane_xy(lane, L["s"])
    off = scene["style"]["lane_w"] / 2 + scene["style"]["lamp_px"] + 10
    side = 1 if scene["style"]["pole"] == "right" else -1
    if lane["dir"] == "h":
        return x, y - off if side < 0 else y + off
    return x + off * side, y


def state(scene, f: int) -> dict:
    """What frame f draws: cars [{id, box}] (only cars on the road), lights [{red}]."""
    t = f / FPS
    res = simulated(scene)
    cars = []
    for c in scene["cars"]:
        s = res["pos"][c["id"]][f]
        if s is not None:
            cars.append({"id": c["id"], "box": car_box(scene, c, s)})
    return {"cars": cars, "lights": [{"red": sim.light_red(L, t)} for L in scene["lights"]]}


@functools.lru_cache(maxsize=8)
def _static(key: str) -> Image.Image:
    scene = json.loads(key)
    st = scene["style"]
    img = Image.new("RGB", SIZE, tuple(st["bg"]))
    d = ImageDraw.Draw(img)
    W = st["lane_w"]
    road, mark = tuple(st["road"]), tuple(st["mark"])
    for lane in scene["lanes"]:
        (x0, y0), (x1, y1) = lane["a"], lane["b"]
        if lane["dir"] == "h":
            box = [x0, y0 - W / 2, x1, y0 + W / 2]
        else:
            box = [x0 - W / 2, y0, x0 + W / 2, y1]
        if st["verge"] == "kerb":
            d.rectangle([box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4], fill=(200, 200, 196))
        d.rectangle(box, fill=road)
        if st["dash"] != "none":
            if lane["dir"] == "h":
                for k in range(int(x0), int(x1), 40):
                    if st["dash"] == "solid" or (k // 40) % 2 == 0:
                        d.line([(k, y0 - W / 2 + 3), (k + 40, y0 - W / 2 + 3)], fill=mark, width=3)
                        d.line([(k, y0 + W / 2 - 3), (k + 40, y0 + W / 2 - 3)], fill=mark, width=3)
            else:
                for k in range(int(y0), int(y1), 40):
                    if st["dash"] == "solid" or (k // 40) % 2 == 0:
                        d.line([(x0 - W / 2 + 3, k), (x0 - W / 2 + 3, k + 40)], fill=mark, width=3)
                        d.line([(x0 + W / 2 - 3, k), (x0 + W / 2 - 3, k + 40)], fill=mark, width=3)
        # the gate the cars come out of
        if lane["dir"] == "h":
            d.rectangle([x0 - 8, y0 - W / 2 - 6, x0, y0 + W / 2 + 6], fill=(30, 30, 34))
        else:
            d.rectangle([x0 - W / 2 - 6, y0 - 8, x0 + W / 2 + 6, y0], fill=(30, 30, 34))
    for li, L in enumerate(scene["lights"]):
        lane = next(l for l in scene["lanes"] if li in l["lights"])
        x, y = sim.lane_xy(lane, L["s"])
        if lane["dir"] == "h":
            d.line([(x, y - W / 2), (x, y + W / 2)], fill=mark, width=5)
        else:
            d.line([(x - W / 2, y), (x + W / 2, y)], fill=mark, width=5)
        lx, ly = lamp_xy(scene, li)
        r = st["lamp_px"]
        d.rectangle([lx - r - 4, ly - r - 4, lx + r + 4, ly + r + 4], fill=(28, 28, 30))
    return img


def _vehicle(d, scene, car, box):
    """The body in its colour plus details that tell the body type apart: car = front + rear window,
    van = front window + one long side window, truck = cab and cargo box with a gap, bus = row of windows."""
    lane = scene["lanes"][car["lane"]]
    x0, y0, x1, y1 = box
    col = sim.COLOURS[car["colour"]]
    d.rounded_rectangle([x0, y0, x1, y1], radius=6, fill=col, outline=(20, 20, 22), width=3)
    glass = (30, 40, 60)
    L = car["len"]

    def rect(u0, u1, v0, v1, fill):                     # u along the car from the back (px), v across (0..1)
        if lane["dir"] == "h":
            d.rectangle([x0 + u0, y0 + (y1 - y0) * v0, x0 + u1, y0 + (y1 - y0) * v1], fill=fill)
        else:
            d.rectangle([x0 + (x1 - x0) * v0, y0 + u0, x0 + (x1 - x0) * v1, y0 + u1], fill=fill)

    rect(L - 14, L - 7, 0.18, 0.82, glass)             # windscreen at the front
    body = car["body"]
    if body == "car":
        rect(6, 11, 0.22, 0.78, glass)
    elif body == "van":
        rect(10, L - 22, 0.12, 0.26, glass)
    elif body == "truck":
        cab = min(30.0, 0.32 * L)
        rect(L - cab - 4, L - cab + 1, -0.02, 1.02, (20, 20, 22))
    elif body == "bus":
        u = 8.0
        while u + 7 < L - 20:
            rect(u, u + 7, 0.12, 0.26, glass)
            rect(u, u + 7, 0.74, 0.88, glass)
            u += 12


def _road_ends(d, scene):
    """Repaint beyond the gate and past the far end so vehicles appear out of the gate and leave cleanly."""
    st = scene["style"]
    W = st["lane_w"]
    bg = tuple(st["bg"])
    for lane in scene["lanes"]:
        (ax, ay), (ex, ey) = lane["a"], lane["b"]
        if lane["dir"] == "h":
            d.rectangle([0, ay - W / 2 - 6, ax - 9, ay + W / 2 + 6], fill=bg)
            d.rectangle([ax - 8, ay - W / 2 - 6, ax, ay + W / 2 + 6], fill=(30, 30, 34))
            d.rectangle([ex + 1, ay - W / 2 - 6, SIZE[0], ay + W / 2 + 6], fill=bg)
        else:
            d.rectangle([ax - W / 2 - 6, 0, ax + W / 2 + 6, ay - 9], fill=bg)
            d.rectangle([ax - W / 2 - 6, ay - 8, ax + W / 2 + 6, ay], fill=(30, 30, 34))
            d.rectangle([ax - W / 2 - 6, ey + 1, ax + W / 2 + 6, SIZE[1]], fill=bg)


def _clock(d, frame):
    x0, y0, x1, y1 = CLOCK_BOX
    d.rounded_rectangle(CLOCK_BOX, radius=12, fill=(18, 18, 22), outline=(235, 235, 235), width=3)
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), clock_text(frame), fill=(255, 255, 255), font=font(44), anchor="mm")


def frame_at(scene, f: int) -> Image.Image:
    img = _static(json.dumps(scene, sort_keys=True)).copy()
    d = ImageDraw.Draw(img)
    S = state(scene, f)
    r = scene["style"]["lamp_px"]
    for li, L in enumerate(S["lights"]):
        lx, ly = lamp_xy(scene, li)
        d.ellipse([lx - r, ly - r, lx + r, ly + r], fill=(230, 40, 40) if L["red"] else (40, 210, 80))
    res = simulated(scene)
    for c in S["cars"]:
        car = scene["cars"][c["id"]]
        _vehicle(d, scene, car, car_box(scene, car, res["pos"][car["id"]][f], clip=False))
    _road_ends(d, scene)
    _clock(d, f)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * FPS)))
