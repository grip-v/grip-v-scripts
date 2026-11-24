"""Shared 2-D shape vocabulary (16 types) for the PIL-rendered datasets.

`unit_shape(kind, aspect)` is an outline of 'radius' ~1 centred on the
origin (y down); `outline(shape)` places one in board pixels from a dict
with type, x, y, size, rot (degrees) and optional aspect."""
from __future__ import annotations
import functools
import math

from shapely.geometry import Polygon
from shapely import affinity

TYPES = ["circle", "ellipse", "square", "rectangle", "diamond", "triangle", "pentagon", "hexagon",
         "octagon", "star", "cross", "heart", "crescent", "trapezoid", "semicircle", "parallelogram"]


def _regular(n, start=90.0):
    return [(math.cos(math.radians(start + 360 * i / n)), math.sin(math.radians(start + 360 * i / n)))
            for i in range(n)]


@functools.lru_cache(maxsize=None)
def unit_shape(kind: str, aspect: float) -> tuple:
    """Outline of a shape of 'radius' ~1 centred on the origin (y down)."""
    if kind == "circle":
        return tuple(_regular(72))
    if kind == "ellipse":
        return tuple((x, y / aspect) for x, y in _regular(72))
    if kind == "square":
        return tuple(_regular(4, 45))
    if kind == "rectangle":
        h = 1 / aspect
        return ((-1, -h), (1, -h), (1, h), (-1, h))
    if kind == "diamond":
        return ((0, -1), (0.62, 0), (0, 1), (-0.62, 0))
    if kind == "triangle":
        return tuple(_regular(3, -90))
    if kind == "pentagon":
        return tuple(_regular(5, -90))
    if kind == "hexagon":
        return tuple(_regular(6, 0))
    if kind == "octagon":
        return tuple(_regular(8, 22.5))
    if kind == "star":
        return tuple(((1 if i % 2 == 0 else 0.45) * math.cos(math.radians(-90 + 36 * i)),
                      (1 if i % 2 == 0 else 0.45) * math.sin(math.radians(-90 + 36 * i))) for i in range(10))
    if kind == "cross":
        a = 0.34
        return ((-a, -1), (a, -1), (a, -a), (1, -a), (1, a), (a, a), (a, 1), (-a, 1), (-a, a), (-1, a),
                (-1, -a), (-a, -a))
    if kind == "heart":
        pts = []
        for i in range(72):
            t = 2 * math.pi * i / 72
            x = 16 * math.sin(t) ** 3
            y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
            pts.append((x / 17, -(y + 2) / 17))
        return tuple(pts)
    if kind == "crescent":
        outer = Polygon(_regular(72))
        inner = affinity.translate(Polygon([(0.82 * x, 0.82 * y) for x, y in _regular(72)]), 0.42, -0.1)
        return tuple(outer.difference(inner).exterior.coords[:-1])
    if kind == "trapezoid":
        return ((-0.55, -0.62), (0.55, -0.62), (1, 0.62), (-1, 0.62))
    if kind == "semicircle":
        return tuple((math.cos(math.pi + math.pi * i / 36), 0.42 + math.sin(math.pi + math.pi * i / 36))
                     for i in range(37))
    if kind == "parallelogram":
        return ((-0.45, -0.55), (1, -0.55), (0.45, 0.55), (-1, 0.55))
    raise ValueError(kind)


def outline(shape: dict) -> list[tuple[float, float]]:
    """Board-pixel outline of a stored shape."""
    th = math.radians(shape["rot"])
    c, s = math.cos(th), math.sin(th)
    cx, cy, r = shape["x"], shape["y"], shape["size"]
    return [(cx + r * (x * c - y * s), cy + r * (x * s + y * c))
            for x, y in unit_shape(shape["type"], shape.get("aspect", 1.0))]


