"""Regenerate the logo (docs/images/logo-light.svg and logo-dark.svg).

The logo is a map of the contiguous United States with 19 NFL cities (the benchmark family of this
project) connected by a Delaunay network, which stands for the distance matrix. The home city is a
football that glows, and its road trip is highlighted: the trip of one team is exactly what the
traveling tournament problem minimizes.

Inputs, downloaded once and cached in ~/.cache/pyttp-logo:

* the US outline from us-atlas (ISC license), which is derived from US Census data (public domain),
* the font Varela Round (SIL Open Font License 1.1). The glyph outlines are embedded as paths in
  the SVG, so no font is needed to display it.

matplotlib and scipy are not dependencies of pyttp. Run

    uv run --with matplotlib --with scipy python docs/make_logo.py [--yard-lines]
"""

from __future__ import annotations

import itertools
import json
import math
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from scipy.spatial import Delaunay

OUT = Path(__file__).parent / "images"
CACHE = Path.home() / ".cache" / "pyttp-logo"
ATLAS_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3/nation-10m.json"
FONT_URL = (
    "https://raw.githubusercontent.com/google/fonts/main/ofl/varelaround/VarelaRound-Regular.ttf"
)

# NFL cities, (latitude, longitude), approximate city centers.
CITIES = {
    "Seattle": (47.61, -122.33),
    "San Francisco": (37.77, -122.42),
    "Los Angeles": (34.05, -118.24),
    "Las Vegas": (36.17, -115.14),
    "Arizona": (33.45, -112.07),
    "Denver": (39.74, -104.99),
    "Dallas": (32.78, -96.80),
    "Houston": (29.76, -95.37),
    "Kansas City": (39.10, -94.58),
    "Minneapolis": (44.98, -93.27),
    "Chicago": (41.88, -87.63),
    "Detroit": (42.33, -83.05),
    "Nashville": (36.16, -86.78),
    "New Orleans": (29.95, -90.07),
    "Atlanta": (33.75, -84.39),
    "Miami": (25.76, -80.19),
    "Washington": (38.91, -77.04),
    "New York": (40.71, -74.01),
    "Boston": (42.36, -71.06),
}
HOME = "Kansas City"
BALL_LENGTH = 150.0  # in logo units (the map is 1000 wide)
BALL_ANGLE = 25.0  # degrees: dynamic, and leaves the nearest road-trip leg a visible arrow
TRIP = ["Kansas City", "Minneapolis", "Detroit", "Atlanta", "Kansas City"]  # 3 away games

THEMES = {
    "light": {
        "land": "#eaf0ff",
        "line": "#3b5bdb",
        "edge_alpha": 0.42,
        "dot": "#1b2a6b",
        "text": "#1b2a6b",
        "muted": "#5c6bb0",
        "accent": "#f76707",
        "glow": "#ffa94d",
        "glow_alpha": 0.022,
        "ball": "#9c4a1a",
        "ball_edge": "#4e230a",
        "lace": "#ffffff",
        "preview_bg": ["#ffffff"],
    },
    "dark": {
        "land": "#1a2140",
        "line": "#8da2fb",
        "edge_alpha": 0.5,
        "dot": "#e3e9ff",
        "text": "#eef1ff",
        "muted": "#9aa7de",
        "accent": "#ffa94d",
        "glow": "#ffa94d",
        "glow_alpha": 0.035,
        "ball": "#c96f27",
        "ball_edge": "#5a2a0c",
        "lace": "#fff6e8",
        "preview_bg": ["#0d1117", "#22272e"],
    },
}


def fetch(url: str, name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / name
    if not target.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            target.write_bytes(response.read())
    return target


# -- geography ---------------------------------------------------------------------------------


def decode_nation(path: Path) -> np.ndarray:
    """Outline of the contiguous United States (lon, lat) from the us-atlas TopoJSON."""
    topo = json.loads(path.read_text())
    sx, sy = topo["transform"]["scale"]
    tx, ty = topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        points = []
        for dx, dy in arc:
            x, y = x + dx, y + dy
            points.append((x * sx + tx, y * sy + ty))
        arcs.append(points)

    def ring(indices):
        points: list = []
        for i in indices:
            segment = arcs[i] if i >= 0 else arcs[~i][::-1]
            points.extend(segment if not points else segment[1:])
        return np.array(points)

    nation = topo["objects"]["nation"]
    geometry = nation["geometries"][0] if "geometries" in nation else nation
    polygons = geometry["arcs"] if geometry["type"] == "MultiPolygon" else [geometry["arcs"]]
    rings = [ring(polygon[0]) for polygon in polygons]
    return max(rings, key=len)  # the mainland; Alaska, Hawaii and islands are separate polygons


def albers(lon, lat, lat1=29.5, lat2=45.5, lat0=37.5, lon0=-96.0):
    """Albers equal-area conic projection, the usual choice for maps of the United States."""
    lon, lat = np.radians(lon), np.radians(lat)
    lat1, lat2, lat0, lon0 = (math.radians(v) for v in (lat1, lat2, lat0, lon0))
    n = (math.sin(lat1) + math.sin(lat2)) / 2
    c = math.cos(lat1) ** 2 + 2 * n * math.sin(lat1)
    rho = np.sqrt(c - 2 * n * np.sin(lat)) / n
    rho0 = math.sqrt(c - 2 * n * math.sin(lat0)) / n
    theta = n * (lon - lon0)
    return np.column_stack([rho * np.sin(theta), rho0 - rho * np.cos(theta)])


def simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer-Douglas-Peucker for a closed ring: fewer points, same shape.

    A closed ring starts and ends in the same point, so it is split at its westernmost and
    easternmost points and the two halves are simplified separately.
    """
    ring = points[:-1] if np.allclose(points[0], points[-1]) else points
    west, east = int(np.argmin(ring[:, 0])), int(np.argmax(ring[:, 0]))
    first, second = sorted((west, east))
    halves = [ring[first : second + 1], np.vstack([ring[second:], ring[: first + 1]])]
    result = []
    for half in halves:
        keep = np.zeros(len(half), dtype=bool)
        keep[0] = keep[-1] = True
        stack = [(0, len(half) - 1)]
        while stack:
            lo, hi = stack.pop()
            if hi <= lo + 1:
                continue
            a, b = half[lo], half[hi]
            ab = b - a
            norm = math.hypot(*ab) or 1e-12
            inner = half[lo + 1 : hi]
            distance = np.abs(ab[0] * (inner[:, 1] - a[1]) - ab[1] * (inner[:, 0] - a[0])) / norm
            i = int(np.argmax(distance))
            if distance[i] > tolerance:
                keep[lo + 1 + i] = True
                stack += [(lo, lo + 1 + i), (lo + 1 + i, hi)]
        result.append(half[keep][:-1])
    return np.vstack(result)


def network(points: np.ndarray, outline: MplPath) -> list[tuple[int, int]]:
    """Delaunay edges that stay on land and are not unreasonably long."""
    edges = set()
    for simplex in Delaunay(points).simplices:
        for i in range(3):
            a, b = sorted((int(simplex[i]), int(simplex[(i + 1) % 3])))
            edges.add((a, b))
    lengths = {e: float(np.hypot(*(points[e[0]] - points[e[1]]))) for e in edges}
    longest = np.percentile(list(lengths.values()), 89)
    kept = []
    for edge in sorted(edges):
        t = np.linspace(0.04, 0.96, 30)[:, None]
        samples = points[edge[0]] * (1 - t) + points[edge[1]] * t
        if lengths[edge] <= longest and outline.contains_points(samples).mean() > 0.96:
            kept.append(edge)
    return kept


# -- drawing -----------------------------------------------------------------------------------


def add_text(ax, text: str, size: float, origin, color: str, font: FontProperties) -> float:
    """Draw text as filled outlines at `origin`; returns its width."""
    glyphs = TextPath((0, 0), text, size=size, prop=font)
    vertices = glyphs.vertices + np.array(origin)
    ax.add_patch(
        PathPatch(MplPath(vertices, glyphs.codes), facecolor=color, edgecolor="none", zorder=7)
    )
    return float(glyphs.get_extents().x1)


def football_outline(center, length: float, angle: float, grow: float = 1.0) -> MplPath:
    """Closed outline of an American football (a parabolic lens) rotated by `angle` degrees."""
    half = length * grow / 2
    control = 0.50 * length * grow / 2 / 0.75  # a cubic reaches 3/4 of its control height
    local = np.array(
        [
            (-half, 0),
            (-half / 3, control),
            (half / 3, control),
            (half, 0),
            (half / 3, -control),
            (-half / 3, -control),
            (-half, 0),
            (-half, 0),
        ]
    )
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 6 + [MplPath.CLOSEPOLY]
    return MplPath(rotate(local, angle) + center, codes)


def rotate(points: np.ndarray, angle: float) -> np.ndarray:
    a = math.radians(angle)
    matrix = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return points @ matrix.T


def draw_football(ax, center, length: float, angle: float, colors: dict) -> None:
    """The football: leather body, two white stripes near the tips and the laces on top."""
    # a ring in the colour of the land separates the ball from the network lines below
    ax.add_patch(
        PathPatch(
            football_outline(center, length, angle, grow=1.10),
            facecolor=colors["land"],
            edgecolor="none",
            zorder=5,
        )
    )
    ax.add_patch(
        PathPatch(
            football_outline(center, length, angle),
            facecolor=colors["ball"],
            edgecolor=colors["ball_edge"],
            linewidth=1.4,
            joinstyle="round",
            zorder=6,
        )
    )
    half, width = length / 2, 0.50 * length

    def half_width(x: float) -> float:
        return width / 2 * (1 - (x / half) ** 2)

    strokes = []
    for x in (-0.6 * half, 0.6 * half):  # the two stripes
        w = half_width(x) * 0.97
        strokes.append(([(x, -w), (x, w)], 2.0))
    strokes.append(([(-0.3 * half, 0), (0.3 * half, 0)], 2.0))  # lace ridge
    for x in np.linspace(-0.21 * half, 0.21 * half, 4):  # the cross laces
        strokes.append(([(x, -0.085 * length), (x, 0.085 * length)], 1.8))
    for points, linewidth in strokes:
        placed = rotate(np.array(points, dtype=float), angle) + center
        ax.plot(
            placed[:, 0],
            placed[:, 1],
            color=colors["lace"],
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=7,
        )


def ball_radius(length: float, angle: float, direction: np.ndarray, gap: float = 10.0) -> float:
    """Distance from the football's center to its edge in `direction`, plus a small gap.

    The ball is a lens: its extent depends on the direction, so arrows that leave across the slim
    side can start much closer than arrows pointing along the long axis.
    """
    local = rotate(np.array([direction / np.hypot(*direction)]), -angle)[0]
    half, width = length / 2, 0.50 * length
    # points t * local on the boundary |y| = (width / 2) * (1 - (x / half) ** 2)
    lo, hi = 0.0, half
    for _ in range(40):
        t = (lo + hi) / 2
        x, y = t * local
        inside = abs(y) <= width / 2 * (1 - (x / half) ** 2) if abs(x) <= half else False
        lo, hi = (t, hi) if inside else (lo, t)
    return lo * 1.10 + gap  # 1.10: the ring around the ball


def vertical_chords(polygon: np.ndarray, x: float) -> list[tuple[float, float]]:
    """Segments of the vertical line at `x` that lie inside the polygon (even-odd rule)."""
    ys = []
    for (x1, y1), (x2, y2) in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        if (x1 <= x < x2) or (x2 <= x < x1):
            ys.append(y1 + (x - x1) / (x2 - x1) * (y2 - y1))
    ys.sort()
    return list(zip(ys[0::2], ys[1::2], strict=False))


def draw_logo(theme: str, preview_bg: str | None, path: Path, yard_lines: bool = False) -> None:
    colors = THEMES[theme]
    nation = decode_nation(fetch(ATLAS_URL, "nation-10m.json"))
    outline_xy = simplify(albers(nation[:, 0], nation[:, 1]), 0.0011)
    names = list(CITIES)
    xy = albers(np.array([CITIES[n][1] for n in names]), np.array([CITIES[n][0] for n in names]))
    index = {name: i for i, name in enumerate(names)}

    # logo units: the map is 1000 wide
    low, high = outline_xy.min(axis=0), outline_xy.max(axis=0)
    scale = 1000 / (high[0] - low[0])
    outline = (outline_xy - low) * scale
    points = (xy - low) * scale
    map_height = float(outline[:, 1].max())
    margin = 40.0
    text_block = 720.0  # room left of the map for the wordmark and the tagline
    outline[:, 0] += text_block
    points[:, 0] += text_block
    outline_path = MplPath(np.vstack([outline, outline[:1]]), closed=True)
    left, right = -margin, text_block + 1000 + margin
    bottom, top = -margin, map_height + margin
    fig = plt.figure(figsize=(10.0, 10.0 * (top - bottom) / (right - left)))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    if preview_bg:
        fig.patch.set_facecolor(preview_bg)
    else:
        fig.patch.set_alpha(0)

    ax.add_patch(
        PathPatch(
            outline_path,
            facecolor=colors["land"],
            edgecolor=colors["line"],
            linewidth=1.8,
            joinstyle="round",
            zorder=1,
        )
    )
    if yard_lines:  # a football field: a faint line every "ten yards"
        for x in np.arange(text_block + 50, text_block + 1000, 100):
            for low_y, high_y in vertical_chords(outline, float(x)):
                ax.plot(
                    [x, x],
                    [low_y, high_y],
                    color=colors["line"],
                    alpha=0.2,
                    linewidth=1.2,
                    zorder=1.5,
                )
    for a, b in network(points, outline_path):
        ax.plot(
            [points[a][0], points[b][0]],
            [points[a][1], points[b][1]],
            color=colors["line"],
            alpha=colors["edge_alpha"],
            linewidth=1.9,
            solid_capstyle="round",
            zorder=2,
        )
    # glow around the home city (concentric translucent discs)
    home = points[index[HOME]]
    for radius in np.linspace(130, 14, 30):
        ax.add_patch(
            Circle(
                home,
                radius,
                facecolor=colors["glow"],
                edgecolor="none",
                alpha=colors["glow_alpha"],
                zorder=3,
            )
        )
    # road trip: three away games and back home. Arrows end at the edge of the football.
    unit_to_pt = fig.get_figwidth() * 72 / (right - left)  # points per logo unit (0.4)

    def shrink(city: str, other: str) -> float:
        if city != HOME:
            return 13.0
        direction = points[index[other]] - points[index[city]]
        return ball_radius(BALL_LENGTH, BALL_ANGLE, direction) * unit_to_pt

    for start, end in itertools.pairwise(TRIP):
        ax.add_patch(
            FancyArrowPatch(
                points[index[start]],
                points[index[end]],
                arrowstyle="-|>",
                mutation_scale=26,
                shrinkA=shrink(start, end),
                shrinkB=shrink(end, start),
                linewidth=5.0,
                color=colors["accent"],
                capstyle="round",
                joinstyle="round",
                zorder=4,
            )
        )
    for name, p in zip(names, points, strict=True):
        if name == HOME:
            draw_football(ax, p, BALL_LENGTH, BALL_ANGLE, colors)
            continue
        on_trip = name in TRIP
        radius = 13 if on_trip else 10.5
        halo = 4.5
        ax.add_patch(Circle(p, radius + halo, facecolor=colors["land"], edgecolor="none", zorder=5))
        ax.add_patch(
            Circle(
                p,
                radius,
                facecolor=colors["accent"] if on_trip else colors["dot"],
                edgecolor="none",
                zorder=6,
            )
        )

    # wordmark below the Mexican border, set in Varela Round
    font = FontProperties(fname=str(fetch(FONT_URL, "VarelaRound-Regular.ttf")))
    x, y, size = 6.0, map_height / 2 - 10, 235.0
    x += add_text(ax, "py", size, (x, y), colors["text"], font) + 4
    add_text(ax, "ttp", size, (x, y), colors["accent"], font)
    add_text(ax, "traveling", 50, (24.0, y - 105), colors["muted"], font)
    add_text(ax, "tournament problem", 50, (24.0, y - 170), colors["muted"], font)

    # No timestamp, and ids derived from a fixed salt: regenerating gives byte-identical files.
    metadata = {"Date": None} if path.suffix == ".svg" else None
    fig.savefig(
        path,
        transparent=preview_bg is None,
        facecolor=fig.get_facecolor(),
        metadata=metadata,
    )
    plt.close(fig)


if __name__ == "__main__":
    import sys

    # `--yard-lines` adds a faint line every "ten yards" across the map, like a football field.
    with_yard_lines = "--yard-lines" in sys.argv[1:]
    matplotlib.rcParams["svg.fonttype"] = "path"
    matplotlib.rcParams["svg.hashsalt"] = "pyttp-logo"
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        draw_logo(theme, None, OUT / f"logo-{theme}.svg", yard_lines=with_yard_lines)
    print("written", *(OUT / f"logo-{t}.svg" for t in THEMES))
