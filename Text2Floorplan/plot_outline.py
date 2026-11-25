"""Turn an uploaded plot drawing into a measured building outline.

1. ``detect_outline``  finds the plot polygon in the image (OpenCV).
2. ``solve_plot``      applies the edge lengths typed by the user, in metres.
3. ``building_outline`` gives the right-angled outline that will be built on.

Image and plan coordinates share one orientation: x to the right, y downwards.
"""
from __future__ import annotations

import math
from string import ascii_uppercase

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

MAX_EDGES = 12
MAX_IMAGE_SIDE = 1400
EXTERIOR_WALL_M = 0.4  # exterior walls straddle the outline, see revit_geometry
MIN_EDGE_M, MAX_EDGE_M = 1.0, 200.0
MIN_BUILDING_AREA_M2, MAX_BUILDING_AREA_M2 = 25.0, 900.0
SNAP_DEGREES = 7.0          # edges this close to a right angle are right angles
MAX_TURN_DEGREES = 20.0     # how far typed lengths may bend the drawn angles


class PlotError(ValueError):
    """The plot drawing or its measurements cannot be used."""


def edge_name(index):
    return ascii_uppercase[index]


# --------------------------------------------------------------- 1. image

def _candidate_masks(image):
    import cv2

    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        if alpha.min() < 32 and alpha.max() > 224:
            yield (alpha > 127).astype(np.uint8) * 255
        weight = alpha[:, :, None].astype(float) / 255
        image = (image[:, :, :3] * weight + 255 * (1 - weight)).astype(np.uint8)
    colour = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gray = cv2.GaussianBlur(cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    saturation = cv2.GaussianBlur(cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)[:, :, 1], (5, 5), 0)
    size = max(3, round(min(gray.shape) / 250) * 2 + 1)
    kernel = np.ones((size, size), np.uint8)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks = [dark, 255 - dark]
    if int(saturation.max()) - int(saturation.min()) > 40:
        masks.append(cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    masks.append(cv2.dilate(cv2.Canny(gray, 40, 120), kernel))
    for mask in masks:
        yield cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _simplify(contour):
    import cv2

    perimeter = cv2.arcLength(contour, True)
    points = None
    for fraction in (0.008, 0.012, 0.016, 0.022, 0.03, 0.04, 0.055):
        points = cv2.approxPolyDP(contour, fraction * perimeter, True).reshape(-1, 2).astype(float)
        points = _drop_straight(points)
        if len(points) <= MAX_EDGES:
            return points
    return None


def _drop_straight(points, degrees=12.0):
    """Remove corners that barely turn."""
    points = [np.asarray(p, dtype=float) for p in points]
    while len(points) > 3:
        turns = []
        for i in range(len(points)):
            a, b, c = points[i - 1], points[i], points[(i + 1) % len(points)]
            u, v = b - a, c - b
            turns.append(abs(math.degrees(math.atan2(u[0] * v[1] - u[1] * v[0], float(u @ v)))))
        i = int(np.argmin(turns))
        if turns[i] >= degrees:
            break
        points.pop(i)
    return np.asarray(points)


def detect_outline(data: bytes) -> dict:
    """Return the plot polygon found in an image, in image pixels."""
    try:
        import cv2
    except ImportError:
        raise PlotError("OpenCV is not installed. Run: pip install -r requirements.txt") from None

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise PlotError("This file is not a readable image")
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    height, width = image.shape[:2]
    if min(height, width) < 40:
        raise PlotError("The image is too small to read")
    shrink = min(1.0, MAX_IMAGE_SIDE / max(height, width))
    if shrink < 1:
        image = cv2.resize(image, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    margin = max(2, round(0.006 * max(h, w)))

    best, best_score = None, 0.0
    for mask in _candidate_masks(image):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not 0.03 * h * w <= area <= 0.97 * h * w:
                continue
            x, y, cw, ch = cv2.boundingRect(contour)
            edges = (x <= margin) + (y <= margin) + (x + cw >= w - margin) + (y + ch >= h - margin)
            if edges >= 2:  # the page itself or its border
                continue
            points = _simplify(contour)
            if points is None or len(points) < 3:
                continue
            polygon = Polygon(points)
            if not polygon.is_valid or polygon.area <= 0:
                continue
            outline = Polygon(contour.reshape(-1, 2)).buffer(0)
            fit = outline.intersection(polygon).area / outline.union(polygon).area
            score = area / (h * w) * fit ** 4
            # The inner and outer side of one drawn line score alike: prefer outer.
            if fit > 0.85 and score > best_score:
                best, best_score = points, score
    if best is None:
        raise PlotError("No closed plot outline was found in this image. Use a clear drawing "
                        "with the plot boundary fully visible, or place the corners yourself.")
    best = best / shrink
    if Polygon(best).exterior.is_ccw is False:
        best = best[::-1]
    # Start at the corner nearest the top-left so edge A is predictable.
    start = int(np.argmin(best[:, 0] + best[:, 1]))
    best = np.roll(best, -start, axis=0)
    return {"points": [[float(x), float(y)] for x, y in best],
            "width": int(width), "height": int(height)}


# --------------------------------------------------- 2. lengths in metres

def _clean_inputs(points, lengths):
    try:
        points = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        raise PlotError("The plot corners are not valid") from None
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise PlotError("The plot corners are not valid")
    if not 3 <= len(points) <= MAX_EDGES:
        raise PlotError(f"A plot needs between 3 and {MAX_EDGES} corners")
    polygon = Polygon(points)
    if not polygon.is_valid or polygon.area <= 0:
        raise PlotError("The plot outline crosses itself. Move the corners so that it does not.")
    if not isinstance(lengths, (list, tuple)) or len(lengths) != len(points):
        raise PlotError("Enter one length for every edge")
    values = []
    for index, value in enumerate(lengths):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise PlotError(f"Enter the length of edge {edge_name(index)} in metres")
        if not MIN_EDGE_M <= value <= MAX_EDGE_M:
            raise PlotError(f"Edge {edge_name(index)} must be between {MIN_EDGE_M:g} and {MAX_EDGE_M:g} m")
        values.append(float(value))
    return points, np.asarray(values)


def _dominant_rotation(angles, weights):
    """Rotation that best lines the edges up with the x and y axes."""
    mean = np.angle(np.sum(weights * np.exp(4j * angles))) / 4
    return -float(mean)


def _close_by_turning(angles, lengths):
    """Turn each edge as little as possible so the typed lengths meet."""
    target = angles.copy()
    current = angles.copy()
    for _ in range(80):
        z = lengths * np.exp(1j * current)
        gap = z.sum()
        residual = np.array([gap.real, gap.imag])
        if np.abs(residual).max() < 1e-10:
            break
        jacobian = np.vstack([-z.imag, z.real])
        drift = current - target
        # Gauss-Newton step of: minimise |angle change|, subject to closure.
        matrix = jacobian @ jacobian.T + 1e-12 * np.eye(2)
        multiplier = np.linalg.solve(matrix, residual - jacobian @ drift)
        current = target - jacobian.T @ multiplier
    z = lengths * np.exp(1j * current)
    if abs(z.sum()) > 1e-4:
        return None
    return current


def solve_plot(points, lengths) -> dict:
    """Build the plot polygon in metres from drawn corners and typed lengths."""
    points, lengths = _clean_inputs(points, lengths)
    edges = np.roll(points, -1, axis=0) - points
    drawn = np.hypot(edges[:, 0], edges[:, 1])
    if (drawn < 1e-6).any():
        raise PlotError("Two plot corners are in the same place")
    angles = np.unwrap(np.arctan2(edges[:, 1], edges[:, 0]))
    angles = angles + _dominant_rotation(angles, drawn)
    quarter = np.round(angles / (math.pi / 2)) * (math.pi / 2)
    near = np.abs(angles - quarter) <= math.radians(SNAP_DEGREES)
    angles = np.where(near, quarter, angles)
    total = float(lengths.sum())
    notes = []

    if 2 * float(lengths.max()) >= total:
        longest = int(np.argmax(lengths))
        raise PlotError(f"Edge {edge_name(longest)} is longer than all the other edges together, "
                        "so the outline cannot close")
    vectors = lengths[:, None] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    gap = vectors.sum(axis=0)
    size = float(np.hypot(*gap))
    square = bool(near.all())
    if size <= 1e-9 * total:
        solved = angles
    elif square and size <= (0.03 if len(points) > 4 else 0.005) * total:
        # Right-angled drawing, small measuring difference: share it out.
        share = np.abs(vectors)
        adjusted = vectors - share / np.maximum(share.sum(axis=0), 1e-12) * gap
        vectors, solved = adjusted, angles
        if size > 0.02:
            worst = float(np.abs(np.hypot(*adjusted.T) - lengths).max())
            notes.append(f"The lengths missed closing by {size:.2f} m. Edges were adjusted by up to {worst:.2f} m.")
    elif square and len(points) > 4:
        raise PlotError(_square_mismatch(angles, lengths))
    else:
        solved = _close_by_turning(angles, lengths)
        turn = None if solved is None else math.degrees(float(np.abs(solved - angles).max()))
        if turn is None or turn > MAX_TURN_DEGREES:
            raise PlotError("These lengths do not match the drawn shape. Check that each length "
                            "belongs to the edge with the same letter.")
        if turn > 3:
            notes.append(f"Corners were turned by up to {turn:.0f}° to fit the lengths.")
        vectors = lengths[:, None] * np.stack([np.cos(solved), np.sin(solved)], axis=1)
    corners = np.vstack([[0, 0], np.cumsum(vectors, axis=0)[:-1]])
    corners = corners - corners.min(axis=0)
    corners = np.round(corners, 6)
    polygon = Polygon(corners)
    if not polygon.is_valid or polygon.area < 1:
        raise PlotError("These lengths fold the outline over itself. Check them against the drawing.")
    return {"corners": corners.tolist(), "lengths": lengths.tolist(),
            "area_m2": float(polygon.area), "notes": notes}


def _square_mismatch(angles, lengths):
    parts = []
    for base, first, second in ((0, "rightward", "leftward"), (1, "downward", "upward")):
        axis = np.cos(angles) if base == 0 else np.sin(angles)
        plus = [i for i in range(len(lengths)) if axis[i] > 0.5]
        minus = [i for i in range(len(lengths)) if axis[i] < -0.5]
        a, b = float(lengths[plus].sum()), float(lengths[minus].sum())
        if abs(a - b) > 0.02:
            parts.append(f"{'+'.join(map(edge_name, plus))} = {a:g} m but "
                         f"{'+'.join(map(edge_name, minus))} = {b:g} m")
    return "These lengths do not close: " + "; ".join(parts) + ". Opposite sides must be equal."


# ----------------------------------------------------- 3. building outline

def is_right_angled(polygon, tolerance=1e-5):
    coords = list(polygon.exterior.coords)
    return all(min(abs(a[0] - b[0]), abs(a[1] - b[1])) < tolerance
               for a, b in zip(coords, coords[1:]))


def _largest_rectangle(grid):
    """Largest all-true rectangle as (area, top, left, bottom, right), in cells."""
    best = (0, 0, 0, 0, 0)
    heights = np.zeros(grid.shape[1], dtype=int)
    for row in range(grid.shape[0]):
        heights = np.where(grid[row], heights + 1, 0)
        stack = []
        for col in range(grid.shape[1] + 1):
            current = heights[col] if col < grid.shape[1] else 0
            start = col
            while stack and stack[-1][1] >= current:
                left, tall = stack.pop()
                area = tall * (col - left)
                if area > best[0]:
                    best = (int(area), row - int(tall) + 1, left, row + 1, col)
                start = left
            stack.append((start, current))
    return best


def _inscribed(polygon, pieces=3):
    """Right-angled shape of a few rectangles that stays inside ``polygon``."""
    from shapely.prepared import prep

    x0, y0, x1, y1 = polygon.bounds
    step = max(0.2, max(x1 - x0, y1 - y0) / 90)
    nx, ny = int((x1 - x0) / step + 1e-9), int((y1 - y0) / step + 1e-9)
    if nx < 2 or ny < 2:
        return Polygon()
    inside = prep(polygon.buffer(1e-7))
    cells = np.array([[inside.contains(box(x0 + c * step, y0 + r * step,
                                           x0 + (c + 1) * step, y0 + (r + 1) * step))
                       for c in range(nx)] for r in range(ny)])
    shape = Polygon()
    narrow = max(1, math.ceil(2.4 / step))
    for piece in range(pieces):
        area, top, left, bottom, right = _largest_rectangle(cells)
        if not area:
            break
        rectangle = box(x0 + left * step, y0 + top * step, x0 + right * step, y0 + bottom * step)
        if piece:
            contact = shape.boundary.intersection(rectangle.boundary).length
            if (rectangle.area < 0.06 * polygon.area or contact < 2.4
                    or min(bottom - top, right - left) < narrow):
                break
        merged = unary_union([shape, rectangle])
        if merged.geom_type != "Polygon" or merged.interiors:
            break
        shape = merged
        cells[top:bottom, left:right] = False
    return shape.simplify(1e-9)


def building_outline(corners) -> dict:
    """Centreline of the outer walls: right-angled and inside the plot.

    A right-angled plot is filled completely (outer wall faces lie on the plot
    edges). For a slanted plot the best right-angled shape inside it is used.
    """
    from shapely import affinity

    plot = Polygon(corners)
    inset = EXTERIOR_WALL_M / 2
    inner = plot.buffer(-inset, join_style=2)
    if inner.geom_type != "Polygon" or inner.is_empty:
        raise PlotError("The plot is too narrow to build on")
    if is_right_angled(plot):
        # The mitred inset is exact up to rounding; make it exactly right-angled.
        ring = np.round(np.asarray(inner.exterior.coords)[:-1], 6)
        shape, rotated = Polygon(ring).simplify(1e-9), plot
    else:
        best, best_area, rotated = Polygon(), 0.0, plot
        ring = np.asarray(plot.exterior.coords)
        deltas = np.diff(ring, axis=0)
        turns = {round(math.degrees(math.atan2(d[1], d[0])) % 90, 1) for d in deltas}
        for turn in sorted({(-t + 45) % 90 - 45 for t in turns}):
            candidate = affinity.rotate(plot, turn, origin=(0, 0))
            # Inscribe the outer wall faces, then step in to the centreline, so
            # that no wall corner crosses a slanted plot edge.
            shape = _inscribed(candidate).buffer(-inset, join_style=2)
            if shape.geom_type != "Polygon":
                continue
            if shape.area > best_area * 1.005:
                best, best_area, rotated = shape, shape.area, candidate
        shape = Polygon(np.round(np.asarray(best.exterior.coords)[:-1], 6)) if not best.is_empty else best
    x0, y0, _, _ = rotated.bounds
    shape = affinity.translate(shape, -x0, -y0)
    rotated = affinity.translate(rotated, -x0, -y0)
    if shape.is_empty:
        raise PlotError("The plot is too narrow to build on")
    if shape.area < MIN_BUILDING_AREA_M2:
        raise PlotError(f"Only {shape.area:.0f} m² can be built on this plot. "
                        f"At least {MIN_BUILDING_AREA_M2:.0f} m² is needed.")
    if shape.area > MAX_BUILDING_AREA_M2:
        raise PlotError(f"This plot is {shape.area:.0f} m². The model draws single homes up to "
                        f"{MAX_BUILDING_AREA_M2:.0f} m².")
    if min(shape.bounds[2] - shape.bounds[0], shape.bounds[3] - shape.bounds[1]) < 3.0:
        raise PlotError("The plot is too narrow to build on")
    footprint = shape.buffer(inset, join_style=2)
    return {"plot": [list(map(float, p)) for p in rotated.exterior.coords[:-1]],
            "outline": [list(map(float, p)) for p in shape.exterior.coords[:-1]],
            "footprint_m2": float(footprint.area),
            "coverage": float(footprint.area / plot.area),
            "right_angled": bool(is_right_angled(plot))}


def measure_plot(points, lengths) -> dict:
    """Everything later steps need, from drawn corners and typed lengths."""
    solved = solve_plot(points, lengths)
    built = building_outline(solved["corners"])
    notes = list(solved["notes"])
    if built["coverage"] < 0.995:
        notes.append(f"Walls must meet at right angles, so the building covers "
                     f"{built['coverage']:.0%} of the plot. The rest stays open ground.")
    return {**built, "lengths": solved["lengths"], "plot_area_m2": solved["area_m2"], "notes": notes}


if __name__ == "__main__":
    import sys

    # python plot_outline.py IMAGE            list the detected edges
    # python plot_outline.py IMAGE 9,6,6,...  also measure the plot
    found = detect_outline(open(sys.argv[1], "rb").read())["points"]
    for i, (a, b) in enumerate(zip(found, found[1:] + found[:1])):
        print(f"edge {edge_name(i)}: ({a[0]:.0f}, {a[1]:.0f}) -> ({b[0]:.0f}, {b[1]:.0f})")
    if len(sys.argv) > 2:
        plot = measure_plot(found, [float(v) for v in sys.argv[2].split(",")])
        print(f"plot {plot['plot_area_m2']:.1f} m2, building {plot['footprint_m2']:.1f} m2", *plot["notes"])
