"""Fit a generated layout to a given building outline.

The released flow model has no outline input. Two steps add one:

1. ``Guide``  runs inside the sampler. At every step it moves the rooms the
   model is about to draw out of the land that is not part of the building,
   and the model rearranges the other rooms around them.
2. ``fit_rooms`` turns the sampled rooms into a right-angled partition that
   fills the outline exactly: overlaps are settled, spill is cut off, and rooms
   grow into the empty land next to them.
"""
from __future__ import annotations

import numpy as np
import torch
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

DOORS = {"front_door", "door"}
SPLIT_M = 0.8
CELL_M = 0.6                 # walls closer than this are drawn as one wall
GUIDE_SIGMA = (0.03, 0.97)   # part of the sampling path that is guided
# Relative floor area each kind of room should get.
SHARE = {"living": 3.0, "bedroom": 1.6, "kitchen": 1.1, "bathroom": 0.5,
         "balcony": 0.8, "storage": 0.45}
MIN_WIDTH_M = {"living": 2.4, "bedroom": 2.2, "kitchen": 1.5, "bathroom": 1.1,
               "balcony": 0.9, "storage": 0.9}
MIN_AREA_M2 = {"living": 9.0, "bedroom": 6.5, "kitchen": 3.5, "bathroom": 2.4,
               "balcony": 1.8, "storage": 1.2}


class PlotFitError(ValueError):
    """This sample does not fit the plot; another sample may."""


# ------------------------------------------------------------- outline maths

def notches(outline):
    """Rectangles of the outline's bounding box that are not building, in [-1, 1]."""
    x0, y0, x1, y1 = outline.bounds
    xs = sorted({round(p[0], 6) for p in outline.exterior.coords})
    ys = sorted({round(p[1], 6) for p in outline.exterior.coords})
    grid = np.array([[not outline.contains(box(a, c, b, d).centroid)
                      for a, b in zip(xs, xs[1:])] for c, d in zip(ys, ys[1:])])
    result = []
    while grid.any():
        # Greedy cover by the widest rectangles.
        row, col = map(int, np.argwhere(grid)[0])
        right = col
        while right < grid.shape[1] and grid[row, right]:
            right += 1
        bottom = row
        while bottom < grid.shape[0] and grid[bottom, col:right].all():
            bottom += 1
        grid[row:bottom, col:right] = False
        result.append((2 * (xs[col] - x0) / (x1 - x0) - 1, 2 * (ys[row] - y0) / (y1 - y0) - 1,
                       2 * (xs[right] - x0) / (x1 - x0) - 1, 2 * (ys[bottom] - y0) / (y1 - y0) - 1))
    return result


# ------------------------------------------------------------- 1. the sampler

class Guide:
    """Keeps the sampler's clean-plan estimate inside the outline."""

    def __init__(self, outline, layout):
        self.notches = notches(outline)
        rooms = np.asarray(layout["room_ids"]).reshape(-1)
        valid = np.asarray(layout["corner_valid"]).reshape(-1).astype(bool)
        self.rooms = [np.flatnonzero(valid & (rooms == i)) for i in np.unique(rooms[valid])]

    def __call__(self, x_t, velocity, sigma):
        sigma = float(sigma)
        if not self.notches or not GUIDE_SIGMA[0] <= sigma <= GUIDE_SIGMA[1]:
            return velocity
        clean = (x_t - sigma * velocity)[0].detach().cpu().numpy()
        moved = torch.as_tensor(self.push(clean), dtype=x_t.dtype, device=x_t.device)
        return (x_t - moved[None]) / sigma

    def push(self, clean):
        xy = clean.copy()
        for index in self.rooms:
            for notch in self.notches:
                points = xy[index]
                (left, top), (right, bottom) = points.min(0), points.max(0)
                if (min(right, notch[2]) - max(left, notch[0]) <= 1e-4
                        or min(bottom, notch[3]) - max(top, notch[1]) <= 1e-4):
                    continue
                if Polygon(points).buffer(0).intersection(box(*notch)).area < 1e-5:
                    continue
                size = (max(right - left, 1e-6), max(bottom - top, 1e-6))
                exits = []
                if notch[0] > -1:
                    exits.append((right - notch[0], 0, -1, notch[0]))
                if notch[2] < 1:
                    exits.append((notch[2] - left, 0, 1, notch[2]))
                if notch[1] > -1:
                    exits.append((bottom - notch[1], 1, -1, notch[1]))
                if notch[3] < 1:
                    exits.append((notch[3] - top, 1, 1, notch[3]))
                if not exits:
                    continue
                depth, axis, sign, edge = min(exits, key=lambda e: e[0] / size[e[1]] + 0.3 * e[0])
                if depth > 0.45 * size[axis]:      # mostly outside: move the whole room
                    xy[index, axis] += sign * depth
                else:                               # partly outside: pull its edge in
                    side = xy[index, axis]
                    xy[index, axis] = np.minimum(side, edge) if sign < 0 else np.maximum(side, edge)
        return np.clip(xy, -1.0, 1.0)


# ------------------------------------------------------------ 2. the partition

def _grid_lines(anchors, values, minimum=CELL_M):
    """Anchors plus as few other lines as needed, at least ``minimum`` apart."""
    anchors = sorted(anchors)
    lines = list(anchors)
    free = sorted(v for v in values if anchors[0] < v < anchors[-1]
                  and min(abs(v - a) for a in anchors) >= minimum)
    cluster = []
    for value in free + [None]:
        if cluster and (value is None or value - cluster[0] >= minimum):
            centre = float(np.median(cluster))
            if min(abs(centre - line) for line in lines) >= minimum - 1e-9:
                lines.append(centre)
            cluster = []
        if value is not None:
            cluster.append(value)
    lines = sorted(lines)
    # Split wide gaps so that rooms can exchange part of them.
    fine = [lines[0]]
    for start, end in zip(lines, lines[1:]):
        parts = max(1, int((end - start) / SPLIT_M))
        fine.extend(start + (end - start) * i / parts for i in range(1, parts + 1))
    return np.asarray(fine)


def _make_room(rooms, outline):
    """Give squeezed rooms a usable size and bring them inside the outline."""
    x0, y0, x1, y1 = outline.bounds
    for room in rooms:
        kind = room.type_name
        if kind == "living":
            continue
        a, b, c, d = room.polygon.bounds
        width, height = max(c - a, 0.05), max(d - b, 0.05)
        least_width, least_area = 1.3 * MIN_WIDTH_M.get(kind, 0.9), 1.4 * MIN_AREA_M2.get(kind, 1.5)
        inside = room.polygon.intersection(outline).area
        if inside >= least_area and min(width, height) >= least_width and inside > 0.6 * room.polygon.area:
            continue
        width, height = max(width, least_width), max(height, least_width)
        grow = max(1.0, least_area / (width * height)) ** 0.5
        width, height = min(width * grow, x1 - x0), min(height * grow, y1 - y0)
        centre = ((a + c) / 2, (b + d) / 2)
        shape = None
        # Nearest position that lies (almost) completely on the building.
        reach = max(x1 - x0, y1 - y0)
        for distance in np.linspace(0, reach, 41):
            spots = [(0.0, 0.0)] if distance == 0 else [
                (distance * np.cos(t), distance * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 17)[:-1]]
            placed = [box(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
                      for dx, dy in spots
                      for cx, cy in [(min(max(centre[0] + dx, x0 + width / 2), x1 - width / 2),
                                      min(max(centre[1] + dy, y0 + height / 2), y1 - height / 2))]]
            best = max(placed, key=lambda p: p.intersection(outline).area)
            if best.intersection(outline).area > 0.97 * best.area:
                shape = best
                break
        if shape is not None:
            room.polygon = shape


class _Partition:
    """Room ownership of the cells between the grid lines (-1 empty, -2 not building)."""

    def __init__(self, rooms, outline):
        self.rooms = {r.room_id: r for r in rooms}
        ring = np.asarray(outline.exterior.coords)[:-1]
        xs = [x for r in rooms for x, _ in r.polygon.exterior.coords]
        ys = [y for r in rooms for _, y in r.polygon.exterior.coords]
        self.xs = _grid_lines({round(float(v), 6) for v in ring[:, 0]}, xs)
        self.ys = _grid_lines({round(float(v), 6) for v in ring[:, 1]}, ys)
        self.w, self.h = np.diff(self.xs), np.diff(self.ys)
        self.area = np.outer(self.h, self.w)
        self.owner = np.full(self.area.shape, -2)
        total = sum(SHARE.get(r.type_name, 1.0) for r in rooms)
        self.target = {r.room_id: outline.area * SHARE.get(r.type_name, 1.0) / total for r in rooms}
        for r in range(len(self.h)):
            for c in range(len(self.w)):
                cell = self.cell(r, c)
                if not outline.contains(cell.centroid):
                    continue
                self.owner[r, c] = -1
                shares = [(room.polygon.intersection(cell).area, room) for room in rooms]
                # Where rooms overlap, the smaller one keeps the space.
                claims = [room for shared, room in shares if shared > 0.5 * cell.area]
                if claims:
                    self.owner[r, c] = min(claims, key=lambda x: x.polygon.area).room_id
                else:
                    shared, room = max(shares, key=lambda s: s[0])
                    if shared > 0.35 * cell.area:
                        self.owner[r, c] = room.room_id
        for room in rooms:
            self._keep_largest(room.room_id)

    def cell(self, r, c):
        return box(self.xs[c], self.ys[r], self.xs[c + 1], self.ys[r + 1])

    def room_area(self, room_id):
        return float(self.area[self.owner == room_id].sum())

    def _components(self, mask):
        seen, parts = np.zeros_like(mask), []
        for start in map(tuple, np.argwhere(mask)):
            if seen[start]:
                continue
            part, stack = [], [start]
            seen[start] = True
            while stack:
                r, c = stack.pop()
                part.append((r, c))
                for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1] and mask[q] and not seen[q]:
                        seen[q] = True
                        stack.append(q)
            parts.append(part)
        return parts

    def _keep_largest(self, room_id):
        parts = self._components(self.owner == room_id)
        parts.sort(key=lambda p: sum(self.area[i] for i in p), reverse=True)
        for part in parts[1:]:
            for index in part:
                self.owner[index] = -1

    def _runs(self, room_id):
        """Straight stretches of the room's edge with empty building cells beyond."""
        rows, cols = self.owner.shape
        mine = self.owner == room_id
        for axis, step in ((0, -1), (0, 1), (1, -1), (1, 1)):
            outer, inner = (rows, cols) if axis == 0 else (cols, rows)
            for line in range(outer):
                beyond = line + step
                if not 0 <= beyond < outer:
                    continue
                at = (lambda i: (line, i)) if axis == 0 else (lambda i: (i, line))
                past = (lambda i: (beyond, i)) if axis == 0 else (lambda i: (i, beyond))
                run = []
                for i in range(inner + 1):
                    edge = i < inner and mine[at(i)] and not mine[past(i)]
                    if edge:
                        run.append(i)
                        continue
                    if run:
                        if all(self.owner[past(j)] == -1 for j in run):
                            yield axis, [past(j) for j in run]
                        run = []

    def _shape_cost(self, room_id, cells):
        """How odd the room would look with ``cells`` added (lower is better)."""
        mask = self.owner == room_id
        for index in cells:
            mask[index] = True
        rows, cols = np.any(mask, 1), np.any(mask, 0)
        r0, r1 = np.flatnonzero(rows)[[0, -1]]
        c0, c1 = np.flatnonzero(cols)[[0, -1]]
        width, height = self.xs[c1 + 1] - self.xs[c0], self.ys[r1 + 1] - self.ys[r0]
        area = float(self.area[mask].sum())
        fill = area / (width * height)
        stretch = max(width, height) / min(width, height)
        kind = self.rooms[room_id].type_name
        return (2.5 * area / self.target[room_id]
                + (1.0 if kind == "living" else 5.0) * (1 - fill)
                + max(0.0, stretch - 1.8) * (0.5 if kind in {"living", "balcony"} else 1.5))

    def grow(self):
        """Give every empty cell to a neighbouring room."""
        while (self.owner == -1).any():
            moves = []
            for room_id in self.rooms:
                for _, cells in self._runs(room_id):
                    moves.append((self._shape_cost(room_id, cells), room_id, cells))
            # Cells that no whole edge can reach go to a neighbour one by one.
            if not moves:
                for r, c in map(tuple, np.argwhere(self.owner == -1)):
                    for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= q[0] < self.owner.shape[0] and 0 <= q[1] < self.owner.shape[1] \
                                and self.owner[q] >= 0:
                            owner = int(self.owner[q])
                            moves.append((self._shape_cost(owner, [(r, c)]), owner, [(r, c)]))
            if not moves:
                raise PlotFitError("part of the plot is not reachable from any room")
            _, room_id, cells = min(moves, key=lambda m: m[0])
            for index in cells:
                self.owner[index] = room_id

    def _fits(self, room_id, slack=1.0):
        """Whether the room is one solid piece of a usable size."""
        try:
            shape = self.polygon(room_id)
        except PlotFitError:
            return False
        kind = self.rooms[room_id].type_name
        return (shape.area >= slack * MIN_AREA_M2.get(kind, 1.0)
                and _narrowest(shape) >= slack * MIN_WIDTH_M.get(kind, 0.9))

    def rescue(self):
        """Let rooms that are too small take space from neighbours that can spare it."""
        for _ in range(40):
            small = [i for i in self.rooms if not self._fits(i)]
            if not small:
                return
            best = None
            for room_id in small:
                for cells in self._takeable(room_id):
                    before = self.owner.copy()
                    donors = {int(self.owner[i]) for i in cells}
                    cost = sum(self._shape_cost(d, []) for d in donors)
                    if any(d in small for d in donors):
                        continue
                    gain = self._shape_cost(room_id, cells)
                    for index in cells:
                        self.owner[index] = room_id
                    if all((self.owner == d).any() and self._fits(d) for d in donors):
                        change = gain + sum(self._shape_cost(d, []) for d in donors) - cost
                        if best is None or change < best[0]:
                            best = (change, room_id, cells)
                    self.owner = before
            if best is None:
                return
            for index in best[2]:
                self.owner[index] = best[1]

    def _takeable(self, room_id):
        """Layers of other rooms' cells along each straight edge of a room."""
        rows, cols = self.owner.shape
        mine = self.owner == room_id
        for r_step, c_step in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            layer = {}
            for r, c in map(tuple, np.argwhere(mine)):
                q = (r + r_step, c + c_step)
                if 0 <= q[0] < rows and 0 <= q[1] < cols and not mine[q]:
                    layer.setdefault(q[0] if r_step else q[1], []).append(q)
            for cells in layer.values():
                if all(self.owner[q] >= 0 for q in cells):
                    yield cells

    def polygon(self, room_id):
        cells = [self.cell(r, c) for r, c in map(tuple, np.argwhere(self.owner == room_id))]
        shape = unary_union(cells).simplify(1e-9) if cells else Polygon()
        if shape.is_empty:
            raise PlotFitError(f"{self.rooms[room_id].name} was pushed off the plot")
        if shape.geom_type != "Polygon" or shape.interiors:
            raise PlotFitError(f"{self.rooms[room_id].name} wraps around another room")
        return orient(shape, sign=1.0)


def _narrowest(polygon):
    """Width of the widest square brush that reaches most of the room."""
    low, high = 0.02, min(polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1])
    for _ in range(12):
        middle = (low + high) / 2
        core = polygon.buffer(-middle / 2, join_style=2)
        if not core.is_empty:
            core = core.buffer(middle / 2, join_style=2)
        low, high = (middle, high) if core.area > 0.75 * polygon.area else (low, middle)
    return low


def fit_rooms(records, outline):
    """Make ``records`` (normalised room polygons) fill ``outline`` (metres).

    Edits the records in place and returns what had to change.
    """
    from shapely import affinity

    x0, y0, x1, y1 = outline.bounds
    sx, sy = (x1 - x0) / 2, (y1 - y0) / 2
    for record in records:
        for name in ("polygon", "source_polygon"):
            shape = getattr(record, name, None)
            if shape is not None:
                setattr(record, name, affinity.affine_transform(shape, [sx, 0, 0, sy, x0 + sx, y0 + sy]))
    rooms = [r for r in records if r.type_name not in DOORS]
    drawn = unary_union([r.polygon.buffer(0) for r in rooms])
    spill = drawn.difference(outline).area / outline.area
    empty = outline.difference(drawn).area / outline.area
    _make_room(rooms, outline)
    partition = _Partition(rooms, outline)
    missing = [r.name for r in rooms if partition.room_area(r.room_id) == 0]
    if missing:
        raise PlotFitError(f"no space was left for {', '.join(missing)}")
    partition.grow()
    partition.rescue()
    for room in rooms:
        room.polygon = partition.polygon(room.room_id)
    penalty = 0.0
    for room in rooms:
        area, width = room.polygon.area, _narrowest(room.polygon)
        kind = room.type_name
        if area < MIN_AREA_M2.get(kind, 1.0):
            raise PlotFitError(f"{room.name} is only {area:.1f} m²")
        if width < MIN_WIDTH_M.get(kind, 0.9):
            raise PlotFitError(f"{room.name} is only {width:.1f} m wide")
        ratio = area / partition.target[room.room_id]
        penalty += abs(np.log(ratio)) * (0.5 if kind == "living" else 1.0)
        if kind != "living" and ratio > 2.2:  # e.g. a bathroom the size of a bedroom
            penalty += 2.0 * (ratio - 2.2)
        box_area = room.polygon.envelope.area
        penalty += (0.5 if kind == "living" else 3.0) * (1 - area / box_area)
    covered = unary_union([r.polygon for r in rooms])
    if covered.symmetric_difference(outline).area > 1e-4:
        raise PlotFitError("the rooms do not fill the plot exactly")
    return {"spill_before_fit": float(spill), "empty_before_fit": float(empty),
            "penalty": float(penalty / max(len(rooms), 1))}


def to_canvas(records, outline, canvas=255.0):
    """Scale fitted records from metres to the editor canvas. Returns m per pixel."""
    from shapely import affinity

    x0, y0, x1, y1 = outline.bounds
    scale = max(x1 - x0, y1 - y0) / canvas
    dx = (canvas - (x1 - x0) / scale) / 2
    dy = (canvas - (y1 - y0) / scale) / 2
    matrix = [1 / scale, 0, 0, 1 / scale, dx - x0 / scale, dy - y0 / scale]
    for record in records:
        for name in ("polygon", "source_polygon"):
            shape = getattr(record, name, None)
            if shape is not None:
                setattr(record, name, affinity.affine_transform(shape, matrix))
    return scale, lambda p: [p[0] / scale + matrix[4], p[1] / scale + matrix[5]]


def check_outline(records, outline):
    """Later clean-up must not have changed the outer shape."""
    covered = unary_union([r.polygon for r in records if r.type_name not in DOORS])
    x0, y0, x1, y1 = outline.bounds
    from shapely import affinity
    a, b, c, d = covered.bounds
    scale = max(x1 - x0, y1 - y0) / max(c - a, d - b)
    metres = affinity.affine_transform(covered, [scale, 0, 0, scale, x0 - a * scale, y0 - b * scale])
    if metres.symmetric_difference(outline).area > 0.01:
        raise PlotFitError("clean-up changed the plot outline")


def improve_access(records, pairs, config, living_id):
    """Connect rooms to the living room rather than through a bathroom or bedroom.

    Returns the connections and how many rooms still have poor access.
    """
    from plan_postprocess import _max_shared_boundary_segment

    by_id = {r.room_id: r for r in records}
    pairs = {tuple(sorted(map(int, p))) for p in pairs}
    private = {"bathroom", "balcony", "storage", "bedroom", "kitchen"}

    def reachable():
        seen, queue = {living_id}, [living_id]
        while queue:
            node = queue.pop()
            if node != living_id and by_id[node].type_name in private:
                continue
            for a, b in pairs:
                other = b if a == node else a if b == node else None
                if other is not None and other not in seen:
                    seen.add(other)
                    queue.append(other)
        return seen

    poor = 0
    for record in sorted(records, key=lambda r: r.room_id):
        if record.type_name not in {"bedroom", "kitchen"} or record.room_id in reachable():
            continue
        shared = _max_shared_boundary_segment(record, by_id[living_id])
        if shared >= config.min_door_shared_length:
            pairs.add(tuple(sorted((record.room_id, living_id))))
        else:
            poor += 1
    return pairs, poor


def settle_ensuites(records, pairs, labels, ensuite, sleeping):
    """Name the bedroom an en-suite bathroom really opens into.

    The plan may connect the en-suite to another bedroom-shaped room than
    planned. Bedroom-shaped rooms only differ by name, so the names are swapped.
    Returns new labels and how many en-suites open into no bedroom at all.
    """
    kinds = {r.room_id: r.type_name for r in records}
    labels, lost, taken = dict(labels), 0, set()
    for node in [n for n, label in labels.items() if label == ensuite and n in kinds]:
        hosts = [b if a == node else a for a, b in pairs if node in (a, b)]
        hosts = [h for h in hosts if kinds.get(h) == "bedroom"]
        if not hosts:
            labels[node], lost = "Bathroom", lost + 1
            continue
        host = next((h for h in hosts if labels.get(h) == "Master bedroom"), hosts[0])
        wanted = next((n for n, label in labels.items() if label == "Master bedroom"
                       and kinds.get(n) == "bedroom" and n not in taken), None)
        if wanted is None and labels.get(host) not in sleeping:
            wanted = next((n for n, label in labels.items() if label in sleeping
                           and kinds.get(n) == "bedroom" and n not in taken), None)
        if wanted is not None and host != wanted and host not in taken:
            labels[host], labels[wanted] = labels[wanted], labels[host]
        taken.add(host)
    return labels, lost
