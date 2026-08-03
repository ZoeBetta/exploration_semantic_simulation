"""Frontiers, paper reward, grid A* and continuous local control.

The global map and A* remain grid based because the paper explicitly uses an
occupancy grid and A*.  Three parts are continuous in this version:
1. standard frontiers are extracted from a 5 cm visibility raster generated
   by the continuous laser, not from the 0.5 m occupancy-cell edges;
2. all frontier positions and paths exposed to the controller are metric;
3. the local planner outputs continuous unicycle controls (v, omega).
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# CODE-REVIEW NOTES
# Purpose: Fine geometric frontier extraction, stair candidates, A*, reward terms, target paths, and local continuous control.
# Coordinate convention: array indices are (row=y, column=x), while
# metric positions and GUI points are written as (x, y). Distances are
# metres unless a name explicitly ends in ``_cells`` or ``_px``.
# Reproducibility: stochastic operations must use the seeded RNG passed
# by the run; avoid module-global random calls in simulation logic.
# Separation of concerns: ground truth, robot-built maps, planner state,
# and rendering overlays are intentionally distinct representations.
# When modifying this file, preserve those boundaries and update the
# corresponding ``verify_*.py`` regression test.
# -----------------------------------------------------------------------------

import heapq
import math
import numpy as np
from scipy import ndimage

from building import FLOOR_W, FLOOR_H, CELL_SIZE
from geometry import cell_center_world, region_center_world, world_to_cell
from robot import (
    UNKNOWN, FREE, OCC, LaserBeam,
    FRONTIER_RESOLUTION_M, FRONTIER_SCALE, FRONTIER_W, FRONTIER_H,
)

# --- default reward weights (overridable from the initial dialog) ---
IW_DEFAULT = 1.0
DW_DEFAULT = 1.0
VW_DEFAULT = 0.3
CW_DEFAULT = 3.0
# Objective O6 of the paper: keep a stronger preference for frontiers aligned
# with the current direction of motion.
OW_DEFAULT = 8.0

# Backward-compatible aliases used by older scripts.
IW = IW_DEFAULT
DW = DW_DEFAULT
VW = VW_DEFAULT

VICINITY_RADIUS_M = 5.0
# h(x_f, x_r) and the vicinity cost V are different terms in Eq. 2.  The old
# implementation reused the whole 5 m vicinity radius for h, abruptly setting
# information gain to zero long before the frontier was reached.  This made a
# farther frontier become best and caused ping-pong.  h now fades only in the
# final approach.
HYSTERESIS_ZERO_RADIUS_M = 0.75
HYSTERESIS_FULL_RADIUS_M = 1.50
INFO_GAIN_RADIUS_M = 3.0
ORIENTATION_TOL_DEG = 15.0
UNREACHABLE_CURVE_COST = 6.0
MIN_FINE_FRONTIER_LENGTH_M = 0.15
FINE_FRONTIER_CLUSTER_LENGTH_M = 1.25


def paper_hysteresis_gain(distance_m: float) -> float:
    """Continuous implementation of h(x_f, x_r) from Equation 2.

    In the paper this gain suppresses the information term for a frontier in
    the robot's immediate vicinity. It is deliberately distinct from the
    target-selection persistence bonus implemented in ``simulation.py``.
    """
    distance_m = max(0.0, float(distance_m))
    if distance_m <= HYSTERESIS_ZERO_RADIUS_M:
        return 0.0
    if distance_m >= HYSTERESIS_FULL_RADIUS_M:
        return 1.0
    return ((distance_m - HYSTERESIS_ZERO_RADIUS_M) /
            (HYSTERESIS_FULL_RADIUS_M - HYSTERESIS_ZERO_RADIUS_M))


# --- continuous path following ---
LOOKAHEAD_DISTANCE_M = 0.65
# Physical robot speed. Playback is real-time by default, so this is no longer
# multiplied by the GUI refresh loop as happened with the Matplotlib viewer.
MAX_LINEAR_SPEED_M_S = 0.60
MAX_ANGULAR_SPEED_RAD_S = 1.65
# The grid A* path is converted to a collision-checked metric polyline and then
# sampled densely.  The local controller therefore follows positions every
# 10 cm rather than "jumping" between 0.5 m cell centres.
CONTROLLER_WAYPOINT_SPACING_M = 0.10
PATH_VISIBILITY_SAMPLE_M = 0.05
# Clearance-aware global planning. The hard band excludes cells directly next
# to known walls, while the soft term continues to favour the centre of wide
# passages. A three-cell door remains traversable through its middle cell.
PATH_HARD_WALL_CLEARANCE_CELLS = 1.45
PATH_PREFERRED_WALL_CLEARANCE_CELLS = 4.0
PATH_CLEARANCE_COST_WEIGHT = 5.0
PATH_STRING_PULL_MIN_CLEARANCE_CELLS = 1.45
HEADING_GAIN = 2.4
REPULSION_RANGE_M = 1.05
REPULSION_GAIN = 0.70
CRITICAL_FRONT_DISTANCE_M = 0.22
SLOWDOWN_FRONT_DISTANCE_M = 0.85


class Frontier:
    __slots__ = (
        'x', 'y', 'kind', 'target_floor', 'score', 'path', 'path_cells',
        'goal_cell', 'segments', 'polylines', 'approach_x', 'approach_y',
        'stair_id'
    )

    def __init__(self, x_m: float, y_m: float, kind: str,
                 target_floor: int | None = None,
                 goal_cell: tuple[int, int] | None = None,
                 segments: list[tuple[float, float, float, float]] | None = None,
                 polylines: list[list[tuple[float, float]]] | None = None,
                 approach_point: tuple[float, float] | None = None,
                 stair_id: int | None = None):
        self.x = float(x_m)
        self.y = float(y_m)
        self.kind = kind
        self.target_floor = target_floor
        self.stair_id = stair_id
        self.score: float | None = None
        self.path: list[tuple[float, float]] | None = None
        self.path_cells: list[tuple[int, int]] | None = None
        self.goal_cell = goal_cell
        self.segments = segments or []
        self.polylines = polylines or []
        if approach_point is None:
            approach_point = (self.x, self.y)
        self.approach_x = float(approach_point[0])
        self.approach_y = float(approach_point[1])


def _point_segment_distance(px: float, py: float,
                            segment: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = segment
    vx, vy = x2 - x1, y2 - y1
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-12:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * vx + (py - y1) * vy) / length_sq
    t = max(0.0, min(1.0, t))
    qx = x1 + t * vx
    qy = y1 + t * vy
    return math.hypot(px - qx, py - qy)


def frontier_distance_to_point(frontier: Frontier,
                               x_m: float, y_m: float) -> float:
    """Distance to the complete frontier curve, not only its centroid."""
    if frontier.kind == 'standard' and frontier.segments:
        return min(_point_segment_distance(x_m, y_m, segment)
                   for segment in frontier.segments)
    return math.hypot(frontier.x - x_m, frontier.y - y_m)


def _segment_key(segment):
    values = tuple(round(float(value), 6) for value in segment)
    first = values[:2]
    second = values[2:]
    return first + second if first <= second else second + first


def frontier_segment_overlap(frontier: Frontier, previous_segments) -> float:
    """Fractional overlap with the curve selected at the previous update."""
    if (frontier.kind != 'standard' or not frontier.segments
            or not previous_segments):
        return 0.0
    current = {_segment_key(segment) for segment in frontier.segments}
    previous = {_segment_key(segment) for segment in previous_segments}
    denominator = max(1, min(len(current), len(previous)))
    return len(current.intersection(previous)) / denominator


def _fine_edge_for_direction(cx: int, cy: int, dx: int, dy: int):
    """Return integer corner coordinates of one fine-grid boundary edge."""
    if dx == -1:
        return ((cx, cy), (cx, cy + 1))
    if dx == 1:
        return ((cx + 1, cy), (cx + 1, cy + 1))
    if dy == -1:
        return ((cx, cy), (cx + 1, cy))
    return ((cx, cy + 1), (cx + 1, cy + 1))


def _canonical_edge(a, b):
    return (a, b) if a <= b else (b, a)


def _chain_edges_to_polylines(edges):
    """Chain pixel-boundary edges into ordered geometric polylines."""
    unused = {_canonical_edge(a, b) for a, b in edges if a != b}
    adjacency = {}
    for a, b in unused:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    polylines = []
    while unused:
        seed_a, seed_b = next(iter(unused))
        start = seed_a
        if len(adjacency.get(seed_a, ())) == 2 and len(adjacency.get(seed_b, ())) != 2:
            start = seed_b
        elif len(adjacency.get(seed_a, ())) == 2 and len(adjacency.get(seed_b, ())) == 2:
            # Closed loop: any vertex is a valid start.
            start = seed_a

        points = [start]
        previous = None
        current = start
        while True:
            candidates = []
            for neighbour in adjacency.get(current, ()):
                edge = _canonical_edge(current, neighbour)
                if edge in unused:
                    candidates.append(neighbour)
            if not candidates:
                break
            if previous is not None and len(candidates) > 1:
                non_backtracking = [p for p in candidates if p != previous]
                if non_backtracking:
                    candidates = non_backtracking
            nxt = candidates[0]
            unused.remove(_canonical_edge(current, nxt))
            points.append(nxt)
            previous, current = current, nxt
            if current == start:
                break
        if len(points) >= 2:
            polylines.append(points)
    return polylines



def _split_polyline_by_length(line, max_length=FINE_FRONTIER_CLUSTER_LENGTH_M):
    """Split a long visibility contour into local frontier candidates.

    A complete 180-degree fan can form one U-shaped connected boundary whose
    global centroid lies close to the robot.  Frontier-based exploration needs
    local pieces instead: each piece has a representative point on the actual
    visibility envelope and can receive its own information-gain score.
    """
    if len(line) < 2:
        return []
    chunks = []
    current = [line[0]]
    accumulated = 0.0
    for point in line[1:]:
        previous = current[-1]
        length = math.hypot(point[0] - previous[0], point[1] - previous[1])
        if (accumulated + length > max_length and len(current) >= 2):
            chunks.append(current)
            current = [previous, point]
            accumulated = length
        else:
            current.append(point)
            accumulated += length
    if len(current) >= 2:
        chunks.append(current)
    return chunks


def _polyline_arc_midpoint(line):
    """Return the point at half of the polyline arc length.

    This is the geometric centre used as the navigation goal for a standard
    frontier.  Unlike an arithmetic centroid, it is guaranteed to lie on the
    actual pixel-resolution frontier curve, even when the curve is bent.
    """
    if not line:
        raise ValueError("A frontier polyline cannot be empty")
    if len(line) == 1:
        return float(line[0][0]), float(line[0][1])

    lengths = [
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(line, line[1:])
    ]
    total = sum(lengths)
    if total <= 1e-12:
        return float(line[0][0]), float(line[0][1])

    target = 0.5 * total
    traversed = 0.0
    for first, second, length in zip(line, line[1:], lengths):
        if length <= 1e-12:
            continue
        if traversed + length >= target:
            ratio = (target - traversed) / length
            return (
                float(first[0] + ratio * (second[0] - first[0])),
                float(first[1] + ratio * (second[1] - first[1])),
            )
        traversed += length
    return float(line[-1][0]), float(line[-1][1])

def _known_occupied_fine(fmap):
    """Upsample known occupied cells to the 5 cm frontier raster."""
    return np.repeat(
        np.repeat(fmap.occ == OCC, FRONTIER_SCALE, axis=0),
        FRONTIER_SCALE,
        axis=1,
    )


def _nearest_known_free_cell(fmap, approach_x, approach_y, fine_xs, fine_ys):
    candidates = set()
    for px, py in zip(fine_xs.tolist(), fine_ys.tolist()):
        wx = (px + 0.5) * FRONTIER_RESOLUTION_M
        wy = (py + 0.5) * FRONTIER_RESOLUTION_M
        cx, cy = world_to_cell(wx, wy)
        if 0 <= cx < FLOOR_W and 0 <= cy < FLOOR_H and fmap.occ[cy, cx] == FREE:
            candidates.add((cx, cy))
    if candidates:
        return min(
            candidates,
            key=lambda cell: math.hypot(
                cell_center_world(*cell)[0] - approach_x,
                cell_center_world(*cell)[1] - approach_y,
            ),
        )

    cx, cy = world_to_cell(approach_x, approach_y)
    for radius in range(0, 6):
        local = []
        for y in range(max(0, cy - radius), min(FLOOR_H, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(FLOOR_W, cx + radius + 1)):
                if fmap.occ[y, x] == FREE:
                    local.append((x, y))
        if local:
            return min(
                local,
                key=lambda cell: math.hypot(
                    cell_center_world(*cell)[0] - approach_x,
                    cell_center_world(*cell)[1] - approach_y,
                ),
            )
    return None


def detect_standard_frontiers(fmap, relaxed: bool = False):
    """Extract 5 cm geometric frontiers from the continuous laser visibility.

    The paper occupancy grid remains unchanged and is still used by A*.  This
    function instead finds the boundary between fine pixels already visible to
    the continuous 180-degree laser and fine pixels that have never been in
    line of sight.  Each component is returned as pixel-resolution polylines;
    only its approach cell is projected back to the coarse grid for A*.
    """
    observed = fmap.visibility_observed
    if observed.shape != (FRONTIER_H, FRONTIER_W) or not np.any(observed):
        return []

    known_occupied = _known_occupied_fine(fmap)
    observed_free = observed & (~known_occupied)
    unknown = ~observed
    cross_structure = np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool
    )
    unknown_adjacent = ndimage.binary_dilation(
        unknown, structure=cross_structure
    )
    wall_adjacent = ndimage.binary_dilation(
        known_occupied, structure=cross_structure
    )
    # Normal operation excludes pixels immediately adjacent to a mapped wall.
    # This suppresses false frontiers caused by the finite thickness of a laser
    # return.  During planner-starvation recovery the caller may request a
    # relaxed mask: occupied pixels are still excluded, but free pixels beside
    # a wall are reconsidered.  A* subsequently rejects candidates that are not
    # connected through known free space, so the relaxation cannot create a
    # path through a physical obstacle.
    frontier_mask = observed_free & unknown_adjacent
    if not relaxed:
        frontier_mask &= ~wall_adjacent

    labeled, count = ndimage.label(
        frontier_mask, structure=np.ones((3, 3), dtype=np.int8)
    )
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    frontiers = []

    for label in range(1, count + 1):
        ys, xs = np.where(labeled == label)
        if len(xs) == 0:
            continue
        component_pixels = set(zip(xs.tolist(), ys.tolist()))
        edges = []
        for px, py in component_pixels:
            for dx, dy in directions:
                nx, ny = px + dx, py + dy
                if not (0 <= nx < FRONTIER_W and 0 <= ny < FRONTIER_H):
                    continue
                if unknown[ny, nx]:
                    edges.append(_fine_edge_for_direction(px, py, dx, dy))
        if not edges:
            continue

        raw_polylines = _chain_edges_to_polylines(edges)
        world_chunks = []
        for raw in raw_polylines:
            line = [
                (vx * FRONTIER_RESOLUTION_M,
                 vy * FRONTIER_RESOLUTION_M)
                for vx, vy in raw
            ]
            world_chunks.extend(_split_polyline_by_length(line))

        # Each local chunk is a separate candidate.  This prevents the
        # centroid of a large U-shaped laser fan from falling near the robot.
        for line in world_chunks:
            segments = []
            total_length = 0.0
            for a, b in zip(line, line[1:]):
                length = math.hypot(b[0] - a[0], b[1] - a[1])
                if length <= 1e-12:
                    continue
                segments.append((a[0], a[1], b[0], b[1]))
                total_length += length

            if total_length < MIN_FINE_FRONTIER_LENGTH_M or not segments:
                continue

            # The selected navigation goal is the arc-length centre of the
            # frontier piece.  It therefore lies on the actual 5 cm contour
            # rather than at a centroid that may fall away from a curved line.
            fx, fy = _polyline_arc_midpoint(line)

            nearest_index = min(
                range(len(xs)),
                key=lambda i: (
                    ((xs[i] + 0.5) * FRONTIER_RESOLUTION_M - fx) ** 2
                    + ((ys[i] + 0.5) * FRONTIER_RESOLUTION_M - fy) ** 2
                ),
            )
            approach_x = (float(xs[nearest_index]) + 0.5) * FRONTIER_RESOLUTION_M
            approach_y = (float(ys[nearest_index]) + 0.5) * FRONTIER_RESOLUTION_M
            goal_cell = _nearest_known_free_cell(
                fmap, approach_x, approach_y, xs, ys
            )
            if goal_cell is None:
                continue

            frontiers.append(
                Frontier(
                    fx,
                    fy,
                    'standard',
                    goal_cell=goal_cell,
                    segments=segments,
                    polylines=[line],
                    approach_point=(approach_x, approach_y),
                )
            )
    return frontiers

def detect_stair_frontiers(fmap, floor):
    """Return every discovered UP/DOWN stair in every corridor core."""
    frontiers: list[Frontier] = []
    up_regions = getattr(floor, 'stair_up_regions', None)
    down_regions = getattr(floor, 'stair_down_regions', None)
    if up_regions is None:
        up_regions = [floor.stair_up_region] if floor.stair_up_region else []
    if down_regions is None:
        down_regions = [floor.stair_down_region] if floor.stair_down_region else []

    pairs = []
    pairs.extend((region, floor.index + 1, core_id)
                 for core_id, region in enumerate(up_regions))
    pairs.extend((region, floor.index - 1, core_id)
                 for core_id, region in enumerate(down_regions))
    for region, target, core_id in pairs:
        if region is None:
            continue
        x0, y0, x1, y1 = region
        observed_y, observed_x = np.where(
            fmap.occ[y0:y1 + 1, x0:x1 + 1] == FREE)
        if len(observed_x) == 0:
            continue
        sx, sy = region_center_world(region)
        candidates = [(x0 + int(x), y0 + int(y))
                      for x, y in zip(observed_x, observed_y)]
        goal_cell = min(candidates,
                        key=lambda c: math.hypot(cell_center_world(*c)[0] - sx,
                                                 cell_center_world(*c)[1] - sy))
        frontiers.append(Frontier(
            sx, sy, 'stair', target_floor=target,
            goal_cell=goal_cell, stair_id=core_id,
        ))
    return frontiers


# ------------------------------------------------------------------ A*
def wall_clearance_cells(occ):
    """Distance in grid cells from each cell to the nearest known wall.

    Unknown cells are not treated as physical walls for the soft-clearance
    term, because a standard frontier necessarily lies beside unknown space.
    Map borders are padded as occupied so paths also avoid the outer boundary.
    """
    wall_mask = (occ == OCC)
    free_from_walls = ~wall_mask
    padded = np.pad(free_from_walls, 1, mode="constant", constant_values=False)
    return ndimage.distance_transform_edt(padded)[1:-1, 1:-1].astype(np.float32)


def _clearance_step_multiplier(clearance_cells):
    preferred = PATH_PREFERRED_WALL_CLEARANCE_CELLS
    if clearance_cells >= preferred:
        return 1.0
    deficit = (preferred - max(0.0, float(clearance_cells))) / preferred
    return 1.0 + PATH_CLEARANCE_COST_WEIGHT * deficit * deficit


def a_star(occ, start_cell, goal_cell, max_iters=30000,
           clearance_map=None, hard_clearance_cells=None):
    """8-connected A* over known-free cells with explicit wall clearance.

    The robot pose and final frontier cell are allowed inside the hard band so
    a close start or target is not made unreachable. Intermediate cells next
    to known walls are rejected, and a continuous cost favours the centre of
    rooms, corridors and three-cell doors.
    """
    height, width = occ.shape
    sx, sy = start_cell
    gx, gy = goal_cell
    sx = min(max(int(sx), 0), width - 1)
    sy = min(max(int(sy), 0), height - 1)
    gx = min(max(int(gx), 0), width - 1)
    gy = min(max(int(gy), 0), height - 1)

    if clearance_map is None:
        clearance_map = wall_clearance_cells(occ)
    # The default preserves the normal wall-clearance behaviour.  A smaller
    # value is used only by the recovery pass when every strict candidate is
    # unreachable; it still requires known FREE cells and never crosses OCC.
    if hard_clearance_cells is None:
        hard_clearance_cells = PATH_HARD_WALL_CLEARANCE_CELLS
    hard_clearance_cells = max(0.0, float(hard_clearance_cells))
    endpoints = {(sx, sy), (gx, gy)}

    def known_free(x, y):
        return 0 <= x < width and 0 <= y < height and occ[y, x] == FREE

    def valid(x, y):
        if not known_free(x, y):
            return False
        if (x, y) in endpoints:
            return True
        return clearance_map[y, x] >= hard_clearance_cells

    if not known_free(sx, sy) or not known_free(gx, gy):
        return None

    neighbours = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    ]

    def heuristic(x, y):
        return math.hypot(x - gx, y - gy)

    open_heap = [(heuristic(sx, sy), 0.0, (sx, sy))]
    came_from = {}
    g_score = {(sx, sy): 0.0}
    closed = set()

    iterations = 0
    while open_heap and iterations < max_iters:
        iterations += 1
        _, cost, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        if current == (gx, gy):
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        for dx, dy, step_cost in neighbours:
            nx, ny = cx + dx, cy + dy
            if not valid(nx, ny):
                continue
            if dx != 0 and dy != 0:
                if not valid(cx + dx, cy) or not valid(cx, cy + dy):
                    continue
            new_cost = (
                cost
                + step_cost * _clearance_step_multiplier(clearance_map[ny, nx])
            )
            if new_cost < g_score.get((nx, ny), math.inf):
                g_score[(nx, ny)] = new_cost
                came_from[(nx, ny)] = current
                heapq.heappush(
                    open_heap,
                    (new_cost + heuristic(nx, ny), new_cost, (nx, ny)),
                )
    return None


def _segment_is_known_free(occ, start, end, clearance_map=None):
    """Check line of sight without undoing A*'s safer wall clearance."""
    if clearance_map is None:
        clearance_map = wall_clearance_cells(occ)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    samples = max(1, int(math.ceil(length / PATH_VISIBILITY_SAMPLE_M)))
    height, width = occ.shape
    previous_cell = None
    for index in range(samples + 1):
        t = index / samples
        x = start[0] + t * dx
        y = start[1] + t * dy
        cx, cy = world_to_cell(x, y)
        if not (0 <= cx < width and 0 <= cy < height):
            return False
        if occ[cy, cx] != FREE:
            return False
        if (index not in (0, samples) and
                clearance_map[cy, cx] <
                PATH_STRING_PULL_MIN_CLEARANCE_CELLS):
            return False
        if previous_cell is not None:
            px, py = previous_cell
            if cx != px and cy != py:
                if not (0 <= cx < width and 0 <= py < height and
                        0 <= px < width and 0 <= cy < height):
                    return False
                if occ[py, cx] != FREE or occ[cy, px] != FREE:
                    return False
                if index not in (0, samples):
                    if (clearance_map[py, cx] <
                            PATH_STRING_PULL_MIN_CLEARANCE_CELLS or
                            clearance_map[cy, px] <
                            PATH_STRING_PULL_MIN_CLEARANCE_CELLS):
                        return False
        previous_cell = (cx, cy)
    return True


def _string_pull_metric_path(occ, points, clearance_map=None):
    """Simplify A* only across known-free, sufficiently clear segments."""
    if len(points) <= 2:
        return points
    if clearance_map is None:
        clearance_map = wall_clearance_cells(occ)
    simplified = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        candidate = len(points) - 1
        while candidate > anchor + 1:
            if _segment_is_known_free(
                    occ, points[anchor], points[candidate], clearance_map):
                break
            candidate -= 1
        simplified.append(points[candidate])
        anchor = candidate
    return simplified


def _densify_metric_path(points, spacing=CONTROLLER_WAYPOINT_SPACING_M):
    """Sample a metric polyline at approximately uniform sub-cell spacing."""
    if not points:
        return None
    dense = [points[0]]
    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        pieces = max(1, int(math.ceil(length / spacing)))
        for index in range(1, pieces + 1):
            t = index / pieces
            dense.append((start[0] + t * dx, start[1] + t * dy))
    return dense


def _append_metric_endpoint(path, endpoint,
                            spacing=CONTROLLER_WAYPOINT_SPACING_M):
    """Densely append an exact continuous endpoint to an existing path.

    Standard frontier centres lie on the boundary between visible free space
    and unknown space.  A* therefore stops in the nearest known-free coarse
    cell, while this final short segment takes the continuous controller to the
    actual centre of the 5 cm frontier curve.
    """
    if not path:
        return None
    endpoint = (float(endpoint[0]), float(endpoint[1]))
    if math.hypot(endpoint[0] - path[-1][0],
                  endpoint[1] - path[-1][1]) <= 1e-9:
        result = list(path)
        result[-1] = endpoint
        return result
    tail = _densify_metric_path([path[-1], endpoint], spacing)
    return list(path) + tail[1:]


def _metric_path(robot, path_cells, occ, clearance_map=None, final_point=None):
    if not path_cells:
        return None
    if clearance_map is None:
        clearance_map = wall_clearance_cells(occ)
    raw = [(robot.x, robot.y)]
    raw.extend(cell_center_world(x, y) for x, y in path_cells[1:])
    if final_point is not None:
        if math.hypot(final_point[0] - raw[-1][0],
                      final_point[1] - raw[-1][1]) > 1e-6:
            if _segment_is_known_free(
                    occ, raw[-1], final_point, clearance_map):
                raw.append(final_point)
    smooth = _string_pull_metric_path(occ, raw, clearance_map)
    return _densify_metric_path(smooth)


def path_length_m(path):
    if not path or len(path) < 2:
        return 0.0
    return sum(math.hypot(path[i][0] - path[i - 1][0],
                          path[i][1] - path[i - 1][1])
               for i in range(1, len(path)))


# ------------------------------------------------------------------ reward terms
def floor_gain(fmap, floor):
    explored = float(np.sum(fmap.occ == FREE)) * CELL_SIZE ** 2
    return max(0.0, floor.explorable_area_m2 - explored)


def information_gain(fmap, fx_m, fy_m):
    """Unknown area in a disk, evaluated on the 5 cm visibility raster."""
    observed = getattr(fmap, 'visibility_observed', None)
    if observed is not None and observed.shape == (FRONTIER_H, FRONTIER_W):
        resolution = FRONTIER_RESOLUTION_M
        cx = int(math.floor(fx_m / resolution))
        cy = int(math.floor(fy_m / resolution))
        radius = int(math.ceil(INFO_GAIN_RADIUS_M / resolution))
        x0 = max(0, cx - radius)
        x1 = min(FRONTIER_W, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(FRONTIER_H, cy + radius + 1)
        xs = (np.arange(x0, x1, dtype=np.float32) + 0.5) * resolution
        ys = (np.arange(y0, y1, dtype=np.float32) + 0.5) * resolution
        disk = (
            (xs[None, :] - fx_m) ** 2
            + (ys[:, None] - fy_m) ** 2
            <= INFO_GAIN_RADIUS_M ** 2
        )
        unknown = ~observed[y0:y1, x0:x1]
        return float(np.count_nonzero(disk & unknown)) * resolution ** 2

    # Compatibility fallback for maps created by older scripts.
    cx, cy = world_to_cell(fx_m, fy_m)
    radius_cells = int(math.ceil(INFO_GAIN_RADIUS_M / CELL_SIZE))
    x0 = max(0, cx - radius_cells)
    x1 = min(FLOOR_W, cx + radius_cells + 1)
    y0 = max(0, cy - radius_cells)
    y1 = min(FLOOR_H, cy + radius_cells + 1)
    count = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            if fmap.occ[y, x] != UNKNOWN:
                continue
            wx, wy = cell_center_world(x, y)
            if (wx - fx_m) ** 2 + (wy - fy_m) ** 2 <= INFO_GAIN_RADIUS_M ** 2:
                count += 1
    return count * CELL_SIZE ** 2

def score_frontiers(
    frontiers,
    robot,
    fmap,
    floor,
    all_floors,
    Fw,
    Cw,
    Ow,
    Iw=IW_DEFAULT,
    Dw=DW_DEFAULT,
    Vw=VW_DEFAULT,
    hard_clearance_cells=None,
):
    """Assign Eq. 2 / Eq. 5 scores and metric A* paths.

    ``Fw`` and ``Opt`` are the only parameters used to create experimental
    combinations.  The other scalar weights are supplied by ``RunConfig`` and
    remain fixed across all combinations of the same experiment.
    """
    start_cell = world_to_cell(robot.x, robot.y)
    clearance_map = wall_clearance_cells(fmap.occ)

    for frontier in frontiers:
        if frontier.goal_cell is None:
            frontier.score = -math.inf
            frontier.path = None
            continue
        path_cells = a_star(
            fmap.occ,
            start_cell,
            frontier.goal_cell,
            clearance_map=clearance_map,
            hard_clearance_cells=hard_clearance_cells,
        )
        frontier.path_cells = path_cells
        if path_cells is None:
            frontier.path = None
            frontier.score = -math.inf
            continue

        # Keep the Eq. 4 curve cost tied to the original A* path, exactly as
        # in the grid model.  Only the copy given to the local controller is
        # string-pulled and densely resampled for smooth continuous motion.
        raw_metric_path = [(robot.x, robot.y)]
        raw_metric_path.extend(
            cell_center_world(x, y) for x, y in path_cells[1:])

        # A* reaches a known-free coarse cell.  For a standard frontier we first
        # connect to the observed-side approach pixel, then append the exact
        # arc-length centre of the geometric frontier as the controller goal.
        # A stair frontier already has its centre in known free space.
        approach_point = (frontier.approach_x, frontier.approach_y)
        if math.hypot(approach_point[0] - raw_metric_path[-1][0],
                      approach_point[1] - raw_metric_path[-1][1]) > 1e-6:
            if _segment_is_known_free(
                    fmap.occ, raw_metric_path[-1], approach_point,
                    clearance_map):
                raw_metric_path.append(approach_point)

        exact_goal = (frontier.x, frontier.y)
        if frontier.kind == 'standard':
            if math.hypot(exact_goal[0] - raw_metric_path[-1][0],
                          exact_goal[1] - raw_metric_path[-1][1]) > 1e-9:
                raw_metric_path.append(exact_goal)
            controller_path = _metric_path(
                robot,
                path_cells,
                fmap.occ,
                clearance_map,
                final_point=approach_point,
            )
            frontier.path = _append_metric_endpoint(
                controller_path, exact_goal
            )
        else:
            frontier.path = _metric_path(
                robot,
                path_cells,
                fmap.occ,
                clearance_map,
                final_point=exact_goal,
            )

        distance = math.hypot(frontier.x - robot.x, frontier.y - robot.y)
        path_length = path_length_m(raw_metric_path)
        curve_cost = min((path_length / distance) if distance > 1e-6 else 1.0,
                         UNREACHABLE_CURVE_COST)

        rel = (math.atan2(frontier.y - robot.y, frontier.x - robot.x)
               - robot.theta + math.pi) % (2 * math.pi) - math.pi
        orientation_gain = (1.0 if abs(math.degrees(rel)) <= ORIENTATION_TOL_DEG
                            else 0.0)

        if distance < VICINITY_RADIUS_M:
            vicinity = (VICINITY_RADIUS_M - distance) / VICINITY_RADIUS_M
        else:
            vicinity = 0.0

        # Paper term h(x_f, x_r): suppress only a frontier that is genuinely
        # in the immediate vicinity of the robot. A continuous ramp avoids a
        # discontinuous score jump at one arbitrary distance.
        hysteresis = paper_hysteresis_gain(distance)

        if frontier.kind == 'standard':
            info = information_gain(fmap, frontier.x, frontier.y)
            gain_floor = floor_gain(fmap, floor)
            frontier.score = (Iw * hysteresis * info - Dw * distance
                              - Vw * vicinity + Fw * gain_floor
                              - Cw * curve_cost + Ow * orientation_gain)
        else:
            target = frontier.target_floor
            if target is not None and 0 <= target < len(all_floors):
                target_floor = all_floors[target]
                gain_floor = (target_floor.explorable_area_m2
                              if not target_floor.fmap.visited_any
                              else floor_gain(target_floor.fmap, target_floor))
            else:
                gain_floor = 0.0
            frontier.score = (-Dw * distance - Vw * vicinity
                              + Fw * gain_floor + Ow * orientation_gain
                              - Cw * curve_cost)
    return frontiers


def best_frontier(frontiers):
    reachable = [f for f in frontiers
                 if f.score is not None and f.score > -math.inf]
    return max(reachable, key=lambda f: f.score) if reachable else None


# ------------------------------------------------------------------ local planner
def _lookahead_point(robot, path, lookahead_m=LOOKAHEAD_DISTANCE_M):
    if not path:
        return None
    # Find the path point nearest to the current continuous pose.
    nearest = min(range(len(path)),
                  key=lambda i: math.hypot(path[i][0] - robot.x,
                                           path[i][1] - robot.y))
    accumulated = 0.0
    previous = (robot.x, robot.y)
    for i in range(nearest, len(path)):
        point = path[i]
        accumulated += math.hypot(point[0] - previous[0],
                                  point[1] - previous[1])
        if accumulated >= lookahead_m:
            return point
        previous = point
    return path[-1]


def continuous_velocity_command(robot, path,
                                scan: list[LaserBeam] | None,
                                rotate_only: bool = False):
    """Pure-pursuit-like control with a mild continuous laser repulsion.

    The A* path dominates the command.  Repulsion only modifies the local
    heading, so the controller does not jump from grid cell to grid cell and
    does not teleport along the path.
    """
    target = _lookahead_point(robot, path)
    if target is None:
        return 0.0, 0.0

    dx = target[0] - robot.x
    dy = target[1] - robot.y
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return 0.0, 0.0
    attractive_x = dx / norm
    attractive_y = dy / norm

    repulsive_x = 0.0
    repulsive_y = 0.0
    repulsive_count = 0
    front_distance = math.inf
    for beam in scan or []:
        if abs(beam.relative_angle) <= math.radians(20.0):
            front_distance = min(front_distance, beam.distance_m)
        if not beam.hit or beam.distance_m >= REPULSION_RANGE_M:
            continue
        weight = ((REPULSION_RANGE_M - max(beam.distance_m, 1e-3))
                  / REPULSION_RANGE_M) ** 2
        repulsive_x -= math.cos(beam.world_angle) * weight
        repulsive_y -= math.sin(beam.world_angle) * weight
        repulsive_count += 1

    if repulsive_count:
        repulsive_x /= repulsive_count
        repulsive_y /= repulsive_count

    desired_x = attractive_x + REPULSION_GAIN * repulsive_x
    desired_y = attractive_y + REPULSION_GAIN * repulsive_y
    if abs(desired_x) + abs(desired_y) < 1e-9:
        desired_x, desired_y = attractive_x, attractive_y

    desired_heading = math.atan2(desired_y, desired_x)
    error = (desired_heading - robot.theta + math.pi) % (2 * math.pi) - math.pi
    omega = max(-MAX_ANGULAR_SPEED_RAD_S,
                min(MAX_ANGULAR_SPEED_RAD_S, HEADING_GAIN * error))

    if rotate_only:
        return 0.0, omega if abs(omega) > 0.15 else MAX_ANGULAR_SPEED_RAD_S * 0.5

    alignment = max(0.0, math.cos(error))
    speed = MAX_LINEAR_SPEED_M_S * alignment
    if front_distance < CRITICAL_FRONT_DISTANCE_M:
        speed = 0.0
    elif front_distance < SLOWDOWN_FRONT_DISTANCE_M:
        scale = ((front_distance - CRITICAL_FRONT_DISTANCE_M) /
                 (SLOWDOWN_FRONT_DISTANCE_M - CRITICAL_FRONT_DISTANCE_M))
        speed *= max(0.0, min(1.0, scale))
    return speed, omega
