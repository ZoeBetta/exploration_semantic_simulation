"""Robot state, continuous LiDAR, camera and maps.

Hybrid representation:
- the 0.5 m occupancy grid used by A*, reward terms and paper metrics is kept
  unchanged for experimental comparability;
- a second probabilistic occupancy grid is built at 5 cm resolution only from
  the area actually observed by the laser and is used by the right-hand live
  view, like a small SLAM occupancy map with known robot pose;
- the semantic map remains at the 0.5 m resolution used by the paper;
- the robot pose and LiDAR rays are continuous metric quantities.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# CODE-REVIEW NOTES
# Purpose: Robot/map data structures and sensor models, including 5 cm probabilistic occupancy mapping.
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

from dataclasses import dataclass
import math
import random
import numpy as np

from building import FLOOR_W, FLOOR_H, CELL_SIZE, is_blocking
from geometry import (cell_center_world, line_of_sight_continuous,
                      raycast_grid_continuous, world_to_cell,
                      point_is_traversable)

# occupancy codes in the map built by the robot
UNKNOWN, FREE, OCC = -1, 0, 1

# --- continuous LiDAR ---
LIDAR_RANGE_M = 4.0
LIDAR_FOV_DEG = 180.0
LIDAR_N_RAYS = 181          # one ray per degree
LIDAR_NOISE_STD_M = 0.0     # can be increased without changing geometry

# Fine visibility raster used only for geometric frontiers.  The occupancy
# and semantic maps remain at the 0.5 m resolution used by the paper, while
# the boundary between visible and not-yet-visible space is stored at 5 cm.
FRONTIER_RESOLUTION_M = 0.05
FRONTIER_SCALE = int(round(CELL_SIZE / FRONTIER_RESOLUTION_M))
if not math.isclose(FRONTIER_SCALE * FRONTIER_RESOLUTION_M, CELL_SIZE):
    raise RuntimeError('FRONTIER_RESOLUTION_M must divide CELL_SIZE')
FRONTIER_W = FLOOR_W * FRONTIER_SCALE
FRONTIER_H = FLOOR_H * FRONTIER_SCALE

# Pixel-resolution probabilistic occupancy map shown in the live viewer.
# It intentionally uses the same 5 cm raster as the geometric frontiers, but
# it is a distinct state: ``visibility_observed`` remains the smooth laser fan
# used for frontier extraction, whereas the arrays below are the SLAM-like
# occupancy estimate displayed on the right.
PIXEL_OCCUPANCY_RESOLUTION_M = FRONTIER_RESOLUTION_M
PIXEL_OCCUPANCY_W = FRONTIER_W
PIXEL_OCCUPANCY_H = FRONTIER_H
PIXEL_LOG_ODDS_FREE = -0.38
PIXEL_LOG_ODDS_OCCUPIED = 0.95
PIXEL_LOG_ODDS_MIN = -5.0
PIXEL_LOG_ODDS_MAX = 5.0

# --- semantic camera ---
CAMERA_RANGE_M = 4.0
CAMERA_FOV_DEG = 180.0

# --- semantic map parameters ---
SEMANTIC_THRESHOLD = 0.8
LAMBDA_NEIGHBOR = 0.5
DECAY_RATE = 0.03
DECAY_EVERY = 4

# --- person detector noise ---
MISS_PROB = 0.15
DETECTION_NOISE_STD = 0.12
FALSE_POSITIVE_PROB = 0.01


@dataclass(frozen=True)
class LaserBeam:
    relative_angle: float
    world_angle: float
    distance_m: float
    hit: bool
    endpoint_x_m: float
    endpoint_y_m: float
    hit_cell: tuple[int, int] | None = None


class FloorMap:
    """Maps constructed by the robot for one floor."""

    def __init__(self):
        # Coarse map retained for the planner and all paper-derived metrics.
        self.occ = np.full((FLOOR_H, FLOOR_W), UNKNOWN, dtype=np.int8)
        self.semantic = np.full((FLOOR_H, FLOOR_W), 0.5, dtype=np.float32)

        # Fine probabilistic occupancy grid for the SLAM-like visual map.
        # Unknown pixels have log-odds 0 but are distinguished by the observed
        # mask.  Only pixels actually visible to the current laser scan are
        # updated; nothing behind a measured obstacle is revealed.
        self.pixel_occ_log_odds = np.zeros(
            (PIXEL_OCCUPANCY_H, PIXEL_OCCUPANCY_W), dtype=np.float32
        )
        self.pixel_occ_observed = np.zeros(
            (PIXEL_OCCUPANCY_H, PIXEL_OCCUPANCY_W), dtype=bool
        )

        # True where the continuous 180-degree laser has actually provided
        # line-of-sight information.  This field is ten times finer in each
        # direction than the occupancy grid and is not used by A*.
        self.visibility_observed = np.zeros(
            (FRONTIER_H, FRONTIER_W), dtype=bool
        )
        self.visited_any = False

    def pixel_occupancy_probability(self) -> np.ndarray:
        """Return P(occupied) for the 5 cm map.

        The observed mask must still be consulted: an unobserved pixel has a
        mathematically neutral probability of 0.5, but is rendered as unknown
        rather than as an observed uncertain cell.
        """
        clipped = np.clip(
            self.pixel_occ_log_odds,
            PIXEL_LOG_ODDS_MIN,
            PIXEL_LOG_ODDS_MAX,
        )
        return 1.0 / (1.0 + np.exp(-clipped))


class Robot:
    """Continuous unicycle pose in metres and radians."""

    def __init__(self, floor_index: int, x_m: float, y_m: float,
                 theta: float = 0.0):
        self.floor = floor_index
        self.x = float(x_m)
        self.y = float(y_m)
        self.theta = float(theta)
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.last_scan: list[LaserBeam] = []



def _mark_coarse_cell_observed_in_fine_map(
    fmap: FloorMap, cell_x: int, cell_y: int
) -> None:
    """Mark the full 0.5 m cell as observed in the fine visibility raster."""
    if not (0 <= cell_x < FLOOR_W and 0 <= cell_y < FLOOR_H):
        return
    x0 = cell_x * FRONTIER_SCALE
    y0 = cell_y * FRONTIER_SCALE
    fmap.visibility_observed[
        y0:y0 + FRONTIER_SCALE,
        x0:x0 + FRONTIER_SCALE,
    ] = True


def _fine_scan_visibility_patch(
    robot: Robot,
    beams: list[LaserBeam],
):
    """Return the local 5 cm visibility mask generated by one laser scan.

    For every fine pixel centre in the local sensor square, the nearest laser
    beam in angle is selected.  The pixel is visible when its radial distance
    is not greater than that beam's measured range.  Thus the frontier follows
    the continuous 180-degree visibility envelope instead of the edges of the
    0.5 m occupancy cells.

    The returned tuple is ``(x0, x1, y0, y1, visible)``.  Keeping this helper
    side-effect free lets the same exact sensor footprint update both the fine
    frontier raster and the probabilistic occupancy map.
    """
    if not beams:
        return None

    resolution = FRONTIER_RESOLUTION_M
    half_fov = math.radians(LIDAR_FOV_DEG / 2.0)
    x0 = max(0, int(math.floor((robot.x - LIDAR_RANGE_M) / resolution)))
    x1 = min(FRONTIER_W, int(math.ceil((robot.x + LIDAR_RANGE_M) / resolution)))
    y0 = max(0, int(math.floor((robot.y - LIDAR_RANGE_M) / resolution)))
    y1 = min(FRONTIER_H, int(math.ceil((robot.y + LIDAR_RANGE_M) / resolution)))
    if x0 >= x1 or y0 >= y1:
        return None

    xs = (np.arange(x0, x1, dtype=np.float32) + 0.5) * resolution
    ys = (np.arange(y0, y1, dtype=np.float32) + 0.5) * resolution
    dx = xs[None, :] - robot.x
    dy = ys[:, None] - robot.y
    distances = np.hypot(dx, dy)
    relative = (np.arctan2(dy, dx) - robot.theta + math.pi) % (
        2.0 * math.pi
    ) - math.pi

    inside = (
        (np.abs(relative) <= half_fov + 1e-7)
        & (distances <= LIDAR_RANGE_M + 0.5 * resolution)
    )
    alpha = (relative + half_fov) / max(2.0 * half_fov, 1e-9)
    indices = np.rint(alpha * (len(beams) - 1)).astype(np.int32)
    indices = np.clip(indices, 0, len(beams) - 1)
    measured_ranges = np.fromiter(
        (beam.distance_m for beam in beams), dtype=np.float32, count=len(beams)
    )
    visible = inside & (
        distances <= measured_ranges[indices] + 0.55 * resolution
    )
    return x0, x1, y0, y1, visible


def _update_fine_visibility_from_laser(
    robot: Robot,
    beams: list[LaserBeam],
    fmap: FloorMap,
) -> None:
    """Rasterise the smooth laser fan used by geometric frontiers."""
    patch = _fine_scan_visibility_patch(robot, beams)
    if patch is None:
        return
    x0, x1, y0, y1, visible = patch
    fmap.visibility_observed[y0:y1, x0:x1] |= visible


def _pixel_cell_from_world(x_m: float, y_m: float) -> tuple[int, int]:
    """Convert metric coordinates to the fine occupancy raster."""
    px = int(math.floor(x_m / PIXEL_OCCUPANCY_RESOLUTION_M))
    py = int(math.floor(y_m / PIXEL_OCCUPANCY_RESOLUTION_M))
    return (
        min(max(px, 0), PIXEL_OCCUPANCY_W - 1),
        min(max(py, 0), PIXEL_OCCUPANCY_H - 1),
    )


def _bresenham_pixel_cells(
    x0: int, y0: int, x1: int, y1: int
) -> list[tuple[int, int]]:
    """Fine-grid cells crossed by one beam, as in the original simulator."""
    cells: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx - dy

    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            return cells
        doubled = 2 * error
        if doubled > -dy:
            error -= dy
            x += sx
        if doubled < dx:
            error += dx
            y += sy


def _beam_endpoint_pixel(beam: LaserBeam) -> tuple[int, int]:
    """Return a robust fine endpoint, nudged inside the hit obstacle.

    Continuous ray casting returns the exact surface coordinate.  A coordinate
    on a cell boundary can be rounded to the free side, so a hit is advanced by
    half a fine pixel and then clamped inside the known coarse hit cell.
    """
    x_m = beam.endpoint_x_m
    y_m = beam.endpoint_y_m
    if beam.hit:
        nudge = 0.55 * PIXEL_OCCUPANCY_RESOLUTION_M
        x_m += nudge * math.cos(beam.world_angle)
        y_m += nudge * math.sin(beam.world_angle)
    px, py = _pixel_cell_from_world(x_m, y_m)

    if beam.hit_cell is not None:
        coarse_x, coarse_y = beam.hit_cell
        min_x = coarse_x * FRONTIER_SCALE
        max_x = (coarse_x + 1) * FRONTIER_SCALE - 1
        min_y = coarse_y * FRONTIER_SCALE
        max_y = (coarse_y + 1) * FRONTIER_SCALE - 1
        px = min(max(px, min_x), max_x)
        py = min(max(py, min_y), max_y)
    return px, py


def _update_pixel_occupancy_from_laser(
    robot: Robot,
    beams: list[LaserBeam],
    fmap: FloorMap,
) -> None:
    """Update the 5 cm probabilistic occupancy grid from one laser scan.

    This is an inverse sensor model with known pose:

    Each of the 181 measured beams is ray-traced through the 5 cm grid, using
    the same Bresenham free/occupied update pattern as the attached original
    simulator: all crossed pixels are free and the final return pixel is
    occupied. Pixels not crossed by a beam remain untouched. The hidden
    thickness of walls and objects therefore stays unknown.
    """
    if not beams:
        return

    start_x, start_y = _pixel_cell_from_world(robot.x, robot.y)
    scan_free = np.zeros_like(fmap.pixel_occ_observed)
    scan_occupied = np.zeros_like(fmap.pixel_occ_observed)

    for beam in beams:
        end_x, end_y = _beam_endpoint_pixel(beam)
        cells = _bresenham_pixel_cells(start_x, start_y, end_x, end_y)
        if beam.hit and cells:
            free_cells = cells[:-1]
            occupied_cell = cells[-1]
            scan_occupied[occupied_cell[1], occupied_cell[0]] = True
        else:
            free_cells = cells

        for px, py in free_cells:
            scan_free[py, px] = True

    # A return dominates a free update if neighbouring beams quantise to the
    # same fine pixel. Each pixel is updated at most once per scan.
    scan_free &= ~scan_occupied
    fmap.pixel_occ_log_odds[scan_free] += PIXEL_LOG_ODDS_FREE
    fmap.pixel_occ_log_odds[scan_occupied] += PIXEL_LOG_ODDS_OCCUPIED
    fmap.pixel_occ_observed |= scan_free | scan_occupied

    np.clip(
        fmap.pixel_occ_log_odds,
        PIXEL_LOG_ODDS_MIN,
        PIXEL_LOG_ODDS_MAX,
        out=fmap.pixel_occ_log_odds,
    )

def lidar_scan(robot: Robot, floor, fmap: FloorMap,
               rng: random.Random | None = None) -> list[LaserBeam]:
    """Simulate a 180-degree continuous laser and update both map scales.

    Distances are computed exactly to the first occupied cell boundary using
    continuous DDA traversal.  The 0.5 m map remains available to the planner,
    while the 5 cm log-odds map records only the laser-visible region.
    """
    beams: list[LaserBeam] = []
    half_fov = math.radians(LIDAR_FOV_DEG / 2.0)

    for k in range(LIDAR_N_RAYS):
        alpha = k / max(1, LIDAR_N_RAYS - 1)
        relative = -half_fov + alpha * (2.0 * half_fov)
        world_angle = robot.theta + relative
        result = raycast_grid_continuous(
            floor.grid, robot.x, robot.y, world_angle, LIDAR_RANGE_M)

        measured = result.distance
        if rng is not None and LIDAR_NOISE_STD_M > 0.0:
            measured += rng.gauss(0.0, LIDAR_NOISE_STD_M)
            measured = max(0.0, min(LIDAR_RANGE_M, measured))

        ex = robot.x + measured * math.cos(world_angle)
        ey = robot.y + measured * math.sin(world_angle)
        beams.append(LaserBeam(
            relative,
            world_angle,
            measured,
            result.hit,
            ex,
            ey,
            result.hit_cell,
        ))

        for cx, cy in result.free_cells:
            if 0 <= cx < FLOOR_W and 0 <= cy < FLOOR_H and fmap.occ[cy, cx] != OCC:
                fmap.occ[cy, cx] = FREE
        if result.hit_cell is not None:
            hx, hy = result.hit_cell
            if 0 <= hx < FLOOR_W and 0 <= hy < FLOOR_H:
                fmap.occ[hy, hx] = OCC
                # A laser return also tells us that the corresponding coarse
                # obstacle cell is known; mark it observed in the fine field
                # so it is not mistaken for unexplored free space.
                _mark_coarse_cell_observed_in_fine_map(fmap, hx, hy)

    _update_fine_visibility_from_laser(robot, beams, fmap)
    _update_pixel_occupancy_from_laser(robot, beams, fmap)

    # The robot's own cell is necessarily free.
    rx, ry = world_to_cell(robot.x, robot.y)
    if 0 <= rx < FLOOR_W and 0 <= ry < FLOOR_H:
        fmap.occ[ry, rx] = FREE
    fmap.visited_any = True
    robot.last_scan = beams
    return beams


def camera_scan(robot: Robot, floor, fmap: FloorMap,
                rng: random.Random):
    """Front camera with continuous FOV, distance and line of sight."""
    detections: list[tuple[int, int, float]] = []
    half_fov = math.radians(CAMERA_FOV_DEG / 2.0)

    for vx_cell, vy_cell in floor.victims:
        vx, vy = cell_center_world(vx_cell, vy_cell)
        ddx, ddy = vx - robot.x, vy - robot.y
        dist = math.hypot(ddx, ddy)
        if dist > CAMERA_RANGE_M or dist < 1e-9:
            continue
        rel = (math.atan2(ddy, ddx) - robot.theta + math.pi) % (2 * math.pi) - math.pi
        if abs(rel) > half_fov:
            continue
        if not line_of_sight_continuous(floor.grid, robot.x, robot.y, vx, vy):
            continue
        if rng.random() < MISS_PROB:
            continue
        base_conf = max(0.05, min(0.99, 1.0 - dist / CAMERA_RANGE_M))
        conf = max(0.0, min(1.0,
                            base_conf + rng.gauss(0.0, DETECTION_NOISE_STD)))
        detections.append((vx_cell, vy_cell, conf))

    # Occasional false positive in a continuous traversable position.
    if rng.random() < FALSE_POSITIVE_PROB:
        for _ in range(20):
            radius = CAMERA_RANGE_M * math.sqrt(rng.random())
            angle = robot.theta + rng.uniform(-half_fov, half_fov)
            fx_m = robot.x + radius * math.cos(angle)
            fy_m = robot.y + radius * math.sin(angle)
            if point_is_traversable(floor.grid, fx_m, fy_m) and \
                    line_of_sight_continuous(floor.grid, robot.x, robot.y, fx_m, fy_m):
                fx, fy = world_to_cell(fx_m, fy_m)
                detections.append((fx, fy, rng.uniform(0.4, 0.7)))
                break

    # Algorithm-1-style update: detected cell and its 8-neighbours.
    for mx, my, conf in detections:
        updates = [(mx, my, conf)]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = mx + dx, my + dy
                if 0 <= nx < FLOOR_W and 0 <= ny < FLOOR_H:
                    updates.append((nx, ny, LAMBDA_NEIGHBOR * conf))
        for cx, cy, value in updates:
            prob = min(max(0.5 * value + 0.5, 1e-4), 1 - 1e-4)
            prev = float(np.clip(fmap.semantic[cy, cx], 1e-4, 1 - 1e-4))
            log_odd = math.log(prob / (1 - prob)) + math.log(prev / (1 - prev))
            fmap.semantic[cy, cx] = 1.0 / (1.0 + math.exp(-log_odd))

    return detections


def semantic_decay(fmap: FloorMap, robot: Robot | None = None, floor=None):
    """Move semantic probabilities toward the uninformative prior 0.5."""
    if robot is None or floor is None:
        fmap.semantic = 0.5 + (fmap.semantic - 0.5) * (1.0 - DECAY_RATE)
        return

    half_fov = math.radians(CAMERA_FOV_DEG / 2.0)
    rx, ry = world_to_cell(robot.x, robot.y)
    radius_cells = int(math.ceil(CAMERA_RANGE_M / CELL_SIZE))
    x0 = max(0, rx - radius_cells)
    x1 = min(FLOOR_W, rx + radius_cells + 1)
    y0 = max(0, ry - radius_cells)
    y1 = min(FLOOR_H, ry + radius_cells + 1)

    for cy in range(y0, y1):
        for cx in range(x0, x1):
            value = float(fmap.semantic[cy, cx])
            if abs(value - 0.5) < 1e-3:
                continue
            wx, wy = cell_center_world(cx, cy)
            ddx, ddy = wx - robot.x, wy - robot.y
            dist = math.hypot(ddx, ddy)
            if dist > CAMERA_RANGE_M:
                continue
            rel = (math.atan2(ddy, ddx) - robot.theta + math.pi) % (2 * math.pi) - math.pi
            if abs(rel) > half_fov:
                continue
            if not line_of_sight_continuous(floor.grid, robot.x, robot.y, wx, wy):
                continue
            fmap.semantic[cy, cx] = 0.5 + (value - 0.5) * (1.0 - DECAY_RATE)
