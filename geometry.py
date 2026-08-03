"""Continuous geometry helpers for the hybrid SAR simulator.

The building and the maps remain occupancy grids, as in the paper.  The
physical pose, rays and motion, however, are expressed in continuous metric
coordinates.  Each grid cell represents the closed/open square
[x*CELL_SIZE, (x+1)*CELL_SIZE) x [y*CELL_SIZE, (y+1)*CELL_SIZE).
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# CODE-REVIEW NOTES
# Purpose: Metric/grid conversion and collision helpers shared by mapping, planning, and continuous motion.
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
from typing import Iterable

from building import CELL_SIZE, FLOOR_H, FLOOR_W, is_blocking, is_traversable


EPS = 1e-9


@dataclass(frozen=True)
class RayResult:
    distance: float
    hit: bool
    endpoint_x: float
    endpoint_y: float
    hit_cell: tuple[int, int] | None
    free_cells: tuple[tuple[int, int], ...]


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def cell_center_world(cell_x: int, cell_y: int) -> tuple[float, float]:
    return ((cell_x + 0.5) * CELL_SIZE, (cell_y + 0.5) * CELL_SIZE)


def world_to_cell(x_m: float, y_m: float) -> tuple[int, int]:
    return (int(math.floor(x_m / CELL_SIZE)), int(math.floor(y_m / CELL_SIZE)))


def world_inside_floor(x_m: float, y_m: float) -> bool:
    return 0.0 <= x_m < FLOOR_W * CELL_SIZE and 0.0 <= y_m < FLOOR_H * CELL_SIZE


def region_center_world(region: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = region
    return ((x0 + x1 + 1) * 0.5 * CELL_SIZE,
            (y0 + y1 + 1) * 0.5 * CELL_SIZE)


def point_in_region_world(x_m: float, y_m: float,
                          region: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = region
    return (x0 * CELL_SIZE <= x_m < (x1 + 1) * CELL_SIZE and
            y0 * CELL_SIZE <= y_m < (y1 + 1) * CELL_SIZE)


def point_is_traversable(grid, x_m: float, y_m: float) -> bool:
    if not world_inside_floor(x_m, y_m):
        return False
    cx, cy = world_to_cell(x_m, y_m)
    return 0 <= cx < FLOOR_W and 0 <= cy < FLOOR_H and is_traversable(grid[cy, cx])


def path_segment_is_traversable(grid, x0: float, y0: float,
                                x1: float, y1: float,
                                sample_step_m: float | None = None) -> bool:
    """Conservative collision check for a point robot along a short segment."""
    if sample_step_m is None:
        sample_step_m = CELL_SIZE / 10.0
    length = math.hypot(x1 - x0, y1 - y0)
    samples = max(1, int(math.ceil(length / sample_step_m)))
    for i in range(1, samples + 1):
        t = i / samples
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        if not point_is_traversable(grid, x, y):
            return False
    return True


def _blocking(grid, cx: int, cy: int) -> bool:
    if cx < 0 or cy < 0 or cx >= FLOOR_W or cy >= FLOOR_H:
        return True
    return is_blocking(grid[cy, cx])


def raycast_grid_continuous(grid, x0: float, y0: float, angle: float,
                            max_range_m: float) -> RayResult:
    """Exact metric ray traversal through an axis-aligned grid.

    This is a 2-D DDA / Amanatides-Woo traversal.  Unlike the original code,
    the ray is not advanced by fixed half-cell jumps.  The returned distance is
    the exact distance from the continuous sensor origin to the boundary of the
    first occupied cell.
    """
    dx = math.cos(angle)
    dy = math.sin(angle)
    if abs(dx) < EPS:
        dx = 0.0
    if abs(dy) < EPS:
        dy = 0.0

    if not world_inside_floor(x0, y0):
        return RayResult(0.0, True, x0, y0, None, tuple())

    cx, cy = world_to_cell(x0, y0)
    free_cells: list[tuple[int, int]] = []
    if not _blocking(grid, cx, cy):
        free_cells.append((cx, cy))

    if dx > 0.0:
        step_x = 1
        next_x = (cx + 1) * CELL_SIZE
        t_max_x = (next_x - x0) / dx
        t_delta_x = CELL_SIZE / dx
    elif dx < 0.0:
        step_x = -1
        next_x = cx * CELL_SIZE
        t_max_x = (next_x - x0) / dx
        t_delta_x = -CELL_SIZE / dx
    else:
        step_x = 0
        t_max_x = math.inf
        t_delta_x = math.inf

    if dy > 0.0:
        step_y = 1
        next_y = (cy + 1) * CELL_SIZE
        t_max_y = (next_y - y0) / dy
        t_delta_y = CELL_SIZE / dy
    elif dy < 0.0:
        step_y = -1
        next_y = cy * CELL_SIZE
        t_max_y = (next_y - y0) / dy
        t_delta_y = -CELL_SIZE / dy
    else:
        step_y = 0
        t_max_y = math.inf
        t_delta_y = math.inf

    while True:
        if abs(t_max_x - t_max_y) <= 1e-10:
            distance = t_max_x
            if distance > max_range_m:
                break
            candidates = []
            if step_x:
                candidates.append((cx + step_x, cy))
            if step_y:
                candidates.append((cx, cy + step_y))
            if step_x and step_y:
                candidates.append((cx + step_x, cy + step_y))
            hit_cell = next((p for p in candidates if _blocking(grid, *p)), None)
            if hit_cell is not None:
                ex = x0 + distance * dx
                ey = y0 + distance * dy
                return RayResult(distance, True, ex, ey, hit_cell, tuple(free_cells))
            cx += step_x
            cy += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            distance = t_max_x
            if distance > max_range_m:
                break
            cx += step_x
            t_max_x += t_delta_x
            if _blocking(grid, cx, cy):
                ex = x0 + distance * dx
                ey = y0 + distance * dy
                hit_cell = (cx, cy) if 0 <= cx < FLOOR_W and 0 <= cy < FLOOR_H else None
                return RayResult(distance, True, ex, ey, hit_cell, tuple(free_cells))
        else:
            distance = t_max_y
            if distance > max_range_m:
                break
            cy += step_y
            t_max_y += t_delta_y
            if _blocking(grid, cx, cy):
                ex = x0 + distance * dx
                ey = y0 + distance * dy
                hit_cell = (cx, cy) if 0 <= cx < FLOOR_W and 0 <= cy < FLOOR_H else None
                return RayResult(distance, True, ex, ey, hit_cell, tuple(free_cells))

        if cx < 0 or cy < 0 or cx >= FLOOR_W or cy >= FLOOR_H:
            distance = min(max_range_m, distance)
            ex = x0 + distance * dx
            ey = y0 + distance * dy
            return RayResult(distance, True, ex, ey, None, tuple(free_cells))
        if (cx, cy) not in free_cells:
            free_cells.append((cx, cy))

    ex = x0 + max_range_m * dx
    ey = y0 + max_range_m * dy
    return RayResult(max_range_m, False, ex, ey, None, tuple(free_cells))


def line_of_sight_continuous(grid, x0: float, y0: float,
                             x1: float, y1: float) -> bool:
    distance = math.hypot(x1 - x0, y1 - y0)
    if distance <= EPS:
        return True
    angle = math.atan2(y1 - y0, x1 - x0)
    hit = raycast_grid_continuous(grid, x0, y0, angle, distance)
    return not hit.hit or hit.distance >= distance - 1e-7
