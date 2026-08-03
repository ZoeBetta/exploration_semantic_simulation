"""Non-graphical verification of the 5 cm SLAM-like occupancy map."""

from __future__ import annotations

# -----------------------------------------------------------------------------
# CODE-REVIEW NOTES
# Purpose: Automated regression checks. Assertions document the invariant being protected and fail loudly on behavioural regressions.
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

import math
import numpy as np

from building import Floor, FREE as GT_FREE, WALL, FLOOR_W, FLOOR_H, CELL_SIZE
from robot import (
    FloorMap,
    Robot,
    lidar_scan,
    PIXEL_OCCUPANCY_RESOLUTION_M,
    PIXEL_OCCUPANCY_W,
    PIXEL_OCCUPANCY_H,
    FRONTIER_RESOLUTION_M,
    FREE,
    OCC,
)


def fine_cell(x_m: float, y_m: float) -> tuple[int, int]:
    return (
        int(math.floor(x_m / PIXEL_OCCUPANCY_RESOLUTION_M)),
        int(math.floor(y_m / PIXEL_OCCUPANCY_RESOLUTION_M)),
    )


def main() -> None:
    floor = Floor(0, 2)
    floor.grid[:, :] = GT_FREE
    floor.grid[0, :] = WALL
    floor.grid[-1, :] = WALL
    floor.grid[:, 0] = WALL
    floor.grid[:, -1] = WALL

    # Vertical wall three metres in front of the robot.
    wall_x_cell = 16  # x in [8.0, 8.5) m
    floor.grid[4:17, wall_x_cell] = WALL

    fmap = FloorMap()
    robot = Robot(0, 5.0, 5.0, theta=0.0)
    beams = lidar_scan(robot, floor, fmap)

    assert len(beams) == 181
    assert PIXEL_OCCUPANCY_RESOLUTION_M == FRONTIER_RESOLUTION_M == 0.05
    assert fmap.pixel_occ_log_odds.shape == (
        PIXEL_OCCUPANCY_H,
        PIXEL_OCCUPANCY_W,
    )
    assert fmap.pixel_occ_observed.shape == fmap.pixel_occ_log_odds.shape
    assert np.count_nonzero(fmap.pixel_occ_observed) > 0
    assert np.count_nonzero(fmap.pixel_occ_observed) < fmap.pixel_occ_observed.size

    # Free point on the central ray, before the wall.
    fx, fy = fine_cell(7.50, 5.025)
    assert fmap.pixel_occ_observed[fy, fx]
    assert fmap.pixel_occ_log_odds[fy, fx] < 0.0

    # First 5 cm layer just inside the laser-facing wall surface is occupied.
    ox, oy = fine_cell(8.025, 5.025)
    assert fmap.pixel_occ_observed[oy, ox]
    assert fmap.pixel_occ_log_odds[oy, ox] > 0.0

    # The hidden thickness of the same coarse wall cell is not revealed.
    hx, hy = fine_cell(8.25, 5.025)
    assert not fmap.pixel_occ_observed[hy, hx]

    # Space behind the wall and behind the 180-degree sensor remains unknown.
    bx, by = fine_cell(8.75, 5.025)
    assert not fmap.pixel_occ_observed[by, bx]
    rx, ry = fine_cell(4.0, 5.025)
    assert not fmap.pixel_occ_observed[ry, rx]

    # The old 0.5 m map still exists and is updated for A* and paper metrics.
    assert fmap.occ[10, 15] == FREE
    assert fmap.occ[10, wall_x_cell] == OCC

    p = fmap.pixel_occupancy_probability()
    assert p[fy, fx] < 0.5
    assert p[oy, ox] > 0.5

    wall_block = fmap.pixel_occ_observed[
        10 * int(CELL_SIZE / PIXEL_OCCUPANCY_RESOLUTION_M):
        11 * int(CELL_SIZE / PIXEL_OCCUPANCY_RESOLUTION_M),
        wall_x_cell * int(CELL_SIZE / PIXEL_OCCUPANCY_RESOLUTION_M):
        (wall_x_cell + 1) * int(CELL_SIZE / PIXEL_OCCUPANCY_RESOLUTION_M),
    ]
    assert np.count_nonzero(wall_block) < wall_block.size

    print("Pixel occupancy verification passed")
    print(f"  fine map: {PIXEL_OCCUPANCY_W} x {PIXEL_OCCUPANCY_H}")
    print(f"  resolution: {PIXEL_OCCUPANCY_RESOLUTION_M:.2f} m")
    print(f"  observed after one scan: {np.count_nonzero(fmap.pixel_occ_observed)} pixels")
    print("  hidden wall thickness remains unknown")
    print("  coarse planning map remains unchanged")


if __name__ == "__main__":
    main()
