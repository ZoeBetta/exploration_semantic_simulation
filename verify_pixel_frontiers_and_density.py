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
"""Non-graphical checks for fine frontiers and clutter density."""

import math
import random

import building
from building import generate_building
from robot import FloorMap, Robot, lidar_scan, FRONTIER_RESOLUTION_M
from planner import detect_standard_frontiers
from geometry import cell_center_world


def object_center_is_near_room_center(floor, obj):
    ox = 0.5 * (obj.region[0] + obj.region[2])
    oy = 0.5 * (obj.region[1] + obj.region[3])
    for room in floor.room_regions:
        rx = 0.5 * (room[0] + room[2])
        ry = 0.5 * (room[1] + room[3])
        if abs(ox - rx) <= 1.5 and abs(oy - ry) <= 1.5:
            return True
    return False


def test_density_and_central_objects():
    ratios = []
    for layout in (building.LAYOUT_OFFICE, building.LAYOUT_FREE):
        for seed in range(12):
            low = generate_building(
                2, seed=seed, layout_mode=layout,
                include_objects=True, object_density=1.0,
            )
            high = generate_building(
                2, seed=seed, layout_mode=layout,
                include_objects=True, object_density=4.0,
            )
            low_count = sum(len(f.environment_objects) for f in low)
            high_count = sum(len(f.environment_objects) for f in high)
            assert low_count > 0
            assert high_count >= low_count
            ratios.append(high_count / low_count)
            for floor in low:
                central_kinds = {
                    obj.kind
                    for obj in floor.environment_objects
                    if object_center_is_near_room_center(floor, obj)
                }
                assert set(building.OBJECT_KINDS).issubset(central_kinds)
    # Packing constraints can prevent an exact 4x count in the smallest rooms,
    # but the high-density mode must be substantially denser in aggregate.
    assert sum(ratios) / len(ratios) >= 2.7, ratios


def test_fine_frontier_geometry():
    floors = generate_building(
        2, seed=5, layout_mode=building.LAYOUT_OFFICE,
        include_objects=False,
    )
    floor = floors[0]
    fmap = FloorMap()
    floor.fmap = fmap

    # Pick a valid free cell and scan from a non-cell-centred continuous pose.
    free = None
    for y in range(2, building.FLOOR_H - 2):
        for x in range(2, building.FLOOR_W - 2):
            if floor.grid[y, x] == building.FREE:
                free = (x, y)
                break
        if free:
            break
    assert free is not None
    cx, cy = cell_center_world(*free)
    robot = Robot(0, cx + 0.073, cy - 0.061, theta=0.37)
    lidar_scan(robot, floor, fmap, random.Random(1))
    frontiers = detect_standard_frontiers(fmap)
    assert frontiers
    assert any(frontier.polylines for frontier in frontiers)

    lengths = []
    coordinates = []
    for frontier in frontiers:
        for x1, y1, x2, y2 in frontier.segments:
            lengths.append(math.hypot(x2 - x1, y2 - y1))
            coordinates.extend((x1, y1, x2, y2))
    assert lengths
    assert max(lengths) <= FRONTIER_RESOLUTION_M * 1.01
    # At least one coordinate must not lie on the old 0.5 m cell lattice.
    assert any(
        not math.isclose(value / building.CELL_SIZE,
                         round(value / building.CELL_SIZE), abs_tol=1e-8)
        for value in coordinates
    )


if __name__ == '__main__':
    test_density_and_central_objects()
    test_fine_frontier_geometry()
    print('OK: frontiere a 5 cm, oggetti centrali e densita 1x..4x verificate.')
