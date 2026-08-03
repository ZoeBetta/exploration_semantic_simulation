"""Non-graphical validation for the v10 geometry and navigation changes."""

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

from collections import deque

import numpy as np
from scipy import ndimage

import building
from building import (
    CELL_SIZE,
    DOOR,
    DOOR_WIDTH_CELLS,
    FREE as GT_FREE,
    LAYOUT_FREE,
    LAYOUT_OFFICE,
    OBJECT_CHAIR,
    OBJECT_DEBRIS,
    OBJECT_TABLE,
    STAIR_DOWN,
    STAIR_UP,
    WALL,
    generate_building,
)
from geometry import world_to_cell
from planner import (
    PATH_HARD_WALL_CLEARANCE_CELLS,
    PATH_PREFERRED_WALL_CLEARANCE_CELLS,
    a_star,
    wall_clearance_cells,
)
from robot import FREE, OCC
from simulation import (
    PHYSICS_DT_S,
    START_MIN_STAIR_DISTANCE_M,
    START_MIN_WALL_CLEARANCE_M,
    RunConfig,
    RunState,
)

TRAVERSABLE_GT = (GT_FREE, DOOR, STAIR_UP, STAIR_DOWN)


def _assert_connected(grid: np.ndarray) -> None:
    traversable = np.isin(grid, TRAVERSABLE_GT)
    ys, xs = np.where(traversable)
    assert len(xs) > 0
    start = (int(xs[0]), int(ys[0]))
    queue = deque([start])
    seen = {start}
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]):
                continue
            if not traversable[ny, nx] or (nx, ny) in seen:
                continue
            seen.add((nx, ny))
            queue.append((nx, ny))
    assert len(seen) == int(np.sum(traversable))


def _assert_doubled_object(item) -> None:
    x0, y0, x1, y1 = item.region
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    if item.kind == OBJECT_CHAIR:
        assert (width, height) == (2, 2)
    elif item.kind == OBJECT_TABLE:
        assert sorted((width, height)) in ([2, 4], [2, 6])
    elif item.kind == OBJECT_DEBRIS:
        assert sorted((width, height)) in ([2, 2], [2, 4], [4, 4])
    else:
        raise AssertionError(item.kind)


def _safe_component(grid: np.ndarray):
    occ = np.where(grid == WALL, OCC, FREE).astype(np.int8)
    clearance = wall_clearance_cells(occ)
    safe = ((occ == FREE)
            & (clearance >= PATH_HARD_WALL_CLEARANCE_CELLS))
    ys, xs = np.where(safe)
    assert len(xs) > 0
    start = (int(xs[0]), int(ys[0]))
    queue = deque([start])
    seen = {start}
    while queue:
        x, y = queue.popleft()
        for dx, dy in (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < safe.shape[1] and 0 <= ny < safe.shape[0]):
                continue
            if not safe[ny, nx] or (nx, ny) in seen:
                continue
            if dx and dy and (not safe[y, nx] or not safe[ny, x]):
                continue
            seen.add((nx, ny))
            queue.append((nx, ny))
    return safe, seen


def _assert_three_cell_door_path() -> None:
    height, width = 17, 31
    occ = np.full((height, width), FREE, dtype=np.int8)
    occ[0, :] = OCC
    occ[-1, :] = OCC
    occ[:, 0] = OCC
    occ[:, -1] = OCC

    wall_x = width // 2
    occ[1:-1, wall_x] = OCC
    door_y0 = height // 2 - 1
    occ[door_y0:door_y0 + DOOR_WIDTH_CELLS, wall_x] = FREE

    start = (3, height // 2)
    goal = (width - 4, height // 2)
    clearance = wall_clearance_cells(occ)
    path = a_star(occ, start, goal, clearance_map=clearance)
    assert path is not None
    # Only the middle cell of a three-cell door is far enough from both jambs;
    # the clearance-aware route must use that centred passage.
    assert (wall_x, height // 2) in path
    assert min(clearance[y, x] for x, y in path[1:-1]) >= (
        PATH_HARD_WALL_CLEARANCE_CELLS
    )


def main() -> None:
    floors_checked = 0
    free_floors_checked = 0

    for mode in (LAYOUT_OFFICE, LAYOUT_FREE):
        for seed in range(18):
            floors = generate_building(
                4,
                seed=seed,
                layout_mode=mode,
                include_objects=True,
            )
            for floor in floors:
                floors_checked += 1
                _assert_connected(floor.grid)

                kinds = {item.kind for item in floor.environment_objects}
                assert {OBJECT_CHAIR, OBJECT_TABLE, OBJECT_DEBRIS}.issubset(kinds)
                for item in floor.environment_objects:
                    _assert_doubled_object(item)
                    x0, y0, x1, y1 = item.region
                    assert np.all(floor.grid[y0:y1 + 1, x0:x1 + 1] == WALL)

                if mode == LAYOUT_FREE:
                    free_floors_checked += 1
                    assert len(floor.wall_regions) >= building.FREE_MIN_WALL_REGIONS
                    assert len(floor.room_regions) >= building.FREE_MIN_ROOM_COUNT
                    assert len(floor.corridor_regions) >= 5

                    safe, connected_safe = _safe_component(floor.grid)
                    # Every generated room has a clearance-valid cell connected
                    # to the same global navigation component.
                    for x0, y0, x1, y1 in floor.room_regions:
                        assert any(
                            safe[y, x] and (x, y) in connected_safe
                            for y in range(y0, y1 + 1)
                            for x in range(x0, x1 + 1)
                        )

    _assert_three_cell_door_path()

    # Start selection is checked on both topologies, with and without clutter.
    starts_checked = 0
    for mode in (LAYOUT_OFFICE, LAYOUT_FREE):
        for include_objects in (False, True):
            for seed in range(24):
                state = RunState(RunConfig(
                    n_floors=4,
                    Tr=1.0,
                    Ts=0.0,
                    Fw=0.3,
                    Opt=True,
                    seed=seed,
                    layout_mode=mode,
                    include_objects=include_objects,
                ))
                starts_checked += 1
                assert state.initial_stair_distance_m >= START_MIN_STAIR_DISTANCE_M

                floor = state.floors[state.robot.floor]
                blocking = floor.grid == WALL
                padded = np.pad(
                    ~blocking, 1, mode="constant", constant_values=False
                )
                wall_clearance_m = (
                    ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
                    * CELL_SIZE
                )
                cx, cy = world_to_cell(state.robot.x, state.robot.y)
                assert wall_clearance_m[cy, cx] >= START_MIN_WALL_CLEARANCE_M

                # Exercise the continuous loop briefly with the new map/planner.
                for _ in range(20):
                    if not state.step(PHYSICS_DT_S):
                        break

    print("Dense-layout, clearance and safe-start verification passed")
    print(f"  Floors checked: {floors_checked}")
    print(f"  Dense free-topology floors checked: {free_floors_checked}")
    print("  Objects: doubled linear footprints")
    print(
        "  Free topology: >= "
        f"{building.FREE_MIN_WALL_REGIONS} walls, >= "
        f"{building.FREE_MIN_ROOM_COUNT} rooms, 5 corridor regions"
    )
    print(
        "  A*: hard clearance "
        f"{PATH_HARD_WALL_CLEARANCE_CELLS:g} cells, preferred "
        f"{PATH_PREFERRED_WALL_CLEARANCE_CELLS:g} cells"
    )
    print(
        "  Safe starts checked: "
        f"{starts_checked}, stair distance >= {START_MIN_STAIR_DISTANCE_M:g} m"
    )


if __name__ == "__main__":
    main()
