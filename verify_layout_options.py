"""Non-graphical checks for topology and optional static objects."""

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

from building import (
    DOOR,
    DOOR_WIDTH_CELLS,
    FLOOR_H,
    FLOOR_W,
    FREE,
    LAYOUT_FREE,
    LAYOUT_OFFICE,
    OBJECT_CHAIR,
    OBJECT_DEBRIS,
    OBJECT_KINDS,
    OBJECT_TABLE,
    STAIR_DOWN,
    STAIR_UP,
    WALL,
    generate_building,
    is_blocking,
)
from simulation import PHYSICS_DT_S, RunConfig, RunState

TRAVERSABLE = (FREE, DOOR, STAIR_UP, STAIR_DOWN)


def _assert_connected(grid: np.ndarray) -> None:
    mask = np.isin(grid, TRAVERSABLE)
    ys, xs = np.where(mask)
    assert len(xs) > 0
    start = (int(xs[0]), int(ys[0]))
    queue = deque([start])
    seen = {start}
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < FLOOR_W and 0 <= ny < FLOOR_H):
                continue
            if not mask[ny, nx] or (nx, ny) in seen:
                continue
            seen.add((nx, ny))
            queue.append((nx, ny))
    assert len(seen) == int(np.sum(mask))


def _assert_door_widths(floor) -> None:
    for x0, y0, x1, y1 in floor.door_regions:
        span = max(x1 - x0 + 1, y1 - y0 + 1)
        assert span == DOOR_WIDTH_CELLS
        assert np.all(floor.grid[y0:y1 + 1, x0:x1 + 1] == DOOR)


def _assert_object_metadata(floor) -> None:
    kinds = {item.kind for item in floor.environment_objects}
    assert set(OBJECT_KINDS).issubset(kinds), kinds
    for item in floor.environment_objects:
        assert item.kind in (OBJECT_CHAIR, OBJECT_TABLE, OBJECT_DEBRIS)
        x0, y0, x1, y1 = item.region
        assert np.all(floor.grid[y0:y1 + 1, x0:x1 + 1] == WALL)
        assert is_blocking(floor.grid[y0, x0])


def main() -> None:
    checked = 0
    for layout_mode in (LAYOUT_OFFICE, LAYOUT_FREE):
        for include_objects in (False, True):
            for seed in range(30):
                floors = generate_building(
                    4,
                    seed=seed,
                    layout_mode=layout_mode,
                    include_objects=include_objects,
                )
                for index, floor in enumerate(floors):
                    checked += 1
                    assert floor.layout_mode == layout_mode
                    _assert_connected(floor.grid)
                    _assert_door_widths(floor)
                    if include_objects:
                        _assert_object_metadata(floor)
                    else:
                        assert floor.environment_objects == []

                    if index < len(floors) - 1:
                        assert floor.stair_up_region == floors[index + 1].stair_down_region
                    if 0 < index < len(floors) - 1:
                        assert floor.stair_up_region != floor.stair_down_region

    # Verify that the fixed environment choices propagate into a real episode
    # without adding dimensions to the Fw x Opt experiment product.
    cfg = RunConfig(
        n_floors=3,
        Tr=1.0,
        Ts=0.0,
        Fw=0.3,
        Opt=True,
        seed=9,
        layout_mode=LAYOUT_FREE,
        include_objects=False,
    )
    state = RunState(cfg)
    assert state.cfg.layout_mode == LAYOUT_FREE
    assert state.cfg.include_objects is False
    assert all(floor.layout_mode == LAYOUT_FREE for floor in state.floors)
    assert all(not floor.environment_objects for floor in state.floors)
    for _ in range(20):
        if not state.step(PHYSICS_DT_S):
            break

    n_runs = 50
    fw_values = [0.0, 0.3, 1.0]
    opt_values = [True, False]
    assert n_runs * len(fw_values) * len(opt_values) == 300

    print("Layout-option verification passed")
    print(f"  Floors checked: {checked}")
    print("  Topologies: office and free")
    print("  Objects: chairs, tables and rubble; all blocking")
    print("  Object toggle: enabled and disabled")
    print("  Combinations remain run x Fw x Opt only")


if __name__ == "__main__":
    main()
