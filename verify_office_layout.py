"""Non-graphical validation for the office-style building generator."""

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
    MAIN_CORRIDOR_MIN_WIDTH_CELLS,
    MIN_CORRIDOR_WIDTH_CELLS,
    MIN_ROOM_DEPTH_CELLS,
    MIN_ROOM_WIDTH_CELLS,
    STAIR_DOWN,
    STAIR_UP,
    generate_building,
)
from planner import DW_DEFAULT


TRAVERSABLE_VALUES = (FREE, DOOR, STAIR_UP, STAIR_DOWN)


def _assert_connected(grid) -> None:
    traversable = np.isin(grid, TRAVERSABLE_VALUES)
    ys, xs = np.where(traversable)
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
            if not traversable[ny, nx] or (nx, ny) in seen:
                continue
            seen.add((nx, ny))
            queue.append((nx, ny))

    assert len(seen) == int(np.sum(traversable))


def _inside(inner, outer) -> bool:
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    return ox0 <= ix0 <= ix1 <= ox1 and oy0 <= iy0 <= iy1 <= oy1


def _assert_floor(floor) -> None:
    grid = floor.grid
    assert floor.office_orientation in ("horizontal", "vertical")
    assert len(floor.corridor_regions) == 1
    assert len(floor.room_regions) == len(floor.room_door_regions)
    assert len(floor.room_regions) >= 6

    corridor = floor.corridor_regions[0]
    cx0, cy0, cx1, cy1 = corridor
    corridor_width = (
        cy1 - cy0 + 1
        if floor.office_orientation == "horizontal"
        else cx1 - cx0 + 1
    )
    assert corridor_width >= MAIN_CORRIDOR_MIN_WIDTH_CELLS
    assert np.all(np.isin(grid[cy0:cy1 + 1, cx0:cx1 + 1], TRAVERSABLE_VALUES))

    primary_sizes = []
    for room, door in zip(floor.room_regions, floor.room_door_regions):
        rx0, ry0, rx1, ry1 = room
        dx0, dy0, dx1, dy1 = door
        width = rx1 - rx0 + 1
        height = ry1 - ry0 + 1
        assert width >= MIN_ROOM_WIDTH_CELLS or height >= MIN_ROOM_WIDTH_CELLS
        assert min(width, height) >= MIN_ROOM_DEPTH_CELLS

        span = max(dx1 - dx0 + 1, dy1 - dy0 + 1)
        assert span == DOOR_WIDTH_CELLS
        assert np.all(grid[dy0:dy1 + 1, dx0:dx1 + 1] == DOOR)

        if floor.office_orientation == "horizontal":
            assert rx0 <= dx0 <= dx1 <= rx1
            assert dy0 == dy1
            assert dy0 in (ry0 - 1, ry1 + 1)
            assert cy0 - 1 <= dy0 <= cy1 + 1
            primary_sizes.append(width)
        else:
            assert ry0 <= dy0 <= dy1 <= ry1
            assert dx0 == dx1
            assert dx0 in (rx0 - 1, rx1 + 1)
            assert cx0 - 1 <= dx0 <= cx1 + 1
            primary_sizes.append(height)

    # Room rows should contain more than one room size, not a uniform lattice.
    assert len(set(primary_sizes)) >= 2

    stairs = [
        region for region in
        (floor.stair_up_region, floor.stair_down_region)
        if region is not None
    ]
    for stair in stairs:
        assert _inside(stair, corridor)
        sx0, sy0, sx1, sy1 = stair
        if floor.office_orientation == "horizontal":
            assert cy1 - sy1 >= MIN_CORRIDOR_WIDTH_CELLS
        else:
            assert cx1 - sx1 >= MIN_CORRIDOR_WIDTH_CELLS

    _assert_connected(grid)


def main() -> None:
    assert DW_DEFAULT == 1.0

    checked_floors = 0
    for seed in range(120):
        floors = generate_building(4, seed=seed)
        for index, floor in enumerate(floors):
            checked_floors += 1
            _assert_floor(floor)
            if index < len(floors) - 1:
                assert floor.stair_up_region == floors[index + 1].stair_down_region
            if 0 < index < len(floors) - 1:
                assert floor.stair_up_region != floor.stair_down_region

    # Also exercise a broader range of floor counts.  Alternating service
    # cores must keep up/down footprints distinct on every intermediate floor.
    for n_floors in range(2, 11):
        floors = generate_building(n_floors, seed=1000 + n_floors)
        for index, floor in enumerate(floors):
            _assert_floor(floor)
            if 0 < index < n_floors - 1:
                assert floor.stair_up_region != floor.stair_down_region

    print("Office layout verification passed")
    print(f"  Floors checked: {checked_floors}")
    print(f"  Door width: {DOOR_WIDTH_CELLS} cells")
    print(f"  Minimum usable corridor: {MIN_CORRIDOR_WIDTH_CELLS} cells")
    print(f"  Main corridor: >= {MAIN_CORRIDOR_MIN_WIDTH_CELLS} cells")
    print(f"  Default Dw: {DW_DEFAULT:g}")


if __name__ == "__main__":
    main()
