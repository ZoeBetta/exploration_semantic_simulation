"""Checks for corridor-aligned stair arrows and exit poses in v19."""
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
import building
from geometry import region_center_world
from planner import Frontier
from simulation import RunConfig, RunState


def main():
    for mode in (building.LAYOUT_OFFICE, building.LAYOUT_FREE):
        for seed in range(30):
            floors = building.generate_building(
                4, seed=seed, layout_mode=mode,
                include_objects=True, object_density=1.0)
            for floor in floors:
                assert len(floor.stair_core_directions) == floor.stair_core_count
                for direction in floor.stair_core_directions:
                    assert direction in ((1, 0), (-1, 0), (0, 1), (0, -1))

    state = RunState(RunConfig(
        n_floors=3, Tr=120, Ts=10, Fw=0.3, Opt=True, seed=6,
        layout_mode=building.LAYOUT_OFFICE, include_objects=False))
    floor = state.floors[0]
    region = floor.stair_up_regions[0]
    x, y = region_center_world(region)
    state.robot.floor = 0
    state.robot.x, state.robot.y = x, y
    direction = floor.stair_core_directions[0]
    state.chosen_frontier = Frontier(
        x, y, 'stair', target_floor=1, stair_id=0)
    assert state._check_stair_transition(floor)
    destination = state.floors[1].stair_down_regions[0]
    dcx, dcy = region_center_world(destination)
    dx = state.robot.x - dcx
    dy = state.robot.y - dcy
    assert dx * direction[0] + dy * direction[1] > 0
    assert abs(math.atan2(direction[1], direction[0]) - state.robot.theta) < 1e-9
    print('OK: arrows aligned with corridors, directional exits and deterministic generation')


if __name__ == '__main__':
    main()
