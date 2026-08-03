"""Non-graphical checks for v13 stair gating, centred goals and rubble icons."""

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
import random

import building
from geometry import region_center_world, world_to_cell
from gui import DEBRIS_STONE_ASPECT, debris_stone_layout
from planner import (
    Frontier,
    detect_standard_frontiers,
    frontier_distance_to_point,
    score_frontiers,
)
from robot import lidar_scan
from simulation import RunConfig, RunState


def make_state(seed: int = 3) -> RunState:
    return RunState(
        RunConfig(
            n_floors=3,
            Tr=120.0,
            Ts=10.0,
            Fw=0.3,
            Opt=True,
            seed=seed,
            layout_mode=building.LAYOUT_OFFICE,
            include_objects=False,
        )
    )


def test_stair_requires_explicit_matching_target() -> None:
    state = make_state()
    middle_index = 1
    state.robot.floor = middle_index
    floor = state.floors[middle_index]
    assert floor.stair_up_region is not None
    assert floor.stair_down_region is not None

    up_x, up_y = region_center_world(floor.stair_up_region)
    state.robot.x = up_x
    state.robot.y = up_y
    state.stair_cooldown = 0.0

    # Merely standing on the stair footprint with a normal frontier selected
    # must not change floor.
    state.chosen_frontier = Frontier(
        up_x,
        up_y,
        "standard",
        goal_cell=world_to_cell(up_x, up_y),
    )
    assert not state._check_stair_transition(floor)
    assert state.robot.floor == middle_index

    # Selecting the other staircase also must not activate this footprint.
    state.chosen_frontier = Frontier(
        up_x,
        up_y,
        "stair",
        target_floor=middle_index - 1,
        goal_cell=world_to_cell(up_x, up_y),
    )
    assert not state._check_stair_transition(floor)
    assert state.robot.floor == middle_index

    # A stair frontier is never discarded by the generic proximity threshold;
    # it remains active until the matching transition actually occurs.
    selected = Frontier(
        up_x,
        up_y,
        "stair",
        target_floor=middle_index + 1,
        goal_cell=world_to_cell(up_x, up_y),
    )
    state.chosen_frontier = selected
    assert not state._frontier_reached(selected)
    before_time = state.remaining_time
    assert state._check_stair_transition(floor)
    assert state.robot.floor == middle_index + 1
    assert math.isclose(state.remaining_time, before_time - state.cfg.Ts)


def test_standard_goal_is_frontier_centre() -> None:
    state = make_state(seed=7)
    floor = state.floors[state.robot.floor]
    lidar_scan(state.robot, floor, floor.fmap, random.Random(1))
    frontiers = detect_standard_frontiers(floor.fmap)
    assert frontiers

    score_frontiers(
        frontiers,
        state.robot,
        floor.fmap,
        floor,
        state.floors,
        state.cfg.Fw,
        state.cfg.Cw,
        state.cfg.Ow,
        Iw=state.cfg.Iw,
        Dw=state.cfg.Dw,
        Vw=state.cfg.Vw,
    )
    reachable = [frontier for frontier in frontiers if frontier.path]
    assert reachable

    for frontier in reachable:
        # The centre is on the actual 5 cm frontier curve, not merely inside
        # its bounding box or at a coarse-cell representative.
        assert frontier_distance_to_point(
            frontier, frontier.x, frontier.y
        ) <= 1e-8
        # The continuous controller's final waypoint is exactly that centre.
        end_x, end_y = frontier.path[-1]
        assert math.hypot(end_x - frontier.x, end_y - frontier.y) <= 1e-9


def test_rubble_stones_scale_by_count_not_distortion() -> None:
    base = debris_stone_layout(40, 40, region_cells=(2, 2))
    wide = debris_stone_layout(80, 40, region_cells=(4, 2))
    tall = debris_stone_layout(40, 80, region_cells=(2, 4))
    large = debris_stone_layout(80, 80, region_cells=(4, 4))

    assert len(base) > 1
    assert len(wide) == 2 * len(base)
    assert len(tall) == 2 * len(base)
    assert len(large) == 4 * len(base)

    for collection in (base, wide, tall, large):
        for _cx, _cy, rx, ry, _angle, _colour in collection:
            assert math.isclose(rx / ry, DEBRIS_STONE_ASPECT, rel_tol=1e-12)

    # Every repeated 2x2 tile uses the same small/large stone dimensions.
    base_sizes = sorted((round(rx, 8), round(ry, 8))
                        for _cx, _cy, rx, ry, _a, _c in base)
    wide_sizes = sorted((round(rx, 8), round(ry, 8))
                        for _cx, _cy, rx, ry, _a, _c in wide)
    assert wide_sizes == sorted(base_sizes + base_sizes)


if __name__ == "__main__":
    test_stair_requires_explicit_matching_target()
    test_standard_goal_is_frontier_centre()
    test_rubble_stones_scale_by_count_not_distortion()
    print("OK: scale esplicite, goal al centro e macerie a sassi verificati.")
