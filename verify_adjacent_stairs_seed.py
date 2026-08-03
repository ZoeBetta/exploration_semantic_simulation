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
"""Verification for v18: adjacent stair cores, exact 1/3 ratio, room width and seeds."""
from building import generate_building, _regions_adjacent_pair
from simulation import RunConfig, RunState


def signature(floors):
    return tuple(
        (
            floor.office_variant,
            floor.grid.tobytes(),
            tuple(floor.stair_up_regions),
            tuple(floor.stair_down_regions),
        )
        for floor in floors
    )


def main():
    for mode in ("office", "free"):
        counts = {1: 0, 2: 0}
        for seed in range(60):
            floors = generate_building(
                4, seed=seed, layout_mode=mode,
                include_objects=True, object_density=1.0,
            )
            expected = 2 if seed % 3 == 0 else 1
            assert floors[0].stair_core_count == expected
            counts[expected] += 1
            for floor in floors:
                assert floor.stair_core_count == expected
                assert all(
                    region[2] - region[0] + 1 >= 3
                    and region[3] - region[1] + 1 >= 3
                    for region in floor.room_regions
                )
                if 0 < floor.index < 3:
                    assert len(floor.stair_up_regions) == expected
                    assert len(floor.stair_down_regions) == expected
                    for up, down in zip(
                        floor.stair_up_regions, floor.stair_down_regions
                    ):
                        assert _regions_adjacent_pair(up, down)
        assert counts == {1: 40, 2: 20}, counts

    # Same seed gives the exact same building; next run seed gives a new one.
    a = generate_building(4, seed=123, layout_mode="office")
    b = generate_building(4, seed=123, layout_mode="office")
    c = generate_building(4, seed=124, layout_mode="office")
    assert signature(a) == signature(b)
    assert signature(a) != signature(c)

    # All stochastic subsystems derive from the episode seed.
    cfg1 = RunConfig(4, 5, 1, .3, True, seed=123)
    cfg2 = RunConfig(4, 5, 1, .3, True, seed=123)
    r1, r2 = RunState(cfg1), RunState(cfg2)
    assert (r1.robot.floor, r1.robot.x, r1.robot.y, r1.robot.theta) == (
        r2.robot.floor, r2.robot.x, r2.robot.y, r2.robot.theta
    )
    print("OK: scale affiancate, rapporto 1/3, stanze >= 3 celle e seed globale")


if __name__ == "__main__":
    main()
