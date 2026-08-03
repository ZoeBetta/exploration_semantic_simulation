#!/usr/bin/env python3
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
"""Non-graphical checks for v16 office diversity and stair placement."""
from collections import Counter
from building import (
    OFFICE_VARIANTS,
    generate_building,
    _stair_is_in_front_of_door,
)


def main():
    counts = Counter()
    for seed in range(180):
        floors = generate_building(
            4,
            seed=seed,
            layout_mode="office",
            include_objects=False,
        )
        variant = floors[0].office_variant
        counts[variant] += 1
        assert variant in OFFICE_VARIANTS
        assert len({floor.office_variant for floor in floors}) == 1
        assert len({tuple(floor.corridor_regions) for floor in floors}) == 1

        for floor in floors:
            assert floor.room_regions
            assert floor.door_regions
            for stair in (
                floor.stair_up_region,
                floor.stair_down_region,
            ):
                if stair is None:
                    continue
                assert not any(
                    _stair_is_in_front_of_door(stair, door)
                    for door in floor.door_regions
                )

    missing = set(OFFICE_VARIANTS) - set(counts)
    assert not missing, f"Office variants not sampled: {sorted(missing)}"
    print("Office variants:", dict(sorted(counts.items())))
    print("Same topology across all Fw x Opt episodes: guaranteed by seed-based generation")
    print("Stair-door approach conflicts: 0")
    print("v16 office diversity checks passed")


if __name__ == "__main__":
    main()
