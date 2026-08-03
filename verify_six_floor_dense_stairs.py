"""Regression checks for six-floor buildings with dense clutter.

The failure fixed in v34 was structural, not related to the robot start pose:
with many floors the old generator intersected the door-free stair candidates
of every floor.  The union could be empty, so retrying ``start_attempt`` merely
rebuilt the same impossible geometry indefinitely.
"""
from __future__ import annotations

import building
from simulation import RunConfig, RunState


def building_signature(floors):
    """Compact deterministic signature used to verify seed reproducibility."""
    return tuple(
        (
            floor.grid.tobytes(),
            tuple(floor.door_regions),
            tuple(floor.stair_up_regions),
            tuple(floor.stair_down_regions),
            tuple(floor.stair_core_directions),
        )
        for floor in floors
    )


def combined(up_region, down_region):
    return (
        min(up_region[0], down_region[0]),
        min(up_region[1], down_region[1]),
        max(up_region[2], down_region[2]),
        max(up_region[3], down_region[3]),
    )


def check_building(mode: str, seed: int):
    floors = building.generate_building(
        6,
        seed=seed,
        layout_mode=mode,
        include_objects=True,
        object_density=4.0,
    )
    expected_cores = 2 if seed % 3 == 0 else 1
    assert len(floors) == 6
    assert all(floor.stair_core_count == expected_cores for floor in floors)

    # Core coordinates and arrow directions are identical on every floor.
    reference_directions = tuple(floors[0].stair_core_directions)
    reference_up = tuple(floors[0].stair_up_regions)
    reference_down = tuple(floors[-1].stair_down_regions)
    assert len(reference_up) == expected_cores
    assert len(reference_down) == expected_cores

    for floor in floors:
        assert tuple(floor.stair_core_directions) == reference_directions
        for core_id in range(expected_cores):
            up = reference_up[core_id]
            down = reference_down[core_id]
            assert building._regions_adjacent_pair(up, down)
            footprint = combined(up, down)
            assert not any(
                building._stair_is_in_front_of_door(footprint, door)
                for door in floor.door_regions
            )

    return floors


def main():
    # Seed 1 reproduces the Run 2 case shown in the user screenshot when the
    # global seed is zero.  Additional seeds cover both one-core and two-core
    # buildings and both topology families.
    seed_sets = {
        building.LAYOUT_OFFICE: (0, 1, 3, 5),
        building.LAYOUT_FREE: (0, 1, 3),
    }
    saved_office_seed_one = None
    for mode, seeds in seed_sets.items():
        for seed in seeds:
            floors = check_building(mode, seed)
            if mode == building.LAYOUT_OFFICE and seed == 1:
                saved_office_seed_one = floors

    # Reproducibility is checked explicitly on the screenshot seed without
    # duplicating every relatively expensive 4x-clutter case.
    duplicate = building.generate_building(
        6, seed=1, layout_mode=building.LAYOUT_OFFICE,
        include_objects=True, object_density=4.0,
    )
    assert building_signature(saved_office_seed_one) == building_signature(duplicate)

    # Full RunState initialization must also succeed with the exact dense,
    # six-floor configuration that previously failed before start diagnostics.
    state = RunState(RunConfig(
        n_floors=6,
        Tr=600,
        Ts=20,
        Fw=0.0,
        Opt=True,
        seed=1,
        layout_mode=building.LAYOUT_OFFICE,
        include_objects=True,
        object_density=4.0,
    ))
    assert len(state.floors) == 6
    print("OK: six-floor dense buildings reserve valid common stair cores")


if __name__ == "__main__":
    main()
