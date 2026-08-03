"""Regression test for the pause observed after episode 144.

With the default six Fw x Opt conditions, episode 144 is the final condition
of building 24.  The next building uses seed 24 when the global seed is zero.
At object density 3x, the old high-density packing fallback repeatedly scanned
all room cells and reran global connectivity/distance-transform checks for the
same impossible 2x2 footprints.  RunState construction could therefore take
minutes before the start-diagnostic heartbeat had a chance to run.
"""
from __future__ import annotations

import time

import building
import main


def experiment_config() -> dict:
    """Return the configuration matching the reported 30-run experiment."""
    return dict(
        n_floors=4,
        Tr=600.0,
        Ts=60.0,
        n_runs=30,
        base_seed=0,
        Fw_values=[0.0, 0.3, 1.0],
        Opt_values=[True, False],
        Iw=1.0,
        Dw=1.0,
        Vw=1.0,
        Cw=1.0,
        Ow=1.0,
        persistence_weight=1.0,
        target_switch_margin=0.0,
        layout_mode=building.LAYOUT_OFFICE,
        include_objects=True,
        object_density=3.0,
    )


def main_test() -> None:
    building.clear_building_cache()

    # Run 25 is zero-based index 24 and follows episode 144 when there are six
    # conditions per building.  Building generation must now be bounded and
    # finish well inside the user-facing 30 s watchdog on ordinary hardware.
    started = time.perf_counter()
    floors = building.generate_building(
        4,
        seed=24,
        layout_mode=building.LAYOUT_OFFICE,
        include_objects=True,
        object_density=3.0,
    )
    first_generation_s = time.perf_counter() - started
    assert first_generation_s < 15.0, first_generation_s
    assert len(floors) == 4
    assert all(floor.environment_objects for floor in floors)

    # Cached templates must be fast but independent.  Mutating one episode's
    # copy cannot contaminate any later Fw/Opt condition.
    original = int(floors[0].grid[1, 1])
    floors[0].grid[1, 1] = 99
    started = time.perf_counter()
    fresh = building.generate_building(
        4,
        seed=24,
        layout_mode=building.LAYOUT_OFFICE,
        include_objects=True,
        object_density=3.0,
    )
    cached_copy_s = time.perf_counter() - started
    assert int(fresh[0].grid[1, 1]) == original
    assert cached_copy_s < first_generation_s

    # Exercise the exact transition preparation, including all six start
    # diagnostics.  It used to appear frozen before condition 1/6 completed.
    started = time.perf_counter()
    attempt = main._select_valid_start_attempt(experiment_config(), 24)
    transition_s = time.perf_counter() - started
    assert 0 <= attempt < main.MAX_START_ATTEMPTS
    assert transition_s < 30.0, transition_s

    print(
        "OK: density-3 seed-24 transition completed; "
        f"build={first_generation_s:.3f}s, cached={cached_copy_s:.3f}s, "
        f"six-condition diagnostic={transition_s:.3f}s"
    )


if __name__ == "__main__":
    main_test()
