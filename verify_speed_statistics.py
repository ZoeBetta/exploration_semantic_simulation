"""Non-graphical checks for v14 playback persistence and statistics."""

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

from gui import RunViewer, summarize_results_by_condition


def check_persistent_speed() -> None:
    RunViewer._persistent_time_scale = 1.0

    first = RunViewer(object(), "first")
    assert first.time_scale == 1.0
    first._set_time_scale(4.0)
    assert RunViewer.persistent_time_scale() == 4.0

    second = RunViewer(object(), "second")
    assert second.time_scale == 4.0, (
        "A new viewer must inherit the multiplier selected in the previous run"
    )
    second._set_time_scale(8.0)

    third = RunViewer(
        object(),
        "third",
        initial_time_scale=RunViewer.persistent_time_scale(),
    )
    assert third.time_scale == 8.0

    # Unsupported values are normalised to the nearest supported multiplier.
    third._set_time_scale(3.6)
    assert third.time_scale == 4.0


def check_condition_statistics() -> None:
    results = [
        {"Fw": 0.0, "Opt": False, "Af": 10.0, "Af_star": 20.0, "Atot": 30.0},
        {"Fw": 0.0, "Opt": False, "Af": 20.0, "Af_star": 30.0, "Atot": 50.0},
        {"Fw": 0.0, "Opt": True, "Af": 40.0, "Af_star": 50.0, "Atot": 60.0},
        {"Fw": 1.0, "Opt": False, "Af": 25.0, "Af_star": 35.0, "Atot": 45.0},
        {"Fw": 1.0, "Opt": False, "Af": 35.0, "Af_star": 45.0, "Atot": 65.0},
    ]

    summary = summarize_results_by_condition(results)
    by_condition = {(row["Fw"], row["Opt"]): row for row in summary}

    row = by_condition[(0.0, False)]
    assert row["N"] == 2
    assert math.isclose(row["Af_mean"], 15.0)
    assert math.isclose(row["Af_std"], math.sqrt(50.0), rel_tol=1e-12)
    assert math.isclose(row["Af_star_mean"], 25.0)
    assert math.isclose(row["Af_star_std"], math.sqrt(50.0), rel_tol=1e-12)
    assert math.isclose(row["Atot_mean"], 40.0)
    assert math.isclose(row["Atot_std"], math.sqrt(200.0), rel_tol=1e-12)

    single = by_condition[(0.0, True)]
    assert single["N"] == 1
    assert single["Af_mean"] == 40.0
    assert single["Af_std"] == 0.0
    assert single["Af_star_std"] == 0.0
    assert single["Atot_std"] == 0.0

    other = by_condition[(1.0, False)]
    assert other["N"] == 2
    assert math.isclose(other["Atot_mean"], 55.0)
    assert math.isclose(other["Atot_std"], math.sqrt(200.0), rel_tol=1e-12)


if __name__ == "__main__":
    check_persistent_speed()
    check_condition_statistics()
    print("OK: playback speed persists and condition statistics are correct.")
