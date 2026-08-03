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
"""Compatibility entry point for all geometry and clearance validations."""

from verify_office_layout import main as verify_office
from verify_layout_options import main as verify_options
from verify_dense_clearance_start import main as verify_dense


if __name__ == "__main__":
    verify_office()
    verify_options()
    verify_dense()
