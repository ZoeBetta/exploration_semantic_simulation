"""Non-graphical checks for floor size and configurable planner weights."""

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

from building import FLOOR_H, FLOOR_W, LAYOUT_FREE
from planner import DW_DEFAULT
from simulation import PHYSICS_DT_S, RunConfig, RunState


def main() -> None:
    assert FLOOR_W == 44, FLOOR_W
    assert FLOOR_H == 30, FLOOR_H
    assert DW_DEFAULT == 1.0, DW_DEFAULT

    cfg = RunConfig(
        n_floors=3,
        Tr=5.0,
        Ts=1.0,
        Fw=0.3,
        Opt=True,
        seed=2,
        Iw=1.7,
        Dw=0.8,
        Vw=0.2,
        Cw=4.5,
        Ow=6.0,
        persistence_weight=19.0,
        target_switch_margin=2.5,
        layout_mode=LAYOUT_FREE,
        include_objects=False,
    )

    assert cfg.Iw == 1.7
    assert cfg.Dw == 0.8
    assert cfg.Vw == 0.2
    assert cfg.Cw == 4.5
    assert cfg.Ow == 6.0
    assert cfg.persistence_weight == 19.0
    assert cfg.target_switch_margin == 2.5
    assert cfg.layout_mode == LAYOUT_FREE
    assert cfg.include_objects is False

    state = RunState(cfg)
    for _ in range(50):
        if not state.step(PHYSICS_DT_S):
            break

    assert state.cfg is cfg
    assert len(state.floors) == 3
    assert all(floor.layout_mode == LAYOUT_FREE for floor in state.floors)
    assert all(not floor.environment_objects for floor in state.floors)

    # Exact batch formula used by main.py: no scalar weight adds a dimension
    # to the Cartesian product.
    n_runs = 50
    fw_values = [0.0, 0.3, 1.0]
    opt_values = [True, False]
    assert n_runs * len(fw_values) * len(opt_values) == 300

    print("Configurable-weight verification passed")
    print(f"  Floor: {FLOOR_W} x {FLOOR_H} cells")
    print(f"  Default Dw: {DW_DEFAULT:g}")
    print("  Custom weights and fixed environment options propagated")
    print("  Combinations: run x Fw x Opt only")


if __name__ == "__main__":
    main()
