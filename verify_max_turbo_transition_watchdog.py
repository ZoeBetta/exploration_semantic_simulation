"""Regression checks for the v33 between-building Max Turbo watchdog.

The test deliberately avoids requiring a real Pygame display.  It verifies the
control-flow hooks that keep the window responsive and checks that cancellation
and progress propagation work without changing the diagnostic result.
"""

from __future__ import annotations

import inspect

import main
from gui import RunViewer
from simulation import RunConfig


def _small_cfg() -> dict:
    return dict(
        n_floors=2,
        Tr=0.5,
        Ts=0.0,
        n_runs=1,
        base_seed=7,
        Fw_values=[0.0],
        Opt_values=[False],
        Iw=1.0,
        Dw=1.0,
        Vw=0.3,
        Cw=3.0,
        Ow=8.0,
        persistence_weight=24.0,
        target_switch_margin=1.0,
        layout_mode="office",
        include_objects=False,
        object_density=1.0,
    )


def test_diagnostic_heartbeat_and_cancel() -> None:
    cfg = _small_cfg()
    run_cfg = main._make_run_config(cfg, 0, 0.0, False, 0)
    heartbeats: list[tuple[float, str]] = []

    def callback(progress: float, detail: str) -> bool:
        heartbeats.append((progress, detail))
        return True

    main._diagnose_initial_start(
        run_cfg,
        seconds=0.5,
        progress_callback=callback,
        wall_timeout_s=30.0,
    )
    assert len(heartbeats) >= 2, "The diagnostic must publish periodic heartbeats"
    assert all(0.0 <= item[0] <= 1.0 for item in heartbeats)

    def cancel(_progress: float, _detail: str) -> bool:
        return False

    try:
        main._diagnose_initial_start(
            run_cfg,
            seconds=0.5,
            progress_callback=cancel,
        )
    except main.ExperimentCancelled:
        pass
    else:
        raise AssertionError("A rejected heartbeat must cancel preparation")


def test_condition_progress_plumbing() -> None:
    cfg = _small_cfg()
    calls: list[tuple[str, str, float]] = []
    original = main._diagnose_initial_start

    def fake_diagnostic(_run_cfg, **kwargs):
        cb = kwargs.get("progress_callback")
        if cb is not None:
            assert cb(0.5, "halfway")
            assert cb(1.0, "done")
        return True, "ok"

    main._diagnose_initial_start = fake_diagnostic
    try:
        attempt = main._select_valid_start_attempt(
            cfg,
            0,
            progress_callback=lambda title, detail, progress: (
                calls.append((title, detail, progress)) or True
            ),
        )
    finally:
        main._diagnose_initial_start = original

    assert attempt == 0
    assert calls
    assert any("Preparazione Run 1/1" in title for title, _, _ in calls)
    assert any("condizione 1/1" in detail for _, detail, _ in calls)


def test_gui_service_contract() -> None:
    source = inspect.getsource(RunViewer.service_batch_transition)
    assert "pygame.event.get" in source
    assert "_batch_abort_requested" in source
    assert "pygame.display.update" in source
    assert "progress" in source
    assert hasattr(RunViewer, "reset_abort_request")


def main_test() -> None:
    test_diagnostic_heartbeat_and_cancel()
    test_condition_progress_plumbing()
    test_gui_service_contract()
    print("Max Turbo transition watchdog checks passed.")


if __name__ == "__main__":
    main_test()
