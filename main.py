"""Entry point for the continuous multi-floor SAR simulator.

Flow:
1. Initial configuration dialog: floors, Tr, Ts, number of environments,
   topology family, optional chairs/tables/rubble, combinatorial Fw/Opt values
   and all fixed planner weights.
2. For every environment seed, first validate the initial pose for the whole
   Fw x Opt batch.  If the robot cannot plan a path or does not translate
   during the first 5 simulated seconds, the same building is restarted from a
   new random free pose and all conditions are re-run from that pose.
3. For every valid environment, run the Cartesian product Fw x Opt.  Topology,
   object presence, Iw, Dw, Vw, Cw, Ow, Wp and Ms stay fixed across the whole
   experiment.
4. Show the live Pygame simulation, export CSV/XLSX, run statistical tests and
   display the final table.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# CODE-REVIEW NOTES
# Purpose: Experiment orchestration, reproducible run/condition scheduling, retries, exports, and persistent playback speed.
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

import csv
import gc
import math
import sys
from pathlib import Path
import time
from typing import Callable

from gui import (
    run_config_dialog,
    RunViewer,
    show_results_table,
    summarize_results_by_condition,
)
from simulation import RunConfig, RunState, PHYSICS_DT_S
from i18n import set_language, tr, yes_no
from welcome import show_welcome_screen
from statistical_analysis import (
    analyze_results,
    print_statistical_analysis,
    write_results_excel,
)

START_DIAGNOSTIC_SECONDS = 5.0
START_DIAGNOSTIC_MIN_MOVEMENT_M = 0.10
MAX_START_ATTEMPTS = 80
# A five-second simulated start probe normally completes in a few wall-clock
# seconds.  The watchdog prevents a pathological map/planner case from making
# the experiment look permanently frozen between two buildings.
START_DIAGNOSTIC_WALL_TIMEOUT_S = 30.0
# Process GUI events and publish progress frequently while the invisible probe
# runs.  This is especially important in Max Turbo, where the Pygame window is
# intentionally kept alive across episodes.
START_DIAGNOSTIC_HEARTBEAT_STEPS = 20


class ExperimentCancelled(RuntimeError):
    """Raised when the user closes the persistent viewer during preparation."""


def _make_run_config(cfg: dict, run_idx: int, Fw: float, Opt: bool,
                     start_attempt: int) -> RunConfig:
    """Create a RunConfig while keeping building seed and start-attempt separate."""
    return RunConfig(
        n_floors=cfg["n_floors"],
        Tr=cfg["Tr"],
        Ts=cfg["Ts"],
        Fw=Fw,
        Opt=Opt,
        seed=cfg["base_seed"] + run_idx,
        Iw=cfg["Iw"],
        Dw=cfg["Dw"],
        Vw=cfg["Vw"],
        Cw=cfg["Cw"],
        Ow=cfg["Ow"],
        persistence_weight=cfg["persistence_weight"],
        target_switch_margin=cfg["target_switch_margin"],
        layout_mode=cfg["layout_mode"],
        include_objects=cfg["include_objects"],
        object_density=cfg["object_density"],
        start_attempt=start_attempt,
    )


def _diagnose_initial_start(
    run_cfg: RunConfig,
    seconds: float = START_DIAGNOSTIC_SECONDS,
    progress_callback: Callable[[float, str], bool] | None = None,
    wall_timeout_s: float = START_DIAGNOSTIC_WALL_TIMEOUT_S,
) -> tuple[bool, str]:
    """Run a short invisible probe to detect blocked initial poses.

    A start is accepted only if, during the first five simulated seconds, the
    robot manages to produce at least one planned path and translates by at
    least START_DIAGNOSTIC_MIN_MOVEMENT_M.  This catches starts placed in small
    pockets, behind objects, or in areas where the known-space planner cannot
    connect to any frontier after the first scans.
    """
    try:
        state = RunState(run_cfg)
    except Exception as exc:
        return False, tr("initialization_failed", error=exc)

    start_floor = state.robot.floor
    start_x, start_y = state.robot.x, state.robot.y
    max_translation = 0.0
    ever_planned_path = False
    ever_had_candidate = False

    # If Tr is shorter than 5 seconds, use the available budget but keep at
    # least half a second to allow the first sensing/planning cycle.
    diagnostic_seconds = min(seconds, max(0.5, float(run_cfg.Tr)))
    steps = max(1, int(math.ceil(diagnostic_seconds / PHYSICS_DT_S)))

    wall_start = time.perf_counter()
    for step_index in range(steps):
        # The diagnostic used to run as one long, silent CPU-bound block.  In
        # v32 this left the persistent Pygame window without event processing
        # between buildings, so Windows could label it "Not responding" even
        # though the simulator was still validating the next start.  The
        # heartbeat keeps both the OS event queue and the visible progress text
        # alive without changing a single simulated step.
        if (step_index % START_DIAGNOSTIC_HEARTBEAT_STEPS == 0 or
                step_index == steps - 1):
            fraction = step_index / max(1, steps)
            detail = (
                tr("start_check", step=step_index, steps=steps, percent=fraction * 100.0)
            )
            if progress_callback is not None and not progress_callback(
                    fraction, detail):
                raise ExperimentCancelled(tr("experiment_stopped"))

        if time.perf_counter() - wall_start > max(1.0, wall_timeout_s):
            return False, tr(
                "start_timeout",
                seconds=wall_timeout_s,
                simulated=diagnostic_seconds,
            )

        state.step(PHYSICS_DT_S)
        if state.current_frontiers:
            ever_had_candidate = True
        if (state.chosen_frontier is not None and
                state.chosen_frontier.path is not None and
                len(state.chosen_frontier.path) >= 2):
            ever_planned_path = True
        if state.robot.floor != start_floor:
            max_translation = max(max_translation, START_DIAGNOSTIC_MIN_MOVEMENT_M)
        else:
            max_translation = max(
                max_translation,
                math.hypot(state.robot.x - start_x, state.robot.y - start_y),
            )
        if ever_planned_path and max_translation >= START_DIAGNOSTIC_MIN_MOVEMENT_M:
            if progress_callback is not None and not progress_callback(
                    1.0, tr("diagnostic_passed")):
                raise ExperimentCancelled(tr("experiment_stopped"))
            return True, tr("valid_start", movement=max_translation)

    reasons = []
    if not ever_had_candidate:
        reasons.append(tr("no_frontier"))
    if not ever_planned_path:
        reasons.append(tr("no_path"))
    if max_translation < START_DIAGNOSTIC_MIN_MOVEMENT_M:
        reasons.append(tr(
            "insufficient_movement",
            movement=max_translation,
            seconds=diagnostic_seconds,
        ))
    if not reasons:
        reasons.append(tr("diagnostic_failed"))
    return False, "; ".join(reasons)


def _select_valid_start_attempt(
    cfg: dict,
    run_idx: int,
    progress_callback: Callable[[str, str, float], bool] | None = None,
) -> int:
    """Choose a start attempt that works for every Fw x Opt condition.

    The building seed is base_seed + run_idx.  Only the initial-pose RNG is advanced
    when an attempt is rejected, so all Fw x Opt episodes for that building
    continue to share exactly the same environment and starting pose.
    """
    total_conditions = len(cfg["Fw_values"]) * len(cfg["Opt_values"])
    for attempt in range(MAX_START_ATTEMPTS):
        failed_reason = None
        condition_index = 0
        for Fw in cfg["Fw_values"]:
            for Opt in cfg["Opt_values"]:
                condition_index += 1
                preparation_label = tr(
                    "start_preparation",
                    run=run_idx + 1,
                    total=cfg['n_runs'],
                )
                condition_text = (
                    f"start attempt {attempt} | {tr('condition')} "
                    f"{condition_index}/{total_conditions} | "
                    f"Fw={Fw:g} | Opt={'ON' if Opt else 'OFF'}"
                )
                print(f"  {preparation_label}: {condition_text}...", flush=True)
                if progress_callback is not None and not progress_callback(
                        preparation_label, condition_text, 0.0):
                    raise ExperimentCancelled(tr("experiment_stopped"))

                probe_cfg = _make_run_config(cfg, run_idx, Fw, Opt, attempt)

                def condition_progress(fraction: float, detail: str) -> bool:
                    if progress_callback is None:
                        return True
                    return progress_callback(
                        preparation_label,
                        f"{condition_text} | {detail}",
                        fraction,
                    )

                ok, reason = _diagnose_initial_start(
                    probe_cfg,
                    progress_callback=condition_progress,
                )
                if not ok:
                    # Changing ``start_attempt`` only changes the initial pose.
                    # It cannot repair a structural building-generation error.
                    # The old loop retried the exact same impossible geometry
                    # up to 80 times and misleadingly printed "Nuova posizione
                    # casuale".  Abort immediately with a precise diagnostic;
                    # building.py now handles recoverable geometry cases through
                    # deterministic layout retries before RunState is returned.
                    if "initialization" in reason.lower() or "inizializzazione" in reason.lower():
                        raise RuntimeError(tr(
                            "building_uninitializable",
                            building=run_idx + 1,
                            error=reason,
                        ))
                    failed_reason = (
                        f"Fw={Fw:g}, Opt={'ON' if Opt else 'OFF'}: {reason}"
                    )
                    break
            if failed_reason:
                break
        if failed_reason is None:
            if attempt > 0:
                print(tr(
                    "valid_start_found",
                    building=run_idx + 1,
                    restarts=attempt,
                ))
            else:
                print(tr("initial_start_valid", building=run_idx + 1))
            return attempt

        print(tr(
            "start_rejected",
            building=run_idx + 1,
            attempt=attempt,
            reason=failed_reason,
        ))

    raise RuntimeError(tr(
        "cannot_find_start",
        building=run_idx + 1,
        attempts=MAX_START_ATTEMPTS,
    ))


def main() -> None:
    language = show_welcome_screen(Path(__file__).resolve().parent)
    if language is None:
        sys.exit(0)
    set_language(language)
    cfg = run_config_dialog()
    if not cfg:
        print(tr("config_cancelled"))
        sys.exit(0)

    n_combo = len(cfg["Fw_values"]) * len(cfg["Opt_values"])
    total = cfg["n_runs"] * n_combo

    print("=" * 78)
    print(tr("experiment_configuration"))
    print(f"  {tr('label_floors')}: {cfg['n_floors']}")
    print(f"  Tr (s): {cfg['Tr']}")
    print(f"  Ts (s): {cfg['Ts']}")
    print(f"  {tr('label_buildings')}: {cfg['n_runs']}")
    print(f"  {tr('label_initial_seed')}: {cfg['base_seed']}")
    print(
        f"  {tr('label_topology')}: "
        f"{tr('office') if cfg['layout_mode'] == 'office' else tr('free')}"
    )
    print(f"  {tr('label_objects')}: {yes_no(cfg['include_objects'])}")
    print(f"  {tr('label_object_density')}: {cfg['object_density']:g}x")
    print(f"  {tr('label_fw_values')}: {cfg['Fw_values']}")
    print(
        f"  {tr('label_opt_values')}: "
        f"{['ON' if value else 'OFF' for value in cfg['Opt_values']]}"
    )
    print(
        f"  {tr('label_fixed_weights')}: "
        f"Iw={cfg['Iw']:g}, Dw={cfg['Dw']:g}, Vw={cfg['Vw']:g}, "
        f"Cw={cfg['Cw']:g}, Ow={cfg['Ow']:g}, "
        f"Wp={cfg['persistence_weight']:g}, "
        f"Ms={cfg['target_switch_margin']:g}"
    )
    print(
        f"  {tr('label_total_episodes')}: {total} "
        f"({cfg['n_runs']} {tr('buildings')} x {len(cfg['Fw_values'])} Fw "
        f"x {len(cfg['Opt_values'])} Opt)"
    )
    print(f"  {tr('start_diagnostic_note')}")
    print(f"  {tr('fixed_batch_note')}")
    print(f"  {tr('viewer_note')}")
    print("=" * 78)

    all_results: list[dict] = []
    episode = 0
    playback_speed = RunViewer.persistent_time_scale()

    RunViewer.reset_abort_request()

    try:
        for run_idx in range(cfg["n_runs"]):
            def preparation_progress(
                title: str,
                detail: str,
                fraction: float,
            ) -> bool:
                return RunViewer.service_batch_transition(
                    title=title,
                    detail=detail,
                    progress=fraction,
                )

            start_attempt = _select_valid_start_attempt(
                cfg,
                run_idx,
                progress_callback=preparation_progress,
            )

            for Fw in cfg["Fw_values"]:
                for Opt in cfg["Opt_values"]:
                    episode += 1
                    label = (
                        f"Run {run_idx + 1}/{cfg['n_runs']} "
                        f"(episodio {episode}/{total})"
                    )
                    print(
                        f"\n>>> {label} | Fw={Fw:g} | "
                        f"Opt={'ON' if Opt else 'OFF'} | seed={cfg['base_seed'] + run_idx} | "
                        f"start_attempt={start_attempt}"
                    )

                    run_cfg = _make_run_config(cfg, run_idx, Fw, Opt, start_attempt)
                    state = RunState(run_cfg)
                    viewer = RunViewer(
                        state,
                        label,
                        initial_time_scale=playback_speed,
                    )
                    playback_speed = viewer.run()

                    metrics = state.compute_metrics()
                    metrics.update(
                        run=run_idx + 1,
                        seed=cfg["base_seed"] + run_idx,
                        start_attempt=start_attempt,
                        diagnostic_restarts=start_attempt,
                        Fw=Fw,
                        Opt=Opt,
                        Iw=cfg["Iw"],
                        Dw=cfg["Dw"],
                        Vw=cfg["Vw"],
                        Cw=cfg["Cw"],
                        Ow=cfg["Ow"],
                        Wp=cfg["persistence_weight"],
                        Ms=cfg["target_switch_margin"],
                        layout_mode=cfg["layout_mode"],
                        include_objects=cfg["include_objects"],
                        object_density=cfg["object_density"],
                    )
                    all_results.append(metrics)

                    print(
                        f"    Af={metrics['Af']:.1f}%  "
                        f"Af*={metrics['Af_star']:.1f}%  "
                        f"Atot={metrics['Atot']:.1f}%  "
                        f"Vf={metrics['Vf']}  Cf={metrics['Cf']}  "
                        f"t_expl={metrics['texpl']:.0f}s"
                    )
                    print(
                        f"    TP={metrics['TP']} FP={metrics['FP']} "
                        f"FN={metrics['FN']} TN={metrics['TN']}  "
                        f"Sens={metrics['SAR_Sensitivity']:.3f} "
                        f"Spec={metrics['SAR_Specificity']:.3f} "
                        f"BalAcc={metrics['SAR_BalancedAccuracy']:.3f} "
                        f"MCC={metrics['SAR_MCC']:.3f}"
                    )

                    # Release large fine-resolution occupancy arrays and cached
                    # Pygame surfaces before preparing the next condition.  The
                    # periodic collection is cheap compared with a full episode
                    # and prevents long batches from accumulating cyclic GUI
                    # objects until an unpredictable automatic GC pause.
                    del viewer, state
                    if episode % n_combo == 0:
                        gc.collect()

    except ExperimentCancelled:
        print("\n" + tr("experiment_stopped"))
        RunViewer.shutdown_display()
        return

    # The viewer deliberately keeps one shared display alive across episodes
    # so Max Turbo can retain a frozen frame while only Run/Episode changes.
    # Close it once the complete experiment has finished, before opening the
    # Matplotlib result windows.
    RunViewer.shutdown_display()

    fields = [
        "run", "seed", "start_attempt", "diagnostic_restarts",
        "Fw", "Opt",
        "Iw", "Dw", "Vw", "Cw", "Ow", "Wp", "Ms",
        "layout_mode", "include_objects", "object_density",
        "Af", "Af_star", "Atot", "Vf", "Cf", "texpl",
        "TP", "FP", "FN", "TN",
        "SAR_Sensitivity", "SAR_Specificity",
        "SAR_BalancedAccuracy", "SAR_MCC",
    ]

    condition_summary = summarize_results_by_condition(all_results)
    analysis = analyze_results(all_results)

    try:
        with open("sar_simulation_results.csv", "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(all_results)
        print("\n" + tr("saved_episode_csv"))
    except OSError as error:
        print("\n" + tr("cannot_save", item=tr("episode_csv"), error=error))

    summary_fields = [
        "Fw", "Opt", "N",
        "Af_mean", "Af_std",
        "Af_star_mean", "Af_star_std",
        "Atot_mean", "Atot_std",
    ]
    try:
        with open(
            "sar_simulation_condition_statistics.csv",
            "w",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=summary_fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(condition_summary)
        print(tr("saved_condition_csv"))
    except OSError as error:
        print(tr("cannot_save", item=tr("aggregate_csv"), error=error))

    relationship_fields = [
        "scope", "N", "unique_times",
        "pearson_r", "pearson_p", "spearman_rho", "spearman_p",
        "linear_slope", "linear_slope_p", "linear_r2",
        "linear_adjusted_r2", "best_model", "best_model_label",
        "best_model_r2", "best_model_adjusted_r2",
        "delta_aicc_vs_linear", "relationship", "relationship_strength",
        "nonlinearity_evidence", "quadratic_term_p", "note",
    ]
    try:
        with open(
            "sar_simulation_time_area_relationship.csv",
            "w",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=relationship_fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(analysis.get("time_area_relationship", []))
        print(tr("saved_relationship_csv"))
    except OSError as error:
        print(tr("cannot_save", item=tr("time_area_csv"), error=error))

    print("\n" + tr("condition_stats_console"))
    for item in condition_summary:
        print(
            f"  Fw={item['Fw']:g}, Opt={'ON' if item['Opt'] else 'OFF'}, "
            f"N={item['N']}: "
            f"Af={item['Af_mean']:.2f} +/- {item['Af_std']:.2f}, "
            f"Af*={item['Af_star_mean']:.2f} +/- "
            f"{item['Af_star_std']:.2f}, "
            f"Atot={item['Atot_mean']:.2f} +/- {item['Atot_std']:.2f}"
        )

    print_statistical_analysis(analysis)

    try:
        excel_path = "sar_simulation_analysis.xlsx"
        write_results_excel(excel_path, all_results, condition_summary, analysis)
        print(tr("saved_excel", path=excel_path))
    except Exception as error:
        print(tr("cannot_save", item=tr("excel_file"), error=error))

    print(tr("opening_final_tables"))
    show_results_table(all_results, condition_summary, analysis)


if __name__ == "__main__":
    main()
