"""Lightweight checks for v15 statistical analysis and start-attempt plumbing."""

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

from pathlib import Path

from simulation import RunConfig
from statistical_analysis import analyze_results, write_results_excel


def make_dummy_results():
    rows = []
    for run in range(1, 5):
        for fw in (0.0, 0.3):
            for opt in (False, True):
                opt_bonus = 2.0 if opt else 0.0
                fw_bonus = 3.0 * fw
                base = 20.0 + run
                rows.append({
                    "run": run,
                    "start_attempt": 0,
                    "diagnostic_restarts": 0,
                    "Fw": fw,
                    "Opt": opt,
                    "Iw": 1.0,
                    "Dw": 1.0,
                    "Vw": 0.3,
                    "Cw": 3.0,
                    "Ow": 8.0,
                    "Wp": 24.0,
                    "Ms": 1.0,
                    "layout_mode": "office",
                    "include_objects": True,
                    "object_density": 1.0,
                    "Af": base + fw_bonus + opt_bonus,
                    "Af_star": base + 0.5 * fw_bonus + 0.25 * opt_bonus,
                    "Atot": base + 0.75 * fw_bonus + 0.5 * opt_bonus,
                    "Vf": 2,
                    "Cf": 1,
                    "texpl": 600.0,
                    "TP": 1,
                    "FP": 0,
                    "FN": 0,
                    "TN": 10.0,
                    "SAR_Sensitivity": 1.0,
                    "SAR_Specificity": 1.0,
                    "SAR_BalancedAccuracy": 1.0,
                    "SAR_MCC": 1.0,
                })
    return rows


def main():
    # The start_attempt argument must not alter the building seed; it only
    # changes the random start selector.
    cfg0 = RunConfig(4, 600, 60, 0.3, True, 7, start_attempt=0)
    cfg1 = RunConfig(4, 600, 60, 0.3, True, 7, start_attempt=3)
    assert cfg0.seed == cfg1.seed == 7
    assert cfg0.start_attempt == 0
    assert cfg1.start_attempt == 3

    results = make_dummy_results()
    analysis = analyze_results(results)
    assert len(analysis["anova_all_conditions"]) == 3
    assert len(analysis["welch_opt"]) == 3
    assert len(analysis["anova_fw"]) == 3
    assert analysis["tukey_all_conditions"]
    assert analysis["tukey_fw"]

    condition_summary = []
    for fw in (0.0, 0.3):
        for opt in (False, True):
            group = [r for r in results if r["Fw"] == fw and r["Opt"] == opt]
            condition_summary.append({
                "Fw": fw,
                "Opt": opt,
                "N": len(group),
                "Af_mean": sum(r["Af"] for r in group) / len(group),
                "Af_std": 0.0,
                "Af_star_mean": sum(r["Af_star"] for r in group) / len(group),
                "Af_star_std": 0.0,
                "Atot_mean": sum(r["Atot"] for r in group) / len(group),
                "Atot_std": 0.0,
            })

    output = Path("_verify_v15_statistics.xlsx")
    write_results_excel(output, results, condition_summary, analysis)
    assert output.exists() and output.stat().st_size > 0
    output.unlink()
    print("OK: statistical analysis, Excel export and start_attempt plumbing")


if __name__ == "__main__":
    main()
