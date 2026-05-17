"""
Reproduce the final modelling pipeline from raw data and checked-in scripts.

Run from the project root:

    python script/run_reproducible_pipeline.py

The script intentionally delegates the detailed model logic to the task scripts
used during the modelling process, but fixes the execution order and writes the
baseline files that later analyses expect.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = ROOT / "script"
RESULT_DIR = ROOT / "result"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


@dataclass(frozen=True)
class Step:
    name: str
    script: str
    required: bool = True


CORE_STEPS: tuple[Step, ...] = (
    Step("Clean domestic events and oil-price windows", "prepare_clean_data.py"),
    Step("Estimate first-question oil-price weights", "estimate_parameters.py"),
    Step("Estimate structural price-adjustment parameters", "estimate_profit_parameters.py"),
    Step("Export first-question prediction comparison", "export_final_prediction_compare.py"),
    Step("Analyse price-transmission asymmetry", "transmission_asymmetry_analysis.py"),
    Step("Prepare task-2 event and monthly model data", "prepare_task2_model_data.py"),
    Step("Fit task-2 welfare-loss parameters", "fit_task2_parameters.py"),
    Step("Smoke-test task-2 welfare-loss function", "task2_social_welfare_loss.py"),
)


ANALYSIS_STEPS: tuple[Step, ...] = (
    Step("Analyse raw task-2 loss distribution", "analyze_task2_raw_loss_distribution.py"),
    Step("Plot raw task-2 loss histograms", "plot_task2_raw_loss_histograms.py"),
    Step("Compare current, full-transmission, and optimal policies", "compare_task2_policy_strategies.py"),
    Step("Analyse alpha correlations", "analyze_alpha_relationships.py"),
    Step("Analyse alpha partial effects", "analyze_alpha_partial_effects.py"),
    Step("Fit gasoline upward alpha function", "fit_gasoline_alpha_upward_function.py"),
    Step("Fit diesel upward alpha function", "fit_diesel_alpha_upward_function.py"),
    Step("Fit downward alpha functions", "fit_downward_alpha_functions.py"),
    Step("Refit gasoline combined alpha function", "evaluate_refit_gasoline_alpha_combined.py"),
    Step("Refit diesel combined alpha function", "evaluate_refit_diesel_alpha_combined.py"),
    Step("Compare before/after alpha adjustment", "compare_task2_before_after_adjustment.py"),
    Step("Run task-2 sensitivity optimisations", "optimize_task2_policy.py"),
    Step("Extract task-3 simple rules and robustness checks", "task3_extract_simple_rules_robustness.py"),
)


def run_script(step: Step) -> None:
    script_path = SCRIPT_DIR / step.script
    if not script_path.exists():
        if step.required:
            raise FileNotFoundError(script_path)
        print(f"[skip] {step.name}: missing {script_path.name}")
        return

    print(f"\n[run] {step.name}")
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"{step.script} failed with exit code {completed.returncode}")


def write_equal_weight_baselines() -> None:
    """Write the canonical equal-weight optimisation outputs used downstream."""
    from optimize_task2_policy import optimize_equal_weight

    jobs = [
        (
            "task2_event_model_input_clean.csv",
            "backtest_clean",
            "task2_optimization_equal_weight_final_step001_updated.csv",
            "task2_optimization_equal_weight_final_step001_summary.csv",
        ),
        (
            "task2_event_model_input_forecast.csv",
            "forecast_to_2026",
            "task2_optimization_equal_weight_forecast_final_step001.csv",
            "task2_optimization_equal_weight_forecast_final_step001_summary.csv",
        ),
    ]

    print("\n[run] Write canonical equal-weight baseline optimisations")
    for event_file, scope, result_name, summary_name in jobs:
        source = RESULT_DIR / event_file
        if not source.exists():
            print(f"[skip] {event_file}: missing input")
            continue

        result, summary = optimize_equal_weight(
            step=0.01,
            event_filename=event_file,
            data_scope=scope,
            scenario="equal_weight",
        )
        result_path = RESULT_DIR / result_name
        summary_path = RESULT_DIR / summary_name
        result.to_csv(result_path, index=False, encoding="utf-8-sig")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"- {result_path.relative_to(ROOT)}: {len(result)} rows")
        print(f"- {summary_path.relative_to(ROOT)}")


def write_manifest() -> None:
    manifest_path = RESULT_DIR / "pipeline_manifest.csv"
    rows = []
    for index, step in enumerate((*CORE_STEPS, *ANALYSIS_STEPS), start=1):
        rows.append(
            {
                "order": index,
                "step": step.name,
                "script": f"script/{step.script}",
                "required": step.required,
            }
        )
    pd.DataFrame(rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"\n[done] Wrote {manifest_path.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Stop after data preparation, parameter fitting, and baseline optimisation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for step in CORE_STEPS:
        run_script(step)

    write_equal_weight_baselines()

    if not args.core_only:
        for step in ANALYSIS_STEPS:
            run_script(step)

    write_manifest()
    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
