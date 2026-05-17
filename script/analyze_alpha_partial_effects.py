"""
Partial-effect analysis for the optimized policy-control coefficient alpha.

For each candidate driver, this script fixes all other inputs at representative
historical medians, varies the selected driver over its P5-P95 range, and
solves the one-period welfare minimization problem with a 0.01 alpha grid.
The resulting curves show the model-implied response of alpha, not historical
correlations.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = ROOT / "script"
RESULT_DIR = ROOT / "result"
sys.path.insert(0, str(SCRIPT_DIR))

from optimize_task2_policy import EPS, _evaluate_candidate  # noqa: E402
from task2_social_welfare_loss import get_weight_scenario, load_parameters  # noqa: E402


EVENT_FILE = RESULT_DIR / "task2_event_model_input_clean.csv"
BASELINE_ALPHA_FILE = RESULT_DIR / "task2_optimization_equal_weight_final_step001_updated.csv"

GRID_STEP = 0.01
N_X_POINTS = 81

VARIABLES = [
    "f_gasoline",
    "f_diesel",
    "S_gasoline_lag",
    "S_diesel_lag",
    "prev_u_gasoline",
    "prev_u_diesel",
    "cpi_yoy",
    "oil_import_dependency",
    "brent_pressure_h",
    "processing_shortage_b",
    "omega_gasoline",
    "energy_A_lag",
]

VARIABLE_LABELS = {
    "f_gasoline": "gasoline f",
    "f_diesel": "diesel f",
    "S_gasoline_lag": "lagged S, gasoline",
    "S_diesel_lag": "lagged S, diesel",
    "prev_u_gasoline": "previous u, gasoline",
    "prev_u_diesel": "previous u, diesel",
    "cpi_yoy": "CPI YoY",
    "oil_import_dependency": "oil import dependency",
    "brent_pressure_h": "Brent pressure",
    "processing_shortage_b": "processing shortage",
    "omega_gasoline": "gasoline consumption weight",
    "energy_A_lag": "lagged energy pressure A",
}

SCENARIOS = {
    "upward_base": "Representative upward theoretical adjustment",
    "downward_base": "Representative downward theoretical adjustment",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(EVENT_FILE, encoding="utf-8-sig")
    baseline = pd.read_csv(BASELINE_ALPHA_FILE, encoding="utf-8-sig")
    events["date"] = pd.to_datetime(events["date"])
    baseline["date"] = pd.to_datetime(baseline["date"])
    baseline = baseline.sort_values("date").reset_index(drop=True)
    baseline["S_gasoline_lag"] = baseline["S_gasoline"].shift(1).fillna(0.0)
    baseline["S_diesel_lag"] = baseline["S_diesel"].shift(1).fillna(0.0)
    baseline["prev_u_gasoline"] = baseline["optimal_u_gasoline"].shift(1).fillna(0.0)
    baseline["prev_u_diesel"] = baseline["optimal_u_diesel"].shift(1).fillna(0.0)
    return events, baseline


def parameter_bundle() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    params = load_parameters()
    weights = get_weight_scenario("equal_weight")
    candidate_params = {
        "rho": params["rho"],
        "rho_a": params["rho_A"],
        "theta_1": params["theta_1"],
        "theta_2": params["theta_2"],
        "b_g": params["B_gasoline"],
        "b_d": params["B_diesel"],
        "u_g_scale": params["U_gasoline"],
        "u_d_scale": params["U_diesel"],
        "d_g_scale": params["D_gasoline"],
        "d_d_scale": params["D_diesel"],
        "t_g_scale": params["T_gasoline"],
        "t_d_scale": params["T_diesel"],
        "loss_scale_lc": params.get("loss_scale_LC", 1.0),
        "loss_scale_lf": params.get("loss_scale_LF", 1.0),
        "loss_scale_lv": params.get("loss_scale_LV", 1.0),
        "loss_scale_lt": params.get("loss_scale_LT", 1.0),
        "loss_scale_le": params.get("loss_scale_LE", 1.0),
        "loss_scale_lpi": params.get("loss_scale_Lpi", 1.0),
        "loss_cap_lc": params.get("loss_cap_LC", 1e12),
        "loss_cap_lf": params.get("loss_cap_LF", 1e12),
        "loss_cap_lv": params.get("loss_cap_LV", 1e12),
        "loss_cap_lt": params.get("loss_cap_LT", 1e12),
        "loss_cap_le": params.get("loss_cap_LE", 1e12),
        "loss_cap_lpi": params.get("loss_cap_Lpi", 1e12),
        "tau": params["tau"],
        "sigma_pi": params["sigma_pi"],
        "cpi_lower_target": params.get("cpi_lower_target", 0.02),
        "beta0": params["beta0"],
        "beta_gasoline": params["beta_gasoline"],
        "beta_diesel": params["beta_diesel"],
        "beta_pi": params["beta_pi"],
    }
    candidate_weights = {
        "lambda_c": weights["lambda_C"],
        "lambda_f": weights["lambda_F"],
        "lambda_pi": weights["lambda_pi"],
        "lambda_v": weights["lambda_V"],
        "lambda_e": weights["lambda_E"],
        "lambda_t": weights.get("lambda_T", 0.0),
    }
    return params, candidate_params, candidate_weights


def nonzero_median(series: pd.Series, sign: str) -> float:
    if sign == "positive":
        values = series.loc[series > EPS]
    else:
        values = series.loc[series < -EPS]
    if values.empty:
        return float(series.median())
    return float(values.median())


def variable_range(variable: str, events: pd.DataFrame, baseline: pd.DataFrame) -> np.ndarray:
    if variable in ["S_gasoline_lag", "S_diesel_lag", "prev_u_gasoline", "prev_u_diesel"]:
        source = baseline[variable]
    elif variable == "energy_A_lag":
        source = events["energy_security_A_raw"]
    else:
        source = events[variable]

    low = float(source.quantile(0.05))
    high = float(source.quantile(0.95))
    if abs(high - low) < EPS:
        low = float(source.min())
        high = float(source.max())
    if abs(high - low) < EPS:
        return np.array([low], dtype=float)
    return np.linspace(low, high, N_X_POINTS)


def build_base_state(events: pd.DataFrame, baseline: pd.DataFrame, scenario: str) -> dict[str, float | pd.Series]:
    sign = "positive" if scenario == "upward_base" else "negative"
    f_g = nonzero_median(events["f_gasoline"], sign)
    f_d = nonzero_median(events["f_diesel"], sign)

    omega_g = float(events["omega_gasoline"].median())
    row = pd.Series(
        {
            "f_gasoline": f_g,
            "f_diesel": f_d,
            "omega_gasoline": omega_g,
            "omega_diesel": 1.0 - omega_g,
            "oil_import_dependency": float(events["oil_import_dependency"].median()),
            "brent_pressure_h": float(events["brent_pressure_h"].median()),
            "processing_shortage_b": float(events["processing_shortage_b"].median()),
            "cpi_yoy": float(events["cpi_yoy"].median()),
            "cpi_target": float(events["cpi_target"].median()),
        }
    )
    state = {
        "row": row,
        "p_g_prev": float(events["gasoline_price_before"].median()),
        "p_d_prev": float(events["diesel_price_before"].median()),
        "s_g_prev": float(baseline["S_gasoline_lag"].median()),
        "s_d_prev": float(baseline["S_diesel_lag"].median()),
        "prev_u_g": float(baseline["prev_u_gasoline"].median()),
        "prev_u_d": float(baseline["prev_u_diesel"].median()),
        "energy_a_prev": float(events["energy_security_A_raw"].median()),
        "month_u_g_sum": 0.0,
        "month_u_d_sum": 0.0,
        "month_p_g_base": float(events["gasoline_price_before"].median()),
        "month_p_d_base": float(events["diesel_price_before"].median()),
    }
    return state


def apply_variable(state: dict[str, float | pd.Series], variable: str, value: float) -> dict[str, float | pd.Series]:
    updated = state.copy()
    updated["row"] = state["row"].copy()
    row = updated["row"]

    if variable in ["f_gasoline", "f_diesel", "cpi_yoy", "oil_import_dependency", "brent_pressure_h", "processing_shortage_b"]:
        row[variable] = value
    elif variable == "omega_gasoline":
        row["omega_gasoline"] = value
        row["omega_diesel"] = 1.0 - value
    elif variable == "S_gasoline_lag":
        updated["s_g_prev"] = value
    elif variable == "S_diesel_lag":
        updated["s_d_prev"] = value
    elif variable == "prev_u_gasoline":
        updated["prev_u_g"] = value
    elif variable == "prev_u_diesel":
        updated["prev_u_d"] = value
    elif variable == "energy_A_lag":
        updated["energy_a_prev"] = value
    else:
        raise ValueError(f"Unknown variable: {variable}")
    return updated


def optimize_one_state(
    state: dict[str, float | pd.Series],
    candidate_params: dict[str, float],
    candidate_weights: dict[str, float],
    alpha_grid: np.ndarray,
) -> dict[str, float]:
    best = None
    for alpha_g in alpha_grid:
        for alpha_d in alpha_grid:
            candidate = _evaluate_candidate(
                row=state["row"],
                idx=0,
                alpha_g=float(alpha_g),
                alpha_d=float(alpha_d),
                p_g_prev=float(state["p_g_prev"]),
                p_d_prev=float(state["p_d_prev"]),
                s_g_prev=float(state["s_g_prev"]),
                s_d_prev=float(state["s_d_prev"]),
                prev_u_g=float(state["prev_u_g"]),
                prev_u_d=float(state["prev_u_d"]),
                energy_a_prev=float(state["energy_a_prev"]),
                month_u_g_sum=float(state["month_u_g_sum"]),
                month_u_d_sum=float(state["month_u_d_sum"]),
                month_p_g_base=float(state["month_p_g_base"]),
                month_p_d_base=float(state["month_p_d_base"]),
                is_month_last_event=True,
                params=candidate_params,
                weights=candidate_weights,
            )
            if best is None or candidate["current_weighted_loss"] < best["current_weighted_loss"]:
                best = candidate
    return best


def run_partial_effects() -> pd.DataFrame:
    events, baseline = load_data()
    _, candidate_params, candidate_weights = parameter_bundle()
    alpha_grid = np.round(np.arange(0.0, 1.0 + GRID_STEP / 2.0, GRID_STEP), 10)

    rows = []
    for scenario in SCENARIOS:
        base_state = build_base_state(events, baseline, scenario)
        for variable in VARIABLES:
            for value in variable_range(variable, events, baseline):
                state = apply_variable(base_state, variable, float(value))
                best = optimize_one_state(state, candidate_params, candidate_weights, alpha_grid)
                rows.append(
                    {
                        "scenario": scenario,
                        "variable": variable,
                        "variable_label": VARIABLE_LABELS[variable],
                        "x_value": float(value),
                        "alpha_gasoline": best["alpha_gasoline"],
                        "alpha_diesel": best["alpha_diesel"],
                        "u_gasoline": best["u_gasoline"],
                        "u_diesel": best["u_diesel"],
                        "LC": best["LC"],
                        "LF": best["LF"],
                        "LV": best["LV"],
                        "LT": best["LT"],
                        "Lpi": best["Lpi"],
                        "LE": best["LE"],
                        "weighted_loss": best["current_weighted_loss"],
                        "base_f_gasoline": float(state["row"]["f_gasoline"]),
                        "base_f_diesel": float(state["row"]["f_diesel"]),
                    }
                )
    return pd.DataFrame(rows)


def plot_partial_grid(result: pd.DataFrame, scenario: str) -> Path:
    subset = result.loc[result["scenario"] == scenario]
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    axes = axes.ravel()
    for ax, variable in zip(axes, VARIABLES):
        data = subset.loc[subset["variable"] == variable]
        ax.scatter(data["x_value"], data["alpha_gasoline"], s=14, alpha=0.8, label="gasoline alpha")
        ax.plot(data["x_value"], data["alpha_gasoline"], linewidth=1.8)
        ax.scatter(data["x_value"], data["alpha_diesel"], s=14, alpha=0.8, label="diesel alpha")
        ax.plot(data["x_value"], data["alpha_diesel"], linewidth=1.8)
        ax.set_title(VARIABLE_LABELS[variable], fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("optimal alpha")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(VARIABLES) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.985))
    fig.suptitle(f"Partial-effect curves: {SCENARIOS[scenario]}", y=0.995)
    fig.tight_layout(rect=[0, 0, 0.98, 0.98])
    path = RESULT_DIR / f"task2_alpha_partial_effects_{scenario}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def summarize_effects(result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, variable), group in result.groupby(["scenario", "variable"]):
        ordered = group.sort_values("x_value")
        for fuel in ["gasoline", "diesel"]:
            y = ordered[f"alpha_{fuel}"]
            rows.append(
                {
                    "scenario": scenario,
                    "variable": variable,
                    "fuel": fuel,
                    "alpha_min": y.min(),
                    "alpha_max": y.max(),
                    "alpha_range": y.max() - y.min(),
                    "alpha_first": y.iloc[0],
                    "alpha_last": y.iloc[-1],
                    "direction": "increasing"
                    if y.iloc[-1] > y.iloc[0] + 1e-9
                    else ("decreasing" if y.iloc[-1] < y.iloc[0] - 1e-9 else "flat"),
                    "n_unique_alpha": y.nunique(),
                }
            )
    return pd.DataFrame(rows).sort_values(["scenario", "fuel", "alpha_range"], ascending=[True, True, False])


def main() -> None:
    result = run_partial_effects()
    result_path = RESULT_DIR / "task2_alpha_partial_effects.csv"
    result.to_csv(result_path, index=False, encoding="utf-8-sig")

    summary = summarize_effects(result)
    summary_path = RESULT_DIR / "task2_alpha_partial_effects_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    plot_paths = [plot_partial_grid(result, scenario) for scenario in SCENARIOS]

    print("Generated alpha partial-effect analysis:")
    print(f"- {result_path.name}: {len(result)} rows")
    print(f"- {summary_path.name}: {len(summary)} rows")
    for path in plot_paths:
        print(f"- {path.name}")
    print()
    print("Largest partial effects by scenario/fuel:")
    for scenario in SCENARIOS:
        print(f"\n{scenario}")
        print(
            summary.loc[summary["scenario"] == scenario, ["variable", "fuel", "alpha_range", "direction", "n_unique_alpha"]]
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
