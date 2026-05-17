"""
Optimize the task-2 policy-control coefficients alpha_g and alpha_d.

The first implementation uses a transparent rolling grid search. For each
pricing window, it enumerates alpha_g, alpha_d in {0, step, ..., 1}, chooses
the pair with the smallest current-period welfare loss, and then recomputes
the final path with the canonical social_welfare_loss function.

Outputs for equal-weight baseline:
    result/task2_optimization_equal_weight.csv
    result/task2_optimization_equal_weight_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from task2_social_welfare_loss import (
    get_weight_scenario,
    load_parameters,
    social_welfare_loss,
)


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result"
EPS = 1e-12
MIN_PRICE_CNY_PER_TON = 1000.0
FINAL_GRID_STEP = 0.01


def load_event_data(filename: str = "task2_event_model_input_clean.csv") -> pd.DataFrame:
    data = pd.read_csv(
        RESULT_DIR / filename,
        encoding="utf-8-sig",
    )
    data["date"] = pd.to_datetime(data["date"])
    data["month"] = pd.to_datetime(data["month"]).dt.to_period("M").dt.to_timestamp()
    return data.sort_values("date").reset_index(drop=True)


def optimize_equal_weight(
    step: float = 0.05,
    event_filename: str = "task2_event_model_input_clean.csv",
    data_scope: str = "backtest_clean",
    scenario: str = "equal_weight",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_event_data(event_filename)
    params = load_parameters()
    weights = get_weight_scenario(scenario)

    alpha_grid = np.round(np.arange(0.0, 1.0 + step / 2.0, step), 10)
    last_index_by_month = data.groupby("month").tail(1).index
    is_month_last_event = set(last_index_by_month)

    alpha_g_path = []
    alpha_d_path = []

    rho = params["rho"]
    rho_a = params["rho_A"]
    theta_1 = params["theta_1"]
    theta_2 = params["theta_2"]
    b_g = params["B_gasoline"]
    b_d = params["B_diesel"]
    u_g_scale = params["U_gasoline"]
    u_d_scale = params["U_diesel"]
    d_g_scale = params["D_gasoline"]
    d_d_scale = params["D_diesel"]
    t_g_scale = params["T_gasoline"]
    t_d_scale = params["T_diesel"]
    loss_scale_lc = params.get("loss_scale_LC", 1.0)
    loss_scale_lf = params.get("loss_scale_LF", 1.0)
    loss_scale_lv = params.get("loss_scale_LV", 1.0)
    loss_scale_lt = params.get("loss_scale_LT", 1.0)
    loss_scale_le = params.get("loss_scale_LE", 1.0)
    loss_scale_lpi = params.get("loss_scale_Lpi", 1.0)
    loss_cap_lc = params.get("loss_cap_LC", 1e12)
    loss_cap_lf = params.get("loss_cap_LF", 1e12)
    loss_cap_lv = params.get("loss_cap_LV", 1e12)
    loss_cap_lt = params.get("loss_cap_LT", 1e12)
    loss_cap_le = params.get("loss_cap_LE", 1e12)
    loss_cap_lpi = params.get("loss_cap_Lpi", 1e12)
    tau = params["tau"]
    sigma_pi = params["sigma_pi"]
    cpi_lower_target = params.get("cpi_lower_target", 0.02)

    lambda_c = weights["lambda_C"]
    lambda_f = weights["lambda_F"]
    lambda_pi = weights["lambda_pi"]
    lambda_v = weights["lambda_V"]
    lambda_e = weights["lambda_E"]
    lambda_t = weights.get("lambda_T", 0.0)

    p_g_prev = float(data.loc[0, "gasoline_price_before"])
    p_d_prev = float(data.loc[0, "diesel_price_before"])
    s_g = 0.0
    s_d = 0.0
    prev_u_g = 0.0
    prev_u_d = 0.0
    energy_a = 0.0

    month_u_g_sum = 0.0
    month_u_d_sum = 0.0
    current_month = data.loc[0, "month"]
    month_p_g_base = p_g_prev
    month_p_d_base = p_d_prev

    chosen_rows = []

    for idx, row in data.iterrows():
        if row["month"] != current_month:
            current_month = row["month"]
            month_u_g_sum = 0.0
            month_u_d_sum = 0.0
            month_p_g_base = p_g_prev
            month_p_d_base = p_d_prev

        best = None
        for alpha_g in alpha_grid:
            for alpha_d in alpha_grid:
                candidate = _evaluate_candidate(
                    row=row,
                    idx=idx,
                    alpha_g=float(alpha_g),
                    alpha_d=float(alpha_d),
                    p_g_prev=p_g_prev,
                    p_d_prev=p_d_prev,
                    s_g_prev=s_g,
                    s_d_prev=s_d,
                    prev_u_g=prev_u_g,
                    prev_u_d=prev_u_d,
                    energy_a_prev=energy_a,
                    month_u_g_sum=month_u_g_sum,
                    month_u_d_sum=month_u_d_sum,
                    month_p_g_base=month_p_g_base,
                    month_p_d_base=month_p_d_base,
                    is_month_last_event=idx in is_month_last_event,
                    params={
                        "rho": rho,
                        "rho_a": rho_a,
                        "theta_1": theta_1,
                        "theta_2": theta_2,
                        "b_g": b_g,
                        "b_d": b_d,
                        "u_g_scale": u_g_scale,
                        "u_d_scale": u_d_scale,
                        "d_g_scale": d_g_scale,
                        "d_d_scale": d_d_scale,
                        "t_g_scale": t_g_scale,
                        "t_d_scale": t_d_scale,
                        "loss_scale_lc": loss_scale_lc,
                        "loss_scale_lf": loss_scale_lf,
                        "loss_scale_lv": loss_scale_lv,
                        "loss_scale_lt": loss_scale_lt,
                        "loss_scale_le": loss_scale_le,
                        "loss_scale_lpi": loss_scale_lpi,
                        "loss_cap_lc": loss_cap_lc,
                        "loss_cap_lf": loss_cap_lf,
                        "loss_cap_lv": loss_cap_lv,
                        "loss_cap_lt": loss_cap_lt,
                        "loss_cap_le": loss_cap_le,
                        "loss_cap_lpi": loss_cap_lpi,
                        "tau": tau,
                        "sigma_pi": sigma_pi,
                        "cpi_lower_target": cpi_lower_target,
                        "beta0": params["beta0"],
                        "beta_gasoline": params["beta_gasoline"],
                        "beta_diesel": params["beta_diesel"],
                        "beta_pi": params["beta_pi"],
                    },
                    weights={
                        "lambda_c": lambda_c,
                        "lambda_f": lambda_f,
                        "lambda_pi": lambda_pi,
                        "lambda_v": lambda_v,
                        "lambda_e": lambda_e,
                        "lambda_t": lambda_t,
                    },
                )
                if best is None or candidate["current_weighted_loss"] < best["current_weighted_loss"]:
                    best = candidate

        alpha_g_path.append(best["alpha_gasoline"])
        alpha_d_path.append(best["alpha_diesel"])
        chosen_rows.append(best)

        p_g_prev = best["price_gasoline"]
        p_d_prev = best["price_diesel"]
        s_g = best["S_gasoline"]
        s_d = best["S_diesel"]
        prev_u_g = best["u_gasoline"]
        prev_u_d = best["u_diesel"]
        energy_a = best["energy_A"]
        month_u_g_sum += best["u_gasoline"]
        month_u_d_sum += best["u_diesel"]

    final = social_welfare_loss(
        event_data=data,
        parameters=params,
        preference_weights=weights,
        alpha_gasoline=np.array(alpha_g_path),
        alpha_diesel=np.array(alpha_d_path),
    )

    result = final["event_loss"].copy()
    result = result.rename(
        columns={
            "u_gasoline": "optimal_u_gasoline",
            "u_diesel": "optimal_u_diesel",
            "price_gasoline": "optimal_price_gasoline",
            "price_diesel": "optimal_price_diesel",
            "weighted_event_loss": "total_loss",
        }
    )
    result["scenario"] = scenario
    result["data_scope"] = data_scope
    result["grid_step"] = step

    compare_cols = [
        "date",
        "f_gasoline",
        "f_diesel",
        "u_hist_gasoline",
        "u_hist_diesel",
        "gasoline_price_after",
        "diesel_price_after",
    ]
    result = result.merge(data[compare_cols], on="date", how="left")
    result["historical_alpha_gasoline"] = np.where(
        result["f_gasoline"].abs() > EPS,
        result["u_hist_gasoline"] / result["f_gasoline"],
        np.nan,
    )
    result["historical_alpha_diesel"] = np.where(
        result["f_diesel"].abs() > EPS,
        result["u_hist_diesel"] / result["f_diesel"],
        np.nan,
    )

    ordered_cols = [
        "scenario",
        "data_scope",
        "date",
        "month",
        "alpha_gasoline",
        "alpha_diesel",
        "optimal_u_gasoline",
        "optimal_u_diesel",
        "optimal_price_gasoline",
        "optimal_price_diesel",
        "S_gasoline",
        "S_diesel",
        "LC",
        "LF",
        "LV",
        "LT",
        "Lpi",
        "LE",
        "total_loss",
        "f_gasoline",
        "f_diesel",
        "u_hist_gasoline",
        "u_hist_diesel",
        "historical_alpha_gasoline",
        "historical_alpha_diesel",
        "gasoline_price_after",
        "diesel_price_after",
        "grid_step",
    ]
    result = result[ordered_cols]

    summary = final["component_summary"].copy()
    summary.insert(0, "scenario", scenario)
    summary.insert(1, "data_scope", data_scope)
    summary.insert(2, "grid_step", step)
    summary["mean_alpha_gasoline"] = result["alpha_gasoline"].mean()
    summary["mean_alpha_diesel"] = result["alpha_diesel"].mean()
    summary["mean_optimal_u_gasoline"] = result["optimal_u_gasoline"].mean()
    summary["mean_optimal_u_diesel"] = result["optimal_u_diesel"].mean()
    summary["mean_historical_u_gasoline"] = result["u_hist_gasoline"].mean()
    summary["mean_historical_u_diesel"] = result["u_hist_diesel"].mean()
    summary["n_events"] = len(result)

    return result, summary


def _evaluate_candidate(
    row: pd.Series,
    idx: int,
    alpha_g: float,
    alpha_d: float,
    p_g_prev: float,
    p_d_prev: float,
    s_g_prev: float,
    s_d_prev: float,
    prev_u_g: float,
    prev_u_d: float,
    energy_a_prev: float,
    month_u_g_sum: float,
    month_u_d_sum: float,
    month_p_g_base: float,
    month_p_d_base: float,
    is_month_last_event: bool,
    params: dict[str, float],
    weights: dict[str, float],
) -> dict[str, float]:
    f_g = float(row["f_gasoline"])
    f_d = float(row["f_diesel"])
    omega_g = float(row["omega_gasoline"])
    omega_d = float(row["omega_diesel"])

    u_g = alpha_g * f_g
    u_d = alpha_d * f_d
    p_g = p_g_prev + u_g
    p_d = p_d_prev + u_d
    if p_g <= MIN_PRICE_CNY_PER_TON or p_d <= MIN_PRICE_CNY_PER_TON:
        return {
            "row_index": idx,
            "alpha_gasoline": alpha_g,
            "alpha_diesel": alpha_d,
            "u_gasoline": u_g,
            "u_diesel": u_d,
            "price_gasoline": p_g,
            "price_diesel": p_d,
            "S_gasoline": s_g_prev,
            "S_diesel": s_d_prev,
            "LC": 0.0,
            "LF": 0.0,
            "LV": 0.0,
            "LT": 0.0,
            "Lpi": 0.0,
            "LE": 0.0,
            "energy_A": energy_a_prev,
            "current_weighted_loss": 1e18,
        }

    s_g = params["rho"] * s_g_prev + f_g - u_g
    s_d = params["rho"] * s_d_prev + f_d - u_d

    consumer_increase_loss = (
        omega_g * (max(u_g, 0.0) / max(p_g_prev, EPS)) ** 2
        + omega_d * (max(u_d, 0.0) / max(p_d_prev, EPS)) ** 2
    )
    consumer_decrease_reward = (
        omega_g * (max(-u_g, 0.0) / max(p_g_prev, EPS)) ** 2
        + omega_d * (max(-u_d, 0.0) / max(p_d_prev, EPS)) ** 2
    )
    lc_raw = consumer_increase_loss - consumer_decrease_reward
    lc = float(np.clip(lc_raw, -params["loss_cap_lc"], params["loss_cap_lc"])) / params["loss_scale_lc"]
    lf_raw = (
        omega_g * (max(s_g, 0.0) / params["b_g"]) ** 2
        + omega_d * (max(s_d, 0.0) / params["b_d"]) ** 2
    )
    lf = min(lf_raw, params["loss_cap_lf"]) / params["loss_scale_lf"]
    lv_raw = (
        omega_g
        * (
            0.5 * (u_g / params["u_g_scale"]) ** 2
            + 0.5 * ((u_g - prev_u_g) / params["d_g_scale"]) ** 2
        )
        + omega_d
        * (
            0.5 * (u_d / params["u_d_scale"]) ** 2
            + 0.5 * ((u_d - prev_u_d) / params["d_d_scale"]) ** 2
        )
    )
    lv = min(lv_raw, params["loss_cap_lv"]) / params["loss_scale_lv"]
    lt_raw = (
        omega_g * ((u_g - f_g) / params["t_g_scale"]) ** 2
        + omega_d * ((u_d - f_d) / params["t_d_scale"]) ** 2
    )
    lt = min(lt_raw, params["loss_cap_lt"]) / params["loss_scale_lt"]

    p_g_star = p_g_prev + f_g
    p_d_star = p_d_prev + f_d
    aggregate_actual_price = omega_g * p_g + omega_d * p_d
    aggregate_theoretical_price = omega_g * p_g_star + omega_d * p_d_star
    price_gap = max(
        0.0,
        (aggregate_theoretical_price - aggregate_actual_price)
        / max(aggregate_theoretical_price, EPS),
    )
    dependency_component = (
        float(row["oil_import_dependency"])
        * float(row["brent_pressure_h"])
        * price_gap
    )
    shortage_component = float(row["processing_shortage_b"])
    energy_a = max(
        0.0,
        params["rho_a"] * energy_a_prev
        + params["theta_1"] * dependency_component
        + params["theta_2"] * shortage_component,
    )
    le_raw = max(0.0, energy_a - params["tau"]) ** 2
    le = min(le_raw, params["loss_cap_le"]) / params["loss_scale_le"]

    lpi = 0.0
    if is_month_last_event:
        x_g = max(0.0, (month_u_g_sum + u_g) / max(month_p_g_base, EPS))
        x_d = max(0.0, (month_u_d_sum + u_d) / max(month_p_d_base, EPS))
        cpi_hat_next = (
            params["beta0"]
            + params["beta_gasoline"] * x_g
            + params["beta_diesel"] * x_d
            + params["beta_pi"] * float(row["cpi_yoy"])
        )
        inflation_excess = max(0.0, cpi_hat_next - float(row["cpi_target"]))
        deflation_gap = max(0.0, params["cpi_lower_target"] - cpi_hat_next)
        lpi_raw = (inflation_excess / params["sigma_pi"]) ** 2 + (
            deflation_gap / params["sigma_pi"]
        ) ** 2
        lpi = min(lpi_raw, params["loss_cap_lpi"]) / params["loss_scale_lpi"]

    current_weighted_loss = (
        weights["lambda_c"] * lc
        + weights["lambda_f"] * lf
        + weights["lambda_v"] * lv
        + weights["lambda_e"] * le
        + weights["lambda_t"] * lt
        + weights["lambda_pi"] * lpi
    )

    return {
        "row_index": idx,
        "alpha_gasoline": alpha_g,
        "alpha_diesel": alpha_d,
        "u_gasoline": u_g,
        "u_diesel": u_d,
        "price_gasoline": p_g,
        "price_diesel": p_d,
        "S_gasoline": s_g,
        "S_diesel": s_d,
        "LC": lc,
        "LF": lf,
        "LV": lv,
        "LT": lt,
        "Lpi": lpi,
        "LE": le,
        "energy_A": energy_a,
        "current_weighted_loss": current_weighted_loss,
    }

def _write_outputs(result: pd.DataFrame, summary: pd.DataFrame, stem: str) -> tuple[Path, Path]:
    result_path = RESULT_DIR / f"{stem}.csv"
    summary_path = RESULT_DIR / f"{stem}_summary.csv"
    try:
        result.to_csv(result_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        result_path = RESULT_DIR / f"{stem}_updated.csv"
        try:
            result.to_csv(result_path, index=False, encoding="utf-8-sig")
        except PermissionError:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = RESULT_DIR / f"{stem}_{stamp}.csv"
            result.to_csv(result_path, index=False, encoding="utf-8-sig")

    try:
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        summary_path = RESULT_DIR / f"{stem}_summary_updated.csv"
        try:
            summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        except PermissionError:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = RESULT_DIR / f"{stem}_summary_{stamp}.csv"
            summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return result_path, summary_path


def main() -> None:
    scenarios = [
        "consumer_priority",
        "firm_priority",
        "cpi_priority",
        "volatility_priority",
        "energy_priority",
    ]
    runs = [
        (
            "task2_event_model_input_clean.csv",
            "backtest_clean",
            "task2_sensitivity",
        ),
        (
            "task2_event_model_input_forecast.csv",
            "forecast_to_2026",
            "task2_sensitivity_forecast",
        ),
    ]

    for event_filename, data_scope, stem in runs:
        if not (RESULT_DIR / event_filename).exists():
            continue
        combined_summaries = []
        for scenario in scenarios:
            result, summary = optimize_equal_weight(
                step=FINAL_GRID_STEP,
                event_filename=event_filename,
                data_scope=data_scope,
                scenario=scenario,
            )
            scenario_stem = f"{stem}_{scenario}_step001"
            result_path, summary_path = _write_outputs(result, summary, scenario_stem)
            combined_summaries.append(summary)
            print("Generated sensitivity optimization outputs:")
            print(f"- {result_path.name}: {len(result)} rows")
            print(f"- {summary_path.name}")
            print(summary.to_string(index=False))

        combined_summary = pd.concat(combined_summaries, ignore_index=True)
        combined_path = RESULT_DIR / f"{stem}_summary_step001.csv"
        combined_summary.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"- {combined_path.name}: {len(combined_summary)} scenario summaries")


if __name__ == "__main__":
    main()
