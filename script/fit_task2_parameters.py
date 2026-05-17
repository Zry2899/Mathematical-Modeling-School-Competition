"""
Fit and set parameters for the simplified task-2 price-control model.

Inputs:
    result/task2_event_model_input_clean.csv

Outputs:
    result/task2_fitted_parameters.csv
    result/task2_weight_scenarios.csv
    result/task2_cpi_regression_data.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result"

RHO = 0.7
RHO_A = 0.7
THETA_1 = 1.0
THETA_2 = 1.0
EPS = 1e-9


def pctl(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[np.isfinite(values)]
    if values.empty:
        return float("nan")
    return float(np.quantile(values, q))


def safe_scale(value: float, fallback: float = 1.0) -> float:
    if pd.isna(value) or abs(value) < EPS:
        return fallback
    return float(value)


def load_event_data() -> pd.DataFrame:
    data = pd.read_csv(RESULT_DIR / "task2_event_model_input_clean.csv")
    data["date"] = pd.to_datetime(data["date"])
    data["month"] = pd.to_datetime(data["month"])
    return data.sort_values("date").reset_index(drop=True)


def add_historical_pressure(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    pressure_g = []
    pressure_d = []
    s_g = 0.0
    s_d = 0.0

    for row in data.itertuples(index=False):
        s_g = RHO * s_g + row.f_gasoline - row.u_hist_gasoline
        s_d = RHO * s_d + row.f_diesel - row.u_hist_diesel
        pressure_g.append(s_g)
        pressure_d.append(s_d)

    data["S_hist_gasoline"] = pressure_g
    data["S_hist_diesel"] = pressure_d
    return data


def build_cpi_regression_data(data: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        data.groupby("month", as_index=False)
        .agg(
            cpi_yoy=("cpi_yoy", "first"),
            cpi_target=("cpi_target", "first"),
            u_gasoline_sum=("u_hist_gasoline", "sum"),
            u_diesel_sum=("u_hist_diesel", "sum"),
            gasoline_price_base=("gasoline_price_before", "first"),
            diesel_price_base=("diesel_price_before", "first"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )

    monthly["x_gasoline"] = (
        monthly["u_gasoline_sum"] / monthly["gasoline_price_base"]
    ).clip(lower=0.0)
    monthly["x_diesel"] = (
        monthly["u_diesel_sum"] / monthly["diesel_price_base"]
    ).clip(lower=0.0)
    monthly["cpi_yoy_next"] = monthly["cpi_yoy"].shift(-1)
    monthly = monthly.dropna(
        subset=["x_gasoline", "x_diesel", "cpi_yoy", "cpi_yoy_next"]
    ).reset_index(drop=True)

    return monthly


def fit_cpi_regression(monthly: pd.DataFrame) -> dict[str, float]:
    x = monthly[["x_gasoline", "x_diesel", "cpi_yoy"]].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    y = monthly["cpi_yoy_next"].to_numpy(dtype=float)

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else float("nan")
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))

    monthly["cpi_yoy_pred_next"] = y_hat

    return {
        "beta0": float(beta[0]),
        "beta_gasoline": float(beta[1]),
        "beta_diesel": float(beta[2]),
        "beta_pi": float(beta[3]),
        "cpi_regression_r2": r2,
        "cpi_regression_rmse": rmse,
        "cpi_regression_n": float(len(monthly)),
    }


def compute_historical_loss_scales(
    data: pd.DataFrame,
    cpi_monthly: pd.DataFrame,
    cpi_fit: dict[str, float],
    scales: dict[str, float],
) -> dict[str, float]:
    rows = []
    s_g = 0.0
    s_d = 0.0
    prev_u_g = 0.0
    prev_u_d = 0.0
    energy_a = 0.0

    for row in data.itertuples(index=False):
        f_g = float(row.f_gasoline)
        f_d = float(row.f_diesel)
        u_g = float(row.u_hist_gasoline)
        u_d = float(row.u_hist_diesel)
        p_g_prev = float(row.gasoline_price_before)
        p_d_prev = float(row.diesel_price_before)
        omega_g = float(row.omega_gasoline)
        omega_d = float(row.omega_diesel)

        s_g = RHO * s_g + f_g - u_g
        s_d = RHO * s_d + f_d - u_d

        consumer_increase_loss = (
            omega_g * (max(u_g, 0.0) / max(p_g_prev, EPS)) ** 2
            + omega_d * (max(u_d, 0.0) / max(p_d_prev, EPS)) ** 2
        )
        consumer_decrease_reward = (
            omega_g * (max(-u_g, 0.0) / max(p_g_prev, EPS)) ** 2
            + omega_d * (max(-u_d, 0.0) / max(p_d_prev, EPS)) ** 2
        )
        lc = consumer_increase_loss - consumer_decrease_reward

        lf = (
            omega_g * (max(s_g, 0.0) / scales["B_gasoline"]) ** 2
            + omega_d * (max(s_d, 0.0) / scales["B_diesel"]) ** 2
        )
        lv = (
            omega_g
            * (
                0.5 * (u_g / scales["U_gasoline"]) ** 2
                + 0.5 * ((u_g - prev_u_g) / scales["D_gasoline"]) ** 2
            )
            + omega_d
            * (
                0.5 * (u_d / scales["U_diesel"]) ** 2
                + 0.5 * ((u_d - prev_u_d) / scales["D_diesel"]) ** 2
            )
        )
        lt = (
            omega_g * ((u_g - f_g) / scales["T_gasoline"]) ** 2
            + omega_d * ((u_d - f_d) / scales["T_diesel"]) ** 2
        )

        energy_a = max(
            0.0,
            RHO_A * energy_a
            + THETA_1
            * float(row.oil_import_dependency)
            * float(row.brent_pressure_h)
            * float(row.price_gap_g)
            + THETA_2 * float(row.processing_shortage_b),
        )
        le = max(0.0, energy_a - scales["tau"]) ** 2

        rows.append({
            "month": row.month,
            "LC": lc,
            "LF": lf,
            "LV": lv,
            "LT": lt,
            "LE": le,
        })
        prev_u_g = u_g
        prev_u_d = u_d

    event_loss = pd.DataFrame(rows)

    monthly = cpi_monthly.copy()
    monthly["cpi_hat_next"] = (
        cpi_fit["beta0"]
        + cpi_fit["beta_gasoline"] * monthly["x_gasoline"]
        + cpi_fit["beta_diesel"] * monthly["x_diesel"]
        + cpi_fit["beta_pi"] * monthly["cpi_yoy"]
    )
    monthly["inflation_excess"] = (
        monthly["cpi_hat_next"] - monthly["cpi_target"]
    ).clip(lower=0.0)
    monthly["deflation_gap"] = (0.02 - monthly["cpi_hat_next"]).clip(lower=0.0)
    monthly["Lpi"] = (
        (monthly["inflation_excess"] / scales["sigma_pi"]) ** 2
        + (monthly["deflation_gap"] / scales["sigma_pi"]) ** 2
    )

    return {
        "loss_scale_LC": _loss_scale(event_loss["LC"].abs()),
        "loss_scale_LF": _loss_scale(event_loss["LF"]),
        "loss_scale_LV": _loss_scale(event_loss["LV"]),
        "loss_scale_LT": _loss_scale(event_loss["LT"]),
        "loss_scale_LE": _loss_scale(event_loss["LE"]),
        "loss_scale_Lpi": _loss_scale(monthly["Lpi"]),
        "loss_cap_LC": _loss_cap(event_loss["LC"].abs()),
        "loss_cap_LF": _loss_cap(event_loss["LF"]),
        "loss_cap_LV": _loss_cap(event_loss["LV"]),
        "loss_cap_LT": _loss_cap(event_loss["LT"]),
        "loss_cap_LE": _loss_cap(event_loss["LE"]),
        "loss_cap_Lpi": _loss_cap(monthly["Lpi"]),
        "loss_scale_method": "P90 of raw loss components after substituting the historical actual adjustment path.",
    }


def _loss_scale(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[np.isfinite(values)]
    if values.empty:
        return 1.0
    clipped = np.minimum(values.abs(), np.quantile(values.abs(), 0.95))
    scale = float(np.sqrt(np.mean(clipped ** 2)))
    if scale > EPS:
        return scale
    mean_abs = float(clipped.mean())
    if mean_abs > EPS:
        return mean_abs
    return 1.0


def _loss_cap(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[np.isfinite(values)]
    if values.empty:
        return 1.0
    cap = float(np.quantile(values.abs(), 0.95))
    if cap > EPS:
        return cap
    return 1.0


def compute_energy_pressure(data: pd.DataFrame) -> pd.Series:
    a_values = []
    a = 0.0
    for row in data.itertuples(index=False):
        dependency_component = (
            row.oil_import_dependency * row.brent_pressure_h * row.price_gap_g
        )
        shortage_component = row.processing_shortage_b
        a = max(0.0, RHO_A * a + THETA_1 * dependency_component + THETA_2 * shortage_component)
        a_values.append(a)
    return pd.Series(a_values, index=data.index)


def fit_parameters(data: pd.DataFrame, cpi_monthly: pd.DataFrame) -> pd.DataFrame:
    data = add_historical_pressure(data)

    positive_s_g = data["S_hist_gasoline"].clip(lower=0.0)
    positive_s_d = data["S_hist_diesel"].clip(lower=0.0)

    u_g = data["u_hist_gasoline"]
    u_d = data["u_hist_diesel"]
    du_g = u_g.diff()
    du_d = u_d.diff()

    energy_a = compute_energy_pressure(data)

    cpi_fit = fit_cpi_regression(cpi_monthly)

    sigma_pi = safe_scale(float(data["cpi_yoy"].std(ddof=1)), fallback=0.01)
    b_g = safe_scale(pctl(positive_s_g[positive_s_g > 0], 0.9))
    b_d = safe_scale(pctl(positive_s_d[positive_s_d > 0], 0.9))
    u_scale_g = safe_scale(pctl(u_g.abs(), 0.9))
    u_scale_d = safe_scale(pctl(u_d.abs(), 0.9))
    d_scale_g = safe_scale(pctl(du_g.abs(), 0.9))
    d_scale_d = safe_scale(pctl(du_d.abs(), 0.9))
    t_scale_g = safe_scale(pctl(data["f_gasoline"].abs(), 0.9))
    t_scale_d = safe_scale(pctl(data["f_diesel"].abs(), 0.9))
    tau = safe_scale(pctl(energy_a, 0.75), fallback=0.0)
    base_scales = {
        "B_gasoline": b_g,
        "B_diesel": b_d,
        "U_gasoline": u_scale_g,
        "U_diesel": u_scale_d,
        "D_gasoline": d_scale_g,
        "D_diesel": d_scale_d,
        "T_gasoline": t_scale_g,
        "T_diesel": t_scale_d,
        "sigma_pi": sigma_pi,
        "tau": tau,
    }
    loss_scales = compute_historical_loss_scales(data, cpi_monthly, cpi_fit, base_scales)

    rows = [
        ("rho", RHO, "set", "Historical pressure decay coefficient."),
        ("rho_A", RHO_A, "set", "Energy-security pressure decay coefficient."),
        ("theta_1", THETA_1, "set", "Weight for import-price-gap energy pressure component."),
        ("theta_2", THETA_2, "set", "Weight for processing-shortage energy pressure component."),
        ("B_gasoline", b_g, "fitted_p90", "P90 of positive historical gasoline pressure."),
        ("B_diesel", b_d, "fitted_p90", "P90 of positive historical diesel pressure."),
        ("U_gasoline", u_scale_g, "fitted_p90", "P90 of absolute historical gasoline adjustment."),
        ("U_diesel", u_scale_d, "fitted_p90", "P90 of absolute historical diesel adjustment."),
        ("D_gasoline", d_scale_g, "fitted_p90", "P90 of absolute change in gasoline adjustment."),
        ("D_diesel", d_scale_d, "fitted_p90", "P90 of absolute change in diesel adjustment."),
        ("T_gasoline", t_scale_g, "fitted_p90", "P90 of absolute theoretical gasoline adjustment; scale for transmission-deviation loss."),
        ("T_diesel", t_scale_d, "fitted_p90", "P90 of absolute theoretical diesel adjustment; scale for transmission-deviation loss."),
        ("sigma_pi", sigma_pi, "fitted_std", "Standard deviation of historical CPI yoy."),
        ("cpi_lower_target", 0.02, "set", "Lower CPI bound; CPI below 2% is treated as deflation-pressure loss."),
        ("tau", tau, "fitted_p75", "P75 of historical recursive energy-security pressure A_t."),
        ("loss_scale_LC", loss_scales["loss_scale_LC"], "historical_path_winsorized_rms", "Component-level normalization scale for absolute consumer loss/reward; values are clipped at historical P95 before RMS."),
        ("loss_scale_LF", loss_scales["loss_scale_LF"], "historical_path_winsorized_rms", "Component-level normalization scale for firm-pressure loss; values are clipped at historical P95 before RMS."),
        ("loss_scale_LV", loss_scales["loss_scale_LV"], "historical_path_winsorized_rms", "Component-level normalization scale for volatility loss; values are clipped at historical P95 before RMS."),
        ("loss_scale_LT", loss_scales["loss_scale_LT"], "historical_path_winsorized_rms", "Component-level normalization scale for transmission-deviation loss; values are clipped at historical P95 before RMS."),
        ("loss_scale_LE", loss_scales["loss_scale_LE"], "historical_path_winsorized_rms", "Component-level normalization scale for energy-security loss; values are clipped at historical P95 before RMS."),
        ("loss_scale_Lpi", loss_scales["loss_scale_Lpi"], "historical_path_winsorized_rms", "Component-level normalization scale for CPI loss; values are clipped at historical P95 before RMS."),
        ("loss_cap_LC", loss_scales["loss_cap_LC"], "historical_path_p95", "P95 clipping cap for absolute consumer loss/reward."),
        ("loss_cap_LF", loss_scales["loss_cap_LF"], "historical_path_p95", "P95 clipping cap for firm-pressure loss."),
        ("loss_cap_LV", loss_scales["loss_cap_LV"], "historical_path_p95", "P95 clipping cap for volatility loss."),
        ("loss_cap_LT", loss_scales["loss_cap_LT"], "historical_path_p95", "P95 clipping cap for transmission-deviation loss."),
        ("loss_cap_LE", loss_scales["loss_cap_LE"], "historical_path_p95", "P95 clipping cap for energy-security loss."),
        ("loss_cap_Lpi", loss_scales["loss_cap_Lpi"], "historical_path_p95", "P95 clipping cap for CPI loss."),
        ("beta0", cpi_fit["beta0"], "fitted_ols", "CPI transmission equation intercept."),
        ("beta_gasoline", cpi_fit["beta_gasoline"], "fitted_ols", "CPI transmission coefficient for gasoline adjustment shock."),
        ("beta_diesel", cpi_fit["beta_diesel"], "fitted_ols", "CPI transmission coefficient for diesel adjustment shock."),
        ("beta_pi", cpi_fit["beta_pi"], "fitted_ols", "CPI autoregressive coefficient."),
        ("cpi_regression_r2", cpi_fit["cpi_regression_r2"], "diagnostic", "OLS in-sample R2."),
        ("cpi_regression_rmse", cpi_fit["cpi_regression_rmse"], "diagnostic", "OLS in-sample RMSE."),
        ("cpi_regression_n", cpi_fit["cpi_regression_n"], "diagnostic", "Number of monthly observations in CPI regression."),
        ("lambda_C", 1.0 / 6.0, "baseline_policy_weight", "Equal weight across all six welfare-loss terms."),
        ("lambda_F", 1.0 / 6.0, "baseline_policy_weight", "Equal weight across all six welfare-loss terms."),
        ("lambda_pi", 1.0 / 6.0, "baseline_policy_weight", "Equal weight across all six welfare-loss terms."),
        ("lambda_V", 1.0 / 6.0, "baseline_policy_weight", "Equal weight across all six welfare-loss terms."),
        ("lambda_E", 1.0 / 6.0, "baseline_policy_weight", "Equal weight across all six welfare-loss terms."),
        ("lambda_T", 1.0 / 6.0, "baseline_policy_weight", "Equal weight across all six welfare-loss terms."),
    ]

    return pd.DataFrame(rows, columns=["parameter", "value", "method", "note"])


def build_weight_scenarios() -> pd.DataFrame:
    high = 0.30
    low = (1.0 - high) / 5.0
    rows = [
        ("equal_weight", 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, "Equal weights across all six welfare-loss terms."),
        ("consumer_priority", high, low, low, low, low, low, "Sensitivity case: consumer welfare loss weight raised to 0.30."),
        ("firm_priority", low, high, low, low, low, low, "Sensitivity case: firm/fiscal pressure loss weight raised to 0.30."),
        ("cpi_priority", low, low, high, low, low, low, "Sensitivity case: CPI-stability loss weight raised to 0.30."),
        ("volatility_priority", low, low, low, high, low, low, "Sensitivity case: adjustment-volatility loss weight raised to 0.30."),
        ("energy_priority", low, low, low, low, high, low, "Sensitivity case: energy-security loss weight raised to 0.30."),
    ]
    return pd.DataFrame(
        rows,
        columns=["scenario", "lambda_C", "lambda_F", "lambda_pi", "lambda_V", "lambda_E", "lambda_T", "note"],
    )


def main() -> None:
    data = load_event_data()
    cpi_monthly = build_cpi_regression_data(data)
    parameters = fit_parameters(data, cpi_monthly)
    scenarios = build_weight_scenarios()

    parameters.to_csv(
        RESULT_DIR / "task2_fitted_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scenarios.to_csv(
        RESULT_DIR / "task2_weight_scenarios.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cpi_monthly.to_csv(
        RESULT_DIR / "task2_cpi_regression_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Generated task-2 fitted parameters:")
    print(f"- task2_fitted_parameters.csv: {len(parameters)} rows")
    print(f"- task2_weight_scenarios.csv: {len(scenarios)} rows")
    print(f"- task2_cpi_regression_data.csv: {len(cpi_monthly)} rows")
    print()
    print(parameters.to_string(index=False))


if __name__ == "__main__":
    main()
