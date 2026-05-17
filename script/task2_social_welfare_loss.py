"""
Social welfare loss function for the simplified task-2 model.

The core callable is social_welfare_loss(...). Optimizers can pass candidate
alpha_gasoline and alpha_diesel arrays and receive total loss plus detailed
event/month loss tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result"
EPS = 1e-12
MIN_PRICE_CNY_PER_TON = 1000.0


def load_parameters(
    path: Path | str = RESULT_DIR / "task2_fitted_parameters.csv",
) -> dict[str, float]:
    params_df = pd.read_csv(path, encoding="utf-8-sig")
    return {
        str(row.parameter): float(row.value)
        for row in params_df.itertuples(index=False)
    }


def load_weight_scenarios(
    path: Path | str = RESULT_DIR / "task2_weight_scenarios.csv",
) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def get_weight_scenario(
    scenario: str = "equal_weight",
    path: Path | str = RESULT_DIR / "task2_weight_scenarios.csv",
) -> dict[str, float]:
    scenarios = load_weight_scenarios(path)
    row = scenarios.loc[scenarios["scenario"] == scenario]
    if row.empty:
        raise ValueError(f"Unknown weight scenario: {scenario}")
    row = row.iloc[0]
    return {
        "lambda_C": float(row["lambda_C"]),
        "lambda_F": float(row["lambda_F"]),
        "lambda_pi": float(row["lambda_pi"]),
        "lambda_V": float(row["lambda_V"]),
        "lambda_E": float(row["lambda_E"]),
        "lambda_T": float(row.get("lambda_T", 0.0)),
    }


def social_welfare_loss(
    event_data: pd.DataFrame,
    parameters: dict[str, float],
    preference_weights: dict[str, float],
    alpha_gasoline: Any,
    alpha_diesel: Any,
) -> dict[str, Any]:
    """
    Compute social welfare loss for candidate gasoline/diesel alpha paths.

    Parameters
    ----------
    event_data:
        Clean task-2 event table, normally result/task2_event_model_input_clean.csv.
    parameters:
        Fitted/model parameters from task2_fitted_parameters.csv.
    preference_weights:
        Dict with lambda_C, lambda_F, lambda_pi, lambda_V, lambda_E, lambda_T.
    alpha_gasoline, alpha_diesel:
        Scalars or arrays with length equal to len(event_data). Values are clipped
        to [0, 1] inside the function.

    Returns
    -------
    dict with:
        total_loss: scalar objective value
        event_loss: per-event dataframe
        monthly_loss: per-month CPI loss dataframe
        component_summary: one-row dataframe with weighted and unweighted totals
    """
    data = event_data.copy().sort_values("date").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"])
    data["month"] = pd.to_datetime(data["month"]).dt.to_period("M").dt.to_timestamp()

    n = len(data)
    alpha_g = _as_alpha_array(alpha_gasoline, n, "alpha_gasoline")
    alpha_d = _as_alpha_array(alpha_diesel, n, "alpha_diesel")

    _validate_weights(preference_weights)

    rho = parameters["rho"]
    rho_a = parameters["rho_A"]
    theta_1 = parameters["theta_1"]
    theta_2 = parameters["theta_2"]

    b_g = _positive(parameters["B_gasoline"], "B_gasoline")
    b_d = _positive(parameters["B_diesel"], "B_diesel")
    u_g_scale = _positive(parameters["U_gasoline"], "U_gasoline")
    u_d_scale = _positive(parameters["U_diesel"], "U_diesel")
    d_g_scale = _positive(parameters["D_gasoline"], "D_gasoline")
    d_d_scale = _positive(parameters["D_diesel"], "D_diesel")
    t_g_scale = _positive(parameters["T_gasoline"], "T_gasoline")
    t_d_scale = _positive(parameters["T_diesel"], "T_diesel")
    loss_scale_lc = _positive(parameters.get("loss_scale_LC", 1.0), "loss_scale_LC")
    loss_scale_lf = _positive(parameters.get("loss_scale_LF", 1.0), "loss_scale_LF")
    loss_scale_lv = _positive(parameters.get("loss_scale_LV", 1.0), "loss_scale_LV")
    loss_scale_lt = _positive(parameters.get("loss_scale_LT", 1.0), "loss_scale_LT")
    loss_scale_le = _positive(parameters.get("loss_scale_LE", 1.0), "loss_scale_LE")
    loss_cap_lc = _positive(parameters.get("loss_cap_LC", 1e12), "loss_cap_LC")
    loss_cap_lf = _positive(parameters.get("loss_cap_LF", 1e12), "loss_cap_LF")
    loss_cap_lv = _positive(parameters.get("loss_cap_LV", 1e12), "loss_cap_LV")
    loss_cap_lt = _positive(parameters.get("loss_cap_LT", 1e12), "loss_cap_LT")
    loss_cap_le = _positive(parameters.get("loss_cap_LE", 1e12), "loss_cap_LE")
    tau = parameters["tau"]

    p_g_prev = float(data.loc[0, "gasoline_price_before"])
    p_d_prev = float(data.loc[0, "diesel_price_before"])
    s_g = 0.0
    s_d = 0.0
    prev_u_g = 0.0
    prev_u_d = 0.0
    energy_a = 0.0

    rows = []
    for i, row in data.iterrows():
        f_g = float(row["f_gasoline"])
        f_d = float(row["f_diesel"])
        omega_g = float(row["omega_gasoline"])
        omega_d = float(row["omega_diesel"])

        u_g = alpha_g[i] * f_g
        u_d = alpha_d[i] * f_d
        p_g = p_g_prev + u_g
        p_d = p_d_prev + u_d
        price_floor_penalty = 0.0
        if p_g <= MIN_PRICE_CNY_PER_TON or p_d <= MIN_PRICE_CNY_PER_TON:
            price_floor_penalty = 1e12

        s_g = rho * s_g + f_g - u_g
        s_d = rho * s_d + f_d - u_d

        consumer_increase_loss = (
            omega_g * (max(u_g, 0.0) / max(p_g_prev, EPS)) ** 2
            + omega_d * (max(u_d, 0.0) / max(p_d_prev, EPS)) ** 2
        )
        consumer_decrease_reward = (
            omega_g * (max(-u_g, 0.0) / max(p_g_prev, EPS)) ** 2
            + omega_d * (max(-u_d, 0.0) / max(p_d_prev, EPS)) ** 2
        )
        lc_raw = consumer_increase_loss - consumer_decrease_reward
        lc_winsorized = float(np.clip(lc_raw, -loss_cap_lc, loss_cap_lc))
        lc = lc_winsorized / loss_scale_lc + price_floor_penalty

        lf_raw = (
            omega_g * (max(s_g, 0.0) / b_g) ** 2
            + omega_d * (max(s_d, 0.0) / b_d) ** 2
        )
        lf_winsorized = min(lf_raw, loss_cap_lf)
        lf = lf_winsorized / loss_scale_lf

        lv_raw = (
            omega_g
            * (
                0.5 * (u_g / u_g_scale) ** 2
                + 0.5 * ((u_g - prev_u_g) / d_g_scale) ** 2
            )
            + omega_d
            * (
                0.5 * (u_d / u_d_scale) ** 2
                + 0.5 * ((u_d - prev_u_d) / d_d_scale) ** 2
            )
        )
        lv_winsorized = min(lv_raw, loss_cap_lv)
        lv = lv_winsorized / loss_scale_lv

        lt_raw = (
            omega_g * ((u_g - f_g) / t_g_scale) ** 2
            + omega_d * ((u_d - f_d) / t_d_scale) ** 2
        )
        lt_winsorized = min(lt_raw, loss_cap_lt)
        lt = lt_winsorized / loss_scale_lt

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
            rho_a * energy_a
            + theta_1 * dependency_component
            + theta_2 * shortage_component,
        )
        le_raw = max(0.0, energy_a - tau) ** 2
        le_winsorized = min(le_raw, loss_cap_le)
        le = le_winsorized / loss_scale_le

        rows.append(
            {
                "date": row["date"],
                "month": row["month"],
                "alpha_gasoline": alpha_g[i],
                "alpha_diesel": alpha_d[i],
                "u_gasoline": u_g,
                "u_diesel": u_d,
                "price_gasoline": p_g,
                "price_diesel": p_d,
                "S_gasoline": s_g,
                "S_diesel": s_d,
                "LC": lc,
                "LC_raw": lc_raw,
                "LC_winsorized": lc_winsorized,
                "consumer_increase_loss": consumer_increase_loss,
                "consumer_decrease_reward": consumer_decrease_reward,
                "LF": lf,
                "LF_raw": lf_raw,
                "LF_winsorized": lf_winsorized,
                "LV": lv,
                "LV_raw": lv_raw,
                "LV_winsorized": lv_winsorized,
                "LT": lt,
                "LT_raw": lt_raw,
                "LT_winsorized": lt_winsorized,
                "price_gap_g": price_gap,
                "energy_A": energy_a,
                "LE": le,
                "LE_raw": le_raw,
                "LE_winsorized": le_winsorized,
                "cpi_yoy": row["cpi_yoy"],
                "cpi_target": row["cpi_target"],
                "gasoline_price_before": p_g_prev,
                "diesel_price_before": p_d_prev,
                "omega_gasoline": omega_g,
                "omega_diesel": omega_d,
            }
        )

        p_g_prev = p_g
        p_d_prev = p_d
        prev_u_g = u_g
        prev_u_d = u_d

    event_loss = pd.DataFrame(rows)
    monthly_loss = _compute_monthly_cpi_loss(event_loss, parameters)

    event_loss = event_loss.merge(
        monthly_loss[["month", "Lpi"]],
        on="month",
        how="left",
    )
    event_loss["Lpi_contribution"] = 0.0
    last_event_in_month = event_loss.groupby("month").tail(1).index
    event_loss.loc[last_event_in_month, "Lpi_contribution"] = event_loss.loc[
        last_event_in_month, "Lpi"
    ]

    lambda_c = preference_weights["lambda_C"]
    lambda_f = preference_weights["lambda_F"]
    lambda_pi = preference_weights["lambda_pi"]
    lambda_v = preference_weights["lambda_V"]
    lambda_e = preference_weights["lambda_E"]
    lambda_t = preference_weights.get("lambda_T", 0.0)

    event_loss["weighted_event_loss"] = (
        lambda_c * event_loss["LC"]
        + lambda_f * event_loss["LF"]
        + lambda_v * event_loss["LV"]
        + lambda_e * event_loss["LE"]
        + lambda_t * event_loss["LT"]
        + lambda_pi * event_loss["Lpi_contribution"]
    )

    component_summary = pd.DataFrame(
        [
            {
                "LC_sum": event_loss["LC"].sum(),
                "LF_sum": event_loss["LF"].sum(),
                "LV_sum": event_loss["LV"].sum(),
                "LT_sum": event_loss["LT"].sum(),
                "LE_sum": event_loss["LE"].sum(),
                "Lpi_sum": monthly_loss["Lpi"].sum(),
                "weighted_LC": lambda_c * event_loss["LC"].sum(),
                "weighted_LF": lambda_f * event_loss["LF"].sum(),
                "weighted_LV": lambda_v * event_loss["LV"].sum(),
                "weighted_LT": lambda_t * event_loss["LT"].sum(),
                "weighted_LE": lambda_e * event_loss["LE"].sum(),
                "weighted_Lpi": lambda_pi * monthly_loss["Lpi"].sum(),
                "total_loss": event_loss["weighted_event_loss"].sum(),
            }
        ]
    )

    return {
        "total_loss": float(component_summary.loc[0, "total_loss"]),
        "event_loss": event_loss,
        "monthly_loss": monthly_loss,
        "component_summary": component_summary,
    }


def _compute_monthly_cpi_loss(
    event_loss: pd.DataFrame,
    parameters: dict[str, float],
) -> pd.DataFrame:
    monthly = (
        event_loss.groupby("month", as_index=False)
        .agg(
            u_gasoline_sum=("u_gasoline", "sum"),
            u_diesel_sum=("u_diesel", "sum"),
            gasoline_price_base=("gasoline_price_before", "first"),
            diesel_price_base=("diesel_price_before", "first"),
            cpi_yoy=("cpi_yoy", "first"),
            cpi_target=("cpi_target", "first"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    monthly["x_gasoline"] = (
        monthly["u_gasoline_sum"] / monthly["gasoline_price_base"].clip(lower=EPS)
    ).clip(lower=0.0)
    monthly["x_diesel"] = (
        monthly["u_diesel_sum"] / monthly["diesel_price_base"].clip(lower=EPS)
    ).clip(lower=0.0)
    monthly["cpi_hat_next"] = (
        parameters["beta0"]
        + parameters["beta_gasoline"] * monthly["x_gasoline"]
        + parameters["beta_diesel"] * monthly["x_diesel"]
        + parameters["beta_pi"] * monthly["cpi_yoy"]
    )
    sigma_pi = _positive(parameters["sigma_pi"], "sigma_pi")
    lower_target = float(parameters.get("cpi_lower_target", 0.02))
    monthly["inflation_excess"] = (
        monthly["cpi_hat_next"] - monthly["cpi_target"]
    ).clip(lower=0.0)
    monthly["deflation_gap"] = (
        lower_target - monthly["cpi_hat_next"]
    ).clip(lower=0.0)
    monthly["Lpi_inflation"] = (monthly["inflation_excess"] / sigma_pi) ** 2
    monthly["Lpi_deflation"] = (monthly["deflation_gap"] / sigma_pi) ** 2
    monthly["Lpi_raw"] = monthly["Lpi_inflation"] + monthly["Lpi_deflation"]
    loss_scale_lpi = _positive(parameters.get("loss_scale_Lpi", 1.0), "loss_scale_Lpi")
    loss_cap_lpi = _positive(parameters.get("loss_cap_Lpi", 1e12), "loss_cap_Lpi")
    monthly["Lpi_winsorized"] = monthly["Lpi_raw"].clip(upper=loss_cap_lpi)
    monthly["Lpi"] = monthly["Lpi_winsorized"] / loss_scale_lpi
    return monthly


def _as_alpha_array(value: Any, n: int, name: str) -> np.ndarray:
    if np.isscalar(value):
        alpha = np.full(n, float(value), dtype=float)
    else:
        alpha = np.asarray(value, dtype=float)
        if alpha.shape != (n,):
            raise ValueError(f"{name} must be scalar or length {n}, got {alpha.shape}")
    return np.clip(alpha, 0.0, 1.0)


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate_weights(weights: dict[str, float]) -> None:
    required = ["lambda_C", "lambda_F", "lambda_pi", "lambda_V", "lambda_E"]
    missing = [key for key in required if key not in weights]
    if missing:
        raise ValueError(f"Missing preference weights: {missing}")
    total = sum(float(weights[key]) for key in required) + float(weights.get("lambda_T", 0.0))
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Preference weights must sum to 1, got {total}")
    if any(float(weights[key]) < 0 for key in required) or float(weights.get("lambda_T", 0.0)) < 0:
        raise ValueError("Preference weights must be non-negative")


def main() -> None:
    data = pd.read_csv(
        RESULT_DIR / "task2_event_model_input_clean.csv",
        encoding="utf-8-sig",
    )
    params = load_parameters()
    scenarios = load_weight_scenarios()

    rows = []
    for scenario in scenarios["scenario"]:
        weights = get_weight_scenario(scenario)
        result = social_welfare_loss(
            data,
            params,
            weights,
            alpha_gasoline=1.0,
            alpha_diesel=1.0,
        )
        summary = result["component_summary"].iloc[0].to_dict()
        summary["scenario"] = scenario
        summary["alpha_policy"] = "full_release_alpha_1"
        rows.append(summary)

    out = pd.DataFrame(rows)
    first_cols = ["scenario", "alpha_policy", "total_loss"]
    out = out[first_cols + [col for col in out.columns if col not in first_cols]]
    out.to_csv(
        RESULT_DIR / "task2_loss_function_smoke_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("Generated task2_loss_function_smoke_test.csv")
    print(out[["scenario", "total_loss"]].to_string(index=False))


if __name__ == "__main__":
    main()
