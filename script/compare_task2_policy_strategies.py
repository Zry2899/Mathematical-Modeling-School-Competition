"""
Compare the welfare-optimal refined-oil pricing path with the current mechanism.

The comparison uses the same six-loss model as task 2, but evaluates the
current mechanism with its observed adjustment amounts directly. This matters
because the historical rule is not always representable as an alpha in [0, 1].

Outputs:
    result/task2_policy_comparison_event_losses.csv
    result/task2_policy_comparison_monthly_cpi.csv
    result/task2_policy_comparison_key_metrics.csv
    result/task2_policy_comparison_dimension_summary.csv
    result/task2_policy_comparison_loss_components.png
    result/task2_policy_comparison_price_path.png
    result/task2_policy_comparison_cumulative_loss.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from task2_social_welfare_loss import (
    EPS,
    MIN_PRICE_CNY_PER_TON,
    _compute_monthly_cpi_loss,
    _positive,
    _validate_weights,
    get_weight_scenario,
    load_parameters,
)


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result"


MECHANISM_LABELS = {
    "current": "Current mechanism",
    "optimal": "Welfare-optimal model",
    "before_adjustment": "Before policy adjustment",
    "after_adjustment": "After policy adjustment",
}


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write a CSV, falling back when the target is open in another program."""
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated{path.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        return fallback


def evaluate_adjustment_path(
    event_data: pd.DataFrame,
    parameters: dict[str, float],
    preference_weights: dict[str, float],
    u_gasoline: Any,
    u_diesel: Any,
    mechanism: str,
) -> dict[str, pd.DataFrame | float]:
    """Evaluate a direct gasoline/diesel adjustment path under the task-2 loss."""
    data = event_data.copy().sort_values("date").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"])
    data["month"] = pd.to_datetime(data["month"]).dt.to_period("M").dt.to_timestamp()

    n = len(data)
    u_g_path = _as_path_array(u_gasoline, n, "u_gasoline")
    u_d_path = _as_path_array(u_diesel, n, "u_diesel")

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

        u_g = float(u_g_path[i])
        u_d = float(u_d_path[i])
        p_g = p_g_prev + u_g
        p_d = p_d_prev + u_d
        price_floor_penalty = 1e12 if (p_g <= MIN_PRICE_CNY_PER_TON or p_d <= MIN_PRICE_CNY_PER_TON) else 0.0

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
        lc = float(np.clip(lc_raw, -loss_cap_lc, loss_cap_lc)) / loss_scale_lc + price_floor_penalty

        lf_raw = (
            omega_g * (max(s_g, 0.0) / b_g) ** 2
            + omega_d * (max(s_d, 0.0) / b_d) ** 2
        )
        lf = min(lf_raw, loss_cap_lf) / loss_scale_lf

        lv_raw = (
            omega_g
            * (0.5 * (u_g / u_g_scale) ** 2 + 0.5 * ((u_g - prev_u_g) / d_g_scale) ** 2)
            + omega_d
            * (0.5 * (u_d / u_d_scale) ** 2 + 0.5 * ((u_d - prev_u_d) / d_d_scale) ** 2)
        )
        lv = min(lv_raw, loss_cap_lv) / loss_scale_lv

        lt_raw = (
            omega_g * ((u_g - f_g) / t_g_scale) ** 2
            + omega_d * ((u_d - f_d) / t_d_scale) ** 2
        )
        lt = min(lt_raw, loss_cap_lt) / loss_scale_lt

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
        le = min(le_raw, loss_cap_le) / loss_scale_le

        rows.append(
            {
                "mechanism": mechanism,
                "mechanism_label": MECHANISM_LABELS.get(mechanism, mechanism),
                "date": row["date"],
                "month": row["month"],
                "u_gasoline": u_g,
                "u_diesel": u_d,
                "f_gasoline": f_g,
                "f_diesel": f_d,
                "price_gasoline": p_g,
                "price_diesel": p_d,
                "S_gasoline": s_g,
                "S_diesel": s_d,
                "LC": lc,
                "LC_raw": lc_raw,
                "consumer_increase_loss": consumer_increase_loss,
                "consumer_decrease_reward": consumer_decrease_reward,
                "LF": lf,
                "LF_raw": lf_raw,
                "LV": lv,
                "LV_raw": lv_raw,
                "LT": lt,
                "LT_raw": lt_raw,
                "price_gap_g": price_gap,
                "energy_A": energy_a,
                "LE": le,
                "LE_raw": le_raw,
                "cpi_yoy": row["cpi_yoy"],
                "cpi_target": row["cpi_target"],
                "gasoline_price_before": p_g_prev,
                "diesel_price_before": p_d_prev,
                "omega_gasoline": omega_g,
                "omega_diesel": omega_d,
                "weighted_adjustment": omega_g * u_g + omega_d * u_d,
                "weighted_abs_adjustment": omega_g * abs(u_g) + omega_d * abs(u_d),
                "weighted_abs_transmission_gap": omega_g * abs(u_g - f_g) + omega_d * abs(u_d - f_d),
            }
        )

        p_g_prev = p_g
        p_d_prev = p_d
        prev_u_g = u_g
        prev_u_d = u_d

    event_loss = pd.DataFrame(rows)
    monthly_loss = _compute_monthly_cpi_loss(event_loss, parameters)
    monthly_loss["mechanism"] = mechanism
    monthly_loss["mechanism_label"] = MECHANISM_LABELS.get(mechanism, mechanism)

    event_loss = event_loss.merge(monthly_loss[["month", "Lpi"]], on="month", how="left")
    event_loss["Lpi_contribution"] = 0.0
    last_event_in_month = event_loss.groupby("month").tail(1).index
    event_loss.loc[last_event_in_month, "Lpi_contribution"] = event_loss.loc[last_event_in_month, "Lpi"]

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
                "mechanism": mechanism,
                "mechanism_label": MECHANISM_LABELS.get(mechanism, mechanism),
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


def _as_path_array(value: Any, n: int, name: str) -> np.ndarray:
    if np.isscalar(value):
        out = np.full(n, float(value), dtype=float)
    else:
        out = np.asarray(value, dtype=float)
        if out.shape != (n,):
            raise ValueError(f"{name} must be scalar or length {n}, got {out.shape}")
    return out


def build_key_metrics(event_all: pd.DataFrame, month_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mechanism, event in event_all.groupby("mechanism", sort=False):
        month = month_all.loc[month_all["mechanism"] == mechanism].copy()
        weighted_terminal_s = (
            event["omega_gasoline"].iloc[-1] * max(event["S_gasoline"].iloc[-1], 0)
            + event["omega_diesel"].iloc[-1] * max(event["S_diesel"].iloc[-1], 0)
        )
        metric_values = {
            "total_loss": event["weighted_event_loss"].sum(),
            "LC_sum": event["LC"].sum(),
            "LF_sum": event["LF"].sum(),
            "LV_sum": event["LV"].sum(),
            "LT_sum": event["LT"].sum(),
            "LE_sum": event["LE"].sum(),
            "Lpi_sum": month["Lpi"].sum(),
            "mean_abs_adjustment": event["weighted_abs_adjustment"].mean(),
            "std_adjustment": event["weighted_adjustment"].std(ddof=1),
            "max_abs_adjustment": event["weighted_abs_adjustment"].max(),
            "total_upward_adjustment": (
                event["omega_gasoline"] * event["u_gasoline"].clip(lower=0)
                + event["omega_diesel"] * event["u_diesel"].clip(lower=0)
            ).sum(),
            "total_downward_adjustment": (
                event["omega_gasoline"] * (-event["u_gasoline"]).clip(lower=0)
                + event["omega_diesel"] * (-event["u_diesel"]).clip(lower=0)
            ).sum(),
            "consumer_increase_loss_sum": event["consumer_increase_loss"].sum(),
            "consumer_decrease_reward_sum": event["consumer_decrease_reward"].sum(),
            "mean_positive_S": (
                event["omega_gasoline"] * event["S_gasoline"].clip(lower=0)
                + event["omega_diesel"] * event["S_diesel"].clip(lower=0)
            ).mean(),
            "terminal_positive_S": weighted_terminal_s,
            "mean_price_gap": event["price_gap_g"].mean(),
            "mean_abs_transmission_gap": event["weighted_abs_transmission_gap"].mean(),
            "mean_abs_cpi_band_gap": (month["inflation_excess"] + month["deflation_gap"]).mean(),
            "months_outside_cpi_band": int(((month["inflation_excess"] + month["deflation_gap"]) > 0).sum()),
        }
        for metric, value in metric_values.items():
            rows.append(
                {
                "mechanism": mechanism,
                "mechanism_label": MECHANISM_LABELS.get(mechanism, mechanism),
                    "metric": metric,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def build_dimension_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    wide = metrics.pivot(index="metric", columns="mechanism", values="value")

    def pct_change(metric: str) -> float:
        current = float(wide.loc[metric, "current"])
        optimal = float(wide.loc[metric, "optimal"])
        if abs(current) < EPS:
            return np.nan
        return (optimal - current) / abs(current)

    rows = [
        {
            "dimension": "价格稳定性",
            "main_metric": "LV_sum",
            "current_value": wide.loc["LV_sum", "current"],
            "optimal_value": wide.loc["LV_sum", "optimal"],
            "change_pct": pct_change("LV_sum"),
            "auxiliary_metrics": (
                f"mean_abs_adjustment: {wide.loc['mean_abs_adjustment', 'current']:.2f} -> "
                f"{wide.loc['mean_abs_adjustment', 'optimal']:.2f}; "
                f"std_adjustment: {wide.loc['std_adjustment', 'current']:.2f} -> "
                f"{wide.loc['std_adjustment', 'optimal']:.2f}"
            ),
            "interpretation_cn": "优化模型的平均调价幅度和离散度略高，但综合平滑损失LV小幅下降，说明其调价节奏更贴近损失函数设定的平稳路径。",
        },
        {
            "dimension": "消费者福利",
            "main_metric": "LC_sum",
            "current_value": wide.loc["LC_sum", "current"],
            "optimal_value": wide.loc["LC_sum", "optimal"],
            "change_pct": pct_change("LC_sum"),
            "auxiliary_metrics": (
                f"upward_adjustment: {wide.loc['total_upward_adjustment', 'current']:.2f} -> "
                f"{wide.loc['total_upward_adjustment', 'optimal']:.2f}; "
                f"downward_adjustment: {wide.loc['total_downward_adjustment', 'current']:.2f} -> "
                f"{wide.loc['total_downward_adjustment', 'optimal']:.2f}"
            ),
            "interpretation_cn": "优化模型在下调阶段释放更多降价收益，同时减少上调传导总量，消费者净福利改善。",
        },
        {
            "dimension": "企业与财政压力",
            "main_metric": "LF_sum",
            "current_value": wide.loc["LF_sum", "current"],
            "optimal_value": wide.loc["LF_sum", "optimal"],
            "change_pct": pct_change("LF_sum"),
            "auxiliary_metrics": (
                f"mean_positive_S: {wide.loc['mean_positive_S', 'current']:.2f} -> "
                f"{wide.loc['mean_positive_S', 'optimal']:.2f}; "
                f"terminal_positive_S: {wide.loc['terminal_positive_S', 'current']:.2f} -> "
                f"{wide.loc['terminal_positive_S', 'optimal']:.2f}"
            ),
            "interpretation_cn": "优化模型显著降低未传导上涨压力，缓解企业或财政补偿压力。",
        },
        {
            "dimension": "宏观稳定性",
            "main_metric": "Lpi_sum",
            "current_value": wide.loc["Lpi_sum", "current"],
            "optimal_value": wide.loc["Lpi_sum", "optimal"],
            "change_pct": pct_change("Lpi_sum"),
            "auxiliary_metrics": (
                f"mean_abs_cpi_band_gap: {wide.loc['mean_abs_cpi_band_gap', 'current']:.6f} -> "
                f"{wide.loc['mean_abs_cpi_band_gap', 'optimal']:.6f}; "
                f"months_outside_cpi_band: {wide.loc['months_outside_cpi_band', 'current']:.0f} -> "
                f"{wide.loc['months_outside_cpi_band', 'optimal']:.0f}"
            ),
            "interpretation_cn": "优化模型的CPI稳定损失小幅下降，但改善幅度有限，说明宏观稳定不是本轮优化收益的主要来源。",
        },
        {
            "dimension": "能源安全与市场传导",
            "main_metric": "LE_sum + LT_sum",
            "current_value": wide.loc["LE_sum", "current"] + wide.loc["LT_sum", "current"],
            "optimal_value": wide.loc["LE_sum", "optimal"] + wide.loc["LT_sum", "optimal"],
            "change_pct": (
                (wide.loc["LE_sum", "optimal"] + wide.loc["LT_sum", "optimal"])
                - (wide.loc["LE_sum", "current"] + wide.loc["LT_sum", "current"])
            )
            / abs(wide.loc["LE_sum", "current"] + wide.loc["LT_sum", "current"]),
            "auxiliary_metrics": (
                f"LE_sum: {wide.loc['LE_sum', 'current']:.2f} -> {wide.loc['LE_sum', 'optimal']:.2f}; "
                f"LT_sum: {wide.loc['LT_sum', 'current']:.2f} -> {wide.loc['LT_sum', 'optimal']:.2f}; "
                f"mean_transmission_gap: {wide.loc['mean_abs_transmission_gap', 'current']:.2f} -> "
                f"{wide.loc['mean_abs_transmission_gap', 'optimal']:.2f}"
            ),
            "interpretation_cn": "优化模型明显减少理论价格与实际价格的传导偏离，能源安全损失也略有下降，主要改进来自市场传导效率。",
        },
    ]
    return pd.DataFrame(rows)


def make_plots(event_all: pd.DataFrame) -> None:
    plot_data = event_all.copy()
    plot_data["date"] = pd.to_datetime(plot_data["date"])

    component = (
        plot_data.groupby("mechanism_label")[["LC", "LF", "LV", "LT", "LE"]]
        .sum()
        .reset_index()
        .melt(id_vars="mechanism_label", var_name="loss_component", value_name="sum")
    )
    lpi = (
        plot_data.groupby(["mechanism_label", "month"], as_index=False)
        .tail(1)
        .groupby("mechanism_label")["Lpi"]
        .sum()
        .reset_index()
    )
    lpi["loss_component"] = "Lpi"
    lpi = lpi.rename(columns={"Lpi": "sum"})
    component = pd.concat([component, lpi], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = ["LC", "LF", "LV", "LT", "LE", "Lpi"]
    x = np.arange(len(labels))
    width = 0.36
    current = component.loc[component["mechanism_label"] == MECHANISM_LABELS["current"]].set_index("loss_component")
    optimal = component.loc[component["mechanism_label"] == MECHANISM_LABELS["optimal"]].set_index("loss_component")
    ax.bar(x - width / 2, [current.loc[k, "sum"] for k in labels], width, label=MECHANISM_LABELS["current"])
    ax.bar(x + width / 2, [optimal.loc[k, "sum"] for k in labels], width, label=MECHANISM_LABELS["optimal"])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Loss component sum")
    ax.set_title("Task 2 policy comparison: loss components")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "task2_policy_comparison_loss_components.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for mechanism, sub in plot_data.groupby("mechanism"):
        label = MECHANISM_LABELS[mechanism]
        axes[0].plot(sub["date"], sub["price_gasoline"], label=label, linewidth=1.5)
        axes[1].plot(sub["date"], sub["price_diesel"], label=label, linewidth=1.5)
    axes[0].set_title("Gasoline price path")
    axes[1].set_title("Diesel price path")
    for ax in axes:
        ax.set_ylabel("CNY / ton")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "task2_policy_comparison_price_path.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for mechanism, sub in plot_data.groupby("mechanism"):
        label = MECHANISM_LABELS[mechanism]
        ax.plot(sub["date"], sub["weighted_event_loss"].cumsum(), label=label, linewidth=1.7)
    ax.set_title("Cumulative weighted welfare loss")
    ax.set_ylabel("Cumulative loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "task2_policy_comparison_cumulative_loss.png", dpi=180)
    plt.close(fig)


def main() -> None:
    events = pd.read_csv(RESULT_DIR / "task2_event_model_input_clean.csv", encoding="utf-8-sig")
    optimal = pd.read_csv(RESULT_DIR / "task2_optimization_equal_weight_final_step001_updated.csv", encoding="utf-8-sig")
    params = load_parameters()
    weights = get_weight_scenario("equal_weight")

    current_eval = evaluate_adjustment_path(
        events,
        params,
        weights,
        events["u_hist_gasoline"].to_numpy(),
        events["u_hist_diesel"].to_numpy(),
        "current",
    )
    optimal_eval = evaluate_adjustment_path(
        events,
        params,
        weights,
        optimal["optimal_u_gasoline"].to_numpy(),
        optimal["optimal_u_diesel"].to_numpy(),
        "optimal",
    )

    event_all = pd.concat(
        [current_eval["event_loss"], optimal_eval["event_loss"]],
        ignore_index=True,
    )
    month_all = pd.concat(
        [current_eval["monthly_loss"], optimal_eval["monthly_loss"]],
        ignore_index=True,
    )
    component_all = pd.concat(
        [current_eval["component_summary"], optimal_eval["component_summary"]],
        ignore_index=True,
    )
    key_metrics = build_key_metrics(event_all, month_all)
    dimension_summary = build_dimension_summary(key_metrics)

    output_paths = [
        safe_to_csv(event_all, RESULT_DIR / "task2_policy_comparison_event_losses.csv"),
        safe_to_csv(month_all, RESULT_DIR / "task2_policy_comparison_monthly_cpi.csv"),
        safe_to_csv(component_all, RESULT_DIR / "task2_policy_comparison_component_summary.csv"),
        safe_to_csv(key_metrics, RESULT_DIR / "task2_policy_comparison_key_metrics.csv"),
        safe_to_csv(dimension_summary, RESULT_DIR / "task2_policy_comparison_dimension_summary.csv"),
    ]
    make_plots(event_all)

    print("Generated task-2 policy comparison outputs:")
    for path in output_paths:
        print(f"- {path.name}")
    print("- task2_policy_comparison_loss_components.png")
    print("- task2_policy_comparison_price_path.png")
    print("- task2_policy_comparison_cumulative_loss.png")
    print()
    print(dimension_summary.to_string(index=False))


if __name__ == "__main__":
    main()
