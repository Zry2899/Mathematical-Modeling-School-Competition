"""
Compare the task-2 model before and after the policy-control coefficient.

Definition:
    before adjustment: u_i,t = f_i,t
        Uses the predicted theoretical adjustment from the first-question model.

    after adjustment: u_i,t = alpha_hat_i,t * f_i,t
        Uses the first-question predicted adjustment multiplied by the
        second-question fitted dynamic alpha function.

The two paths are evaluated under the same task-2 welfare-loss model. The final
comparison reports the average loss for the five paper dimensions:
    1. Consumer welfare: LC
    2. Firm/fiscal pressure: LF
    3. Price stability: LV
    4. Macro/CPI stability: Lpi
    5. Energy security and market transmission: LE + LT
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_task2_policy_strategies import evaluate_adjustment_path
from diesel_alpha_combined_adjustment_function import diesel_alpha_adjustment
from gasoline_alpha_combined_adjustment_function import gasoline_alpha_adjustment
from task2_social_welfare_loss import get_weight_scenario, load_parameters


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result"


def build_alpha_adjusted_path(
    events: pd.DataFrame,
    params: dict[str, float],
) -> pd.DataFrame:
    """Generate recursive alpha and adjusted u paths from fitted alpha functions."""
    data = events.copy().sort_values("date").reset_index(drop=True)
    rho = float(params["rho"])
    s_g = 0.0
    s_d = 0.0
    prev_u_g = 0.0
    prev_u_d = 0.0

    rows = []
    for row in data.itertuples(index=False):
        f_g = float(row.f_gasoline)
        f_d = float(row.f_diesel)
        cpi = float(row.cpi_yoy)

        alpha_g = gasoline_alpha_adjustment(
            f_gasoline=f_g,
            f_diesel=f_d,
            S_gasoline_lag=s_g,
            prev_u_gasoline=prev_u_g,
            cpi_yoy=cpi,
        )
        alpha_d = diesel_alpha_adjustment(
            f_gasoline=f_g,
            f_diesel=f_d,
            S_diesel_lag=s_d,
            prev_u_diesel=prev_u_d,
            cpi_yoy=cpi,
        )

        if not np.isfinite(alpha_g):
            alpha_g = 0.0
        if not np.isfinite(alpha_d):
            alpha_d = 0.0

        u_g = alpha_g * f_g
        u_d = alpha_d * f_d
        rows.append(
            {
                "date": row.date,
                "f_gasoline": f_g,
                "f_diesel": f_d,
                "alpha_hat_gasoline": alpha_g,
                "alpha_hat_diesel": alpha_d,
                "adjusted_u_gasoline": u_g,
                "adjusted_u_diesel": u_d,
                "S_gasoline_lag": s_g,
                "S_diesel_lag": s_d,
                "prev_u_gasoline": prev_u_g,
                "prev_u_diesel": prev_u_d,
            }
        )

        s_g = rho * s_g + f_g - u_g
        s_d = rho * s_d + f_d - u_d
        prev_u_g = u_g
        prev_u_d = u_d

    return pd.DataFrame(rows)


def build_five_dimension_average(
    before_event: pd.DataFrame,
    before_month: pd.DataFrame,
    after_event: pd.DataFrame,
    after_month: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    specs = [
        ("消费者福利", "LC", "event", "LC"),
        ("企业与财政压力", "LF", "event", "LF"),
        ("价格稳定性", "LV", "event", "LV"),
        ("宏观稳定性", "Lpi", "month", "Lpi"),
        ("能源安全与市场传导", "LE + LT", "event", "LE_plus_LT"),
    ]

    before_event = before_event.copy()
    after_event = after_event.copy()
    before_event["LE_plus_LT"] = before_event["LE"] + before_event["LT"]
    after_event["LE_plus_LT"] = after_event["LE"] + after_event["LT"]

    for dimension, loss_component, scope, column in specs:
        if scope == "event":
            before_values = before_event[column]
            after_values = after_event[column]
            n = len(before_values)
        else:
            before_values = before_month[column]
            after_values = after_month[column]
            n = len(before_values)

        before_mean = float(before_values.mean())
        after_mean = float(after_values.mean())
        improvement = before_mean - after_mean
        improvement_pct = improvement / abs(before_mean) if abs(before_mean) > 1e-12 else np.nan
        rows.append(
            {
                "dimension": dimension,
                "loss_component": loss_component,
                "averaging_scope": scope,
                "n": n,
                "before_mean": before_mean,
                "after_mean": after_mean,
                "mean_change_after_minus_before": after_mean - before_mean,
                "improvement_before_minus_after": improvement,
                "improvement_pct": improvement_pct,
            }
        )
    return pd.DataFrame(rows)


def build_event_detail(before_event: pd.DataFrame, after_event: pd.DataFrame, alpha_path: pd.DataFrame) -> pd.DataFrame:
    before_event = before_event.copy()
    after_event = after_event.copy()
    alpha_path = alpha_path.copy()
    before_event["date"] = pd.to_datetime(before_event["date"])
    after_event["date"] = pd.to_datetime(after_event["date"])
    alpha_path["date"] = pd.to_datetime(alpha_path["date"])

    before_keep = before_event[
        [
            "date",
            "u_gasoline",
            "u_diesel",
            "price_gasoline",
            "price_diesel",
            "LC",
            "LF",
            "LV",
            "LT",
            "LE",
            "Lpi_contribution",
            "weighted_event_loss",
        ]
    ].rename(
        columns={
            "u_gasoline": "before_u_gasoline",
            "u_diesel": "before_u_diesel",
            "price_gasoline": "before_price_gasoline",
            "price_diesel": "before_price_diesel",
            "LC": "before_LC",
            "LF": "before_LF",
            "LV": "before_LV",
            "LT": "before_LT",
            "LE": "before_LE",
            "Lpi_contribution": "before_Lpi_contribution",
            "weighted_event_loss": "before_weighted_event_loss",
        }
    )
    after_keep = after_event[
        [
            "date",
            "u_gasoline",
            "u_diesel",
            "price_gasoline",
            "price_diesel",
            "LC",
            "LF",
            "LV",
            "LT",
            "LE",
            "Lpi_contribution",
            "weighted_event_loss",
        ]
    ].rename(
        columns={
            "u_gasoline": "after_u_gasoline",
            "u_diesel": "after_u_diesel",
            "price_gasoline": "after_price_gasoline",
            "price_diesel": "after_price_diesel",
            "LC": "after_LC",
            "LF": "after_LF",
            "LV": "after_LV",
            "LT": "after_LT",
            "LE": "after_LE",
            "Lpi_contribution": "after_Lpi_contribution",
            "weighted_event_loss": "after_weighted_event_loss",
        }
    )
    detail = before_keep.merge(after_keep, on="date", how="inner")
    alpha_keep = alpha_path[
        [
            "date",
            "f_gasoline",
            "f_diesel",
            "alpha_hat_gasoline",
            "alpha_hat_diesel",
        ]
    ]
    return detail.merge(alpha_keep, on="date", how="left")


def make_plots(summary: pd.DataFrame, event_detail: pd.DataFrame) -> None:
    plot_summary = summary.copy()
    english_dimensions = {
        "消费者福利": "Consumer welfare",
        "企业与财政压力": "Firm/fiscal pressure",
        "价格稳定性": "Price stability",
        "宏观稳定性": "Macro stability",
        "能源安全与市场传导": "Energy and transmission",
    }
    plot_summary["dimension_en"] = plot_summary["dimension"].map(english_dimensions)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(plot_summary))
    width = 0.36
    ax.bar(x - width / 2, plot_summary["before_mean"], width, label="Before adjustment: u=f")
    ax.bar(x + width / 2, plot_summary["after_mean"], width, label="After adjustment: u=alpha*f")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, plot_summary["dimension_en"], rotation=18, ha="right")
    ax.set_ylabel("Average loss")
    ax.set_title("Average loss by comparison dimension")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "task2_before_after_adjustment_dimension_means.png", dpi=180)
    plt.close(fig)

    detail = event_detail.copy()
    detail["date"] = pd.to_datetime(detail["date"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(detail["date"], detail["before_price_gasoline"], label="Before adjustment", linewidth=1.5)
    axes[0].plot(detail["date"], detail["after_price_gasoline"], label="After adjustment", linewidth=1.5)
    axes[1].plot(detail["date"], detail["before_price_diesel"], label="Before adjustment", linewidth=1.5)
    axes[1].plot(detail["date"], detail["after_price_diesel"], label="After adjustment", linewidth=1.5)
    axes[0].set_title("Gasoline price path")
    axes[1].set_title("Diesel price path")
    for ax in axes:
        ax.set_ylabel("CNY / ton")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "task2_before_after_adjustment_price_path.png", dpi=180)
    plt.close(fig)


def main() -> None:
    events = pd.read_csv(RESULT_DIR / "task2_event_model_input_clean.csv", encoding="utf-8-sig")
    params = load_parameters()
    weights = get_weight_scenario("equal_weight")

    alpha_path = build_alpha_adjusted_path(events, params)
    before_eval = evaluate_adjustment_path(
        events,
        params,
        weights,
        events["f_gasoline"].to_numpy(),
        events["f_diesel"].to_numpy(),
        "before_adjustment",
    )
    after_eval = evaluate_adjustment_path(
        events,
        params,
        weights,
        alpha_path["adjusted_u_gasoline"].to_numpy(),
        alpha_path["adjusted_u_diesel"].to_numpy(),
        "after_adjustment",
    )

    before_event = before_eval["event_loss"]
    after_event = after_eval["event_loss"]
    before_month = before_eval["monthly_loss"]
    after_month = after_eval["monthly_loss"]

    summary = build_five_dimension_average(before_event, before_month, after_event, after_month)
    event_detail = build_event_detail(before_event, after_event, alpha_path)
    total_summary = pd.DataFrame(
        [
            {
                "path": "before_adjustment",
                "description": "u = f, first-question predicted theoretical adjustment",
                "total_weighted_loss": before_event["weighted_event_loss"].sum(),
                "mean_weighted_event_loss": before_event["weighted_event_loss"].mean(),
            },
            {
                "path": "after_adjustment",
                "description": "u = alpha_hat * f, first-question prediction adjusted by task-2 alpha function",
                "total_weighted_loss": after_event["weighted_event_loss"].sum(),
                "mean_weighted_event_loss": after_event["weighted_event_loss"].mean(),
            },
        ]
    )

    alpha_path.to_csv(
        RESULT_DIR / "task2_before_after_adjustment_alpha_path.csv",
        index=False,
        encoding="utf-8-sig",
    )
    event_detail.to_csv(
        RESULT_DIR / "task2_before_after_adjustment_event_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        RESULT_DIR / "task2_before_after_adjustment_dimension_mean_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    total_summary.to_csv(
        RESULT_DIR / "task2_before_after_adjustment_total_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    make_plots(summary, event_detail)

    print("Generated before/after adjustment comparison outputs:")
    print("- task2_before_after_adjustment_alpha_path.csv")
    print("- task2_before_after_adjustment_event_detail.csv")
    print("- task2_before_after_adjustment_dimension_mean_summary.csv")
    print("- task2_before_after_adjustment_total_summary.csv")
    print("- task2_before_after_adjustment_dimension_means.png")
    print("- task2_before_after_adjustment_price_path.png")
    print()
    print(summary.to_string(index=False))
    print()
    print(total_summary.to_string(index=False))


if __name__ == "__main__":
    main()
