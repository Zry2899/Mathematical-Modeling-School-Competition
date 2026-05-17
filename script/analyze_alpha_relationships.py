"""
Explore empirical relationships between optimized alpha and candidate drivers.

The analysis uses the six-loss equal-weight optimized alpha as the baseline
policy-control coefficient. Rows with f_i = 0 are excluded because alpha_i has
no economic meaning when the theoretical adjustment is zero.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result"

BASELINE_ALPHA_FILE = RESULT_DIR / "task2_optimization_equal_weight_final_step001_updated.csv"
EVENT_FILE = RESULT_DIR / "task2_event_model_input_clean.csv"

VARIABLES = [
    "f",
    "abs_f",
    "is_price_up",
    "S_lag",
    "prev_u",
    "cpi_yoy",
    "oil_import_dependency",
    "brent_pressure_h",
    "processing_shortage_b",
    "time_index",
]

VARIABLE_LABELS = {
    "f": "theoretical adjustment f",
    "abs_f": "|f|",
    "is_price_up": "price-up indicator",
    "S_lag": "lagged accumulated pressure",
    "prev_u": "previous actual adjustment",
    "cpi_yoy": "CPI YoY",
    "oil_import_dependency": "oil import dependency",
    "brent_pressure_h": "Brent pressure",
    "processing_shortage_b": "processing shortage",
    "time_index": "time index",
}

FUEL_CONFIG = {
    "gasoline": {
        "alpha": "alpha_gasoline",
        "f": "f_gasoline",
        "S": "S_gasoline",
        "u": "optimal_u_gasoline",
        "title": "Gasoline",
    },
    "diesel": {
        "alpha": "alpha_diesel",
        "f": "f_diesel",
        "S": "S_diesel",
        "u": "optimal_u_diesel",
        "title": "Diesel",
    },
}


def load_analysis_frame() -> pd.DataFrame:
    alpha = pd.read_csv(BASELINE_ALPHA_FILE, encoding="utf-8-sig")
    events = pd.read_csv(EVENT_FILE, encoding="utf-8-sig")

    keep_cols = [
        "date",
        "cpi_yoy",
        "oil_import_dependency",
        "brent_pressure_h",
        "processing_shortage_b",
    ]
    data = alpha.merge(events[keep_cols], on="date", how="left")
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)
    data["time_index"] = np.arange(len(data), dtype=float)

    for fuel, cfg in FUEL_CONFIG.items():
        data[f"{fuel}_f"] = data[cfg["f"]]
        data[f"{fuel}_abs_f"] = data[cfg["f"]].abs()
        data[f"{fuel}_is_price_up"] = (data[cfg["f"]] > 0).astype(int)
        data[f"{fuel}_S_lag"] = data[cfg["S"]].shift(1).fillna(0.0)
        data[f"{fuel}_prev_u"] = data[cfg["u"]].shift(1).fillna(0.0)
    return data


def build_fuel_frame(data: pd.DataFrame, fuel: str) -> pd.DataFrame:
    cfg = FUEL_CONFIG[fuel]
    frame = pd.DataFrame(
        {
            "date": data["date"],
            "alpha": data[cfg["alpha"]],
            "f": data[f"{fuel}_f"],
            "abs_f": data[f"{fuel}_abs_f"],
            "is_price_up": data[f"{fuel}_is_price_up"],
            "S_lag": data[f"{fuel}_S_lag"],
            "prev_u": data[f"{fuel}_prev_u"],
            "cpi_yoy": data["cpi_yoy"],
            "oil_import_dependency": data["oil_import_dependency"],
            "brent_pressure_h": data["brent_pressure_h"],
            "processing_shortage_b": data["processing_shortage_b"],
            "time_index": data["time_index"],
        }
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.loc[frame["f"].abs() > 1e-12].copy()
    return frame


def calculate_correlations(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for fuel, frame in frames.items():
        for var in VARIABLES:
            pair = frame[["alpha", var]].dropna()
            rows.append(
                {
                    "fuel": fuel,
                    "variable": var,
                    "n": len(pair),
                    "pearson": pair["alpha"].corr(pair[var], method="pearson"),
                    "spearman": pair["alpha"].corr(pair[var], method="spearman"),
                    "kendall": pair["alpha"].corr(pair[var], method="kendall"),
                }
            )
    result = pd.DataFrame(rows)
    result["max_abs_corr"] = result[["pearson", "spearman", "kendall"]].abs().max(axis=1)
    return result.sort_values(["fuel", "max_abs_corr"], ascending=[True, False])


def plot_correlation_heatmap(corr: pd.DataFrame, fuel: str) -> Path:
    subset = corr.loc[corr["fuel"] == fuel].set_index("variable")
    plot_data = subset.loc[VARIABLES, ["pearson", "spearman", "kendall"]]

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    sns.heatmap(
        plot_data,
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "correlation with alpha"},
        ax=ax,
    )
    ax.set_title(f"{FUEL_CONFIG[fuel]['title']} alpha correlation diagnostics")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels([VARIABLE_LABELS[v] for v in VARIABLES], rotation=0)
    fig.tight_layout()
    path = RESULT_DIR / f"task2_alpha_corr_heatmap_{fuel}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _binned_means(frame: pd.DataFrame, var: str, bins: int = 10) -> pd.DataFrame:
    data = frame[["alpha", var]].dropna().copy()
    if data[var].nunique() <= 2:
        grouped = data.groupby(var, as_index=False).agg(alpha_mean=("alpha", "mean"), n=("alpha", "size"))
        grouped["x"] = grouped[var]
        return grouped

    try:
        data["bin"] = pd.qcut(data[var], q=min(bins, data[var].nunique()), duplicates="drop")
    except ValueError:
        data["bin"] = pd.cut(data[var], bins=min(bins, data[var].nunique()), duplicates="drop")

    grouped = (
        data.groupby("bin", observed=True)
        .agg(x=(var, "mean"), alpha_mean=("alpha", "mean"), n=("alpha", "size"))
        .reset_index(drop=True)
    )
    return grouped


def plot_relationship_grid(frame: pd.DataFrame, fuel: str) -> Path:
    fig, axes = plt.subplots(4, 3, figsize=(14, 14))
    axes = axes.ravel()

    for ax, var in zip(axes, VARIABLES):
        sns.scatterplot(
            data=frame,
            x=var,
            y="alpha",
            s=18,
            alpha=0.45,
            edgecolor=None,
            ax=ax,
        )
        bins = _binned_means(frame, var)
        sns.lineplot(
            data=bins,
            x="x",
            y="alpha_mean",
            marker="o",
            linewidth=2,
            color="#c0392b",
            ax=ax,
        )
        ax.set_title(VARIABLE_LABELS[var], fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("alpha")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.2)

    for ax in axes[len(VARIABLES) :]:
        ax.axis("off")

    fig.suptitle(f"{FUEL_CONFIG[fuel]['title']} alpha: scatter and binned means", y=0.995)
    fig.tight_layout()
    path = RESULT_DIR / f"task2_alpha_relationship_grid_{fuel}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_direction_boxplot(frames: dict[str, pd.DataFrame]) -> Path:
    rows = []
    for fuel, frame in frames.items():
        temp = frame.copy()
        temp["fuel"] = FUEL_CONFIG[fuel]["title"]
        temp["direction"] = np.where(temp["f"] > 0, "upward f", "downward f")
        rows.append(temp[["fuel", "direction", "alpha"]])
    data = pd.concat(rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.boxplot(data=data, x="fuel", y="alpha", hue="direction", width=0.55, ax=ax)
    sns.stripplot(
        data=data,
        x="fuel",
        y="alpha",
        hue="direction",
        dodge=True,
        alpha=0.25,
        size=3,
        legend=False,
        ax=ax,
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Alpha distribution by theoretical adjustment direction")
    ax.set_xlabel("")
    ax.set_ylabel("alpha")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    path = RESULT_DIR / "task2_alpha_direction_boxplot.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    data = load_analysis_frame()
    frames = {fuel: build_fuel_frame(data, fuel) for fuel in FUEL_CONFIG}
    corr = calculate_correlations(frames)
    corr.to_csv(RESULT_DIR / "task2_alpha_correlation_tests.csv", index=False, encoding="utf-8-sig")

    sample_summary = pd.DataFrame(
        [
            {
                "fuel": fuel,
                "n_used": len(frame),
                "alpha_mean": frame["alpha"].mean(),
                "alpha_median": frame["alpha"].median(),
                "alpha_std": frame["alpha"].std(),
                "alpha_min": frame["alpha"].min(),
                "alpha_max": frame["alpha"].max(),
            }
            for fuel, frame in frames.items()
        ]
    )
    sample_summary.to_csv(
        RESULT_DIR / "task2_alpha_analysis_sample_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_paths = []
    for fuel, frame in frames.items():
        plot_paths.append(plot_correlation_heatmap(corr, fuel))
        plot_paths.append(plot_relationship_grid(frame, fuel))
    plot_paths.append(plot_direction_boxplot(frames))

    print("Generated alpha relationship analysis:")
    print(f"- task2_alpha_correlation_tests.csv: {len(corr)} rows")
    print("- task2_alpha_analysis_sample_summary.csv")
    for path in plot_paths:
        print(f"- {path.name}")
    print()
    print("Top correlations by fuel:")
    for fuel in FUEL_CONFIG:
        print(f"\n{fuel}")
        print(
            corr.loc[corr["fuel"] == fuel, ["variable", "pearson", "spearman", "kendall", "max_abs_corr"]]
            .head(6)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
