"""
Fit and plot the gasoline upward alpha approximation function.

The fitted function is intentionally simple and interpretable:

alpha_g^up = clip[
    beta0
    + beta1 * log(1 + max(f_g - k_g, 0))
    + beta2 * max(f_d - k_d, 0)
    + beta3 * S_g(t-1)
    + beta4 * u_g(t-1)
    + beta5 * max(cpi_low - cpi, 0)
    + beta6 * max(cpi - cpi_high, 0),
    0, 1
]

It is fitted to the model-implied partial-effect points from the six-loss
equal-weight welfare optimization, not to historical observed policy alpha.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from analyze_alpha_partial_effects import build_base_state, load_data
from gasoline_alpha_upward_adjustment_function import gasoline_alpha_upward_adjustment


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result"
FUNCTION_FILE = ROOT / "script" / "gasoline_alpha_upward_adjustment_function.py"

PARTIAL_EFFECT_FILE = RESULT_DIR / "task2_alpha_partial_effects.csv"
SCENARIO = "upward_base"
MODELED_VARIABLES = [
    "f_gasoline",
    "f_diesel",
    "S_gasoline_lag",
    "prev_u_gasoline",
    "cpi_yoy",
]
VARIABLE_LABELS = {
    "f_gasoline": "gasoline f",
    "f_diesel": "diesel f",
    "S_gasoline_lag": "lagged S, gasoline",
    "prev_u_gasoline": "previous u, gasoline",
    "cpi_yoy": "CPI YoY",
}


def build_base_features() -> dict[str, float]:
    events, baseline = load_data()
    state = build_base_state(events, baseline, SCENARIO)
    row = state["row"]
    return {
        "f_gasoline": float(row["f_gasoline"]),
        "f_diesel": float(row["f_diesel"]),
        "S_gasoline_lag": float(state["s_g_prev"]),
        "prev_u_gasoline": float(state["prev_u_g"]),
        "cpi_yoy": float(row["cpi_yoy"]),
    }


def load_training_data(base: dict[str, float]) -> pd.DataFrame:
    data = pd.read_csv(PARTIAL_EFFECT_FILE, encoding="utf-8-sig")
    data = data.loc[
        data["scenario"].eq(SCENARIO) & data["variable"].isin(MODELED_VARIABLES)
    ].copy()

    rows = []
    for row in data.itertuples(index=False):
        features = base.copy()
        features[row.variable] = float(row.x_value)
        if features["f_gasoline"] <= 0 or features["f_diesel"] <= 0:
            continue
        rows.append(
            {
                "variable": row.variable,
                "x_value": float(row.x_value),
                "alpha_target": float(row.alpha_gasoline),
                **features,
            }
        )
    return pd.DataFrame(rows)


def make_params(theta: np.ndarray, base: dict[str, float]) -> dict[str, float]:
    names = [
        "intercept",
        "beta_fg_log",
        "beta_fd_hinge",
        "beta_s_g",
        "beta_prev_u_g",
        "beta_cpi_low",
        "beta_cpi_high",
    ]
    params = {name: float(value) for name, value in zip(names, theta)}
    params.update(
        {
            "fg_threshold": 100.0,
            "fd_threshold": 90.0,
            "cpi_low": 0.02,
            "cpi_high": 0.03,
            "fg_log_center": float(np.log1p(max(base["f_gasoline"] - 100.0, 0.0))),
            "fd_hinge_center": float(max(base["f_diesel"] - 90.0, 0.0) / 100.0),
            "S_g_center": float(base["S_gasoline_lag"]),
            "prev_u_g_center": float(base["prev_u_gasoline"]),
            "cpi_low_gap_center": float(max(0.02 - base["cpi_yoy"], 0.0) / 0.01),
            "cpi_high_gap_center": float(max(base["cpi_yoy"] - 0.03, 0.0) / 0.01),
        }
    )
    return params


def predict_frame(data: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            gasoline_alpha_upward_adjustment(
                f_gasoline=row.f_gasoline,
                f_diesel=row.f_diesel,
                S_gasoline_lag=row.S_gasoline_lag,
                prev_u_gasoline=row.prev_u_gasoline,
                cpi_yoy=row.cpi_yoy,
                params=params,
            )
            for row in data.itertuples(index=False)
        ],
        dtype=float,
    )


def fit_model(train: pd.DataFrame, base: dict[str, float]) -> tuple[dict[str, float], pd.DataFrame]:
    def residual(theta: np.ndarray) -> np.ndarray:
        params = make_params(theta, base)
        pred = predict_frame(train, params)
        return pred - train["alpha_target"].to_numpy()

    # Coefficient signs follow the inspected partial-effect shapes.
    lower = np.array([0.60, -0.40, 0.00, 0.00, 0.00, -0.40, -0.40])
    upper = np.array([1.00, 0.00, 0.20, 0.20, 0.20, 0.40, 0.00])
    initial = np.array([0.93, -0.02, 0.02, 0.02, 0.02, 0.00, -0.03])
    fit = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.02,
        max_nfev=20000,
    )
    params = make_params(fit.x, base)
    train = train.copy()
    train["alpha_fitted"] = predict_frame(train, params)
    train["residual"] = train["alpha_fitted"] - train["alpha_target"]
    return params, train


def metrics(train: pd.DataFrame) -> dict[str, float]:
    err = train["residual"].to_numpy()
    target = train["alpha_target"].to_numpy()
    sse = float(np.sum(err**2))
    sst = float(np.sum((target - target.mean()) ** 2))
    return {
        "n": float(len(train)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
    }


def save_parameter_outputs(params: dict[str, float], fit_metrics: dict[str, float]) -> None:
    rows = [{"parameter": key, "value": value} for key, value in params.items()]
    for key, value in fit_metrics.items():
        rows.append({"parameter": f"metric_{key}", "value": value})
    pd.DataFrame(rows).to_csv(
        RESULT_DIR / "task2_alpha_gasoline_upward_function_params.csv",
        index=False,
        encoding="utf-8-sig",
    )


def update_function_file(params: dict[str, float]) -> None:
    text = FUNCTION_FILE.read_text(encoding="utf-8")
    formatted = "{\n" + "\n".join(
        f'    "{key}": {value:.12g},' for key, value in params.items()
    ) + "\n}"
    pattern = r"GASOLINE_UPWARD_ADJUSTMENT_ALPHA_PARAMS = \{.*?\n\}"
    replacement = f"GASOLINE_UPWARD_ADJUSTMENT_ALPHA_PARAMS = {formatted}"
    new_text = re.sub(pattern, replacement, text, flags=re.S)
    FUNCTION_FILE.write_text(new_text, encoding="utf-8")


def build_overlay_data(partial: pd.DataFrame, base: dict[str, float], params: dict[str, float]) -> pd.DataFrame:
    rows = []
    for variable in MODELED_VARIABLES:
        subset = partial.loc[partial["variable"].eq(variable)].copy()
        if variable in ["f_gasoline", "f_diesel"]:
            subset = subset.loc[subset["x_value"] > 0]
        for row in subset.sort_values("x_value").itertuples(index=False):
            features = base.copy()
            features[variable] = float(row.x_value)
            if features["f_gasoline"] <= 0 or features["f_diesel"] <= 0:
                continue
            rows.append(
                {
                    "variable": variable,
                    "x_value": float(row.x_value),
                    "alpha_target": float(row.alpha_gasoline),
                "alpha_fitted": gasoline_alpha_upward_adjustment(**features, params=params),
                }
            )
    return pd.DataFrame(rows)


def plot_overlay(overlay: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for ax, variable in zip(axes, MODELED_VARIABLES):
        data = overlay.loc[overlay["variable"].eq(variable)].sort_values("x_value")
        ax.scatter(
            data["x_value"],
            data["alpha_target"],
            s=24,
            alpha=0.75,
            label="grid-search alpha",
        )
        ax.plot(
            data["x_value"],
            data["alpha_fitted"],
            color="#c0392b",
            linewidth=2.2,
            label="fitted function",
        )
        ax.set_title(VARIABLE_LABELS[variable])
        ax.set_ylim(0.35, 1.03)
        ax.set_xlabel("")
        ax.set_ylabel("gasoline alpha")
        ax.grid(True, alpha=0.25)

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.96, 0.96))
    fig.suptitle("Gasoline upward alpha function: partial-effect fit check", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.98, 0.95])
    path = RESULT_DIR / "task2_alpha_gasoline_upward_function_overlay.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    base = build_base_features()
    train = load_training_data(base)
    params, fitted_train = fit_model(train, base)
    fit_metrics = metrics(fitted_train)
    save_parameter_outputs(params, fit_metrics)
    update_function_file(params)

    fitted_train.to_csv(
        RESULT_DIR / "task2_alpha_gasoline_upward_function_fit_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    partial = pd.read_csv(PARTIAL_EFFECT_FILE, encoding="utf-8-sig")
    partial = partial.loc[partial["scenario"].eq(SCENARIO)].copy()
    overlay = build_overlay_data(partial, base, params)
    overlay.to_csv(
        RESULT_DIR / "task2_alpha_gasoline_upward_function_overlay_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_path = plot_overlay(overlay)

    print("Fitted gasoline upward alpha function")
    print(pd.DataFrame([params]).T.rename(columns={0: "value"}).to_string())
    print("\nMetrics")
    print(pd.DataFrame([fit_metrics]).to_string(index=False))
    print("\nOutputs")
    print("- task2_alpha_gasoline_upward_function_params.csv")
    print("- task2_alpha_gasoline_upward_function_fit_data.csv")
    print("- task2_alpha_gasoline_upward_function_overlay_data.csv")
    print(f"- {plot_path.name}")
    print("- script/gasoline_alpha_upward_adjustment_function.py")


if __name__ == "__main__":
    main()
