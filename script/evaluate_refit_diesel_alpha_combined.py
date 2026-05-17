"""
Evaluate and refit the combined diesel alpha function on optimized alpha data.

The initial combined function stitches together the partial-effect-fitted upward
and downward functions. Upward and downward coefficients are refitted separately
on the corresponding optimized-alpha samples.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = ROOT / "script"
RESULT_DIR = ROOT / "result"
sys.path.insert(0, str(SCRIPT_DIR))

from diesel_alpha_combined_adjustment_function import (  # noqa: E402
    DIESEL_COMBINED_DOWNWARD_ALPHA_PARAMS,
    DIESEL_COMBINED_UPWARD_ALPHA_PARAMS,
    diesel_alpha_adjustment,
)


BASELINE_ALPHA_FILE = RESULT_DIR / "task2_optimization_equal_weight_final_step001_updated.csv"
EVENT_FILE = RESULT_DIR / "task2_event_model_input_clean.csv"


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write a CSV, using an _updated suffix if the target is locked."""
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated{path.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        return fallback
FUNCTION_FILE = SCRIPT_DIR / "diesel_alpha_combined_adjustment_function.py"


def load_modeling_data() -> pd.DataFrame:
    alpha = pd.read_csv(BASELINE_ALPHA_FILE, encoding="utf-8-sig")
    events = pd.read_csv(EVENT_FILE, encoding="utf-8-sig")
    alpha["date"] = pd.to_datetime(alpha["date"])
    events["date"] = pd.to_datetime(events["date"])
    data = alpha.merge(events[["date", "cpi_yoy"]], on="date", how="left")
    data = data.sort_values("date").reset_index(drop=True)
    data["S_diesel_lag"] = data["S_diesel"].shift(1).fillna(0.0)
    data["prev_u_diesel"] = data["optimal_u_diesel"].shift(1).fillna(0.0)
    data["direction"] = np.where(data["f_diesel"] > 0, "upward", "downward")
    data.loc[data["f_diesel"].abs() <= 1e-12, "direction"] = "zero_f"
    return data.loc[data["direction"].ne("zero_f")].copy()


def predict(data: pd.DataFrame, upward_params: dict[str, float], downward_params: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            diesel_alpha_adjustment(
                f_gasoline=row.f_gasoline,
                f_diesel=row.f_diesel,
                S_diesel_lag=row.S_diesel_lag,
                prev_u_diesel=row.prev_u_diesel,
                cpi_yoy=row.cpi_yoy,
                upward_params=upward_params,
                downward_params=downward_params,
            )
            for row in data.itertuples(index=False)
        ],
        dtype=float,
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    sse = float(np.sum(err**2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": float(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
    }


def refit_upward_params(data: pd.DataFrame) -> dict[str, float]:
    upward = data.loc[data["direction"].eq("upward")].copy()
    y = upward["alpha_diesel"].to_numpy(dtype=float)
    up0 = DIESEL_COMBINED_UPWARD_ALPHA_PARAMS.copy()
    up_names = [
        "intercept",
        "beta_fd_log",
        "beta_fg_hinge",
        "beta_s_d",
        "beta_prev_u_d",
        "beta_cpi_low",
        "beta_cpi_high",
    ]
    theta0 = np.array([up0[k] for k in up_names], dtype=float)
    lower = np.array([0.40, -0.80, -0.40, -0.40, -0.40, -0.80, -0.80], dtype=float)
    upper = np.array([1.20, 0.20, 0.80, 0.80, 0.80, 0.80, 0.80], dtype=float)

    def unpack(theta: np.ndarray) -> dict[str, float]:
        up = up0.copy()
        for key, value in zip(up_names, theta):
            up[key] = float(value)
        return up

    def residual(theta: np.ndarray) -> np.ndarray:
        up = unpack(theta)
        pred = predict(upward, up, DIESEL_COMBINED_DOWNWARD_ALPHA_PARAMS)
        return pred - y

    fit = least_squares(
        residual,
        theta0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.03,
        max_nfev=30000,
    )
    return unpack(fit.x)


def refit_downward_params(data: pd.DataFrame) -> dict[str, float]:
    downward = data.loc[data["direction"].eq("downward")].copy()
    y = downward["alpha_diesel"].to_numpy(dtype=float)
    down0 = DIESEL_COMBINED_DOWNWARD_ALPHA_PARAMS.copy()
    down_names = [
        "intercept",
        "beta_abs_fd_log",
        "beta_s_d",
        "beta_prev_u_d",
    ]
    theta0 = np.array([down0[k] for k in down_names], dtype=float)
    lower = np.array([0.20, 0.00, -0.60, -0.60], dtype=float)
    upper = np.array([1.20, 1.20, 0.60, 0.60], dtype=float)

    def unpack(theta: np.ndarray) -> dict[str, float]:
        down = down0.copy()
        for key, value in zip(down_names, theta):
            down[key] = float(value)
        return down

    def residual(theta: np.ndarray) -> np.ndarray:
        down = unpack(theta)
        pred = predict(downward, DIESEL_COMBINED_UPWARD_ALPHA_PARAMS, down)
        return pred - y

    fit = least_squares(
        residual,
        theta0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.03,
        max_nfev=30000,
    )
    return unpack(fit.x)


def refit_params(data: pd.DataFrame) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
    up = refit_upward_params(data)
    down = refit_downward_params(data)
    fitted = data.copy()
    fitted["alpha_pred_refit"] = predict(data, up, down)
    fitted["residual_refit"] = fitted["alpha_pred_refit"] - fitted["alpha_diesel"]
    return up, down, fitted


def format_params(params: dict[str, float]) -> str:
    return "{\n" + "\n".join(f'    "{key}": {value:.12g},' for key, value in params.items()) + "\n}"


def update_combined_function_file(up: dict[str, float], down: dict[str, float]) -> None:
    text = FUNCTION_FILE.read_text(encoding="utf-8")
    replacements = {
        "DIESEL_COMBINED_UPWARD_ALPHA_PARAMS": up,
        "DIESEL_COMBINED_DOWNWARD_ALPHA_PARAMS": down,
    }
    for name, params in replacements.items():
        pattern = rf"{name} = \{{.*?\n\}}"
        text = re.sub(pattern, f"{name} = {format_params(params)}", text, flags=re.S)
    FUNCTION_FILE.write_text(text, encoding="utf-8")


def plot_predictions(data: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pred_col, title in [
        (axes[0], "alpha_pred_initial", "Initial partial-effect coefficients"),
        (axes[1], "alpha_pred_refit", "Refitted on optimized alpha"),
    ]:
        ax.scatter(data["alpha_diesel"], data[pred_col], s=26, alpha=0.75)
        ax.plot([0, 1], [0, 1], color="#c0392b", linewidth=1.8)
        ax.set_xlim(0.2, 1.03)
        ax.set_ylim(0.2, 1.03)
        ax.set_xlabel("optimized alpha")
        ax.set_ylabel("predicted alpha")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Combined diesel alpha function: prediction check", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = RESULT_DIR / "task2_alpha_diesel_combined_prediction_check.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    data = load_modeling_data()
    y = data["alpha_diesel"].to_numpy(dtype=float)
    data["alpha_pred_initial"] = predict(
        data,
        DIESEL_COMBINED_UPWARD_ALPHA_PARAMS,
        DIESEL_COMBINED_DOWNWARD_ALPHA_PARAMS,
    )
    data["residual_initial"] = data["alpha_pred_initial"] - data["alpha_diesel"]
    initial_metrics = metrics(y, data["alpha_pred_initial"].to_numpy(dtype=float))

    up_refit, down_refit, fitted = refit_params(data)
    data["alpha_pred_refit"] = fitted["alpha_pred_refit"]
    data["residual_refit"] = fitted["residual_refit"]
    refit_metrics = metrics(y, data["alpha_pred_refit"].to_numpy(dtype=float))
    update_combined_function_file(up_refit, down_refit)

    output_cols = [
        "date",
        "direction",
        "alpha_diesel",
        "alpha_pred_initial",
        "residual_initial",
        "alpha_pred_refit",
        "residual_refit",
        "f_gasoline",
        "f_diesel",
        "S_diesel_lag",
        "prev_u_diesel",
        "cpi_yoy",
    ]
    prediction_path = safe_to_csv(
        data[output_cols],
        RESULT_DIR / "task2_alpha_diesel_combined_predictions.csv",
    )

    param_rows = []
    for scope, params in [("upward", up_refit), ("downward", down_refit)]:
        for key, value in params.items():
            param_rows.append({"scope": scope, "parameter": key, "value": value})
    for scope, result in [("initial", initial_metrics), ("refit", refit_metrics)]:
        for key, value in result.items():
            param_rows.append({"scope": scope, "parameter": f"metric_{key}", "value": value})
    param_path = safe_to_csv(
        pd.DataFrame(param_rows),
        RESULT_DIR / "task2_alpha_diesel_combined_refit_params.csv",
    )

    plot_path = plot_predictions(data)

    by_direction = []
    for direction, group in data.groupby("direction"):
        by_direction.append(
            {
                "direction": direction,
                **{
                    f"initial_{k}": v
                    for k, v in metrics(group["alpha_diesel"].to_numpy(), group["alpha_pred_initial"].to_numpy()).items()
                },
                **{
                    f"refit_{k}": v
                    for k, v in metrics(group["alpha_diesel"].to_numpy(), group["alpha_pred_refit"].to_numpy()).items()
                },
            }
        )
    metrics_path = safe_to_csv(
        pd.DataFrame(by_direction),
        RESULT_DIR / "task2_alpha_diesel_combined_metrics_by_direction.csv",
    )

    print("Combined diesel alpha evaluation")
    print("Initial metrics:", initial_metrics)
    print("Refit metrics:", refit_metrics)
    print("By direction:")
    print(pd.DataFrame(by_direction).to_string(index=False))
    print("Outputs:")
    print(f"- {prediction_path.name}")
    print(f"- {param_path.name}")
    print(f"- {metrics_path.name}")
    print(f"- {plot_path.name}")
    print("- script/diesel_alpha_combined_adjustment_function.py")


if __name__ == "__main__":
    main()
