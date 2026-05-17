"""
Fit and plot downward alpha approximation functions for gasoline and diesel.

The fitted functions are intentionally simple and interpretable. For each fuel,
the own theoretical downward adjustment enters through log(1 + |f_i|), while
lagged pressure and previous adjustment enter linearly. The functions are fitted
to model-implied partial-effect points from the six-loss equal-weight welfare
optimization, not to historical observed policy alpha.
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
from diesel_alpha_downward_adjustment_function import diesel_alpha_downward_adjustment
from gasoline_alpha_downward_adjustment_function import gasoline_alpha_downward_adjustment


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result"
SCRIPT_DIR = ROOT / "script"

PARTIAL_EFFECT_FILE = RESULT_DIR / "task2_alpha_partial_effects.csv"
SCENARIO = "downward_base"


FUEL_CONFIG = {
    "gasoline": {
        "target": "alpha_gasoline",
        "variables": ["f_gasoline", "S_gasoline_lag", "prev_u_gasoline"],
        "labels": {
            "f_gasoline": "gasoline f",
            "S_gasoline_lag": "lagged S, gasoline",
            "prev_u_gasoline": "previous u, gasoline",
        },
        "function": gasoline_alpha_downward_adjustment,
        "param_file": SCRIPT_DIR / "gasoline_alpha_downward_adjustment_function.py",
        "param_name": "GASOLINE_DOWNWARD_ADJUSTMENT_ALPHA_PARAMS",
        "result_prefix": "task2_alpha_gasoline_downward_function",
        "f_key": "f_gasoline",
        "s_key": "S_gasoline_lag",
        "u_key": "prev_u_gasoline",
        "threshold_key": "fg_abs_threshold",
        "log_center_key": "abs_fg_log_center",
        "s_center_key": "S_g_center",
        "u_center_key": "prev_u_g_center",
        "beta_names": ["intercept", "beta_abs_fg_log", "beta_s_g", "beta_prev_u_g"],
    },
    "diesel": {
        "target": "alpha_diesel",
        "variables": ["f_diesel", "S_diesel_lag", "prev_u_diesel"],
        "labels": {
            "f_diesel": "diesel f",
            "S_diesel_lag": "lagged S, diesel",
            "prev_u_diesel": "previous u, diesel",
        },
        "function": diesel_alpha_downward_adjustment,
        "param_file": SCRIPT_DIR / "diesel_alpha_downward_adjustment_function.py",
        "param_name": "DIESEL_DOWNWARD_ADJUSTMENT_ALPHA_PARAMS",
        "result_prefix": "task2_alpha_diesel_downward_function",
        "f_key": "f_diesel",
        "s_key": "S_diesel_lag",
        "u_key": "prev_u_diesel",
        "threshold_key": "fd_abs_threshold",
        "log_center_key": "abs_fd_log_center",
        "s_center_key": "S_d_center",
        "u_center_key": "prev_u_d_center",
        "beta_names": ["intercept", "beta_abs_fd_log", "beta_s_d", "beta_prev_u_d"],
    },
}


def build_base_features(fuel: str) -> dict[str, float]:
    events, baseline = load_data()
    state = build_base_state(events, baseline, SCENARIO)
    row = state["row"]
    if fuel == "gasoline":
        return {
            "f_gasoline": float(row["f_gasoline"]),
            "S_gasoline_lag": float(state["s_g_prev"]),
            "prev_u_gasoline": float(state["prev_u_g"]),
        }
    return {
        "f_diesel": float(row["f_diesel"]),
        "S_diesel_lag": float(state["s_d_prev"]),
        "prev_u_diesel": float(state["prev_u_d"]),
    }


def load_training_data(fuel: str, base: dict[str, float]) -> pd.DataFrame:
    cfg = FUEL_CONFIG[fuel]
    data = pd.read_csv(PARTIAL_EFFECT_FILE, encoding="utf-8-sig")
    data = data.loc[
        data["scenario"].eq(SCENARIO) & data["variable"].isin(cfg["variables"])
    ].copy()

    rows = []
    for row in data.itertuples(index=False):
        features = base.copy()
        features[row.variable] = float(row.x_value)
        if features[cfg["f_key"]] >= 0:
            continue
        rows.append(
            {
                "variable": row.variable,
                "x_value": float(row.x_value),
                "alpha_target": float(getattr(row, cfg["target"])),
                **features,
            }
        )
    return pd.DataFrame(rows)


def make_params(fuel: str, theta: np.ndarray, base: dict[str, float]) -> dict[str, float]:
    cfg = FUEL_CONFIG[fuel]
    params = {name: float(value) for name, value in zip(cfg["beta_names"], theta)}
    threshold = 0.0
    params.update(
        {
            cfg["threshold_key"]: threshold,
            cfg["log_center_key"]: float(
                np.log1p(max(abs(base[cfg["f_key"]]) - threshold, 0.0))
            ),
            cfg["s_center_key"]: float(base[cfg["s_key"]]),
            cfg["u_center_key"]: float(base[cfg["u_key"]]),
        }
    )
    return params


def predict_frame(fuel: str, data: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    cfg = FUEL_CONFIG[fuel]
    fn = cfg["function"]
    if fuel == "gasoline":
        return np.array(
            [
                fn(
                    f_gasoline=row.f_gasoline,
                    S_gasoline_lag=row.S_gasoline_lag,
                    prev_u_gasoline=row.prev_u_gasoline,
                    params=params,
                )
                for row in data.itertuples(index=False)
            ],
            dtype=float,
        )
    return np.array(
        [
            fn(
                f_diesel=row.f_diesel,
                S_diesel_lag=row.S_diesel_lag,
                prev_u_diesel=row.prev_u_diesel,
                params=params,
            )
            for row in data.itertuples(index=False)
        ],
        dtype=float,
    )


def fit_model(fuel: str, train: pd.DataFrame, base: dict[str, float]) -> tuple[dict[str, float], pd.DataFrame]:
    def residual(theta: np.ndarray) -> np.ndarray:
        params = make_params(fuel, theta, base)
        pred = predict_frame(fuel, train, params)
        return pred - train["alpha_target"].to_numpy()

    # Downward own |f| effect is increasing and saturating; lagged pressure and
    # previous adjustment have mild negative slopes in the partial-effect data.
    lower = np.array([0.20, 0.00, -0.20, -0.20])
    upper = np.array([1.00, 0.60, 0.00, 0.00])
    initial = np.array([0.98, 0.05, -0.02, -0.02])
    fit = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.02,
        max_nfev=20000,
    )
    params = make_params(fuel, fit.x, base)
    train = train.copy()
    train["alpha_fitted"] = predict_frame(fuel, train, params)
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


def update_function_file(fuel: str, params: dict[str, float]) -> None:
    cfg = FUEL_CONFIG[fuel]
    text = cfg["param_file"].read_text(encoding="utf-8")
    formatted = "{\n" + "\n".join(
        f'    "{key}": {value:.12g},' for key, value in params.items()
    ) + "\n}"
    pattern = rf"{cfg['param_name']} = \{{.*?\n\}}"
    replacement = f"{cfg['param_name']} = {formatted}"
    new_text = re.sub(pattern, replacement, text, flags=re.S)
    cfg["param_file"].write_text(new_text, encoding="utf-8")


def build_overlay_data(fuel: str, partial: pd.DataFrame, base: dict[str, float], params: dict[str, float]) -> pd.DataFrame:
    cfg = FUEL_CONFIG[fuel]
    rows = []
    for variable in cfg["variables"]:
        subset = partial.loc[partial["variable"].eq(variable)].copy()
        if variable == cfg["f_key"]:
            subset = subset.loc[subset["x_value"] < 0]
        for row in subset.sort_values("x_value").itertuples(index=False):
            features = base.copy()
            features[variable] = float(row.x_value)
            if features[cfg["f_key"]] >= 0:
                continue
            if fuel == "gasoline":
                fitted = cfg["function"](
                    f_gasoline=features["f_gasoline"],
                    S_gasoline_lag=features["S_gasoline_lag"],
                    prev_u_gasoline=features["prev_u_gasoline"],
                    params=params,
                )
            else:
                fitted = cfg["function"](
                    f_diesel=features["f_diesel"],
                    S_diesel_lag=features["S_diesel_lag"],
                    prev_u_diesel=features["prev_u_diesel"],
                    params=params,
                )
            rows.append(
                {
                    "variable": variable,
                    "x_value": float(row.x_value),
                    "alpha_target": float(getattr(row, cfg["target"])),
                    "alpha_fitted": fitted,
                }
            )
    return pd.DataFrame(rows)


def plot_overlay(fuel: str, overlay: pd.DataFrame) -> Path:
    cfg = FUEL_CONFIG[fuel]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, variable in zip(axes, cfg["variables"]):
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
        ax.set_title(cfg["labels"][variable])
        ax.set_ylim(0.15, 1.03)
        ax.set_xlabel("")
        ax.set_ylabel(f"{fuel} alpha")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(f"{fuel.title()} downward alpha function: partial-effect fit check", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.98, 0.92])
    path = RESULT_DIR / f"{cfg['result_prefix']}_overlay.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_one(fuel: str) -> None:
    cfg = FUEL_CONFIG[fuel]
    base = build_base_features(fuel)
    train = load_training_data(fuel, base)
    params, fitted_train = fit_model(fuel, train, base)
    fit_metrics = metrics(fitted_train)
    update_function_file(fuel, params)

    rows = [{"parameter": key, "value": value} for key, value in params.items()]
    for key, value in fit_metrics.items():
        rows.append({"parameter": f"metric_{key}", "value": value})
    pd.DataFrame(rows).to_csv(
        RESULT_DIR / f"{cfg['result_prefix']}_params.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fitted_train.to_csv(
        RESULT_DIR / f"{cfg['result_prefix']}_fit_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    partial = pd.read_csv(PARTIAL_EFFECT_FILE, encoding="utf-8-sig")
    partial = partial.loc[partial["scenario"].eq(SCENARIO)].copy()
    overlay = build_overlay_data(fuel, partial, base, params)
    overlay.to_csv(
        RESULT_DIR / f"{cfg['result_prefix']}_overlay_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_path = plot_overlay(fuel, overlay)

    print(f"Fitted {fuel} downward alpha function")
    print(pd.DataFrame([params]).T.rename(columns={0: "value"}).to_string())
    print("\nMetrics")
    print(pd.DataFrame([fit_metrics]).to_string(index=False))
    print("\nOutputs")
    print(f"- {cfg['result_prefix']}_params.csv")
    print(f"- {cfg['result_prefix']}_fit_data.csv")
    print(f"- {cfg['result_prefix']}_overlay_data.csv")
    print(f"- {plot_path.name}")
    print(f"- {cfg['param_file'].relative_to(ROOT)}")
    print()


def main() -> None:
    for fuel in ["gasoline", "diesel"]:
        run_one(fuel)


if __name__ == "__main__":
    main()
