"""
Price transmission asymmetry analysis.

This script estimates regression-based transmission indices from international
oil-cost changes to domestic product-oil price changes.
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover - optional dependency
    stats = None


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"

FUEL_FILES = {
    "gasoline": RESULT_DIR / "structural_backtest_gasoline.csv",
    "diesel": RESULT_DIR / "structural_backtest_diesel.csv",
}

TARGETS = {
    "actual": "actual_delta",
    "model": "pred_delta_no_special",
}

FIRST_REFORM_DATE = pd.Timestamp("2016-01-14")


def oil_range_label(oil_price):
    """Classify by the current regulation intervals."""
    if pd.isna(oil_price):
        return np.nan
    if oil_price < 40:
        return "<40"
    if oil_price <= 80:
        return "40-80"
    if oil_price < 130:
        return "80-130"
    return ">=130"


def load_sample(fuel):
    """Load one fuel sample and apply the agreed sample filters."""
    df = pd.read_csv(FUEL_FILES[fuel])
    df["date"] = pd.to_datetime(df["date"])

    for col in ["actual_delta", "pred_delta_no_special", "delta_X", "weighted_oil"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "is_special_regulated" in df.columns:
        df["is_special_regulated"] = (
            df["is_special_regulated"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
        )
    else:
        df["is_special_regulated"] = False

    df["special_type"] = df.get("special_type", "").astype(str)

    sample = df[
        (df["date"] != FIRST_REFORM_DATE)
        & (df["is_special_regulated"] == False)
        & (df["special_type"] != "mechanism_reform")
    ].copy()

    sample = sample.dropna(subset=["delta_X", "weighted_oil"])
    sample["oil_range"] = sample["weighted_oil"].apply(oil_range_label)
    sample["fuel_type"] = fuel
    return sample


def ols_transmission(df, y_col):
    """
    Estimate Y_t = alpha + beta_up * Z_plus - beta_down * Z_minus + error.

    Z_plus = max(delta_X, 0), Z_minus = max(-delta_X, 0).
    The design matrix uses -Z_minus, so beta_down is directly comparable
    with beta_up and should be positive under normal pass-through.
    """
    work = df.dropna(subset=["delta_X", y_col]).copy()
    z = work["delta_X"].to_numpy(dtype=float)
    y = work[y_col].to_numpy(dtype=float)
    z_up = np.maximum(z, 0.0)
    z_down = np.maximum(-z, 0.0)

    n = len(work)
    n_up = int((z > 0).sum())
    n_down = int((z < 0).sum())
    n_zero = int((z == 0).sum())

    base = {
        "n": n,
        "n_up": n_up,
        "n_down": n_down,
        "n_zero": n_zero,
        "z_up_sum": float(z_up.sum()) if n else np.nan,
        "z_down_sum": float(z_down.sum()) if n else np.nan,
        "y_sum": float(y.sum()) if n else np.nan,
    }

    if n < 4 or n_up == 0 or n_down == 0:
        return {
            **base,
            "intercept": np.nan,
            "beta_up": np.nan,
            "beta_down": np.nan,
            "asymmetry_index": np.nan,
            "beta_diff": np.nan,
            "t_beta_equal": np.nan,
            "p_beta_equal": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
            "status": "insufficient_sample",
        }

    x = np.column_stack([np.ones(n), z_up, -z_down])
    rank = np.linalg.matrix_rank(x)
    if rank < x.shape[1]:
        return {
            **base,
            "intercept": np.nan,
            "beta_up": np.nan,
            "beta_down": np.nan,
            "asymmetry_index": np.nan,
            "beta_diff": np.nan,
            "t_beta_equal": np.nan,
            "p_beta_equal": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
            "status": "rank_deficient",
        }

    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coef
    resid = y - fitted
    sse = float(resid @ resid)
    tss = float(((y - y.mean()) @ (y - y.mean())))
    dof = n - x.shape[1]
    sigma2 = sse / dof
    xtx_inv = np.linalg.inv(x.T @ x)
    cov = sigma2 * xtx_inv

    beta_up = float(coef[1])
    beta_down = float(coef[2])
    beta_diff = beta_up - beta_down
    var_diff = cov[1, 1] + cov[2, 2] - 2.0 * cov[1, 2]
    se_diff = float(np.sqrt(max(var_diff, 0.0)))
    t_stat = beta_diff / se_diff if se_diff > 0 else np.nan

    if stats is not None and not np.isnan(t_stat):
        p_value = float(2.0 * stats.t.sf(abs(t_stat), dof))
    else:
        p_value = np.nan

    denom = beta_up + beta_down
    asymmetry_index = beta_diff / denom if abs(denom) > 1e-12 else np.nan

    return {
        **base,
        "intercept": float(coef[0]),
        "beta_up": beta_up,
        "beta_down": beta_down,
        "asymmetry_index": float(asymmetry_index),
        "beta_diff": float(beta_diff),
        "t_beta_equal": float(t_stat),
        "p_beta_equal": p_value,
        "r2": float(1.0 - sse / tss) if tss > 0 else np.nan,
        "rmse": float(np.sqrt(sse / n)),
        "status": "ok",
    }

 
def ratio_transmission(df, y_col):
    """
    Estimate pass-through by direct cumulative ratios instead of OLS.

    Upward cost periods use beta_up = sum(Y) / sum(delta_X).
    Downward cost periods use beta_down = sum(-Y) / sum(-delta_X), so a
    symmetric response has beta_up close to beta_down.
    """
    work = df.dropna(subset=["delta_X", y_col]).copy()
    z = work["delta_X"].to_numpy(dtype=float)
    y = work[y_col].to_numpy(dtype=float)

    up_mask = z > 0
    down_mask = z < 0
    n = len(work)
    n_up = int(up_mask.sum())
    n_down = int(down_mask.sum())
    n_zero = int((z == 0).sum())

    z_up = z[up_mask]
    y_up = y[up_mask]
    z_down = -z[down_mask]
    y_down = -y[down_mask]

    z_up_sum = float(z_up.sum()) if n_up else np.nan
    z_down_sum = float(z_down.sum()) if n_down else np.nan
    y_up_sum = float(y_up.sum()) if n_up else np.nan
    y_down_sum = float(y_down.sum()) if n_down else np.nan

    base = {
        "n": n,
        "n_up": n_up,
        "n_down": n_down,
        "n_zero": n_zero,
        "z_up_sum": z_up_sum,
        "z_down_sum": z_down_sum,
        "y_up_sum": y_up_sum,
        "y_down_sum": y_down_sum,
        "y_sum": float(y.sum()) if n else np.nan,
    }

    if n_up == 0 or n_down == 0 or abs(z_up_sum) < 1e-12 or abs(z_down_sum) < 1e-12:
        return {
            **base,
            "beta_up": np.nan,
            "beta_down": np.nan,
            "asymmetry_index": np.nan,
            "beta_diff": np.nan,
            "ratio_up_mean": np.nan,
            "ratio_down_mean": np.nan,
            "ratio_up_median": np.nan,
            "ratio_down_median": np.nan,
            "status": "insufficient_sample",
        }

    beta_up = y_up_sum / z_up_sum
    beta_down = y_down_sum / z_down_sum
    denom = beta_up + beta_down
    asymmetry_index = (beta_up - beta_down) / denom if abs(denom) > 1e-12 else np.nan
    ratio_up = y_up / z_up
    ratio_down = y_down / z_down

    return {
        **base,
        "beta_up": float(beta_up),
        "beta_down": float(beta_down),
        "asymmetry_index": float(asymmetry_index),
        "beta_diff": float(beta_up - beta_down),
        "ratio_up_mean": float(np.mean(ratio_up)),
        "ratio_down_mean": float(np.mean(ratio_down)),
        "ratio_up_median": float(np.median(ratio_up)),
        "ratio_down_median": float(np.median(ratio_down)),
        "status": "ok",
    }


def add_judgement(row):
    """Attach a compact qualitative judgement."""
    if row["status"] != "ok":
        return "样本不足，不作强结论"
    a = row["asymmetry_index"]
    p = row["p_beta_equal"]
    if pd.notna(p) and p < 0.05:
        if a > 0:
            return "上涨传递显著强于下跌"
        if a < 0:
            return "下跌传递显著强于上涨"
    if abs(a) < 0.05:
        return "双向传递基本对称"
    if a > 0:
        return "上涨传递略强，但统计证据有限"
    return "下跌传递略强，但统计证据有限"


def add_ratio_judgement(row):
    """Attach a compact qualitative judgement for direct-ratio estimates."""
    if row["status"] != "ok":
        return "样本不足，不作强结论"
    a = row["asymmetry_index"]
    if pd.isna(a):
        return "无法判断"
    if abs(a) < 0.05:
        return "双向传递基本对称"
    if a > 0:
        return "上涨传递更强"
    return "下跌传递更强"


def build_results():
    overall_rows = []
    range_rows = []

    for fuel in FUEL_FILES:
        sample = load_sample(fuel)

        for target_name, y_col in TARGETS.items():
            result = ols_transmission(sample, y_col)
            overall_rows.append(
                {
                    "fuel_type": fuel,
                    "target": target_name,
                    "y_column": y_col,
                    "oil_range": "overall",
                    **result,
                }
            )

            for label in ["<40", "40-80", "80-130", ">=130"]:
                sub = sample[sample["oil_range"] == label].copy()
                result = ols_transmission(sub, y_col)
                range_rows.append(
                    {
                        "fuel_type": fuel,
                        "target": target_name,
                        "y_column": y_col,
                        "oil_range": label,
                        **result,
                    }
                )

    overall = pd.DataFrame(overall_rows)
    by_range = pd.DataFrame(range_rows)

    for df in [overall, by_range]:
        df["judgement"] = df.apply(add_judgement, axis=1)

    return overall, by_range


def build_ratio_results():
    overall_rows = []
    range_rows = []

    for fuel in FUEL_FILES:
        sample = load_sample(fuel)

        for target_name, y_col in TARGETS.items():
            result = ratio_transmission(sample, y_col)
            overall_rows.append(
                {
                    "fuel_type": fuel,
                    "target": target_name,
                    "y_column": y_col,
                    "oil_range": "overall",
                    "method": "direct_ratio",
                    **result,
                }
            )

            for label in ["<40", "40-80", "80-130", ">=130"]:
                sub = sample[sample["oil_range"] == label].copy()
                result = ratio_transmission(sub, y_col)
                range_rows.append(
                    {
                        "fuel_type": fuel,
                        "target": target_name,
                        "y_column": y_col,
                        "oil_range": label,
                        "method": "direct_ratio",
                        **result,
                    }
                )

    overall = pd.DataFrame(overall_rows)
    by_range = pd.DataFrame(range_rows)

    for df in [overall, by_range]:
        df["judgement"] = df.apply(add_ratio_judgement, axis=1)

    return overall, by_range


def write_csv_with_fallback(df, path):
    """Write a CSV; if the target is open elsewhere, add an _updated suffix."""
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(path.stem + "_updated" + path.suffix)
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        return fallback


def main():
    overall, by_range = build_results()
    ratio_overall, ratio_by_range = build_ratio_results()

    overall_path = RESULT_DIR / "transmission_asymmetry_overall.csv"
    range_path = RESULT_DIR / "transmission_asymmetry_by_range.csv"
    ratio_overall_path = RESULT_DIR / "transmission_asymmetry_ratio_overall.csv"
    ratio_range_path = RESULT_DIR / "transmission_asymmetry_ratio_by_range.csv"

    overall_path = write_csv_with_fallback(overall, overall_path)
    range_path = write_csv_with_fallback(by_range, range_path)
    ratio_overall_path = write_csv_with_fallback(ratio_overall, ratio_overall_path)
    ratio_range_path = write_csv_with_fallback(ratio_by_range, ratio_range_path)

    display_cols = [
        "fuel_type",
        "target",
        "oil_range",
        "n",
        "n_up",
        "n_down",
        "beta_up",
        "beta_down",
        "asymmetry_index",
        "p_beta_equal",
        "r2",
        "judgement",
    ]

    print("Overall results:")
    print(overall[display_cols].to_string(index=False))
    print()
    print("By-range results:")
    print(by_range[display_cols].to_string(index=False))
    print()
    ratio_display_cols = [
        "fuel_type",
        "target",
        "oil_range",
        "n",
        "n_up",
        "n_down",
        "beta_up",
        "beta_down",
        "asymmetry_index",
        "judgement",
    ]
    print("Ratio overall results:")
    print(ratio_overall[ratio_display_cols].to_string(index=False))
    print()
    print("Ratio by-range results:")
    print(ratio_by_range[ratio_display_cols].to_string(index=False))
    print()
    print(f"Saved: {overall_path}")
    print(f"Saved: {range_path}")
    print(f"Saved: {ratio_overall_path}")
    print(f"Saved: {ratio_range_path}")


if __name__ == "__main__":
    main()
