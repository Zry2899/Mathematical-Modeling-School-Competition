"""
两步法估计包含利润项 pi0 的结构参数。

步骤：
1. 读取 estimate_parameters.py 输出的第一阶段权重 w1,w2,w3；
2. 固定权重，计算 O_t、X_t = mu * nu * O_t、h(O_t)；
3. 用干净样本回归：
       actual_delta = alpha * Delta_X + pi0 * Delta_h + c
4. 保存最终结构参数；
5. 使用最终结构参数完整回代预测。
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "result"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import base_model


NU = base_model.PARAMS["barrel_per_ton"]
THRESHOLD = base_model.PARAMS["rule_threshold"]


def load_model_input():
    """
    读取国内调价事件、10 个交易日油价均值、汇率，并合并。
    """
    domestic = pd.read_csv(RESULT_DIR / "domestic_events_clean.csv")
    oil = pd.read_csv(RESULT_DIR / "oil_10_workday_average.csv")
    exchange = pd.read_csv(DATA_DIR / "cny_usd_exchange_rate.csv")

    exchange = exchange.iloc[:, 0:2].copy()
    exchange.columns = ["date", "exchange_rate"]

    domestic["date"] = pd.to_datetime(domestic["date"])
    domestic["notice_date"] = pd.to_datetime(domestic["notice_date"])
    oil["date"] = pd.to_datetime(oil["date"])
    oil["notice_date"] = pd.to_datetime(oil["notice_date"])
    exchange["date"] = pd.to_datetime(exchange["date"])

    data = domestic.merge(
        oil[["date", "notice_date", "wti_mean", "brent_mean", "basket_mean"]],
        on=["date", "notice_date"],
        how="left",
    )

    data = pd.merge_asof(
        data.sort_values("date"),
        exchange.sort_values("date"),
        on="date",
        direction="backward",
    )

    return data.sort_values("date").reset_index(drop=True)


def h_profit(oil_price):
    """
    利润分段函数中的 h(O)。

    pi(O) = pi0 * h(O)
    """
    oil_price = np.asarray(oil_price, dtype=float)
    return np.where(
        oil_price <= 80,
        1.0,
        np.where(oil_price <= 130, (130 - oil_price) / 50, 0.0),
    )


def load_stage1_weights(fuel_type):
    """
    读取第一阶段 estimate_parameters.py 得到的权重。
    """
    path = RESULT_DIR / f"parameter_estimation_{fuel_type}.csv"
    if not path.exists():
        raise FileNotFoundError(f"请先运行 estimate_parameters.py，缺少文件: {path}")

    row = pd.read_csv(path).iloc[0]
    return {
        "w_wti": float(row["w1_wti"]),
        "w_brent": float(row["w2_brent"]),
        "w_basket": float(row["w3_basket"]),
    }


def build_features(data, weights, fuel_type):
    """
    固定权重后构造 X、h 及其差分。
    """
    df = data.copy()

    if fuel_type == "gasoline":
        df["actual_delta"] = pd.to_numeric(df["gasoline_change"], errors="coerce")
    elif fuel_type == "diesel":
        df["actual_delta"] = pd.to_numeric(df["diesel_change"], errors="coerce")
    else:
        raise ValueError("fuel_type 必须是 gasoline 或 diesel")

    df["weighted_oil"] = (
        weights["w_wti"] * df["wti_mean"]
        + weights["w_brent"] * df["brent_mean"]
        + weights["w_basket"] * df["basket_mean"]
    )

    # 利润函数用原始综合油价 O_t；如果需要地板价，可在这里替换为 clip(lower=40)
    df["h_oil"] = h_profit(df["weighted_oil"])
    df["X"] = df["exchange_rate"] * NU * df["weighted_oil"]

    df["delta_X"] = df["X"].diff()
    df["delta_h"] = df["h_oil"].diff()
    df["prev_actual_delta"] = df["actual_delta"].shift(1)
    df["prev_is_special_regulated"] = df["is_special_regulated"].shift(1).fillna(False)
    df["prev_is_mechanism_reform"] = df["special_type"].eq("mechanism_reform").shift(1).fillna(False)
    df["is_mechanism_reform"] = df["special_type"].eq("mechanism_reform")

    return df


def build_clean_sample(feature_df):
    """
    构造第二阶段回归的干净样本。
    """
    clean = feature_df[
        (feature_df["is_special_regulated"] == False)
        & (feature_df["prev_is_special_regulated"] == False)
        & (feature_df["is_mechanism_reform"] == False)
        & (feature_df["prev_is_mechanism_reform"] == False)
        & (feature_df["prev_actual_delta"].abs() >= THRESHOLD)
        & (feature_df["actual_delta"].abs() >= THRESHOLD)
    ].copy()

    clean = clean.dropna(subset=["delta_X", "delta_h", "actual_delta"])
    return clean


def fit_alpha_pi0_c(clean):
    """
    回归 actual_delta = alpha * delta_X + pi0 * delta_h + c。

    alpha、pi0 非负；c 不限正负。
    因为只有两个非负变量，这里用主动集枚举实现约束最小二乘。
    """
    A = np.column_stack([
        clean["delta_X"].to_numpy(dtype=float),
        clean["delta_h"].to_numpy(dtype=float),
        np.ones(len(clean)),
    ])
    y = clean["actual_delta"].to_numpy(dtype=float)

    best_coef = None
    best_sse = None

    # mask 控制 alpha/pi0 哪些变量参与回归；未参与的固定为 0，截距始终参与。
    for mask in range(1, 4):
        active = [i for i in range(2) if (mask >> i) & 1]
        cols = active + [2]
        coef_sub, _, _, _ = np.linalg.lstsq(A[:, cols], y, rcond=None)

        coef = np.zeros(3)
        coef[cols] = coef_sub

        if coef[0] < -1e-8 or coef[1] < -1e-8:
            continue

        pred = A @ coef
        sse = float(np.sum((pred - y) ** 2))
        if best_sse is None or sse < best_sse:
            best_sse = sse
            best_coef = coef

    # alpha=pi0=0，只估截距，作为兜底
    coef = np.array([0.0, 0.0, float(np.mean(y))])
    sse = float(np.sum((A @ coef - y) ** 2))
    if best_sse is None or sse < best_sse:
        best_coef = coef

    pred = A @ best_coef
    residual = pred - y
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    return {
        "alpha": float(best_coef[0]),
        "pi0": float(best_coef[1]),
        "intercept": float(best_coef[2]),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "n": len(clean),
    }


def backtest(feature_df, params):
    """
    使用最终结构参数完整回代预测。
    """
    carry = 0.0
    rows = []

    for _, row in feature_df.iterrows():
        if pd.isna(row["delta_X"]) or pd.isna(row["delta_h"]):
            raw_rule_delta = np.nan
            rule_delta = np.nan
            pred_delta = np.nan
            carry_before = carry
            carry_after = carry
        else:
            raw_rule_delta = (
                params["alpha"] * row["delta_X"]
                + params["pi0"] * row["delta_h"]
                + params["intercept"]
            )

            carry_before = carry
            rule_delta = carry_before + raw_rule_delta

            if abs(rule_delta) >= THRESHOLD:
                pred_delta = rule_delta
                carry = 0.0
            else:
                pred_delta = 0.0
                carry = rule_delta

            carry_after = carry

        rows.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "actual_delta": row["actual_delta"],
            "weighted_oil": row["weighted_oil"],
            "h_oil": row["h_oil"],
            "delta_X": row["delta_X"],
            "delta_h": row["delta_h"],
            "raw_rule_delta": raw_rule_delta,
            "rule_delta": rule_delta,
            "pred_delta_no_special": pred_delta,
            "carry_before": carry_before,
            "carry_after": carry_after,
            "is_special_regulated": row["is_special_regulated"],
            "special_type": row["special_type"],
            "abs_error": abs(pred_delta - row["actual_delta"]) if not pd.isna(pred_delta) else np.nan,
        })

    return pd.DataFrame(rows)


def run_for_fuel(data, fuel_type):
    """
    完成某个油品的第二阶段参数估计与回代。
    """
    weights = load_stage1_weights(fuel_type)
    features = build_features(data, weights, fuel_type)
    clean = build_clean_sample(features)
    params = fit_alpha_pi0_c(clean)
    params.update(weights)
    params["fuel_type"] = fuel_type

    result = backtest(features, params)

    return params, result


def main():
    """
    程序入口。
    """
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_model_input()

    gas_params, gas_backtest = run_for_fuel(data, "gasoline")
    diesel_params, diesel_backtest = run_for_fuel(data, "diesel")

    pd.DataFrame([gas_params]).to_csv(
        RESULT_DIR / "structural_parameters_gasoline.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([diesel_params]).to_csv(
        RESULT_DIR / "structural_parameters_diesel.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gas_backtest.to_csv(RESULT_DIR / "structural_backtest_gasoline.csv", index=False, encoding="utf-8-sig")
    diesel_backtest.to_csv(RESULT_DIR / "structural_backtest_diesel.csv", index=False, encoding="utf-8-sig")

    print("结构参数估计完成")
    print("汽油参数:")
    print(pd.DataFrame([gas_params]).to_string(index=False))
    print()
    print("柴油参数:")
    print(pd.DataFrame([diesel_params]).to_string(index=False))


if __name__ == "__main__":
    main()
