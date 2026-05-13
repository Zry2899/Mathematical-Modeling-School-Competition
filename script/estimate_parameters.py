"""
新的参数估计流程。

核心原则：
    正常样本估计基础参数
    -> 完整时序模型回代
    -> 特殊调控样本反推 theta_i
    -> 判断 theta_i 是否可近似为常数

本脚本不修改原始数据，所有输出写入 result 目录。
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
EPS = 1e-9


def scan_data_files():
    """
    扫描 data、result、script 目录下已有的 csv/xlsx 文件。

    返回一个列表，用于报告中说明数据来源。
    """
    files = []
    for folder in [DATA_DIR, RESULT_DIR, SCRIPT_DIR]:
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if path.suffix.lower() in [".csv", ".xlsx", ".xls"]:
                files.append(path)
    return files


def normalize_name(name):
    """
    标准化字段名，便于模糊匹配。
    """
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def find_column(df, candidates, required=True, label=""):
    """
    在 DataFrame 中自动匹配字段名。

    如果找不到必需字段，则打印现有字段并抛出清晰错误。
    """
    normalized = {normalize_name(col): col for col in df.columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    if required:
        print("无法识别字段:", label or candidates[0])
        print("候选字段:", candidates)
        print("现有字段:", list(df.columns))
        raise ValueError("关键字段缺失: " + (label or candidates[0]))

    return None


def identify_columns(df):
    """
    识别合并后特征表中的关键字段。
    """
    return {
        "date": find_column(df, ["date", "adjust_date", "effective_date"], label="date"),
        "gasoline_change": find_column(
            df,
            ["gasoline_change", "gas_change", "gasoline_delta", "gasoline_adjust_cny_per_ton"],
            label="gasoline_change",
        ),
        "diesel_change": find_column(
            df,
            ["diesel_change", "diesel_delta", "diesel_adjust_cny_per_ton"],
            label="diesel_change",
        ),
        "is_special_regulated": find_column(
            df,
            ["is_special_regulated", "special", "is_special", "temporary_control"],
            required=False,
            label="is_special_regulated",
        ),
        "wti": find_column(df, ["wti_mean", "wti", "WTI"], label="WTI"),
        "brent": find_column(df, ["brent_mean", "brent", "Brent"], label="Brent"),
        "basket": find_column(df, ["basket_mean", "basket", "BA", "Basket"], label="Basket"),
        "exchange_rate": find_column(df, ["exchange_rate", "mu", "汇率"], label="exchange_rate"),
    }


def read_exchange_rate():
    """
    读取汇率数据。中文列名不稳定，所以按前两列位置读取。
    """
    path = DATA_DIR / "cny_usd_exchange_rate.csv"
    if not path.exists():
        raise ValueError("找不到汇率文件: " + str(path))

    df = pd.read_csv(path)
    df = df.iloc[:, 0:2].copy()
    df.columns = ["date", "exchange_rate"]
    df["date"] = pd.to_datetime(df["date"])
    df["exchange_rate"] = pd.to_numeric(df["exchange_rate"], errors="coerce")
    df = df.dropna(subset=["date", "exchange_rate"])
    return df.sort_values("date").reset_index(drop=True)


def load_preferred_merged_data():
    """
    优先读取 result 中已经整理好的国内事件表和 10 工作日均价表。
    """
    domestic_path = RESULT_DIR / "domestic_events_clean.csv"
    oil_path = RESULT_DIR / "oil_10_workday_average.csv"

    if not domestic_path.exists() or not oil_path.exists():
        return None

    domestic = pd.read_csv(domestic_path)
    oil = pd.read_csv(oil_path)

    domestic["date"] = pd.to_datetime(domestic["date"])
    domestic["notice_date"] = pd.to_datetime(domestic["notice_date"])
    oil["date"] = pd.to_datetime(oil["date"])
    oil["notice_date"] = pd.to_datetime(oil["notice_date"])

    merged = domestic.merge(
        oil[["date", "notice_date", "wti_mean", "brent_mean", "basket_mean"]],
        on=["date", "notice_date"],
        how="left",
    )

    exchange = read_exchange_rate()
    merged = pd.merge_asof(
        merged.sort_values("date"),
        exchange,
        on="date",
        direction="backward",
    )

    return merged


def load_fallback_data():
    """
    如果没有合并好的特征表，则根据现有 result/data 文件尽量自动合并。
    """
    domestic_candidates = [
        RESULT_DIR / "domestic_events_clean.csv",
        RESULT_DIR / "domastic_price.csv",
        DATA_DIR / "rare-domastic.csv",
    ]
    domestic_path = next((p for p in domestic_candidates if p.exists()), None)
    if domestic_path is None:
        raise ValueError("找不到国内成品油调价数据")

    domestic = pd.read_csv(domestic_path)
    if "effective_date" in domestic.columns:
        domestic["date"] = pd.to_datetime(domestic["effective_date"])
    else:
        date_col = find_column(domestic, ["date", "adjust_date"], label="domestic date")
        domestic["date"] = pd.to_datetime(domestic[date_col])

    rename_map = {}
    for old, new in [
        ("gasoline_adjust_cny_per_ton", "gasoline_change"),
        ("diesel_adjust_cny_per_ton", "diesel_change"),
        ("beijing_gasoline_ceiling_after_cny_per_ton", "gasoline_price_after"),
        ("beijing_diesel_ceiling_after_cny_per_ton", "diesel_price_after"),
    ]:
        if old in domestic.columns and new not in domestic.columns:
            rename_map[old] = new
    domestic = domestic.rename(columns=rename_map)

    oil_path = RESULT_DIR / "oil_10_workday_average.csv"
    if oil_path.exists():
        oil = pd.read_csv(oil_path)
        oil["date"] = pd.to_datetime(oil["date"])
        merged = domestic.merge(
            oil[["date", "wti_mean", "brent_mean", "basket_mean"]],
            on="date",
            how="left",
        )
    else:
        raise ValueError("找不到国际油价窗口均价数据，请先运行 prepare_clean_data.py")

    exchange = read_exchange_rate()
    merged = pd.merge_asof(
        merged.sort_values("date"),
        exchange,
        on="date",
        direction="backward",
    )
    return merged


def load_data():
    """
    读取并标准化完整特征数据。
    """
    files = scan_data_files()
    print("已扫描到数据/结果文件数量:", len(files))

    df = load_preferred_merged_data()
    source = "result/domestic_events_clean.csv + result/oil_10_workday_average.csv"
    if df is None:
        df = load_fallback_data()
        source = "fallback merge"

    columns = identify_columns(df)

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[columns["date"]])
    out["gasoline_change"] = pd.to_numeric(df[columns["gasoline_change"]], errors="coerce")
    out["diesel_change"] = pd.to_numeric(df[columns["diesel_change"]], errors="coerce")
    out["wti"] = pd.to_numeric(df[columns["wti"]], errors="coerce")
    out["brent"] = pd.to_numeric(df[columns["brent"]], errors="coerce")
    out["basket"] = pd.to_numeric(df[columns["basket"]], errors="coerce")
    out["exchange_rate"] = pd.to_numeric(df[columns["exchange_rate"]], errors="coerce")

    if "notice_date" in df.columns:
        out["notice_date"] = pd.to_datetime(df["notice_date"])
    else:
        out["notice_date"] = out["date"]

    if "special_type" in df.columns:
        out["special_type"] = df["special_type"].astype(str)
    else:
        out["special_type"] = "unknown"

    special_col = columns["is_special_regulated"]
    if special_col is None:
        out["is_special_regulated"] = out["special_type"].eq("temporary_control")
    else:
        out["is_special_regulated"] = (
            df[special_col].astype(str).str.upper().isin(["TRUE", "1", "YES"])
        )

    out = out.dropna(subset=[
        "date",
        "gasoline_change",
        "diesel_change",
        "wti",
        "brent",
        "basket",
        "exchange_rate",
    ])
    out = out.sort_values("date").reset_index(drop=True)

    # 机制切换样本不算特殊调控，但不适合训练。
    out["is_mechanism_reform"] = out["special_type"].eq("mechanism_reform")

    meta = {
        "source": source,
        "files": files,
        "columns": columns,
    }
    return out, meta


def build_features(df, fuel_type):
    """
    构造回归特征。

    X1 = mu * nu * WTI
    X2 = mu * nu * Brent
    X3 = mu * nu * Basket

    回归使用相邻窗口差分 Delta_X。
    """
    if fuel_type == "gasoline":
        y_col = "gasoline_change"
    elif fuel_type == "diesel":
        y_col = "diesel_change"
    else:
        raise ValueError("fuel_type 必须是 gasoline 或 diesel")

    out = df.copy()
    out["actual_delta"] = out[y_col]
    out["X1"] = out["exchange_rate"] * NU * out["wti"]
    out["X2"] = out["exchange_rate"] * NU * out["brent"]
    out["X3"] = out["exchange_rate"] * NU * out["basket"]

    for col in ["X1", "X2", "X3"]:
        out["delta_" + col] = out[col].diff()

    out["prev_actual_delta"] = out["actual_delta"].shift(1)
    out["prev_is_special_regulated"] = out["is_special_regulated"].shift(1).fillna(False)
    out["prev_is_mechanism_reform"] = out["is_mechanism_reform"].shift(1).fillna(False)

    return out


def build_clean_sample(feature_df, fuel_type):
    """
    构造干净样本。

    条件：
    1. 当前不是特殊调控；
    2. 上一期不是特殊调控；
    3. 上一期实际发生正常调价；
    4. 当前不是搁浅样本；
    5. 当前和上一期都不是机制切换样本；
    6. 关键字段不缺失。
    """
    clean = feature_df[
        (feature_df["is_special_regulated"] == False)
        & (feature_df["prev_is_special_regulated"] == False)
        & (feature_df["is_mechanism_reform"] == False)
        & (feature_df["prev_is_mechanism_reform"] == False)
        & (feature_df["prev_actual_delta"].abs() >= THRESHOLD)
        & (feature_df["actual_delta"].abs() >= THRESHOLD)
    ].copy()

    clean = clean.dropna(subset=["delta_X1", "delta_X2", "delta_X3", "actual_delta"])
    return clean


def constrained_lstsq(A, y):
    """
    最小二乘估计，约束 gamma1、gamma2、gamma3 非负，截距不限。

    优先使用 scipy.optimize.lsq_linear。
    如果环境没有 scipy，则枚举 gamma 的活动集做 fallback。
    """
    try:
        from scipy.optimize import lsq_linear

        lower = np.array([0.0, 0.0, 0.0, -np.inf])
        upper = np.array([np.inf, np.inf, np.inf, np.inf])
        result = lsq_linear(A, y, bounds=(lower, upper), lsmr_tol="auto")
        return result.x
    except Exception:
        pass

    best_coef = None
    best_sse = None

    # active 表示哪些 gamma 可以自由估计；未激活的 gamma 固定为 0。
    for mask in range(1, 8):
        active_gamma = [i for i in range(3) if (mask >> i) & 1]
        cols = active_gamma + [3]  # 截距始终保留
        coef_sub, _, _, _ = np.linalg.lstsq(A[:, cols], y, rcond=None)

        coef = np.zeros(4)
        coef[cols] = coef_sub

        if np.any(coef[:3] < -1e-8):
            continue

        pred = A @ coef
        sse = float(np.sum((pred - y) ** 2))
        if best_sse is None or sse < best_sse:
            best_sse = sse
            best_coef = coef

    # 所有 gamma 都为 0，只估截距，也作为备选。
    coef = np.array([0.0, 0.0, 0.0, float(np.mean(y))])
    sse = float(np.sum((A @ coef - y) ** 2))
    if best_sse is None or sse < best_sse:
        best_coef = coef

    return best_coef


def regression_metrics(y, pred, k):
    """
    计算 MAE、RMSE、R2。
    """
    residual = pred - y
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > EPS else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "n": len(y), "k": k}


def recover_weights(params):
    """
    根据 gamma 恢复 alpha 和 w1,w2,w3。
    """
    gamma_sum = params["gamma1"] + params["gamma2"] + params["gamma3"]
    params["alpha"] = gamma_sum

    if gamma_sum > EPS:
        params["w1_wti"] = params["gamma1"] / gamma_sum
        params["w2_brent"] = params["gamma2"] / gamma_sum
        params["w3_basket"] = params["gamma3"] / gamma_sum
    else:
        params["w1_wti"] = np.nan
        params["w2_brent"] = np.nan
        params["w3_basket"] = np.nan

    return params


def fit_nonnegative_regression(clean_df, fuel_type):
    """
    对干净样本做非负约束回归。
    """
    A = clean_df[["delta_X1", "delta_X2", "delta_X3"]].to_numpy(dtype=float)
    y = clean_df["actual_delta"].to_numpy(dtype=float)

    # 有截距模型：A = [Delta_X1, Delta_X2, Delta_X3, 1]
    A_with_intercept = np.column_stack([A, np.ones(len(clean_df))])
    coef = constrained_lstsq(A_with_intercept, y)
    pred = A_with_intercept @ coef

    params = {
        "fuel_type": fuel_type,
        "gamma1": float(coef[0]),
        "gamma2": float(coef[1]),
        "gamma3": float(coef[2]),
        "intercept": float(coef[3]),
    }
    params = recover_weights(params)
    params.update(regression_metrics(y, pred, k=4))

    return params


def save_parameter_result(params, fuel_type):
    """
    保存单个油品的参数估计结果。
    """
    output = RESULT_DIR / f"parameter_estimation_{fuel_type}.csv"
    pd.DataFrame([params]).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def backtest_full_sequence(feature_df, params, fuel_type):
    """
    将估计出的参数放回完整时序模型中验证。

    这里使用差分模型：
        raw_rule_delta = gamma1*dX1 + gamma2*dX2 + gamma3*dX3 + intercept

    然后加入 50 元门槛和累计未调幅度。
    """
    rows = []
    carry = 0.0

    for _, row in feature_df.iterrows():
        if pd.isna(row["delta_X1"]) or pd.isna(row["delta_X2"]) or pd.isna(row["delta_X3"]):
            raw_rule_delta = np.nan
            cumulative_rule_delta = np.nan
            pred_delta_no_special = np.nan
            carry_before = carry
            carry_after = carry
        else:
            raw_rule_delta = (
                params["gamma1"] * row["delta_X1"]
                + params["gamma2"] * row["delta_X2"]
                + params["gamma3"] * row["delta_X3"]
                + params["intercept"]
            )

            carry_before = carry
            cumulative_rule_delta = carry_before + raw_rule_delta

            if abs(cumulative_rule_delta) >= THRESHOLD:
                pred_delta_no_special = cumulative_rule_delta
                carry = 0.0
            else:
                pred_delta_no_special = 0.0
                carry = cumulative_rule_delta
            carry_after = carry

        rows.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "actual_delta": row["actual_delta"],
            "raw_rule_delta": raw_rule_delta,
            "rule_delta": cumulative_rule_delta,
            "pred_delta_no_special": pred_delta_no_special,
            "carry_before": carry_before,
            "carry_after": carry_after,
            "is_special_regulated": row["is_special_regulated"],
            "special_type": row["special_type"],
            "abs_error": abs(pred_delta_no_special - row["actual_delta"])
            if not pd.isna(pred_delta_no_special)
            else np.nan,
        })

    backtest = pd.DataFrame(rows)
    return backtest


def backtest_metrics(backtest_df):
    """
    计算完整回代误差。
    """
    valid = backtest_df.dropna(subset=["abs_error"]).copy()
    normal = valid[valid["is_special_regulated"] == False]
    special = valid[valid["is_special_regulated"] == True]

    return {
        "normal_sample_MAE": float(normal["abs_error"].mean()) if len(normal) else np.nan,
        "all_sample_MAE": float(valid["abs_error"].mean()) if len(valid) else np.nan,
        "special_sample_MAE": float(special["abs_error"].mean()) if len(special) else np.nan,
    }


def compute_special_theta(backtest_df, fuel_type):
    """
    对特殊调控样本反推 theta_i。
    """
    special = backtest_df[backtest_df["is_special_regulated"] == True].copy()
    rows = []

    for _, row in special.iterrows():
        rule_delta = row["rule_delta"]
        actual_delta = row["actual_delta"]

        if pd.isna(rule_delta):
            theta = np.nan
            valid = False
            reason = "rule_delta 缺失"
        elif abs(rule_delta) < THRESHOLD:
            theta = np.nan
            valid = False
            reason = "abs(rule_delta) < 50，theta 不稳定"
        elif abs(rule_delta) < EPS:
            theta = np.nan
            valid = False
            reason = "rule_delta 接近 0"
        else:
            theta = actual_delta / rule_delta
            valid = True
            reason = ""

        rows.append({
            "date": row["date"],
            "fuel_type": fuel_type,
            "actual_delta": actual_delta,
            "rule_delta": rule_delta,
            "theta_i": theta,
            "valid_theta": valid,
            "reason_if_invalid": reason,
        })

    theta_df = pd.DataFrame(rows)
    return theta_df


def theta_summary(theta_gasoline, theta_diesel):
    """
    统计 theta_i 是否可近似为常数。
    """
    rows = []
    for fuel_type, theta_df in [("gasoline", theta_gasoline), ("diesel", theta_diesel)]:
        valid = theta_df[theta_df["valid_theta"] == True].copy()
        if len(valid) == 0:
            mean = std = cv = np.nan
            judgement = "无有效 theta，无法判断"
        else:
            mean = float(valid["theta_i"].mean())
            std = float(valid["theta_i"].std(ddof=0))
            cv = std / abs(mean) if abs(mean) > EPS else np.inf

            if cv < 0.05:
                judgement = "theta_i 高度稳定，可以近似为常数"
            elif cv < 0.10:
                judgement = "theta_i 基本稳定，可以近似为常数"
            else:
                judgement = "theta_i 波动较大，不建议设为常数"

        rows.append({
            "fuel_type": fuel_type,
            "valid_n": len(valid),
            "theta_mean": mean,
            "theta_std": std,
            "theta_cv": cv,
            "judgement": judgement,
        })

    # 如果同日期汽油和柴油 theta 接近，计算综合 theta。
    if len(theta_gasoline) and len(theta_diesel):
        merged = theta_gasoline.merge(
            theta_diesel,
            on="date",
            suffixes=("_gasoline", "_diesel"),
        )
        merged = merged[
            (merged["valid_theta_gasoline"] == True)
            & (merged["valid_theta_diesel"] == True)
        ].copy()
        if len(merged):
            merged["theta_i"] = (merged["theta_i_gasoline"] + merged["theta_i_diesel"]) / 2
            mean = float(merged["theta_i"].mean())
            std = float(merged["theta_i"].std(ddof=0))
            cv = std / abs(mean) if abs(mean) > EPS else np.inf

            if cv < 0.05:
                judgement = "综合 theta_i 高度稳定，可以近似为常数"
            elif cv < 0.10:
                judgement = "综合 theta_i 基本稳定，可以近似为常数"
            else:
                judgement = "综合 theta_i 波动较大，不建议设为常数"

            rows.append({
                "fuel_type": "combined",
                "valid_n": len(merged),
                "theta_mean": mean,
                "theta_std": std,
                "theta_cv": cv,
                "judgement": judgement,
            })

    return pd.DataFrame(rows)


def write_backtest_output(backtest_df, fuel_type):
    """
    保存完整时序回代结果。
    """
    output = RESULT_DIR / f"backtest_{fuel_type}.csv"
    backtest_df.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def write_theta_output(theta_df, fuel_type):
    """
    保存特殊调控 theta 结果。
    """
    output = RESULT_DIR / f"theta_special_{fuel_type}.csv"
    theta_df.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def write_report(meta, results):
    """
    生成 Markdown 报告。
    """
    report_path = RESULT_DIR / "parameter_estimation_report.md"

    gas_params = results["gasoline"]["params"]
    diesel_params = results["diesel"]["params"]
    gas_metrics = results["gasoline"]["backtest_metrics"]
    diesel_metrics = results["diesel"]["backtest_metrics"]
    theta_sum = results["theta_summary"]

    lines = []
    lines.append("# 参数估计报告")
    lines.append("")
    lines.append("## 1. 数据文件来源和使用字段")
    lines.append("")
    lines.append(f"- 主数据来源：`{meta['source']}`")
    lines.append("- 已扫描文件：")
    for path in meta["files"]:
        lines.append(f"  - `{path.relative_to(ROOT)}`")
    lines.append("")
    lines.append("字段识别结果：")
    for key, value in meta["columns"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    lines.append("## 2. 干净样本筛选规则")
    lines.append("")
    lines.append("clean_sample 同时满足：当前样本非特殊调控、上一期非特殊调控、上一期实际调价幅度不小于 50 元/吨、当前实际调价幅度不小于 50 元/吨、当前和上一期均非机制切换样本。")
    lines.append("")

    lines.append("## 3. 回归模型公式")
    lines.append("")
    lines.append("令 `X1_t = mu_t * nu * WTI_t`，`X2_t = mu_t * nu * Brent_t`，`X3_t = mu_t * nu * Basket_t`。")
    lines.append("")
    lines.append("拟合调价幅度模型：")
    lines.append("")
    lines.append("```text")
    lines.append("Delta_P_t = gamma1 * Delta_X1_t + gamma2 * Delta_X2_t + gamma3 * Delta_X3_t + intercept")
    lines.append("```")
    lines.append("")
    lines.append("并令：")
    lines.append("")
    lines.append("```text")
    lines.append("alpha = gamma1 + gamma2 + gamma3")
    lines.append("w_i = gamma_i / alpha")
    lines.append("```")
    lines.append("")

    def add_param_section(title, params):
        lines.append(f"## {title}")
        lines.append("")
        for key in [
            "gamma1",
            "gamma2",
            "gamma3",
            "alpha",
            "w1_wti",
            "w2_brent",
            "w3_basket",
            "intercept",
            "MAE",
            "RMSE",
            "R2",
            "n",
        ]:
            lines.append(f"- {key}: {params.get(key)}")
        lines.append("")

    add_param_section("4. 汽油参数估计结果", gas_params)
    add_param_section("5. 柴油参数估计结果", diesel_params)

    lines.append("## 6. 完整时序回代误差")
    lines.append("")
    lines.append("汽油：")
    for key, value in gas_metrics.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("柴油：")
    for key, value in diesel_metrics.items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## 7. 特殊调控 theta_i 计算结果")
    lines.append("")
    lines.append("详见：")
    lines.append("- `result/theta_special_gasoline.csv`")
    lines.append("- `result/theta_special_diesel.csv`")
    lines.append("")

    lines.append("## 8. theta_i 是否近似常数")
    lines.append("")
    lines.extend(dataframe_to_markdown(theta_sum))
    lines.append("")

    lines.append("## 9. 如果模型效果仍然较差，可能原因")
    lines.append("")
    lines.append("- 调价窗口是否严格使用了 10 个工作日，而不是 10 个自然日。")
    lines.append("- 国内调价日期是否区分公告日和生效日。")
    lines.append("- 特殊调控标记是否准确。")
    lines.append("- 40、80、130 美元分段利润函数是否应纳入差分模型。")
    lines.append("- 累计未调幅度是否按政策逻辑正确递推。")
    lines.append("- 官方一揽子油种权重和部分成本项并未公开，公开数据只能近似识别。")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def dataframe_to_markdown(df):
    """
    不依赖 tabulate，手写一个简单 Markdown 表格。
    """
    if len(df) == 0:
        return ["无数据"]

    columns = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    return lines


def run_for_fuel(df, fuel_type):
    """
    对一个油品完成：构造特征、干净样本、回归、回代、theta。
    """
    feature_df = build_features(df, fuel_type)
    clean_df = build_clean_sample(feature_df, fuel_type)

    if len(clean_df) < 5:
        raise ValueError(f"{fuel_type} 干净样本过少，无法回归。样本数: {len(clean_df)}")

    params = fit_nonnegative_regression(clean_df, fuel_type)
    param_output = save_parameter_result(params, fuel_type)

    backtest_df = backtest_full_sequence(feature_df, params, fuel_type)
    backtest_output = write_backtest_output(backtest_df, fuel_type)
    metrics = backtest_metrics(backtest_df)

    theta_df = compute_special_theta(backtest_df, fuel_type)
    theta_output = write_theta_output(theta_df, fuel_type)

    return {
        "feature_df": feature_df,
        "clean_df": clean_df,
        "params": params,
        "param_output": param_output,
        "backtest_df": backtest_df,
        "backtest_output": backtest_output,
        "backtest_metrics": metrics,
        "theta_df": theta_df,
        "theta_output": theta_output,
    }


def main():
    """
    主流程。
    """
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    df, meta = load_data()

    results = {
        "gasoline": run_for_fuel(df, "gasoline"),
        "diesel": run_for_fuel(df, "diesel"),
    }

    theta_sum = theta_summary(
        results["gasoline"]["theta_df"],
        results["diesel"]["theta_df"],
    )
    theta_summary_output = RESULT_DIR / "theta_summary.csv"
    theta_sum.to_csv(theta_summary_output, index=False, encoding="utf-8-sig")
    results["theta_summary"] = theta_sum

    report_path = write_report(meta, results)

    print("参数估计流程完成")
    print("汽油参数:", results["gasoline"]["param_output"].resolve())
    print("柴油参数:", results["diesel"]["param_output"].resolve())
    print("汽油回代:", results["gasoline"]["backtest_output"].resolve())
    print("柴油回代:", results["diesel"]["backtest_output"].resolve())
    print("theta 汇总:", theta_summary_output.resolve())
    print("报告:", report_path.resolve())
    print()
    print("汽油参数估计:")
    print(pd.DataFrame([results["gasoline"]["params"]]).to_string(index=False))
    print()
    print("柴油参数估计:")
    print(pd.DataFrame([results["diesel"]["params"]]).to_string(index=False))
    print()
    print("theta 稳定性:")
    print(theta_sum.to_string(index=False))


if __name__ == "__main__":
    main()
