"""
成品油价格调控机制 base 模型。

当前版本优先读取修正后的数据口径：
1. result/domestic_events_clean.csv
   国内调价事件表，区分公告日 notice_date 和生效日 date/effective_date。
2. result/oil_10_workday_average.csv
   按公告日前最近 10 个国际原油有效交易日计算的均价。

如果上述两个文件不存在，则回退到旧的 result/domastic_price.csv 和自然日均价结果。
"""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "result"


PARAMS = {
    "w_wti": 1 / 3,
    "w_brent": 1 / 3,
    "w_basket": 1 / 3,
    "alpha": 1.0,
    "beta": None,
    "pi0": 600.0,
    "lambda_factor": 0.30,
    "theta_special": 1.0,
    "barrel_per_ton": 7.33,
    "oil_floor": 40.0,
    "rule_threshold": 50.0,
}


def read_average_price(oil_name, include_adjust_day):
    """
    读取旧口径下某一种国际油价的窗口均价。

    这个函数只作为 fallback 使用。新口径会直接读取
    result/oil_10_workday_average.csv。
    """
    suffix = "Incl" if include_adjust_day else "NIncl"
    file_path = RESULT_DIR / f"{oil_name}-average-{suffix}_detail.csv"

    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])

    new_price_name = oil_name + "_mean"
    return df[["date", "mean_price"]].rename(columns={"mean_price": new_price_name})


def read_exchange_rate():
    """
    读取人民币兑美元汇率。

    汇率文件第二列是中文列名。为了避免终端编码影响，这里直接按列位置重命名。
    """
    file_path = DATA_DIR / "cny_usd_exchange_rate.csv"

    df = pd.read_csv(file_path)
    df = df.iloc[:, 0:2]
    df.columns = ["date", "exchange_rate"]

    df["date"] = pd.to_datetime(df["date"])
    df["exchange_rate"] = pd.to_numeric(df["exchange_rate"], errors="coerce")
    df = df.dropna(subset=["date", "exchange_rate"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def load_model_data(include_adjust_day=True):
    """
    读取并合并模型需要的数据。

    返回的每一行对应一次国内调价事件，包含：
    - 生效日 date；
    - 公告日 notice_date；
    - 国内真实调价幅度和调价后价格；
    - WTI、Brent、Basket 的窗口均价；
    - 当日或最近可得汇率；
    - 是否属于临时特殊调控。
    """
    clean_domestic_path = RESULT_DIR / "domestic_events_clean.csv"
    clean_average_path = RESULT_DIR / "oil_10_workday_average.csv"

    if clean_domestic_path.exists() and clean_average_path.exists():
        domestic = pd.read_csv(clean_domestic_path)
        averages = pd.read_csv(clean_average_path)

        domestic["date"] = pd.to_datetime(domestic["date"])
        domestic["notice_date"] = pd.to_datetime(domestic["notice_date"])
        averages["date"] = pd.to_datetime(averages["date"])
        averages["notice_date"] = pd.to_datetime(averages["notice_date"])

        average_columns = [
            "date",
            "notice_date",
            "wti_mean",
            "brent_mean",
            "basket_mean",
            "wti_valid_days",
            "brent_valid_days",
            "basket_valid_days",
        ]
        data = domestic.merge(averages[average_columns], on=["date", "notice_date"], how="left")
    else:
        domestic = pd.read_csv(RESULT_DIR / "domastic_price.csv")
        domestic["date"] = pd.to_datetime(domestic["date"])
        domestic = domestic.sort_values("date").reset_index(drop=True)

        data = domestic
        data = data.merge(read_average_price("wti", include_adjust_day), on="date", how="left")
        data = data.merge(read_average_price("brent", include_adjust_day), on="date", how="left")
        data = data.merge(read_average_price("basket", include_adjust_day), on="date", how="left")

    # 汇率按生效日向前匹配最近可得值。
    # 若后续要更严格，也可以改成按 notice_date 匹配。
    exchange = read_exchange_rate()
    data = pd.merge_asof(
        data.sort_values("date"),
        exchange,
        on="date",
        direction="backward",
    )

    number_columns = [
        "gasoline_change",
        "diesel_change",
        "gasoline_price_after",
        "diesel_price_after",
        "wti_mean",
        "brent_mean",
        "basket_mean",
        "exchange_rate",
    ]
    for col in number_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if "is_changed" not in data.columns:
        data["is_changed"] = (data["gasoline_change"] != 0) | (data["diesel_change"] != 0)
    else:
        data["is_changed"] = data["is_changed"].astype(str).str.upper().isin(["TRUE", "1"])

    if "is_special_regulated" not in data.columns:
        data["is_special_regulated"] = False
    else:
        data["is_special_regulated"] = (
            data["is_special_regulated"].astype(str).str.upper().isin(["TRUE", "1"])
        )

    if "special_type" not in data.columns:
        data["special_type"] = data["is_special_regulated"].map({
            True: "temporary_control",
            False: "unknown",
        })

    data = data.dropna(subset=["wti_mean", "brent_mean", "basket_mean", "exchange_rate"])
    data = data.sort_values("date").reset_index(drop=True)

    return data


def check_weights(params):
    """
    检查三种国际油价权重是否相加为 1。
    """
    weight_sum = params["w_wti"] + params["w_brent"] + params["w_basket"]
    if abs(weight_sum - 1.0) > 0.000001:
        raise ValueError("三种油价权重之和必须等于 1，当前为：" + str(weight_sum))


def calculate_weighted_oil_mean(row, params):
    """
    计算过滤前的三油种加权均价。
    """
    return (
        params["w_wti"] * row["wti_mean"]
        + params["w_brent"] * row["brent_mean"]
        + params["w_basket"] * row["basket_mean"]
    )


def apply_oil_floor(weighted_oil_mean, params):
    """
    应用 40 美元/桶地板价。
    """
    if weighted_oil_mean < params["oil_floor"]:
        return params["oil_floor"]
    return weighted_oil_mean


def calculate_profit(oil_price, params):
    """
    计算利润函数 pi(x)。
    """
    pi0 = params["pi0"]

    if oil_price <= 80:
        return pi0
    if oil_price <= 130:
        return pi0 * (130 - oil_price) / 50
    return 0.0


def calculate_theory_price(oil_price, exchange_rate, beta, params):
    """
    计算理论成品油价格。
    """
    alpha = params["alpha"]
    barrel_per_ton = params["barrel_per_ton"]

    oil_cost = oil_price * exchange_rate * barrel_per_ton
    cap_cost = 130 * exchange_rate * barrel_per_ton

    if oil_price < 130:
        return alpha * oil_cost + beta + calculate_profit(oil_price, params)

    lambda_above_130 = alpha * params["lambda_factor"]
    extra_cost = (oil_price - 130) * exchange_rate * barrel_per_ton
    return alpha * cap_cost + beta + lambda_above_130 * extra_cost


def infer_beta_from_first_row(data, params):
    """
    如果没有手动给 beta，就用第一行反推 beta。

    这只是 base 模型的兜底方式。正式拟合时建议显式给 beta。
    """
    first_row = data.iloc[0]

    weighted_oil_mean = calculate_weighted_oil_mean(first_row, params)
    filtered_oil_price = apply_oil_floor(weighted_oil_mean, params)
    previous_actual_price = first_row["gasoline_price_after"] - first_row["gasoline_change"]

    theory_without_beta = calculate_theory_price(
        oil_price=filtered_oil_price,
        exchange_rate=first_row["exchange_rate"],
        beta=0,
        params=params,
    )

    return previous_actual_price - theory_without_beta


def run_model(data, params):
    """
    给定参数，逐期运行规则模型。
    """
    check_weights(params)

    if params["beta"] is None:
        beta = infer_beta_from_first_row(data, params)
    else:
        beta = params["beta"]

    carry = 0.0
    previous_actual_price = None
    result_rows = []

    for _, row in data.iterrows():
        actual_change = row["gasoline_change"]
        actual_after = row["gasoline_price_after"]

        if previous_actual_price is None:
            previous_actual_price = actual_after - actual_change

        weighted_oil_mean = calculate_weighted_oil_mean(row, params)
        filtered_oil_price = apply_oil_floor(weighted_oil_mean, params)

        theory_price = calculate_theory_price(
            oil_price=filtered_oil_price,
            exchange_rate=row["exchange_rate"],
            beta=beta,
            params=params,
        )

        theory_change = theory_price - previous_actual_price
        stacked_change = theory_change + carry

        # 50 元/吨门槛按绝对幅度判断。
        if abs(stacked_change) < params["rule_threshold"]:
            rule_change = 0.0
            carry = stacked_change
        else:
            rule_change = stacked_change
            carry = 0.0

        if row["is_special_regulated"]:
            theta = params["theta_special"]
        else:
            theta = 1.0

        predicted_change = theta * rule_change
        predicted_after = previous_actual_price + predicted_change

        result_rows.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "notice_date": row["notice_date"].strftime("%Y-%m-%d") if "notice_date" in row else "",
            "special_type": row["special_type"] if "special_type" in row else "",
            "weighted_oil_mean": round(weighted_oil_mean, 4),
            "filtered_oil_price": round(filtered_oil_price, 4),
            "exchange_rate": round(row["exchange_rate"], 4),
            "previous_actual": round(previous_actual_price, 4),
            "theory_price": round(theory_price, 4),
            "theory_change": round(theory_change, 4),
            "carry_after_rule": round(carry, 4),
            "predicted_change": round(predicted_change, 4),
            "predicted_after": round(predicted_after, 4),
            "actual_change": round(actual_change, 4),
            "actual_after": round(actual_after, 4),
            "change_error": round(predicted_change - actual_change, 4),
            "price_error": round(predicted_after - actual_after, 4),
            "is_special_regulated": bool(row["is_special_regulated"]),
            "beta_used": round(beta, 4),
        })

        previous_actual_price = actual_after

    return pd.DataFrame(result_rows)


def parse_args():
    """
    读取命令行参数。
    """
    parser = argparse.ArgumentParser(description="成品油价格调控机制 base 模型")

    parser.add_argument("--exclude-adjust-day", action="store_true", help="旧口径下使用不包含调价当天的均价")
    parser.add_argument("--output", default=str(RESULT_DIR / "base_model_predictions.csv"))

    parser.add_argument("--w-wti", type=float, default=PARAMS["w_wti"])
    parser.add_argument("--w-brent", type=float, default=PARAMS["w_brent"])
    parser.add_argument("--w-basket", type=float, default=PARAMS["w_basket"])
    parser.add_argument("--alpha", type=float, default=PARAMS["alpha"])
    parser.add_argument("--beta", type=float, default=PARAMS["beta"])
    parser.add_argument("--pi0", type=float, default=PARAMS["pi0"])
    parser.add_argument("--lambda-factor", type=float, default=PARAMS["lambda_factor"])
    parser.add_argument("--theta-special", type=float, default=PARAMS["theta_special"])

    return parser.parse_args()


def build_params_from_args(args):
    """
    把命令行参数整理成参数字典。
    """
    params = PARAMS.copy()
    params["w_wti"] = args.w_wti
    params["w_brent"] = args.w_brent
    params["w_basket"] = args.w_basket
    params["alpha"] = args.alpha
    params["beta"] = args.beta
    params["pi0"] = args.pi0
    params["lambda_factor"] = args.lambda_factor
    params["theta_special"] = args.theta_special
    return params


def main():
    """
    程序入口。
    """
    args = parse_args()
    params = build_params_from_args(args)

    data = load_model_data(include_adjust_day=not args.exclude_adjust_day)
    result = run_model(data, params)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    change_mae = result["change_error"].abs().mean()
    price_mae = result["price_error"].abs().mean()

    print("模型运行完成")
    print("样本行数:", len(result))
    print("输出文件:", output_path.resolve())
    print("本次使用的 beta:", round(result["beta_used"].iloc[0], 4))
    print("调价幅度 MAE:", round(change_mae, 4))
    print("调价后价格 MAE:", round(price_mae, 4))
    print()
    print("前 10 行结果预览:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
