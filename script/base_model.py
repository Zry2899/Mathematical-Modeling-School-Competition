"""
成品油价格调控机制 base 模型。

这个脚本的目标不是一次性把参数估到最优，而是先把数学模型中的
“计算流程”搭起来：

1. 读取国内成品油调价数据；
2. 读取 WTI、Brent、Basket 三种国际油价的 10 天均价；
3. 读取人民币兑美元汇率；
4. 根据给定参数计算理论价格；
5. 根据 50 元/吨门槛和累计未调幅度，得到模型预测调价幅度；
6. 与真实调价幅度比较，输出误差表。

后面讨论参数求解时，主要改 PARAMS 里的参数，或者写一个搜索函数反复调用 run_model。
"""

import argparse
from pathlib import Path

import pandas as pd


# =====================================================
# 1. 路径设置
# =====================================================

# 当前文件在 script 文件夹中，所以 parents[1] 是项目根目录
ROOT = Path(__file__).resolve().parents[1]

# 基础数据目录
DATA_DIR = ROOT / "data"

# 处理结果目录
RESULT_DIR = ROOT / "result"


# =====================================================
# 2. 默认模型参数
# =====================================================

# 这里先给一组能跑通流程的默认参数。
# 注意：这些参数还没有经过拟合，所以预测误差大是正常的。
PARAMS = {
    # 三种国际油价的权重，要求三者相加等于 1
    "w_wti": 1 / 3,
    "w_brent": 1 / 3,
    "w_basket": 1 / 3,

    # 国际原油成本向国内成品油价格的传导系数
    "alpha": 1.0,

    # 固定项：税费、加工成本、流通费用等
    # 如果设为 None，脚本会用第一行数据自动反推一个 beta，使第一期理论价格贴近实际价格
    "beta": None,

    # 正常加工利润
    "pi0": 600.0,

    # 高于 130 美元/桶之后的边际传导比例
    # lambda_above_130 = alpha * lambda_factor
    "lambda_factor": 0.30,

    # 特殊调控时期的调控系数
    # 先设为 1，表示暂时不额外削弱或放大调价幅度
    "theta_special": 1.0,

    # 桶到吨换算系数，约 1 吨 = 7.33 桶
    "barrel_per_ton": 7.33,

    # 国内机制中的地板油价，低于 40 美元/桶时按 40 计算
    "oil_floor": 40.0,

    # 调价门槛，理论调价幅度不足 50 元/吨时暂不调整
    "rule_threshold": 50.0,
}


# =====================================================
# 3. 数据读取函数
# =====================================================

def read_average_price(oil_name, include_adjust_day):
    """
    读取某一种国际油价的 10 天均价结果。

    参数：
    oil_name:
        字符串，可取 "wti"、"brent"、"basket"。

    include_adjust_day:
        True 表示读取包含调价当天的窗口结果；
        False 表示读取不包含调价当天的窗口结果。

    返回：
        只保留 date 和 mean_price 两列，并把 mean_price 改名为
        wti_mean、brent_mean 或 basket_mean，方便后面合并。
    """
    if include_adjust_day:
        suffix = "Incl"
    else:
        suffix = "NIncl"

    file_path = RESULT_DIR / f"{oil_name}-average-{suffix}_detail.csv"

    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])

    new_price_name = oil_name + "_mean"
    df = df[["date", "mean_price"]].rename(columns={"mean_price": new_price_name})

    return df


def read_exchange_rate():
    """
    读取人民币兑美元汇率。

    原始文件中列名包含中文，终端有时会显示乱码。
    为了让脚本更稳，这里不依赖具体列名，而是直接把前两列改成：
    date 和 exchange_rate。
    """
    file_path = DATA_DIR / "cny_usd_exchange_rate.csv"

    df = pd.read_csv(file_path)
    df = df.iloc[:, 0:2]
    df.columns = ["date", "exchange_rate"]

    df["date"] = pd.to_datetime(df["date"])
    df["exchange_rate"] = pd.to_numeric(df["exchange_rate"], errors="coerce")

    # 删除日期或汇率缺失的行，并按日期排序
    df = df.dropna(subset=["date", "exchange_rate"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def load_model_data(include_adjust_day=True):
    """
    读取并合并 base 模型所需的所有数据。

    合并后的每一行对应一次国内成品油调价窗口，包含：
    - 真实汽油调价幅度；
    - 真实调价后的汽油价格；
    - WTI、Brent、Basket 的 10 天均价；
    - 当日或最近可得的人民币兑美元汇率；
    - 是否属于特殊调控期。
    """
    domestic_path = RESULT_DIR / "domastic_price.csv"
    domestic = pd.read_csv(domestic_path)

    # 统一日期格式，方便后面按 date 合并
    domestic["date"] = pd.to_datetime(domestic["date"])
    domestic = domestic.sort_values("date").reset_index(drop=True)

    # 依次合并三种国际油价均值
    data = domestic
    data = data.merge(read_average_price("wti", include_adjust_day), on="date", how="left")
    data = data.merge(read_average_price("brent", include_adjust_day), on="date", how="left")
    data = data.merge(read_average_price("basket", include_adjust_day), on="date", how="left")

    # 合并汇率。
    # merge_asof 表示：如果调价当天没有汇率，就使用之前最近一天的汇率。
    exchange = read_exchange_rate()
    data = pd.merge_asof(
        data.sort_values("date"),
        exchange,
        on="date",
        direction="backward",
    )

    # 把后面要计算的列全部转成数值，避免字符串参与计算
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

    # 把 TRUE/FALSE 字符串转成布尔值
    data["is_special_regulated"] = (
        data["is_special_regulated"].astype(str).str.upper() == "TRUE"
    )

    # 如果某一行缺少油价或汇率，就暂时删掉，保证模型能正常计算
    data = data.dropna(subset=["wti_mean", "brent_mean", "basket_mean", "exchange_rate"])

    return data


# =====================================================
# 4. 模型公式函数
# =====================================================

def check_weights(params):
    """
    检查 WTI、Brent、Basket 三个权重是否相加为 1。

    如果不为 1，说明参数设置有问题，直接报错。
    """
    weight_sum = params["w_wti"] + params["w_brent"] + params["w_basket"]

    if abs(weight_sum - 1.0) > 0.000001:
        raise ValueError("三种油价权重之和必须等于 1，当前为：" + str(weight_sum))


def calculate_weighted_oil_mean(row, params):
    """
    计算三种国际油价的加权均价，也就是“过滤前”的均价。

    公式：
        O_t = w1 * WTI_t + w2 * Brent_t + w3 * Basket_t

    注意：
        这一步只计算均价，不做 40 美元/桶地板价过滤。
        这样可以在输出结果中同时看到“原始均价”和“过滤后油价”。
    """
    weighted_oil = (
        params["w_wti"] * row["wti_mean"]
        + params["w_brent"] * row["brent_mean"]
        + params["w_basket"] * row["basket_mean"]
    )

    return weighted_oil


def apply_oil_floor(weighted_oil_mean, params):
    """
    对加权均价执行 40 美元/桶地板价过滤。

    Word/PDF 中先定义 10 天均价，再定义过滤后的数值：
        filtered_oil_price = max(40, weighted_oil_mean)

    后续理论价格公式使用过滤后的油价。
    但为了检查模型，我们仍然保留过滤前的 weighted_oil_mean。
    """
    if weighted_oil_mean < params["oil_floor"]:
        return params["oil_floor"]

    return weighted_oil_mean


def calculate_profit(oil_price, params):
    """
    计算加工利润函数 pi(x)。

    规则来自 Word 中的模型：
    - x <= 80 时，利润为正常加工利润 pi0；
    - 80 < x <= 130 时，利润从 pi0 线性下降到 0；
    - x > 130 时，利润为 0。
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

    oil_price:
        国际油价加权均价，单位是美元/桶。

    exchange_rate:
        人民币兑美元汇率，单位是元/美元。

    beta:
        固定项，代表税费、加工成本、流通费用等。

    返回：
        理论价格，单位是元/吨。
    """
    alpha = params["alpha"]
    barrel_per_ton = params["barrel_per_ton"]

    # 把美元/桶转换成人民币/吨：
    # 美元/桶 * 元/美元 * 桶/吨 = 元/吨
    oil_cost = oil_price * exchange_rate * barrel_per_ton

    # 130 美元/桶对应的人民币/吨成本
    cap_cost = 130 * exchange_rate * barrel_per_ton

    if oil_price < 130:
        profit = calculate_profit(oil_price, params)
        theory_price = alpha * oil_cost + beta + profit
    else:
        # 高于 130 美元/桶之后，超出部分只按较低的边际传导系数传导
        lambda_above_130 = alpha * params["lambda_factor"]
        extra_cost = (oil_price - 130) * exchange_rate * barrel_per_ton
        theory_price = alpha * cap_cost + beta + lambda_above_130 * extra_cost

    return theory_price


def infer_beta_from_first_row(data, params):
    """
    如果没有手动给 beta，就用第一行数据反推 beta。

    这样做的目的：
    - base 模型刚开始还没有估计参数；
    - 如果 beta 随便设，理论价格水平可能整体偏高或偏低；
    - 用第一期实际价格校准 beta，可以让模型先站在一个合理价格水平上。
    """
    first_row = data.iloc[0]

    weighted_oil_mean = calculate_weighted_oil_mean(first_row, params)
    filtered_oil_price = apply_oil_floor(weighted_oil_mean, params)
    exchange_rate = first_row["exchange_rate"]

    # 第一行的“上一期实际价格” = 本期调价后价格 - 本期调价幅度
    previous_actual_price = first_row["gasoline_price_after"] - first_row["gasoline_change"]

    # 先假设 beta = 0，算出除 beta 之外的理论价格部分
    theory_without_beta = calculate_theory_price(
        oil_price=filtered_oil_price,
        exchange_rate=exchange_rate,
        beta=0,
        params=params,
    )

    # 让第一期理论价格等于上一期实际价格，由此反推 beta
    beta = previous_actual_price - theory_without_beta

    return beta


# =====================================================
# 5. 主模型计算函数
# =====================================================

def run_model(data, params):
    """
    对所有调价窗口逐行运行 base 模型。

    每一行的计算步骤：
    1. 计算三种国际油价的加权均价；
    2. 计算理论成品油价格；
    3. 用理论价格减去上一期实际价格，得到理论调价幅度；
    4. 加上上一轮没有调整的累计幅度；
    5. 判断是否达到 50 元/吨调价门槛；
    6. 如果是特殊调控期，再乘以特殊调控系数；
    7. 得到预测调价幅度和预测调价后价格；
    8. 与真实调价幅度、真实调价后价格比较。
    """
    check_weights(params)

    # 如果 beta 没有手动设置，就用第一行数据自动反推
    if params["beta"] is None:
        beta = infer_beta_from_first_row(data, params)
    else:
        beta = params["beta"]

    # carry 表示累计未调整幅度，也就是 Word 模型中的 A_t
    carry = 0.0

    # previous_actual_price 表示上一期真实调价后的价格
    previous_actual_price = None

    # 用普通列表收集每一期的计算结果，最后再转成 DataFrame
    result_rows = []

    for _, row in data.iterrows():
        actual_change = row["gasoline_change"]
        actual_after = row["gasoline_price_after"]

        # 第一行没有上一期价格，所以用“本期调价后价格 - 本期调价幅度”反推
        if previous_actual_price is None:
            previous_actual_price = actual_after - actual_change

        # 第一步：三种国际油价加权，得到过滤前的 10 天加权均价
        weighted_oil_mean = calculate_weighted_oil_mean(row, params)

        # 第二步：对加权均价应用 40 美元/桶地板价，得到进入价格公式的油价
        filtered_oil_price = apply_oil_floor(weighted_oil_mean, params)

        # 第三步：计算理论价格
        theory_price = calculate_theory_price(
            oil_price=filtered_oil_price,
            exchange_rate=row["exchange_rate"],
            beta=beta,
            params=params,
        )

        # 第四步：理论调价幅度 = 理论价格 - 上一期实际价格
        theory_change = theory_price - previous_actual_price

        # 第五步：叠加上一轮累计未调幅度
        stacked_change = theory_change + carry

        # 第六步：执行 50 元/吨门槛规则
        # 题目背景表述为“调价幅度低于每吨 50 元时，不作调整”。
        # 这里的“幅度”应理解为绝对幅度，所以不论上调还是下调，都用 abs(stacked_change) 判断。
        if abs(stacked_change) < params["rule_threshold"]:
            # 不足 50 元/吨，不调价，幅度累计到下一轮
            rule_change = 0.0
            carry = stacked_change
        else:
            # 达到 50 元/吨，本轮调价，累计幅度清零
            rule_change = stacked_change
            carry = 0.0

        # 第七步：特殊调控期乘以 theta_special
        if row["is_special_regulated"]:
            theta = params["theta_special"]
        else:
            theta = 1.0

        predicted_change = theta * rule_change

        # 第八步：模型预测的调价后价格
        predicted_after = previous_actual_price + predicted_change

        # 第九步：保存本轮结果
        result_rows.append({
            "date": row["date"].strftime("%Y-%m-%d"),
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
            "is_special_regulated": row["is_special_regulated"],
            "beta_used": round(beta, 4),
        })

        # 下一轮计算时，“上一期实际价格”使用真实价格，而不是模型预测价格
        # 这样做可以避免模型误差一路累积，便于先拟合单期调价幅度。
        previous_actual_price = actual_after

    result = pd.DataFrame(result_rows)

    return result


# =====================================================
# 6. 命令行参数
# =====================================================

def parse_args():
    """
    读取命令行参数。

    例如：
        python script/base_model.py --alpha 1.2 --pi0 500

    如果不传参数，就使用 PARAMS 中的默认值。
    """
    parser = argparse.ArgumentParser(description="成品油价格调控机制 base 模型")

    parser.add_argument("--exclude-adjust-day", action="store_true", help="使用不包含调价当天的 10 天均价")
    parser.add_argument("--output", default=str(RESULT_DIR / "base_model_predictions.csv"), help="输出文件路径")

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
    把命令行参数整理成一个普通字典。

    使用字典的好处是结构简单，后面讨论参数求解时也方便修改。
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


# =====================================================
# 7. 程序入口
# =====================================================

def main():
    """
    程序主入口。

    负责：
    1. 读取命令行参数；
    2. 读取并合并数据；
    3. 运行模型；
    4. 保存结果；
    5. 打印简单误差指标。
    """
    args = parse_args()
    params = build_params_from_args(args)

    # 是否包含调价当天：
    # 默认包含；如果命令行传入 --exclude-adjust-day，则不包含。
    include_adjust_day = not args.exclude_adjust_day

    data = load_model_data(include_adjust_day=include_adjust_day)
    result = run_model(data, params)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 误差指标：平均绝对误差
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
