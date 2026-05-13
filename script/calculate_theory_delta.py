"""
计算全部样本的理论调价幅度 Delta P theory。

这里使用已经确定的推荐参数：

    w_wti    = 0.82
    w_brent  = 0.10
    w_basket = 0.08
    alpha    = 1.337214
    beta     = 1820.751832
    pi0      = 2533.074724
    lambda   = 0.0

本脚本只计算理论价格和理论调价幅度：

    Delta P theory = P theory - P actual(t-1)

然后把理论调价幅度与真实调价幅度进行比较。

注意：
这里不考虑 50 元/吨门槛，不考虑累计未调幅度，也不考虑特殊调控 theta。
目的只是先完整得到理论机制下每一期“应该变化多少”。
"""

import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import base_model


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"


def build_fixed_params():
    """
    构造已经确定的推荐参数。
    """
    params = base_model.PARAMS.copy()

    params["w_wti"] = 0.82
    params["w_brent"] = 0.10
    params["w_basket"] = 0.08

    params["alpha"] = 1.337214
    params["beta"] = 1820.751832
    params["pi0"] = 2533.074724

    # lambda = 0，因此 lambda_factor = lambda / alpha = 0
    params["lambda_factor"] = 0.0

    # 本脚本不使用 theta，只计算 Delta P theory
    params["theta_special"] = 1.0

    return params


def calculate_all_theory_delta():
    """
    对全部调价窗口计算理论价格和理论调价幅度。

    返回字段说明：
    - weighted_oil_mean: 过滤前的加权均价；
    - filtered_oil_price: 经过 40 美元地板价后的油价；
    - exchange_rate: 当日或最近可得汇率；
    - previous_actual: 上一期真实调价后价格；
    - theory_price: 理论价格；
    - delta_p_theory: 理论调价幅度；
    - actual_change: 真实调价幅度；
    - delta_error: 理论调价幅度 - 真实调价幅度。
    """
    params = build_fixed_params()
    data = base_model.load_model_data(include_adjust_day=True)
    data = data.sort_values("date").reset_index(drop=True)

    rows = []
    previous_actual = None

    for _, row in data.iterrows():
        actual_change = row["gasoline_change"]
        actual_after = row["gasoline_price_after"]

        # 第一行没有上一期价格，用“本期调价后价格 - 本期调价幅度”反推
        if previous_actual is None:
            previous_actual = actual_after - actual_change

        # 计算过滤前加权均价
        weighted_oil_mean = base_model.calculate_weighted_oil_mean(row, params)

        # 计算过滤后的油价
        filtered_oil_price = base_model.apply_oil_floor(weighted_oil_mean, params)

        # 计算理论价格
        theory_price = base_model.calculate_theory_price(
            oil_price=filtered_oil_price,
            exchange_rate=row["exchange_rate"],
            beta=params["beta"],
            params=params,
        )

        # 理论调价幅度
        delta_p_theory = theory_price - previous_actual

        rows.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "wti_mean": round(row["wti_mean"], 4),
            "brent_mean": round(row["brent_mean"], 4),
            "basket_mean": round(row["basket_mean"], 4),
            "weighted_oil_mean": round(weighted_oil_mean, 4),
            "filtered_oil_price": round(filtered_oil_price, 4),
            "exchange_rate": round(row["exchange_rate"], 4),
            "previous_actual": round(previous_actual, 4),
            "theory_price": round(theory_price, 4),
            "delta_p_theory": round(delta_p_theory, 4),
            "actual_change": round(actual_change, 4),
            "actual_after": round(actual_after, 4),
            "delta_error": round(delta_p_theory - actual_change, 4),
            "abs_delta_error": round(abs(delta_p_theory - actual_change), 4),
            "is_changed": row["is_changed"],
            "is_special_regulated": row["is_special_regulated"],
        })

        # 下一期的上一期实际价格，使用真实调价后价格
        previous_actual = actual_after

    return pd.DataFrame(rows)


def print_error_summary(result):
    """
    打印整体误差、普通时期误差、特殊调控时期误差。
    """
    all_mae = result["abs_delta_error"].mean()

    normal = result[result["is_special_regulated"] == False]
    special = result[result["is_special_regulated"] == True]

    normal_mae = normal["abs_delta_error"].mean()
    special_mae = special["abs_delta_error"].mean()

    print("全部样本数:", len(result))
    print("普通样本数:", len(normal))
    print("特殊调控样本数:", len(special))
    print()
    print("全部样本 Delta P theory MAE:", round(all_mae, 4))
    print("普通样本 Delta P theory MAE:", round(normal_mae, 4))
    print("特殊调控样本 Delta P theory MAE:", round(special_mae, 4))


def main():
    """
    程序入口。
    """
    result = calculate_all_theory_delta()

    output_path = RESULT_DIR / "all_delta_p_theory_compare.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("全部 Delta P theory 计算完成")
    print("输出文件:", output_path.resolve())
    print()
    print_error_summary(result)
    print()
    print("前 15 行预览:")
    print(result.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
