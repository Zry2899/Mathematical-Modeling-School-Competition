"""
计算全部样本的 Delta P rule，并提取特殊调控期的 theta_t。

当前口径：
1. 使用已经固定的参数；
2. 对全部数据逐期计算理论价格 Delta P theory；
3. 加入 50 元/吨门槛和累计未调幅度，得到 Delta P rule；
4. 不考虑特殊调控，也就是先令 theta_t = 1；
5. 在特殊调控日期上，用实际调价幅度除以 Delta P rule：

       theta_t = actual_change / delta_p_rule

这样得到的 theta_t 表示：
    在该特殊调控日，实际调幅相当于规则理论调幅的多少倍。
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
    固定已经选定的第二组参数。
    """
    params = base_model.PARAMS.copy()

    params["w_wti"] = 0.82
    params["w_brent"] = 0.10
    params["w_basket"] = 0.08

    params["alpha"] = 1.337214
    params["beta"] = 1820.751832
    params["pi0"] = 2533.074724

    # 当前估计下 lambda = 0
    params["lambda_factor"] = 0.0

    # 计算 Delta P rule 时不考虑特殊调控，因此特殊调控系数先设为 1
    params["theta_special"] = 1.0

    return params


def calculate_all_rule_values():
    """
    对全部调价窗口计算 Delta P theory 和 Delta P rule。

    这里自己展开计算过程，而不是直接只读 base_model 输出，
    是为了把 stacked_change、carry_before、carry_after 等中间量也保留下来。
    """
    params = build_fixed_params()
    data = base_model.load_model_data(include_adjust_day=True)
    data = data.sort_values("date").reset_index(drop=True)

    rows = []
    previous_actual = None
    carry = 0.0

    for _, row in data.iterrows():
        actual_change = row["gasoline_change"]
        actual_after = row["gasoline_price_after"]

        if previous_actual is None:
            previous_actual = actual_after - actual_change

        carry_before = carry

        weighted_oil_mean = base_model.calculate_weighted_oil_mean(row, params)
        filtered_oil_price = base_model.apply_oil_floor(weighted_oil_mean, params)

        theory_price = base_model.calculate_theory_price(
            oil_price=filtered_oil_price,
            exchange_rate=row["exchange_rate"],
            beta=params["beta"],
            params=params,
        )

        delta_p_theory = theory_price - previous_actual
        stacked_change = delta_p_theory + carry_before

        # 50 元/吨门槛规则：
        # 如果叠加后的绝对调幅不足 50，则本期不调，累计到下一期；
        # 否则本期按 stacked_change 调整，并清空累计值。
        if abs(stacked_change) < params["rule_threshold"]:
            delta_p_rule = 0.0
            carry = stacked_change
        else:
            delta_p_rule = stacked_change
            carry = 0.0

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
            "carry_before": round(carry_before, 4),
            "stacked_change": round(stacked_change, 4),
            "delta_p_rule": round(delta_p_rule, 4),
            "carry_after": round(carry, 4),
            "actual_change": round(actual_change, 4),
            "actual_after": round(actual_after, 4),
            "rule_error": round(delta_p_rule - actual_change, 4),
            "abs_rule_error": round(abs(delta_p_rule - actual_change), 4),
            "is_changed": row["is_changed"],
            "is_special_regulated": row["is_special_regulated"],
        })

        # 下一期仍然使用真实调价后价格，避免模型误差在价格水平上滚动累积
        previous_actual = actual_after

    return pd.DataFrame(rows)


def calculate_special_theta(rule_result):
    """
    从全部 Delta P rule 结果中挑出特殊调控日，计算 theta_t。
    """
    special = rule_result[rule_result["is_special_regulated"] == True].copy()

    theta_values = []

    for _, row in special.iterrows():
        delta_p_rule = row["delta_p_rule"]
        actual_change = row["actual_change"]

        if abs(delta_p_rule) < 1e-6:
            theta = None
        else:
            theta = actual_change / delta_p_rule

        theta_values.append(theta)

    special["theta_t"] = theta_values

    output = special[[
        "date",
        "weighted_oil_mean",
        "filtered_oil_price",
        "delta_p_theory",
        "carry_before",
        "stacked_change",
        "delta_p_rule",
        "actual_change",
        "theta_t",
        "rule_error",
    ]].copy()

    return output


def print_theta_values(theta_table):
    """
    打印 theta_i，便于直接观察规律。
    """
    valid = theta_table.dropna(subset=["theta_t"]).copy()

    print("特殊调控日 theta_i:")
    for i, (_, row) in enumerate(valid.iterrows(), start=1):
        print(
            "theta_{} = {:.6f}    date = {}    actual = {:.4f}    delta_p_rule = {:.4f}".format(
                i,
                row["theta_t"],
                row["date"],
                row["actual_change"],
                row["delta_p_rule"],
            )
        )

    print()
    if len(valid) > 0:
        print("theta 简单统计:")
        print("数量:", len(valid))
        print("均值:", round(valid["theta_t"].mean(), 6))
        print("中位数:", round(valid["theta_t"].median(), 6))
        print("最小值:", round(valid["theta_t"].min(), 6))
        print("最大值:", round(valid["theta_t"].max(), 6))


def main():
    """
    程序入口。
    """
    rule_result = calculate_all_rule_values()
    theta_table = calculate_special_theta(rule_result)

    rule_output = RESULT_DIR / "all_delta_p_rule_compare.csv"
    theta_output = RESULT_DIR / "theta_from_delta_p_rule.csv"

    rule_result.to_csv(rule_output, index=False, encoding="utf-8-sig")
    theta_table.to_csv(theta_output, index=False, encoding="utf-8-sig")

    print("Delta P rule 和 theta_t 计算完成")
    print("全部 Delta P rule 输出:", rule_output.resolve())
    print("特殊调控 theta 输出:", theta_output.resolve())
    print()
    print_theta_values(theta_table)
    print()
    print("特殊调控明细:")
    print(theta_table.to_string(index=False))


if __name__ == "__main__":
    main()
