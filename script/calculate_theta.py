"""
计算特殊调控期的 theta_t。

这里沿用 fit_params.py 中推荐的第二组参数：

    w_wti    = 0.82
    w_brent  = 0.10
    w_basket = 0.08
    alpha    = 1.337214
    beta     = 1820.751832
    pi0      = 2533.074724
    lambda   = 0.0

计算思路：
1. 先用 theta_special = 1 跑 base 模型；
2. 此时特殊调控期的 predicted_change 可以理解为“机制应调幅度”；
3. 对特殊调控样本计算：

       theta_t = actual_change / predicted_change

4. 如果 predicted_change 太接近 0，则无法计算 theta_t，记为空。
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


def build_recommended_params():
    """
    构造推荐的第二组参数。
    """
    params = base_model.PARAMS.copy()

    params["w_wti"] = 0.82
    params["w_brent"] = 0.10
    params["w_basket"] = 0.08

    params["alpha"] = 1.337214
    params["beta"] = 1820.751832
    params["pi0"] = 2533.074724

    # 上一步回归得到 lambda = 0，因此 lambda_factor = lambda / alpha = 0
    params["lambda_factor"] = 0.0

    # 这里必须设为 1，目的是先算出“没有特殊削弱时”的机制应调幅度
    params["theta_special"] = 1.0

    return params


def calculate_theta_table():
    """
    计算特殊调控期的 theta_t 明细表。
    """
    params = build_recommended_params()

    data = base_model.load_model_data(include_adjust_day=True)
    result = base_model.run_model(data, params)

    special = result[result["is_special_regulated"] == True].copy()

    theta_values = []
    for _, row in special.iterrows():
        mechanism_change = row["predicted_change"]
        actual_change = row["actual_change"]

        if abs(mechanism_change) < 1e-6:
            theta = None
        else:
            theta = actual_change / mechanism_change

        theta_values.append(theta)

    special["theta_t"] = theta_values

    # 为了论文和检查方便，输出字段尽量简洁
    output = special[[
        "date",
        "weighted_oil_mean",
        "filtered_oil_price",
        "previous_actual",
        "theory_price",
        "theory_change",
        "predicted_change",
        "actual_change",
        "theta_t",
    ]].copy()

    output = output.rename(columns={
        "predicted_change": "mechanism_change",
    })

    return output


def main():
    """
    程序入口。
    """
    theta_table = calculate_theta_table()

    output_path = RESULT_DIR / "theta_special_values.csv"
    theta_table.to_csv(output_path, index=False, encoding="utf-8-sig")

    valid_theta = theta_table.dropna(subset=["theta_t"]).copy()

    print("特殊调控 theta 计算完成")
    print("输出文件:", output_path.resolve())
    print()
    print("全部特殊调控样本:")
    print(theta_table.to_string(index=False))
    print()
    print("可计算的 theta_i:")
    for i, (_, row) in enumerate(valid_theta.iterrows(), start=1):
        print(
            "theta_{} = {:.6f}    date = {}    actual = {:.4f}    mechanism = {:.4f}".format(
                i,
                row["theta_t"],
                row["date"],
                row["actual_change"],
                row["mechanism_change"],
            )
        )


if __name__ == "__main__":
    main()
