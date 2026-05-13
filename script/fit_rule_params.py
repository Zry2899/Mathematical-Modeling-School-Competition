"""
用“规则调价幅度”重新拟合机制参数。

为什么要写这个脚本：
之前直接用实际调价后价格回归理论价格，会把政策平滑、特殊调控、
累计未调幅度等因素混在一起，导致参数在 2026 年题目给出的两个
官方机制调幅点上失真。

本脚本改为直接拟合 Delta P rule：

1. 固定 pi0，避免 beta 和 pi0 严重共线；
2. 网格搜索 w_wti、w_brent、w_basket、alpha、beta；
3. 对每组参数跑完整规则：
   P_theory -> Delta P_theory -> 累计未调 -> 50 元门槛 -> Delta P_rule；
4. 正常时期要求 Delta P_rule 接近实际调价幅度；
5. 题目给出的 2026 两个机制应调幅作为强锚点：
   2026-03-24: Delta P_rule 约为 2205；
   2026-04-08: Delta P_rule 约为 800；
6. 选综合损失最小的一组参数；
7. 用最优参数重新计算特殊调控期 theta_t。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import base_model


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"


# 题目中给出的两个官方机制应调幅。
# 数据表中对应的实际调价日期是 2026-03-24 和 2026-04-08。
ANCHORS = {
    "2026-03-24": 2205.0,
    "2026-04-08": 800.0,
}


def prepare_data():
    """
    读取全部模型数据，并补充上一期实际价格。
    """
    data = base_model.load_model_data(include_adjust_day=True)
    data = data.sort_values("date").reset_index(drop=True)

    data["is_changed"] = data["is_changed"].astype(str).str.upper() == "TRUE"
    data["previous_actual"] = data["gasoline_price_after"].shift(1)

    # 第一行没有上一期价格，用“本期调价后价格 - 本期调价幅度”反推
    data.loc[0, "previous_actual"] = (
        data.loc[0, "gasoline_price_after"] - data.loc[0, "gasoline_change"]
    )

    return data


def make_params(w_wti, w_brent, w_basket, alpha, beta, pi0, lambda_factor):
    """
    把一组网格参数整理成 base_model 能使用的参数字典。
    """
    params = base_model.PARAMS.copy()

    params["w_wti"] = w_wti
    params["w_brent"] = w_brent
    params["w_basket"] = w_basket
    params["alpha"] = alpha
    params["beta"] = beta
    params["pi0"] = pi0
    params["lambda_factor"] = lambda_factor

    # 拟合 Delta P rule 时不考虑特殊调控，先令 theta = 1
    params["theta_special"] = 1.0

    return params


def calculate_rule_with_params(data, params):
    """
    给定参数，计算全部样本的 Delta P rule。

    返回 DataFrame，包含理论价格、理论调幅、规则调幅、累计未调幅度等。
    """
    rows = []
    carry = 0.0

    for _, row in data.iterrows():
        weighted_oil_mean = base_model.calculate_weighted_oil_mean(row, params)
        filtered_oil_price = base_model.apply_oil_floor(weighted_oil_mean, params)

        theory_price = base_model.calculate_theory_price(
            oil_price=filtered_oil_price,
            exchange_rate=row["exchange_rate"],
            beta=params["beta"],
            params=params,
        )

        previous_actual = row["previous_actual"]
        delta_p_theory = theory_price - previous_actual

        carry_before = carry
        stacked_change = delta_p_theory + carry_before

        if abs(stacked_change) < params["rule_threshold"]:
            delta_p_rule = 0.0
            carry = stacked_change
        else:
            delta_p_rule = stacked_change
            carry = 0.0

        rows.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "weighted_oil_mean": weighted_oil_mean,
            "filtered_oil_price": filtered_oil_price,
            "exchange_rate": row["exchange_rate"],
            "previous_actual": previous_actual,
            "theory_price": theory_price,
            "delta_p_theory": delta_p_theory,
            "carry_before": carry_before,
            "stacked_change": stacked_change,
            "delta_p_rule": delta_p_rule,
            "carry_after": carry,
            "actual_change": row["gasoline_change"],
            "actual_after": row["gasoline_price_after"],
            "is_changed": row["is_changed"],
            "is_special_regulated": row["is_special_regulated"],
        })

    result = pd.DataFrame(rows)
    result["rule_error"] = result["delta_p_rule"] - result["actual_change"]
    result["abs_rule_error"] = result["rule_error"].abs()

    return result


def calculate_loss(rule_result, anchor_weight):
    """
    计算综合损失。

    normal_mae:
        非特殊调控时期，Delta P rule 与实际调幅的 MAE。

    anchor_mae:
        2026 两个官方机制调幅锚点的 MAE。

    total_loss:
        normal_mae + anchor_weight * anchor_mae。
    """
    normal = rule_result[rule_result["is_special_regulated"] == False]
    normal_mae = normal["abs_rule_error"].mean()

    anchor_errors = []
    anchor_predictions = {}

    for date, official_value in ANCHORS.items():
        matched = rule_result[rule_result["date"] == date]
        if len(matched) == 0:
            continue

        predicted = float(matched.iloc[0]["delta_p_rule"])
        anchor_predictions[date] = predicted
        anchor_errors.append(abs(predicted - official_value))

    if len(anchor_errors) == 0:
        anchor_mae = np.inf
    else:
        anchor_mae = float(np.mean(anchor_errors))

    total_loss = normal_mae + anchor_weight * anchor_mae

    return {
        "normal_mae": normal_mae,
        "anchor_mae": anchor_mae,
        "total_loss": total_loss,
        "anchor_predictions": anchor_predictions,
    }


def grid_values(start, stop, step):
    """
    生成包含 stop 附近端点的网格。
    """
    return np.arange(start, stop + step / 2, step)


def search_params(data, args):
    """
    网格搜索参数。
    """
    results = []
    best = None
    best_rule_result = None

    previous_actual = data["previous_actual"].to_numpy(dtype=float)
    actual_change = data["gasoline_change"].to_numpy(dtype=float)
    exchange_rate = data["exchange_rate"].to_numpy(dtype=float)
    is_special = data["is_special_regulated"].to_numpy(dtype=bool)
    dates = data["date"].dt.strftime("%Y-%m-%d").to_numpy()

    normal_mask = ~is_special
    anchor_index = {}
    for date in ANCHORS:
        matched = np.where(dates == date)[0]
        if len(matched) > 0:
            anchor_index[date] = int(matched[0])

    weight_values = grid_values(0, 1, args.weight_step)
    alpha_values = grid_values(args.alpha_min, args.alpha_max, args.alpha_step)
    beta_values = grid_values(args.beta_min, args.beta_max, args.beta_step)

    for w_wti in weight_values:
        for w_brent in weight_values:
            w_basket = 1 - w_wti - w_brent
            if w_basket < -1e-9:
                continue
            w_basket = max(w_basket, 0)

            weighted_oil = (
                w_wti * data["wti_mean"].to_numpy(dtype=float)
                + w_brent * data["brent_mean"].to_numpy(dtype=float)
                + w_basket * data["basket_mean"].to_numpy(dtype=float)
            )
            oil = np.maximum(weighted_oil, base_model.PARAMS["oil_floor"])

            # 利润函数中 pi0 的系数
            pi_factor = np.where(
                oil <= 80,
                1.0,
                np.where(oil <= 130, (130 - oil) / 50, 0.0),
            )

            barrel_per_ton = base_model.PARAMS["barrel_per_ton"]
            oil_cost = oil * exchange_rate * barrel_per_ton
            cap_cost = 130 * exchange_rate * barrel_per_ton
            extra_cost = (oil - 130) * exchange_rate * barrel_per_ton
            extra_cost = np.where(oil >= 130, extra_cost, 0.0)

            for alpha in alpha_values:
                lambda_above_130 = alpha * args.lambda_factor

                variable_part = np.where(
                    oil < 130,
                    alpha * oil_cost + args.pi0 * pi_factor,
                    alpha * cap_cost + lambda_above_130 * extra_cost,
                )

                for beta in beta_values:
                    theory_price = variable_part + beta
                    delta_p_theory = theory_price - previous_actual

                    delta_p_rule = np.zeros(len(data))
                    carry = 0.0
                    for i, delta in enumerate(delta_p_theory):
                        stacked = delta + carry
                        if abs(stacked) < base_model.PARAMS["rule_threshold"]:
                            delta_p_rule[i] = 0.0
                            carry = stacked
                        else:
                            delta_p_rule[i] = stacked
                            carry = 0.0

                    abs_error = np.abs(delta_p_rule - actual_change)
                    normal_mae = float(abs_error[normal_mask].mean())

                    anchor_errors = []
                    anchor_predictions = {}
                    for date, official_value in ANCHORS.items():
                        if date not in anchor_index:
                            continue
                        predicted = float(delta_p_rule[anchor_index[date]])
                        anchor_predictions[date] = predicted
                        anchor_errors.append(abs(predicted - official_value))

                    if len(anchor_errors) == 0:
                        anchor_mae = np.inf
                    else:
                        anchor_mae = float(np.mean(anchor_errors))

                    total_loss = normal_mae + args.anchor_weight * anchor_mae

                    row = {
                        "w_wti": w_wti,
                        "w_brent": w_brent,
                        "w_basket": w_basket,
                        "alpha": alpha,
                        "beta": beta,
                        "pi0": args.pi0,
                        "lambda_factor": args.lambda_factor,
                        "normal_mae": normal_mae,
                        "anchor_mae": anchor_mae,
                        "total_loss": total_loss,
                    }

                    for date in ANCHORS:
                        row["rule_" + date] = anchor_predictions.get(date, np.nan)

                    results.append(row)

                    if best is None or row["total_loss"] < best["total_loss"]:
                        best = row
                        params = make_params(
                            w_wti=w_wti,
                            w_brent=w_brent,
                            w_basket=w_basket,
                            alpha=alpha,
                            beta=beta,
                            pi0=args.pi0,
                            lambda_factor=args.lambda_factor,
                        )
                        best_rule_result = calculate_rule_with_params(data, params)

    results_df = pd.DataFrame(results).sort_values("total_loss").reset_index(drop=True)

    return results_df, best, best_rule_result


def calculate_theta(rule_result):
    """
    用最优参数下的 Delta P rule 计算特殊调控期 theta_t。
    """
    special = rule_result[rule_result["is_special_regulated"] == True].copy()

    theta_values = []
    for _, row in special.iterrows():
        if abs(row["delta_p_rule"]) < 1e-6:
            theta = None
        else:
            theta = row["actual_change"] / row["delta_p_rule"]
        theta_values.append(theta)

    special["theta_t"] = theta_values

    return special[[
        "date",
        "filtered_oil_price",
        "delta_p_theory",
        "carry_before",
        "delta_p_rule",
        "actual_change",
        "theta_t",
        "rule_error",
    ]]


def parse_args():
    """
    命令行参数。
    """
    parser = argparse.ArgumentParser(description="用官方机制调幅锚点重新拟合 Delta P rule 参数")

    parser.add_argument("--pi0", type=float, default=800.0, help="固定正常加工利润")
    parser.add_argument("--lambda-factor", type=float, default=0.0, help="高于 130 美元后的边际传导比例")
    parser.add_argument("--anchor-weight", type=float, default=30.0, help="官方机制调幅锚点的损失权重")

    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--alpha-min", type=float, default=0.8)
    parser.add_argument("--alpha-max", type=float, default=2.8)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--beta-min", type=float, default=1000.0)
    parser.add_argument("--beta-max", type=float, default=6000.0)
    parser.add_argument("--beta-step", type=float, default=200.0)

    parser.add_argument("--search-output", default=str(RESULT_DIR / "fit_rule_params_search.csv"))
    parser.add_argument("--rule-output", default=str(RESULT_DIR / "fit_rule_params_best_rule.csv"))
    parser.add_argument("--theta-output", default=str(RESULT_DIR / "fit_rule_params_theta.csv"))

    return parser.parse_args()


def main():
    """
    程序入口。
    """
    args = parse_args()
    data = prepare_data()

    results_df, best, best_rule_result = search_params(data, args)
    theta_table = calculate_theta(best_rule_result)

    search_output = Path(args.search_output)
    rule_output = Path(args.rule_output)
    theta_output = Path(args.theta_output)

    search_output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(search_output, index=False, encoding="utf-8-sig")
    best_rule_result.to_csv(rule_output, index=False, encoding="utf-8-sig")
    theta_table.to_csv(theta_output, index=False, encoding="utf-8-sig")

    print("规则参数拟合完成")
    print("搜索结果:", search_output.resolve())
    print("最优规则明细:", rule_output.resolve())
    print("theta 输出:", theta_output.resolve())
    print()
    print("最优参数:")
    for key, value in best.items():
        print(key + ":", round(value, 6) if isinstance(value, float) else value)
    print()
    print("特殊调控 theta:")
    valid_theta = theta_table.dropna(subset=["theta_t"])
    for i, (_, row) in enumerate(valid_theta.iterrows(), start=1):
        print(
            "theta_{} = {:.6f}    date = {}    actual = {:.4f}    delta_p_rule = {:.4f}".format(
                i,
                row["theta_t"],
                row["date"],
                row["actual_change"],
                row["delta_p_rule"],
            )
        )


if __name__ == "__main__":
    main()
