"""
参数拟合脚本。

本脚本实现一种“外层搜索权重 + 内层线性回归”的估计策略：

1. 先只使用非特殊调控时期的数据；
2. 为了尽量撇去累计未调幅度和时序传递影响，只保留“上一期真实发生过调价”的样本；
3. 对每一组 WTI、Brent、Basket 权重，先计算加权油价；
4. 按汇率波动幅度把时间序列切成若干段；
5. 在每一段内，近似认为汇率变化不大，于是可以用结构化线性回归估计参数；
6. 将各时间段估计出的参数按时间长度加权平均；
7. 用加权平均后的参数在全部训练样本上计算误差；
8. 网格搜索多组权重，选择误差最小的一组。

注意：
这里是第一版可解释参数估计，不考虑累计涨幅，也不考虑特殊调控系数。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# 让当前脚本可以直接导入同目录下的 base_model.py
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import base_model


# =====================================================
# 1. 基础设置
# =====================================================

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"


# =====================================================
# 2. 样本筛选
# =====================================================

def prepare_fit_data(include_adjust_day=True):
    """
    读取 base_model 合并好的数据，并筛选用于线性回归的样本。

    筛选逻辑：
    1. 当前样本不能是特殊调控期；
    2. 上一期必须真实发生过调价；

    为什么要求上一期真实发生过调价？
    因为我们这一版拟合暂时不考虑累计未调幅度。
    如果上一期没有调价，可能有未释放的累计涨跌幅，会污染本期回归。
    """
    data = base_model.load_model_data(include_adjust_day=include_adjust_day)
    data = data.sort_values("date").reset_index(drop=True)

    # 把 TRUE/FALSE 字符串转成布尔值
    data["is_changed"] = data["is_changed"].astype(str).str.upper() == "TRUE"
    data["prev_is_changed"] = data["is_changed"].shift(1).fillna(False)

    # 上一期调价后的真实价格
    data["previous_actual"] = data["gasoline_price_after"].shift(1)

    # 只保留非特殊调控，并且上一期真实调价的样本
    fit_data = data[
        (data["is_special_regulated"] == False)
        & (data["prev_is_changed"] == True)
    ].copy()

    fit_data = fit_data.dropna(subset=[
        "previous_actual",
        "gasoline_price_after",
        "wti_mean",
        "brent_mean",
        "basket_mean",
        "exchange_rate",
    ])

    return fit_data.reset_index(drop=True)


# =====================================================
# 3. 给定权重时，构造线性回归变量
# =====================================================

def add_oil_price_columns(data, w_wti, w_brent, w_basket):
    """
    给定一组油种权重，计算过滤前和过滤后的油价。

    weighted_oil_mean:
        三种油价的加权均价。

    filtered_oil_price:
        执行 40 美元/桶地板价后的油价。
    """
    data = data.copy()

    data["weighted_oil_mean"] = (
        w_wti * data["wti_mean"]
        + w_brent * data["brent_mean"]
        + w_basket * data["basket_mean"]
    )

    data["filtered_oil_price"] = data["weighted_oil_mean"].clip(
        lower=base_model.PARAMS["oil_floor"]
    )

    return data


def build_regression_matrix(data, fixed_exchange_rate=None):
    """
    构造结构化线性回归矩阵。

    在每个时间段内，近似认为汇率是常数，理论价格可以写成：

        P = beta + alpha * X_alpha + pi0 * X_pi + lambda * X_lambda

    其中：
        X_alpha:
            O < 130 时为 μ * ν * O；
            O >= 130 时为 μ * ν * 130。

        X_pi:
            O <= 80 时为 1；
            80 < O <= 130 时为 (130 - O) / 50；
            O > 130 时为 0。

        X_lambda:
            O < 130 时为 0；
            O >= 130 时为 μ * ν * (O - 130)。

    fixed_exchange_rate:
        如果给定，就用这个固定汇率构造特征。
        分段回归时使用该时间段的平均汇率。

        如果不给定，就使用每一行自己的汇率。
        计算整体误差时可以用这种方式，更接近完整模型。

    回归参数顺序为：
        beta, alpha, pi0, lambda
    """
    oil = data["filtered_oil_price"].to_numpy()
    barrel_per_ton = base_model.PARAMS["barrel_per_ton"]

    if fixed_exchange_rate is None:
        exchange_rate = data["exchange_rate"].to_numpy()
    else:
        exchange_rate = fixed_exchange_rate

    x_alpha = np.where(
        oil < 130,
        exchange_rate * barrel_per_ton * oil,
        exchange_rate * barrel_per_ton * 130,
    )

    x_pi = np.where(
        oil <= 80,
        1.0,
        np.where(oil <= 130, (130 - oil) / 50, 0.0),
    )

    x_lambda = np.where(
        oil < 130,
        0.0,
        exchange_rate * barrel_per_ton * (oil - 130),
    )

    # 第一列全为 1，对应截距 beta
    ones = np.ones(len(data))
    x_matrix = np.column_stack([ones, x_alpha, x_pi, x_lambda])

    # 目标变量使用真实调价后的汽油价格
    y = data["gasoline_price_after"].to_numpy()

    return x_matrix, y


# =====================================================
# 4. 按汇率波动切分时间段
# =====================================================

def split_by_exchange_rate(data, exchange_threshold=0.03, min_rows=8):
    """
    按汇率波动幅度把样本切成若干时间段。

    exchange_threshold:
        汇率最大值与最小值的相对差异阈值。
        例如 0.03 表示一个时间段内汇率波动尽量不超过 3%。

    min_rows:
        每段最少样本数。

    切分方式：
        从前往后贪心扩展区间。
        如果加入新样本后汇率波动超过阈值，并且当前段已有足够样本，
        就结束当前段，从新样本开始下一段。
    """
    data = data.sort_values("date").reset_index(drop=True)

    segments = []
    start = 0

    for i in range(len(data)):
        current = data.iloc[start:i + 1]
        rate_max = current["exchange_rate"].max()
        rate_min = current["exchange_rate"].min()
        rate_mean = current["exchange_rate"].mean()

        if rate_mean == 0:
            rate_range = 0
        else:
            rate_range = (rate_max - rate_min) / rate_mean

        # 如果汇率波动超过阈值，并且当前段已经有足够样本，就切段
        if rate_range > exchange_threshold and (i - start) >= min_rows:
            segments.append(data.iloc[start:i].copy())
            start = i

    # 处理最后一段
    last_segment = data.iloc[start:].copy()
    if len(last_segment) > 0:
        segments.append(last_segment)

    # 如果最后一段太短，就合并到上一段，避免回归不稳定
    if len(segments) >= 2 and len(segments[-1]) < min_rows:
        merged = pd.concat([segments[-2], segments[-1]], ignore_index=True)
        segments = segments[:-2] + [merged]

    return segments


# =====================================================
# 5. 单段线性回归
# =====================================================

def fit_one_segment(segment, fixed_pi0=None, fixed_beta=None):
    """
    对一个时间段做结构化线性回归。

    fixed_pi0:
        如果给定，就固定正常加工利润 pi0，只回归 beta、alpha、lambda。

    fixed_beta:
        如果给定，就固定固定项 beta，只回归 alpha、pi0、lambda。

    返回：
        params:
            beta, alpha, pi0, lambda

        error:
            该段价格拟合 MAE
    """
    # 该策略的核心假设：
    # 在一个时间段内汇率波动不大，因此用该段平均汇率作为常数参与线性回归。
    segment_exchange_rate = segment["exchange_rate"].mean()
    x_matrix, y = build_regression_matrix(segment, fixed_exchange_rate=segment_exchange_rate)

    # 原始矩阵列含义：
    # 0: beta 截距项
    # 1: alpha 对应项
    # 2: pi0 对应项
    # 3: lambda 对应项
    if fixed_pi0 is not None and fixed_beta is not None:
        # beta 和 pi0 都固定，只估 alpha 和 lambda
        y_adjusted = y - fixed_beta * x_matrix[:, 0] - fixed_pi0 * x_matrix[:, 2]
        x_reg = x_matrix[:, [1, 3]]
        coef, _, _, _ = np.linalg.lstsq(x_reg, y_adjusted, rcond=None)

        beta = fixed_beta
        alpha = coef[0]
        pi0 = fixed_pi0
        lambda_above_130 = coef[1]

    elif fixed_pi0 is not None:
        # 固定 pi0，只估 beta、alpha、lambda
        y_adjusted = y - fixed_pi0 * x_matrix[:, 2]
        x_reg = x_matrix[:, [0, 1, 3]]
        coef, _, _, _ = np.linalg.lstsq(x_reg, y_adjusted, rcond=None)

        beta = coef[0]
        alpha = coef[1]
        pi0 = fixed_pi0
        lambda_above_130 = coef[2]

    elif fixed_beta is not None:
        # 固定 beta，只估 alpha、pi0、lambda
        y_adjusted = y - fixed_beta * x_matrix[:, 0]
        x_reg = x_matrix[:, [1, 2, 3]]
        coef, _, _, _ = np.linalg.lstsq(x_reg, y_adjusted, rcond=None)

        beta = fixed_beta
        alpha = coef[0]
        pi0 = coef[1]
        lambda_above_130 = coef[2]

    else:
        # 不固定任何参数，直接估 beta、alpha、pi0、lambda
        coef, _, _, _ = np.linalg.lstsq(x_matrix, y, rcond=None)

        beta = coef[0]
        alpha = coef[1]
        pi0 = coef[2]
        lambda_above_130 = coef[3]

    full_coef = np.array([beta, alpha, pi0, lambda_above_130])
    y_pred = x_matrix @ full_coef
    error = np.mean(np.abs(y_pred - y))

    params = {
        "beta": beta,
        "alpha": alpha,
        "pi0": pi0,
        "lambda": lambda_above_130,
    }

    return params, error


def is_reasonable_params(params):
    """
    判断一组回归参数是否基本合理。

    这里不做太严格的经济学约束，只过滤明显不合理的情况：
    - alpha 应该为正；
    - pi0 应该非负；
    - 130 美元以上的边际传导系数 lambda 应该非负；
    - lambda 不应大于 alpha 太多。
    """
    if params["alpha"] <= 0:
        return False

    if params["pi0"] < 0:
        return False

    if params["lambda"] < 0:
        return False

    if params["lambda"] > params["alpha"] * 1.2:
        return False

    return True


# =====================================================
# 6. 多段参数加权
# =====================================================

def segment_time_weight(segment):
    """
    计算某个时间段的权重。

    权重使用时间长度，也就是最后日期与最早日期的天数差。
    如果一个时间段内只有很短间隔，则至少给 1 的权重。
    """
    days = (segment["date"].max() - segment["date"].min()).days
    return max(days, 1)


def weighted_average_params(segment_results):
    """
    将多个时间段估计出的参数按时间长度加权平均。
    """
    total_weight = sum(item["weight"] for item in segment_results)

    avg = {
        "beta": 0.0,
        "alpha": 0.0,
        "pi0": 0.0,
        "lambda": 0.0,
    }

    for item in segment_results:
        weight = item["weight"] / total_weight
        for key in avg:
            avg[key] += item["params"][key] * weight

    return avg


# =====================================================
# 7. 误差计算
# =====================================================

def predict_price(data, params):
    """
    使用拟合得到的 beta、alpha、pi0、lambda 预测理论价格。

    这里使用每一行自己的汇率来计算整体误差。
    分段回归时是“段内汇率近似常数”，但最终评价时仍尽量贴近完整模型。
    """
    x_matrix, _ = build_regression_matrix(data)

    coef = np.array([
        params["beta"],
        params["alpha"],
        params["pi0"],
        params["lambda"],
    ])

    return x_matrix @ coef


def calculate_errors(data, params):
    """
    计算整体样本上的拟合误差。
    """
    y_true = data["gasoline_price_after"].to_numpy()
    y_pred = predict_price(data, params)

    price_error = y_pred - y_true
    price_mae = np.mean(np.abs(price_error))
    price_rmse = np.sqrt(np.mean(price_error ** 2))

    # 调价幅度误差：
    # 由于上一期实际价格相同，所以预测价格误差也等于调价幅度误差。
    actual_change = data["gasoline_change"].to_numpy()
    predicted_change = y_pred - data["previous_actual"].to_numpy()
    change_mae = np.mean(np.abs(predicted_change - actual_change))

    return {
        "price_mae": price_mae,
        "price_rmse": price_rmse,
        "change_mae": change_mae,
    }


# =====================================================
# 8. 给定一组权重时完成全部拟合
# =====================================================

def fit_for_one_weight(
    data,
    w_wti,
    w_brent,
    w_basket,
    exchange_threshold,
    min_rows,
    fixed_pi0=None,
    fixed_beta=None,
):
    """
    对固定的一组 w1、w2、w3 执行完整拟合。

    步骤：
    1. 计算该权重下的加权油价；
    2. 按汇率波动切分时间段；
    3. 每段分别做线性回归；
    4. 过滤掉明显不合理的段参数；
    5. 按时间长度加权平均参数；
    6. 在整体样本上计算误差。
    """
    weighted_data = add_oil_price_columns(data, w_wti, w_brent, w_basket)
    segments = split_by_exchange_rate(weighted_data, exchange_threshold, min_rows)

    segment_results = []

    for segment_id, segment in enumerate(segments, start=1):
        if len(segment) < 4:
            continue

        params, segment_error = fit_one_segment(
            segment,
            fixed_pi0=fixed_pi0,
            fixed_beta=fixed_beta,
        )

        if not is_reasonable_params(params):
            continue

        segment_results.append({
            "segment_id": segment_id,
            "start_date": segment["date"].min(),
            "end_date": segment["date"].max(),
            "rows": len(segment),
            "weight": segment_time_weight(segment),
            "params": params,
            "segment_mae": segment_error,
        })

    # 如果有效时间段太少，说明这组权重下回归不稳定，直接返回 None
    if len(segment_results) == 0:
        return None

    avg_params = weighted_average_params(segment_results)
    errors = calculate_errors(weighted_data, avg_params)

    result = {
        "w_wti": w_wti,
        "w_brent": w_brent,
        "w_basket": w_basket,
        "beta": avg_params["beta"],
        "alpha": avg_params["alpha"],
        "pi0": avg_params["pi0"],
        "lambda": avg_params["lambda"],
        "lambda_factor": avg_params["lambda"] / avg_params["alpha"],
        "price_mae": errors["price_mae"],
        "price_rmse": errors["price_rmse"],
        "change_mae": errors["change_mae"],
        "segments_used": len(segment_results),
        "rows_used": len(weighted_data),
    }

    return result, segment_results


# =====================================================
# 9. 权重网格搜索
# =====================================================

def grid_search_weights(data, step, exchange_threshold, min_rows, fixed_pi0=None, fixed_beta=None):
    """
    对 WTI、Brent、Basket 权重做网格搜索。

    为了保证 w1 + w2 + w3 = 1，只枚举 w_wti 和 w_brent，
    然后令：
        w_basket = 1 - w_wti - w_brent
    """
    all_results = []
    best_result = None
    best_segments = None

    grid_values = np.arange(0, 1 + step / 2, step)

    for w_wti in grid_values:
        for w_brent in grid_values:
            w_basket = 1 - w_wti - w_brent

            # 跳过非法权重
            if w_basket < -0.000001:
                continue

            w_basket = max(w_basket, 0)

            fit_result = fit_for_one_weight(
                data=data,
                w_wti=w_wti,
                w_brent=w_brent,
                w_basket=w_basket,
                exchange_threshold=exchange_threshold,
                min_rows=min_rows,
                fixed_pi0=fixed_pi0,
                fixed_beta=fixed_beta,
            )

            if fit_result is None:
                continue

            result, segment_results = fit_result
            all_results.append(result)

            if best_result is None or result["change_mae"] < best_result["change_mae"]:
                best_result = result
                best_segments = segment_results

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("change_mae").reset_index(drop=True)

    return results_df, best_result, best_segments


# =====================================================
# 10. 输出结果
# =====================================================

def save_segment_details(segment_results, output_path):
    """
    保存最优权重下，每个时间段的回归参数。
    """
    rows = []

    for item in segment_results:
        params = item["params"]
        rows.append({
            "segment_id": item["segment_id"],
            "start_date": item["start_date"].strftime("%Y-%m-%d"),
            "end_date": item["end_date"].strftime("%Y-%m-%d"),
            "rows": item["rows"],
            "time_weight_days": item["weight"],
            "segment_mae": item["segment_mae"],
            "beta": params["beta"],
            "alpha": params["alpha"],
            "pi0": params["pi0"],
            "lambda": params["lambda"],
        })

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def parse_args():
    """
    读取命令行参数。
    """
    parser = argparse.ArgumentParser(description="网格搜索油种权重，并用分段线性回归估计参数")

    parser.add_argument("--step", type=float, default=0.1, help="权重网格步长")
    parser.add_argument("--exchange-threshold", type=float, default=0.03, help="单段内汇率相对波动阈值")
    parser.add_argument("--min-rows", type=int, default=8, help="每个时间段最少样本数")
    parser.add_argument("--exclude-adjust-day", action="store_true", help="使用不包含调价当天的均价")
    parser.add_argument("--fixed-pi0", type=float, default=None, help="固定 pi0，只回归其他参数")
    parser.add_argument("--fixed-beta", type=float, default=None, help="固定 beta，只回归其他参数")

    parser.add_argument(
        "--summary-output",
        default=str(RESULT_DIR / "fit_params_grid_search.csv"),
        help="保存所有权重搜索结果",
    )
    parser.add_argument(
        "--segment-output",
        default=str(RESULT_DIR / "fit_params_best_segments.csv"),
        help="保存最优权重下的分段回归结果",
    )

    return parser.parse_args()


def main():
    """
    程序入口。
    """
    args = parse_args()

    include_adjust_day = not args.exclude_adjust_day
    data = prepare_fit_data(include_adjust_day=include_adjust_day)

    results_df, best_result, best_segments = grid_search_weights(
        data=data,
        step=args.step,
        exchange_threshold=args.exchange_threshold,
        min_rows=args.min_rows,
        fixed_pi0=args.fixed_pi0,
        fixed_beta=args.fixed_beta,
    )

    summary_output = Path(args.summary_output)
    segment_output = Path(args.segment_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    segment_output.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(summary_output, index=False, encoding="utf-8-sig")

    if best_segments is not None:
        save_segment_details(best_segments, segment_output)

    print("参数搜索完成")
    print("训练样本数:", len(data))
    print("权重搜索结果:", summary_output.resolve())
    print("最优分段结果:", segment_output.resolve())
    print()
    print("最优参数:")
    for key, value in best_result.items():
        if isinstance(value, float):
            print(key + ":", round(value, 6))
        else:
            print(key + ":", value)

    print()
    print("误差最小的前 10 组权重:")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
