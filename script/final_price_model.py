"""
最终成品油调价幅度预测函数。

采用已确定的方案一：
1. 先按照调价幅度模型计算 raw_rule_delta；
2. 再加入 50 元/吨门槛和累计未调幅度 carry；
3. 输出经过完整机制后的预测调价幅度 pred_delta_no_special。
"""

import numpy as np
import pandas as pd


NU = 7.33
RULE_THRESHOLD = 50.0


GASOLINE_PARAMS = {
    "w_wti": 0.47376354384215497,
    "w_brent": 0.07934593782510575,
    "w_basket": 0.4468905183327393,
    "alpha": 1.3175617132622648,
    "pi0": 613.3017012254874,
    "intercept": -2.270083650854798,
}


DIESEL_PARAMS = {
    "w_wti": 0.47511053691233207,
    "w_brent": 0.07910902734113263,
    "w_basket": 0.44578043574653536,
    "alpha": 1.2661333155600092,
    "pi0": 579.0517221651798,
    "intercept": -2.074435036922397,
}


def predict_gasoline_price_adjustment(df):
    """
    预测汽油调价幅度。

    输入数据需要包含 date、WTI/Brent/Basket 窗口均价和汇率。
    返回原数据副本，并新增 raw_rule_delta、rule_delta、pred_delta_no_special、carry 等列。
    """
    return _predict_price_adjustment(df, GASOLINE_PARAMS)


def predict_diesel_price_adjustment(df):
    """
    预测柴油调价幅度。

    输入数据需要包含 date、WTI/Brent/Basket 窗口均价和汇率。
    返回原数据副本，并新增 raw_rule_delta、rule_delta、pred_delta_no_special、carry 等列。
    """
    return _predict_price_adjustment(df, DIESEL_PARAMS)


def _predict_price_adjustment(df, params):
    """
    内部通用预测函数。
    """
    data = df.copy()

    if "date" not in data.columns:
        raise ValueError("输入数据缺少 date 列")

    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    wti_col = _find_column(data, ["wti_mean", "wti", "WTI"])
    brent_col = _find_column(data, ["brent_mean", "brent", "Brent"])
    basket_col = _find_column(data, ["basket_mean", "basket", "Basket"])
    exchange_col = _find_column(data, ["exchange_rate", "mu", "rate"])

    wti = pd.to_numeric(data[wti_col], errors="coerce")
    brent = pd.to_numeric(data[brent_col], errors="coerce")
    basket = pd.to_numeric(data[basket_col], errors="coerce")
    exchange_rate = pd.to_numeric(data[exchange_col], errors="coerce")

    # 综合国际油价 O_t。
    data["weighted_oil"] = (
        params["w_wti"] * wti
        + params["w_brent"] * brent
        + params["w_basket"] * basket
    )

    # 利润分段函数 h(O_t)。
    data["h_oil"] = _profit_shape(data["weighted_oil"])

    # X_t = 汇率 * 单位换算系数 * 综合国际油价。
    data["X"] = exchange_rate * NU * data["weighted_oil"]

    # 调价幅度模型使用相邻两次调价窗口之间的变化。
    data["delta_X"] = data["X"].diff()
    data["delta_h"] = data["h_oil"].diff()
    data["raw_rule_delta"] = (
        params["alpha"] * data["delta_X"]
        + params["pi0"] * data["delta_h"]
        + params["intercept"]
    )

    carry = 0.0
    rule_delta_list = []
    pred_delta_list = []
    carry_before_list = []
    carry_after_list = []

    for raw_rule_delta in data["raw_rule_delta"]:
        carry_before = carry

        if pd.isna(raw_rule_delta):
            rule_delta = np.nan
            pred_delta = np.nan
            carry_after = carry
        else:
            rule_delta = carry_before + raw_rule_delta

            # 50 元/吨门槛：不足 50 元不调价，并累计到下一期。
            if abs(rule_delta) >= RULE_THRESHOLD:
                pred_delta = rule_delta
                carry = 0.0
            else:
                pred_delta = 0.0
                carry = rule_delta

            carry_after = carry

        rule_delta_list.append(rule_delta)
        pred_delta_list.append(pred_delta)
        carry_before_list.append(carry_before)
        carry_after_list.append(carry_after)

    data["rule_delta"] = rule_delta_list
    data["pred_delta_no_special"] = pred_delta_list
    data["pred_delta"] = pred_delta_list
    data["carry_before"] = carry_before_list
    data["carry_after"] = carry_after_list

    return data


def _profit_shape(oil_price):
    """
    利润分段函数 h(O)。
    """
    oil_price = np.asarray(oil_price, dtype=float)
    return np.where(
        oil_price <= 80,
        1.0,
        np.where(oil_price <= 130, (130 - oil_price) / 50, 0.0),
    )


def _find_column(df, candidates):
    """
    从候选列名中找到实际存在的列。
    """
    lower_map = {str(col).lower(): col for col in df.columns}

    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]

    raise ValueError("输入数据缺少字段，候选列名为: " + ", ".join(candidates))
