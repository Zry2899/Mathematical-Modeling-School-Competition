"""
导出最终模型预测结果对比表。

输出文件：
    result/final_prediction_compare_structural.csv

字段：
    gasoline_actual_change
    gasoline_predicted_change
    diesel_actual_change
    diesel_predicted_change
"""

from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "result"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from final_price_model import (
    predict_diesel_price_adjustment,
    predict_gasoline_price_adjustment,
)


def load_prediction_input():
    """
    读取最终模型需要的原始/中间数据，并合并成一个表。
    """
    domestic = pd.read_csv(RESULT_DIR / "domestic_events_clean.csv")
    oil = pd.read_csv(RESULT_DIR / "oil_10_workday_average.csv")
    exchange = pd.read_csv(DATA_DIR / "cny_usd_exchange_rate.csv")

    # 汇率文件第二列是中文列名，这里直接按位置重命名
    exchange = exchange.iloc[:, 0:2].copy()
    exchange.columns = ["date", "exchange_rate"]

    domestic["date"] = pd.to_datetime(domestic["date"])
    domestic["notice_date"] = pd.to_datetime(domestic["notice_date"])
    oil["date"] = pd.to_datetime(oil["date"])
    oil["notice_date"] = pd.to_datetime(oil["notice_date"])
    exchange["date"] = pd.to_datetime(exchange["date"])

    merged = domestic.merge(
        oil[["date", "notice_date", "wti_mean", "brent_mean", "basket_mean"]],
        on=["date", "notice_date"],
        how="left",
    )

    merged = pd.merge_asof(
        merged.sort_values("date"),
        exchange.sort_values("date"),
        on="date",
        direction="backward",
    )

    return merged


def main():
    """
    程序入口。
    """
    data = load_prediction_input()

    gasoline_pred = predict_gasoline_price_adjustment(data)
    diesel_pred = predict_diesel_price_adjustment(data)

    output = pd.DataFrame({
        "gasoline_actual_change": data["gasoline_change"],
        "gasoline_predicted_change": gasoline_pred["pred_delta_no_special"],
        "diesel_actual_change": data["diesel_change"],
        "diesel_predicted_change": diesel_pred["pred_delta_no_special"],
    })

    output_path = RESULT_DIR / "final_prediction_compare_scheme1.csv"
    try:
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        output_path = RESULT_DIR / "final_prediction_compare_structural_updated.csv"
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        print("目标文件被占用，已改存为:", output_path.resolve())

    print("预测对比文件已生成")
    print("输出文件:", output_path.resolve())
    print(output.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
