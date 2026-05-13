"""
重新整理第一问需要的基础数据。

本脚本修正两个关键口径：

1. 公告日和生效日分开：
   - notice_date 表示发改委公告日；
   - date / effective_date 表示价格生效日。

2. 国际油价窗口改为公告日前最近 10 个有效交易日：
   - 不再使用 10 个自然日；
   - 不把生效日之后的油价放进窗口；
   - 默认使用 crude_date < notice_date，避免未来信息。

输出：
    result/domestic_events_clean.csv
    result/oil_10_workday_average.csv
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "result"


def classify_event(row):
    """
    给国内调价事件重新分类。

    normal:
        普通按机制调价。

    threshold_no_adjust:
        调价幅度低于 50 元/吨，按机制不作调整。

    temporary_control:
        公告中明确属于临时调控、继续实施调控、调控后实际调价的样本。
        当前数据中最明确的是 2026-03-24 和 2026-04-08 两次。

    mechanism_reform:
        2016-01-14 是机制完善同步下调，不作为常规机制拟合样本。
    """
    effective_date = row["date"].strftime("%Y-%m-%d")
    title = str(row["notice_title"])
    gasoline_change = row["gasoline_change"]
    diesel_change = row["diesel_change"]

    if effective_date == "2016-01-14":
        return "mechanism_reform"

    if effective_date in ["2026-03-24", "2026-04-08"]:
        return "temporary_control"

    if "临时调控" in title or "继续实施调控" in title or "调控后" in title:
        return "temporary_control"

    if gasoline_change == 0 and diesel_change == 0:
        return "threshold_no_adjust"

    return "normal"


def build_domestic_events():
    """
    从 data/rare-domastic.csv 构建标准国内调价事件表。
    """
    path = DATA_DIR / "rare-domastic.csv"
    df = pd.read_csv(path)

    df = df.rename(columns={
        "date": "effective_date",
        "gasoline_adjust_cny_per_ton": "gasoline_change",
        "diesel_adjust_cny_per_ton": "diesel_change",
        "beijing_gasoline_ceiling_after_cny_per_ton": "gasoline_price_after",
        "beijing_diesel_ceiling_after_cny_per_ton": "diesel_price_after",
    })

    df["effective_date"] = pd.to_datetime(df["effective_date"])
    df["notice_date"] = pd.to_datetime(df["notice_date"])

    number_columns = [
        "gasoline_change",
        "diesel_change",
        "gasoline_price_after",
        "diesel_price_after",
    ]
    for col in number_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["effective_date", "notice_date"] + number_columns)

    # 题目讨论的是 2016 年《石油价格管理办法》后的现行机制。
    # 2013-2015 年数据口径和规则可能不同，这里不纳入第一问主模型。
    df = df[df["effective_date"] >= pd.Timestamp("2016-01-14")]

    df = df.sort_values("effective_date").reset_index(drop=True)

    # 为了兼容已有模型，保留 date 列作为生效日。
    df["date"] = df["effective_date"]
    df["is_changed"] = (df["gasoline_change"] != 0) | (df["diesel_change"] != 0)

    df["special_type"] = df.apply(classify_event, axis=1)
    df["is_special_regulated"] = df["special_type"] == "temporary_control"

    output_columns = [
        "date",
        "notice_date",
        "effective_date",
        "gasoline_change",
        "diesel_change",
        "gasoline_price_after",
        "diesel_price_after",
        "is_changed",
        "special_type",
        "is_special_regulated",
        "notice_title",
        "source_url",
    ]

    out = df[output_columns].copy()
    for col in ["date", "notice_date", "effective_date"]:
        out[col] = out[col].dt.strftime("%Y-%m-%d")

    return out


def read_crude_price(name):
    """
    读取一种国际原油价格。
    """
    path = DATA_DIR / f"{name}-daily.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def ten_trading_day_mean(crude, notice_date):
    """
    计算公告日前最近 10 个有效交易日均价。

    这里使用 crude.date < notice_date，避免把公告日当天或生效日之后的价格
    纳入窗口。若之后想做稳健性检验，可以改成 <= notice_date。
    """
    window = crude[crude["date"] < notice_date].tail(10)

    if len(window) == 0:
        return {
            "mean": None,
            "valid_days": 0,
            "window_start": None,
            "window_end": None,
        }

    return {
        "mean": window["price"].mean(),
        "valid_days": len(window),
        "window_start": window["date"].min(),
        "window_end": window["date"].max(),
    }


def build_oil_averages(domestic_events):
    """
    对每个公告日计算 WTI、Brent、Basket 的前 10 个交易日均价。
    """
    crude_data = {
        "wti": read_crude_price("wti"),
        "brent": read_crude_price("brent"),
        "basket": read_crude_price("basket"),
    }

    rows = []

    for _, event in domestic_events.iterrows():
        notice_date = pd.to_datetime(event["notice_date"])
        row = {
            "date": event["date"],
            "notice_date": event["notice_date"],
            "effective_date": event["effective_date"],
        }

        for name, crude in crude_data.items():
            info = ten_trading_day_mean(crude, notice_date)
            row[f"{name}_mean"] = info["mean"]
            row[f"{name}_valid_days"] = info["valid_days"]

            if info["window_start"] is None:
                row[f"{name}_window_start"] = ""
                row[f"{name}_window_end"] = ""
            else:
                row[f"{name}_window_start"] = info["window_start"].strftime("%Y-%m-%d")
                row[f"{name}_window_end"] = info["window_end"].strftime("%Y-%m-%d")

        rows.append(row)

    out = pd.DataFrame(rows)

    for col in ["wti_mean", "brent_mean", "basket_mean"]:
        out[col] = out[col].round(4)

    return out


def main():
    """
    程序入口。
    """
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    domestic_events = build_domestic_events()
    oil_averages = build_oil_averages(domestic_events)

    domestic_output = RESULT_DIR / "domestic_events_clean.csv"
    oil_output = RESULT_DIR / "oil_10_workday_average.csv"

    domestic_events.to_csv(domestic_output, index=False, encoding="utf-8-sig")
    oil_averages.to_csv(oil_output, index=False, encoding="utf-8-sig")

    print("清洗完成")
    print("国内调价事件:", domestic_output.resolve())
    print("10个交易日均价:", oil_output.resolve())
    print()
    print("事件分类统计:")
    print(domestic_events["special_type"].value_counts().to_string())
    print()
    print("10个交易日均价预览:")
    print(oil_averages.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
