"""
Fetch and normalize external data used by the simplified task 2 model.

Outputs:
    data/task2_cpi_monthly.csv
    data/task2_crude_production_monthly_eia.csv
    data/task2_petroleum_consumption_annual_eia.csv
    data/task2_data_requirements.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

EIA_BASE_URL = "https://api.eia.gov/v2/international/data/"
EIA_API_KEY = "DEMO_KEY"


def fetch_cpi_monthly() -> pd.DataFrame:
    """Fetch national CPI from Eastmoney via AkShare."""
    import akshare as ak

    raw = ak.macro_china_cpi()
    out = raw.rename(
        columns={
            "月份": "month",
            "全国-当月": "cpi_index_same_month_last_year_100",
            "全国-同比增长": "cpi_yoy_pct",
            "全国-环比增长": "cpi_mom_pct",
            "全国-累计": "cpi_cumulative_same_period_last_year_100",
        }
    )
    out["month"] = (
        out["month"]
        .astype(str)
        .str.replace("年", "-", regex=False)
        .str.replace("月份", "-01", regex=False)
    )
    out["month"] = pd.to_datetime(out["month"]).dt.strftime("%Y-%m-%d")
    keep = [
        "month",
        "cpi_index_same_month_last_year_100",
        "cpi_yoy_pct",
        "cpi_mom_pct",
        "cpi_cumulative_same_period_last_year_100",
    ]
    out = out[keep].sort_values("month").reset_index(drop=True)
    return out


def fetch_eia_series(
    frequency: str,
    product_id: str,
    activity_id: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch a China energy series from EIA international API."""
    params = {
        "api_key": EIA_API_KEY,
        "frequency": frequency,
        "data[0]": "value",
        "facets[countryRegionId][]": "CHN",
        "facets[productId][]": product_id,
        "facets[activityId][]": activity_id,
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }
    response = requests.get(EIA_BASE_URL, params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("response", {}).get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    keep = [
        "period",
        "productId",
        "productName",
        "activityId",
        "activityName",
        "unitName",
        "unit",
        "value",
    ]
    return df[keep].sort_values("period").reset_index(drop=True)


def build_requirements() -> pd.DataFrame:
    rows = [
        {
            "variable": "f_g,t / f_d,t",
            "meaning": "任务一模型输出的汽油/柴油理论调价幅度",
            "status": "available",
            "local_file": "result/final_prediction_compare_scheme1.csv",
            "note": "gasoline_predicted_change, diesel_predicted_change",
        },
        {
            "variable": "P_g,t / P_d,t",
            "meaning": "汽油/柴油价格水平",
            "status": "available",
            "local_file": "result/domestic_events_clean.csv",
            "note": "gasoline_price_after, diesel_price_after",
        },
        {
            "variable": "u_hist_g,t / u_hist_d,t",
            "meaning": "历史实际调价幅度",
            "status": "available",
            "local_file": "result/domestic_events_clean.csv",
            "note": "gasoline_change, diesel_change",
        },
        {
            "variable": "pi_m",
            "meaning": "月度 CPI 同比/环比",
            "status": "fetched",
            "local_file": "data/task2_cpi_monthly.csv",
            "note": "AkShare Eastmoney CPI, 2008-01 onward",
        },
        {
            "variable": "Brent_t",
            "meaning": "Brent 原油价格",
            "status": "available",
            "local_file": "data/brent-daily.csv",
            "note": "already available, daily",
        },
        {
            "variable": "Qprod_t",
            "meaning": "中国原油产量",
            "status": "fetched",
            "local_file": "data/task2_crude_production_monthly_eia.csv",
            "note": "EIA productId=57, activityId=1, monthly, TBPD",
        },
        {
            "variable": "Qimp_t",
            "meaning": "中国原油进口量",
            "status": "partial",
            "local_file": "",
            "note": "EIA monthly import is unavailable for crude; use customs/NBS source or simplify energy term",
        },
        {
            "variable": "Qprocess_t",
            "meaning": "原油加工量",
            "status": "proxy_fetched",
            "local_file": "data/task2_petroleum_consumption_annual_eia.csv",
            "note": "EIA annual petroleum consumption proxy; NBS monthly A030106 is preferable if accessible",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    cpi = fetch_cpi_monthly()
    cpi.to_csv(DATA_DIR / "task2_cpi_monthly.csv", index=False, encoding="utf-8-sig")

    crude_prod = fetch_eia_series(
        frequency="monthly",
        product_id="57",
        activity_id="1",
        start="2016-01",
        end="2026-05",
    )
    crude_prod.to_csv(
        DATA_DIR / "task2_crude_production_monthly_eia.csv",
        index=False,
        encoding="utf-8-sig",
    )

    petroleum_consumption = fetch_eia_series(
        frequency="annual",
        product_id="5",
        activity_id="2",
        start="2016",
        end="2024",
    )
    petroleum_consumption.to_csv(
        DATA_DIR / "task2_petroleum_consumption_annual_eia.csv",
        index=False,
        encoding="utf-8-sig",
    )

    requirements = build_requirements()
    requirements.to_csv(
        DATA_DIR / "task2_data_requirements.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Task 2 data files generated:")
    for path in [
        DATA_DIR / "task2_cpi_monthly.csv",
        DATA_DIR / "task2_crude_production_monthly_eia.csv",
        DATA_DIR / "task2_petroleum_consumption_annual_eia.csv",
        DATA_DIR / "task2_data_requirements.csv",
    ]:
        df = pd.read_csv(path)
        print(f"- {path.name}: rows={len(df)}, columns={list(df.columns)}")


if __name__ == "__main__":
    main()
