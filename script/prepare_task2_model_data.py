"""
Prepare local data for the simplified task-2 price-control model.

This script only uses files already present in data/ and result/.

Outputs:
    result/task2_monthly_model_data.csv
    result/task2_event_model_input.csv
    result/task2_monthly_model_data_clean.csv
    result/task2_event_model_input_clean.csv
    result/task2_monthly_model_data_forecast.csv
    result/task2_event_model_input_forecast.csv
    result/task2_data_audit.csv
    result/task2_missing_data_detail.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "result"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


def build_monthly_model_data() -> pd.DataFrame:
    macro = read_csv(DATA_DIR / "03_monthly_macro.csv")
    oil_supply = read_csv(DATA_DIR / "04_monthly_oil_supply.csv")
    product = read_csv(DATA_DIR / "05_monthly_product_consumption.csv")

    macro["month"] = month_start(macro["month"])
    oil_supply["month"] = month_start(oil_supply["month"])
    product["month"] = month_start(product["month"])

    monthly = (
        macro.merge(oil_supply, on="month", how="outer", suffixes=("", "_oil"))
        .merge(product, on="month", how="outer", suffixes=("", "_product"))
        .sort_values("month")
        .reset_index(drop=True)
    )

    monthly = apply_monthly_missing_rules(monthly)

    monthly["omega_gasoline"] = monthly["gasoline_weight"]
    monthly["omega_diesel"] = monthly["diesel_weight"]

    monthly["oil_import_dependency"] = monthly["crude_import_ton"] / (
        monthly["crude_production_ton"] + monthly["crude_import_ton"]
    )

    monthly["brent_pressure_h"] = (
        (monthly["brent_month_avg_usd_per_barrel"] - 80.0) / 50.0
    ).clip(lower=0.0, upper=1.0)

    monthly["crude_processing_ma12"] = (
        monthly["crude_processing_ton"].rolling(window=12, min_periods=3).mean()
    )
    monthly["processing_shortage_b"] = (
        1.0 - monthly["crude_processing_ton"] / monthly["crude_processing_ma12"]
    ).clip(lower=0.0)
    monthly["processing_shortage_b"] = monthly["processing_shortage_b"].fillna(0.0)

    monthly["energy_dependency_pressure"] = (
        monthly["oil_import_dependency"]
        * monthly["brent_pressure_h"]
    )

    monthly["energy_security_A_raw"] = (
        monthly["energy_dependency_pressure"] + monthly["processing_shortage_b"]
    )

    return monthly


def apply_monthly_missing_rules(monthly: pd.DataFrame) -> pd.DataFrame:
    """Apply documented missing-value rules before derived variables are built."""
    monthly = monthly.copy()
    monthly["year"] = monthly["month"].dt.year
    monthly["month_num"] = monthly["month"].dt.month
    monthly["energy_imputation_flag"] = 0
    monthly["energy_imputation_note"] = ""

    for col in [
        "crude_production_ton",
        "crude_processing_ton",
        "gasoline_consumption_ton",
        "diesel_consumption_ton",
    ]:
        if col not in monthly.columns:
            continue
        march_by_year = (
            monthly.loc[monthly["month_num"] == 3, ["year", col]]
            .dropna(subset=[col])
            .set_index("year")[col]
        )
        fill_mask = monthly[col].isna() & monthly["month_num"].isin([1, 2])
        monthly.loc[fill_mask, col] = monthly.loc[fill_mask, "year"].map(march_by_year)
        _mark_imputed(
            monthly,
            fill_mask & monthly[col].notna(),
            f"{col}: Jan-Feb filled with same-year March",
        )

    import_col = "crude_import_ton"
    value_col = "crude_import_value_usd"
    avg_col = "crude_import_avg_usd_per_ton"
    for col in [import_col, value_col, avg_col]:
        if col in monthly.columns:
            before = monthly[col].isna()
            monthly[col] = monthly[col].interpolate(
                method="linear",
                limit_area="inside",
            )
            _mark_imputed(
                monthly,
                before & monthly[col].notna(),
                f"{col}: internal linear interpolation",
            )

    forecast_cols = [
        "crude_production_ton",
        "crude_import_ton",
        "crude_processing_ton",
        "crude_import_value_usd",
        "crude_import_avg_usd_per_ton",
        "gasoline_consumption_ton",
        "diesel_consumption_ton",
    ]
    for col in forecast_cols:
        if col in monthly.columns:
            _fill_remaining_by_previous_year_same_month(monthly, col)

    has_consumption = (
        monthly["gasoline_consumption_ton"].notna()
        & monthly["diesel_consumption_ton"].notna()
    )
    total_product = monthly["gasoline_consumption_ton"] + monthly["diesel_consumption_ton"]
    monthly.loc[has_consumption, "gasoline_weight"] = (
        monthly.loc[has_consumption, "gasoline_consumption_ton"]
        / total_product.loc[has_consumption]
    )
    monthly.loc[has_consumption, "diesel_weight"] = (
        monthly.loc[has_consumption, "diesel_consumption_ton"]
        / total_product.loc[has_consumption]
    )

    return monthly.drop(columns=["month_num"])


def _fill_remaining_by_previous_year_same_month(monthly: pd.DataFrame, col: str) -> None:
    for idx, row in monthly.loc[monthly[col].isna()].iterrows():
        prev_month = row["month"] - pd.DateOffset(years=1)
        candidate = monthly.loc[monthly["month"] == prev_month, col]
        if not candidate.empty and pd.notna(candidate.iloc[0]):
            monthly.at[idx, col] = candidate.iloc[0]
            _mark_imputed(
                monthly,
                monthly.index == idx,
                f"{col}: filled with previous-year same-month value",
            )


def _mark_imputed(monthly: pd.DataFrame, mask: pd.Series | np.ndarray, note: str) -> None:
    if not np.any(mask):
        return
    monthly.loc[mask, "energy_imputation_flag"] = 1
    current = monthly.loc[mask, "energy_imputation_note"].fillna("")
    separator = np.where(current.astype(str).str.len() > 0, "; ", "")
    monthly.loc[mask, "energy_imputation_note"] = current + separator + note


def build_event_model_input(monthly: pd.DataFrame) -> pd.DataFrame:
    domestic = read_csv(RESULT_DIR / "domestic_events_clean.csv")
    prediction = read_csv(RESULT_DIR / "final_prediction_compare_scheme1.csv")

    domestic["date"] = pd.to_datetime(domestic["date"])
    prediction["date"] = pd.to_datetime(prediction["date"])

    event = domestic.merge(
        prediction[
            [
                "date",
                "gasoline_predicted_change",
                "diesel_predicted_change",
            ]
        ],
        on="date",
        how="left",
    )

    event["month"] = month_start(event["date"])

    event["gasoline_price_before"] = event["gasoline_price_after"].shift(1)
    event["diesel_price_before"] = event["diesel_price_after"].shift(1)

    first_idx = event.index[0]
    event.loc[first_idx, "gasoline_price_before"] = (
        event.loc[first_idx, "gasoline_price_after"]
        - event.loc[first_idx, "gasoline_change"]
    )
    event.loc[first_idx, "diesel_price_before"] = (
        event.loc[first_idx, "diesel_price_after"]
        - event.loc[first_idx, "diesel_change"]
    )

    event["f_gasoline"] = event["gasoline_predicted_change"]
    event["f_diesel"] = event["diesel_predicted_change"]
    event["u_hist_gasoline"] = event["gasoline_change"]
    event["u_hist_diesel"] = event["diesel_change"]

    event["alpha_hist_gasoline"] = np.where(
        event["f_gasoline"].abs() > 1e-9,
        event["u_hist_gasoline"] / event["f_gasoline"],
        np.nan,
    )
    event["alpha_hist_diesel"] = np.where(
        event["f_diesel"].abs() > 1e-9,
        event["u_hist_diesel"] / event["f_diesel"],
        np.nan,
    )

    event["gasoline_theoretical_price_after"] = (
        event["gasoline_price_before"] + event["f_gasoline"]
    )
    event["diesel_theoretical_price_after"] = (
        event["diesel_price_before"] + event["f_diesel"]
    )

    monthly_keep = [
        "month",
        "cpi_yoy",
        "cpi_mom",
        "ppi_yoy",
        "usd_cny_avg",
        "brent_month_avg_usd_per_barrel",
        "cpi_target",
        "crude_production_ton",
        "crude_import_ton",
        "crude_processing_ton",
        "crude_import_avg_usd_per_ton",
        "gasoline_consumption_ton",
        "diesel_consumption_ton",
        "omega_gasoline",
        "omega_diesel",
        "oil_import_dependency",
        "brent_pressure_h",
        "crude_processing_ma12",
        "processing_shortage_b",
        "energy_dependency_pressure",
        "energy_security_A_raw",
        "energy_imputation_flag",
        "energy_imputation_note",
    ]
    event = event.merge(monthly[monthly_keep], on="month", how="left")

    event["aggregate_actual_price_after"] = (
        event["omega_gasoline"] * event["gasoline_price_after"]
        + event["omega_diesel"] * event["diesel_price_after"]
    )
    event["aggregate_theoretical_price_after"] = (
        event["omega_gasoline"] * event["gasoline_theoretical_price_after"]
        + event["omega_diesel"] * event["diesel_theoretical_price_after"]
    )
    event["price_gap_g"] = (
        (
            event["aggregate_theoretical_price_after"]
            - event["aggregate_actual_price_after"]
        )
        / event["aggregate_theoretical_price_after"]
    ).clip(lower=0.0)

    keep = [
        "date",
        "month",
        "notice_date",
        "effective_date",
        "special_type",
        "gasoline_price_before",
        "diesel_price_before",
        "gasoline_price_after",
        "diesel_price_after",
        "f_gasoline",
        "f_diesel",
        "u_hist_gasoline",
        "u_hist_diesel",
        "alpha_hist_gasoline",
        "alpha_hist_diesel",
        "gasoline_theoretical_price_after",
        "diesel_theoretical_price_after",
        "omega_gasoline",
        "omega_diesel",
        "cpi_yoy",
        "cpi_mom",
        "ppi_yoy",
        "cpi_target",
        "brent_month_avg_usd_per_barrel",
        "crude_production_ton",
        "crude_import_ton",
        "crude_processing_ton",
        "crude_import_avg_usd_per_ton",
        "oil_import_dependency",
        "brent_pressure_h",
        "processing_shortage_b",
        "energy_dependency_pressure",
        "energy_security_A_raw",
        "energy_imputation_flag",
        "energy_imputation_note",
        "aggregate_actual_price_after",
        "aggregate_theoretical_price_after",
        "price_gap_g",
        "notice_title",
        "source_url",
    ]

    return event[keep]


def build_clean_event_model_input(event: pd.DataFrame) -> pd.DataFrame:
    clean = event.copy()
    clean["date"] = pd.to_datetime(clean["date"])
    clean["month"] = pd.to_datetime(clean["month"])

    clean = clean[clean["f_gasoline"].notna() & clean["f_diesel"].notna()]
    clean = clean[clean["date"] != pd.Timestamp("2016-01-14")]
    clean = clean[clean["month"] <= pd.Timestamp("2024-12-01")]

    required = [
        "omega_gasoline",
        "omega_diesel",
        "cpi_yoy",
        "cpi_mom",
        "brent_month_avg_usd_per_barrel",
        "crude_production_ton",
        "crude_import_ton",
        "crude_processing_ton",
        "oil_import_dependency",
    ]
    clean = clean.dropna(subset=required).reset_index(drop=True)
    return clean


def build_forecast_event_model_input(event: pd.DataFrame) -> pd.DataFrame:
    forecast = event.copy()
    forecast["date"] = pd.to_datetime(forecast["date"])
    forecast["month"] = pd.to_datetime(forecast["month"])

    forecast = forecast[forecast["f_gasoline"].notna() & forecast["f_diesel"].notna()]
    forecast = forecast[forecast["date"] != pd.Timestamp("2016-01-14")]
    forecast = forecast[forecast["month"] <= pd.Timestamp("2026-04-01")]

    required = [
        "omega_gasoline",
        "omega_diesel",
        "cpi_yoy",
        "cpi_mom",
        "brent_month_avg_usd_per_barrel",
        "crude_production_ton",
        "crude_import_ton",
        "crude_processing_ton",
        "oil_import_dependency",
    ]
    forecast = forecast.dropna(subset=required).reset_index(drop=True)
    return forecast


def build_audit(monthly: pd.DataFrame, event: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "formula_variable": "alpha_k,t",
            "prepared_column": "decision variable, not observed",
            "status": "to_optimize",
            "note": "Historical alpha columns are included only for backtest/reference.",
        },
        {
            "formula_variable": "f_k,t",
            "prepared_column": "f_gasoline, f_diesel",
            "status": "available",
            "note": "From final price model predicted adjustment.",
        },
        {
            "formula_variable": "u_k,t",
            "prepared_column": "decision variable; u_hist_* for historical actual",
            "status": "available_for_history",
            "note": "Optimization should use u_k,t = alpha_k,t * f_k,t.",
        },
        {
            "formula_variable": "P_k,t-1",
            "prepared_column": "gasoline_price_before, diesel_price_before",
            "status": "available",
            "note": "Derived from domestic regulated price sequence.",
        },
        {
            "formula_variable": "omega_k",
            "prepared_column": "omega_gasoline, omega_diesel",
            "status": "available_with_assumption",
            "note": "Monthly gasoline/diesel apparent-consumption weights; Jan-Feb 2016 use 0.5/0.5 fallback.",
        },
        {
            "formula_variable": "pi_m",
            "prepared_column": "cpi_yoy",
            "status": "available",
            "note": "Stored as decimal, e.g. 0.018 means 1.8%.",
        },
        {
            "formula_variable": "pi_target",
            "prepared_column": "cpi_target",
            "status": "available",
            "note": "Current data uses 0.03.",
        },
        {
            "formula_variable": "Brent_t",
            "prepared_column": "brent_month_avg_usd_per_barrel",
            "status": "available",
            "note": "Monthly average Brent price.",
        },
        {
            "formula_variable": "d_t",
            "prepared_column": "oil_import_dependency",
            "status": "available",
            "note": "crude_import_ton / (crude_production_ton + crude_import_ton).",
        },
        {
            "formula_variable": "h_t",
            "prepared_column": "brent_pressure_h",
            "status": "available",
            "note": "clip((Brent - 80) / 50, 0, 1).",
        },
        {
            "formula_variable": "b_t",
            "prepared_column": "processing_shortage_b",
            "status": "available_after_ma12",
            "note": "First months can be missing because MA12 needs enough observations.",
        },
        {
            "formula_variable": "Q_exp_t",
            "prepared_column": "not prepared",
            "status": "not_needed_current_formula",
            "note": "The simplified formula uses imports, production, and processing; crude export is unavailable and not used.",
        },
        {
            "formula_variable": "missing-value rules",
            "prepared_column": "clean input",
            "status": "applied",
            "note": "Drop 2016-01-14; cap historical backtest at 2024-12; fill crude output/processing Jan-Feb with same-year March; interpolate 2020-01 crude imports; MA12 uses min_periods=3.",
        },
        {
            "formula_variable": "beta_0,beta_g,beta_d,beta_pi",
            "prepared_column": "not estimated yet",
            "status": "still_needed_or_estimate",
            "note": "Needed if CPI loss uses a fitted CPI forecast equation.",
        },
        {
            "formula_variable": "lambda_C,lambda_F,lambda_pi,lambda_V,lambda_E",
            "prepared_column": "not data",
            "status": "set_by_model",
            "note": "Document suggests equal weights 0.2 as a baseline.",
        },
        {
            "formula_variable": "rho,rho_A,theta_1,theta_2",
            "prepared_column": "not data",
            "status": "set_by_model",
            "note": "Document suggests rho=0.7, rho_A=0.7, theta_1=theta_2=1.",
        },
    ]

    audit = pd.DataFrame(rows)

    event_missing = event.isna().sum().rename("missing_event_rows").reset_index()
    event_missing.columns = ["prepared_column", "missing_event_rows"]
    audit = audit.merge(event_missing, on="prepared_column", how="left")

    monthly_missing = monthly.isna().sum().rename("missing_monthly_rows").reset_index()
    monthly_missing.columns = ["prepared_column", "missing_monthly_rows"]
    audit = audit.merge(monthly_missing, on="prepared_column", how="left")

    return audit


def build_missing_detail(monthly: pd.DataFrame, event: pd.DataFrame) -> pd.DataFrame:
    key_columns = [
        "f_gasoline",
        "f_diesel",
        "omega_gasoline",
        "omega_diesel",
        "cpi_yoy",
        "cpi_mom",
        "brent_month_avg_usd_per_barrel",
        "crude_production_ton",
        "crude_import_ton",
        "crude_processing_ton",
        "oil_import_dependency",
        "processing_shortage_b",
        "energy_dependency_pressure",
        "energy_security_A_raw",
    ]

    rows = []
    for col in key_columns:
        monthly_months = []
        if col in monthly.columns:
            monthly_months = monthly.loc[monthly[col].isna(), "month"].dt.strftime(
                "%Y-%m"
            )

        event_dates = []
        if col in event.columns:
            event_dates = event.loc[event[col].isna(), "date"].dt.strftime("%Y-%m-%d")

        rows.append(
            {
                "column": col,
                "missing_month_count": len(monthly_months),
                "missing_months": ";".join(monthly_months),
                "missing_event_count": len(event_dates),
                "missing_event_dates": ";".join(event_dates),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    monthly = build_monthly_model_data()
    event = build_event_model_input(monthly)
    clean_event = build_clean_event_model_input(event)
    clean_monthly = monthly[monthly["month"] <= pd.Timestamp("2024-12-01")].copy()
    forecast_event = build_forecast_event_model_input(event)
    forecast_monthly = monthly[monthly["month"] <= pd.Timestamp("2026-04-01")].copy()
    audit = build_audit(monthly, event)
    missing_detail = build_missing_detail(clean_monthly, clean_event)

    monthly.to_csv(
        RESULT_DIR / "task2_monthly_model_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    event.to_csv(
        RESULT_DIR / "task2_event_model_input.csv",
        index=False,
        encoding="utf-8-sig",
    )
    clean_monthly.to_csv(
        RESULT_DIR / "task2_monthly_model_data_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )
    clean_event.to_csv(
        RESULT_DIR / "task2_event_model_input_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )
    forecast_monthly.to_csv(
        RESULT_DIR / "task2_monthly_model_data_forecast.csv",
        index=False,
        encoding="utf-8-sig",
    )
    forecast_event.to_csv(
        RESULT_DIR / "task2_event_model_input_forecast.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(
        RESULT_DIR / "task2_data_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    missing_detail.to_csv(
        RESULT_DIR / "task2_missing_data_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Generated task-2 model data:")
    print(f"- task2_monthly_model_data.csv: {len(monthly)} rows")
    print(f"- task2_event_model_input.csv: {len(event)} rows")
    print(f"- task2_monthly_model_data_clean.csv: {len(clean_monthly)} rows")
    print(f"- task2_event_model_input_clean.csv: {len(clean_event)} rows")
    print(f"- task2_monthly_model_data_forecast.csv: {len(forecast_monthly)} rows")
    print(f"- task2_event_model_input_forecast.csv: {len(forecast_event)} rows")
    print(f"- task2_data_audit.csv: {len(audit)} rows")
    print(f"- task2_missing_data_detail.csv: {len(missing_detail)} rows")

    key_cols = [
        "f_gasoline",
        "f_diesel",
        "omega_gasoline",
        "omega_diesel",
        "cpi_yoy",
        "crude_production_ton",
        "crude_import_ton",
        "crude_processing_ton",
        "oil_import_dependency",
        "processing_shortage_b",
    ]
    print("\nMissing values in clean event input:")
    print(clean_event[key_cols].isna().sum().to_string())


if __name__ == "__main__":
    main()
