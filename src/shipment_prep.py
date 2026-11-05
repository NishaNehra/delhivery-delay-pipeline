"""Shared constants and the cleaning / feature functions used by every Part."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
WEIGHT_SENTINEL_LIMIT = 100


def clean_shipments(raw, medians=None):
    """Clean a raw shipments export"""
    df = raw.copy()

    # 1. duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. service_type extra space remove and title case
    df["service_type"] = df["service_type"].str.strip().str.title()

    # 3. payment_mode fill na with Unknown
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. declared_value text to number
    value_text = (df["declared_value"].astype("string")
                  .str.replace("\u20b9", "", regex=False)
                  .str.replace(",", "", regex=False)
                  .str.strip())
    df["declared_value_inr"] = pd.to_numeric(value_text, errors="coerce").astype(float)
    df = df.drop(columns=["declared_value"])

    # 5. dispatch_time to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], format="%Y-%m-%d %H:%M")

    # 6. scanner-fault sentinel weights handling
    df["parcel_weight_kg"] = df["parcel_weight_kg"].astype(float)
    df.loc[df["parcel_weight_kg"] > WEIGHT_SENTINEL_LIMIT, "parcel_weight_kg"] = np.nan

    # 7. median fill
    if medians is None:
        medians = {col: float(df[col].median()) for col in MEDIAN_COLUMNS}
    else:
        medians = {col: float(medians[col]) for col in MEDIAN_COLUMNS}
    for col in MEDIAN_COLUMNS:
        df[col] = df[col].astype(float).fillna(medians[col])

    return df, medians


def build_features(clean, hubs):
    """Attach hub columns and derive model features"""
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    df = clean.merge(hubs[hub_cols], on="hub_id", how="left", validate="many_to_one")

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.day_of_week >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df
