"""Part A - clean the raw export, build the modelling table and the delay evidence files."""
import os

import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, clean_shipments, build_features

DIST_BINS = [0, 200, 400, 600, 2000]
DIST_LABELS = ["0-200", "201-400", "401-600", "601+"]


def r4(x):
    """Round floats to 4 places"""
    if isinstance(x, (int, np.integer)):
        return int(x)
    return round(float(x), 4)


def main():
    os.makedirs("outputs", exist_ok=True)
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # before counts after removing duplicates
    dedup = raw.drop_duplicates()
    before = {
        "missing_distance_km_before": int(dedup["distance_km"].isna().sum()),
        "missing_parcel_weight_before": int(dedup["parcel_weight_kg"].isna().sum()),
        "sentinel_weight_rows": int((dedup["parcel_weight_kg"] > 100).sum()),
        "missing_declared_value_before": int(dedup["declared_value"].isna().sum()),
        "missing_payment_mode_before": int(dedup["payment_mode"].isna().sum()),
        "service_type_spellings_before": int(dedup["service_type"].nunique()),
        "service_type_spellings_after": int(dedup["service_type"].str.strip().str.title().nunique()),
    }

    clean, medians = clean_shipments(raw)
    table = build_features(clean, hubs)
    table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
    table.to_csv("data/modelling_table.csv", index=False)

    # checks
    hours = table["delivery_hours"]
    q1, q3 = hours.quantile(0.25), hours.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    checks = {
        "raw_rows": len(raw),
        "duplicate_rows_removed": len(raw) - len(dedup),
        "rows_after_dedup": len(clean),
        **before,
        "median_distance_km": medians["distance_km"],
        "median_parcel_weight_kg": medians["parcel_weight_kg"],
        "median_declared_value_inr": medians["declared_value_inr"],
        "rows_after_hub_merge": len(table),
        "nulls_in_features_and_target": int(table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
        "delayed_count": int(table["is_delayed"].sum()),
        "delay_rate": table["is_delayed"].mean(),
        "hours_q1": q1,
        "hours_q3": q3,
        "hours_lower_bound": lower,
        "hours_upper_bound": upper,
        "hours_outlier_count": int(((hours < lower) | (hours > upper)).sum()),
        "declared_value_mean_inr": table["declared_value_inr"].mean(),
        "declared_value_median_inr": table["declared_value_inr"].median(),
        "log_declared_value_mean": table["log_declared_value"].mean(),
        "log_declared_value_median": table["log_declared_value"].median(),
    }
    pd.DataFrame({
        "metric": list(checks.keys()),
        "value": pd.Series([r4(v) for v in checks.values()], dtype=object),
    }).to_csv("outputs/part_a_checks.csv", index=False)

    # delay evidence by group
    table["distance_band"] = pd.cut(table["distance_km"], bins=DIST_BINS, labels=DIST_LABELS, right=True)
    parts = []
    for dim in ["service_type", "hub_tier", "distance_band"]:
        g = (table.groupby(dim, observed=False)
             .agg(shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
                  delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"))
             .reset_index()
             .rename(columns={dim: "group"}))
        g.insert(0, "dimension", dim)
        g["group"] = g["group"].astype(str)
        parts.append(g)
    by_group = pd.concat(parts, ignore_index=True)
    by_group[["delay_rate", "mean_hours"]] = by_group[["delay_rate", "mean_hours"]].round(4)
    by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

    # correlations of the numeric columns of the modelling table
    numeric = table.drop(columns=["distance_band"]).select_dtypes(include="number")
    numeric.corr().round(4).to_csv("outputs/part_a_correlations.csv", index_label="column")

    print(pd.read_csv("outputs/part_a_checks.csv").to_string(index=False))
    print(by_group.to_string(index=False))


if __name__ == "__main__":
    main()
