"""Score new parcels and turn them into an action list for operations"""
import os

import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import clean_shipments, build_features

ESCALATE_SEGMENT = "Escalate"


def main():
    os.makedirs("outputs", exist_ok=True)
    new = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    model = joblib.load("models/delay_model.pkl")
    threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    impute = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute["column"], impute["median"].astype(float)))
    segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

    clean, _ = clean_shipments(new, medians=medians)
    features = build_features(clean, hubs)

    scored = features[["shipment_id", "hub_id"]].copy()
    scored["delay_probability"] = model.predict_proba(features[feature_order])[:, 1].round(4)
    scored["flagged"] = (scored["delay_probability"] >= threshold).astype(int)
    scored = scored.merge(segments, on="hub_id", how="left", validate="many_to_one")
    scored["action"] = np.select(
        [(scored["flagged"] == 1) & (scored["segment"] == ESCALATE_SEGMENT), scored["flagged"] == 1],
        ["Escalate to hub manager", "Proactive customer call"],
        default="No action",
    )

    actions = scored[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
    actions.to_csv("outputs/new_shipment_actions.csv", index=False)
    print(f"{len(actions)} parcels scored, {int(actions['flagged'].sum())} flagged")
    print(actions["action"].value_counts().to_string())


if __name__ == "__main__":
    main()
