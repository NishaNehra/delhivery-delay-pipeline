"""Part D - refit Part C's final classifier and save it as a reusable artefact in models"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments

TARGET = "is_delayed"
MODEL_PATH = "models/delay_model.pkl"


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
    max_depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
    threshold = float(config["threshold"])

    # same stratified split as Part C
    table = pd.read_csv("data/modelling_table.csv")
    X, y = table[FEATURES], table[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    model = Pipeline([
        ("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
        ("model", RandomForestClassifier(random_state=42, n_estimators=int(config["n_estimators"]),
                                         max_depth=max_depth, min_samples_leaf=int(config["min_samples_leaf"]))),
    ]).fit(X_train, y_train)

    _, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))

    joblib.dump(model, MODEL_PATH)
    pd.DataFrame({"threshold": [round(threshold, 4)]}).to_csv("models/delay_threshold.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
    pd.DataFrame({"column": list(medians.keys()),
                  "median": [round(v, 4) for v in medians.values()]}).to_csv("models/impute_values.csv", index=False)

    # reload check on test rows
    reloaded = joblib.load(MODEL_PATH)
    diff = np.abs(reloaded.predict_proba(X_test)[:, 1] - model.predict_proba(X_test)[:, 1]).max()
    pd.DataFrame({
        "metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
        "value": pd.Series([len(X_test), round(float(diff), 4), round(threshold, 4)], dtype=object),
    }).to_csv("outputs/part_d_reload_check.csv", index=False)
    print(pd.read_csv("outputs/part_d_reload_check.csv").to_string(index=False))


if __name__ == "__main__":
    main()
