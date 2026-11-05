"""Part B - delivery-hours regression on the Part A modelling table"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES

TARGET = "delivery_hours"
DEPTHS = [1, 2, 3, 4, 6, 8, 12, None]
LEARNING_RATES = [0.01, 0.1, 1.0]


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    os.makedirs("outputs", exist_ok=True)
    table = pd.read_csv("data/modelling_table.csv")
    X, y = table[FEATURES], table[TARGET]

    # 1. split 70 / 15 / 15
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 2. scaler fitted on training rows only
    scaler = StandardScaler().fit(X_train)
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)
    pd.DataFrame({
        "metric": ["train_rows", "val_rows", "test_rows",
                   "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
        "value": pd.Series([len(X_train), len(X_val), len(X_test),
                            round(float(np.abs(train_scaled.mean(axis=0)).max()), 4),
                            round(float(np.abs(test_scaled.mean(axis=0)).max()), 4)], dtype=object),
    }).to_csv("outputs/part_b_checks.csv", index=False)

    # 3. mean baseline
    metric_rows = [{"model": "mean_baseline", "split": "val",
                    **regression_metrics(y_val, np.full(len(y_val), y_train.mean()))}]

    # 4. tree-depth diagnostic
    depth_rows = []
    for d in DEPTHS:
        tree = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
        depth_rows.append({"max_depth": "unlimited" if d is None else d,
                           "train_rmse": rmse(y_train, tree.predict(X_train)),
                           "val_rmse": rmse(y_val, tree.predict(X_val))})
    depth_df = pd.DataFrame(depth_rows)
    best_depth = DEPTHS[int(depth_df["val_rmse"].values.argmin())]
    depth_df.round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)

    # 5. four candidates scored on validation
    candidates = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }
    val_rmse = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        m = regression_metrics(y_val, model.predict(X_val))
        val_rmse[name] = m["RMSE"]
        metric_rows.append({"model": name, "split": "val", **m})

    # 6. learning-rate experiment
    lr_rows = []
    for lr in LEARNING_RATES:
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
        pred = gb.predict(X_val)
        lr_rows.append({"learning_rate": lr, "val_rmse": rmse(y_val, pred), "val_r2": r2_score(y_val, pred)})
    pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

    # 7. 5-fold CV of the two best, then a single test score for the winner
    ranking = sorted(val_rmse, key=val_rmse.get)
    X_trval, y_trval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    for name in ranking[:2]:
        scores = cross_val_score(candidates[name], X_trval, y_trval, cv=kfold)
        cv_rows.append({"model": name, "mean_r2": scores.mean(), "std_r2": scores.std()})
    pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)

    final_name = ranking[0]
    metric_rows.append({"model": final_name, "split": "test",
                        **regression_metrics(y_test, candidates[final_name].predict(X_test))})
    metrics_df = pd.DataFrame(metric_rows).round(4)
    metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)

    print("Validation ranking (RMSE):", ranking)
    print(metrics_df.to_string(index=False))
    print(depth_df.round(4).to_string(index=False))
    print(pd.DataFrame(lr_rows).round(4).to_string(index=False))
    print(pd.DataFrame(cv_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
