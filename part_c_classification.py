"""Part C - delay classifier, feature selection, tuning and the alert threshold"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS

TARGET = "is_delayed"
TREE_DEPTHS = [2, 3, 4, 5, 6, 8, 10]
K_VALUES = [5, 7, 10, 12, 17]
SELECTOR_K = 7
PARAM_GRID = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
DEFAULT_PARAMS = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MIN_RECALL = 0.80
TIE_ORDER = ["default_forest_same_7_features", "grid_search", "randomized_search"]


def classification_row(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


def params_text(params):
    return str({key: params[key] for key in PARAM_GRID})


def main():
    os.makedirs("outputs", exist_ok=True)
    table = pd.read_csv("data/modelling_table.csv")
    X, y = table[FEATURES], table[TARGET]

    # 1. stratified 70 / 15 / 15 split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print("sizes:", len(X_train), len(X_val), len(X_test),
          "delayed:", int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))

    # 2. blind baseline and three classifiers at the 0.5 rule
    model_rows = [{"model": "blind_all_on_time", **classification_row(y_val, np.zeros(len(y_val), dtype=int))}]
    logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
    model_rows.append({"model": "logistic_regression", **classification_row(y_val, logreg.predict(X_val))})
    tree_rows = {}
    for d in TREE_DEPTHS:
        tree = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
        tree_rows[d] = classification_row(y_val, tree.predict(X_val))
    best_d = max(TREE_DEPTHS, key=lambda d: (tree_rows[d]["f1"], -d))
    model_rows.append({"model": f"decision_tree_depth_{best_d}", **tree_rows[best_d]})
    rf_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    model_rows.append({"model": "random_forest_default", **classification_row(y_val, rf_default.predict(X_val))})
    pd.DataFrame(model_rows).round(4).to_csv("outputs/part_c_models.csv", index=False)

    # 3. signal versus noise
    f_scores = SelectKBest(f_classif, k="all").fit(X_train, y_train).scores_
    scores = pd.DataFrame({"feature": FEATURES, "f_score": f_scores,
                           "rf_importance": rf_default.feature_importances_})
    scores.insert(2, "f_rank", scores["f_score"].rank(ascending=False, method="min").astype(int))
    scores["is_noise"] = scores["feature"].isin(NOISE_COLUMNS)
    scores.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)

    selectk_rows = []
    for k in K_VALUES:
        pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                         ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
        selectk_rows.append({"k": k, "val_f1": f1_score(y_val, pipe.predict(X_val))})
    pd.DataFrame(selectk_rows).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

    # 4. tuning on the 7 selected columns
    selector = SelectKBest(f_classif, k=SELECTOR_K).fit(X_train, y_train)
    X_train_k, X_val_k = selector.transform(X_train), selector.transform(X_val)
    grid = GridSearchCV(RandomForestClassifier(random_state=42), PARAM_GRID, cv=5).fit(X_train_k, y_train)
    rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), PARAM_GRID, n_iter=9, cv=5,
                              random_state=42).fit(X_train_k, y_train)
    default_k = RandomForestClassifier(random_state=42, **DEFAULT_PARAMS).fit(X_train_k, y_train)
    tuning = pd.DataFrame([
        {"search": "grid_search", "candidates": len(grid.cv_results_["params"]), "cv_folds": 5,
         "best_params": params_text(grid.best_params_), "best_cv_accuracy": grid.best_score_,
         "val_f1": f1_score(y_val, grid.predict(X_val_k)), "_params": grid.best_params_},
        {"search": "randomized_search", "candidates": len(rand.cv_results_["params"]), "cv_folds": 5,
         "best_params": params_text(rand.best_params_), "best_cv_accuracy": rand.best_score_,
         "val_f1": f1_score(y_val, rand.predict(X_val_k)), "_params": rand.best_params_},
        {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0,
         "best_params": params_text(DEFAULT_PARAMS), "best_cv_accuracy": np.nan,
         "val_f1": f1_score(y_val, default_k.predict(X_val_k)), "_params": DEFAULT_PARAMS},
    ])
    tuning["total_fits"] = np.where(tuning["cv_folds"] > 0, tuning["candidates"] * tuning["cv_folds"], 1)
    out_cols = ["search", "candidates", "cv_folds", "total_fits", "best_params", "best_cv_accuracy", "val_f1"]
    tuning[out_cols].round(4).to_csv("outputs/part_c_tuning.csv", index=False)

    # 5. final configuration: highest val_f1, ties broken by TIE_ORDER
    by_name = tuning.set_index("search")
    chosen_from = max(TIE_ORDER, key=lambda s: (round(by_name.at[s, "val_f1"], 10), -TIE_ORDER.index(s)))
    chosen = by_name.at[chosen_from, "_params"]
    final_model = Pipeline([("selector", SelectKBest(f_classif, k=SELECTOR_K)),
                            ("model", RandomForestClassifier(random_state=42, **chosen))]).fit(X_train, y_train)

    # 6. alert threshold from validation probabilities
    val_proba = final_model.predict_proba(X_val)[:, 1]
    thr = pd.DataFrame([{"threshold": t, **classification_row(y_val, (val_proba >= t).astype(int))}
                        for t in THRESHOLDS])
    thr.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)
    passing = thr.loc[thr["recall"] >= MIN_RECALL, "threshold"]
    threshold = float(passing.max()) if len(passing) else 0.1

    # 7. single test evaluation
    test_pred = (final_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    final_test = pd.DataFrame([{"threshold": threshold, **classification_row(y_test, test_pred)}])
    final_test.round(4).to_csv("outputs/part_c_final_test.csv", index=False)
    pd.DataFrame([{"selector_k": SELECTOR_K, "n_estimators": chosen["n_estimators"],
                   "max_depth": "unlimited" if chosen["max_depth"] is None else chosen["max_depth"],
                   "min_samples_leaf": chosen["min_samples_leaf"], "threshold": threshold,
                   "chosen_from": chosen_from}]).to_csv("outputs/part_c_final_config.csv", index=False)

    print(pd.DataFrame(model_rows).round(4).to_string(index=False))
    print(scores.sort_values("f_rank").round(4).to_string(index=False))
    print("noise importance share:", round(scores.loc[scores["is_noise"], "rf_importance"].sum(), 4))
    print(pd.DataFrame(selectk_rows).round(4).to_string(index=False))
    print(tuning[out_cols].round(4).to_string(index=False))
    print("chosen:", chosen_from, chosen)
    print(thr.round(4).to_string(index=False))
    print(final_test.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
