"""
Quick train-vs-val diagnostic to spot overfitting.

Usage:
    python scripts/analysis/check_overfitting.py \
        --data data/final/sample_dataset_1000_balanced_normalized.csv

Outputs mean ± std for train/val metrics across stratified CV folds.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV with features + has_promotion target")
    parser.add_argument("--target", default="has_promotion", help="Target column name")
    parser.add_argument("--folds", type=int, default=5, help="Stratified CV folds")
    return parser.parse_args()


def build_pipeline(cat_cols, num_cols, model):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def metric_dict(y_true, y_prob, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def evaluate_model(df, target_col, model, folds):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pipe = build_pipeline(cat_cols, num_cols, model)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    train_metrics = []
    val_metrics = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe.fit(X_train, y_train)

        y_train_prob = pipe.predict_proba(X_train)[:, 1]
        y_train_pred = (y_train_prob >= 0.5).astype(int)
        train_metrics.append(metric_dict(y_train, y_train_prob, y_train_pred))

        y_val_prob = pipe.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_prob >= 0.5).astype(int)
        val_metrics.append(metric_dict(y_val, y_val_prob, y_val_pred))

    return train_metrics, val_metrics


def summarize(metrics_list):
    keys = metrics_list[0].keys()
    summary = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics_list])
        summary[k] = (vals.mean(), vals.std())
    return summary


def print_summary(name, train_summary, val_summary):
    print(f"\n=== {name} ===")
    print("Train metrics (mean ± std):")
    for k, (mean, std) in train_summary.items():
        print(f"  {k:9s}: {mean:0.4f} ± {std:0.4f}")
    print("Validation metrics (mean ± std):")
    for k, (mean, std) in val_summary.items():
        print(f"  {k:9s}: {mean:0.4f} ± {std:0.4f}")


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in data.")

    models = [
        ("LogReg", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("RandomForest", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ]

    for name, model in models:
        train_metrics, val_metrics = evaluate_model(df, args.target, model, args.folds)
        train_summary = summarize(train_metrics)
        val_summary = summarize(val_metrics)
        print_summary(name, train_summary, val_summary)

        # Quick diagnostic: large gaps between train and val indicate overfit
        print("  Gap (train - val) for F1 and ROC-AUC:")
        print(f"    F1 gap     : {train_summary['f1'][0] - val_summary['f1'][0]:0.4f}")
        print(f"    ROC-AUC gap: {train_summary['roc_auc'][0] - val_summary['roc_auc'][0]:0.4f}")


if __name__ == "__main__":
    main()
