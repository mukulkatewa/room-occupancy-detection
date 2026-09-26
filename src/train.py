"""
Train and evaluate a room-occupancy classifier from ambient sensor data.

Uses the canonical UCI split: datatraining.csv for training, datatest.csv for
validation/model-selection, and datatest2.csv as a held-out final test set.

Usage:
    python src/train.py --data-dir data --model-out models/model.joblib
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
TARGET = "Occupancy"
RANDOM_STATE = 42


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates()
    return df


def build_candidates() -> dict:
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        "decision_tree": Pipeline(
            [("model", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE))]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
                    ),
                )
            ]
        ),
    }


def evaluate(model, x, y) -> dict:
    preds = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds)),
        "recall": float(recall_score(y, preds)),
        "f1": float(f1_score(y, preds)),
        "report": classification_report(y, preds, output_dict=True),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-out", default="models/model.joblib")
    parser.add_argument("--metrics-out", default="models/metrics.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_df = load(data_dir / "datatraining.csv")
    val_df = load(data_dir / "datatest.csv")
    holdout_df = load(data_dir / "datatest2.csv")

    x_train, y_train = train_df[FEATURES], train_df[TARGET]
    x_val, y_val = val_df[FEATURES], val_df[TARGET]
    x_holdout, y_holdout = holdout_df[FEATURES], holdout_df[TARGET]

    candidates = build_candidates()
    val_results = {}
    fitted = {}
    for name, pipeline in candidates.items():
        cv_scores = cross_val_score(pipeline, x_train, y_train, cv=5, scoring="f1")
        pipeline.fit(x_train, y_train)
        metrics = evaluate(pipeline, x_val, y_val)
        metrics["cv_f1_mean"] = float(cv_scores.mean())
        metrics["cv_f1_std"] = float(cv_scores.std())
        val_results[name] = metrics
        fitted[name] = pipeline
        print(f"[{name}] val_accuracy={metrics['accuracy']:.4f} val_f1={metrics['f1']:.4f} "
              f"cv_f1={metrics['cv_f1_mean']:.4f}+/-{metrics['cv_f1_std']:.4f}")

    best_name = max(val_results, key=lambda k: val_results[k]["f1"])
    best_model = fitted[best_name]
    holdout_metrics = evaluate(best_model, x_holdout, y_holdout)
    print(f"\nBest model on validation: {best_name}")
    print(f"Holdout (datatest2) accuracy={holdout_metrics['accuracy']:.4f} "
          f"f1={holdout_metrics['f1']:.4f}")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.model_out)

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "validation_results": val_results,
                "holdout_results": holdout_metrics,
            },
            f,
            indent=2,
        )

    print(f"Saved best model to {args.model_out}")
    print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
