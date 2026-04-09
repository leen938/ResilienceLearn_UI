"""Train RF classifier on survey CSV and save artifacts/model.joblib (used if notebook not run)."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.preprocessing import prepare_X_y_classification  # noqa: E402


def main() -> None:
    data_path = ROOT / "data" / "AI ResilienceLearn (Responses) - Form Responses 1 (1).csv"
    df = pd.read_csv(data_path)
    X, y, _proc = prepare_X_y_classification(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    feature_importances = {
        name: float(imp) for name, imp in zip(X.columns, rf.feature_importances_, strict=True)
    }
    feature_medians = X_train.median().astype(float).to_dict()

    art = ROOT / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": rf,
        "feature_names": list(X.columns),
        "target": "risk_label",
        "positive_class": 1,
        "positive_label": "at_risk",
        "feature_importances": feature_importances,
        "feature_medians": feature_medians,
    }
    joblib.dump(payload, art / "model.joblib")
    print("Saved", art / "model.joblib", "test_score", rf.score(X_test, y_test))


if __name__ == "__main__":
    main()
