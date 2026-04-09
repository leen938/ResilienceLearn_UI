"""One-off: patch notebooks/phase1_preprocessing.ipynb for classification + ml module."""
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    path = root / "notebooks" / "phase1_preprocessing.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]

    def set_code(idx: int, source: str) -> None:
        lines = source.strip("\n").split("\n")
        cells[idx]["source"] = [ln + "\n" for ln in lines]
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None

    set_code(
        0,
        """
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NOTEBOOK_DIR = Path.cwd().resolve()
PROJECT_ROOT = NOTEBOOK_DIR if (NOTEBOOK_DIR / "ml").is_dir() else NOTEBOOK_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "AI ResilienceLearn (Responses) - Form Responses 1 (1).csv"
df = pd.read_csv(DATA_PATH)

df.head()
""",
    )

    set_code(
        4,
        """
# `ml.preprocessing` performs canonical column cleaning. Raw `df` from the CSV is enough here.
df.head(2)
""",
    )

    set_code(
        5,
        """
from ml.preprocessing import MODEL_FEATURE_ORDER, prepare_X_y_classification

X, y, proc = prepare_X_y_classification(df)
print("Features:", list(X.columns))
assert list(X.columns) == list(MODEL_FEATURE_ORDER)
X.shape, y.value_counts()
""",
    )

    set_code(
        6,
        """
# High-cardinality columns are dropped inside `prepare_X_y_classification`.
proc.head(2)
""",
    )

    set_code(
        7,
        """
print("Rows, Columns:", proc.shape)
display(proc.head(3))

missing = proc.isna().sum().sort_values(ascending=False)
display(missing[missing > 0])

uniq = proc.nunique().sort_values(ascending=False)
display(uniq)
""",
    )

    set_code(
        8,
        """
# Superseded: `risk_label` is created inside `prepare_X_y_classification` (same 1–2 vs 3–5 rule).
pass
""",
    )

    set_code(
        9,
        """
# Superseded: performance column is dropped inside shared preprocessing to avoid leakage.
pass
""",
    )

    set_code(
        10,
        """
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
""",
    )

    set_code(
        11,
        """
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
lr_model.fit(X_train, y_train)
y_proba_lr = lr_model.predict_proba(X_test)[:, 1]
y_pred_lr = lr_model.predict(X_test)

print("Logistic Regression (baseline)")
print("  accuracy :", accuracy_score(y_test, y_pred_lr))
print("  precision:", precision_score(y_test, y_pred_lr, zero_division=0))
print("  recall   :", recall_score(y_test, y_pred_lr, zero_division=0))
print("  f1       :", f1_score(y_test, y_pred_lr, zero_division=0))
try:
    print("  roc_auc  :", roc_auc_score(y_test, y_proba_lr))
except ValueError as e:
    print("  roc_auc  : n/a ({})".format(e))
""",
    )

    set_code(
        12,
        """
# Train / test sets are shared with the baseline above.
""",
    )

    set_code(
        13,
        """
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = rf_model.predict(X_test)

print("Random Forest (main)")
print("  accuracy :", accuracy_score(y_test, y_pred_rf))
print("  precision:", precision_score(y_test, y_pred_rf, zero_division=0))
print("  recall   :", recall_score(y_test, y_pred_rf, zero_division=0))
print("  f1       :", f1_score(y_test, y_pred_rf, zero_division=0))
try:
    print("  roc_auc  :", roc_auc_score(y_test, y_proba_rf))
except ValueError as e:
    print("  roc_auc  : n/a ({})".format(e))
""",
    )

    save_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            ln + "\n"
            for ln in """
import joblib

ART = PROJECT_ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

payload = {
    "model": rf_model,
    "feature_names": list(X.columns),
    "target": "risk_label",
    "positive_class": 1,
    "positive_label": "at_risk",
}
joblib.dump(payload, ART / "model.joblib")
print("Saved", ART / "model.joblib")
""".strip("\n").split("\n")
        ],
    }
    cells.insert(14, save_cell)

    feat_idx = 16
    set_code(
        feat_idx,
        """
import pandas as pd

importance = rf_model.feature_importances_
feat_imp = pd.Series(importance, index=X.columns).sort_values(ascending=False)

print("Feature importance (Random Forest)")
print(feat_imp)
""",
    )

    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Patched", path, "cells:", len(nb["cells"]))


if __name__ == "__main__":
    main()
