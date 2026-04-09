"""
Model evaluation + comparison for FYP presentation.

Constraints:
- Reuse shared preprocessing only: ml.preprocessing.prepare_X_y_classification
- Do NOT modify backend/UI artifacts here.

Outputs (thesis-friendly):
- reports/model_comparison/metrics_table.csv
- reports/model_comparison/metrics_table.json
- reports/model_comparison/roc_curves.png
- reports/model_comparison/confusion_<model>.png
- reports/model_comparison/summary.md
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless-safe on Windows (avoid Tkinter backend issues)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.preprocessing import prepare_X_y_classification  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "AI ResilienceLearn (Responses) - Form Responses 1 (1).csv"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    needs_scaling: bool = False
    supports_class_weight: bool = False


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _get_proba_or_score(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Returns a 1D score for the positive class (risk_label==1).
    Prefers predict_proba; falls back to decision_function.
    """
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return np.ravel(proba)
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        s = np.ravel(s)
        # Convert unbounded scores to (0,1) via logistic for AUC/ROC plotting consistency
        return 1.0 / (1.0 + np.exp(-s))
    # Worst case: use hard predictions (AUC becomes less meaningful)
    return np.asarray(estimator.predict(X), dtype=float)


def _build_pipeline(spec: ModelSpec) -> Any:
    est = spec.estimator
    if spec.needs_scaling:
        return Pipeline([("scaler", StandardScaler()), ("model", est)])
    return est


def _confusion_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["on_track(0)", "at_risk(1)"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=140)
    disp.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _safe_name(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")


def main() -> int:
    data_path = Path(os.environ.get("RESILIENCE_DATA_PATH", str(DEFAULT_DATA)))
    out_dir = ROOT / "reports" / "model_comparison"
    _ensure_dir(out_dir)

    df = pd.read_csv(data_path)
    X, y, _proc = prepare_X_y_classification(df)

    y = np.asarray(y, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Basic imbalance hint (used for class_weight where supported)
    pos_rate = float(np.mean(y_train == 1))
    use_balanced = (pos_rate < 0.4) or (pos_rate > 0.6)

    specs: list[ModelSpec] = [
        ModelSpec("Baseline (most_frequent)", DummyClassifier(strategy="most_frequent")),
        ModelSpec(
            "Logistic Regression",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced" if use_balanced else None,
                solver="lbfgs",
            ),
            needs_scaling=True,
        ),
        ModelSpec(
            "Decision Tree",
            DecisionTreeClassifier(
                max_depth=None,
                class_weight="balanced" if use_balanced else None,
                random_state=42,
            ),
        ),
        ModelSpec(
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced" if use_balanced else None,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "Extra Trees",
            ExtraTreesClassifier(
                n_estimators=600,
                class_weight="balanced" if use_balanced else None,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=42),
        ),
        ModelSpec(
            "HistGradientBoosting",
            HistGradientBoostingClassifier(random_state=42),
        ),
        ModelSpec(
            "AdaBoost",
            AdaBoostClassifier(random_state=42),
        ),
        ModelSpec(
            "KNN",
            KNeighborsClassifier(n_neighbors=7),
            needs_scaling=True,
        ),
        ModelSpec(
            "SVM (RBF, calibrated)",
            # SVC(probability=True) is slower; calibration is explicit + stable for AUC.
            CalibratedClassifierCV(
                estimator=SVC(
                    kernel="rbf",
                    C=2.0,
                    gamma="scale",
                    class_weight="balanced" if use_balanced else None,
                ),
                method="sigmoid",
                cv=3,
            ),
            needs_scaling=True,
        ),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows: list[dict[str, Any]] = []
    roc_lines: list[tuple[str, np.ndarray, np.ndarray, float]] = []

    for spec in specs:
        est = _build_pipeline(spec)
        t0 = time.time()
        est.fit(X_train, y_train)
        train_ms = int((time.time() - t0) * 1000)

        y_pred = est.predict(X_test)
        y_score = _get_proba_or_score(est, X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_test, y_score))
        except Exception:
            auc = float("nan")

        # Cross-validated ROC-AUC (more stable for small datasets)
        try:
            cv_scores = cross_val_predict(
                est,
                X,
                y,
                cv=cv,
                method="predict_proba",
                n_jobs=None,
            )
            cv_auc = float(roc_auc_score(y, cv_scores[:, 1]))
        except Exception:
            cv_auc = float("nan")

        rows.append(
            {
                "model": spec.name,
                "train_ms": train_ms,
                "test_accuracy": round(acc, 4),
                "test_precision": round(prec, 4),
                "test_recall": round(rec, 4),
                "test_f1": round(f1, 4),
                "test_roc_auc": round(auc, 4) if np.isfinite(auc) else None,
                "cv_roc_auc_5fold": round(cv_auc, 4) if np.isfinite(cv_auc) else None,
                "notes": "class_weight=balanced" if ("balanced" in str(spec.estimator)) else "",
            }
        )

        # Save confusion matrix plot
        _confusion_plot(
            y_test,
            y_pred,
            title=f"Confusion Matrix — {spec.name}",
            out_path=out_dir / f"confusion_{_safe_name(spec.name)}.png",
        )

        if np.isfinite(auc):
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_lines.append((spec.name, fpr, tpr, auc))

    metrics_df = pd.DataFrame(rows)
    # Rank primarily by CV ROC-AUC, then test F1 (helps small datasets)
    metrics_df["rank_key_auc"] = metrics_df["cv_roc_auc_5fold"].fillna(-1.0)
    metrics_df["rank_key_f1"] = metrics_df["test_f1"].fillna(-1.0)
    metrics_df = metrics_df.sort_values(
        by=["rank_key_auc", "rank_key_f1"], ascending=False
    ).drop(columns=["rank_key_auc", "rank_key_f1"])

    metrics_csv = out_dir / "metrics_table.csv"
    metrics_json = out_dir / "metrics_table.json"
    metrics_df.to_csv(metrics_csv, index=False)
    metrics_json.write_text(json.dumps(metrics_df.to_dict(orient="records"), indent=2), encoding="utf-8")

    # ROC curves plot
    if roc_lines:
        fig = plt.figure(figsize=(7.2, 5.2), dpi=150)
        for name, fpr, tpr, auc in roc_lines:
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (test split)")
        plt.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        fig.savefig(out_dir / "roc_curves.png")
        plt.close(fig)

    # Written summary
    best = metrics_df.iloc[0].to_dict()
    summary_md = out_dir / "summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Model comparison summary",
                "",
                f"- Dataset: `{data_path}`",
                f"- Samples: {int(X.shape[0])}, features: {int(X.shape[1])}",
                f"- Positive class rate (train split): {pos_rate:.3f}",
                "",
                "## Ranking rule",
                "- Primary: **5-fold CV ROC-AUC** (higher is better)",
                "- Tie-breaker: **test F1**",
                "",
                "## Best model (by ranking rule)",
                f"- **{best.get('model')}**",
                f"- CV ROC-AUC: **{best.get('cv_roc_auc_5fold')}**",
                f"- Test ROC-AUC: **{best.get('test_roc_auc')}**",
                f"- Test F1: **{best.get('test_f1')}**",
                "",
                "## Notes (interpretability)",
                "- Logistic Regression: most interpretable linear baseline.",
                "- Decision Tree: human-readable but can overfit.",
                "- Random Forest / Extra Trees: strong tabular performance + global importances (less transparent locally unless SHAP used).",
                "- Boosting models: often strong; interpretability via SHAP/feature importance.",
                "",
                "See `metrics_table.csv`, confusion matrices, and `roc_curves.png`.",
            ]
        ),
        encoding="utf-8",
    )

    print("Wrote", out_dir)
    print(metrics_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

