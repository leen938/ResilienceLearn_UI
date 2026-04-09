import json
from pathlib import Path

NEW_SOURCE = r'''import joblib

ART = PROJECT_ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

_feature_importances = {
    name: float(imp)
    for name, imp in zip(X.columns, rf_model.feature_importances_, strict=True)
}
_feature_medians = X_train.median().astype(float).to_dict()

payload = {
    "model": rf_model,
    "feature_names": list(X.columns),
    "target": "risk_label",
    "positive_class": 1,
    "positive_label": "at_risk",
    "feature_importances": _feature_importances,
    "feature_medians": _feature_medians,
}
joblib.dump(payload, ART / "model.joblib")
print("Saved", ART / "model.joblib")
'''


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    p = root / "notebooks" / "phase1_preprocessing.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        if "joblib.dump(payload" in "".join(c.get("source", [])):
            nb["cells"][i]["source"] = [line + "\n" for line in NEW_SOURCE.strip().split("\n")]
            nb["cells"][i]["outputs"] = []
            nb["cells"][i]["execution_count"] = None
            p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
            print("Patched cell", i)
            return
    raise SystemExit("cell not found")


if __name__ == "__main__":
    main()
