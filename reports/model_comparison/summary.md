# Model comparison summary

- Dataset: `C:\Users\VICTUS\Desktop\ResilienceLearn_UI\data\AI ResilienceLearn (Responses) - Form Responses 1 (1).csv`
- Samples: 196, features: 15
- Positive class rate (train split): 0.449

## Ranking rule
- Primary: **5-fold CV ROC-AUC** (higher is better)
- Tie-breaker: **test F1**

## Best model (by ranking rule)
- **Random Forest**
- CV ROC-AUC: **0.8603**
- Test ROC-AUC: **0.8712**
- Test F1: **0.7895**

## Notes (interpretability)
- Logistic Regression: most interpretable linear baseline.
- Decision Tree: human-readable but can overfit.
- Random Forest / Extra Trees: strong tabular performance + global importances (less transparent locally unless SHAP used).
- Boosting models: often strong; interpretability via SHAP/feature importance.

See `metrics_table.csv`, confusion matrices, and `roc_curves.png`.