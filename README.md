# ResilienceLearn (FYP)

## Overview

ResilienceLearn is a **crisis-aware academic risk** prototype: a binary classifier (**on track** vs **at risk**) trained on survey data, exposed through a **FastAPI** backend, and consumed by a **Streamlit** dashboard. All training-time preprocessing, feature order, and crisis exposure logic live in **`ml/`** so the notebook, batch training script, and API stay aligned.

**Support chat** is **rule-based** (no external LLM): it offers emotional validation and gentle coping ideas. It is **not** therapy, **not** a clinical tool, and **not** a substitute for professional or emergency help.

## Architecture

```text
data/*.csv
    → notebooks/phase1_preprocessing.ipynb  OR  scripts/train_and_save_model.py
        → ml/preprocessing.py (prepare_X_y_classification, CEI, feature order)
        → artifacts/model.joblib  { model, feature_names, feature_importances, feature_medians }

streamlit_app_ui_only.py
    → maps sidebar → ml/ui_mapping.ui_checkin_to_predict_body
    → HTTP client → FastAPI (single source of truth for predictions & explanations)

backend/app.py
    → loads artifacts/model.joblib
    → /predict, /explain, /feature-importance, /chat/support
```

- **Backend** is the **single source of truth** for model outputs, SHAP/approx explanations, and chat replies.  
- **Streamlit** does **not** run the model locally; it only gathers inputs, calls the API, charts results, and shows cached explainability responses.

## Dataset location

Raw responses (Google Form export) are expected at:

`data/AI ResilienceLearn (Responses) - Form Responses 1 (1).csv`

The training script `scripts/train_and_save_model.py` reads this path by default. You can swap the file if you preserve the column semantics expected by `ml/preprocessing.py`.

## Shared preprocessing

- **`ml/preprocessing.py`** — survey cleaning, ordinals, `crisis_exposure_index`, `MODEL_FEATURE_ORDER`, and `prepare_X_y_classification` used for training and analysis.  
- **`ml/features.py`** — CEI computation shared with **`ml/ui_mapping.py`** so dashboard-derived CEI matches training logic.  
- **`notebooks/preprocessing.py`** (under `notebooks/`) is a **legacy / teaching** helper inside the notebook tree; **authoritative** pipeline code for the project is under **`ml/`**.

## Training workflow

1. Install dependencies (see below).  
2. **Notebook:** open `notebooks/phase1_preprocessing.ipynb`, run cells through training; ensure the save step writes `artifacts/model.joblib` (or run `scripts/patch_notebook_joblib_cell.py` if you are syncing the notebook save cell).  
3. **Script (alternative):** from the repo root:
   ```bash
   python scripts/train_and_save_model.py
   ```
   This fits a `RandomForestClassifier`, embeds `feature_importances_` and training medians, and saves `artifacts/model.joblib`.

## Environment consistency (train and serve)

- Use the **same Python environment** and the **same `scikit-learn` version** for training and for running `backend` / `streamlit`.  
- The artefact is **pickled**; loading with a different `scikit-learn` can trigger warnings or incorrect behaviour.  
- This repo pins **`scikit-learn==1.8.0`** in `requirements.txt`. After changing it, **retrain** and overwrite `artifacts/model.joblib`.

## Run the API

From the **project root**:

```bash
pip install -r requirements.txt
uvicorn backend.app:app --reload --app-dir .
```

- Interactive docs: `http://127.0.0.1:8000/docs`  
- Optional: set **`RESILIENCE_MODEL_PATH`** to a different `model.joblib`.

## Run the Streamlit UI

With the API running (recommended), from the project root:

```bash
python -m streamlit run streamlit_app_ui_only.py
```

(Optional) Set **`RESILIENCE_API_BASE`** (default in the app: `http://127.0.0.1:8000`).

On Windows, if `streamlit` is not on `PATH`, `python -m streamlit` is the reliable form.

## HTTP API (summary)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness + whether the model artifact loaded |
| `POST` | `/predict` | Risk label, probability, uncertainty text, fast approximate top factors |
| `GET` | `/feature-importance` | Global Gini importances from the saved forest |
| `POST` | `/explain` | Local explanation (Tree SHAP when available, else same-family approximate ranker) |
| `POST` | `/chat/support` | Supportive reply (OpenAI-backed if enabled; otherwise fallback rule-based); optional `context` + `history` |

Request/response schemas are documented in OpenAPI (`/docs`).

## Support chat (safety note)

Responses are **empathetic and non-diagnostic**. They **do not** provide therapy, diagnosis, or treatment. For crisis situations, users should contact **local emergency services** or a **human crisis line**; the app includes keyword-based escalation messaging, which is **not** a substitute for professional care.

## Optional: Enable OpenAI for Support Chat

By default, support chat uses a local, rule-based fallback.

To enable OpenAI for `POST /chat/support`:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a local `.env` in the repo root (do not commit it) with:

```text
OPENAI_API_KEY=your_key_here
USE_OPENAI_CHAT=true
OPENAI_MODEL=gpt-4o-mini
```

3. Start the backend normally:

```bash
uvicorn backend.app:app --reload --app-dir .
```

When enabled, the backend will return `provider="openai"` in the `/chat/support` response. If the key is missing or the OpenAI call fails, it will fallback to the local rule-based chat with `provider="fallback"`.

## Tests

From the project root:

```bash
pip install -r requirements.txt
pytest tests/ -q
```

Some tests **skip** if `artifacts/model.joblib` is missing; place an artifact (via training) to run the full set.
