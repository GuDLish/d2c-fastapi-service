# Part 4 — D2C Churn Scoring API (FastAPI)

FastAPI service that loads the churn model trained in Part 3 and exposes
prediction endpoints for an internal CRM tool.

- **Snapshot date:** `2025-09-30`
- **Target:** `churn_next_60d`
- **Model artifact:** `model.pkl` produced by Part 3 (`{pipeline, threshold, feature_cols, num_cols, cat_cols, snapshot_date, champion}`)

## Project layout
```
.
├── app/
│   ├── main.py          # FastAPI app + endpoints
│   ├── schemas.py       # Pydantic request/response models with validation
│   ├── model_loader.py  # Cached joblib loader
│   └── explain.py       # Rule-based risk-level + explanation
├── train_model.py       # Re-creates model.pkl from raw CSVs in ./data/
├── tests/test_api.py    # 5 API test cases (pytest + TestClient)
├── sample_requests/     # curl scripts + sample JSON
├── monitoring_plan.md
├── responsible_use.md
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the model

Either (a) drop the `model.pkl` produced by Part 3's notebook into the project
root, **or** (b) place the raw CSVs from the data package into `./data/` and
run:

```bash
python train_model.py
```

This regenerates `model.pkl` using the exact same leakage-safe pipeline as
Part 3 (calibrated GradientBoosting / RandomForest, cost-tuned threshold).

## Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Open interactive docs at http://127.0.0.1:8000/docs.

## Run tests

```bash
pytest -v
```

Prediction tests automatically skip if `model.pkl` is missing.

## Run with Docker

```bash
docker build -t churn-api .
docker run --rm -p 8000:8000 -v "$PWD/model.pkl:/app/model.pkl" churn-api
```

## Endpoints

### `GET /health`
Returns service status, whether the model is loaded, champion model name,
snapshot date and the tuned threshold.

**Sample response**
```json
{
  "status": "ok",
  "model_loaded": true,
  "champion": "GradientBoosting",
  "snapshot_date": "2025-09-30",
  "threshold": 0.42
}
```

### `POST /predict`
Accepts one `CustomerFeatures` payload (see `app/schemas.py` for the full
schema and value ranges).

**Sample request:** `sample_requests/predict.json`

```bash
bash sample_requests/predict.sh
```

**Sample response**
```json
{
  "customer_id": "CUST00012",
  "churn_probability": 0.71,
  "predicted_class": 1,
  "risk_level": "high",
  "risk_explanation": "Key risk drivers: very low 30-day session activity; 1 support tickets in 90 days.",
  "threshold_used": 0.42
}
```

### `POST /batch_predict`
Accepts up to 5000 `CustomerFeatures` rows.

```bash
bash sample_requests/batch_predict.sh
```

**Sample response**
```json
{
  "predictions": [
    {
      "customer_id": "CUST00012",
      "churn_probability": 0.71,
      "predicted_class": 1,
      "risk_level": "high",
      "risk_explanation": "...",
      "threshold_used": 0.42
    }
  ]
}
```

## Input validation
- All numeric fields enforce sensible ranges (e.g. `avg_discount ∈ [0,1]`,
  `avg_rating ∈ [0,5]`, `avg_sentiment_90d ∈ [-1,1]`).
Unknown categorical values are handled by the training-time encoder (unknown categories are ignored).
- Missing optional features fall back to the training-time defaults (`0` for
  numeric, `"Missing"` for categorical).

## Error handling
| Condition | HTTP code |
|---|---|
| Payload fails Pydantic validation | 422 |
| Empty `customers` list in batch | 400 |
| Batch larger than 5000 rows | 413 |
| Model artifact missing on disk | 503 |
| Any other prediction failure | 500 |

## Notes on the model
See `monitoring_plan.md` for drift / retraining triggers and
`responsible_use.md` for how the retention team should and should not act on
the score.
