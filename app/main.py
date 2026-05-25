import logging
from typing import List
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.schemas import (
    CustomerFeatures, PredictionResponse,
    BatchRequest, BatchResponse, HealthResponse,
)
from app.model_loader import get_artifact
from app.explain import risk_level, explain

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("churn-api")

app = FastAPI(
    title="D2C Churn Scoring API",
    description="Returns 60-day churn risk for D2C customers. "
                "Part 4 of the Customer Churn Intelligence capstone.",
    version="1.0.0",
)

# --------- error handlers ---------
@app.exception_handler(FileNotFoundError)
async def _model_missing(_: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})

@app.exception_handler(ValueError)
async def _bad_value(_: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# --------- helpers ---------
def _score(rows: List[CustomerFeatures]) -> List[PredictionResponse]:
    art = get_artifact()
    pipeline   = art["pipeline"]
    threshold  = float(art["threshold"])
    feat_cols  = art["feature_cols"]

    df = pd.DataFrame([r.model_dump() for r in rows])
    # ensure every expected feature exists, fill missing with sensible defaults
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0 if c in art["num_cols"] else "Missing"
    X = df[feat_cols]

    probs = pipeline.predict_proba(X)[:, 1]
    out = []
    for raw, p in zip(rows, probs):
        feats = raw.model_dump()
        out.append(PredictionResponse(
            customer_id      = raw.customer_id,
            churn_probability= round(float(p), 4),
            predicted_class  = int(p >= threshold),
            risk_level       = risk_level(float(p), threshold),
            risk_explanation = explain(feats, float(p)),
            threshold_used   = threshold,
        ))
    return out


# --------- endpoints ---------
@app.get("/health", response_model=HealthResponse)
def health():
    try:
        art = get_artifact()
        return HealthResponse(status="ok", model_loaded=True,
                              champion=art.get("champion"),
                              snapshot_date=art.get("snapshot_date"),
                              threshold=float(art["threshold"]))
    except Exception as e:
        log.warning("health: model not loaded: %s", e)
        return HealthResponse(status="degraded", model_loaded=False)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures):
    try:
        return _score([payload])[0]
    except FileNotFoundError:
        raise
    except Exception as e:
        log.exception("predict failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/batch_predict", response_model=BatchResponse)
def batch_predict(payload: BatchRequest):
    if len(payload.customers) == 0:
        raise HTTPException(status_code=400, detail="customers list is empty")
    if len(payload.customers) > 5000:
        raise HTTPException(status_code=413, detail="Batch too large (max 5000)")
    try:
        return BatchResponse(predictions=_score(payload.customers))
    except FileNotFoundError:
        raise
    except Exception as e:
        log.exception("batch_predict failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")
