"""API tests. Run with:  pytest -v

These tests require model.pkl in the project root. If you don't have one,
run `python train_model.py` first (needs CSVs in ./data/)."""
import os, pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

HIGH_RISK = {
    "customer_id": "TEST_HIGH",
    "city_tier": "Tier 2", "age_group": "35-44",
    "acquisition_channel": "Instagram", "loyalty_tier": "Missing",
    "preferred_category": "Skin Care", "skin_type": "Dry",
    "marketing_consent": "No", "last_campaign_received": "none",
    "manual_priority_bucket": "low",
    "days_since_signup": 400, "recency_days": 250,
    "frequency_all": 1, "monetary_all": 400, "avg_order_value": 400,
    "avg_discount": 0.4, "return_rate": 0.5, "avg_rating": 2.0,
    "category_div": 1, "avg_delivery": 6,
    "frequency_180d": 0, "monetary_180d": 0,
    "sessions_30d": 0, "product_views_30d": 0, "cart_adds_30d": 0,
    "wishlist_adds_30d": 0, "abandoned_carts_30d": 0,
    "email_opens_30d": 0, "campaign_clicks_30d": 0,
    "last_visit_days_ago": 55,
    "ticket_count_90d": 3, "avg_sentiment_90d": -0.7,
    "neg_ticket_rate_90d": 0.8, "avg_resolution_hours_90d": 50,
    "reopened_count_90d": 2, "last_campaign_cost": 0, "had_ticket": 1,
}

LOW_RISK = {**HIGH_RISK,
    "customer_id": "TEST_LOW",
    "loyalty_tier": "Gold", "marketing_consent": "Yes",
    "recency_days": 5, "frequency_all": 12, "monetary_all": 9000,
    "frequency_180d": 6, "monetary_180d": 5500,
    "sessions_30d": 18, "product_views_30d": 90, "cart_adds_30d": 5,
    "wishlist_adds_30d": 3, "email_opens_30d": 8, "campaign_clicks_30d": 3,
    "last_visit_days_ago": 1, "return_rate": 0.0, "avg_rating": 4.7,
    "ticket_count_90d": 0, "avg_sentiment_90d": 0, "neg_ticket_rate_90d": 0,
    "avg_resolution_hours_90d": 0, "reopened_count_90d": 0, "had_ticket": 0,
}

MODEL_PRESENT = os.path.exists(os.environ.get("MODEL_PATH", "model.pkl"))
requires_model = pytest.mark.skipif(not MODEL_PRESENT, reason="model.pkl not present")


def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "model_loaded" in body


@requires_model
def test_predict_high_risk_customer():
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["predicted_class"] in (0, 1)
    assert body["risk_level"] in ("low", "medium", "high")
    assert body["customer_id"] == "TEST_HIGH"
    assert isinstance(body["risk_explanation"], str) and body["risk_explanation"]


@requires_model
def test_batch_predict_returns_one_per_customer():
    r = client.post("/batch_predict", json={"customers": [HIGH_RISK, LOW_RISK]})
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert {p["customer_id"] for p in preds} == {"TEST_HIGH", "TEST_LOW"}


def test_predict_rejects_invalid_payload():
    bad = {"avg_discount": 5}  # discount must be 0..1
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_batch_predict_rejects_empty_list():
    r = client.post("/batch_predict", json={"customers": []})
    assert r.status_code == 400
