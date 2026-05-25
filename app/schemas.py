from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal

class CustomerFeatures(BaseModel):
    """Input payload — must mirror the feature_cols used at training time.
    Unknown categorical values are handled by OneHotEncoder(handle_unknown='ignore').
    Missing numeric fields default to 0 (same convention as training)."""
    model_config = ConfigDict(extra="allow")

    customer_id: Optional[str] = Field(None, description="Optional identifier echoed in response")

    # categorical
    city_tier: str = "Missing"
    age_group: str = "Missing"
    acquisition_channel: str = "Missing"
    loyalty_tier: str = "Missing"
    preferred_category: str = "Missing"
    skin_type: str = "Missing"
    marketing_consent: str = "Missing"
    last_campaign_received: str = "Missing"
    manual_priority_bucket: str = "Missing"

    # numeric
    days_since_signup: float = 0
    recency_days: float = 0
    frequency_all: float = 0
    monetary_all: float = 0
    avg_order_value: float = 0
    avg_discount: float = Field(0, ge=0, le=1)
    return_rate: float = Field(0, ge=0, le=1)
    avg_rating: float = Field(0, ge=0, le=5)
    category_div: float = Field(0, ge=0)
    avg_delivery: float = Field(0, ge=0)
    frequency_180d: float = Field(0, ge=0)
    monetary_180d: float = Field(0, ge=0)
    sessions_30d: float = Field(0, ge=0)
    product_views_30d: float = Field(0, ge=0)
    cart_adds_30d: float = Field(0, ge=0)
    wishlist_adds_30d: float = Field(0, ge=0)
    abandoned_carts_30d: float = Field(0, ge=0)
    email_opens_30d: float = Field(0, ge=0)
    campaign_clicks_30d: float = Field(0, ge=0)
    last_visit_days_ago: float = Field(0, ge=0)
    ticket_count_90d: float = Field(0, ge=0)
    avg_sentiment_90d: float = Field(0, ge=-1, le=1)
    neg_ticket_rate_90d: float = Field(0, ge=0, le=1)
    avg_resolution_hours_90d: float = Field(0, ge=0)
    reopened_count_90d: float = Field(0, ge=0)
    last_campaign_cost: float = Field(0, ge=0)
    had_ticket: int = Field(0, ge=0, le=1)


class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float
    predicted_class: int
    risk_level: Literal["low", "medium", "high"]
    risk_explanation: str
    threshold_used: float


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]

class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    champion: Optional[str] = None
    snapshot_date: Optional[str] = None
    threshold: Optional[float] = None
