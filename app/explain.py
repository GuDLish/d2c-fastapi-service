"""Simple, rule-based risk explanation (transparent for CRM users)."""
def risk_level(prob: float, threshold: float) -> str:
    if prob >= max(threshold, 0.6): return "high"
    if prob >= threshold * 0.6:     return "medium"
    return "low"

def explain(features: dict, prob: float) -> str:
    reasons = []
    if features.get("recency_days", 0) >= 90:
        reasons.append(f"no purchase in {int(features['recency_days'])} days")
    if features.get("last_visit_days_ago", 0) >= 21:
        reasons.append(f"last app/web visit {int(features['last_visit_days_ago'])} days ago")
    if features.get("sessions_30d", 0) <= 1:
        reasons.append("very low 30-day session activity")
    if features.get("frequency_180d", 0) == 0:
        reasons.append("no orders in the last 180 days")
    if features.get("ticket_count_90d", 0) >= 2:
        reasons.append(f"{int(features['ticket_count_90d'])} support tickets in 90 days")
    if features.get("neg_ticket_rate_90d", 0) >= 0.5:
        reasons.append("majority of recent tickets had negative sentiment")
    if features.get("return_rate", 0) >= 0.3:
        reasons.append("high return rate")
    if features.get("avg_rating", 5) and features["avg_rating"] <= 2.5:
        reasons.append(f"low avg rating ({features['avg_rating']:.1f})")

    if not reasons:
        if prob >= 0.5:
            return "Combined behavioural signals indicate elevated churn risk."
        return "Engagement and purchase signals look healthy."
    return "Key risk drivers: " + "; ".join(reasons[:4]) + "."
