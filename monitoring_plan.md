# Monitoring Plan — Churn Scoring API

This plan covers what the retention/ML team should track after this API is
deployed to the internal CRM.

## 1. Service health (real-time)
| Metric | Tool | Alert threshold |
|---|---|---|
| API uptime / availability | uptime check on `GET /health` every 60s | < 99% over 15 min |
| p95 latency `/predict` | request logs / APM | > 300 ms for 10 min |
| p95 latency `/batch_predict` (1k rows) | request logs | > 5 s |
| 5xx error rate | request logs | > 1% over 5 min |
| 4xx validation rate | request logs | sudden spike vs 7-day baseline (Pydantic schema drift) |

## 2. Data drift (weekly)
Compare the live request distribution against the training snapshot
(`2025-09-30`). For each feature track:
- **Numeric:** mean, median, PSI (Population Stability Index). Alert at PSI > 0.2.
- **Categorical:** category mix and share of `"Missing"` / `handle_unknown` hits.
  Alert if any single category share moves by > 10 absolute points.

Watch especially: `recency_days`, `sessions_30d`, `last_visit_days_ago`,
`ticket_count_90d`, `acquisition_channel`, `loyalty_tier`. These drove the
top feature importances in Part 3.

## 3. Prediction distribution (daily)
- Mean predicted churn probability per day.
- Share of customers scored as `high`, `medium`, `low`.
- Alert if `high`-risk share moves > 25% vs the prior 7-day average — usually a
  drift signal, not a real churn surge.

## 4. Business outcomes (monthly)
Hold-out a small random control group (e.g. 5% of `high` risk) that receives
**no** retention intervention. Then track:
- Realised 60-day churn rate by predicted bucket (calibration check).
- Save rate = (control churn − treatment churn) / control churn.
- Net value per contacted customer vs the threshold assumptions
  (`SAVE_VALUE = ₹1,500`, `CONTACT_COST = ₹150`).

## 5. Retraining triggers
Retrain (or recalibrate) if **any** is true:
- PSI > 0.25 on three or more top-10 features for two consecutive weeks.
- Realised test PR-AUC drops > 15% vs the value recorded in `metrics.json`.
- Calibration error (Brier score) on the next true snapshot exceeds the
  training-snapshot Brier by > 0.05.
- Business inputs change (`SAVE_VALUE`, `CONTACT_COST`) — re-tune threshold
  via the cost curve in Part 3, you don't need a full retrain.
- More than 90 days have passed since the last snapshot (calendar trigger).

## 6. Logging & audit
- Log every prediction with: `customer_id`, timestamp, model version
  (`champion + snapshot_date`), threshold used, probability, predicted class.
- Keep 12 months. This lets you do post-hoc calibration plots and
  fairness audits.
