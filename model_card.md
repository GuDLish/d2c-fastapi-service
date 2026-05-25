# Model Card — Customer Churn Prediction Model

## Model Overview

This machine learning model predicts whether a customer is likely to churn within the next 60 days.

The model was trained using transactional customer behavior data from a D2C business environment.

---

## Model Type

**Calibrated ensemble** (as implemented in the training pipeline): the model artifact is saved as a single `pipeline` that may include tree models (e.g., Random Forest / Gradient Boosting) with calibration.


---

## Input Features

The model uses the following features:

- total_orders
- total_spent
- avg_order_value
- avg_delivery_days
- return_rate
- avg_rating

---

## Target Variable

`churn_next_60d`

- 1 = customer likely to churn
- 0 = customer likely to stay

---

## Performance Metrics

| Metric | Score |
|---|---|
| Accuracy | 65.2% |
| Precision | 65% |
| Recall | 52.9% |
| F1 Score | 58.3% |

---

## Intended Use

The model is intended for:

- churn risk identification,
- customer retention campaigns,
- customer segmentation support,
- marketing prioritization.

---

## Limitations

The model currently relies mainly on transaction-based customer features.

It does not yet include:

- web activity,
- support interaction intensity,
- campaign engagement,
- behavioral event streams.

---

## Author

Prateek Parmar