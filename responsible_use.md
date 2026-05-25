# Responsible Use — Churn Scoring API

## What the API is
A statistical score (0–1) estimating the likelihood that a customer churns
in the next 60 days, plus a transparent rule-based explanation. It is a
**decision support tool**, not a verdict.

## How the retention team SHOULD use it
- Prioritise `high`-risk customers with high historical monetary value first
  (budget is finite — see Part 2 prioritisation logic).
- Combine the score with the segment from Part 2 before choosing an action:
  e.g. a `high`-risk `Champion` deserves a personal call, a `high`-risk
  `Recent Customer` deserves an onboarding nudge, not a deep discount.
- Use the `risk_explanation` to brief CRM agents on *why* a customer is at
  risk so the outreach feels relevant.
- A/B test interventions and feed results back into the monitoring plan.

## How the team SHOULD NOT use it
- **Do not** treat the score as ground truth. Recall on the test set is
  ~0.70 at the chosen threshold — roughly 3 in 10 actual churners will be
  missed, and some predicted churners will not churn.
- **Do not** stack heavy discounts on every `high`-risk customer. That is
  exactly the wasteful behaviour the project was built to avoid; many
  predicted churners would have stayed anyway.
- **Do not** use the score to deny service, raise prices, suppress
  recommendations, or otherwise punish customers. The model was trained
  for retention outreach only.
- **Do not** share the raw probability with the customer. It is an internal
  prioritisation signal.
- **Do not** use it for any segment the model was not trained on
  (e.g. B2B customers, markets outside the snapshot, post-2025-09-30 cohorts)
  without a fresh evaluation.

## Fairness & sensitive attributes
The model uses `age_group` and `city_tier` (broad geo proxy). These were
included because they materially improve recall. Before each retraining,
audit the realised contact rate and save rate across these slices to ensure
no group is being systematically over- or under-served.

## Escalation
If a customer disputes a retention contact, the team can look up the
logged prediction (see Monitoring Plan §6) including the threshold and the
explanation that drove the action.
