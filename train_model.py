"""Reproducible training script — rebuilds model.pkl identical in schema to Part 3.

Place the raw CSVs from the data package into ./data/ then run:
    python train_model.py
"""
import os, json, joblib, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
RNG = 42
DATA = "./data/"
SNAP = pd.Timestamp("2025-09-30")
CONTACT_COST, SAVE_VALUE = 150, 1500

print("Loading data...")
customers = pd.read_csv(DATA + "customers.csv", parse_dates=["signup_date"])
orders    = pd.read_csv(DATA + "orders.csv",    parse_dates=["order_date"])
support   = pd.read_csv(DATA + "support_tickets.csv", parse_dates=["ticket_date"])
web       = pd.read_csv(DATA + "web_events_snapshot.csv")
labels    = pd.read_csv(DATA + "churn_labels.csv")
interv    = pd.read_csv(DATA + "intervention_history.csv")

orders_pre  = orders[orders.order_date <= SNAP]
support_pre = support[support.ticket_date <= SNAP]

order_feats = (orders_pre.groupby("customer_id").agg(
    recency_days   = ("order_date", lambda x: (SNAP - x.max()).days),
    frequency_all  = ("order_id", "count"),
    monetary_all   = ("gross_amount", "sum"),
    avg_order_value= ("gross_amount", "mean"),
    avg_discount   = ("discount_pct", "mean"),
    return_rate    = ("returned", "mean"),
    avg_rating     = ("rating", "mean"),
    category_div   = ("category", "nunique"),
    avg_delivery   = ("delivery_days", "mean"),
).reset_index())

cutoff = SNAP - pd.Timedelta(days=180)
o180 = orders_pre[orders_pre.order_date >= cutoff]
f180 = o180.groupby("customer_id").agg(
    frequency_180d=("order_id","count"),
    monetary_180d =("gross_amount","sum")).reset_index()
order_feats = order_feats.merge(f180, on="customer_id", how="left")\
                         .fillna({"frequency_180d":0,"monetary_180d":0})

s90 = support_pre[support_pre.ticket_date >= SNAP - pd.Timedelta(days=90)]
sup_feats = s90.groupby("customer_id").agg(
    ticket_count_90d        = ("ticket_id","count"),
    avg_sentiment_90d       = ("sentiment_score","mean"),
    neg_ticket_rate_90d     = ("sentiment_score", lambda x: (x < -0.3).mean()),
    avg_resolution_hours_90d= ("resolution_hours","mean"),
    reopened_count_90d      = ("reopened","sum"),
).reset_index()

web_feats  = web.drop(columns=["snapshot_date"])
camp_feats = interv[["customer_id","last_campaign_received","last_campaign_cost","manual_priority_bucket"]]

base = customers.copy()
base["days_since_signup"] = (SNAP - base.signup_date).dt.days
base = base.drop(columns=["signup_date"])

df = (base.merge(order_feats, on="customer_id", how="left")
          .merge(web_feats,   on="customer_id", how="left")
          .merge(sup_feats,   on="customer_id", how="left")
          .merge(camp_feats,  on="customer_id", how="left")
          .merge(labels[["customer_id","churn_next_60d","split"]], on="customer_id", how="left"))

num_zero = ["recency_days","frequency_all","monetary_all","avg_order_value",
    "avg_discount","return_rate","avg_rating","category_div","avg_delivery",
    "frequency_180d","monetary_180d","ticket_count_90d","avg_sentiment_90d",
    "neg_ticket_rate_90d","avg_resolution_hours_90d","reopened_count_90d","last_campaign_cost"]
for c in num_zero: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
mask = df["frequency_all"] == 0
df.loc[mask,"recency_days"] = df.loc[mask,"days_since_signup"]

cat_cols = ["city_tier","age_group","acquisition_channel","loyalty_tier",
            "preferred_category","skin_type","marketing_consent",
            "last_campaign_received","manual_priority_bucket"]
for c in cat_cols: df[c] = df[c].fillna("Missing").astype(str)
df["had_ticket"] = (df.ticket_count_90d > 0).astype(int)

TARGET, DROP = "churn_next_60d", ["customer_id","split","churn_next_60d"]
feature_cols = [c for c in df.columns if c not in DROP]
num_cols     = [c for c in feature_cols if c not in cat_cols]

tr, va, te = df[df.split=="train"], df[df.split=="validation"], df[df.split=="test"]
Xtr,ytr = tr[feature_cols], tr[TARGET].astype(int)
Xva,yva = va[feature_cols], va[TARGET].astype(int)
Xte,yte = te[feature_cols], te[TARGET].astype(int)

pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])

models = {
    "LogReg":           LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RNG),
    "RandomForest":     RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                              n_jobs=-1, random_state=RNG, class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                                   learning_rate=0.05, random_state=RNG),
}

best_name, best_score = None, -1
for n, est in models.items():
    p = Pipeline([("pre", pre), ("clf", est)]).fit(Xtr, ytr)
    s = average_precision_score(yva, p.predict_proba(Xva)[:,1])
    print(f"  {n}: val PR-AUC = {s:.3f}")
    if s > best_score: best_name, best_score = n, s
print(f"Champion: {best_name}")

calibrated = Pipeline([("pre", pre),
    ("clf", CalibratedClassifierCV(estimator=models[best_name], method="isotonic", cv=5))]).fit(Xtr, ytr)
proba_va = calibrated.predict_proba(Xva)[:,1]

thrs = np.linspace(0.05, 0.95, 91)
def ev(p, t):
    yhat = (p>=t).astype(int); y = yva.values
    tp=((yhat==1)&(y==1)).sum(); fp=((yhat==1)&(y==0)).sum(); fn=((yhat==0)&(y==1)).sum()
    return tp*(SAVE_VALUE-CONTACT_COST) - fp*CONTACT_COST - fn*SAVE_VALUE
best_t = float(thrs[int(np.argmax([ev(proba_va,t) for t in thrs]))])
print(f"Chosen threshold (max EV): {best_t:.3f}")

proba_te = calibrated.predict_proba(Xte)[:,1]
print(f"Test ROC-AUC={roc_auc_score(yte,proba_te):.3f}  PR-AUC={average_precision_score(yte,proba_te):.3f}")

joblib.dump({
    "pipeline": calibrated, "threshold": best_t,
    "feature_cols": feature_cols, "num_cols": num_cols, "cat_cols": cat_cols,
    "snapshot_date": str(SNAP.date()), "champion": best_name,
}, "model.pkl")
print("Saved model.pkl")
