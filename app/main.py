from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np



with open("model.pkl", "rb") as file:

    model = pickle.load(file)


app = FastAPI()


class CustomerData(BaseModel):

    recency_days: int

    frequency_180d: int

    monetary_180d: float

    return_rate_180d: float

    avg_discount_pct_180d: float

    avg_rating_180d: float

    category_diversity_180d: int

    ticket_count_90d: int

    negative_ticket_rate_90d: float

    avg_resolution_hours_90d: float

    days_since_signup: int

    sessions_30d: int



@app.get("/")

def home():

    return {

        "message": "D2C Churn Prediction API is running successfully"

    }


@app.post("/predict")

def predict(data: CustomerData):

    features = [[

        data.recency_days,

        data.frequency_180d,

        data.monetary_180d,

        data.return_rate_180d,

        data.avg_discount_pct_180d,

        data.avg_rating_180d,

        data.category_diversity_180d,

        data.ticket_count_90d,

        data.negative_ticket_rate_90d,

        data.avg_resolution_hours_90d,

        data.days_since_signup,

        data.sessions_30d

    ]]



    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]



    return {

        "prediction": int(prediction),

        "churn_probability": float(probability)

    }