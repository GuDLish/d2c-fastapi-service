from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

class CustomerData(BaseModel):
    total_orders: int
    total_spent: float
    avg_order_value: float
    avg_delivery_days: float
    return_rate: float
    avg_rating: float

@app.get("/")
def home():
    return {
        "message": "D2C Customer Churn Prediction API"
    }

@app.post("/predict")
def predict(data: CustomerData):

    features = np.array([[
        data.total_orders,
        data.total_spent,
        data.avg_order_value,
        data.avg_delivery_days,
        data.return_rate,
        data.avg_rating
    ]])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability)
    }