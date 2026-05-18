# d2c-fastapi-service

# D2C Customer Churn Capstone — Part 4: FastAPI Prediction Service

## Project Overview

This project exposes the churn prediction machine learning model through a FastAPI backend service.

The API accepts customer behavioral features and returns:

- churn prediction
- churn probability

This simulates a production-style ML deployment workflow.

---

## Repository Structure

```bash
Part4_FastAPI_Service/

│── app/
│   ├── main.py
│   └── __init__.py

│── model.pkl
│── README.md
│── requirements.txt
│── monitoring_plan.md
│── test_api.py
```

---

## Features

- FastAPI backend
- REST API endpoints
- ML model integration
- JSON request/response handling
- Swagger API documentation
- Real-time prediction service

---

## API Endpoints

### Home Endpoint

```bash
GET /
```

Returns API status message.

---

### Prediction Endpoint

```bash
POST /predict
```

Accepts customer features and returns churn prediction.

---

## Example Request

```json
{
    "total_orders": 5,
    "total_spent": 3200,
    "avg_order_value": 640,
    "avg_delivery_days": 4,
    "return_rate": 0.1,
    "avg_rating": 4.2
}
```

---

## Example Response

```json
{
    "prediction": 1,
    "churn_probability": 0.58
}
```

---

## Technologies Used

- Python
- FastAPI
- Scikit-learn
- NumPy
- Uvicorn
- Pydantic

---

## Installation & Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Run API

```bash
uvicorn app.main:app --reload
```

---

## Swagger Documentation

Open:

```bash
http://127.0.0.1:8000/docs
```

to access interactive API testing interface.

---

## Author

Prateek Parmar