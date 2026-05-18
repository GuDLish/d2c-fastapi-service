import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "total_orders": 5,
    "total_spent": 3200,
    "avg_order_value": 640,
    "avg_delivery_days": 4,
    "return_rate": 0.1,
    "avg_rating": 4.2
}

response = requests.post(
    url,
    json=data
)

print(response.json())