import requests

url = "http://127.0.0.1:8000/predict"

data = {

    "recency_days": 20,

    "frequency_180d": 12,

    "monetary_180d": 5000,

    "return_rate_180d": 0.1,

    "avg_discount_pct_180d": 15,

    "avg_rating_180d": 4.2,

    "category_diversity_180d": 5,

    "ticket_count_90d": 2,

    "negative_ticket_rate_90d": 0.2,

    "avg_resolution_hours_90d": 12,

    "days_since_signup": 400,

    "sessions_30d": 18

}

response = requests.post(

    url,

    json=data

)

print(response.json())
