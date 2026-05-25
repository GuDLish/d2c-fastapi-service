curl -s -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d "{\"customers\": [$(cat sample_requests/predict.json)]}" | python -m json.tool
