curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_requests/predict.json | python -m json.tool
