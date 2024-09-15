#!/bin/bash

# Base URL
BASE_URL="http://localhost:5000"

echo "1. Testing Health Check Endpoint:"
curl -X GET "$BASE_URL/" | jq .
echo -e "\n----------------------------------------"

echo "2. Testing Prediction Endpoint with Valid Data:"
curl -X POST "$BASE_URL/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "proto": "HTTP",
           "service": "web",
           "state": "S0",
           "sbytes": 1024.0,
           "dbytes": 2048.0,
           "sttl": 64.0,
           "dttl": 128.0,
           "sloss": 0.0,
           "dloss": 0.0,
           "sload": 0.0,
           "dload": 0.0,
           "spkts": 10.0,
           "dpkts": 20.0
         }' | jq .
echo -e "\n----------------------------------------"

echo "3. Testing Prediction Endpoint with Missing Fields:"
curl -X POST "$BASE_URL/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "proto": "HTTP",
           "service": "web"
         }' | jq .
echo -e "\n----------------------------------------"

echo "4. Testing Prediction Endpoint with Incorrect Data Types:"
curl -X POST "$BASE_URL/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "proto": 123,
           "service": "web",
           "state": "S0",
           "sbytes": "one thousand",
           "dbytes": 2048.0,
           "sttl": 64.0,
           "dttl": 128.0,
           "sloss": 0.0,
           "dloss": 0.0,
           "sload": 0.0,
           "dload": 0.0,
           "spkts": 10.0,
           "dpkts": 20.0
         }' | jq .
echo -e "\n----------------------------------------"

echo "5. Testing Chat Endpoint:"
curl -X POST "$BASE_URL/chat" \
     -H "Content-Type: application/json" \
     -d '{
           "question": "Who just scanned the network?"
         }' | jq .
echo -e "\n----------------------------------------"

echo "6. Testing Slack Events Endpoint:"
curl -X POST "$BASE_URL/slack/events" \
     -H "Content-Type: application/json" \
     -d '{
           "token": "XXYYZZ",
           "team_id": "TXXXXXXXX",
           "api_app_id": "AXXXXXXXXX",
           "event": {
             "type": "message",
             "user": "UXXXXXXX",
             "text": "Hello bot!",
             "ts": "1612095967.000200",
             "channel": "CXXXXXXX",
             "event_ts": "1612095967.000200"
           },
           "type": "event_callback",
           "authed_users": ["UXXXXXXX"]
         }'
echo -e "\n----------------------------------------"

echo "7. Testing Model Status Endpoint (If Available):"
curl -X GET "$BASE_URL/model/status" | jq .
echo -e "\n----------------------------------------"
