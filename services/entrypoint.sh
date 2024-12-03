#!/bin/bash
set -e

if [ "$#" -eq 0 ]; then
    echo "Run one of following"
    echo "python app/create_mock_data.py"
    echo "python app/process_mock_data.py"
    echo "python app/predict_and_store.py"
else
    exec "$@"
fi
