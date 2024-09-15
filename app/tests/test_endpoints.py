# tests/test_endpoints.py

import pytest
from unittest.mock import MagicMock
from app import create_app
from prometheus_client import CollectorRegistry

@pytest.fixture
def client(monkeypatch):
    # Create mock models
    mock_predictive_model = MagicMock()
    mock_predictive_model.predict.return_value = [1]  # Mock prediction

    mock_llm_model = MagicMock()
    mock_llm_model.generate_response.return_value = "This is a mock response."  # Mock LLM response

    mock_slack_client = MagicMock()
    mock_slack_client.send_message.return_value = True  # Mock Slack client

    # Create a separate CollectorRegistry for testing to prevent duplicated metrics
    test_registry = CollectorRegistry()

    # Create app with test registry
    app = create_app(registry=test_registry)
    app.config['TESTING'] = True

    # Mock the app.persistent_state with mock models
    app.persistent_state = {
        'predictive_model': mock_predictive_model,
        'llm_model': mock_llm_model,
        'slack_client': mock_slack_client
    }

    with app.test_client() as client:
        with app.app_context():
            yield client

def test_health_check(client):
    """Test the health check endpoint."""
    rv = client.get('/')
    assert rv.status_code == 200
    assert rv.data == b'OK'

def test_predict_valid(client):
    """Test the /predict endpoint with valid data."""
    data = {
        'proto': 'TCP',
        'service': 'http',
        'state': 'FIN',
        'sbytes': 123,
        'dbytes': 456,
        'sttl': 50,
        'dttl': 60,
        'sloss': 0,
        'dloss': 0,
        'sload': 1.23,
        'dload': 2.34,
        'spkts': 10,
        'dpkts': 12
    }
    rv = client.post('/predict', json=data)
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert "prediction" in json_data
    assert json_data["prediction"] == 1

def test_predict_missing_fields(client):
    """Test the /predict endpoint with missing fields."""
    data = {
        'proto': 'TCP',
        'sbytes': 123
        # Missing other required fields
    }
    rv = client.post('/predict', json=data)
    assert rv.status_code == 400  # Expecting bad request for missing fields
    json_data = rv.get_json()
    assert "error" in json_data
    assert "Missing required fields" in json_data["error"]
