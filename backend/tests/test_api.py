import json
import math
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from backend.main import app, traces, models
from backend.models.demo import samples

@pytest.fixture
def client():
    models.clear()
    traces.clear()
    with TestClient(app) as c:
        yield c

def test_health(client):
    assert client.get('/health').json()['status'] == 'ok'

def test_forward_endpoint_and_neuron(client):
    result = client.post('/run-demo', json={'input': samples()[7]['pixels']})
    assert result.status_code == 200
    trace = result.json()
    assert trace['predicted_class'] == 7
    assert len(trace['layers']) == 6
    assert sum(trace['probabilities']) == pytest.approx(1)
    details = client.get(f"/traces/{trace['trace_id']}/layers/0/neurons/42?limit=10")
    assert details.status_code == 200
    data = details.json()
    assert len(data['top_contributions']) == 10
    assert data['pre_activation'] == pytest.approx(data['contribution_sum'] + data['bias'], abs=1e-5)
    assert client.get(f"/traces/{trace['trace_id']}/layers/0/neurons/9999").status_code == 422
    assert client.get(f"/traces/{trace['trace_id']}/layers/99/neurons/0").status_code == 404

def test_input_validation(client):
    assert client.post('/run-demo', json={'input': [0, 1]}).status_code == 422
    assert client.post('/run-demo', json={'input': [-1] * 784}).status_code == 422
    assert client.post('/run-demo', json={'input': ['NaN'] * 784}).status_code == 422
    assert client.post('/run-model', json={'input': [1], 'model_id': 'missing'}).status_code == 404

def test_controlled_upload_and_tiny_prediction(client):
    spec = json.loads((Path(__file__).resolve().parents[2] / 'examples/tiny-model.json').read_text())
    response = client.post('/upload-model', json=spec)
    assert response.status_code == 201
    model_id = response.json()['id']
    trace = client.post('/run-model', json={'model_id': model_id, 'input': [1, .5, -1]}).json()
    assert trace['layers'][0]['activations'] == pytest.approx([2.35, -.5, -1.45, -.5])
    assert trace['predicted_class'] == 0
    assert trace['probabilities'][0] == pytest.approx(1 / (1 + math.exp(-3.425)), abs=1e-6)

def test_invalid_uploads(client):
    assert client.post('/upload-model', json={'name': 'bad', 'input_features': 3, 'layers': [{'type': 'Python', 'source': 'anything'}]}).status_code == 422
    assert client.post('/upload-model', json={'name': 'bad', 'input_features': 3, 'layers': [{'type': 'Linear', 'weight': [[1, 2]], 'bias': [0]}]}).status_code == 422
    assert client.post('/upload-model', content=b'not a pickle loader', headers={'Content-Type': 'application/octet-stream'}).status_code == 422

def test_expired_trace_and_payload_limit(client):
    assert client.get('/traces/missing/layers/0/neurons/0').status_code == 404
    response = client.post('/upload-model', content=b' ' * (8 * 1024 * 1024 + 1))
    assert response.status_code == 413
