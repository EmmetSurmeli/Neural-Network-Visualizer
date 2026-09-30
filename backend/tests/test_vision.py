import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from backend.main import app, models
from backend.models.catalog import IDS
from backend.models.vision import load, samples, generate
from backend.instrumentation.trace import run_trace

client = TestClient(app)

@pytest.mark.parametrize('kind', ['shapes-cnn', 'strokes-cnn'])
def test_trained_cnn_evaluation_and_trace_are_real(kind):
    model, info = load(kind)
    x, y = generate(kind, 600, 4042)
    with torch.inference_mode():
        accuracy = (model(x).argmax(-1) == y).float().mean().item()
    assert accuracy >= .95
    assert accuracy == info['test_accuracy']
    for sample in samples(kind):
        tensor = torch.tensor(sample['pixels']).reshape(1, 1, 28, 28)
        trace, raw = run_trace(model, tensor, info['name'])
        assert trace['predicted_class'] == sample['label']
        assert trace['input_shape'] == [1, 1, 28, 28]
        assert trace['layers'][0]['type'] == 'Conv2d'
        assert trace['layers'][2]['output_shape'] == [1, 8, 14, 14]
        assert trace['layers'][5]['output_shape'] == [1, 16, 7, 7]
        assert trace['layers'][6]['output_shape'] == [1, 784]
        for layer, capture in zip(trace['layers'], raw):
            maps = layer['feature_maps']
            if capture['output'].ndim == 4:
                assert maps['values'] == capture['output'][0].flatten(1).tolist()
                assert maps['scale'] == max(capture['output'].abs().max().item(), 1e-12)
            else:
                assert maps is None
        graph = trace['visualization']
        assert graph['input_label'] == 'Flatten'
        assert [c['size'] for c in graph['columns']] == [784, 32, 3]
        assert len(graph['columns'][1]['indices']) == 32
        for col in graph['columns']:
            for state in col['states']:
                assert state['values'] == raw[state['step']]['output'].flatten()[col['indices']].tolist()
        for connection in graph['connections']:
            assert raw[connection['step']]['module'].__class__.__name__ == 'Linear'
            for e in connection['edges']:
                assert e['contribution'] == pytest.approx(e['input'] * e['weight'], abs=1e-6)


def test_catalog_and_model_specific_api_inputs():
    models.clear()
    catalog = client.get('/demo-models').json()
    assert [m['id'] for m in catalog] == IDS
    for info in catalog:
        if info.get('input_kind') == 'image':
            options = client.get('/samples', params={'model_id': info['id']}).json()
            sample = options[0]
            values = sample['pixels']
        else:
            values = info['default_input']
        response = client.post('/run-model', json={'model_id': info['id'], 'input': values})
        assert response.status_code == 200
        data = response.json()
        assert np.isfinite(data['output']).all()
        if info.get('input_kind') == 'image':
            assert data['predicted_class'] == sample['label']
            assert data['classes'] == info['classes']
            assert data['input_shape'] == [1, *info['input_shape']]
    assert client.get('/samples', params={'model_id': 'missing'}).status_code == 404
    assert client.post('/run-model', json={'model_id': 'shapes-cnn', 'input': [0, 1]}).status_code == 422
    assert client.post('/run-model', json={'model_id': 'shapes-cnn', 'input': [-1] * 784}).status_code == 422


def test_builtin_models_do_not_consume_import_slots():
    models.clear()
    client.get('/demo-models')
    spec = {'name': 'Small', 'input_features': 1, 'layers': [{'type': 'Linear', 'weight': [[1.]]}]}
    for _ in range(8):
        assert client.post('/upload-model', json=spec).status_code == 201
    assert client.post('/upload-model', json=spec).status_code == 409
    models.clear()
