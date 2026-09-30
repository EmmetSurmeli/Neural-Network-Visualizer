import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from backend.main import app, models
from backend.services.playground import PlaygroundConfig, dataset_points, train_model

client = TestClient(app)

@pytest.mark.parametrize('dataset', ['xor', 'moons', 'circles', 'spiral'])
def test_datasets_are_balanced_reproducible_and_standardized(dataset):
    x, y, train, bound = dataset_points(dataset)
    again = dataset_points(dataset)
    assert all(np.array_equal(a, b) for a, b in zip((x, y, train), again[:3]))
    assert x.shape == (200, 2) and np.isfinite(x).all()
    assert np.bincount(y).tolist() == [100, 100]
    assert np.bincount(y[train]).tolist() == [80, 80]
    assert np.bincount(y[~train]).tolist() == [20, 20]
    assert np.allclose(x[train].mean(axis=0), 0, atol=1e-6)
    assert np.allclose(x[train].std(axis=0), 1, atol=1e-6)
    assert bound > np.abs(x).max()

@pytest.mark.parametrize('overrides', [
    {'activation': 'ReLU', 'hidden_layers': [2], 'learning_rate': .001, 'epochs': 100},
    {'activation': 'Sigmoid', 'hidden_layers': [4, 16, 2], 'learning_rate': .05, 'epochs': 1000},
    {'activation': 'Tanh', 'hidden_layers': [8, 8], 'learning_rate': .01, 'epochs': 300},
])
def test_allowed_training_configurations_and_snapshot_parity(overrides):
    config = PlaygroundConfig(**overrides)
    model, result = train_model(config)
    snapshots = result['snapshots']
    assert [s['epoch'] for s in snapshots] == list(range(0, config.epochs + 1, 5))
    for s in snapshots:
        assert np.isfinite(s['loss'])
        assert 0 <= s['train_accuracy'] <= 1 and 0 <= s['validation_accuracy'] <= 1
        assert len(s['grid_probabilities']) == 48 * 48
        assert len(s['point_probabilities']) == 200
        assert all(0 <= p <= 1 for p in s['grid_probabilities'])
    with torch.inference_mode():
        x = torch.tensor([p['input'] for p in result['points']])
        assert model(x)[:, 1].tolist() == pytest.approx(snapshots[-1]['point_probabilities'])
        bound = result['grid']['max']
        corner = model(torch.tensor([[-bound, -bound]]))[0, 1].item()
        assert corner == pytest.approx(snapshots[-1]['grid_probabilities'][0], abs=1e-7)
    assert not model.training


def test_default_training_reproduces_and_learns():
    _, first = train_model(PlaygroundConfig())
    _, second = train_model(PlaygroundConfig())
    assert first['snapshots'] == second['snapshots']
    assert first['points'] == second['points']
    assert first['metrics']['validation_accuracy'] >= .9
    assert first['snapshots'][-1]['loss'] < first['snapshots'][0]['loss']

@pytest.mark.parametrize('dataset', ['xor', 'moons', 'circles', 'spiral'])
def test_api_training_to_visualizer(dataset):
    preview = client.get('/playground/dataset', params={'dataset': dataset}).json()
    response = client.post('/playground/train', json={'dataset': dataset, 'epochs': 100})
    assert response.status_code == 200
    run = response.json()
    assert preview['points'] == run['points']
    assert run['model_id'] in models
    trace = client.post('/run-model', json={'model_id': run['model_id'], 'input': run['points'][0]['input']})
    assert trace.status_code == 200
    data = trace.json()
    assert data['input_shape'] == [1, 2]
    assert [l['type'] for l in data['layers']] == ['Linear', 'Tanh', 'Linear', 'Tanh', 'Linear', 'Softmax']
    assert data['probabilities'][1] == pytest.approx(run['snapshots'][-1]['point_probabilities'][0], abs=1e-6)
    assert len(data['visualization']['columns']) == 4
    details = client.get(f"/traces/{data['trace_id']}/layers/0/neurons/0")
    assert details.status_code == 200

@pytest.mark.parametrize('config', [
    {'dataset': 'digits'}, {'hidden_layers': []}, {'hidden_layers': [8] * 4},
    {'hidden_layers': [3]}, {'activation': 'Softmax'}, {'learning_rate': .5},
    {'epochs': 999999}, {'seed': -1}, {'seed': '42'}, {'extra': True},
])
def test_training_rejects_invalid_requests(config):
    assert client.post('/playground/train', json=config).status_code == 422


def test_retraining_reuses_one_slot_and_preserves_import_capacity():
    original = models.copy()
    try:
        first = client.post('/playground/train', json={'epochs': 100}).json()
        keys = set(models)
        second = client.post('/playground/train', json={'dataset': 'xor', 'epochs': 100}).json()
        assert first['model_id'] != second['model_id']
        assert first['model_id'] not in models
        assert len(models) == len(keys)
        assert client.post('/run-model', json={'model_id': first['model_id'], 'input': [0, 0]}).status_code == 404
        models.clear()
        client.get('/demo-models')
        models['playground-latest'] = original.get('playground-latest', (None, {}))
        spec = {'name': 'Tiny', 'input_features': 2, 'layers': [{'type': 'Linear', 'weight': [[1., 1.]]}]}
        for _ in range(8):
            assert client.post('/upload-model', json=spec).status_code == 201
        assert client.post('/upload-model', json=spec).status_code == 409
    finally:
        models.clear()
        models.update(original)


def test_failed_training_preserves_previous_model(monkeypatch):
    import backend.main as main
    response = client.post('/playground/train', json={'epochs': 100})
    model_id = response.json()['model_id']
    def fail(_):
        raise ValueError('Training became unstable. Try a smaller learning rate.')
    monkeypatch.setattr(main, 'train_model', fail)
    response = client.post('/playground/train', json={})
    assert response.status_code == 422
    assert 'smaller learning rate' in response.json()['detail']
    assert model_id in models
    assert client.post('/run-model', json={'model_id': model_id, 'input': [0, 0]}).status_code == 200


@pytest.mark.parametrize('dataset', ['xor', 'moons', 'circles', 'spiral'])
def test_noise_changes_points_but_preserves_labels_and_split(dataset):
    clean, labels, split, _ = dataset_points(dataset, noise=0)
    noisy, noisy_labels, noisy_split, _ = dataset_points(dataset, noise=.5)
    repeat, _, _, _ = dataset_points(dataset, noise=.5)
    assert not np.allclose(clean, noisy)
    assert np.array_equal(labels, noisy_labels)
    assert np.array_equal(split, noisy_split)
    assert np.array_equal(noisy, repeat)
    preview = client.get('/playground/dataset', params={'dataset': dataset, 'noise': .5}).json()
    result = client.post('/playground/train', json={'dataset': dataset, 'noise': .5, 'epochs': 100}).json()
    assert preview['points'] == result['points']
    assert result['config']['noise'] == .5
    assert all(np.isfinite(s['validation_loss']) for s in result['snapshots'])

@pytest.mark.parametrize('noise', [-.1, .81, 'NaN'])
def test_invalid_noise_is_rejected(noise):
    assert client.get('/playground/dataset', params={'noise': noise}).status_code == 422
    assert client.post('/playground/train', json={'noise': noise}).status_code == 422
