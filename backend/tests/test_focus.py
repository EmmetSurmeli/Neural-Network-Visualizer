import numpy as np
import pytest
import torch
from torch import nn
from fastapi.testclient import TestClient
from backend.main import app
from backend.instrumentation.trace import run_trace
from backend.instrumentation.visualization import focused_visualization, activation_summary


def test_focus_ranks_all_inputs_and_preserves_real_values():
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(80, 64), nn.Tanh(), nn.Linear(64, 2))
    x = torch.randn(1, 80)
    trace, captured = run_trace(model, x, 'Focus test')
    # A low-activation input with a large weight can matter most: it must not
    # disappear merely because the overview chooses different source neurons.
    for layer, target in [(0, 61), (2, 1)]:
        focus = focused_visualization(captured, layer, target)
        values = captured[layer]['input'].numpy().reshape(-1)
        weights = model[layer].weight.detach().numpy()[target]
        products = values * weights
        expected = sorted(np.argsort(-np.abs(products), kind='stable')[:16].tolist())
        source, destination = focus['diagram']['columns']
        assert source['indices'] == expected
        assert destination['indices'] == [target]
        assert source['states'][-1]['values'] == pytest.approx(values[expected])
        for edge in focus['diagram']['connections'][0]['edges']:
            assert edge['target'] == target
            assert edge['contribution'] == pytest.approx(products[edge['source']])
        assert focus['magnitude_coverage'] == pytest.approx(np.abs(products[expected]).sum() / np.abs(products).sum())
        assert focus['pre_activation'] == pytest.approx(products.sum() + focus['bias'], abs=1e-6)
    assert len(trace['visualization']['columns']) == 3


def test_focus_recovers_important_edge_missing_from_overview():
    model = nn.Sequential(nn.Linear(80, 2))
    x = torch.ones(1, 80)
    x[0, 79] = .01
    with torch.no_grad():
        model[0].weight.fill_(1)
        model[0].weight[0, 79] = 10000
    trace, captured = run_trace(model, x, 'Omitted edge')
    assert 79 not in trace['visualization']['columns'][0]['indices']
    focus = focused_visualization(captured, 0, 0)
    edge = next(e for e in focus['diagram']['connections'][0]['edges'] if e['source'] == 79)
    assert edge['contribution'] == pytest.approx(100)


def test_focus_cnn_flatten_indices_and_zero_contributions():
    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Flatten(), nn.Linear(18, 3))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    _, captured = run_trace(model, torch.ones(1, 1, 5, 5), 'CNN')
    focus = focused_visualization(captured, 3, 2)
    assert focus['diagram']['columns'][0]['states'][0]['step'] == 2
    assert focus['diagram']['columns'][0]['label'] == 'Flatten'
    assert focus['magnitude_coverage'] == 0
    assert all(e['contribution'] == 0 for e in focus['diagram']['connections'][0]['edges'])


def test_activity_summary_covers_full_tensor_in_index_order():
    values = torch.tensor([0., -2., 4., 0.] * 30)
    summary = activation_summary(values)
    assert summary['zero_fraction'] == .5
    expected = [np.abs(v).mean() / 4 for v in np.array_split(values.numpy(), 48)]
    assert summary['activity'] == pytest.approx(expected)


def test_focus_api_validation_and_expired_trace():
    client = TestClient(app)
    trace = client.post('/run-model', json={'model_id': 'tiny-mlp', 'input': [1, .5, -1]}).json()
    url = f"/traces/{trace['trace_id']}/focus"
    focus = client.get(url, params={'layer': 0, 'neuron': 0})
    assert focus.status_code == 200
    assert focus.json()['pre_activation'] == pytest.approx(2.35)
    assert focus.json()['shown'] == 3
    for layer, neuron in [(-1, 0), (100, 0), (1, 0), (0, -1), (0, 4)]:
        assert client.get(url, params={'layer': layer, 'neuron': neuron}).status_code == 422
    assert client.get('/traces/missing/focus?layer=0&neuron=0').status_code == 404
