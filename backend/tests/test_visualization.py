import pytest
import torch
from torch import nn
from backend.instrumentation.trace import run_trace
from backend.models.demo import load_demo, samples

def test_diagram_matches_the_captured_execution_and_weights():
    model = load_demo()
    trace, captured = run_trace(model, torch.tensor([samples()[7]['pixels']]), 'MNIST')
    graph = trace['visualization']
    assert [c['size'] for c in graph['columns']] == [784, 128, 64, 10]
    assert [[s['step'] for s in c['states']] for c in graph['columns']] == [[-1], [0, 1], [2, 3], [4, 5]]
    assert [c['step'] for c in graph['connections']] == [0, 2, 4]
    assert all(len(c['indices']) <= 32 for c in graph['columns'])
    assert sum(len(c['edges']) for c in graph['connections']) <= 600
    for column in graph['columns']:
        for state in column['states']:
            raw = captured[0]['input'] if state['step'] < 0 else captured[state['step']]['output']
            assert state['values'] == raw.flatten()[column['indices']].tolist()
    for boundary in graph['connections']:
        raw = captured[boundary['step']]
        for edge in boundary['edges']:
            assert edge['source'] in graph['columns'][boundary['source_column']]['indices']
            assert edge['target'] in graph['columns'][boundary['target_column']]['indices']
            assert edge['weight'] == raw['module'].weight[edge['target'], edge['source']].item()
            assert edge['contribution'] == pytest.approx(edge['input'] * edge['weight'], abs=1e-6)

def test_nonlinear_steps_keep_neuron_identity_and_zero_negative_values():
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1., -2.], [3., 1.]]))
        linear.bias.zero_()
    trace, _ = run_trace(nn.Sequential(linear, nn.ReLU(), nn.Softmax(dim=-1)), torch.tensor([[1., 1.]]), 'small')
    graph = trace['visualization']
    assert len(graph['columns']) == 2
    assert graph['columns'][1]['indices'] == [0, 1]
    assert graph['columns'][1]['states'][0]['values'] == [-1, 4]
    assert graph['columns'][1]['states'][1]['values'] == [0, 4]
    assert graph['columns'][1]['states'][2]['values'] == trace['probabilities']
    assert len(graph['connections'][0]['edges']) == 4

def test_many_layers_respect_global_edge_budget_and_pure_nonlinear_model():
    model = nn.Sequential(*[nn.Linear(32, 32) for _ in range(24)])
    trace, _ = run_trace(model, torch.ones(1, 32), 'deep')
    graph = trace['visualization']
    assert len(graph['columns']) == 25
    assert sum(len(c['edges']) for c in graph['connections']) <= 600
    assert len(graph['columns'][0]['indices']) == 16
    assert all(len(c['indices']) == 32 for c in graph['columns'][1:])
    trace, _ = run_trace(nn.Sequential(nn.ReLU(), nn.Sigmoid()), torch.tensor([[-1., 1.]]), 'activation')
    assert len(trace['visualization']['columns']) == 1
    assert trace['visualization']['connections'] == []
