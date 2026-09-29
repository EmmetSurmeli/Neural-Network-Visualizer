import json
from pathlib import Path

import pytest
import torch
from torch import nn

from backend.instrumentation.trace import run_trace
from backend.schemas.requests import ModelSpec
from backend.services.export import export_sequential
from backend.services.models import from_spec, summarize_spec

ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.parametrize(
    ('filename', 'sample', 'layer_types', 'probabilities'),
    [
        ('xor-model.json', [0, 1], ['Linear', 'Sigmoid', 'Linear', 'Sigmoid'], False),
        ('tanh-classifier.json', [1, -0.5, 0.25, 0.75], ['Linear', 'Tanh', 'Dropout', 'Linear', 'Softmax'], True),
        ('tiny-model.json', [1, 0.5, -1], ['Linear', 'ReLU', 'Linear', 'Softmax'], True),
    ],
)
def test_example_models_import_run_trace_and_visualize(filename, sample, layer_types, probabilities):
    spec = ModelSpec.model_validate_json((ROOT / 'examples' / filename).read_text())
    model = from_spec(spec)
    trace, captured = run_trace(model, torch.tensor([sample], dtype=torch.float32), spec.name)
    assert [layer['type'] for layer in trace['layers']] == layer_types
    assert len(captured) == len(layer_types)
    assert bool(trace['probabilities']) is probabilities
    assert torch.isfinite(torch.tensor(trace['output'])).all()
    assert trace['visualization']['columns']
    assert len(trace['visualization']['connections']) == sum(kind == 'Linear' for kind in layer_types)
    if probabilities:
        assert sum(trace['probabilities']) == pytest.approx(1)

def test_xor_example_matches_truth_table():
    spec = ModelSpec.model_validate_json((ROOT / 'examples/xor-model.json').read_text())
    model = from_spec(spec)
    with torch.inference_mode():
        values = model(torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])).flatten()
    assert values.tolist() == pytest.approx([0.007153, 0.992356, 0.992356, 0.007153], abs=0.00001)

def test_exported_pytorch_model_round_trips_exactly(tmp_path):
    original = nn.Sequential(
        nn.Flatten(), nn.Linear(6, 5), nn.Tanh(), nn.Dropout(0.25),
        nn.Linear(5, 3, bias=False), nn.Softmax(dim=-1),
    ).eval()
    path = tmp_path / 'exported.json'
    export_sequential(original, input_features=6, path=str(path), name='Round trip')
    spec = ModelSpec.model_validate_json(path.read_text())
    imported = from_spec(spec)
    sample = torch.tensor([[1., -2., 0.5, 0., 3., -1.]])
    with torch.inference_mode():
        assert torch.equal(original(sample), imported(sample))
    assert summarize_spec(spec) == '6 → 5 → Tanh → 3 → Softmax'

def test_relu_regression_model_returns_raw_output(tmp_path):
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    path = tmp_path / 'regression.json'
    export_sequential(model, 3, str(path), 'Regression')
    spec = ModelSpec.model_validate_json(path.read_text())
    trace, _ = run_trace(from_spec(spec), torch.tensor([[1., 2., 3.]]), spec.name)
    assert trace['probabilities'] is None
    assert trace['predicted_class'] is None
    assert len(trace['output']) == 2
