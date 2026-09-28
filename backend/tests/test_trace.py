import pytest
import torch
from torch import nn
from backend.instrumentation.trace import run_trace, statistics, histogram, neuron_details
from backend.models.demo import load_demo, samples

def test_demo_inference_and_all_presets():
    model = load_demo()
    x = torch.tensor([s['pixels'] for s in samples()])
    with torch.inference_mode():
        probs = model(x)
    assert probs.shape == (10, 10)
    assert probs.argmax(1).tolist() == list(range(10))
    assert torch.allclose(probs.sum(-1), torch.ones(10), atol=1e-6)

def test_hook_registration_order_and_cleanup():
    model = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1)).train()
    model[1].eval()  # preserve mixed train/eval states too
    baseline = [len(m._forward_hooks) for m in model]
    trace, raw = run_trace(model, torch.ones(1, 3), 'test')
    assert [l['name'] for l in trace['layers']] == ['0', '1', '2']
    assert [l['order'] for l in trace['layers']] == [0, 1, 2]
    assert trace['layers'][0]['input_shape'] == [1, 3]
    assert trace['layers'][-1]['output_shape'] == [1, 1]
    assert [len(m._forward_hooks) for m in model] == baseline
    assert model.training and not model[1].training
    assert all(not item['output'].requires_grad for item in raw)
    assert all(item['output'].device.type == 'cpu' for item in raw)

def test_cleanup_on_forward_failure():
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(5, 1)).train()
    with pytest.raises(RuntimeError):
        run_trace(model, torch.ones(1, 3), 'bad')
    assert all(not m._forward_hooks for m in model)
    assert model.training

def test_activation_statistics_and_histogram():
    values = torch.tensor([-2., 0., 0., 2.])
    stats = statistics(values)
    assert stats == pytest.approx({'min': -2, 'max': 2, 'mean': 0, 'std': 2 ** .5, 'percent_zero': 50})
    bins = histogram(values)
    assert sum(b['count'] for b in bins) == 4
    assert sum(b['count'] for b in histogram(torch.zeros(10))) == 10

def test_linear_contributions_known_arithmetic_and_activation():
    layer = nn.Linear(3, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2., -3., 1.], [-1., 0., 0.]]))
        layer.bias.copy_(torch.tensor([.5, -.5]))
    model = nn.Sequential(layer, nn.ReLU())
    _, raw = run_trace(model, torch.tensor([[1., 2., 3.]]), 'known')
    details = neuron_details(raw, 0, 0)
    assert details['contribution_sum'] == -1
    assert details['bias'] == .5
    assert details['pre_activation'] == -.5
    assert details['post_activation'] == 0
    assert details['activation'] == 'ReLU'
    assert [c['index'] for c in details['top_contributions']] == [1, 2, 0]
    assert details['negative'][0]['contribution'] == -6
    assert details['positive'][0]['contribution'] == 3
    assert details['contribution_sum'] + details['bias'] == details['pre_activation']

def test_supported_activation_types_dropout_and_flatten():
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 3), nn.Sigmoid(), nn.Tanh(), nn.Dropout(.9), nn.Softmax(dim=-1))
    trace, raw = run_trace(model, torch.ones(1, 2, 2), 'activations')
    assert trace['layers'][0]['output_shape'] == [1, 4]
    assert torch.equal(raw[3]['output'], raw[4]['output'])
    assert sum(trace['probabilities']) == pytest.approx(1)

def test_large_tensor_summaries_are_bounded():
    trace, _ = run_trace(nn.Sequential(nn.ReLU()), torch.arange(2048.).unsqueeze(0), 'wide')
    layer = trace['layers'][0]
    assert layer['truncated'] and len(layer['activations']) == 512
    assert len(layer['strongest']) == 20
    assert layer['strongest'][0]['index'] == 2047
    assert layer['weakest'][0]['index'] == 0
    assert sum(b['count'] for b in layer['histogram']) == 2048
