"""Independent NumPy forward pass: no PyTorch execution or trace helpers in the reference."""
import numpy as np
import pytest
import torch
from torch import nn
from backend.models.catalog import load_catalog, IDS
from backend.models.demo import samples as digit_samples
from backend.models.vision import samples as vision_samples
from backend.instrumentation.trace import run_trace, neuron_details


def array(tensor):
    return tensor.detach().cpu().numpy().astype(np.float64)


def reference_step(module, x):
    if isinstance(module, nn.Linear):
        return x @ array(module.weight).T + (array(module.bias) if module.bias is not None else 0)
    if isinstance(module, nn.Conv2d):
        py, px = module.padding
        padded = np.pad(x, ((0, 0), (0, 0), (py, py), (px, px)))
        kh, kw = module.kernel_size
        windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(-2, -1))
        windows = windows[:, :, ::module.stride[0], ::module.stride[1]]
        result = np.einsum('ncyxij,ocij->noyx', windows, array(module.weight))
        return result + array(module.bias)[None, :, None, None]
    if isinstance(module, nn.MaxPool2d):
        k = module.kernel_size
        windows = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(-2, -1))
        return windows[:, :, ::module.stride, ::module.stride].max(axis=(-2, -1))
    if isinstance(module, nn.Flatten):
        return x.reshape(x.shape[0], -1)
    if isinstance(module, nn.ReLU):
        return np.maximum(x, 0)
    if isinstance(module, nn.Sigmoid):
        return 1 / (1 + np.exp(-x))
    if isinstance(module, nn.Tanh):
        return np.tanh(x)
    if isinstance(module, nn.Softmax):
        exp = np.exp(x - x.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)
    raise AssertionError(f'Unimplemented independent reference: {type(module).__name__}')


@pytest.mark.parametrize('model_id', IDS)
def test_visualization_against_independent_forward_math(model_id):
    model, info = load_catalog()[model_id]
    if model_id == 'mnist-mlp':
        inputs = [s['pixels'] for s in digit_samples()]
    elif model_id.endswith('cnn'):
        inputs = [s['pixels'] for s in vision_samples(model_id)]
    elif model_id == 'xor-mlp':
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    else:
        inputs = [[1, .5, -1], [-1, .5, 1], [0, 0, 0]]
    rng = np.random.default_rng(704)
    inputs += [rng.uniform(0, 1, info['input_features']).tolist()]
    max_error = 0.
    for values in inputs:
        tensor = torch.tensor([values], dtype=torch.float32).reshape(1, *info.get('input_shape', [len(values)]))
        trace, captured = run_trace(model, tensor, info['name'])
        reference = array(tensor)
        expected = []
        for index, (module, layer) in enumerate(zip(model.children(), trace['layers'])):
            incoming = reference.copy()
            reference = reference_step(module, reference)
            expected.append(reference.copy())
            actual = array(captured[index]['output'])
            max_error = max(max_error, float(np.abs(actual - reference).max()))
            np.testing.assert_allclose(actual, reference, rtol=3e-5, atol=2e-5)
            assert layer['input_shape'] == list(incoming.shape)
            assert layer['output_shape'] == list(reference.shape)
            assert layer['type'] == type(module).__name__
            np.testing.assert_allclose(layer['activations'], reference.flatten()[:512], rtol=3e-5, atol=2e-5)
            if layer['feature_maps']:
                np.testing.assert_allclose(layer['feature_maps']['values'], reference[0].reshape(reference.shape[1], -1), rtol=3e-5, atol=2e-5)
            if isinstance(module, nn.ReLU):
                # All meaningfully negative inputs must produce exactly dark/zero outputs.
                assert np.all(actual[incoming < -2e-5] == 0)
            if isinstance(module, nn.Linear):
                for neuron in [0, module.out_features - 1]:
                    details = neuron_details(captured, index, neuron, limit=50)
                    expected_z = reference.flatten()[neuron]
                    assert details['pre_activation'] == pytest.approx(expected_z, abs=2e-5, rel=3e-5)
                    assert details['contribution_sum'] + details['bias'] == pytest.approx(expected_z, abs=2e-5, rel=3e-5)
        np.testing.assert_allclose(trace['output'], reference.flatten(), rtol=3e-5, atol=2e-5)
        if trace['probabilities']:
            assert trace['predicted_class'] == int(reference.argmax())
        graph = trace['visualization']
        for col in graph['columns']:
            assert len(set(col['indices'])) == len(col['indices'])
            assert all(0 <= i < col['size'] for i in col['indices'])
            for state in col['states']:
                source = array(tensor) if state['step'] == -1 else expected[state['step']]
                np.testing.assert_allclose(state['values'], source.flatten()[col['indices']], rtol=3e-5, atol=2e-5)
        for connection in graph['connections']:
            step = connection['step']
            module = captured[step]['module']
            assert isinstance(module, nn.Linear)
            incoming = array(tensor).flatten() if step == 0 else expected[step - 1].flatten()
            for edge in connection['edges']:
                assert edge['source'] in graph['columns'][connection['source_column']]['indices']
                assert edge['target'] in graph['columns'][connection['target_column']]['indices']
                expected_contribution = incoming[edge['source']] * array(module.weight)[edge['target'], edge['source']]
                assert edge['contribution'] == pytest.approx(expected_contribution, abs=2e-5, rel=3e-5)
    print(f'{model_id}: {len(inputs)} inputs, max absolute activation error {max_error:.3g}')
