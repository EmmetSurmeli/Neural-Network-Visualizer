"""Bounded JSON summaries plus private CPU captures for on-demand inspection."""
from time import perf_counter
import torch
from torch import nn
from backend.instrumentation.visualization import visualization

SUPPORTED = (nn.Linear, nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softmax, nn.Dropout, nn.Flatten, nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)

def statistics(tensor):
    values = tensor.detach().double().flatten().cpu()
    return {'min': values.min().item(), 'max': values.max().item(),
            'mean': values.mean().item(), 'std': values.std(unbiased=False).item(),
            'percent_zero': (values == 0).float().mean().item() * 100}

def histogram(tensor, bins=24):
    values = tensor.detach().double().flatten().cpu()
    lo, hi = values.min().item(), values.max().item()
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    counts = torch.histc(values, bins=bins, min=lo, max=hi).int().tolist()
    return [{'start': lo + i * (hi - lo) / bins, 'end': lo + (i + 1) * (hi - lo) / bins,
             'count': count} for i, count in enumerate(counts)]

def ranked(tensor, limit=20, strongest=True):
    values = tensor.flatten()
    ids = torch.argsort(values.abs(), descending=strongest, stable=True)[:limit]
    return [{'index': i.item(), 'value': values[i].item()} for i in ids]

def feature_maps(tensor):
    if tensor.ndim != 4:
        return None
    maps = tensor[0, :32]
    h, w = maps.shape[-2:]
    if h > 28 or w > 28:
        maps = torch.nn.functional.adaptive_avg_pool2d(maps, (min(h, 28), min(w, 28)))
    return {'channels': tensor.shape[1], 'height': maps.shape[-2], 'width': maps.shape[-1],
            'scale': max(maps.abs().max().item(), 1e-12), 'downsampled': h > 28 or w > 28,
            'values': [m.flatten().tolist() for m in maps]}

def run_trace(model: nn.Sequential, tensor: torch.Tensor, model_name: str):
    if not isinstance(model, nn.Sequential):
        raise ValueError('Only nn.Sequential models are supported.')
    modules = [(name, module) for name, module in model.named_modules() if name and not isinstance(module, nn.Sequential)]
    if any(not isinstance(module, SUPPORTED) for _, module in modules):
        raise ValueError('The model contains an unsupported layer.')
    captured, handles = [], []
    training_states = [(m, m.training) for m in model.modules()]

    def hook(name):
        def capture(module, args, output):
            x, y = args[0].detach().cpu().clone(), output.detach().cpu().clone()
            if not torch.isfinite(y).all():
                raise ValueError('The model produced non-finite activations; reduce input or weight magnitudes.')
            captured.append({'name': name, 'module': module, 'input': x, 'output': y})
        return capture

    try:
        model.eval()
        for name, module in modules:
            handles.append(module.register_forward_hook(hook(name)))
        start = perf_counter()
        with torch.inference_mode():
            output = model(tensor)
        elapsed = (perf_counter() - start) * 1000
    finally:
        for handle in handles:
            handle.remove()
        for module, training in training_states:
            module.training = training

    layers = []
    for index, item in enumerate(captured):
        module, x, y = item['module'], item['input'], item['output']
        params = {'count': sum(p.numel() for p in module.parameters(recurse=False))}
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            params.update(weight_shape=list(module.weight.shape),
                          bias_shape=list(module.bias.shape) if module.bias is not None else None,
                          weight_histogram=histogram(module.weight), weight_stats=statistics(module.weight))
        values = y.flatten()
        layers.append({'id': str(index), 'name': item['name'], 'type': type(module).__name__,
                       'order': index, 'input_shape': list(x.shape), 'output_shape': list(y.shape),
                       'activations': values[:512].tolist(), 'activation_count': values.numel(),
                       'truncated': values.numel() > 512, 'stats': statistics(y), 'histogram': histogram(y),
                       'feature_maps': feature_maps(y), 'strongest': ranked(y), 'weakest': ranked(y, strongest=False), 'parameters': params})
    result = output.flatten()
    is_probability = isinstance(captured[-1]['module'], nn.Softmax)
    return {'model_name': model_name, 'layers': layers, 'output': result.tolist(),
            'probabilities': result.tolist() if is_probability else None,
            'predicted_class': int(result.argmax()) if is_probability else None,
            'elapsed_ms': elapsed, 'parameter_count': sum(p.numel() for p in model.parameters()),
            'input_shape': list(tensor.shape), 'input_maps': feature_maps(tensor), 'visualization': visualization(captured)}, captured

def neuron_details(captured, layer_index, neuron_index, limit=20):
    item = captured[layer_index]
    module = item['module']
    if not isinstance(module, nn.Linear):
        raise ValueError('Select a Linear layer to inspect weighted input contributions.')
    if not 0 <= neuron_index < module.out_features:
        raise ValueError('Neuron index is outside this layer.')
    x = item['input'].flatten()
    w = module.weight.detach().cpu()[neuron_index]
    contributions = x * w
    def entries(indices):
        return [{'index': int(i), 'input': x[i].item(), 'weight': w[i].item(),
                 'contribution': contributions[i].item()} for i in indices]
    ordered = torch.argsort(contributions.abs(), descending=True, stable=True)
    z = item['output'].flatten()[neuron_index].item()
    post, activation = z, 'Identity'
    if layer_index + 1 < len(captured):
        nxt = captured[layer_index + 1]
        if isinstance(nxt['module'], (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softmax)):
            post = nxt['output'].flatten()[neuron_index].item()
            activation = type(nxt['module']).__name__
    bias = module.bias.detach().cpu()[neuron_index].item() if module.bias is not None else 0.0
    return {'index': neuron_index, 'pre_activation': z, 'post_activation': post,
            'activation': activation, 'bias': bias, 'contribution_sum': contributions.sum().item(),
            'top_contributions': entries(ordered[:limit]),
            'positive': entries(ordered[contributions[ordered] > 0][:limit]),
            'negative': entries(ordered[contributions[ordered] < 0][:limit]),
            'input_count': x.numel()}
