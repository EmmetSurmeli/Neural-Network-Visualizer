"""Controlled, declarative JSON import. No Python, pickle or torch.load uploads."""
import torch
from torch import nn
from backend.schemas.requests import ModelSpec

def from_spec(spec: ModelSpec):
    layers, width, total = [], spec.input_features, 0
    for layer in spec.layers:
        if layer.type == 'Linear':
            if not layer.weight or len(layer.weight) > 1024 or any(len(row) != width for row in layer.weight):
                raise ValueError(f'Linear weight must have 1–1024 rows and {width} columns.')
            out = len(layer.weight)
            total += out * width + (out if layer.bias is not None else 0)
            if total > 500_000:
                raise ValueError('V1 supports at most 500,000 parameters per imported model.')
            weight = torch.tensor(layer.weight, dtype=torch.float32)
            if not torch.isfinite(weight).all() or weight.abs().max() > 1e6:
                raise ValueError('Weights must be finite and at most 1,000,000 in magnitude.')
            module = nn.Linear(width, out, bias=layer.bias is not None)
            with torch.no_grad():
                module.weight.copy_(weight)
                if layer.bias is not None:
                    if len(layer.bias) != out or any(abs(x) > 1e6 for x in layer.bias):
                        raise ValueError('Bias length must equal output features; magnitude must be at most 1,000,000.')
                    module.bias.copy_(torch.tensor(layer.bias))
            width = out
        else:
            if layer.weight is not None or layer.bias is not None:
                raise ValueError('Only Linear layers accept weight and bias arrays.')
            constructors = {'ReLU': nn.ReLU, 'Sigmoid': nn.Sigmoid, 'Tanh': nn.Tanh,
                            'Softmax': lambda: nn.Softmax(dim=-1),
                            'Dropout': lambda: nn.Dropout(layer.p), 'Flatten': nn.Flatten}
            module = constructors[layer.type]()
        layers.append(module)
    return nn.Sequential(*layers).eval()
