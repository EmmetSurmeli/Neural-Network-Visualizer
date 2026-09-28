"""Call locally with your already-constructed, trusted PyTorch model."""
import json
from pathlib import Path
from torch import nn
from backend.schemas.requests import ModelSpec
from backend.services.models import from_spec

def export_sequential(model: nn.Sequential, input_features: int, path: str, name='Imported MLP'):
    if not isinstance(model, nn.Sequential):
        raise ValueError('Only Sequential models are supported.')
    layers = []
    for module in model.modules():
        if isinstance(module, nn.Sequential):
            continue
        layer = {'type': type(module).__name__}
        if isinstance(module, nn.Linear):
            layer['weight'] = module.weight.detach().cpu().tolist()
            layer['bias'] = module.bias.detach().cpu().tolist() if module.bias is not None else None
        elif isinstance(module, nn.Dropout):
            layer['p'] = module.p
        elif isinstance(module, nn.Softmax) and module.dim not in (1, -1):
            raise ValueError('Softmax must operate on the last feature dimension.')
        elif isinstance(module, nn.Flatten) and (module.start_dim != 1 or module.end_dim != -1):
            raise ValueError('Only Flatten(start_dim=1, end_dim=-1) is supported.')
        layers.append(layer)
    spec = ModelSpec(name=name, input_features=input_features, layers=layers)
    from_spec(spec)  # Validate dimensions and budgets before writing.
    Path(path).write_text(json.dumps(spec.model_dump(), allow_nan=False))
