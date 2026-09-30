import json
from pathlib import Path
from backend.models.demo import load_demo, metadata
from backend.models.vision import load as load_vision, VISION_CLASSES
from backend.schemas.requests import ModelSpec
from backend.services.models import from_spec, summarize_spec

IDS = ['mnist-mlp', 'shapes-cnn', 'strokes-cnn', 'xor-mlp', 'tiny-mlp']

def load_catalog():
    info = {**metadata(), 'input_shape': [784], 'input_kind': 'image', 'classes': [str(i) for i in range(10)],
            'description': 'Trained on MNIST handwritten digits.'}
    entries = {'mnist-mlp': (load_demo(), info)}
    for kind in VISION_CLASSES:
        entries[kind] = load_vision(kind)
    for model_id, filename, sample in [('xor-mlp', 'xor-model.json', [0, 1]), ('tiny-mlp', 'tiny-model.json', [1, .5, -1])]:
        spec = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / 'examples' / filename).read_text())
        model = from_spec(spec)
        entries[model_id] = (model, {'id': model_id, 'name': spec.name, 'input_features': spec.input_features,
            'parameters': sum(p.numel() for p in model.parameters()), 'dataset': 'Numeric example',
            'architecture': summarize_spec(spec), 'default_input': sample,
            'description': 'Hand-set weights demonstrating XOR.' if model_id == 'xor-mlp' else 'Hand-set weights for checking the arithmetic.'})
    return entries
