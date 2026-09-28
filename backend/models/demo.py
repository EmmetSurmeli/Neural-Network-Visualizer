"""The bundled checkpoint contains tensors only; never load user pickle files."""
from collections import OrderedDict
from pathlib import Path
import json
import torch
from torch import nn

ROOT = Path(__file__).parent

def architecture():
    return nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(784, 128)), ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(128, 64)), ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(64, 10)), ('probabilities', nn.Softmax(dim=-1)),
    ]))

def load_demo():
    model = architecture()
    model.load_state_dict(torch.load(ROOT / 'digits.pt', map_location='cpu', weights_only=True))
    return model.eval()

def metadata():
    return json.loads((ROOT / 'metadata.json').read_text())

def samples():
    return json.loads((ROOT / 'samples.json').read_text())
