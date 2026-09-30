"""Small trained CNNs on reproducible synthetic images (no downloads)."""
from collections import OrderedDict
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).parent
VISION_CLASSES = {'shapes-cnn': ['Circle', 'Square', 'Triangle'], 'strokes-cnn': ['Horizontal', 'Vertical', 'Diagonal']}

def architecture():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1, 8, 3, padding=1)), ('relu1', nn.ReLU()), ('pool1', nn.MaxPool2d(2)),
        ('conv2', nn.Conv2d(8, 16, 3, padding=1)), ('relu2', nn.ReLU()), ('pool2', nn.MaxPool2d(2)),
        ('flatten', nn.Flatten()), ('fc1', nn.Linear(16 * 7 * 7, 32)), ('relu3', nn.ReLU()),
        ('fc2', nn.Linear(32, 3)), ('probabilities', nn.Softmax(dim=-1)),
    ]))

def generate(kind, count, seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:28, :28]
    images, labels = [], []
    for i in range(count):
        label = i % 3
        xx, yy = x - rng.uniform(12, 16), y - rng.uniform(12, 16)
        size, thickness = rng.uniform(7, 10), rng.uniform(1.2, 2.5)
        if kind == 'shapes-cnn':
            if label == 0:
                distance = np.abs(np.sqrt(xx ** 2 + yy ** 2) - size)
                mask = distance <= thickness / 2
            elif label == 1:
                distance = np.abs(np.maximum(np.abs(xx), np.abs(yy)) - size)
                mask = distance <= thickness / 2
            else:
                # Distance to the three finite edges of an upright triangle.
                vertices = [(-size, size * .8), (size, size * .8), (0, -size)]
                distances = []
                for a, b in zip(vertices, vertices[1:] + vertices[:1]):
                    dx, dy = b[0] - a[0], b[1] - a[1]
                    t = np.clip(((xx - a[0]) * dx + (yy - a[1]) * dy) / (dx * dx + dy * dy), 0, 1)
                    distances.append(np.hypot(xx - a[0] - t * dx, yy - a[1] - t * dy))
                mask = np.minimum.reduce(distances) <= thickness / 2
        else:
            angle = [0, np.pi / 2, np.pi / 4][label] + rng.uniform(-.16, .16)
            along, across = xx * np.cos(angle) + yy * np.sin(angle), -xx * np.sin(angle) + yy * np.cos(angle)
            mask = (np.abs(across) <= thickness / 2) & (np.abs(along) <= size)
        image = np.clip(mask * rng.uniform(.7, 1.) + rng.normal(0, .035, (28, 28)), 0, 1).astype(np.float32)
        images.append(image)
        labels.append(label)
    return torch.from_numpy(np.array(images)[:, None]), torch.tensor(labels)

def load(kind):
    model = architecture()
    model.load_state_dict(torch.load(ROOT / f'{kind}.pt', map_location='cpu', weights_only=True))
    return model.eval(), json.loads((ROOT / f'{kind}.json').read_text())

def samples(kind):
    return json.loads((ROOT / f'{kind}-samples.json').read_text())
