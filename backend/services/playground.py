"""Small deterministic classification experiments and recorded training snapshots."""
import numpy as np
import torch
from torch import nn
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from uuid import uuid4

Dataset = Literal['xor', 'moons', 'circles', 'spiral']
NAMES = {'xor': 'XOR', 'moons': 'Two moons', 'circles': 'Circles', 'spiral': 'Spiral'}

class PlaygroundConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dataset: Dataset = 'moons'
    hidden_layers: list[Literal[2, 4, 8, 16]] = Field(default_factory=lambda: [8, 8], min_length=1, max_length=3)
    activation: Literal['ReLU', 'Tanh', 'Sigmoid'] = 'Tanh'
    learning_rate: Literal[0.001, 0.01, 0.05] = 0.01
    epochs: Literal[100, 300, 1000] = 300
    noise: float = Field(default=0.1, ge=0, le=0.8, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**31 - 1, strict=True)

def dataset_points(dataset: Dataset, seed=42, noise=0.1):
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], 100)
    if dataset == 'xor':
        centers = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])
        points = np.repeat(centers, 50, axis=0)
    elif dataset == 'moons':
        angle = np.linspace(0, np.pi, 100)
        points = np.concatenate([np.c_[np.cos(angle), np.sin(angle)], np.c_[1 - np.cos(angle), .5 - np.sin(angle)]])
    elif dataset == 'circles':
        angle = rng.uniform(0, 2 * np.pi, 200)
        radius = np.where(labels == 0, .5, 1.)
        points = np.c_[radius * np.cos(angle), radius * np.sin(angle)]
    else:
        radius = np.linspace(.08, 1, 100)
        angle = radius * 2 * np.pi
        arm = np.c_[radius * np.cos(angle), radius * np.sin(angle)]
        points = np.concatenate([arm, -arm])
    # Fit standardization on the training split only.
    train = np.zeros(200, dtype=bool)
    for label in (0, 1):
        train[rng.permutation(np.flatnonzero(labels == label))[:80]] = True
    # Noise is measured relative to each clean feature's training standard deviation.
    points = points + rng.normal(size=points.shape) * noise * points[train].std(axis=0)
    points = (points - points[train].mean(axis=0)) / points[train].std(axis=0)
    points = points.astype(np.float32)
    bound = float(np.ceil(float(np.abs(points).max()) * 1.1 * 10) / 10)
    return points, labels, train, bound

def dataset_payload(dataset: Dataset, seed=42, noise=0.1):
    points, labels, train, bound = dataset_points(dataset, seed, noise)
    return {'dataset': dataset, 'points': [
        {'id': i, 'input': p.tolist(), 'label': int(labels[i]), 'split': 'train' if train[i] else 'validation'}
        for i, p in enumerate(points)
    ], 'grid': {'size': 48, 'min': -bound, 'max': bound}}

def train_model(config: PlaygroundConfig):
    points, labels, mask, bound = dataset_points(config.dataset, config.seed, config.noise)
    x, y = torch.from_numpy(points), torch.from_numpy(labels)
    train_mask = torch.from_numpy(mask)
    coordinates = torch.linspace(-bound, bound, 48)
    gy, gx = torch.meshgrid(coordinates, coordinates, indexing='ij')
    grid = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    # The API serializes this RNG scope with other model operations.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        layers, width = [], 2
        for next_width in config.hidden_layers:
            layers.extend([nn.Linear(width, next_width), getattr(nn, config.activation)()])
            width = next_width
        layers.append(nn.Linear(width, 2))
        logits_model = nn.Sequential(*layers)
        optimizer = torch.optim.Adam(logits_model.parameters(), lr=config.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        snapshots = []
        for epoch in range(config.epochs + 1):
            if epoch:
                optimizer.zero_grad()
                loss = loss_fn(logits_model(x[train_mask]), y[train_mask])
                loss.backward()
                optimizer.step()
            if epoch % 5 == 0:
                with torch.inference_mode():
                    logits = logits_model(x)
                    probabilities = logits.softmax(-1)[:, 1]
                    loss = loss_fn(logits[train_mask], y[train_mask]).item()
                    if not torch.isfinite(logits).all() or not np.isfinite(loss):
                        raise ValueError('Training became unstable. Try a smaller learning rate.')
                    correct = logits.argmax(-1) == y
                    snapshots.append({
                        'epoch': epoch, 'loss': loss,
                        'validation_loss': loss_fn(logits[~train_mask], y[~train_mask]).item(),
                        'train_accuracy': correct[train_mask].float().mean().item(),
                        'validation_accuracy': correct[~train_mask].float().mean().item(),
                        'grid_probabilities': logits_model(grid).softmax(-1)[:, 1].tolist(),
                        'point_probabilities': probabilities.tolist(),
                    })
        model = nn.Sequential(*layers, nn.Softmax(dim=-1)).eval()
    info = {'id': f'playground-{uuid4()}', 'name': f'{NAMES[config.dataset]} · Playground',
            'input_features': 2, 'parameters': sum(p.numel() for p in model.parameters()),
            'dataset': NAMES[config.dataset],
            'architecture': ' → '.join(['2'] + [f'{w} → {config.activation}' for w in config.hidden_layers] + ['2', 'Softmax'])}
    last = snapshots[-1]
    return model, {**dataset_payload(config.dataset, config.seed, config.noise), 'config': config.model_dump(),
                   'model_id': info['id'], 'model': info, 'snapshots': snapshots,
                   'metrics': {k: last[k] for k in ('loss', 'train_accuracy', 'validation_accuracy')}}
