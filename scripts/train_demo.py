"""Download MNIST, reproducibly train a small MLP, and save held-out samples."""
import gzip
import json
import sys
import urllib.request
from pathlib import Path
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.models.demo import ROOT, architecture

torch.manual_seed(42)
torch.set_num_threads(4)
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)

def read_file(name, offset):
    path = DATA / name
    if not path.exists():
        print(f'Downloading {name}', flush=True)
        urllib.request.urlretrieve('https://ossci-datasets.s3.amazonaws.com/mnist/' + name, path)
    return np.frombuffer(gzip.decompress(path.read_bytes()), dtype=np.uint8, offset=offset).copy()

def dataset(prefix):
    x = read_file(prefix + '-images-idx3-ubyte.gz', 16).reshape(-1, 784)
    y = read_file(prefix + '-labels-idx1-ubyte.gz', 8)
    return torch.tensor(x, dtype=torch.float32) / 255, torch.tensor(y, dtype=torch.long)

x, y = dataset('train')
test_x, test_y = dataset('t10k')
model = architecture()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(12):
    model.train()
    order = torch.randperm(len(x))
    for batch in order.split(256):
        optimizer.zero_grad()
        logits = model[:-1](x[batch])
        loss = loss_fn(logits, y[batch])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        accuracy = (model(test_x).argmax(1) == test_y).float().mean().item()
    print(f'Epoch {epoch + 1}/12 · test accuracy {accuracy:.2%}', flush=True)
torch.save(model.state_dict(), ROOT / 'digits.pt')
meta = {'id': 'mnist-mlp', 'name': 'MNIST · Digit classifier', 'dataset': 'MNIST',
        'input_features': 784, 'input_shape': [1, 784], 'classes': list(range(10)),
        'parameters': sum(p.numel() for p in model.parameters()), 'test_accuracy': accuracy,
        'training_samples': len(x), 'test_samples': len(test_x), 'seed': 42, 'epochs': 12}
(ROOT / 'metadata.json').write_text(json.dumps(meta, indent=2))
# Use the first correctly classified test example for each class; disclose selection.
with torch.inference_mode():
    pred = model(test_x).argmax(1)
examples = []
for digit in range(10):
    index = int(torch.where((test_y == digit) & (pred == digit))[0][0])
    examples.append({'label': digit, 'test_index': index, 'pixels': test_x[index].tolist()})
(ROOT / 'samples.json').write_text(json.dumps(examples))
example_dir = ROOT.parents[1] / 'examples'
example_dir.mkdir(exist_ok=True)
(example_dir / 'digit-7.json').write_text(json.dumps({'input': examples[7]['pixels']}, indent=2))
print(json.dumps(meta), flush=True)
