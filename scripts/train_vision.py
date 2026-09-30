"""Rebuild synthetic CNN checkpoints and held-out evaluation metadata."""
import json
import torch
from backend.models.vision import ROOT, VISION_CLASSES, architecture, generate

def main():
    torch.set_num_threads(2)
    for kind, classes in VISION_CLASSES.items():
        torch.manual_seed(42)
        model = architecture()
        # Train logits; retain Softmax for the exported inference architecture.
        logits = torch.nn.Sequential(*list(model.children())[:-1])
        x, y = generate(kind, 1800, 42)
        test_x, test_y = generate(kind, 600, 4042)
        opt = torch.optim.Adam(logits.parameters(), lr=.003)
        for epoch in range(15):
            order = torch.randperm(len(y))
            for ids in order.split(100):
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(logits(x[ids]), y[ids])
                loss.backward()
                opt.step()
        model.eval()
        with torch.inference_mode():
            pred = model(test_x).argmax(-1)
            accuracy = (pred == test_y).float().mean().item()
        torch.save(model.state_dict(), ROOT / f'{kind}.pt')
        info = {'id': kind, 'name': 'Shapes · CNN' if kind == 'shapes-cnn' else 'Line orientation · CNN',
                'input_features': 784, 'input_shape': [1, 28, 28], 'input_kind': 'image', 'classes': classes,
                'parameters': sum(p.numel() for p in model.parameters()), 'dataset': 'Synthetic shapes' if kind == 'shapes-cnn' else 'Synthetic strokes',
                'architecture': 'Conv 8 → Pool → Conv 16 → Pool → 32 → 3', 'test_accuracy': accuracy,
                'description': 'Trained on generated outline shapes.' if kind == 'shapes-cnn' else 'Trained on generated lines at three orientations.',
                'training_samples': 1800, 'test_samples': 600, 'epochs': 15, 'training_seed': 42, 'test_seed': 4042}
        (ROOT / f'{kind}.json').write_text(json.dumps(info, indent=2))
        samples = []
        for label in range(3):
            i = int(torch.where((test_y == label) & (pred == label))[0][0])
            samples.append({'label': label, 'test_index': i, 'pixels': test_x[i].flatten().tolist()})
        (ROOT / f'{kind}-samples.json').write_text(json.dumps(samples))
        print(kind, 'held-out accuracy:', accuracy, flush=True)

if __name__ == '__main__':
    main()
