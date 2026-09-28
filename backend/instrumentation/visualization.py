"""Bounded, faithful projection of captured tensors for the neuron diagram."""
import torch
from torch import nn

NODE_LIMIT = 16
EDGE_LIMIT = 600

def representative_indices(values, limit=NODE_LIMIT):
    values = values.flatten()
    if values.numel() <= limit:
        return list(range(values.numel()))
    chosen = set(torch.argsort(values.abs(), descending=True, stable=True)[:limit // 2].tolist())
    for index in torch.linspace(0, values.numel() - 1, limit).round().long().tolist():
        chosen.add(index)
        if len(chosen) == limit:
            break
    return sorted(chosen)

def visualization(captured):
    first = captured[0]['input'].flatten()
    groups = [{'size': first.numel(), 'raw_states': [(-1, 'Input', 'Input', first)], 'linear_layer': None}]
    boundaries = []
    for index, item in enumerate(captured):
        values = item['output'].flatten()
        if isinstance(item['module'], nn.Linear):
            boundaries.append((index, len(groups) - 1, len(groups), item))
            groups.append({'size': values.numel(), 'raw_states': [], 'linear_layer': index})
        groups[-1]['raw_states'].append((index, item['name'], type(item['module']).__name__, values))
    columns = []
    for group in groups:
        indices = representative_indices(group['raw_states'][-1][3])
        states = [{'step': step, 'name': name, 'type': kind, 'values': values[indices].tolist(),
                   'scale': max(values.abs().max().item(), 1e-12)}
                  for step, name, kind, values in group['raw_states']]
        columns.append({'size': group['size'], 'indices': indices, 'states': states, 'linear_layer': group['linear_layer']})
    connections = []
    per_boundary = min(96, EDGE_LIMIT // max(1, len(boundaries)))
    for layer, source, target, item in boundaries:
        x = item['input'].flatten()
        weight = item['module'].weight.detach().cpu()
        edges = []
        for j in columns[target]['indices']:
            for i in columns[source]['indices']:
                edges.append({'source': i, 'target': j, 'weight': weight[j, i].item(), 'input': x[i].item(),
                              'contribution': (weight[j, i] * x[i]).item()})
        edges.sort(key=lambda e: abs(e['contribution']), reverse=True)
        connections.append({'step': layer, 'source_column': source, 'target_column': target,
                            'edges': edges[:per_boundary]})
    return {'columns': columns, 'connections': connections, 'node_limit': NODE_LIMIT, 'edge_limit': EDGE_LIMIT}
