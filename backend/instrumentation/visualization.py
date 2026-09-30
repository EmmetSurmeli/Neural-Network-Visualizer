"""Bounded, faithful projection of captured tensors for the neuron diagram."""
import torch
from torch import nn

NODE_LIMIT = 32
EDGE_LIMIT = 600

def activation_summary(values):
    values = values.flatten()
    scale = max(values.abs().max().item(), 1e-12)
    return {'activity': [chunk.abs().mean().item() / scale for chunk in torch.tensor_split(values, min(48, values.numel()))],
            'zero_fraction': (values == 0).float().mean().item()}

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
    start = 0
    if captured[0]['input'].ndim == 4:
        start = next((i for i, item in enumerate(captured) if isinstance(item['module'], nn.Flatten)), None)
        if start is None:
            raise ValueError('CNN visualization requires Flatten before its dense layers.')
    first = captured[start]['input'].flatten()
    groups = [{'size': first.numel(), 'raw_states': [(start if start else -1, 'Flatten' if start else 'Input', 'Flatten' if start else 'Input', first)], 'linear_layer': None}]
    boundaries = []
    for index, item in enumerate(captured):
        if index < start or (start and index == start):
            continue
        values = item['output'].flatten()
        if isinstance(item['module'], nn.Linear):
            boundaries.append((index, len(groups) - 1, len(groups), item))
            groups.append({'size': values.numel(), 'raw_states': [], 'linear_layer': index})
        groups[-1]['raw_states'].append((index, item['name'], type(item['module']).__name__, values))
    columns = []
    for group in groups:
        indices = representative_indices(group['raw_states'][-1][3], 16 if group is groups[0] else NODE_LIMIT)
        states = [{'step': step, 'name': name, 'type': kind, 'values': values[indices].tolist(),
                   'scale': max(values.abs().max().item(), 1e-12), **activation_summary(values)}
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
    return {'columns': columns, 'connections': connections, 'node_limit': NODE_LIMIT, 'edge_limit': EDGE_LIMIT, 'input_label': 'Flatten' if start else 'Input'}

def focused_visualization(captured, layer_index, neuron_index):
    """Show true top incoming contributions across the full input, not the overview sample."""
    if not 0 <= layer_index < len(captured):
        raise ValueError('Layer not found.')
    item = captured[layer_index]
    module = item['module']
    if not isinstance(module, nn.Linear) or not 0 <= neuron_index < module.out_features:
        raise ValueError('Select a valid neuron in a Linear layer.')
    graph = visualization(captured)
    target = next(i for i, c in enumerate(graph['columns']) if c['linear_layer'] == layer_index)
    x = item['input'].flatten()
    weight = module.weight.detach().cpu()[neuron_index]
    contributions = x * weight
    indices = sorted(torch.argsort(contributions.abs(), descending=True, stable=True)[:16].tolist())
    columns = []
    for position, selected in [(target - 1, indices), (target, [neuron_index])]:
        original = graph['columns'][position]
        states = []
        for state in original['states']:
            if state['step'] == -1:
                values = captured[0]['input'].flatten()
            else:
                values = captured[state['step']]['output'].flatten()
            states.append({**state, 'values': values[selected].tolist()})
        label = graph['input_label'] if position == 0 else 'Output' if position == len(graph['columns']) - 1 else f'Hidden {position}'
        columns.append({**original, 'indices': selected, 'states': states, 'label': label})
    edges = [{'source': i, 'target': neuron_index, 'input': x[i].item(), 'weight': weight[i].item(),
              'contribution': contributions[i].item()} for i in indices]
    total = contributions.abs().sum().item()
    return {'diagram': {**graph, 'columns': columns, 'connections': [{'step': layer_index,
             'source_column': 0, 'target_column': 1, 'edges': edges}]},
            'shown': len(indices), 'total': x.numel(),
            'magnitude_coverage': contributions[indices].abs().sum().item() / total if total else 0,
            'bias': module.bias[neuron_index].item() if module.bias is not None else 0,
            'pre_activation': item['output'].flatten()[neuron_index].item()}
