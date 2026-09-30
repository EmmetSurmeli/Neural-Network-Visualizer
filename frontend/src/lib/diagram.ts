import type {Trace, VisualColumn} from '../types';
const clamp = (v: number) => Math.max(0, Math.min(1, v));

export function projectDiagram(trace: Trace, nodeLimit: number) {
    const columns = trace.visualization.columns.map((c, i, all) => {
      const keep = c.indices.map((_, j) => j).filter(j => i === 0 || i === all.length - 1 || c.indices.length <= nodeLimit || j % Math.ceil(c.indices.length / nodeLimit) === 0);
      return {...c, indices: keep.map(j => c.indices[j]), states: c.states.map(s => ({...s, values: keep.map(j => s.values[j])}))};
    });
    const connections = trace.visualization.connections.map(c => ({...c, edges: c.edges.filter(e => columns[c.source_column].indices.includes(e.source) && columns[c.target_column].indices.includes(e.target))}));
    return {columns, connections};
}

export function displayed(column: VisualColumn, at: number, step: number, progress: number) {
    const eligible = column.states.filter(state => state.step <= step);
    const active = eligible.at(-1);
    if (!active) return {value: 0, brightness: 0, ready: false, name: column.states[0].name};
    const previous = eligible.at(-2);
    const blend = active.step === -1 || active.step < step ? 1 : active.type === 'Linear' ? clamp((progress - .55) / .45) : clamp(progress);
    // Numeric labels always use recorded values; only brightness is interpolated.
    const value = active.values[at];
    const oldIntensity = active.type === 'Linear' ? 0 : Math.abs(previous?.values[at] ?? 0) / (previous?.scale ?? 1);
    const intensity = oldIntensity + (Math.abs(active.values[at]) / active.scale - oldIntensity) * blend;
    return {value, brightness: Math.round(clamp(intensity) * 232), ready: true, name: active.name};
  }

export function featureReveal(order: number, step: number, progress: number) {
  if (step < order) return 0;
  return step === order && order >= 0 ? clamp(progress) : 1;
}
export function neuronRadius(count: number) {return count > 16 ? 5 : 10;}

export function replayPosition(value: number, layerCount: number) {
  const bounded = Math.max(0, Math.min(layerCount + 1, value));
  if (bounded === layerCount + 1) return {step: layerCount - 1, progress: 1};
  return {step: Math.floor(bounded) - 1, progress: bounded % 1};
}
