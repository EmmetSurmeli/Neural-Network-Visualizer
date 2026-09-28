import {useMemo, useState} from 'react';
import type {NeuronSelection, Trace, VisualColumn} from '../types';
import {fmt} from './Histogram';

interface Props {trace: Trace; step: number; progress: number; selected: number; neuron: NeuronSelection | null; select: (layer: number) => void; inspectNeuron: (layer: number, index: number) => void}
const CYAN = '#65b4c6', RED = '#ca7476';
const clamp = (v: number) => Math.max(0, Math.min(1, v));

export default function NeuronDiagram({trace, step, progress, selected, neuron, select, inspectNeuron}: Props) {
  const [hover, setHover] = useState<{column: number; index: number} | null>(null);
  const {columns, connections} = trace.visualization;
  const width = Math.max(820, columns.length * 230), height = 516;
  const x = (c: number) => columns.length === 1 ? width / 2 : 68 + c * (width - 148) / (columns.length - 1);
  const y = (c: number, index: number) => {
    const count = columns[c].indices.length, at = columns[c].indices.indexOf(index);
    const spacing = Math.min(33, 352 / Math.max(1, count - 1));
    return 272 - (count - 1) * spacing / 2 + at * spacing;
  };
  const edges = useMemo(() => connections.flatMap(connection => {
    const max = Math.max(1e-12, ...connection.edges.map(e => Math.abs(e.contribution)));
    return connection.edges.map(edge => ({...edge, step: connection.step, from: connection.source_column, to: connection.target_column, strength: Math.abs(edge.contribution) / max}));
  }), [connections]);
  function displayed(column: VisualColumn, at: number) {
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
  const focused = hover ?? (neuron ? {column: columns.findIndex(c => c.linear_layer === neuron.layer), index: neuron.index} : null);
  return <svg className="neuron-diagram" viewBox={`0 0 ${width} ${height}`} style={{minWidth: columns.length > 4 ? width : 660}} role="group" aria-label="Animated neurons and weighted connections">
    <title>Neuron activations and input-weight contributions for this forward pass</title>
    <g className="neural-edges" aria-hidden="true">{edges.map((edge, i) => {
      const x1 = x(edge.from) + 10, x2 = x(edge.to) - 10, y1 = y(edge.from, edge.source), y2 = y(edge.to, edge.target);
      const active = step === edge.step, visited = step >= edge.step;
      const connected = focused && ((focused.column === edge.from && focused.index === edge.source) || (focused.column === edge.to && focused.index === edge.target));
      const opacity = focused ? connected ? .85 : .025 : visited ? .12 + edge.strength * .6 : .035;
      const color = edge.contribution === 0 ? '#535353' : edge.contribution > 0 ? CYAN : RED;
      return <g key={`${edge.step}-${i}`}><line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeOpacity={opacity} strokeWidth={connected ? 1.6 : .65 + edge.strength * .7}><title>x{edge.source} → n{edge.target}: {fmt(edge.input)} × {fmt(edge.weight)} = {fmt(edge.contribution)}</title></line>
        {active && progress < .85 && edge.contribution !== 0 && <circle className="signal-particle" cx={x1 + (x2 - x1) * clamp(progress / .85)} cy={y1 + (y2 - y1) * clamp(progress / .85)} r={1 + edge.strength * 1.4} fill={color} opacity={focused && !connected ? .08 : .85}/>}</g>;
    })}</g>
    {columns.map((column, c) => {
      const latest = column.states.filter(s => s.step <= step).at(-1);
      const active = column.states.some(s => s.step === step);
      const isSelected = column.states.some(s => s.step === selected);
      const label = c === 0 ? 'Input' : c === columns.length - 1 ? 'Output' : `Hidden ${c}`;
      const operation = column.states.filter(s => s.step >= 0).map(s => s.type).join(' → ');
      return <g key={c} className={`neuron-column ${active ? 'executing' : ''}`} data-column={c} data-state={latest?.name ?? 'pending'}>
        <text className="column-title" x={x(c)} y="26" textAnchor="middle">{label}</text>
        <text className="column-operation" x={x(c)} y="46" textAnchor="middle">{operation || 'x'}</text>
        <text className="column-count" x={x(c)} y="68" textAnchor="middle">{column.size} {c === 0 ? 'inputs' : 'neurons'}</text>
        {active && <line className="active-column-marker" x1={x(c) - 27} x2={x(c) + 27} y1="77" y2="77" stroke={CYAN} strokeWidth="1.5"/>}
        {column.indices.map((index, at) => {
          const value = displayed(column, at), yy = y(c, index);
          const chosen = focused?.column === c && focused.index === index;
          const winner = c === columns.length - 1 && index === trace.predicted_class && step >= column.states.at(-1)!.step && progress >= 1;
          const inspect = () => {if (column.linear_layer !== null) inspectNeuron(column.linear_layer, index); else if (latest && latest.step >= 0) select(latest.step);};
          return <g key={index} role={column.linear_layer !== null ? 'button' : undefined} tabIndex={column.linear_layer !== null ? 0 : undefined} aria-label={`${label} neuron ${index}, ${value.name}: ${value.ready ? fmt(value.value) : 'not reached'}`} className="neuron" onClick={inspect} onKeyDown={e => {if (e.key === 'Enter' || e.key === ' ') {e.preventDefault(); inspect();}}} onMouseEnter={() => setHover({column: c, index})} onMouseLeave={() => setHover(null)} onFocus={() => setHover({column: c, index})} onBlur={() => setHover(null)}>
            <title>{value.name} · n{index}: {value.ready ? fmt(value.value) : 'not reached'}{column.linear_layer !== null ? ' · Click to inspect' : ''}</title>
            <circle cx={x(c)} cy={yy} r={10} fill={`rgb(${value.brightness},${value.brightness},${value.brightness})`} stroke={chosen || winner ? CYAN : value.value < 0 ? RED : value.ready ? '#a2a2a2' : '#353535'} strokeWidth={chosen || winner ? 2 : 1.1}/>
            {c === columns.length - 1 && column.size <= 20 ? <text className={`output-digit ${winner ? 'winner' : ''}`} x={x(c) + 20} y={yy + 7}>{index}</text> : <text className="neuron-index" x={x(c) - 17} y={yy + 3} textAnchor="end">{index}</text>}
            {chosen && <text className="neuron-value" x={x(c) + 16} y={yy - 14}>{fmt(value.value)}</text>}
          </g>;
        })}
        <text className="column-subset" x={x(c)} y="474" textAnchor="middle">{column.indices.length} of {column.size} shown</text>
        <text className={`column-state ${isSelected ? 'selected' : ''}`} x={x(c)} y="496" textAnchor="middle">{latest ? latest.name : '—'}</text>
      </g>;
    })}
  </svg>;
}
