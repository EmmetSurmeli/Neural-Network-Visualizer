import {useMemo, useState} from 'react';
import type {NeuronSelection, Trace} from '../types';
import {projectDiagram, displayed, neuronRadius} from '../lib/diagram';
import {fmt} from './Histogram';

interface Props {zoom: number; nodeLimit: number; trace: Trace; step: number; progress: number; selected: number; neuron: NeuronSelection | null; select: (layer: number) => void; inspectNeuron: (layer: number, index: number) => void}
const CYAN = '#65b4c6', RED = '#ca7476';
const clamp = (v: number) => Math.max(0, Math.min(1, v));

export default function NeuronDiagram({zoom, nodeLimit, trace, step, progress, selected, neuron, select, inspectNeuron}: Props) {
  const [hover, setHover] = useState<{column: number; index: number} | null>(null);
  const {columns, connections} = useMemo(() => projectDiagram(trace, nodeLimit), [trace, nodeLimit]);
  const width = Math.max(820, columns.length * 230), height = 536;
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
  const focused = hover ?? (neuron ? {column: columns.findIndex(c => c.linear_layer === neuron.layer), index: neuron.index} : null);
  return <svg className="neuron-diagram" viewBox={`0 0 ${width} ${height}`} style={{height: 460 * zoom}} role="group" aria-label="Animated neurons and weighted connections">
    <title>Neuron activations and input-weight contributions for this forward pass</title>
    <g className="neural-edges" aria-hidden="true">{edges.map((edge, i) => {
      const x1 = x(edge.from) + neuronRadius(columns[edge.from].indices.length), x2 = x(edge.to) - neuronRadius(columns[edge.to].indices.length), y1 = y(edge.from, edge.source), y2 = y(edge.to, edge.target);
      const active = step === edge.step, visited = step >= edge.step;
      const connected = focused && ((focused.column === edge.from && focused.index === edge.source) || (focused.column === edge.to && focused.index === edge.target));
      const opacity = focused ? connected ? .85 : .025 : visited ? .12 + edge.strength * .6 : .035;
      const color = edge.contribution === 0 ? '#535353' : edge.contribution > 0 ? CYAN : RED;
      return <g key={`${edge.step}-${i}`}><line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeOpacity={opacity} strokeWidth={connected ? 1.6 : .65 + edge.strength * .7}><title>x{edge.source} → n{edge.target}: {fmt(edge.input)} × {fmt(edge.weight)} = {fmt(edge.contribution)}</title></line>
        {active && progress < .85 && edge.contribution !== 0 && <g className="signal-particle" opacity={focused && !connected ? .05 : .9}>{[.08,.04,0].map((lag, j) => <circle key={j} cx={x1 + (x2 - x1) * clamp(progress / .85 - lag)} cy={y1 + (y2 - y1) * clamp(progress / .85 - lag)} r={1 + edge.strength * 1.8} fill={color} opacity={(j + 1) / 3}/>)}</g>}</g>;
    })}</g>
    {columns.map((column, c) => {
      const latest = column.states.filter(s => s.step <= step).at(-1);
      const active = column.states.some(s => s.step === step);
      const isSelected = column.states.some(s => s.step === selected);
      const label = column.label ?? (c === 0 ? trace.visualization.input_label ?? 'Input' : c === columns.length - 1 ? 'Output' : `Hidden ${c}`);
      const operation = column.states.filter(s => s.step >= 0).map(s => s.type).join(' → ');
      return <g key={c} className={`neuron-column ${active ? 'executing' : ''}`} data-column={c} data-state={latest?.name ?? 'pending'}>
        <text className="column-title" x={x(c)} y="26" textAnchor="middle">{label}</text>
        <text className="column-operation" x={x(c)} y="46" textAnchor="middle">{operation || 'x'}</text>
        <text className="column-count" x={x(c)} y="68" textAnchor="middle">{column.size} {column.linear_layer === null ? 'inputs' : 'neurons'}</text>
        {active && <line className="active-column-marker" x1={x(c) - 27} x2={x(c) + 27} y1="77" y2="77" stroke={CYAN} strokeWidth="1.5"/>}
        {column.indices.map((index, at) => {
          const value = displayed(column, at, step, progress), yy = y(c, index);
          const chosen = focused?.column === c && focused.index === index;
          const winner = c === columns.length - 1 && index === trace.predicted_class && step >= column.states.at(-1)!.step && progress >= 1;
          const inspect = () => {if (column.linear_layer !== null) inspectNeuron(column.linear_layer, index); else if (latest && latest.step >= 0) select(latest.step);};
          return <g key={index} role={column.linear_layer !== null ? 'button' : undefined} tabIndex={column.linear_layer !== null ? 0 : undefined} aria-label={`${label} neuron ${index}, ${value.name}: ${value.ready ? fmt(value.value) : 'not reached'}`} className="neuron" onClick={inspect} onKeyDown={e => {if (e.key === 'Enter' || e.key === ' ') {e.preventDefault(); inspect();}}} onMouseEnter={() => setHover({column: c, index})} onMouseLeave={() => setHover(null)} onFocus={() => setHover({column: c, index})} onBlur={() => setHover(null)}>
            <title>{value.name} · n{index}: {value.ready ? fmt(value.value) : 'not reached'}{column.linear_layer !== null ? ' · Click to inspect' : ''}</title>
            <rect x={x(c) - 32} y={yy - neuronRadius(column.indices.length) - 1} width="86" height={2 * neuronRadius(column.indices.length) + 2} fill="transparent"/>
            <circle cx={x(c)} cy={yy} r={neuronRadius(column.indices.length)} fill={`rgb(${value.brightness},${value.brightness},${value.brightness})`} stroke={chosen || winner ? CYAN : value.value < 0 ? RED : value.ready ? '#a2a2a2' : '#353535'} strokeWidth={chosen || winner ? 2 : 1.1}/>
            {label === 'Output' && column.size <= 20 ? <text className={`output-digit ${winner ? 'winner' : ''}`} x={x(c) + 20} y={yy + 7} style={{fontSize: trace.classes?.[index]?.length && trace.classes[index].length > 2 ? 12 : undefined}}>{trace.classes?.[index] ?? index}</text> : <text className="neuron-index" x={x(c) - 17} y={yy + 3} textAnchor="end">{index}</text>}
            {chosen && <text className="neuron-value" x={x(c) + 16} y={yy - 14}>{fmt(value.value)}</text>}
          </g>;
        })}
        <g aria-label={`${label} full layer activity`}><title>All {column.size} values: mean absolute activation per block; {latest ? ((latest.zero_fraction ?? 0) * 100).toFixed(1) : '—'}% exactly zero</title>{latest?.activity?.map((v, i, all) => <rect key={i} x={x(c) - 49 + i * 98 / all.length} y="464" width={98 / all.length - .4} height="8" fill={CYAN} opacity={.08 + .92 * v * (latest.step === step ? progress : 1)}/>)}</g>
        <text className="column-subset" x={x(c)} y="491" textAnchor="middle">{column.indices.length}/{column.size} shown{latest?.zero_fraction !== undefined ? ` · ${Math.round(latest.zero_fraction * 100)}% zero` : ''}</text>
        <text className={`column-state ${isSelected ? 'selected' : ''}`} x={x(c)} y="515" textAnchor="middle">{latest ? latest.name : '—'}</text>
      </g>;
    })}
  </svg>;
}
