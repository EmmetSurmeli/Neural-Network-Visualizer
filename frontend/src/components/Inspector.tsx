import {useEffect, useState} from 'react';
import {Activity, ArrowUpRight, ChevronDown} from 'lucide-react';
import type {Trace, Neuron, NeuronSelection} from '../types';
import {api} from '../services/api';
import Histogram, {fmt} from './Histogram';
export default function Inspector({trace, selected, focusedNeuron}: {trace: Trace | null; selected: number; focusedNeuron: NeuronSelection | null}) {
  const [tab, setTab] = useState('layer'), [index, setIndex] = useState(0), [limit, setLimit] = useState(20);
  const [neuron, setNeuron] = useState<Neuron | null>(null), [error, setError] = useState('');
  const layer = trace?.layers[selected];
  useEffect(() => {setIndex(0); setNeuron(null); setError(''); setTab('layer');}, [selected, trace?.trace_id]);
  useEffect(() => {if (focusedNeuron?.layer === selected) {setIndex(focusedNeuron.index); setTab('neuron');}}, [focusedNeuron, selected]);
  useEffect(() => {
    setNeuron(null); setError('');
    if (!trace || layer?.type !== 'Linear' || tab !== 'neuron') return;
    const abort = new AbortController();
    api.neuron(trace.trace_id, selected, index, limit, abort.signal).then(setNeuron).catch(e => {if (!abort.signal.aborted) setError(e.message);});
    return () => abort.abort();
  }, [trace?.trace_id, selected, index, limit, tab, layer?.type]);
  return <aside className="inspector panel" id="inspector"><div className="panel-heading"><h2>Inspector</h2></div>
    {!layer ? <div className="inspector-empty"><Activity size={28}/><h3>No layer selected</h3><p>Run a forward pass and select a layer to explore its activations.</p></div> : <>
      <div className="inspector-title"><span className="eyebrow">LAYER {String(selected + 1).padStart(2, '0')}</span><h3>{layer.type}<span className="mono">{layer.name}</span></h3><p className="mono">[{layer.input_shape.join(', ')}] <span className="muted">→</span> [{layer.output_shape.join(', ')}]</p></div>
      <div className="tabs"><button className={tab === 'layer' ? 'active' : ''} onClick={() => setTab('layer')}>Layer details</button><button disabled={layer.type !== 'Linear'} title={layer.type !== 'Linear' ? 'Select a Linear layer to inspect weights' : undefined} className={tab === 'neuron' ? 'active' : ''} onClick={() => setTab('neuron')}>Neuron explorer <ArrowUpRight size={12}/></button></div>
      <div className="inspector-content">{tab === 'layer' ? <>
        <div className="section-label">Activation statistics</div><div className="stats-grid">{(['min', 'max', 'mean', 'std'] as const).map(key => <div key={key}><span>{key === 'std' ? 'Std. deviation' : key}</span><strong className="mono">{fmt(layer.stats[key])}</strong></div>)}</div>
        <div className="zero-stat"><span>Zero activations</span><strong className="mono">{layer.stats.percent_zero.toFixed(1)}%</strong></div><div className="zero-track"><i style={{width: `${layer.stats.percent_zero}%`}}/></div>
        <div className="section-label space-between">Activation distribution <span>{layer.activation_count} values</span></div><Histogram bins={layer.histogram} label="Histogram of layer activations"/>
        <div className="section-label space-between">Strongest activations <span>by magnitude</span></div>
        <div className="activation-grid">{layer.strongest.map(v => <button key={v.index} disabled={layer.type !== 'Linear'} title={`Neuron ${v.index}: ${fmt(v.value)}`} style={{backgroundColor: `rgba(130, 140, 145, ${0.04 + Math.abs(v.value) / (Math.max(Math.abs(layer.stats.min), Math.abs(layer.stats.max)) || 1) * .22})`}} onClick={() => {setIndex(v.index); setTab('neuron');}}><span>n{v.index}</span><strong>{v.value.toFixed(2)}</strong></button>)}</div>
        <details className="detail-section"><summary>Weakest activations <ChevronDown size={13}/></summary><div className="ranked-list">{layer.weakest.map(v => <span key={v.index} className="mono">n{v.index}: {fmt(v.value)}</span>)}</div></details>
        {layer.parameters.weight_histogram && <details className="detail-section"><summary>Weights & biases <span className="mono">{layer.parameters.count.toLocaleString()} params</span></summary><div className="weight-info"><span>Weight <b className="mono">[{layer.parameters.weight_shape?.join(' × ')}]</b></span><span>Bias <b className="mono">{layer.parameters.bias_shape ? `[${layer.parameters.bias_shape.join(', ')}]` : 'None'}</b></span></div><Histogram bins={layer.parameters.weight_histogram} color="var(--purple)" label="Weight distribution"/></details>}
      </> : <>
        <div className="neuron-controls"><label>Neuron<select value={index} onChange={e => setIndex(Number(e.target.value))}>{Array.from({length: layer.output_shape.at(-1) ?? 0}, (_, i) => <option key={i} value={i}>n{i}</option>)}</select></label><label>Top inputs<select value={limit} onChange={e => setLimit(Number(e.target.value))}>{[10, 20, 50].map(n => <option key={n}>{n}</option>)}</select></label></div>
        <div className="formula">z = Σ wᵢxᵢ + b</div>
        {error ? <p className="error" role="alert">{error}</p> : !neuron ? <p className="muted">Calculating contributions…</p> : <>
          <div className="stats-grid"><div><span>Pre-activation z</span><strong className="mono">{fmt(neuron.pre_activation)}</strong></div><div><span>After {neuron.activation}</span><strong className="mono accent">{fmt(neuron.post_activation)}</strong></div><div><span>Σ contributions</span><strong className="mono">{fmt(neuron.contribution_sum)}</strong></div><div><span>Bias</span><strong className="mono">{fmt(neuron.bias)}</strong></div></div>
          <div className="section-label space-between">Input contributions <span>top {Math.min(limit, neuron.input_count)} of {neuron.input_count}</span></div><p className="tiny muted">Each bar is input × weight, ranked by absolute value.</p>
          <div className="contribution-legend"><span><i/> Positive</span><span><i/> Negative</span></div>
          <div className="contributions">{neuron.top_contributions.map(c => <div key={c.index} className="contribution-row" title={`${fmt(c.input)} × ${fmt(c.weight)} = ${fmt(c.contribution)}`}><span className="mono">x{c.index}</span><div className="signed-track"><i className={c.contribution < 0 ? 'negative' : ''} style={{width: `${Math.abs(c.contribution) / (Math.max(...neuron.top_contributions.map(v => Math.abs(v.contribution))) || 1) * 48}%`}}/></div><strong className="mono">{c.contribution > 0 ? '+' : ''}{c.contribution.toFixed(3)}</strong></div>)}</div>
          {(['positive', 'negative'] as const).map(sign => <details className="detail-section" key={sign}><summary>Strongest {sign} inputs <ChevronDown size={13}/></summary><div className="ranked-list">{neuron[sign].length ? neuron[sign].map(c => <span key={c.index} className="mono">x{c.index}: {fmt(c.input)} × {fmt(c.weight)} = {fmt(c.contribution)}</span>) : <p className="muted">No {sign} contributions.</p>}</div></details>)}
        </>}
      </>}</div>
    </>}
  </aside>;
}
