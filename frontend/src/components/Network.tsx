import {ArrowLeft, Minus, Plus, Pause, Play, RotateCcw, SkipBack, SkipForward} from 'lucide-react';
import {useEffect, useRef, useState} from 'react';
import type {FocusView, NeuronSelection, Trace} from '../types';
import {api} from '../services/api';
import type usePlayback from '../hooks/usePlayback';
import FeatureMaps from './FeatureMaps';
import NeuronDiagram from './NeuronDiagram';
import {replayPosition} from '../lib/diagram';

interface Props {trace: Trace | null; selected: number; select: (i: number) => void; playback: ReturnType<typeof usePlayback>; neuron: NeuronSelection | null; inspectNeuron: (layer: number, index: number) => void; stale: boolean}
export default function Network({trace, selected, select, playback, neuron, inspectNeuron, stale}: Props) {
  const [nodeLimit, setNodeLimit] = useState(16);
  const [zoom, setZoom] = useState(1);
  const [focus, setFocus] = useState<{key: string; view: FocusView} | null>(null);
  const [focusError, setFocusError] = useState('');
  const focusKey = trace && neuron ? `${trace.trace_id}:${neuron.layer}:${neuron.index}` : '';
  const focused = focus?.key === focusKey ? focus.view : null;
  useEffect(() => {
    setZoom(1); setFocus(null); setFocusError('');
    if (!trace || !neuron) return;
    const controller = new AbortController();
    api.focus(trace.trace_id, neuron.layer, neuron.index, controller.signal).then(view => {
      if (!controller.signal.aborted) setFocus({key: focusKey, view});
    }).catch(e => {if (!controller.signal.aborted) setFocusError(e instanceof Error ? e.message : 'Could not load connections.');});
    return () => controller.abort();
  }, [focusKey]);
  const {step, progress, playing} = playback;
  const rail = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const container = rail.current, current = container?.querySelector<HTMLElement>('.current');
    if (!container || !current) return;
    const parent = container.getBoundingClientRect(), child = current.getBoundingClientRect();
    if (child.right > parent.right - 10) container.scrollLeft += child.right - parent.right + 10;
    else if (child.left < parent.left + 10) container.scrollLeft += child.left - parent.left - 10;
  }, [step]);
  function jump(index: number) {playback.setPlaying(false); playback.seek(index); if (index >= 0) select(index);}
  function focusNeuron(layer: number, index: number) {
    playback.setPlaying(false);
    const column = trace?.visualization.columns.find(c => c.linear_layer === layer);
    playback.seek(column?.states.at(-1)?.step ?? layer);
    inspectNeuron(layer, index);
  }
  const diagramTrace = trace && focused ? {...trace, visualization: focused.diagram, predicted_class: null} : trace;
  return <section className="network-panel panel" aria-label="Network visualization">
    <div className="panel-heading"><h2>{neuron ? 'Incoming connections' : 'Network'}</h2>{!neuron && <label className="neuron-limit">Hidden neurons<select aria-label="Hidden neurons shown" value={nodeLimit} onChange={e => setNodeLimit(+e.target.value)}><option value={16}>16</option><option value={32}>32</option></select></label>}<div className="network-legend"><span><i className="positive"/> +wx</span><span><i className="negative"/> −wx</span><span><i className="brightness"/> |activation|</span></div></div>
    <div className="view-toolbar">{neuron ? <><button onClick={() => select(selected)}><ArrowLeft size={13}/> Whole network</button><span>{trace?.layers[neuron.layer].name} · neuron {neuron.index}</span></> : <span>Select a neuron to follow its incoming connections</span>}<div className="zoom-controls"><button aria-label="Zoom out" disabled={zoom <= 1} onClick={() => setZoom(z => Math.max(1, z - .5))}><Minus size={13}/></button><span>{Math.round(zoom * 100)}%</span><button aria-label="Zoom in" disabled={zoom >= 2} onClick={() => setZoom(z => Math.min(2, z + .5))}><Plus size={13}/></button></div></div>
    {trace?.input_maps && !neuron && <><div className="cnn-caption">Convolution & pooling · select a stage to inspect its feature maps</div><FeatureMaps trace={trace} step={step} progress={progress} select={jump}/><div className="cnn-caption">Flatten → dense layers · individual neuron activations</div></>}
    <div className="diagram-scroll" tabIndex={0} aria-label="Network canvas; scroll to explore" style={{height: 500}}>
      {focusError ? <div className="graph-empty" role="alert">{focusError}<button className="secondary-button" onClick={() => select(selected)}>Back to network</button></div> : neuron && !focused ? <div className="graph-empty" role="status">Loading incoming connections…</div> : diagramTrace ? <div style={{width: `${zoom * 100}%`, minWidth: Math.max(660, diagramTrace.visualization.columns.length * 210) * zoom}}><NeuronDiagram key={focusKey} zoom={zoom} nodeLimit={nodeLimit} trace={diagramTrace} step={step} progress={progress} selected={selected} neuron={neuron} select={select} inspectNeuron={focusNeuron}/></div> : <div className="graph-empty">Run a forward pass to display the network.</div>}
    </div>
    <div className="diagram-caption"><span>{focused ? `${focused.shown} strongest of ${focused.total} inputs · ${(focused.magnitude_coverage * 100).toFixed(1)}% of absolute incoming contribution · bias ${focused.bias.toFixed(3)} · full linear score ${focused.pre_activation.toFixed(3)}` : trace ? `Up to ${nodeLimit} hidden neurons per column · sampled connections · layer strips summarize all neurons` : 'Waiting for input'}</span>{stale && <span className="stale">Input changed. Run again.</span>}</div>
    <details className="diagram-reading"><summary>How to read this view</summary><p>Every displayed neuron is an actual model index. Values and connections come from this input’s recorded forward pass. Larger layers and their connections are sampled; omitted connections still participate in the calculation.</p><p>Select a neuron to see its 16 strongest incoming contributions, ranked across every input to that neuron. You can select a source neuron to continue backward. These are local weighted contributions, not an attribution of the whole prediction. Coverage measures absolute contribution magnitude, excluding bias.</p><p>Brightness shows absolute activation relative to the largest value in that operation. Layer strips average absolute activations in consecutive groups covering all neurons. The zero percentage covers the full layer. Cyan/red edges show positive/negative input × weight contributions, not just weight signs. Biases are included in the neuron inspector.</p><p>The animation replays layer order. Moving particles and fades illustrate computation; they are not measured signal travel or intermediate tensor values. Neurons in a dense layer are calculated together.</p>{trace?.input_maps && <p>CNN tiles show actual channel activations, with a shared scale within each operation. These tiles do not draw individual convolution receptive fields or their connections. The dense graph begins after Flatten.</p>}</details>
    <div className="execution-heading">Execution graph <span>{trace ? `${trace.layers.length} operations` : ''}</span></div>
    <div className="execution-rail" aria-label="Execution graph" ref={rail}>
      <button className={`execution-node ${step === -1 ? 'current' : ''}`} disabled={!trace} onClick={() => jump(-1)} aria-label="Show input step"><strong>Input</strong><small className="mono">{trace ? `[${trace.input_shape.join(', ')}]` : '—'}</small></button>
      {trace?.layers.map((layer, i) => <div className="execution-part" key={layer.id}><span aria-hidden="true">→</span><button className={`execution-node ${selected === i ? 'selected' : ''} ${step === i ? 'current' : ''}`} onClick={() => jump(i)} aria-label={`Inspect ${layer.name} ${layer.type}`} aria-pressed={selected === i}><strong>{layer.type} <span>{layer.name}</span></strong><small className="mono">[{layer.input_shape.join(', ')}] → [{layer.output_shape.join(', ')}]</small><i className="step-fill" style={{width: `${step > i ? 100 : step === i ? progress * 100 : 0}%`}}/></button></div>)}
    </div>
    <div className="playback"><div className="playback-buttons">
      <button title="Reset animation" aria-label="Reset animation" disabled={!trace} onClick={playback.reset}><RotateCcw size={15}/></button>
      <button title="Previous step" aria-label="Previous step" disabled={!trace || step < 0} onClick={playback.previous}><SkipBack size={15}/></button>
      <button className="play-button" aria-label={playing ? 'Pause animation' : 'Play animation'} disabled={!trace} onClick={playback.toggle}>{playing ? <Pause size={15}/> : <Play size={15}/>}</button>
      <button title="Next step" aria-label="Next step" disabled={!trace || step >= trace.layers.length - 1} onClick={playback.next}><SkipForward size={15}/></button>
      <select aria-label="Animation speed" value={playback.speed} onChange={e => playback.setSpeed(Number(e.target.value))}>{[0.5, 1, 2].map(n => <option key={n} value={n}>{n}×</option>)}</select>
    </div><input className="replay-scrubber" type="range" aria-label="Replay position" min={0} max={(trace?.layers.length ?? 0) + 1} step={.01} disabled={!trace} value={step + 1 + progress} onChange={e => {const position = replayPosition(Number(e.target.value), trace?.layers.length ?? 0); playback.setPlaying(false); playback.seek(position.step, position.progress);}}/><span className="mono playback-status">{trace ? `${step + 2} / ${trace.layers.length + 1} · ${step < 0 ? 'Input' : trace.layers[step].name + ' · ' + trace.layers[step].type}` : '—'}</span></div>
  </section>;
}
