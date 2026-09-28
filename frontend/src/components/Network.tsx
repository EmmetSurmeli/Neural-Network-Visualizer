import {Pause, Play, RotateCcw, SkipBack, SkipForward} from 'lucide-react';
import {useEffect, useRef} from 'react';
import type {NeuronSelection, Trace} from '../types';
import type usePlayback from '../hooks/usePlayback';
import NeuronDiagram from './NeuronDiagram';

interface Props {trace: Trace | null; selected: number; select: (i: number) => void; playback: ReturnType<typeof usePlayback>; neuron: NeuronSelection | null; inspectNeuron: (layer: number, index: number) => void; stale: boolean}
export default function Network({trace, selected, select, playback, neuron, inspectNeuron, stale}: Props) {
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
  return <section className="network-panel panel" aria-label="Network visualization">
    <div className="panel-heading"><h2>Network</h2><div className="network-legend"><span><i className="positive"/> +wx</span><span><i className="negative"/> −wx</span><span><i className="brightness"/> |activation|</span></div></div>
    <div className="diagram-scroll">
      {trace ? <NeuronDiagram trace={trace} step={step} progress={progress} selected={selected} neuron={neuron} select={select} inspectNeuron={inspectNeuron}/> : <div className="graph-empty">Run a forward pass to display the network.</div>}
    </div>
    <div className="diagram-caption"><span>{trace ? `Up to ${trace.visualization.node_limit} neurons per column · strongest connections between shown neurons` : 'Waiting for input'}</span>{stale && <span className="stale">Input changed. Run again.</span>}</div>
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
    </div><span className="mono playback-status">{trace ? `${step + 2} / ${trace.layers.length + 1} · ${step < 0 ? 'Input' : trace.layers[step].name + ' · ' + trace.layers[step].type}` : '—'}</span></div>
  </section>;
}
