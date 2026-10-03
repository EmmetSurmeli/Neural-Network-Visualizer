import {useEffect, useRef, useState} from 'react';
import {Pause, Play, RotateCcw, SkipForward, ArrowRight, LoaderCircle} from 'lucide-react';
import {api} from '../services/api';
import RequestNotice from './RequestNotice';
import useRequestState, {useOnline} from '../hooks/useRequestState';
import {track} from '../lib/telemetry';
import type {DatasetName, PlaygroundConfig, PlaygroundDataset, PlaygroundRun, PlaygroundHandoff} from '../types';

const names: Record<DatasetName, string> = {xor: 'XOR', moons: 'Two moons', circles: 'Circles', spiral: 'Spiral'};
const initial: PlaygroundConfig = {dataset: 'moons', hidden_layers: [8, 8], activation: 'Tanh', learning_rate: .01, epochs: 300, seed: 42, noise: .1};
const descriptions: Record<DatasetName, string> = {
  moons: 'Two curved groups. Can the network follow the curve when the classes overlap?',
  xor: 'Opposite corners share a class. A single straight dividing line cannot separate them.',
  circles: 'One class surrounds the other. The network needs to learn a closed boundary.',
  spiral: 'Two intertwined arms. Watch where a small network misses the tighter turns.',
};
const percent = (n: number) => `${(n * 100).toFixed(1)}%`;

export default function Playground({active, inspect}: {active: boolean; inspect: (handoff: PlaygroundHandoff) => void}) {
  const [config, setConfig] = useState<PlaygroundConfig>(initial);
  const [preview, setPreview] = useState<PlaygroundDataset | null>(null);
  const [run, setRun] = useState<PlaygroundRun | null>(null);
  const [busy, setBusy] = useState(false), [loading, setLoading] = useState(true);
  const [index, setIndex] = useState(0), [playing, setPlaying] = useState(false), [selected, setSelected] = useState(0);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [showMistakes, setShowMistakes] = useState(false);
  const [previous, setPrevious] = useState<PlaygroundRun | null>(null);
  const request = useRequestState(), online = useOnline();
  const retryAction = useRef<() => void>(() => {});
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); request.start('Loading dataset…'); retryAction.current = () => setRetry(v => v + 1);
    api.dataset(config.dataset, config.seed, config.noise, controller.signal).then(data => {if (!controller.signal.aborted) {setPreview(data); request.success();}})
      .catch(e => {if (!controller.signal.aborted) request.fail(e, !!run || !!preview);})
      .finally(() => {if (!controller.signal.aborted) setLoading(false);});
    return () => controller.abort();
  }, [config.dataset, config.seed, config.noise, retry]);
  useEffect(() => {
    if (!active) setPlaying(false);
  }, [active]);
  useEffect(() => {
    if (!playing || !run || !active) return;
    if (index >= run.snapshots.length - 1) {setPlaying(false); return;}
    const timer = window.setTimeout(() => setIndex(i => i + 1), 110);
    return () => clearTimeout(timer);
  }, [playing, index, run, active]);
  const data = run ?? preview;
  const snapshot = run?.snapshots[index];
  const point = data?.points[selected];
  const probability = snapshot?.point_probabilities[selected];
  const predicted = probability === undefined ? null : probability >= .5 ? 1 : 0;
  useEffect(() => {
    const ctx = canvas.current?.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, 48, 48);
    if (!snapshot) return;
    const image = ctx.createImageData(48, 48);
    snapshot.grid_probabilities.forEach((p, i) => {
      const x = i % 48, y = 47 - Math.floor(i / 48), offset = (y * 48 + x) * 4;
      const a = [47, 104, 117], b = [125, 55, 62];
      for (let c = 0; c < 3; c++) image.data[offset + c] = a[c] * (1 - p) + b[c] * p;
      image.data[offset + 3] = 210;
    });
    ctx.putImageData(image, 0, 0);
  }, [snapshot]);
  function change(next: Partial<PlaygroundConfig>) {
    if (run) setPrevious(run);
    setConfig(c => ({...c, ...next})); setRun(null); setIndex(0); setPlaying(false); request.reset();
    if (next.dataset) setSelected(0);
  }
  async function train() {
    setBusy(true); setPlaying(false); request.start('Training network…'); retryAction.current = () => void train();
    try {if (run) setPrevious(run); const result = await api.train(config); setRun(result); setIndex(0); setPlaying(true); request.success();}
    catch (e) {request.fail(e, !!run);}
    finally {setBusy(false);}
  }
  const coord = (v: number) => data ? 28 + (v - data.grid.min) / (data.grid.max - data.grid.min) * 544 : 0;
  const lossMax = Math.max(...(run?.snapshots.flatMap(s => [s.loss, s.validation_loss]) ?? [1]), .001);
  const curve = (key: 'loss' | 'validation_loss') => run?.snapshots.map((s, i) => `${i === 0 ? 'M' : 'L'}${8 + i / (run.snapshots.length - 1) * 584},${74 - s[key] / lossMax * 64}`).join(' ');
  const mistakes = data?.points.filter(p => snapshot && (snapshot.point_probabilities[p.id] >= .5 ? 1 : 0) !== p.label) ?? [];
  function experiment(kind: 'clean' | 'overlap' | 'small') {
    change({...initial, noise: kind === 'overlap' ? .5 : .05, hidden_layers: kind === 'small' ? [2] : [8, 8]});
  }
  return <main className="playground">
    <div className="page-intro"><div><h1>When does a network get it wrong?</h1><p>Change the data or the network. Train it, check the mistakes, then inspect a prediction.</p></div><span className="tiny muted">2 inputs · 2 classes</span></div>
    <RequestNotice state={request.state} retry={() => retryAction.current()}/>
    <div className="experiment-strip"><span>Start an experiment</span><button disabled={busy} onClick={() => experiment('clean')}>1. Separate the classes<small>Low noise · 8 + 8 neurons</small></button><button disabled={busy} onClick={() => experiment('overlap')}>2. Make them overlap<small>High noise · same network</small></button><button disabled={busy} onClick={() => experiment('small')}>3. Limit the network<small>Low noise · just 2 neurons</small></button></div>
    <div className="playground-grid">
      <aside className="panel training-controls"><div className="panel-heading"><h2>Build & train</h2></div><fieldset disabled={busy}>
        <button className="run-button" disabled={busy || loading || !data || !online} onClick={() => void train()}>{busy ? <LoaderCircle size={15} className="spin"/> : <Play size={15}/>} {busy ? 'Training…' : run ? 'Train again' : 'Train network'}</button>
        <label>Dataset<select value={config.dataset} onChange={e => change({dataset: e.target.value as DatasetName})}>{Object.entries(names).map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></label>
        <div className="noise-control"><label htmlFor="noise">Data noise <output>{Math.round(config.noise * 100)}%</output></label><input id="noise" type="range" min="0" max="0.8" step="0.05" value={config.noise} onChange={e => change({noise: +e.target.value})}/><p>Moves points away from the pattern. More noise makes the classes overlap.</p></div>
        <button className="text-button resample" onClick={() => change({seed: (config.seed + 1) % 2147483648})}>New sample ↻</button>
        <div className="control-divider"/>
        <label>Hidden layers<select value={config.hidden_layers.length} onChange={e => change({hidden_layers: Array.from({length: +e.target.value}, (_, i) => config.hidden_layers[i] ?? 8)})}>{[1, 2, 3].map(n => <option key={n}>{n}</option>)}</select></label>
        {config.hidden_layers.map((width, i) => <label key={i}>Layer {i + 1} neurons<select value={width} onChange={e => change({hidden_layers: config.hidden_layers.map((w, j) => j === i ? +e.target.value : w)})}>{[2, 4, 8, 16].map(n => <option key={n}>{n}</option>)}</select></label>)}
        <label>Activation<select value={config.activation} onChange={e => change({activation: e.target.value as PlaygroundConfig['activation']})}>{['ReLU', 'Tanh', 'Sigmoid'].map(a => <option key={a}>{a}</option>)}</select></label>
        <div className="control-divider"/>
        <label>Learning rate<select value={config.learning_rate} onChange={e => change({learning_rate: +e.target.value})}>{[.001, .01, .05].map(n => <option key={n}>{n}</option>)}</select></label>
        <label>Epochs<select value={config.epochs} onChange={e => change({epochs: +e.target.value})}>{[100, 300, 1000].map(n => <option key={n}>{n}</option>)}</select></label>
        <div className="architecture-mini mono" aria-label="Network architecture">{[2, ...config.hidden_layers, 2].join(' → ')}</div>

        <p className="tiny muted training-note" role="status">{busy ? 'Computing the full training run. Replay will start when ready.' : '200 points · 160 training / 40 validation'}</p>
      </fieldset></aside>
      <section className="panel boundary-panel"><div className="panel-heading"><h2>{names[config.dataset]}</h2><div className="class-legend"><span><i/>Class 0</span><span><i/>Class 1</span></div></div>
        <p className="dataset-explanation">{descriptions[config.dataset]}</p>
        <div className="boundary-wrap" aria-busy={busy || loading}>
          <canvas ref={canvas} width={48} height={48} aria-hidden="true"/>
          <svg viewBox="0 0 600 600" className="boundary-plot" role="group" aria-label="Dataset points and decision boundary">
            <rect x="28" y="28" width="544" height="544" fill="none" stroke="#444"/>
            <path d="M300 28V572M28 300H572" stroke="#888" strokeOpacity=".22"/>
            <text x="582" y="304" className="plot-label">x</text><text x="296" y="18" className="plot-label">y</text>
            {data && <><text x="28" y="591" className="plot-label">{data.grid.min.toFixed(1)}</text><text x="555" y="591" className="plot-label">{data.grid.max.toFixed(1)}</text></>}
            {data?.points.map(p => <circle key={p.id} role="button" tabIndex={selected === p.id ? 0 : -1} aria-label={`Point ${p.id + 1}, class ${p.label}, ${p.split}`} aria-pressed={selected === p.id} cx={coord(p.input[0])} cy={600 - coord(p.input[1])} r={selected === p.id ? 6.5 : 4} fill={p.label === 0 ? '#7fc5d4' : '#e18c8e'} stroke={selected === p.id ? '#fff' : p.split === 'validation' ? '#eee' : '#101010'} strokeWidth={selected === p.id ? 2 : 1} opacity={showMistakes && snapshot && (snapshot.point_probabilities[p.id] >= .5 ? 1 : 0) === p.label ? .15 : 1} className="data-point" onClick={() => setSelected(p.id)} onKeyDown={e => {if (e.key === 'Enter' || e.key === ' ') {e.preventDefault(); setSelected(p.id);} if (['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp'].includes(e.key)) {e.preventDefault(); const n = (p.id + (['ArrowRight', 'ArrowDown'].includes(e.key) ? 1 : 199)) % 200; setSelected(n); const parent = e.currentTarget.parentElement; (parent?.querySelectorAll('.data-point')[n] as SVGElement)?.focus();}}}/>)}
          </svg>
          {(loading || busy) && <div className="plot-status" role="status">{busy ? 'Training network…' : 'Loading dataset…'}</div>}
        </div>
        <div className="mistake-tools"><label><input type="checkbox" checked={showMistakes} disabled={!run} onChange={e => setShowMistakes(e.target.checked)}/> Highlight mistakes{snapshot ? ` (${mistakes.length} / 200)` : ''}</label><button className="text-button" disabled={!mistakes.length} onClick={() => {setPlaying(false); const next = mistakes.find(p => p.id > selected) ?? mistakes[0]; setSelected(next.id);}}>Next mistake →</button></div>
        <div className="plot-caption"><span>Click a point to inspect it · white outlines mark validation points</span><span>{run ? 'Color shows predicted class probability' : 'Train to reveal the decision boundary'}</span></div>
      </section>
      <aside className="training-results">
        <section className="panel"><div className="panel-heading"><h2>Training</h2><span className="tag">{run ? 'RECORDED REPLAY' : 'NOT TRAINED'}</span></div>
          <dl className="training-metrics"><div><dt>Epoch</dt><dd>{snapshot?.epoch ?? '—'}<small> / {config.epochs}</small></dd></div><div><dt>Loss</dt><dd>{snapshot?.loss.toFixed(4) ?? '—'}</dd></div><div><dt>Training accuracy</dt><dd>{snapshot ? percent(snapshot.train_accuracy) : '—'}</dd></div><div><dt>Validation accuracy</dt><dd>{snapshot ? percent(snapshot.validation_accuracy) : '—'}</dd></div></dl>
          <div className="loss-chart"><span className="tiny muted">Loss · <b className="train-key">training</b> / <b className="validation-key">validation</b></span><svg viewBox="0 0 600 90" role="img" aria-label="Training loss curve"><path d="M8 74H592" stroke="#333"/>{run && <><path d={curve('loss')} stroke="#65b4c6" strokeWidth="2" fill="none"/><path d={curve('validation_loss')} stroke="#ca7476" strokeWidth="2" fill="none"/></>}{run && <line x1={8 + index / (run.snapshots.length - 1) * 584} x2={8 + index / (run.snapshots.length - 1) * 584} y1="7" y2="76" stroke="#ddd"/>}</svg></div>
          <div className="training-playback"><div className="playback-buttons"><button aria-label="Replay training" disabled={!run || busy} onClick={() => {setIndex(0); setPlaying(true); request.success();}}><RotateCcw size={15}/></button><button aria-label={playing ? 'Pause training replay' : 'Play training replay'} disabled={!run || busy} onClick={() => {if (run && index === run.snapshots.length - 1) setIndex(0); setPlaying(v => !v);}}>{playing ? <Pause size={15}/> : <Play size={15}/>}</button><button aria-label="Jump to final epoch" disabled={!run || busy} onClick={() => {setPlaying(false); setIndex(run!.snapshots.length - 1);}}><SkipForward size={15}/></button></div><input type="range" aria-label="Training epoch" min={0} max={(run?.snapshots.length ?? 2) - 1} value={index} disabled={!run || busy} onChange={e => {setPlaying(false); setIndex(+e.target.value);}} aria-valuetext={`Epoch ${snapshot?.epoch ?? 0}`}/></div>
          <p className="replay-note tiny muted">Replay shows a snapshot every 5 epochs.</p>
          <div className="results-explanation"><p><strong>Training</strong> measures the 160 points used to adjust the weights. <strong>Validation</strong> measures 40 points held out from training.</p><p>{snapshot ? snapshot.train_accuracy - snapshot.validation_accuracy > .1 ? 'The network does better on familiar points. Try less capacity or fewer epochs and check whether validation improves.' : snapshot.validation_accuracy < .8 ? 'The model is still missing held-out points. Check the highlighted mistakes: overlapping data or limited capacity may be contributing.' : config.noise >= .3 ? 'Inspect the remaining mistakes. Where the classes overlap, even a confident prediction can be wrong.' : 'The network separates most held-out points. Add noise and train again to see where this stops working.' : 'Train a network, then watch both curves. If training loss falls while validation loss rises, it may be fitting details that do not generalize.'}</p></div>
          {previous && <div className="previous-run"><span>Previous completed run</span><p>{names[previous.config.dataset]} · {Math.round(previous.config.noise * 100)}% noise · {previous.config.hidden_layers.join(' + ')} neurons</p><strong>{percent(previous.metrics.validation_accuracy)} validation</strong><p>{previous.config.dataset === config.dataset && previous.config.seed === config.seed && previous.config.noise === config.noise ? 'Same data and split as this experiment.' : 'Data differs; these scores use different examples.'}</p></div>}
        </section>
        <section className="panel selected-point"><div className="panel-heading"><h2>Selected point</h2><span className="tag">{point?.split.toUpperCase()}</span></div><div className="point-content"><label>Point<select aria-label="Selected point" value={selected} disabled={!data} onChange={e => setSelected(+e.target.value)}>{data?.points.map(p => <option key={p.id} value={p.id}>{p.id + 1} · class {p.label}</option>)}</select></label><p className="point-coordinates mono">{point ? `[${point.input.map(v => v.toFixed(3)).join(', ')}]` : '—'}</p><dl><div><dt>True class</dt><dd>{point?.label ?? '—'}</dd></div><div><dt>Predicted class</dt><dd>{predicted ?? '—'}</dd></div><div><dt>Probability of prediction</dt><dd>{probability === undefined ? '—' : percent(Math.max(probability, 1 - probability))}</dd></div></dl><button className="run-button" disabled={!run || !point || busy} onClick={() => {if (run && point) {setPlaying(false); inspect({key: Date.now(), model: run.model, dataset: names[config.dataset], point});}}}>Inspect in Visualizer<ArrowRight size={15}/></button><p className="tiny muted">Follow this point through the final trained network to see how it produced its answer.</p></div></section>
      </aside>
    </div>
  </main>;
}
