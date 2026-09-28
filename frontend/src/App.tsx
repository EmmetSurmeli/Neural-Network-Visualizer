import {useEffect, useRef, useState} from 'react';
import {ArrowRight, Braces, Eraser, FileUp, LoaderCircle, Play, RefreshCw} from 'lucide-react';
import {api} from './services/api';
import type {Model, Sample, Trace, NeuronSelection} from './types';
import DigitCanvas from './components/DigitCanvas';
import Network from './components/Network';
import Inspector from './components/Inspector';
import Output from './components/Output';
import usePlayback from './hooks/usePlayback';

export default function App() {
  const [models, setModels] = useState<Model[]>([]), [samples, setSamples] = useState<Sample[]>([]), [modelId, setModelId] = useState('mnist-mlp');
  const [source, setSource] = useState<number[]>(Array(784).fill(0)), [pixels, setPixels] = useState<number[]>(Array(784).fill(0));
  const [sample, setSample] = useState<number | null>(7), [mode, setMode] = useState('draw'), [raw, setRaw] = useState('[1, 0.5, -1]');
  const [trace, setTrace] = useState<Trace | null>(null), [selected, setSelected] = useState(0);
  const [neuron, setNeuron] = useState<NeuronSelection | null>(null);
  const playback = usePlayback(trace?.layers.length ?? 0);
  const {setPlaying} = playback;
  const selectLayer = (index: number) => {setSelected(index); setNeuron(null);};
  const inspectNeuron = (layer: number, index: number) => {setSelected(layer); setNeuron({layer, index, key: Date.now()});};
  const [busy, setBusy] = useState(false), [loading, setLoading] = useState(true), [error, setError] = useState(''), [stale, setStale] = useState(false);
  const file = useRef<HTMLInputElement>(null), model = models.find(m => m.id === modelId);
  const tinyModelId = useRef<string | null>(null);
  async function initialize() {
    setLoading(true); setError('');
    try {const [m, s] = await Promise.all([api.models(), api.samples()]); setModels(m); setSamples(s); const seven = s.find(v => v.label === 7)!; setSource(seven.pixels); setPixels(seven.pixels); setSample(7); const result = await api.run('mnist-mlp', seven.pixels); setTrace(result); playback.seek(result.layers.length - 1);} catch {setError('Cannot reach the Python backend. Start the app with ./start.sh, then retry.');} finally {setLoading(false);}
  }
  useEffect(() => {void initialize();}, []);
  function changeInput(p: number[], label: number | null) {setPixels(p); setSample(label); setStale(!!trace); setPlaying(false);}
  function chooseSample(s: Sample) {setSource(s.pixels); changeInput(s.pixels, s.label); setMode('draw');}
  async function run() {
    setError(''); setPlaying(false);
    try {
      const input = mode === 'raw' || modelId !== 'mnist-mlp' ? JSON.parse(raw) : pixels;
      if (!Array.isArray(input) || input.length !== model?.input_features || !input.every(v => typeof v === 'number' && Number.isFinite(v))) throw new Error(`Enter a JSON array of exactly ${model?.input_features ?? 784} finite numbers.`);
      if (modelId === 'mnist-mlp' && input.every(v => v === 0)) throw new Error('Draw a digit or choose a sample before running.');
      setBusy(true); const result = await api.run(modelId, input); setTrace(result); setSelected(0); setNeuron(null); playback.seek(-1, 0); setPlaying(true); setStale(false);
    } catch (e) {setError(e instanceof Error ? e.message : 'Forward pass failed.');} finally {setBusy(false);}
  }
  async function upload(f?: File) {
    if (!f) return;
    setError(''); setBusy(true);
    try {
      if (f.size > 8 * 1024 * 1024) throw new Error('Model JSON must be smaller than 8 MB.');
      const next = await api.upload(JSON.parse(await f.text())); setModels(ms => [...ms, next]); switchModel(next.id); setRaw(JSON.stringify(Array(next.input_features).fill(0)));
    } catch (e) {setError(e instanceof Error ? e.message : 'Import failed. Use a supported JSON model.');} finally {setBusy(false); if (file.current) file.current.value = '';}
  }
  function switchModel(id: string) {setModelId(id); setTrace(null); setPlaying(false); setSelected(0); setNeuron(null); playback.seek(-1, 0); setStale(false); setMode(id === 'mnist-mlp' ? 'draw' : 'raw'); setError('');}
  async function tinyDemo() {
    if (tinyModelId.current) {switchModel(tinyModelId.current); setRaw('[1, 0.5, -1]'); return;}
    setBusy(true); setError('');
    try {const spec = await fetch('/tiny-model.json').then(r => r.json()); const m = await api.upload(spec); tinyModelId.current = m.id; setModels(ms => [...ms, m]); switchModel(m.id); setRaw('[1, 0.5, -1]');} catch (e) {setError(e instanceof Error ? e.message : 'Failed to load model.');} finally {setBusy(false);}
  }
  return <div className="app-shell">
    <header className="topbar"><a className="brand" href="/" aria-label="NeuralScope home">NeuralScope</a><span className="header-title">Neural network explorer</span><span className="engine">{loading ? 'Loading model…' : models.length ? 'PyTorch / CPU' : 'Backend offline'}</span></header>
    <main>
      <div className="model-toolbar"><div className="model-select"><label>Model<select aria-label="Active model" value={modelId} disabled={busy || loading} onChange={e => switchModel(e.target.value)}>{models.length ? models.map(m => <option key={m.id} value={m.id}>{m.name}</option>) : <option>Connecting to backend…</option>}</select></label></div><div className="model-badges"><span className="mono">{model?.input_features ?? 784} inputs</span><span className="mono">{model?.parameters.toLocaleString() ?? '—'} params</span></div><input type="file" accept=".json,application/json" ref={file} hidden onChange={e => void upload(e.target.files?.[0])}/><button className="secondary-button import-button" disabled={busy || loading} onClick={() => file.current?.click()}><FileUp size={15}/> Import model</button></div>
      {error && <div className="error-banner" role="alert"><span>{error}</span>{!models.length && <button onClick={() => void initialize()}><RefreshCw size={14}/> Retry connection</button>}<button aria-label="Dismiss error" onClick={() => setError('')}>×</button></div>}
      <div className="workspace-grid"><aside className="input-panel panel"><div className="panel-heading"><h2>Input</h2><span className="tag">{modelId === 'mnist-mlp' ? '28 × 28' : 'FLOAT32'}</span></div>
        <div className="input-content"><div className="tabs input-tabs"><button disabled={modelId !== 'mnist-mlp' || busy} onClick={() => {setMode('draw'); setStale(!!trace);}} className={mode === 'draw' ? 'active' : ''}>Digit canvas</button><button disabled={busy} onClick={() => {if (modelId === 'mnist-mlp') setRaw(JSON.stringify(pixels)); setMode('raw'); setStale(!!trace);}} className={mode === 'raw' ? 'active' : ''}><Braces size={12}/> Numeric input</button></div>
          {mode === 'draw' ? <><div className={`drawing-wrap ${busy ? 'disabled' : ''}`}><DigitCanvas source={source} onChange={p => changeInput(p, null)}/><span className="canvas-coordinate">0,0</span><span className="canvas-label">{sample === null ? 'Drawing' : `Sample ${sample}`}</span></div><div className="canvas-caption"><span>Draw a single digit, 0–9</span><button disabled={busy} className="text-button" onClick={() => {const p = Array(784).fill(0); setSource(p); changeInput(p, null);}}><Eraser size={13}/> Clear</button></div><div className="section-label sample-label">Samples <span>MNIST</span></div><div className="sample-buttons">{samples.map(s => <button disabled={busy} aria-label={`Load sample digit ${s.label}`} aria-pressed={sample === s.label} className={sample === s.label ? 'active' : ''} key={s.label} onClick={() => chooseSample(s)}>{s.label}</button>)}</div></> : <><label className="raw-label" htmlFor="numeric">JSON array · {model?.input_features} values</label><textarea id="numeric" className="numeric-input mono" spellCheck={false} value={raw} disabled={busy} onChange={e => {setRaw(e.target.value); setStale(!!trace); setPlaying(false);}}/><p className="tiny muted">{modelId === 'mnist-mlp' ? '784 pixels, row by row. Black = 0; white = 1.' : 'Use a flat array, for example [1, 0.5, -1] for the three-input model.'}</p></>}
          <button className="run-button" disabled={busy || loading || !model} onClick={() => void run()}>{busy ? <LoaderCircle size={16} className="spin"/> : <Play size={15} fill="currentColor"/>}{busy ? 'Running…' : 'Run forward pass'}<ArrowRight size={16}/></button>
          <div className="model-info"><dl><div><dt>Layers</dt><dd className="mono">{modelId === 'mnist-mlp' ? '784 → 128 → 64 → 10' : `${model?.input_features} inputs`}</dd></div>{model?.test_accuracy && <div><dt>Test accuracy</dt><dd className="mono">{(model.test_accuracy * 100).toFixed(2)}%</dd></div>}</dl></div>
          <button className="tiny-demo" disabled={busy || loading} onClick={() => void tinyDemo()}>3-input example <span>↗</span></button>
        </div>
      </aside><Network trace={trace} selected={selected} select={selectLayer} playback={playback} neuron={neuron} inspectNeuron={inspectNeuron} stale={stale}/></div>
      <div className="detail-grid"><Inspector trace={trace} selected={selected} focusedNeuron={neuron}/><Output trace={trace} stale={stale}/></div>
    </main>
  </div>;
}
