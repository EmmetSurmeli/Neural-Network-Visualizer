import {useEffect, useRef, useState} from 'react';
import {ArrowRight, Braces, CircleHelp, Eraser, FileUp, LoaderCircle, Play} from 'lucide-react';
import {api} from './services/api';
import type {Model, Sample, Trace, NeuronSelection, PlaygroundHandoff} from './types';
import DigitCanvas from './components/DigitCanvas';
import Network from './components/Network';
import Inspector from './components/Inspector';
import Output from './components/Output';
import usePlayback from './hooks/usePlayback';
import ImportGuide from './components/ImportGuide';
import RequestNotice from './components/RequestNotice';
import useRequestState, {useOnline} from './hooks/useRequestState';
import {ApiError} from './lib/http';
import {track, modelProperties} from './lib/telemetry';
import bundled from './data/bundled.json';
const defaultPixels = bundled.samples.find(sample => sample.label === 7)!.pixels;

export default function Visualizer({handoff, active}: {handoff: PlaygroundHandoff | null; active: boolean}) {
  const [models, setModels] = useState<Model[]>(bundled.models), [samples, setSamples] = useState<Sample[]>(bundled.samples), [modelId, setModelId] = useState('mnist-mlp');
  const [source, setSource] = useState<number[]>(defaultPixels), [pixels, setPixels] = useState<number[]>(defaultPixels);
  const [sample, setSample] = useState<number | null>(7), [mode, setMode] = useState('draw'), [raw, setRaw] = useState('[1, 0.5, -1]');
  const [trace, setTrace] = useState<Trace | null>(null), [selected, setSelected] = useState(0);
  const [neuron, setNeuron] = useState<NeuronSelection | null>(null);
  const playback = usePlayback(trace?.layers.length ?? 0);
  const {setPlaying} = playback;
  const selectLayer = (index: number) => {setSelected(index); setNeuron(null);};
  const inspectNeuron = (layer: number, index: number) => {setSelected(layer); setNeuron({layer, index, key: Date.now()});};
  const [busy, setBusy] = useState(false), [loading, setLoading] = useState(true), [stale, setStale] = useState(false);
  const request = useRequestState(), online = useOnline();
  const retryAction = useRef<() => void>(() => {});
  const latestRun = useRef<() => Promise<void>>(async () => {});
  const [showImportGuide, setShowImportGuide] = useState(false);
  const file = useRef<HTMLInputElement>(null), model = models.find(m => m.id === modelId);
  const appliedHandoff = useRef<number | null>(null);
  const isImage = model?.input_kind === 'image' || modelId === 'mnist-mlp';
  async function initialize() {
    setLoading(true); request.start('Loading models and the default sample…'); retryAction.current = () => void initialize();
    try {
      const m = await api.models(); setModels(m);
      const s = await api.samples(); setSamples(s);
      const seven = s.find(v => v.label === 7) ?? s[0];
      setSource(seven.pixels); setPixels(seven.pixels); setSample(seven.label);
      request.success();
    } catch (error) {request.fail(error, !!trace);} finally {setLoading(false);}
  }
  useEffect(() => {void initialize();}, []);
  useEffect(() => {
    if (!active) setPlaying(false);
  }, [active, setPlaying]);
  useEffect(() => {
    if (!handoff || loading || appliedHandoff.current === handoff.key) return;
    appliedHandoff.current = handoff.key;
    retryAction.current = () => void runHandoff();
    async function runHandoff() {
      setBusy(true); setPlaying(false); request.start('Running the selected point…');
      try {
        const result = await api.run(handoff!.model.id, handoff!.point.input);
        setModels(ms => [...ms.filter(m => !m.id.startsWith('playground-')), handoff!.model]);
        switchModel(handoff!.model.id);
        setRaw(JSON.stringify(handoff!.point.input));
        setTrace(result); playback.seek(-1, 0); setPlaying(true); request.success();
      } catch (error) {request.fail(error, !!trace); setStale(!!trace);}
      finally {setBusy(false);}
    }
    void runHandoff();
  }, [handoff, loading]);
  function changeInput(p: number[], label: number | null) {setPixels(p); setSample(label); setStale(!!trace); setPlaying(false);}
  function chooseSample(s: Sample) {setSource(s.pixels); changeInput(s.pixels, s.label); setMode('draw');}
  async function run() {
    request.start('Running forward pass…'); retryAction.current = () => void latestRun.current(); setPlaying(false);
    try {
      const input = mode === 'raw' || !isImage ? JSON.parse(raw) : pixels;
      if (!Array.isArray(input) || input.length !== model?.input_features || !input.every(v => typeof v === 'number' && Number.isFinite(v))) throw new ApiError('INVALID_INPUT', `Enter a JSON array of exactly ${model?.input_features ?? 784} finite numbers.`);
      if (isImage && input.every(v => v === 0)) throw new ApiError('INVALID_INPUT', 'Draw an image or choose a sample before running.');
      setBusy(true); const result = await api.run(modelId, input); setTrace(result); setSelected(0); setNeuron(null); playback.seek(-1, 0); setPlaying(true); setStale(false); request.success();
    } catch (e) {request.fail(e, !!trace); setStale(!!trace);} finally {setBusy(false);}
  }
  latestRun.current = run;
  async function upload(f?: File) {
    if (!f) return;
    track('model_import_started', {model_category: 'imported'});
    request.start('Importing model…'); setBusy(true); retryAction.current = () => void upload(f);
    try {
      if (f.size > 8 * 1024 * 1024) throw new ApiError('INVALID_INPUT', 'Model JSON must be smaller than 8 MB.');
      const next = await api.upload(JSON.parse(await f.text())); setModels(ms => [...ms, next]); switchModel(next.id); setRaw(JSON.stringify(Array(next.input_features).fill(0))); request.success(); track('model_loaded', modelProperties(next.id));
    } catch (e) {track('model_import_failed', {model_category: 'imported', outcome: 'failure'}); request.fail(e, !!trace); setStale(!!trace);} finally {setBusy(false); if (file.current) file.current.value = '';}
  }
  function switchModel(id: string) {setModelId(id); setTrace(null); setPlaying(false); setSelected(0); setNeuron(null); playback.seek(-1, 0); setStale(false); setMode(id === 'mnist-mlp' || models.find(m => m.id === id)?.input_kind === 'image' ? 'draw' : 'raw'); request.reset();}
  async function loadModel(id: string) {
    const next = models.find(m => m.id === id);
    if (!next) return;
    setBusy(true); request.start('Loading model and sample…'); retryAction.current = () => void loadModel(id);
    try {
      let options: Sample[] = [];
      let input = next.default_input ?? Array(next.input_features).fill(0);
      if (next.input_kind === 'image') {
        options = await api.samples(id);
        const first = id === 'mnist-mlp' ? (options.find(s => s.label === 7) ?? options[0]) : options[0];
        input = first.pixels;
      }
      switchModel(id); setSamples(options); setRaw(JSON.stringify(input));
      if (options.length) {setSource(input); setPixels(input); setSample(options.find(s => s.pixels === input)?.label ?? 0);}
      request.success(true); track('model_loaded', modelProperties(id));
    } catch(e) {request.fail(e, !!trace);}
    finally {setBusy(false);}
  }
  return <main>
      {handoff && modelId === handoff.model.id && raw === JSON.stringify(handoff.point.input) && <div className="handoff-context"><a href="#/playground">← Back to Playground</a><span>{handoff.dataset} · [{handoff.point.input.map(v => v.toFixed(3)).join(', ')}] · expected class {handoff.point.label}</span><span>Final trained model</span></div>}
      <div className="model-toolbar"><div className="model-select"><label>Model<select aria-label="Active model" value={modelId} disabled={busy || loading || !online} onChange={e => void loadModel(e.target.value)}>{models.length ? models.map(m => <option key={m.id} value={m.id}>{m.name}</option>) : <option>Connecting to backend…</option>}</select></label></div><div className="model-badges"><span className="mono">{model?.input_features ?? 784} inputs</span><span className="mono">{model?.parameters.toLocaleString() ?? '—'} params</span></div><div className="import-actions"><button className="text-button format-button" onClick={() => setShowImportGuide(true)}><CircleHelp size={14}/> Format</button><input type="file" accept=".json,application/json" ref={file} hidden onChange={e => void upload(e.target.files?.[0])}/><button className="secondary-button import-button" disabled={busy || loading || !online} onClick={() => file.current?.click()}><FileUp size={15}/> Import model</button></div></div>
      <RequestNotice state={request.state} retry={() => retryAction.current()}/>
      {stale && request.state.status !== 'stale' && <RequestNotice state={{status: 'stale', message: 'Previous result · run the current input to update the visualization.'}}/>}
      {request.state.error && <button className="text-button recovery-link" disabled={busy || loading || !online} onClick={() => void loadModel('tiny-mlp')}>Load the 3-input demo</button>}
      <div className="workspace-grid"><aside className="input-panel panel"><div className="panel-heading"><h2>Input</h2><span className="tag">{isImage ? '28 × 28' : 'FLOAT32'}</span></div>
        <div className="input-content"><div className="tabs input-tabs"><button disabled={!isImage || busy} onClick={() => {setMode('draw'); setStale(!!trace);}} className={mode === 'draw' ? 'active' : ''}>Image canvas</button><button disabled={busy} onClick={() => {if (isImage) setRaw(JSON.stringify(pixels)); setMode('raw'); setStale(!!trace);}} className={mode === 'raw' ? 'active' : ''}><Braces size={12}/> Numeric input</button></div>
          {mode === 'draw' ? <><div className={`drawing-wrap ${busy ? 'disabled' : ''}`}><DigitCanvas label={isImage ? `Draw an image for ${model?.name ?? 'the selected model'}. Or choose a sample.` : undefined} source={source} onChange={p => changeInput(p, null)}/><span className="canvas-coordinate">0,0</span><span className="canvas-label">{sample === null ? 'Drawing' : model?.classes?.[sample] ?? `Sample ${sample}`}</span></div><div className="canvas-caption"><span>{modelId === 'mnist-mlp' ? 'Draw a digit, 0–9' : modelId === 'shapes-cnn' ? 'Draw an outline shape' : 'Draw one straight line'}</span><button disabled={busy} className="text-button" onClick={() => {const p = Array(784).fill(0); setSource(p); changeInput(p, null);}}><Eraser size={13}/> Clear</button></div><div className="section-label sample-label">Samples <span>{model?.dataset}</span></div><div className={`sample-buttons ${modelId.includes("cnn") ? "named-samples" : ""}`}>{samples.map(s => <button disabled={busy} aria-label={`Load sample ${model?.classes?.[s.label] ?? s.label}`} aria-pressed={sample === s.label} className={sample === s.label ? 'active' : ''} key={s.label} onClick={() => chooseSample(s)}>{model?.classes?.[s.label] ?? s.label}</button>)}</div></> : <><label className="raw-label" htmlFor="numeric">JSON array · {model?.input_features} values</label><textarea id="numeric" className="numeric-input mono" spellCheck={false} value={raw} disabled={busy} onChange={e => {setRaw(e.target.value); setStale(!!trace); setPlaying(false);}}/><p className="tiny muted">{isImage ? '784 pixels, row by row. Black = 0; white = 1.' : `Enter exactly ${model?.input_features ?? 0} finite numbers in one flat array.`}</p></>}
          <button className="run-button" disabled={busy || loading || !model || !online} onClick={() => void run()}>{busy ? <LoaderCircle size={16} className="spin"/> : <Play size={15} fill="currentColor"/>}{busy ? 'Running…' : 'Run forward pass'}<ArrowRight size={16}/></button>
          {model?.description && <p className="model-description">{model.description} {modelId.includes("cnn") && "Accuracy is measured on generated images; hand drawings may differ."}</p>}
          <div className="model-info"><dl><div><dt>Layers</dt><dd className="mono">{model?.architecture ?? (modelId === 'mnist-mlp' ? '784 → 128 → 64 → 10' : `${model?.input_features} inputs`)}</dd></div>{model?.test_accuracy && <div><dt>Test accuracy</dt><dd className="mono">{(model.test_accuracy * 100).toFixed(2)}%</dd></div>}</dl></div>
          <button className="tiny-demo" disabled={busy || loading || !online} onClick={() => void loadModel('tiny-mlp')}>3-input example <span>↗</span></button>
        </div>
      </aside><Network trace={trace} selected={selected} select={selectLayer} playback={playback} neuron={neuron} inspectNeuron={inspectNeuron} stale={stale}/></div>
      <div className="detail-grid"><Inspector trace={trace} selected={selected} focusedNeuron={neuron}/><Output trace={trace} stale={stale}/></div>
      <ImportGuide open={showImportGuide} close={() => setShowImportGuide(false)}/>
    </main>;
}
