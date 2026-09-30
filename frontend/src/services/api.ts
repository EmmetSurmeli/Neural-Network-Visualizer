import type {Model, Neuron, Sample, Trace, FocusView, DatasetName, PlaygroundConfig, PlaygroundDataset, PlaygroundRun} from '../types';
// Vite proxies /api in development; the production bundle is served by FastAPI.
const BASE = import.meta.env.DEV ? '/api' : '';
async function request<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(BASE + path, {method: body === undefined ? 'GET' : 'POST', headers: body === undefined ? {} : {'Content-Type': 'application/json'}, body: body === undefined ? undefined : JSON.stringify(body), signal});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Invalid input. Check the model format and tensor dimensions.');
  return data;
}
export const api = {
  dataset: (dataset: DatasetName, seed: number, noise: number, signal?: AbortSignal) => request<PlaygroundDataset>(`/playground/dataset?dataset=${dataset}&seed=${seed}&noise=${noise}`, undefined, signal),
  train: (config: PlaygroundConfig) => request<PlaygroundRun>('/playground/train', config),
  models: () => request<Model[]>('/demo-models'), samples: (id = 'mnist-mlp') => request<Sample[]>(`/samples?model_id=${encodeURIComponent(id)}`),
  run: (model_id: string, input: number[]) => request<Trace>('/run-model', {model_id, input}),
  upload: (spec: unknown) => request<Model>('/upload-model', spec),
  focus: (id: string, layer: number, neuron: number, signal?: AbortSignal) => request<FocusView>(`/traces/${id}/focus?layer=${layer}&neuron=${neuron}`, undefined, signal),
  neuron: (id: string, layer: number, neuron: number, limit: number, signal?: AbortSignal) => request<Neuron>(`/traces/${id}/layers/${layer}/neurons/${neuron}?limit=${limit}`, undefined, signal),
};
