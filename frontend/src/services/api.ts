import type {Model, Neuron, Sample, Trace} from '../types';
// Vite proxies /api in development; the production bundle is served by FastAPI.
const BASE = import.meta.env.DEV ? '/api' : '';
async function request<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(BASE + path, {method: body === undefined ? 'GET' : 'POST', headers: body === undefined ? {} : {'Content-Type': 'application/json'}, body: body === undefined ? undefined : JSON.stringify(body), signal});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Invalid input. Check the model format and tensor dimensions.');
  return data;
}
export const api = {
  models: () => request<Model[]>('/demo-models'), samples: () => request<Sample[]>('/samples'),
  run: (model_id: string, input: number[]) => request<Trace>('/run-model', {model_id, input}),
  upload: (spec: unknown) => request<Model>('/upload-model', spec),
  neuron: (id: string, layer: number, neuron: number, limit: number, signal?: AbortSignal) => request<Neuron>(`/traces/${id}/layers/${layer}/neurons/${neuron}?limit=${limit}`, undefined, signal),
};
