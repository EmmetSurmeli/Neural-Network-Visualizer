import type {DatasetName, PlaygroundConfig} from '../types';
import {ApiError, fetchJson} from '../lib/http';
import {elapsedBucket, modelProperties, reportError, track} from '../lib/telemetry';
import * as schema from './schemas';
const BASE = (import.meta.env.VITE_API_URL || (import.meta.env.DEV ? '/api' : '')).replace(/\/$/, '');
let sessionPromise: Promise<string> | undefined;
let token = '';
try {token = sessionStorage.getItem('neuralscope-session') ?? '';} catch { /* Private browsing may disable storage. */ }
function forgetSession() {token = ''; sessionPromise = undefined; try {sessionStorage.removeItem('neuralscope-session');} catch { /* In-memory fallback. */ }}
async function session() {
  if (token) return token;
  if (!sessionPromise) sessionPromise = fetchJson(BASE + '/session', schema.session.parse, {body: {}}).then(data => {token = data.token; try {sessionStorage.setItem('neuralscope-session', token);} catch { /* In-memory fallback. */ } return token;}).finally(() => {sessionPromise = undefined;});
  return sessionPromise;
}
async function request<T>(path: string, validate: (data: unknown) => T, body?: unknown, signal?: AbortSignal, protectedRoute = false, timeout = 25000): Promise<T> {
  try {
    const result = await fetchJson(BASE + path, validate, {body, signal, timeout, token: protectedRoute ? await session() : undefined});
    window.dispatchEvent(new Event('service-available'));
    return result;
  } catch (error) {
    if (error instanceof ApiError) {
      if (error.code === 'SESSION_EXPIRED') forgetSession();
      if (['BACKEND_UNAVAILABLE', 'OFFLINE', 'REQUEST_TIMEOUT'].includes(error.code)) window.dispatchEvent(new Event('service-unavailable'));
      if (['INTERNAL_ERROR', 'INVALID_RESPONSE'].includes(error.code)) reportError(error.code, error.requestId);
    }
    throw error;
  }
}
export const api = {
  health: () => request('/health', schema.health.parse, undefined, undefined, false, 10000),
  clearSession: async () => {await fetchJson(BASE + '/session', () => undefined, {method: 'DELETE', token: await session()}); forgetSession();},
  dataset: (dataset: DatasetName, seed: number, noise: number, signal?: AbortSignal) => request(`/playground/dataset?dataset=${dataset}&seed=${seed}&noise=${noise}`, schema.dataset.parse, undefined, signal),
  train: async (config: PlaygroundConfig) => {const start = performance.now(); try {const run = await request('/playground/train', schema.run.parse, config, undefined, true, 100000); track('playground_training_completed', {elapsed: elapsedBucket(performance.now() - start), outcome: 'success'}); return run;} catch (error) {track('playground_training_failed', {outcome: 'failure'}); throw error;}},
  models: () => request('/demo-models', schema.models.parse),
  samples: (id = 'mnist-mlp') => request(`/samples?model_id=${encodeURIComponent(id)}`, schema.samples.parse),
  run: async (model_id: string, input: number[], record = true) => {
    const properties = {...modelProperties(model_id), input_size: input.length <= 16 ? '1–16' : input.length <= 1024 ? '17–1024' : '1025+'};
    const start = performance.now();
    if (record) track('forward_pass_started', properties);
    try {const result = await request('/run-model', schema.trace.parse, {model_id, input}, undefined, true); if (record) track('forward_pass_completed', {...properties, outcome: 'success', elapsed: elapsedBucket(performance.now() - start), layer_count: result.layers.length}); return result;}
    catch (error) {if (record) track('forward_pass_failed', {...properties, outcome: 'failure'}); throw error;}
  },
  upload: (spec: unknown) => request('/upload-model', schema.model.parse, spec, undefined, true),
  focus: (id: string, layer: number, neuron: number, signal?: AbortSignal) => request(`/traces/${id}/focus?layer=${layer}&neuron=${neuron}`, schema.focus.parse, undefined, signal, true),
  neuron: (id: string, layer: number, neuron: number, limit: number, signal?: AbortSignal) => request(`/traces/${id}/layers/${layer}/neurons/${neuron}?limit=${limit}`, schema.neuron.parse, undefined, signal, true),
};
