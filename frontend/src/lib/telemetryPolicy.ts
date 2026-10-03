export const events = ['page_view', 'model_loaded', 'forward_pass_started', 'forward_pass_completed', 'forward_pass_failed', 'playground_training_completed', 'playground_training_failed', 'visualizer_focus_opened', 'replay_started', 'model_import_started', 'model_import_failed'] as const;
export type ProductEvent = typeof events[number];
export const builtins = ['mnist-mlp', 'shapes-cnn', 'strokes-cnn', 'xor-mlp', 'tiny-mlp'];
export function modelProperties(id: string) {return builtins.includes(id) ? {model_category: 'built-in', model_id: id} : {model_category: id.startsWith('playground-') ? 'playground' : 'imported'};}
export function elapsedBucket(ms: number) {return ms < 1000 ? '<1s' : ms < 5000 ? '1–5s' : ms < 30000 ? '5–30s' : '30s+';}
export function safeProperties(properties: Record<string, unknown>) {
  const result: Record<string, string | number> = {};
  const enums: Record<string, readonly string[]> = {route: ['visualizer', 'playground'], model_id: builtins, model_category: ['built-in', 'imported', 'playground'], outcome: ['success', 'failure'], input_size: ['1–16', '17–1024', '1025+'], elapsed: ['<1s', '1–5s', '5–30s', '30s+'], device_class: ['small', 'large']};
  for (const [key, allowed] of Object.entries(enums)) if (typeof properties[key] === 'string' && allowed.includes(properties[key] as string)) result[key] = properties[key] as string;
  if (Number.isInteger(properties.layer_count) && Number(properties.layer_count) >= 0 && Number(properties.layer_count) <= 64) result.layer_count = Number(properties.layer_count);
  return result;
}
export function scrubErrorEvent(event: Record<string, any>, release = 'local') {
  const tags: Record<string, string> = {};
  if (/^[a-f0-9]{32}$/.test(event.tags?.request_id ?? '')) tags.request_id = event.tags.request_id;
  if (['visualizer', 'playground'].includes(event.tags?.route)) tags.route = event.tags.route;
  if (['built-in', 'imported', 'playground'].includes(event.tags?.model_category)) tags.model_category = event.tags.model_category;
  const codes = ['INTERNAL_ERROR', 'INVALID_RESPONSE', 'UI_ERROR', 'REQUEST_FAILED'];
  tags.error_type = codes.includes(event.tags?.error_type) ? event.tags.error_type : 'UI_ERROR';
  return {type: undefined, event_id: event.event_id, timestamp: event.timestamp, platform: 'javascript', level: 'error' as const, release, environment: 'production', tags, message: tags.error_type};
}
