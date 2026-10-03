export class ApiError extends Error {
  constructor(public code: string, message: string, public requestId?: string, public retryAfter = 0) {super(message);}
}
export function readableError(error: unknown): ApiError {
  if (error instanceof ApiError) return error;
  if (error instanceof SyntaxError) return new ApiError('INVALID_INPUT', 'Enter valid JSON. Check brackets, commas, and numeric values.');
  return new ApiError('REQUEST_FAILED', 'The request failed. Please retry or load a bundled example.');
}
export async function fetchJson<T>(url: string, validate: (value: unknown) => T, options: {body?: unknown; method?: string; signal?: AbortSignal; token?: string; timeout?: number} = {}): Promise<T> {
  if (!navigator.onLine) throw new ApiError('OFFLINE', 'You are offline. Reconnect, then retry. Your previous result is still available.');
  const controller = new AbortController();
  let timedOut = false;
  const abort = () => controller.abort();
  options.signal?.addEventListener('abort', abort, {once: true});
  if (options.signal?.aborted) abort();
  const timer = setTimeout(() => {timedOut = true; abort();}, options.timeout ?? 25000);
  try {
    const response = await fetch(url, {method: options.method ?? (options.body === undefined ? 'GET' : 'POST'), credentials: 'omit', signal: controller.signal,
      headers: {...(options.body === undefined ? {} : {'Content-Type': 'application/json'}), ...(options.token ? {'X-NeuralScope-Session': options.token} : {})},
      body: options.body === undefined ? undefined : JSON.stringify(options.body)});
    const requestId = response.headers.get('X-Request-ID') ?? undefined;
    let value;
    try {value = await response.json();} catch {throw new ApiError(response.ok ? 'INVALID_RESPONSE' : 'BACKEND_UNAVAILABLE', response.ok ? 'The service returned an incomplete response. Please retry.' : 'Service unavailable. It may be starting; retry shortly.', requestId);}
    if (!response.ok) {
      const error = value?.error;
      throw new ApiError(typeof error?.code === 'string' ? error.code : 'REQUEST_FAILED', typeof error?.message === 'string' ? error.message : 'The request failed. Please retry.', requestId, Number(response.headers.get('Retry-After')) || 0);
    }
    try {return validate(value);} catch {throw new ApiError('INVALID_RESPONSE', 'The service returned an incomplete response. Please retry.', requestId);}
  } catch (error) {
    if (options.signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
    if (timedOut) throw new ApiError('REQUEST_TIMEOUT', 'The request took too long. Retry, or use a smaller model or fewer epochs.');
    if (error instanceof ApiError) throw error;
    throw new ApiError('BACKEND_UNAVAILABLE', 'Service unavailable. Live inference and training need the service. It may be starting; retry shortly.');
  } finally {clearTimeout(timer); options.signal?.removeEventListener('abort', abort);}
}
