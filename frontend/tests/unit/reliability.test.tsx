import React from 'react';
import {afterEach, describe, expect, it, vi} from 'vitest';
import {cleanup, fireEvent, render, screen, act, renderHook} from '@testing-library/react';
import RequestNotice from '../../src/components/RequestNotice';
import useRequestState from '../../src/hooks/useRequestState';
import {ApiError, fetchJson} from '../../src/lib/http';
import {events, modelProperties, safeProperties, scrubErrorEvent} from '../../src/lib/telemetryPolicy';
import * as schema from '../../src/services/schemas';
afterEach(() => {cleanup(); vi.unstubAllGlobals(); vi.useRealTimers();});

describe('Request recovery', () => {
  it('handles idle, loading, success, empty, stale, error and offline', () => {
    const {result} = renderHook(useRequestState);
    expect(result.current.state.status).toBe('idle');
    act(() => result.current.start('Running…')); expect(result.current.state.status).toBe('loading');
    act(() => result.current.success()); expect(result.current.state.status).toBe('success');
    act(() => result.current.success(true)); expect(result.current.state.status).toBe('empty');
    act(() => result.current.fail(new ApiError('REQUEST_FAILED', 'Try again'), true)); expect(result.current.state.status).toBe('stale');
    act(() => result.current.fail(new ApiError('REQUEST_FAILED', 'Try again'))); expect(result.current.state.status).toBe('error');
    act(() => result.current.fail(new ApiError('OFFLINE', 'Reconnect'))); expect(result.current.state.status).toBe('offline');
  });
  it('renders progress, request reference, stale result and a rate-aware retry', () => {
    vi.useFakeTimers(); const retry = vi.fn();
    const view = render(<RequestNotice state={{status: 'loading', message: 'Training…'}} retry={retry}/>);
    expect(screen.getByRole('status').textContent).toContain('Training'); expect(screen.queryByRole('button')).toBeNull();
    view.rerender(<RequestNotice state={{status: 'stale', previous: true, error: new ApiError('RATE_LIMITED', 'Wait', 'abc', 2)}} retry={retry}/>);
    expect(screen.getByRole('alert').textContent).toContain('Previous result'); expect(screen.getByText('Request reference: abc')).toBeTruthy();
    fireEvent.click(screen.getByRole('button')); expect(retry).not.toHaveBeenCalled();
    act(() => {vi.advanceTimersByTime(2000);}); fireEvent.click(screen.getByRole('button', {name: 'Retry'})); expect(retry).toHaveBeenCalledOnce();
  });
  it('times out, aborts the underlying request and allows a new request', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch', vi.fn((_url, options) => new Promise((_, reject) => options.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError'))))));
    const result = fetchJson('/test', schema.health.parse, {timeout: 50});
    const assertion = expect(result).rejects.toMatchObject({code: 'REQUEST_TIMEOUT'});
    await vi.advanceTimersByTimeAsync(51); await assertion;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"status":"ok"}')));
    expect(await fetchJson('/test', schema.health.parse)).toEqual({status: 'ok'});
  });
  it('rejects malformed JSON and structurally incomplete successful responses', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(new Response('<html>')).mockResolvedValueOnce(new Response('{}')));
    await expect(fetchJson('/test', schema.health.parse)).rejects.toMatchObject({code: 'INVALID_RESPONSE'});
    await expect(fetchJson('/test', schema.trace.parse)).rejects.toMatchObject({code: 'INVALID_RESPONSE'});
  });
  it('preserves stable server errors and request IDs', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"error":{"code":"TRACE_EXPIRED","message":"Run again"}}', {status: 404, headers: {'X-Request-ID': 'abcd'}})));
    await expect(fetchJson('/test', schema.focus.parse)).rejects.toMatchObject({code: 'TRACE_EXPIRED', requestId: 'abcd', message: 'Run again'});
  });
  it('handles backend-unavailable startup and browser offline', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    await expect(fetchJson('/health', schema.health.parse)).rejects.toMatchObject({code: 'BACKEND_UNAVAILABLE'});
    vi.stubGlobal('navigator', {onLine: false});
    await expect(fetchJson('/health', schema.health.parse)).rejects.toMatchObject({code: 'OFFLINE'});
  });
});
describe('Telemetry privacy', () => {
  it('accepts only the documented event names and coarse properties', () => {
    expect(events).toHaveLength(11);
    expect(safeProperties({route: 'visualizer', model_id: 'uploaded-private-name', weights: [99], input: [1, 2], filename: 'secret.json', trace: {}, layer_count: 4})).toEqual({route: 'visualizer', layer_count: 4});
    expect(modelProperties('my-private-model')).toEqual({model_category: 'imported'});
    expect(modelProperties('tiny-mlp')).toEqual({model_category: 'built-in', model_id: 'tiny-mlp'});
  });
  it('strips messages, request bodies, attachments, URLs and arbitrary Sentry contexts', () => {
    const clean = scrubErrorEvent({message: 'SECRET', exception: {values: [{value: 'SECRET'}]}, request: {url: 'SECRET', data: 'SECRET'}, user: {email: 'SECRET'}, attachments: ['SECRET'], breadcrumbs: ['SECRET'], extra: {input: 'SECRET'}, tags: {filename: 'SECRET', route: 'visualizer', request_id: 'a'.repeat(32)}});
    expect(JSON.stringify(clean)).not.toContain('SECRET'); expect(clean.tags.request_id).toBe('a'.repeat(32));
  });
});
