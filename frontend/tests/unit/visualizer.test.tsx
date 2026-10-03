import React from 'react';
import {afterEach, beforeEach, expect, it, vi} from 'vitest';
import {cleanup, fireEvent, render, screen, waitFor} from '@testing-library/react';
import Visualizer from '../../src/Visualizer';
import ErrorBoundary from '../../src/components/ErrorBoundary';
import {api} from '../../src/services/api';
import {ApiError} from '../../src/lib/http';
import {trace as traceSchema} from '../../src/services/schemas';
import bundled from '../../src/data/bundled.json';
import fixture from '../fixtures/tiny-trace.json';
vi.mock('../../src/services/api', () => ({api: {models: vi.fn(), samples: vi.fn(), run: vi.fn(), upload: vi.fn()}}));
vi.mock('../../src/lib/telemetry', () => ({track: vi.fn(), reportError: vi.fn(), modelProperties: () => ({})}));
vi.mock('../../src/components/DigitCanvas', () => ({default: () => <div>Drawing canvas</div>}));
vi.mock('../../src/components/Network', () => ({default: ({trace}: {trace: {model_name: string} | null}) => <div data-testid="network">{trace?.model_name ?? 'Empty network'}</div>}));
vi.mock('../../src/components/Inspector', () => ({default: () => null}));
vi.mock('../../src/components/Output', () => ({default: () => null}));
beforeEach(() => {
  vi.mocked(api.models).mockResolvedValue(bundled.models);
  vi.mocked(api.samples).mockResolvedValue(bundled.samples);
  vi.mocked(api.run).mockResolvedValue(traceSchema.parse(fixture));
});
afterEach(() => {cleanup(); vi.resetAllMocks();});

it('loads a sample without falsely counting an automatic visualization', async () => {
  render(<Visualizer handoff={null} active/>);
  await waitFor(() => expect((screen.getByRole('button', {name: 'Run forward pass'}) as HTMLButtonElement).disabled).toBe(false));
  expect(api.run).not.toHaveBeenCalled(); expect(screen.getByTestId('network').textContent).toBe('Empty network');
});

it('retains the previous trace on a failed run and retries that operation', async () => {
  render(<Visualizer handoff={null} active/>);
  await waitFor(() => expect((screen.getByRole('combobox', {name: 'Active model'}) as HTMLSelectElement).disabled).toBe(false));
  fireEvent.change(screen.getByRole('combobox', {name: 'Active model'}), {target: {value: 'tiny-mlp'}});
  fireEvent.click(screen.getByRole('button', {name: 'Run forward pass'}));
  await waitFor(() => expect(screen.getByTestId('network').textContent).toBe(fixture.model_name));
  vi.mocked(api.run).mockRejectedValueOnce(new ApiError('BACKEND_UNAVAILABLE', 'Service unavailable', 'a'.repeat(32)));
  fireEvent.click(screen.getByRole('button', {name: 'Run forward pass'}));
  await screen.findByText('Previous result · the current request has not completed.');
  expect(screen.getByTestId('network').textContent).toBe(fixture.model_name);
  fireEvent.click(screen.getByRole('button', {name: 'Retry'}));
  await waitFor(() => expect(screen.queryByRole('alert')).toBeNull());
  expect(api.run).toHaveBeenCalledTimes(3);
});

it('keeps bundled models and format access when startup fails, and retries startup', async () => {
  vi.mocked(api.models).mockRejectedValueOnce(new ApiError('BACKEND_UNAVAILABLE', 'Service unavailable'));
  render(<Visualizer handoff={null} active/>);
  await screen.findByRole('alert');
  expect(screen.getByRole('button', {name: 'Format'})).toBeTruthy();
  expect(screen.getByRole('option', {name: 'Tiny MLP · 3 inputs'})).toBeTruthy();
  fireEvent.click(screen.getByRole('button', {name: 'Retry'}));
  await waitFor(() => expect(screen.queryByRole('alert')).toBeNull());
  expect(api.models).toHaveBeenCalledTimes(2);
});

it('rejects invalid numeric input without sending it to the service', async () => {
  render(<Visualizer handoff={null} active/>);
  await waitFor(() => expect((screen.getByRole('combobox', {name: 'Active model'}) as HTMLSelectElement).disabled).toBe(false));
  fireEvent.change(screen.getByRole('combobox', {name: 'Active model'}), {target: {value: 'tiny-mlp'}});
  fireEvent.change(screen.getByRole('textbox'), {target: {value: '["private data"]'}});
  fireEvent.click(screen.getByRole('button', {name: 'Run forward pass'}));
  expect((await screen.findByRole('alert')).textContent).toContain('exactly 3 finite numbers');
  expect(api.run).not.toHaveBeenCalled();
  fireEvent.change(screen.getByRole('textbox'), {target: {value: '[2,0,-1]'}});
  fireEvent.click(screen.getByRole('button', {name: 'Retry'}));
  await waitFor(() => expect(api.run).toHaveBeenCalledWith('tiny-mlp', [2, 0, -1]));
});

it('recovers from rendering errors without displaying the exception contents', () => {
  const log = vi.spyOn(console, 'error').mockImplementation(() => {});
  function Broken(): React.ReactNode {throw new Error('SECRET_MODEL_CONTENT');}
  render(<ErrorBoundary><Broken/></ErrorBoundary>);
  expect(screen.getByRole('alert').textContent).not.toContain('SECRET_MODEL_CONTENT');
  expect(screen.getByRole('button', {name: 'Reload demo'})).toBeTruthy(); log.mockRestore();
});
