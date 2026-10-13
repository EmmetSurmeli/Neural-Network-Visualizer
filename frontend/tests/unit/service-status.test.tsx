import React from 'react';
import {afterEach, expect, it, vi} from 'vitest';
import {act, cleanup, render, screen} from '@testing-library/react';
import ServiceStatus from '../../src/components/ServiceStatus';
import {api} from '../../src/services/api';

vi.mock('../../src/services/api', () => ({api: {health: vi.fn()}}));
afterEach(() => {cleanup(); vi.clearAllMocks(); vi.useRealTimers();});

it('reconnects automatically after an initial health failure', async () => {
  vi.useFakeTimers();
  vi.mocked(api.health).mockRejectedValueOnce(new Error('starting')).mockResolvedValue({status: 'ok'});
  render(<ServiceStatus/>);
  await act(async () => {await Promise.resolve();});
  expect(screen.getByText('Service not ready')).toBeTruthy();
  await act(async () => {await vi.advanceTimersByTimeAsync(5000);});
  expect(api.health).toHaveBeenCalledTimes(2);
  expect(screen.getByText('Service ready')).toBeTruthy();
});
