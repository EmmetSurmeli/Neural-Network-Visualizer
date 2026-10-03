import {beforeEach, afterEach, expect, it, vi} from 'vitest';
const mock = vi.hoisted(() => ({init: vi.fn(), capture: vi.fn(), opt_out_capturing: vi.fn(), opt_in_capturing: vi.fn()}));
vi.mock('posthog-js/dist/module.no-external', () => ({default: mock}));
beforeEach(() => {
  vi.resetModules(); vi.clearAllMocks(); localStorage.clear();
  vi.stubEnv('PROD', true); vi.stubEnv('VITE_APP_ENV', 'production'); vi.stubEnv('VITE_PRODUCTION_HOST', location.hostname); vi.stubEnv('VITE_POSTHOG_KEY', 'test-public-project-token');
});
afterEach(() => {vi.unstubAllEnvs(); vi.unstubAllGlobals();});
it('does not initialize analytics before consent; scrubs SDK properties and stops after opt-out', async () => {
  const telemetry = await import('../../src/lib/telemetry');
  await telemetry.initializeTelemetry(); expect(mock.init).not.toHaveBeenCalled();
  telemetry.setAnalytics(true);
  await vi.waitFor(() => expect(mock.init).toHaveBeenCalledOnce());
  const config = mock.init.mock.calls[0][1];
  expect(config.autocapture).toBe(false); expect(config.disable_session_recording).toBe(true); expect(config.person_profiles).toBe('never');
  const event = config.before_send({event: 'forward_pass_completed', properties: {distinct_id: 'anonymous', route: 'visualizer', weights: 'SECRET', $current_url: 'SECRET', filename: 'SECRET'}});
  expect(JSON.stringify(event)).not.toContain('SECRET');
  expect(config.before_send({event: 'arbitrary_event', properties: {}})).toBeNull();
  telemetry.setAnalytics(false);
  expect(mock.opt_out_capturing).toHaveBeenCalledOnce();
  expect(config.before_send({event: 'page_view', properties: {}})).toBeNull();
});
it('never sends production analytics from a preview build', async () => {
  vi.stubEnv('VITE_APP_ENV', 'preview'); localStorage.setItem('neuralscope-analytics', 'on');
  const telemetry = await import('../../src/lib/telemetry');
  await telemetry.initializeTelemetry(); telemetry.track('page_view');
  expect(mock.init).not.toHaveBeenCalled(); expect(mock.capture).not.toHaveBeenCalled();
});
it('honors browser privacy controls even when the preference is on', async () => {
  localStorage.setItem('neuralscope-analytics', 'on'); vi.stubGlobal('navigator', {doNotTrack: '1', globalPrivacyControl: true});
  const telemetry = await import('../../src/lib/telemetry'); await telemetry.initializeTelemetry();
  expect(mock.init).not.toHaveBeenCalled();
});
