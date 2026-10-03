import {events, safeProperties, scrubErrorEvent, type ProductEvent} from './telemetryPolicy';
export {modelProperties, elapsedBucket} from './telemetryPolicy';
const release = import.meta.env.VITE_APP_VERSION || '1.0.0';
const production = import.meta.env.PROD && import.meta.env.VITE_APP_ENV === 'production' && location.hostname === import.meta.env.VITE_PRODUCTION_HOST;
let initialized = false;
let initializing = false;
let posthog: typeof import('posthog-js/dist/module.no-external')['default'];
let sentry: typeof import('@sentry/react') | undefined;
export function preference(): string {try {return localStorage.getItem('neuralscope-analytics') ?? 'unset';} catch {return 'off';}}
function permitted() {return production && preference() === 'on' && navigator.doNotTrack !== '1' && !(navigator as Navigator & {globalPrivacyControl?: boolean}).globalPrivacyControl;}
export function setAnalytics(enabled: boolean) {
  try {localStorage.setItem('neuralscope-analytics', enabled ? 'on' : 'off');} catch {return;}
  if (enabled) void initializeAnalytics(); else if (initialized) posthog.opt_out_capturing();
  window.dispatchEvent(new Event('analytics-preference'));
}
async function initializeAnalytics() {
  if (!permitted() || !import.meta.env.VITE_POSTHOG_KEY) return;
  if (initialized) {posthog.opt_in_capturing({captureEventName: false}); return;}
  if (initializing) return;
  initializing = true;
  try {
  posthog = (await import('posthog-js/dist/module.no-external')).default;
  if (!permitted()) return;
  posthog.init(import.meta.env.VITE_POSTHOG_KEY, {
    api_host: import.meta.env.VITE_POSTHOG_HOST || 'https://us.i.posthog.com',
    autocapture: false, capture_pageview: false, capture_pageleave: false, capture_exceptions: false,
    disable_session_recording: true, disable_external_dependency_loading: true, advanced_disable_flags: true,
    person_profiles: 'never', persistence: 'memory', ip: false,
    before_send: event => {
      if (!event || !permitted() || !events.includes(event.event as ProductEvent)) return null;
      event.properties = {...safeProperties(event.properties), distinct_id: event.properties.distinct_id, $process_person_profile: false, $geoip_disable: true, frontend_version: release};
      return event;
    },
  });
  initialized = true;
  track('page_view');
  } catch { /* Optional analytics must never interrupt a visualization. */ } finally {initializing = false;}
}
export function track(event: ProductEvent, properties: Record<string, unknown> = {}) {
  if (initialized && permitted()) posthog.capture(event, safeProperties({route: location.hash === '#/playground' ? 'playground' : 'visualizer', device_class: innerWidth < 700 ? 'small' : 'large', ...properties}));
}
export function reportError(code: string, requestId?: string) {
  if (production && import.meta.env.VITE_SENTRY_DSN) sentry?.captureEvent({message: code, tags: {error_type: code, request_id: requestId, route: location.hash === '#/playground' ? 'playground' : 'visualizer'}});
}
export async function initializeTelemetry() {
  void initializeAnalytics();
  if (production && import.meta.env.VITE_SENTRY_DSN) {
    try {
    sentry = await import('@sentry/react');
    sentry.init({dsn: import.meta.env.VITE_SENTRY_DSN, release, environment: 'production', defaultIntegrations: false, tracesSampleRate: 0, beforeSend: event => scrubErrorEvent(event, release)});
    window.addEventListener('error', () => reportError('UI_ERROR'));
    window.addEventListener('unhandledrejection', () => reportError('UI_ERROR'));
    } catch { /* Monitoring is optional. */ }
  }
}
