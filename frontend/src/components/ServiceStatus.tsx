import {useEffect, useRef} from 'react';
import {api} from '../services/api';
import useRequestState, {useOnline} from '../hooks/useRequestState';
import {ApiError} from '../lib/http';
import RequestNotice from './RequestNotice';
export default function ServiceStatus() {
  const request = useRequestState(), online = useOnline();
  const retry = useRef<() => void>(() => {});
  useEffect(() => {
    let stopped = false, timer: number | undefined, attempts = 0;
    const check = async () => {
      request.start(attempts ? 'The service may be waking up. Checking again…' : 'Connecting to the service…');
      try {await api.health(); if (!stopped) request.success();}
      catch (error) {
        if (stopped) return;
        request.fail(error);
        if (navigator.onLine && attempts < 7) {
          const delay = Math.min(15000, 5000 + attempts * 2500);
          attempts++;
          timer = window.setTimeout(() => void check(), delay);
        }
      }
    };
    retry.current = () => {window.clearTimeout(timer); attempts = 0; void check();};
    if (online) void check(); else request.fail(new ApiError('OFFLINE', 'You are offline. Reconnect to run inputs or train. Recorded results and example downloads remain available if already loaded.'));
    const available = () => {if (navigator.onLine) {window.clearTimeout(timer); request.success();}};
    const unavailable = () => request.fail(new ApiError('BACKEND_UNAVAILABLE', 'Service unavailable. Live inference and training require the service. Retry shortly; the model format guide and bundled examples are still available.'));
    window.addEventListener('service-available', available); window.addEventListener('service-unavailable', unavailable);
    return () => {stopped = true; window.clearTimeout(timer); window.removeEventListener('service-available', available); window.removeEventListener('service-unavailable', unavailable);};
  }, [online]);
  return <div className="service-status"><RequestNotice state={request.state} retry={() => retry.current()}/><span className="service-label" role="status">{request.state.status === 'success' ? 'Service ready' : !online ? 'Browser offline' : 'Service not ready'}</span></div>;
}
