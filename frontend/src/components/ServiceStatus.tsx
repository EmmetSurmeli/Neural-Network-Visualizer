import {useEffect} from 'react';
import {api} from '../services/api';
import useRequestState, {useOnline} from '../hooks/useRequestState';
import {ApiError} from '../lib/http';
import RequestNotice from './RequestNotice';
export default function ServiceStatus() {
  const request = useRequestState(), online = useOnline();
  const check = async () => {request.start('Connecting to the service…'); try {await api.health(); request.success();} catch (error) {request.fail(error);}};
  useEffect(() => {
    if (online) void check(); else request.fail(new ApiError('OFFLINE', 'You are offline. Reconnect to run inputs or train. Recorded results and example downloads remain available if already loaded.'));
    const available = () => {if (navigator.onLine) request.success();};
    const unavailable = () => request.fail(new ApiError('BACKEND_UNAVAILABLE', 'Service unavailable. Live inference and training require the service. Retry shortly; the model format guide and bundled examples are still available.'));
    window.addEventListener('service-available', available); window.addEventListener('service-unavailable', unavailable);
    return () => {window.removeEventListener('service-available', available); window.removeEventListener('service-unavailable', unavailable);};
  }, [online]);
  return <div className="service-status"><RequestNotice state={request.state} retry={() => void check()}/><span className="service-label" role="status">{request.state.status === 'success' ? 'Service ready' : !online ? 'Browser offline' : 'Service not ready'}</span></div>;
}
