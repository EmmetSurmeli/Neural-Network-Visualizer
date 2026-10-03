import {useEffect, useState} from 'react';
import type {RequestState} from '../hooks/useRequestState';
export default function RequestNotice({state, retry}: {state: RequestState; retry?: () => void}) {
  const [remaining, setRemaining] = useState(0);
  useEffect(() => {setRemaining(state.error?.retryAfter ?? 0); if (!state.error?.retryAfter) return; const timer = setInterval(() => setRemaining(n => Math.max(0, n - 1)), 1000); return () => clearInterval(timer);}, [state.error]);
  if (state.status === 'idle' || state.status === 'success') return null;
  const pending = state.status === 'loading';
  return <div className={`request-notice ${state.status}`} role={state.error ? 'alert' : 'status'} aria-busy={pending}>
    <div><span>{state.error?.message ?? state.message ?? (state.status === 'empty' ? 'Nothing to show yet. Load an example and run an input.' : 'Previous result. Run the current input again.')}</span>{state.previous && <small>Previous result · the current request has not completed.</small>}{state.error?.requestId && <small>Request reference: {state.error.requestId}</small>}</div>
    {!pending && retry && <button className="secondary-button" disabled={remaining > 0} onClick={retry}>{remaining ? `Retry in ${remaining}s` : 'Retry'}</button>}
  </div>;
}
