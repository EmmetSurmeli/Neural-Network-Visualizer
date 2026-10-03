import {useRef, useState} from 'react';
import {preference, setAnalytics} from '../lib/telemetry';
import {api} from '../services/api';
import useRequestState from '../hooks/useRequestState';
import RequestNotice from './RequestNotice';
export default function Privacy() {
  const dialog = useRef<HTMLDialogElement>(null), [choice, setChoice] = useState(preference), request = useRequestState();
  const choose = (enabled: boolean) => {setAnalytics(enabled); setChoice(preference());};
  async function clear() {request.start('Clearing temporary models and traces…'); try {await api.clearSession(); location.reload();} catch (error) {request.fail(error);}}
  return <footer className="privacy-footer"><span>Models and inputs are processed in temporary server memory.</span><button className="text-button" onClick={() => dialog.current?.showModal()}>Privacy & session</button>
    {choice === 'unset' && <div className="analytics-choice"><span>Allow anonymous usage counts? Model contents and inputs are never included.</span><button onClick={() => choose(true)}>Allow</button><button onClick={() => choose(false)}>No thanks</button></div>}
    <dialog ref={dialog} className="import-dialog" aria-labelledby="privacy-title"><div className="dialog-heading"><h2 id="privacy-title">Privacy & session</h2><button aria-label="Close privacy notice" onClick={() => dialog.current?.close()}>×</button></div><div className="dialog-content">
      <p>Imports, input values, drawings, and traces are sent to the Python service for computation and kept in memory only. They are never sent to analytics or error monitoring. They are not saved to disk or a database.</p>
      <p>Your temporary session expires after 30 minutes without a session request, or when the server restarts. Older traces may be removed sooner when the cache fills. Clearing the session removes its server models and traces immediately.</p>
      <p>Optional anonymous analytics count actions such as completed forward passes. An in-memory anonymous identifier lasts for this page visit. No accounts, recordings, raw URLs, or uploaded filenames are collected. We respect browser Do Not Track and Global Privacy Control.</p>
      <label className="privacy-toggle"><input type="checkbox" checked={choice === 'on'} onChange={e => choose(e.target.checked)}/> Allow anonymous usage analytics</label>
      <p>Production error monitoring receives generic error codes, route categories, software version, and request references. Hosting and telemetry providers necessarily receive connection metadata such as an IP address. Analytics are disabled on local and preview builds.</p>
      <RequestNotice state={request.state} retry={() => void clear()}/><button className="secondary-button" disabled={request.state.status === 'loading'} onClick={() => void clear()}>Clear temporary session & reload</button>
    </div></dialog>
  </footer>;
}
