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
      <p>Last updated October 5, 2026. NeuralScope is a public demonstration operated by the project maintainer. It has no accounts or saved history.</p>
      <h3>What the site processes</h3>
      <p>Your uploaded JSON models, input values, drawings, and resulting traces go to the Python service on Render so it can compute and display a forward pass. They stay in temporary server memory, not a database or permanent file. Vercel hosts the website. Both hosting providers receive connection data, including IP addresses, to deliver requests. The service logs a random request reference, route category, status, and duration without request bodies, model weights, or inputs.</p>
      <p>The server removes your session after 30 minutes of inactivity or a restart. At capacity, the least recently used session without a request in progress may be removed sooner. Older traces can also expire sooner. Closing the tab does not immediately clear server memory; use the button below to remove your session's models and traces now. Hosting providers may retain operational logs under their own policies.</p>
      <h3>Optional measurement</h3>
      <p>With your permission, PostHog can receive coarse events such as a completed forward pass, a model category, and a broad input-size range. Its anonymous identifier stays in memory for this page visit. Model contents, inputs, traces, full page addresses, and filenames are excluded. You can withdraw permission below. Browser Do Not Track and Global Privacy Control override an Allow choice.</p>
      <label className="privacy-toggle"><input type="checkbox" checked={choice === 'on'} onChange={e => choose(e.target.checked)}/> Allow anonymous usage analytics</label>
      <p>If enabled, Sentry receives limited error codes, route categories, software version, and request references to diagnose failures. Error reports and service logs do not include model contents. PostHog and Sentry are optional service providers; the site works without them. Analytics are disabled on local and preview builds. See the providers' policies for their separate retention and transfer practices: <a href="https://vercel.com/legal/privacy-notice" target="_blank" rel="noreferrer">Vercel</a>, <a href="https://render.com/privacy" target="_blank" rel="noreferrer">Render</a>, <a href="https://posthog.com/privacy" target="_blank" rel="noreferrer">PostHog</a>, and <a href="https://sentry.io/privacy/" target="_blank" rel="noreferrer">Sentry</a>.</p>
      <h3>Your choices</h3>
      <p>Running the demo requires sending the chosen input to Render. Optional analytics rely on your permission; computing the demo, preventing abuse, and diagnosing failures serve the legitimate interest of operating the site. Where privacy law applies, you may request access, correction, deletion, or object to certain processing. You can clear temporary session data or stop analytics here at any time. Data may be processed in the United States and other provider locations. Do not upload private or sensitive information to the public demo.</p>
      <p>For privacy questions or rights requests, contact the maintainer through the <a href="https://github.com/EmmetSurmeli" target="_blank" rel="noreferrer">public GitHub profile</a>. Do not put private information in a public issue.</p>
      <RequestNotice state={request.state} retry={() => void clear()}/><button className="secondary-button" disabled={request.state.status === 'loading'} onClick={() => void clear()}>Clear temporary session & reload</button>
    </div></dialog>
  </footer>;
}
