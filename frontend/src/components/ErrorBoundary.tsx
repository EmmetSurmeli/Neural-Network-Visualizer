import {Component, type ReactNode} from 'react';
import {reportError} from '../lib/telemetry';
export default class ErrorBoundary extends Component<{children: ReactNode}, {failed: boolean}> {
  state = {failed: false};
  static getDerivedStateFromError() {return {failed: true};}
  componentDidCatch() {reportError('UI_ERROR');}
  render() {return this.state.failed ? <main className="recovery-screen" role="alert"><h1>This view could not be displayed.</h1><p>Reload to return to the built-in demo. Temporary server models expire automatically.</p><button className="secondary-button" onClick={() => location.reload()}>Reload demo</button><a href="/tiny-model.json" download>Download an example model</a></main> : this.props.children;}
}
