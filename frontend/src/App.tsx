import {useEffect, useState} from 'react';
import Visualizer from './Visualizer';
import Playground from './components/Playground';
import type {PlaygroundHandoff} from './types';

function currentPage() {return location.hash === '#/playground' ? 'playground' : 'visualizer';}
export default function App() {
  const [page, setPage] = useState(currentPage);
  useEffect(() => {document.title = `NeuralScope · ${page === 'visualizer' ? 'Visualizer' : 'Playground'}`;}, [page]);
  const [handoff, setHandoff] = useState<PlaygroundHandoff | null>(null);
  useEffect(() => {
    const update = () => setPage(currentPage());
    window.addEventListener('hashchange', update);
    if (!location.hash) history.replaceState(null, '', '#/visualizer');
    return () => window.removeEventListener('hashchange', update);
  }, []);
  function inspect(next: PlaygroundHandoff) {setHandoff(next); location.hash = '/visualizer';}
  return <div className="app-shell">
    <header className="topbar"><a className="brand" href="#/visualizer">NeuralScope</a><span className="section-title">{page === 'visualizer' ? 'Visualizer' : 'Playground'}</span><nav aria-label="Main navigation"><a href="#/visualizer" aria-current={page === 'visualizer' ? 'page' : undefined}>Visualizer</a><a href="#/playground" aria-current={page === 'playground' ? 'page' : undefined}>Playground</a></nav></header>
    <div hidden={page !== 'playground'}><Playground active={page === 'playground'} inspect={inspect}/></div>
    <div hidden={page !== 'visualizer'}><Visualizer handoff={handoff} active={page === 'visualizer'}/></div>
  </div>;
}
