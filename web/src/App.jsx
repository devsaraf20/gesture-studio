import { useEffect, useState } from 'react';
import { loadDataset } from './lib/dataset.js';
import { trackerStatus } from './lib/handTracker.js';
import './lib/outputs.js'; // starts the bridge connection (if enabled)
import { useStore } from './lib/store.js';
import CollectPage from './pages/CollectPage.jsx';
import DemoPage from './pages/DemoPage.jsx';
import LivePage from './pages/LivePage.jsx';
import TrainPage from './pages/TrainPage.jsx';

const PAGES = [
  { id: 'collect', label: 'Collect', hint: 'record examples', Component: CollectPage },
  { id: 'train', label: 'Train', hint: 'teach the model', Component: TrainPage },
  { id: 'live', label: 'Live', hint: 'use it', Component: LivePage },
  { id: 'demo', label: 'Demo', hint: '3D playground', Component: DemoPage },
];

function pageFromHash() {
  const id = window.location.hash.replace('#', '');
  return PAGES.some((p) => p.id === id) ? id : 'collect';
}

export default function App() {
  const [page, setPage] = useState(pageFromHash);
  const status = useStore(trackerStatus);

  useEffect(() => {
    loadDataset();
    const onHash = () => setPage(pageFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const { Component } = PAGES.find((p) => p.id === page);
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">✋</span>
          <span>Gesture Studio</span>
        </div>
        <nav className="steps">
          {PAGES.map((p, i) => (
            <a key={p.id} href={`#${p.id}`} className={p.id === page ? 'step active' : 'step'}>
              <span className="step-num">{i + 1}</span>
              <span className="step-label">{p.label}</span>
              <span className="step-hint">{p.hint}</span>
            </a>
          ))}
        </nav>
        <div className={`pill pill-${status.state}`}>
          {status.state === 'running' ? `camera on · ${status.fps} fps` : status.state === 'loading' ? 'starting…' : status.state === 'error' ? 'camera error' : 'camera off'}
        </div>
      </header>
      <main className="page">
        <Component />
      </main>
    </div>
  );
}
