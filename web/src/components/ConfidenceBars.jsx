// One bar per gesture showing how sure the model is right now.
//   filled bar = smoothed confidence, thin white tick = this frame's raw confidence,
//   dashed line = your threshold. The triggered gesture lights up.
import { useEffect, useState } from 'react';
import { REST_CLASSES } from '../lib/smoother.js';
import { liveSettings, runtime } from '../lib/runtime.js';
import { useStore } from '../lib/store.js';

export default function ConfidenceBars({ model }) {
  const [pred, setPred] = useState(null);
  const { threshold } = useStore(liveSettings);

  useEffect(
    () =>
      runtime.on('prediction', (p) => {
        if (p.modelId !== model.id) return;
        setPred({ smoothed: Array.from(p.smoothed), raw: p.raw ? Array.from(p.raw) : null, active: p.active, handVisible: p.handVisible });
      }),
    [model.id],
  );

  const waiting = model.meta.mode === 'motion' && pred?.handVisible && !pred?.raw;
  return (
    <div className="bars">
      <div className="bars-head">
        <strong>{model.meta.name}</strong>
        <span className="muted small">
          {!pred?.handVisible ? 'no hand in view' : waiting ? 'watching movement…' : model.meta.mode === 'motion' ? 'move your hand' : 'hold a pose'}
        </span>
      </div>
      {model.meta.classes.map((name, i) => {
        const v = pred?.smoothed[i] ?? 0;
        const r = pred?.raw?.[i];
        const active = pred?.active === name;
        const rest = REST_CLASSES.has(name.toLowerCase());
        return (
          <div key={name} className={`bar-row ${active ? 'active' : ''} ${rest ? 'rest' : ''}`}>
            <span className="bar-label">{name}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${v * 100}%` }} />
              {r != null && <div className="bar-raw" style={{ left: `${r * 100}%` }} />}
              <div className="bar-threshold" style={{ left: `${threshold * 100}%` }} />
            </div>
            <span className="bar-value">{Math.round(v * 100)}%</span>
          </div>
        );
      })}
    </div>
  );
}
