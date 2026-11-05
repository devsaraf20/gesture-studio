// Step 3: run your trained model on the live webcam and send gestures to other software.
import { useEffect, useState } from 'react';
import CameraView from '../components/CameraView.jsx';
import ConfidenceBars from '../components/ConfidenceBars.jsx';
import ModelPicker from '../components/ModelPicker.jsx';
import { bridgeStatus, mappingFor, messageLog, outputSettings, sendTest, setMapping } from '../lib/outputs.js';
import { DEFAULT_SETTINGS, liveSettings, loadedModels, runtime } from '../lib/runtime.js';
import { REST_CLASSES } from '../lib/smoother.js';
import { useStore } from '../lib/store.js';

export default function LivePage() {
  const models = useStore(loadedModels);
  const [current, setCurrent] = useState(null);

  useEffect(
    () =>
      runtime.on('gesture', (ev) => {
        if (ev.state === 'start') setCurrent({ ...ev, at: performance.now() });
        else setCurrent((c) => (c?.gesture === ev.gesture ? null : c));
      }),
    [],
  );

  return (
    <div className="layout-2col">
      <section>
        <CameraView>
          {current && (
            <div className="gesture-banner" key={current.at}>
              {current.gesture}
              <span>{Math.round(current.confidence * 100)}%</span>
            </div>
          )}
        </CameraView>
        {!models.length && <p className="note note-info">Load a model on the right to start recognising gestures.</p>}
        <div className="bars-grid">
          {models.map(({ model }) => (
            <ConfidenceBars key={model.id} model={model} />
          ))}
        </div>
        <Settings />
      </section>
      <aside className="side">
        <div className="panel">
          <h3>Model</h3>
          <ModelPicker />
          <p className="muted small">You can load one pose model and one motion model at the same time.</p>
        </div>
        <Outputs models={models} />
      </aside>
    </div>
  );
}

function Slider({ label, help, value, min, max, step, format, onChange }) {
  return (
    <label className="slider">
      <div className="slider-head">
        <span>{label}</span>
        <b>{format(value)}</b>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="muted small">{help}</span>
    </label>
  );
}

function Settings() {
  const s = useStore(liveSettings);
  const set = (k) => (v) => liveSettings.set((x) => ({ ...x, [k]: v }));
  const ms = (v) => `${v} ms`;
  return (
    <div className="panel settings">
      <h3>
        Stability controls{' '}
        <button className="btn btn-ghost small" onClick={() => liveSettings.set({ ...DEFAULT_SETTINGS })}>
          reset
        </button>
      </h3>
      <div className="settings-grid">
        <Slider label="Confidence threshold" value={s.threshold} min={0.3} max={0.99} step={0.01} format={(v) => `${Math.round(v * 100)}%`} onChange={set('threshold')}
          help="How sure the model must be before a gesture counts. Higher = fewer accidental triggers." />
        <Slider label="Smoothing" value={s.smoothing} min={0} max={0.95} step={0.05} format={(v) => `${Math.round(v * 100)}%`} onChange={set('smoothing')}
          help="Averages recent frames so flickers are ignored. Higher = steadier but slower to react." />
        <Slider label="Hold time" value={s.holdMs} min={0} max={1000} step={10} format={ms} onChange={set('holdMs')}
          help="A pose must stay on top this long before it triggers." />
        <Slider label="Cooldown (poses)" value={s.cooldownMs} min={0} max={2000} step={50} format={ms} onChange={set('cooldownMs')}
          help="Minimum time between two pose triggers." />
        <Slider label="Cooldown (motions)" value={s.motionCooldownMs} min={200} max={3000} step={50} format={ms} onChange={set('motionCooldownMs')}
          help="Minimum time between two swipes, so one swipe fires once." />
      </div>
    </div>
  );
}

function Outputs({ models }) {
  const s = useStore(outputSettings);
  const status = useStore(bridgeStatus);
  const log = useStore(messageLog);
  const gestures = [...new Set(models.flatMap(({ model }) => model.meta.classes))].filter((g) => !REST_CLASSES.has(g.toLowerCase()));
  const update = (patch) => outputSettings.set((x) => ({ ...x, ...patch }));

  return (
    <div className="panel">
      <h3>Send to TouchDesigner / Unreal / Physical Bridge</h3>
      <label className="check">
        <input type="checkbox" checked={s.enabled} onChange={(e) => update({ enabled: e.target.checked })} />
        Send to bridge
      </label>
      <div className="row">
        <input value={s.url} onChange={(e) => update({ url: e.target.value })} spellCheck={false} />
      </div>
      <p className={`small status-${status.state}`}>{status.state === 'off' ? 'Start the bridge with “npm run bridge” (or use “npm start”), then tick the box.' : status.message}</p>
      <label className="check">
        <input type="checkbox" checked={s.sendHand} onChange={(e) => update({ sendHand: e.target.checked })} />
        Also stream hand position (/hand/x, /hand/y, /hand/size)
      </label>

      <h4>Gesture → OSC address</h4>
      {!gestures.length && <p className="muted small">Load a model to see its gestures here.</p>}
      <table className="mapping">
        <tbody>
          {gestures.map((g) => {
            const m = mappingFor(g);
            return (
              <tr key={g}>
                <td>
                  <input type="checkbox" checked={m.enabled} onChange={(e) => setMapping(g, { enabled: e.target.checked })} title="send this gesture" />
                </td>
                <td className="g-name">{g}</td>
                <td>
                  <input value={m.address} onChange={(e) => setMapping(g, { address: e.target.value })} spellCheck={false} />
                </td>
                <td>
                  <button className="btn btn-ghost small" onClick={() => sendTest(g)} disabled={!s.enabled}>
                    test
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted small">
        Each gesture sends <code>1</code> when it starts and <code>0</code> when it ends, plus <code>/gesture/current</code> with its name. WebSocket clients get the same as
        JSON. See the README for the full message format.
      </p>

      <h4>Sent messages</h4>
      <div className="log">
        {!log.length && <span className="muted small">Nothing sent yet.</span>}
        {log.map((l, i) => (
          <div key={i} className={l.sent ? '' : 'unsent'}>
            <span className="muted">{l.time}</span> {l.msg.gesture} {l.msg.state} {l.sent ? '' : '(not connected)'}
          </div>
        ))}
      </div>
    </div>
  );
}
