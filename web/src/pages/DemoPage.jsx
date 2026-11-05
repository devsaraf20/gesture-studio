// Step 4: a 3D playground controlled by your gestures.
import { useEffect, useRef, useState } from 'react';
import CameraView from '../components/CameraView.jsx';
import ModelPicker from '../components/ModelPicker.jsx';
import { DEMO_ACTIONS, DemoScene, guessAction } from '../lib/demoScene.js';
import { loadedModels, runtime, simulateGesture } from '../lib/runtime.js';
import { REST_CLASSES } from '../lib/smoother.js';
import { useStore } from '../lib/store.js';

const KEY = 'gesture-studio-demo-mapping';

function loadMapping() {
  try {
    return JSON.parse(localStorage.getItem(KEY) || '{}');
  } catch {
    return {};
  }
}

export default function DemoPage() {
  const holder = useRef(null);
  const sceneRef = useRef(null);
  const models = useStore(loadedModels);
  const [mapping, setMapping] = useState(loadMapping);
  const [last, setLast] = useState(null);
  const [followHand, setFollowHand] = useState(true);
  const mappingRef = useRef(mapping);
  mappingRef.current = mapping;
  const followRef = useRef(followHand);
  followRef.current = followHand;

  const gestures = [...new Set(models.flatMap(({ model }) => model.meta.classes))].filter((g) => !REST_CLASSES.has(g.toLowerCase()));
  const actionFor = (g) => mappingRef.current[g] ?? guessAction(g);

  useEffect(() => {
    const scene = new DemoScene(holder.current);
    sceneRef.current = scene;
    const offGesture = runtime.on('gesture', ({ gesture, state, simulated }) => {
      const action = simulated ? gesture : actionFor(gesture);
      if (state === 'start') setLast({ gesture: simulated ? 'test button' : gesture, action, at: Date.now() });
      run(scene, action, state);
    });
    const offHand = runtime.on('hand', (h) => scene.setHand(h.x, h.y, h.present && followRef.current));
    return () => {
      offGesture();
      offHand();
      scene.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function choose(g, action) {
    const next = { ...mapping, [g]: action };
    setMapping(next);
    localStorage.setItem(KEY, JSON.stringify(next));
  }

  return (
    <div className="demo">
      <div className="demo-stage" ref={holder}>
        {last && (
          <div className="demo-toast" key={last.at}>
            {last.gesture} → {DEMO_ACTIONS[last.action] ?? last.action}
          </div>
        )}
        <div className="demo-cam">
          <CameraView compact />
        </div>
      </div>
      <aside className="side">
        <div className="panel">
          <h3>Model</h3>
          {models.length ? null : <p className="muted small">Load a model trained on your gestures (or the sample one) to drive the scene.</p>}
          <ModelPicker />
        </div>
        <div className="panel">
          <h3>Gesture → action</h3>
          {!gestures.length && <p className="muted small">Your gestures will appear here once a model is loaded.</p>}
          {gestures.map((g) => (
            <label key={g} className="map-row">
              <span className="g-name">{g}</span>
              <select value={mapping[g] ?? guessAction(g)} onChange={(e) => choose(g, e.target.value)}>
                {Object.entries(DEMO_ACTIONS).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
          ))}
          <label className="check">
            <input type="checkbox" checked={followHand} onChange={(e) => setFollowHand(e.target.checked)} />
            Object leans towards your hand
          </label>
        </div>
        <div className="panel">
          <h3>Try the actions</h3>
          <p className="muted small">No camera needed — these buttons do what a gesture would.</p>
          <div className="button-grid">
            <button className="btn" onClick={() => simulateGesture('colour')}>change colour</button>
            <button className="btn" onClick={() => simulateGesture('burst')}>particle burst</button>
            <button className="btn" onClick={() => simulateGesture('rotate_left')}>rotate left</button>
            <button className="btn" onClick={() => simulateGesture('rotate_right')}>rotate right</button>
            <button
              className="btn"
              onPointerDown={() => simulateGesture('spin', 'start')}
              onPointerUp={() => simulateGesture('spin', 'end')}
              onPointerLeave={() => simulateGesture('spin', 'end')}
            >
              spin (hold)
            </button>
          </div>
        </div>
      </aside>
    </div>
  );
}

function run(scene, action, state) {
  if (action === 'spin') return scene.setSpinning(state === 'start');
  if (state !== 'start') return;
  if (action === 'colour') scene.nextColour();
  else if (action === 'burst') scene.burst();
  else if (action === 'rotate_left') scene.rotateBy(-90);
  else if (action === 'rotate_right') scene.rotateBy(90);
}
