// Load trained models: from web/public/models (written by train.py) or from files on disk.
import { useEffect, useState } from 'react';
import { listBundledModels, loadModelFromFiles, loadModelFromUrl } from '../lib/models.js';
import { addModel, loadedModels, removeModel } from '../lib/runtime.js';
import { useStore } from '../lib/store.js';

export default function ModelPicker() {
  const [bundled, setBundled] = useState([]);
  const [choice, setChoice] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const loaded = useStore(loadedModels);

  useEffect(() => {
    listBundledModels().then((list) => {
      setBundled(list);
      if (list.length) setChoice(list[0].folder);
    });
  }, []);

  async function run(fn) {
    setBusy(true);
    setError('');
    try {
      const model = await fn();
      // replace a model of the same mode (one pose model + one motion model at a time)
      loadedModels.get().filter((m) => m.model.meta.mode === model.meta.mode).forEach((m) => removeModel(m.model.id));
      addModel(model);
    } catch (e) {
      console.error(e);
      setError(e.message);
    }
    setBusy(false);
  }

  const loadedFolders = new Set(loaded.map((m) => m.model.meta.name));

  return (
    <div className="model-picker">
      <div className="row">
        <select value={choice} onChange={(e) => setChoice(e.target.value)} disabled={!bundled.length}>
          {!bundled.length && <option>No models in web/public/models yet</option>}
          {bundled.map((m) => (
            <option key={m.folder} value={m.folder}>
              {m.name} · {m.mode} · {m.model_type.replace('_', ' ')} · {Math.round(m.test_accuracy * 100)}%
            </option>
          ))}
        </select>
        <button
          className="btn btn-primary"
          disabled={!choice || busy}
          onClick={() => run(() => loadModelFromUrl(`${import.meta.env.BASE_URL}models/${choice}/`))}
        >
          {loadedFolders.has(choice) ? 'Reload' : 'Load'}
        </button>
      </div>
      <label className="file-btn">
        <input
          type="file"
          multiple
          accept=".json,.bin"
          onChange={(e) => {
            const files = e.target.files;
            if (files?.length) run(() => loadModelFromFiles(files));
            e.target.value = '';
          }}
        />
        or load from files… <span className="muted">(select every file in the model folder)</span>
      </label>
      {error && <p className="error">{error}</p>}
      {loaded.map(({ model }) => (
        <div key={model.id} className="loaded-model">
          <div>
            <strong>{model.meta.name}</strong>{' '}
            <span className="tag">{model.meta.mode === 'motion' ? 'motion' : 'pose'}</span>{' '}
            <span className="tag">{model.meta.model_type.replace('_', ' ')}</span>
            <div className="muted small">{model.meta.classes.join(' · ')}</div>
          </div>
          <button className="btn btn-ghost" onClick={() => removeModel(model.id)} title="Unload">
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}
