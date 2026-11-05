// Step 1: record examples of each gesture.
import { useEffect, useMemo, useRef, useState } from 'react';
import CameraView from '../components/CameraView.jsx';
import SampleBrowser from '../components/SampleBrowser.jsx';
import {
  addClass,
  addSample,
  balanceReport,
  clearClass,
  countsByClass,
  dataset,
  deleteClass,
  downloadText,
  exportCSV,
  importCSV,
  newTake,
} from '../lib/dataset.js';
import { DEFAULT_SEQ_LEN, motionFeatures, normalizeHand } from '../lib/features.js';
import { onFrame } from '../lib/handTracker.js';
import { useStore } from '../lib/store.js';

const POSE_RATE = 10; // samples per second while recording a pose
const MIN_MOTION_MS = 250;

export default function CollectPage() {
  const state = useStore(dataset);
  const [selectedId, setSelectedId] = useState(null);
  const [recording, setRecording] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [notice, setNotice] = useState('');
  const [handVisible, setHandVisible] = useState(false);
  const [takeCount, setTakeCount] = useState(0);

  const selected = state.classes.find((c) => c.id === selectedId) || state.classes[0];
  const counts = useMemo(() => countsByClass(state), [state]);

  // Recording lives in refs so the per-frame callback always sees the latest values.
  const rec = useRef({ on: false, take: 0, last: 0, frames: [], classId: null, type: 'pose', count: 0 });

  useEffect(
    () =>
      onFrame((frame) => {
        setHandVisible(!!frame.hand);
        const r = rec.current;
        if (!r.on || !frame.hand) return;
        const { landmarks, isLeft } = frame.hand;
        if (r.type === 'pose') {
          if (frame.t - r.last < 1000 / POSE_RATE) return;
          r.last = frame.t;
          addSample(r.classId, r.take, normalizeHand(landmarks, frame.width, frame.height, isLeft));
          r.count++;
          setTakeCount(r.count);
        } else {
          r.frames.push({ t: frame.t, landmarks, width: frame.width, height: frame.height, isLeft });
          setTakeCount(r.frames.length);
        }
      }),
    [],
  );

  function start() {
    if (!selected || rec.current.on) return;
    rec.current = { on: true, take: newTake(), last: 0, frames: [], classId: selected.id, type: selected.type, count: 0 };
    setTakeCount(0);
    setNotice('');
    setRecording(true);
  }

  function stop() {
    const r = rec.current;
    if (!r.on) return;
    r.on = false;
    setRecording(false);
    if (r.type === 'motion') {
      const duration = r.frames.length ? r.frames[r.frames.length - 1].t - r.frames[0].t : 0;
      if (r.frames.length < 5 || duration < MIN_MOTION_MS) {
        setNotice('That motion was too short or your hand was not visible. Hold the button for the whole movement.');
        return;
      }
      addSample(r.classId, r.take, motionFeatures(r.frames, DEFAULT_SEQ_LEN), duration);
    } else if (r.count === 0) {
      setNotice('No hand was visible, so nothing was recorded.');
    }
  }

  function timed() {
    if (!selected || countdown) return;
    let n = 3;
    setCountdown(n);
    const tick = setInterval(() => {
      n -= 1;
      if (n > 0) return setCountdown(n);
      clearInterval(tick);
      setCountdown(0);
      start();
      setTimeout(stop, selected.type === 'pose' ? 3000 : 1200);
    }, 700);
  }

  // Hold SPACE to record (handy when your hand is busy making the gesture)
  const startRef = useRef(start);
  const stopRef = useRef(stop);
  startRef.current = start;
  stopRef.current = stop;
  useEffect(() => {
    const isTyping = (e) => ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName);
    const down = (e) => {
      if (e.code !== 'Space' || e.repeat || isTyping(e)) return;
      e.preventDefault();
      startRef.current();
    };
    const up = (e) => {
      if (e.code !== 'Space' || isTyping(e)) return;
      e.preventDefault();
      stopRef.current();
    };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
    };
  }, []);

  const drawTrail = (ctx, frame, w, h) => {
    const r = rec.current;
    if (!r.on || r.type !== 'motion' || r.frames.length < 2) return;
    ctx.strokeStyle = '#ff8a5b';
    ctx.lineWidth = 4;
    ctx.beginPath();
    r.frames.forEach((f, i) => {
      const x = (1 - f.landmarks[0].x) * w;
      const y = f.landmarks[0].y * h;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  };

  if (!state.loaded) return <p className="muted">Loading your dataset…</p>;

  return (
    <div className="layout-2col">
      <section>
        <CameraView onDraw={drawTrail}>
          {recording && (
            <div className="rec-badge">
              ● REC {selected?.name} · {selected?.type === 'pose' ? `${takeCount} samples` : `${takeCount} frames`}
              {!handVisible && <span className="rec-warn"> · no hand!</span>}
            </div>
          )}
          {countdown > 0 && <div className="countdown">{countdown}</div>}
          {!recording && !handVisible && <div className="hint-badge">Show your hand to the camera</div>}
        </CameraView>

        <div className="record-bar">
          <button
            className={`btn btn-record ${recording ? 'on' : ''}`}
            disabled={!selected}
            onPointerDown={start}
            onPointerUp={stop}
            onPointerLeave={stop}
          >
            {recording ? 'Recording… release to stop' : `Hold to record "${selected?.name ?? '…'}"`}
          </button>
          <button className="btn" onClick={timed} disabled={!selected || recording || countdown > 0}>
            3·2·1 timed
          </button>
          <span className="muted small">
            or hold <kbd>Space</kbd>.{' '}
            {selected?.type === 'motion'
              ? 'Motion: one press = one sample. Press, do the whole movement, release.'
              : `Pose: records ${POSE_RATE} samples per second while held. Move your hand around a little.`}
          </span>
        </div>
        {notice && <p className="error">{notice}</p>}

        {selected && <SampleBrowser cls={selected} />}
      </section>

      <aside className="side">
        <ClassList title="Poses" subtitle="a still hand shape" type="pose" state={state} counts={counts} selected={selected} onSelect={setSelectedId} />
        <ClassList title="Motions" subtitle="a movement, e.g. a swipe" type="motion" state={state} counts={counts} selected={selected} onSelect={setSelectedId} />
        <AddClassForm onAdded={(c) => setSelectedId(c.id)} />
        <ExportPanel state={state} />
      </aside>
    </div>
  );
}

function ClassList({ title, subtitle, type, state, counts, selected, onSelect }) {
  const classes = state.classes.filter((c) => c.type === type);
  const max = Math.max(1, ...classes.map((c) => counts[c.id] || 0));
  const notes = balanceReport(state, type);
  return (
    <div className="panel">
      <h3>
        {title} <span className="muted small">— {subtitle}</span>
      </h3>
      {!classes.length && <p className="muted small">None yet. Add one below.</p>}
      {classes.map((c) => (
        <div key={c.id} className={`class-row ${selected?.id === c.id ? 'selected' : ''}`} onClick={() => onSelect(c.id)}>
          <span className="dot" style={{ background: c.colour }} />
          <span className="class-name">{c.name}</span>
          <div className="count-track">
            <div className="count-fill" style={{ width: `${((counts[c.id] || 0) / max) * 100}%`, background: c.colour }} />
          </div>
          <span className="count">{counts[c.id] || 0}</span>
          <details className="menu" onClick={(e) => e.stopPropagation()}>
            <summary>⋯</summary>
            <div className="menu-items">
              <button onClick={() => confirm(`Delete all ${counts[c.id] || 0} samples of "${c.name}"?`) && clearClass(c.id)}>Clear samples</button>
              <button onClick={() => confirm(`Delete gesture "${c.name}" and its samples?`) && deleteClass(c.id)}>Delete gesture</button>
            </div>
          </details>
        </div>
      ))}
      {notes.map((n, i) => (
        <p key={i} className={`note note-${n.level}`}>
          {n.level === 'warn' ? '⚠ ' : n.level === 'ok' ? '✓ ' : 'ℹ '}
          {n.text}
        </p>
      ))}
    </div>
  );
}

function AddClassForm({ onAdded }) {
  const [name, setName] = useState('');
  const [type, setType] = useState('pose');
  const [error, setError] = useState('');
  function submit(e) {
    e.preventDefault();
    const res = addClass(name, type);
    if (res.error) return setError(res.error);
    setError('');
    setName('');
    onAdded(res.class);
  }
  return (
    <form className="panel" onSubmit={submit}>
      <h3>Add a gesture</h3>
      <div className="row">
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. thumbs_up" />
        <select value={type} onChange={(e) => setType(e.target.value)}>
          <option value="pose">pose</option>
          <option value="motion">motion</option>
        </select>
        <button className="btn btn-primary">Add</button>
      </div>
      {error && <p className="error">{error}</p>}
      <p className="muted small">Tip: include a "none" pose (relaxed, random hands) so the model can say "nothing".</p>
    </form>
  );
}

function ExportPanel({ state }) {
  const [msg, setMsg] = useState('');
  const typeOf = Object.fromEntries(state.classes.map((c) => [c.id, c.type]));
  const rows = { pose: 0, motion: 0 };
  for (const x of state.samples) if (typeOf[x.classId]) rows[typeOf[x.classId]]++;
  const download = (type, filename) => downloadText(exportCSV(dataset.get(), type).csv, filename);
  return (
    <div className="panel">
      <h3>Export for training</h3>
      <div className="col">
        <button className="btn btn-primary" disabled={!rows.pose} onClick={() => download('pose', 'gesture_poses.csv')}>
          Download poses CSV ({rows.pose})
        </button>
        <button className="btn btn-primary" disabled={!rows.motion} onClick={() => download('motion', 'gesture_motions.csv')}>
          Download motions CSV ({rows.motion})
        </button>
        <label className="file-btn">
          <input
            type="file"
            accept=".csv"
            onChange={async (e) => {
              const file = e.target.files?.[0];
              e.target.value = '';
              if (!file) return;
              try {
                const { added, type } = importCSV(await file.text());
                setMsg(`Imported ${added} ${type} samples.`);
              } catch (err) {
                setMsg(err.message);
              }
            }}
          />
          Import a CSV…
        </label>
        <button
          className="btn btn-ghost danger"
          onClick={() => confirm('Delete ALL gestures and samples? Export first if you want to keep them.') && dataset.set((s) => ({ ...s, classes: [], samples: [] }))}
        >
          Clear everything
        </button>
      </div>
      {msg && <p className="muted small">{msg}</p>}
      <p className="muted small">Your samples are saved in this browser automatically. Next: the Train tab.</p>
    </div>
  );
}
