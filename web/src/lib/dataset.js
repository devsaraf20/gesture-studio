// The dataset you build on the Collect page: gesture classes + recorded samples.
// Saved automatically in the browser (IndexedDB), so a page refresh doesn't lose your work.
import { get, set } from 'idb-keyval';
import { DEFAULT_SEQ_LEN, motionColumns, poseColumns } from './features.js';
import { createStore } from './store.js';

const DB_KEY = 'gesture-studio-dataset-v1';
const PALETTE = ['#4cc9f0', '#ff8a5b', '#3ddc97', '#ffd23f', '#b388ff', '#ff5d8f', '#9ef01a', '#f7a072'];

/**
 * classes: [{ id, name, type: 'pose' | 'motion', colour }]
 * samples: [{ id, classId, take, t, f: number[] (features), d?: duration ms (motion) }]
 */
export const dataset = createStore({ loaded: false, classes: [], samples: [], nextTake: 1, nextId: 1 });

let saveTimer = 0;
dataset.subscribe(() => {
  if (!dataset.get().loaded) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    const { classes, samples, nextTake, nextId } = dataset.get();
    set(DB_KEY, { classes, samples, nextTake, nextId }).catch((e) => console.error('save failed', e));
  }, 400);
});

export async function loadDataset() {
  try {
    const saved = await get(DB_KEY);
    if (saved) {
      dataset.set({ ...saved, loaded: true });
      return;
    }
  } catch (e) {
    console.error('could not load saved dataset', e);
  }
  // first visit: start with the example gestures from the README
  dataset.set({ loaded: true, classes: [], samples: [], nextTake: 1, nextId: 1 });
  for (const [name, type] of [['fist', 'pose'], ['peace', 'pose'], ['point', 'pose'], ['none', 'pose'], ['swipe_left', 'motion'], ['swipe_right', 'motion'], ['idle', 'motion']]) {
    addClass(name, type);
  }
}

export function cleanName(name) {
  return name.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '');
}

export function addClass(rawName, type) {
  const name = cleanName(rawName);
  if (!name) return { error: 'Give the gesture a name.' };
  if (dataset.get().classes.some((c) => c.name === name)) return { error: `"${name}" already exists.` };
  let created;
  dataset.set((s) => {
    created = { id: `c${s.nextId}`, name, type, colour: PALETTE[s.classes.length % PALETTE.length] };
    return { ...s, classes: [...s.classes, created], nextId: s.nextId + 1 };
  });
  return { class: created };
}

export function renameClass(id, rawName) {
  const name = cleanName(rawName);
  if (!name || dataset.get().classes.some((c) => c.name === name && c.id !== id)) return false;
  dataset.set((s) => ({ ...s, classes: s.classes.map((c) => (c.id === id ? { ...c, name } : c)) }));
  return true;
}

export function deleteClass(id) {
  dataset.set((s) => ({
    ...s,
    classes: s.classes.filter((c) => c.id !== id),
    samples: s.samples.filter((x) => x.classId !== id),
  }));
}

export function clearClass(id) {
  dataset.set((s) => ({ ...s, samples: s.samples.filter((x) => x.classId !== id) }));
}

/** A "take" is one press of the record button. Training keeps takes together (see LEARN.md). */
export function newTake() {
  const take = dataset.get().nextTake;
  dataset.set((s) => ({ ...s, nextTake: s.nextTake + 1 }));
  return take;
}

const round = (v) => Math.round(v * 10000) / 10000;

export function addSample(classId, take, features, durationMs) {
  dataset.set((s) => ({
    ...s,
    nextId: s.nextId + 1,
    samples: [
      ...s.samples,
      { id: `s${s.nextId}`, classId, take, t: Date.now(), f: features.map(round), ...(durationMs != null ? { d: Math.round(durationMs) } : {}) },
    ],
  }));
}

export function deleteSample(id) {
  dataset.set((s) => ({ ...s, samples: s.samples.filter((x) => x.id !== id) }));
}

export function deleteTake(take) {
  dataset.set((s) => ({ ...s, samples: s.samples.filter((x) => x.take !== take) }));
}

export function countsByClass(state) {
  const counts = Object.fromEntries(state.classes.map((c) => [c.id, 0]));
  for (const x of state.samples) counts[x.classId] = (counts[x.classId] || 0) + 1;
  return counts;
}

/**
 * Plain-English warnings about the balance of the dataset for one type ('pose' or 'motion').
 * Returns [{ level: 'warn' | 'info' | 'ok', text }]
 */
export function balanceReport(state, type) {
  const classes = state.classes.filter((c) => c.type === type);
  if (!classes.length) return [];
  const counts = countsByClass(state);
  const n = classes.map((c) => counts[c.id] || 0);
  const takes = Object.fromEntries(classes.map((c) => [c.id, new Set()]));
  for (const x of state.samples) takes[x.classId]?.add(x.take);
  const notes = [];
  const target = type === 'pose' ? 100 : 30;
  const empty = classes.filter((c) => !counts[c.id]);
  if (classes.length < 2) notes.push({ level: 'info', text: 'You need at least 2 gestures of this type to train a model.' });
  if (empty.length) notes.push({ level: 'info', text: `No samples yet for: ${empty.map((c) => c.name).join(', ')}.` });
  const max = Math.max(...n);
  const min = Math.min(...n);
  if (min > 0 && max > 1.5 * min) {
    const small = classes[n.indexOf(min)].name;
    const big = classes[n.indexOf(max)].name;
    notes.push({
      level: 'warn',
      text: `Unbalanced: "${big}" has ${max} samples but "${small}" only ${min}. The model will lean towards "${big}". Record more "${small}" so every gesture is within ~1.5× of the others.`,
    });
  }
  const few = classes.filter((c) => counts[c.id] > 0 && counts[c.id] < target);
  if (few.length) notes.push({ level: 'info', text: `Aim for ${target}+ samples each. Still short: ${few.map((c) => `${c.name} (${counts[c.id]})`).join(', ')}.` });
  const fewTakes = classes.filter((c) => counts[c.id] > 0 && takes[c.id].size < 3);
  if (type === 'pose' && fewTakes.length)
    notes.push({ level: 'info', text: `Record in at least 3 separate takes (moving your hand, changing angle in between) for a fair test: ${fewTakes.map((c) => c.name).join(', ')}.` });
  if (!notes.length) notes.push({ level: 'ok', text: 'Looks balanced and big enough. Ready to export and train!' });
  return notes;
}

// ------------------------------------------------------------------ CSV
export function exportCSV(state, type) {
  const classes = state.classes.filter((c) => c.type === type);
  const byId = Object.fromEntries(classes.map((c) => [c.id, c.name]));
  const samples = state.samples.filter((x) => byId[x.classId]);
  const header = type === 'pose' ? ['label', 'take', ...poseColumns()] : ['label', 'take', 'duration_ms', ...motionColumns(DEFAULT_SEQ_LEN)];
  const lines = [header.join(',')];
  for (const x of samples) {
    const meta = type === 'pose' ? [byId[x.classId], x.take] : [byId[x.classId], x.take, x.d ?? ''];
    lines.push([...meta, ...x.f].join(','));
  }
  return { csv: lines.join('\n') + '\n', rows: samples.length };
}

export function downloadText(text, filename) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv' }));
  const a = Object.assign(document.createElement('a'), { href: url, download: filename });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Import a CSV exported earlier (e.g. to continue on another computer). Adds to what you have. */
export function importCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(',');
  const li = header.indexOf('label');
  const ti = header.indexOf('take');
  const di = header.indexOf('duration_ms');
  if (li < 0) throw new Error('This CSV has no "label" column.');
  const type = header.some((h) => /^f\d+_/.test(h)) ? 'motion' : 'pose';
  const featureIdx = header.map((h, i) => i).filter((i) => i !== li && i !== ti && i !== di);
  const takeOffset = dataset.get().nextTake;
  const rows = [];
  for (const line of lines.slice(1)) {
    const cols = line.split(',');
    if (cols.length !== header.length) continue;
    const name = cleanName(cols[li]);
    let cls = dataset.get().classes.find((c) => c.name === name);
    if (!cls) cls = addClass(name, type).class;
    if (!cls || cls.type !== type) continue;
    rows.push({
      classId: cls.id,
      take: takeOffset + (ti >= 0 ? Number(cols[ti]) || 0 : rows.length),
      f: featureIdx.map((i) => round(Number(cols[i]))),
      d: di >= 0 ? Math.round(Number(cols[di])) : undefined,
    });
  }
  // add everything in one go (one save, one re-render)
  dataset.set((s) => {
    let id = s.nextId;
    const added = rows.map((r) => ({ id: `s${id++}`, t: Date.now(), ...r }));
    const maxTake = Math.max(s.nextTake - 1, ...rows.map((r) => r.take));
    return { ...s, samples: [...s.samples, ...added], nextId: id, nextTake: maxTake + 1 };
  });
  const added = rows.length;
  return { added, type };
}
