// Loading trained models in the browser and running them ("inference").
//
// Two kinds, both produced by training/train.py:
//   neural_net    -> TensorFlow.js files: model.json + weights.bin
//   random_forest -> forest.json (a list of decision trees, evaluated right here)
// Each comes with gesture_meta.json describing the gestures and input format.
import * as tf from '@tensorflow/tfjs';
import { FEATURE_VERSION } from './features.js';

let backendReady = null;
function ensureBackend() {
  // Our models are tiny, so the CPU backend is fast and avoids fighting MediaPipe for the GPU.
  backendReady ??= tf.setBackend('cpu').then(() => tf.ready());
  return backendReady;
}

/** Run every tree and average their votes. */
function forestPredict(forest, x) {
  const probs = new Float32Array(forest.n_classes);
  for (const t of forest.trees) {
    let n = 0;
    while (t.left[n] !== -1) n = x[t.feature[n]] <= t.threshold[n] ? t.left[n] : t.right[n];
    const v = t.value[n];
    for (let c = 0; c < v.length; c++) probs[c] += v[c];
  }
  for (let c = 0; c < probs.length; c++) probs[c] /= forest.trees.length;
  return probs;
}

function checkMeta(meta) {
  if (!meta?.classes?.length) throw new Error('gesture_meta.json is missing the list of gestures.');
  if (meta.feature_version && meta.feature_version !== FEATURE_VERSION)
    throw new Error(`This model was trained with feature version ${meta.feature_version}, the app uses ${FEATURE_VERSION}.`);
}

let nextId = 1;
function wrap(meta, predict, dispose = () => {}) {
  return { id: `m${nextId++}`, meta, predict, dispose };
}

async function fromTfjs(meta, loadArg) {
  await ensureBackend();
  const model = await tf.loadLayersModel(loadArg);
  const inputSize = model.inputs[0].shape[1];
  if (inputSize !== meta.input_size) throw new Error(`Model expects ${inputSize} inputs but gesture_meta.json says ${meta.input_size}.`);
  return wrap(
    meta,
    (features) => tf.tidy(() => model.predict(tf.tensor2d([features], [1, inputSize])).dataSync()),
    () => model.dispose(),
  );
}

/** Load from a folder served by the app, e.g. /models/sample_poses/ */
export async function loadModelFromUrl(folderUrl) {
  const base = folderUrl.endsWith('/') ? folderUrl : `${folderUrl}/`;
  const meta = await (await fetch(`${base}gesture_meta.json`)).json();
  checkMeta(meta);
  if (meta.model_type === 'random_forest') {
    const forest = await (await fetch(`${base}forest.json`)).json();
    return wrap(meta, (x) => forestPredict(forest, x));
  }
  return fromTfjs(meta, `${base}model.json`);
}

/** Load from files the user picked (e.g. downloaded from Google Colab). */
export async function loadModelFromFiles(fileList) {
  const files = Array.from(fileList);
  const find = (name) => files.find((f) => f.name === name);
  const metaFile = find('gesture_meta.json');
  if (!metaFile) throw new Error('Please include gesture_meta.json (select all files in the model folder).');
  const meta = JSON.parse(await metaFile.text());
  checkMeta(meta);
  if (meta.model_type === 'random_forest') {
    const forestFile = find('forest.json');
    if (!forestFile) throw new Error('This is a random forest model: please also select forest.json.');
    const forest = JSON.parse(await forestFile.text());
    return wrap(meta, (x) => forestPredict(forest, x));
  }
  const json = find('model.json');
  const bins = files.filter((f) => f.name.endsWith('.bin'));
  if (!json || !bins.length) throw new Error('This is a neural network model: please select model.json and weights.bin too.');
  return fromTfjs(meta, tf.io.browserFiles([json, ...bins]));
}

/** List of models in web/public/models (written by train.py). */
export async function listBundledModels() {
  try {
    const r = await fetch(`${import.meta.env.BASE_URL}models/index.json`, { cache: 'no-store' });
    if (!r.ok) return [];
    return (await r.json()).models || [];
  } catch {
    return [];
  }
}

export { forestPredict };
