// Step 2: training happens in Python (on your laptop or in Google Colab). This page explains how.
import { balanceReport, dataset } from '../lib/dataset.js';
import { useStore } from '../lib/store.js';

const COLAB_URL = 'https://colab.research.google.com/github/devsaraf20/gesture-studio/blob/main/training/gesture_studio_training.ipynb';

export default function TrainPage() {
  const state = useStore(dataset);
  return (
    <div className="prose">
      <h1>Train your model</h1>
      <p className="lead">
        Training means showing the computer your examples until it finds the pattern that separates “fist” from “peace”. It happens in Python because that is where
        the machine learning tools live. It takes about a minute.
      </p>

      <div className="cards">
        <div className="card">
          <h3>1 · Check your data</h3>
          {['pose', 'motion'].map((type) => {
            const notes = balanceReport(state, type);
            if (!notes.length) return null;
            return (
              <div key={type}>
                <h4>{type === 'pose' ? 'Poses' : 'Motions'}</h4>
                {notes.map((n, i) => (
                  <p key={i} className={`note note-${n.level}`}>
                    {n.text}
                  </p>
                ))}
              </div>
            );
          })}
          <p>
            Then go back to <a href="#collect">Collect</a> and click <b>Download poses CSV</b> (and/or motions). Save it into the <code>training</code> folder.
          </p>
        </div>

        <div className="card">
          <h3>2a · Train on your laptop (Windows)</h3>
          <p>Open PowerShell in the project folder, then:</p>
          <pre>
            {`cd training
.venv\\Scripts\\Activate.ps1
python train.py gesture_poses.csv`}
          </pre>
          <p className="muted small">
            First time? Follow “Set up Python” in the README. For motions: <code>python train.py gesture_motions.csv</code>
          </p>
        </div>

        <div className="card">
          <h3>2b · …or in Google Colab</h3>
          <p>No installation needed, runs in your browser on Google's computers.</p>
          <p>
            <a className="btn btn-primary" href={COLAB_URL} target="_blank" rel="noreferrer">
              Open the notebook in Colab
            </a>
          </p>
          <p className="muted small">Upload your CSV when the notebook asks, run every cell, then download the zip it makes at the end.</p>
        </div>

        <div className="card">
          <h3>3 · Read the results</h3>
          <p>The script prints plain-English advice and saves charts in <code>training/exports/&lt;name&gt;/</code>:</p>
          <ul>
            <li>
              <b>training_curves.png</b> — loss and accuracy per epoch
            </li>
            <li>
              <b>confusion_matrices.png</b> — which gestures get mixed up
            </li>
            <li>
              <b>report.md</b> — everything it told you, saved
            </li>
          </ul>
          <p>
            If it says two gestures are confused, record more varied samples of both and train again. New to all this? Read <b>LEARN.md</b>.
          </p>
        </div>

        <div className="card">
          <h3>4 · Use it</h3>
          <p>
            The best model is copied into <code>web/public/models/</code> automatically. Open <a href="#live">Live</a>, pick it from the list and click <b>Load</b>.
          </p>
          <p className="muted small">
            From Colab: unzip the download, then on the Live page use “load from files” and select every file inside the <code>best</code> folder.
          </p>
        </div>
      </div>
    </div>
  );
}
