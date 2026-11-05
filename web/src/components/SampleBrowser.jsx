// Shows the recorded samples of one gesture, grouped by take, so you can delete bad ones.
import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { dataset, deleteSample, deleteTake } from '../lib/dataset.js';
import { drawSampleThumbnail } from '../lib/drawing.js';
import { useStore } from '../lib/store.js';

const PER_TAKE = 24;

const Thumb = memo(function Thumb({ sample, type }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) drawSampleThumbnail(ref.current, sample.f, type);
  }, [sample, type]);
  return (
    <div className="thumb" title={`sample ${sample.id}${sample.d ? ` · ${sample.d} ms` : ''}`}>
      <canvas ref={ref} width={64} height={64} />
      <button className="thumb-x" onClick={() => deleteSample(sample.id)} title="Delete this sample">
        ✕
      </button>
    </div>
  );
});

function SampleBrowser({ cls }) {
  const samples = useStore(dataset, (s) => s.samples);
  const [showTakes, setShowTakes] = useState(8);
  const [expanded, setExpanded] = useState({});

  const takes = useMemo(() => {
    const map = new Map();
    for (const x of samples) {
      if (x.classId !== cls.id) continue;
      if (!map.has(x.take)) map.set(x.take, []);
      map.get(x.take).push(x);
    }
    return [...map.entries()].sort((a, b) => b[0] - a[0]);
  }, [samples, cls.id]);

  const total = takes.reduce((n, [, xs]) => n + xs.length, 0);

  return (
    <div className="panel samples">
      <h3>
        Samples of “{cls.name}” <span className="muted small">— {total} in {takes.length} takes. Hover a sample and click ✕ to delete it.</span>
      </h3>
      {!takes.length && <p className="muted small">Nothing recorded yet.</p>}
      {takes.slice(0, showTakes).map(([take, xs]) => {
        const open = expanded[take];
        const visible = open ? xs : xs.slice(0, PER_TAKE);
        return (
          <div key={take} className="take">
            <div className="take-head">
              <span>
                Take {take} · {xs.length} sample{xs.length > 1 ? 's' : ''}
              </span>
              <button className="btn btn-ghost small" onClick={() => deleteTake(take)}>
                Delete take
              </button>
            </div>
            <div className="thumbs">
              {visible.map((x) => (
                <Thumb key={x.id} sample={x} type={cls.type} />
              ))}
              {xs.length > PER_TAKE && (
                <button className="btn btn-ghost small" onClick={() => setExpanded((e) => ({ ...e, [take]: !open }))}>
                  {open ? 'show less' : `+${xs.length - PER_TAKE} more`}
                </button>
              )}
            </div>
          </div>
        );
      })}
      {takes.length > showTakes && (
        <button className="btn btn-ghost" onClick={() => setShowTakes((n) => n + 10)}>
          Show older takes ({takes.length - showTakes} more)
        </button>
      )}
    </div>
  );
}

export default memo(SampleBrowser);
