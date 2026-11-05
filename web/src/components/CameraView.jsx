// Webcam picture with the hand skeleton drawn on top (mirrored, like a selfie camera).
import { useEffect, useRef } from 'react';
import { drawHand } from '../lib/drawing.js';
import { getVideo, onFrame, startCamera, trackerStatus } from '../lib/handTracker.js';
import { useStore } from '../lib/store.js';

/**
 * onDraw(ctx, frame, width, height) - optional, to draw extra things on top
 * children - HTML overlays (labels, record indicator…)
 */
export default function CameraView({ onDraw, children, compact = false }) {
  const canvasRef = useRef(null);
  const onDrawRef = useRef(onDraw);
  onDrawRef.current = onDraw;
  const status = useStore(trackerStatus);

  useEffect(
    () =>
      onFrame((frame) => {
        const canvas = canvasRef.current;
        const video = getVideo();
        if (!canvas || !video) return;
        if (canvas.width !== frame.width || canvas.height !== frame.height) {
          canvas.width = frame.width;
          canvas.height = frame.height;
        }
        const ctx = canvas.getContext('2d');
        const { width: w, height: h } = canvas;
        ctx.save();
        ctx.translate(w, 0);
        ctx.scale(-1, 1); // mirror
        ctx.drawImage(video, 0, 0, w, h);
        ctx.restore();
        ctx.fillStyle = 'rgba(10,12,20,0.25)';
        ctx.fillRect(0, 0, w, h);
        if (frame.hand) drawHand(ctx, frame.hand.landmarks, w, h, { lineWidth: Math.max(2, w / 260) });
        onDrawRef.current?.(ctx, frame, w, h);
      }),
    [],
  );

  return (
    <div className={`camera ${compact ? 'camera-compact' : ''}`}>
      <canvas ref={canvasRef} width={960} height={540} />
      {status.state !== 'running' && (
        <div className="camera-placeholder">
          {status.state === 'loading' ? (
            <p>{status.message}</p>
          ) : (
            <>
              <button className="btn btn-primary btn-lg" onClick={startCamera}>
                Start camera
              </button>
              <p className="muted small">
                Video stays on your computer. Only 21 hand points per frame are used.
              </p>
              {status.state === 'error' && <p className="error">{status.message}</p>}
            </>
          )}
        </div>
      )}
      {status.state === 'running' && children}
    </div>
  );
}
