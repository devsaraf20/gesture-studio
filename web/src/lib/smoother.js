// Turns jittery per-frame predictions into clean "gesture started / ended" events.
//
//   smoothing  : blend each new prediction with the previous ones (0 = none, 0.9 = very smooth)
//   threshold  : the top gesture must be at least this confident (0..1) to count
//   holdMs     : …and stay on top for this long before it triggers
//   cooldownMs : after a trigger, wait this long before another one
//
// "Rest" classes (none, idle, nothing, background…) never trigger anything: they exist so the
// model has somewhere to put "no gesture".

export const REST_CLASSES = new Set(['none', 'idle', 'nothing', 'background', 'rest', 'neutral']);
const MOTION_PULSE_MS = 250; // motion gestures are momentary: "on" for 250 ms, then "off"

export class GestureSmoother {
  constructor(classes, mode) {
    this.classes = classes;
    this.mode = mode; // 'pose' | 'motion'
    this.reset();
  }

  reset() {
    this.smoothed = new Float32Array(this.classes.length);
    this.candidate = null;
    this.candidateSince = 0;
    this.active = null;
    this.activeSince = 0;
    this.lastTrigger = -Infinity;
  }

  /**
   * probs: Float32Array of raw probabilities, or null when no hand is visible.
   * Returns { smoothed, top, topProb, active, events: [{ gesture, state: 'start'|'end', confidence }] }
   */
  update(probs, now, { smoothing, threshold, holdMs, cooldownMs }) {
    const events = [];
    const a = Math.min(Math.max(smoothing, 0), 0.98);
    for (let i = 0; i < this.smoothed.length; i++) {
      const target = probs ? probs[i] : 0; // no hand: fade everything out
      this.smoothed[i] = a * this.smoothed[i] + (1 - a) * target;
    }
    let top = 0;
    for (let i = 1; i < this.smoothed.length; i++) if (this.smoothed[i] > this.smoothed[top]) top = i;
    const topName = this.classes[top];
    const topProb = this.smoothed[top];

    const confident = topProb >= threshold && !REST_CLASSES.has(topName.toLowerCase());
    const candidate = confident ? topName : null;
    if (candidate !== this.candidate) {
      this.candidate = candidate;
      this.candidateSince = now;
    }
    const stableFor = now - this.candidateSince;

    if (this.mode === 'motion') {
      // momentary: fire once, then switch off after a short pulse
      if (this.active && now - this.activeSince > MOTION_PULSE_MS) {
        events.push({ gesture: this.active, state: 'end', confidence: topProb });
        this.active = null;
      }
      const wait = Math.max(cooldownMs, MOTION_PULSE_MS * 2); // never re-fire during its own pulse
      if (candidate && !this.active && stableFor >= holdMs && now - this.lastTrigger >= wait) {
        this.active = candidate;
        this.activeSince = now;
        this.lastTrigger = now;
        events.push({ gesture: candidate, state: 'start', confidence: topProb });
        this.smoothed.fill(0); // forget this motion so it doesn't fire twice
        this.candidate = null;
        this.wantsBufferReset = true;
      }
    } else if (candidate !== this.active) {
      // pose: "on" while held, "off" when released
      const canStart = candidate && stableFor >= holdMs && now - this.lastTrigger >= cooldownMs;
      const shouldEnd = this.active && stableFor >= holdMs; // released, or a different gesture is shown
      if (shouldEnd) {
        events.push({ gesture: this.active, state: 'end', confidence: topProb });
        this.active = null;
      }
      if (canStart) {
        this.active = candidate;
        this.activeSince = now;
        this.lastTrigger = now;
        events.push({ gesture: candidate, state: 'start', confidence: topProb });
      }
    }
    return { smoothed: this.smoothed, top: topName, topProb, active: this.active, events };
  }
}
