// A tiny "store": holds a value, tells React components when it changes.
import { useSyncExternalStore } from 'react';

export function createStore(initial) {
  let state = initial;
  const listeners = new Set();
  return {
    get: () => state,
    set(update) {
      state = typeof update === 'function' ? update(state) : update;
      listeners.forEach((l) => l());
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

/** Re-render the component whenever the store changes. `select` must return stable references. */
export function useStore(store, select = (s) => s) {
  return useSyncExternalStore(store.subscribe, () => select(store.get()));
}

/** Minimal event emitter: on(type, fn) returns an "unsubscribe" function. */
export function createEmitter() {
  const map = new Map();
  return {
    on(type, fn) {
      if (!map.has(type)) map.set(type, new Set());
      map.get(type).add(fn);
      return () => map.get(type).delete(fn);
    },
    emit(type, payload) {
      map.get(type)?.forEach((fn) => fn(payload));
    },
  };
}
