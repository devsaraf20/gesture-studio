# TouchDesigner setup

Gesture Studio talks to TouchDesigner through the **bridge** (`npm start` or `npm run bridge` in the `web` folder).
Pick one of the three recipes below. Recipe 1 is the easiest place to start.

Before you start:

1. Run `npm start` in the `web` folder. It opens the app and starts the bridge.
2. On the **Live** page, load a model and tick **Send to bridge**. The status should say *Connected*.
3. The OSC port for TouchDesigner is set in `web/bridge/config.json` (default **10000**).

> Windows may ask whether Node.js can use the network the first time the bridge starts. Click **Allow**, otherwise TouchDesigner won't receive anything.

---

## Recipe 1: OSC In CHOP (no code)

Best for driving parameters with smooth 0–1 values.

1. Add an **OSC In CHOP**. Set **Network Port** to `10000`.
2. Make a gesture. You'll see channels appear, for example:
   - `gesture/peace`: `1` while you hold the peace sign, `0` when you stop
   - `gesture/swipe_left`: a short `1` pulse for each swipe
   - `hand/x`, `hand/y`: where your hand is (0–1, as seen in the mirrored camera)
   - `hand/size`: roughly how close your hand is (bigger = closer)
3. Add a **Select CHOP** to pick the channels you want, e.g. `gesture/peace hand/x`.
4. Add a **Lag CHOP** (lag 0.2 / 0.2) after it so values glide instead of jumping.
5. Drag a channel onto any parameter and pick *Export CHOP* (or reference it, e.g. `op('lag1')['gesture/peace']`).

Example ideas:

- `gesture/peace` → **Level TOP** opacity: the image fades in while you show peace.
- `hand/x` → **Transform TOP** rotate: turn your hand left and right to rotate the image.
- `gesture/fist` → **Trigger CHOP** → envelope for a flash on every fist.

*Tip:* the channel names come from the OSC addresses. You can change an address per gesture on the Live page (for example `/td/flash`).

## Recipe 2: OSC In DAT + Python (react to events)

Best when a gesture should *do* something once: jump to the next scene, randomise a seed, trigger a cue.

1. Add an **OSC In DAT**. Set **Network Port** to `10000`.
   *Only one operator can listen on a port: switch off the OSC In CHOP from recipe 1 or change one of the ports.*
2. Open its **callbacks DAT** and replace its content with [`oscin_callbacks.py`](oscin_callbacks.py).
3. Edit the `ACTIONS` dictionary at the top: one line per gesture.
4. The example uses a `noise1` operator (fist gives a new random seed), a `constant1` TOP (peace gives a random colour) and a Text DAT called `gesture_log` (every event is written there). Create those, or change the functions.

## Recipe 3: WebSocket DAT (JSON, no OSC)

Best if you want the confidence values or prefer JSON. It also works across computers (use the bridge computer's IP address).

1. Add a **Table DAT** and name it `gesture_table`.
2. Add a **WebSocket DAT**: Network Address `localhost`, Network Port `8787`, Active `On`.
3. Replace its callbacks DAT content with [`websocket_callbacks.py`](websocket_callbacks.py).
4. Add a **DAT to CHOP**: DAT `gesture_table`, First Row = Names, First Column = Names, Output = Channel per Row.
   Every gesture becomes a channel (`peace`, `fist`, …) plus `hand_x`, `hand_y`, `hand_size`.

## Testing without TouchDesigner

In the `web` folder, run `npm run monitor` (with TouchDesigner closed, since both use port 10000).
It prints every OSC message the bridge sends. The **test** buttons on the Live page send a
gesture without you having to perform it.

## Running the model inside TouchDesigner (advanced)

Training also exports `model.onnx`. TouchDesigner's Python can run it with `onnxruntime`
(`pip install onnxruntime` into TouchDesigner's Python), if you'd rather not use the browser.
You would then need to compute the same features as `web/src/lib/features.js` from your own
hand tracking (for example the MediaPipe TouchDesigner plugin). The bridge approach above is
much simpler, so start there.
