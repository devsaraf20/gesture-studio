# Unreal Engine setup (UE 5.x, Blueprints only)

Gesture Studio sends OSC messages to Unreal through the **bridge** (`npm start` in the `web` folder).
Unreal's built-in **OSC plugin** receives them. You don't need any C++.

Before you start:

1. Run `npm start` in the `web` folder.
2. On the **Live** page, load a model and tick **Send to bridge**.
3. The OSC port for Unreal is set in `web/bridge/config.json` (default **8000**).

## 1. Enable the OSC plugin

**Edit → Plugins**, search **OSC**, tick it, restart the editor.

## 2. Create a receiver actor

1. Content Browser → **Add → Blueprint Class → Actor**, name it `BP_GestureReceiver`.
2. Open it, go to the **Event Graph**, and build this:

```
Event BeginPlay
  └─► Create OSC Server
        Receive IP Address : 0.0.0.0
        Port               : 8000
        Multicast Loopback : false
        Start Listening    : true
        Server Name        : GestureStudio
      Return Value ─► Promote to Variable  (name it "OscServer")
  └─► Bind Event to On Osc Message Received   (Target: OscServer)
        Event ─► (drag out, "Add Custom Event…") name it "OnGesture"
```

3. The custom event `OnGesture` gives you **Message**, **IPAddress** and **Port**. Continue:

```
OnGesture (Message)
  └─► Get OSC Message Address (Message)
        └─► Convert OSC Address To String
              └─► Switch on String
                    Pin "/gesture/peace" ─► Get OSC Message Float (Message, Index 0)
                                              └─► Branch (Value > 0.5)
                                                    True  ─► (peace started: e.g. Set Light Color)
                                                    False ─► (peace ended)
                    Pin "/gesture/fist"  ─► Get OSC Message Float … ─► Branch ─► Spawn System at Location (Niagara burst)
                    Pin "/gesture/swipe_left" ─► … ─► Add Actor Local Rotation (Yaw -90)
                    Pin "/hand/x"        ─► Get OSC Message Float ─► Map Range Clamped (0..1 → -45..45) ─► Set Actor Rotation (Yaw)
```

*Adding pins to Switch on String:* select the node and, in the Details panel, add entries to **Pin Names**.

4. Drag `BP_GestureReceiver` into your level and press **Play**. Make a gesture.

### What the messages mean

| Address | Value | When |
|---|---|---|
| `/gesture/<name>` | `1.0` / `0.0` | gesture starts / ends (motions: a short 1 then 0) |
| `/gesture/current` | string name, float confidence | each start (`"none"`, 0 on end) |
| `/hand/x`, `/hand/y` | 0–1 | continuous, about 30 per second (if enabled) |
| `/hand/size` | ~0.1–0.6 | bigger = hand closer to camera |
| `/hand/present` | 1 / 0 | is a hand visible |

All numbers are floats, so always use **Get OSC Message Float**.
You can change the address for each gesture on the Live page.

### Debugging

- Add a **Print String** right after `OnGesture` showing the address. If nothing prints, check the port and that the bridge says it sends to `Unreal 127.0.0.1:8000`.
- With Unreal closed, `npm run monitor -- 8000` in the `web` folder prints what the bridge sends to that port.
- Windows firewall: allow Node.js and Unreal Editor on private networks.
- The **Create OSC Server** node only runs during Play. To use it in the editor, call it from an Editor Utility Widget instead.

## Alternative: WebSocket JSON

The bridge also serves JSON on `ws://localhost:8787`, the same messages TouchDesigner's WebSocket DAT gets.
Unreal's WebSocket support is C++ only (`WebSockets` module), so OSC is the simpler route for Blueprints.

## Running the model inside Unreal (advanced)

Training exports `model.onnx`, which Unreal's **Neural Network Engine (NNE)** can load.
You would also need hand landmarks inside Unreal (e.g. from a MediaPipe plugin) and must compute
the same features as `web/src/lib/features.js`. Using the browser plus OSC is much simpler, so start with that.
