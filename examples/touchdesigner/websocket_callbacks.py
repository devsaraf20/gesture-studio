# TouchDesigner: paste this into the callbacks DAT of a "WebSocket DAT".
#   Network Address: localhost     Network Port: 8787     Active: On
#
# Gesture Studio's bridge sends one JSON message per event, for example:
#   {"type": "gesture", "gesture": "peace", "state": "start", "value": 1, "confidence": 0.93, ...}
#   {"type": "hand", "present": true, "x": 0.42, "y": 0.61, "size": 0.21}
#
# This script keeps a Table DAT called "gesture_table" up to date:
#     name | value | confidence
# Turn it into CHOP channels with a "DAT to CHOP" (First Column = Names, Output = Channel per Row).
import json


def _table():
    t = op('gesture_table')
    if t is not None and t.numRows == 0:
        t.appendRow(['name', 'value', 'confidence'])
    return t


def _set(name, value, confidence=1.0):
    t = _table()
    if t is None:
        return
    if t.row(name) is None:
        t.appendRow([name, value, confidence])
    else:
        t[name, 'value'] = value
        t[name, 'confidence'] = confidence


def onConnect(dat):
    debug('Connected to Gesture Studio bridge')
    return


def onDisconnect(dat):
    return


def onReceiveText(dat, rowIndex, message):
    try:
        msg = json.loads(message)
    except ValueError:
        return
    if msg.get('type') == 'gesture':
        _set(msg['gesture'], msg['value'], msg.get('confidence', 1.0))
    elif msg.get('type') == 'hand':
        _set('hand_x', msg.get('x', 0.5))
        _set('hand_y', msg.get('y', 0.5))
        _set('hand_size', msg.get('size', 0))
        _set('hand_present', 1 if msg.get('present') else 0)
    return


def onReceiveBinary(dat, contents):
    return


def onReceivePing(dat, contents):
    dat.sendPong(contents)
    return


def onReceivePong(dat, contents):
    return


def onMonitorMessage(dat, message):
    return
