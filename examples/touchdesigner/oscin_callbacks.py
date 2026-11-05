# TouchDesigner: paste this into the callbacks DAT of an "OSC In DAT" (Network Port 10000).
#
# Gesture Studio sends, for every gesture:
#   /gesture/<name>   1.0 when it starts, 0.0 when it ends
#   /gesture/current  "<name>" <confidence>   ("none" 0 when it ends)
#   /hand/x /hand/y /hand/size /hand/present   (if "stream hand position" is ticked)
#
# This script reacts to gestures *starting*. Edit the ACTIONS dictionary: each gesture name
# maps to a small function. The examples assume these operators exist next to the OSC In DAT:
#   noise1      a Noise TOP/CHOP   (fist  -> new random seed)
#   constant1   a Constant TOP     (peace -> random colour)
#   gesture_log a Text DAT          (every gesture is written here)
import random


def on_fist():
    op('noise1').par.seed = random.randint(0, 10000)


def on_peace():
    c = op('constant1')
    c.par.colorr, c.par.colorg, c.par.colorb = random.random(), random.random(), random.random()


def on_swipe_left():
    print('swipe left!')


ACTIONS = {
    'fist': on_fist,
    'peace': on_peace,
    'swipe_left': on_swipe_left,
}


def onReceiveOSC(dat, rowIndex, message, byteData, timeStamp, address, args, peer):
    if not address.startswith('/gesture/') or address == '/gesture/current':
        return
    name = address[len('/gesture/'):]
    started = bool(args) and args[0] > 0.5
    log = op('gesture_log')
    if log is not None:
        log.text = f"{name} {'start' if started else 'end'}\n" + log.text[:2000]
    if started and name in ACTIONS:
        try:
            ACTIONS[name]()
        except Exception as e:  # a missing operator should not stop the others
            debug(f'Gesture action for {name} failed: {e}')
    return
