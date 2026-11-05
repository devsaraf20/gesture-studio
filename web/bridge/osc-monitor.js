// Prints every OSC message arriving on a UDP port - a quick way to check the bridge works
// before opening TouchDesigner or Unreal.
//
//   npm run monitor            (listens on 10000, the TouchDesigner port in config.json)
//   npm run monitor -- 8000    (listens on another port)
//
// Note: only one program can listen on a port. Close TouchDesigner's OSC In first.
import dgram from 'node:dgram';
import { decodeOsc } from './osc.js';

const port = Number(process.argv[2] || 10000);
const sock = dgram.createSocket('udp4');
sock.on('message', (buf) => {
  try {
    const { address, args } = decodeOsc(buf);
    if (address.startsWith('/hand/') && !process.argv.includes('--hand')) return; // too chatty
    console.log(`${new Date().toLocaleTimeString()}  ${address.padEnd(24)} ${args.map((a) => (typeof a === 'number' ? a.toFixed(3) : JSON.stringify(a))).join('  ')}`);
  } catch (e) {
    console.warn('could not decode packet', e.message);
  }
});
sock.bind(port, () => console.log(`Listening for OSC on UDP port ${port} (add --hand to also show /hand/* messages)`));
