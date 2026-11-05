// The 3D playground for the Demo page. Gestures call the action methods below.
import * as THREE from 'three';

const COLOURS = ['#4cc9f0', '#ff8a5b', '#3ddc97', '#ffd23f', '#b388ff', '#ff5d8f'];

/** A soft round dot, so particles are glowing circles instead of squares. */
function dotTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.4, 'rgba(255,255,255,0.6)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

export class DemoScene {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor('#0b0d17');
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog('#0b0d17', 8, 22);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    this.camera.position.set(0, 0.6, 7);
    this.camera.lookAt(0, 0, 0);

    this.scene.add(new THREE.HemisphereLight('#c8d4ff', '#1a1020', 1.2));
    const key = new THREE.DirectionalLight('#ffffff', 2.2);
    key.position.set(3, 5, 4);
    this.scene.add(key);
    const rim = new THREE.PointLight('#ff8a5b', 25, 20);
    rim.position.set(-4, -1, -3);
    this.scene.add(rim);

    // the object you control: an outer group (tilt, follows your hand) holding the mesh (spins)
    this.pivot = new THREE.Group();
    this.scene.add(this.pivot);
    this.colourIndex = 0;
    this.material = new THREE.MeshStandardMaterial({ color: COLOURS[0], roughness: 0.25, metalness: 0.35, flatShading: true });
    this.mesh = new THREE.Mesh(new THREE.IcosahedronGeometry(1.5, 1), this.material);
    this.pivot.add(this.mesh);
    const wire = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.53, 1),
      new THREE.MeshBasicMaterial({ color: '#ffffff', wireframe: true, transparent: true, opacity: 0.12 }),
    );
    this.mesh.add(wire);

    const grid = new THREE.GridHelper(30, 30, '#2a2f4a', '#1a1e33');
    grid.position.y = -2.4;
    this.scene.add(grid);

    this.targetColour = new THREE.Color(COLOURS[0]);
    this.targetYaw = 0;
    this.spinning = false;
    this.spinSpeed = 0;
    this.tilt = { x: 0, y: 0 };
    this.scaleKick = 0;
    this.bursts = [];
    this.dot = dotTexture();
    this.timer = new THREE.Timer();

    this.resize = this.resize.bind(this);
    this.observer = new ResizeObserver(this.resize);
    this.observer.observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  // ------------------------------------------------------------------ actions
  nextColour() {
    this.colourIndex = (this.colourIndex + 1) % COLOURS.length;
    this.targetColour.set(COLOURS[this.colourIndex]);
    this.scaleKick = 0.15;
  }

  setSpinning(on) {
    this.spinning = on;
  }

  rotateBy(degrees) {
    this.targetYaw += THREE.MathUtils.degToRad(degrees);
  }

  burst() {
    const n = 600;
    const positions = new Float32Array(n * 3);
    const velocities = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const dir = new THREE.Vector3().randomDirection();
      const speed = 2 + Math.random() * 5;
      velocities.set([dir.x * speed, dir.y * speed, dir.z * speed], i * 3);
      positions.set([dir.x * 1.4, dir.y * 1.4, dir.z * 1.4], i * 3);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      color: this.targetColour.clone().offsetHSL(0, 0, 0.15),
      size: 0.14,
      map: this.dot,
      transparent: true,
      opacity: 1,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const points = new THREE.Points(geo, mat);
    this.scene.add(points);
    this.bursts.push({ points, velocities, age: 0 });
    this.scaleKick = 0.3;
  }

  /** x, y: 0..1 hand position in the mirrored view (0.5, 0.5 = centre). */
  setHand(x, y, present) {
    this.tilt.x = present ? (y - 0.5) * 0.9 : 0;
    this.tilt.y = present ? (x - 0.5) * 1.2 : 0;
  }

  // ------------------------------------------------------------------ animation
  tick() {
    this.timer.update();
    const dt = Math.min(this.timer.getDelta(), 0.05);
    this.material.color.lerp(this.targetColour, 1 - Math.exp(-dt * 6));

    this.spinSpeed += ((this.spinning ? 4 : 0) - this.spinSpeed) * (1 - Math.exp(-dt * 4));
    this.targetYaw += this.spinSpeed * dt;
    this.mesh.rotation.y += (this.targetYaw - this.mesh.rotation.y) * (1 - Math.exp(-dt * 6));
    this.mesh.rotation.x += dt * 0.15;

    this.pivot.rotation.x += (this.tilt.x - this.pivot.rotation.x) * (1 - Math.exp(-dt * 5));
    this.pivot.rotation.z += (-this.tilt.y * 0.4 - this.pivot.rotation.z) * (1 - Math.exp(-dt * 5));
    this.pivot.position.x += (this.tilt.y * 1.5 - this.pivot.position.x) * (1 - Math.exp(-dt * 5));

    this.scaleKick *= Math.exp(-dt * 6);
    this.mesh.scale.setScalar(1 + this.scaleKick);

    for (const b of this.bursts) {
      b.age += dt;
      const pos = b.points.geometry.attributes.position;
      for (let i = 0; i < pos.count; i++) {
        b.velocities[i * 3 + 1] -= 3 * dt; // a little gravity
        pos.array[i * 3] += b.velocities[i * 3] * dt;
        pos.array[i * 3 + 1] += b.velocities[i * 3 + 1] * dt;
        pos.array[i * 3 + 2] += b.velocities[i * 3 + 2] * dt;
      }
      pos.needsUpdate = true;
      b.points.material.opacity = Math.max(0, 1 - b.age / 1.6);
    }
    for (const b of this.bursts.filter((x) => x.age > 1.6)) {
      this.scene.remove(b.points);
      b.points.geometry.dispose();
      b.points.material.dispose();
    }
    this.bursts = this.bursts.filter((x) => x.age <= 1.6);

    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  dispose() {
    this.renderer.setAnimationLoop(null);
    this.observer.disconnect();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

export const DEMO_ACTIONS = {
  none: 'do nothing',
  colour: 'change colour',
  spin: 'spin while held',
  rotate_left: 'rotate left 90°',
  rotate_right: 'rotate right 90°',
  burst: 'particle burst',
};

/** A sensible default action for a gesture name. */
export function guessAction(name) {
  const n = name.toLowerCase();
  if (/swipe.?left/.test(n)) return 'rotate_left';
  if (/swipe.?right/.test(n)) return 'rotate_right';
  if (/fist|swipe.?up|push/.test(n)) return 'burst';
  if (/peace|victory|thumb/.test(n)) return 'colour';
  if (/point/.test(n)) return 'spin';
  return 'none';
}
