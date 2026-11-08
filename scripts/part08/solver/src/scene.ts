import {
  ArrowHelper, BoxGeometry, BufferGeometry, CatmullRomCurve3, Color, Group, Line, LineBasicMaterial,
  Mesh, MeshBasicMaterial, PerspectiveCamera, PlaneGeometry, Scene, SphereGeometry, TubeGeometry,
  Vector3, WebGLRenderer,
} from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { goal, release, type Flight, type Vec3 } from './flight.ts'

function world(point: Vec3): Vector3 {
  return new Vector3(point[1], point[2], point[0])
}

function line(points: Vector3[], color: number, opacity = 1) {
  return new Line(
    new BufferGeometry().setFromPoints(points),
    new LineBasicMaterial({ color, opacity, transparent: opacity < 1 }),
  )
}

function marker(point: Vec3, color: number, size: number) {
  const mesh = new Mesh(new SphereGeometry(size, 16, 12), new MeshBasicMaterial({ color }))
  mesh.position.copy(world(point))
  return mesh
}

function field(scene: Scene) {
  for (let index = 0; index < 8; index++) {
    const stripe = new Mesh(new PlaneGeometry(14, 2), new MeshBasicMaterial({
      color: index % 2 ? 0x215044 : 0x295a49,
    }))
    stripe.rotation.x = -Math.PI / 2
    stripe.position.set(0, 0, index * 2 - 2)
    scene.add(stripe)
  }

  const h = goal.halfWidth
  const x = goal.x
  const fieldLine = (coordinates: Vec3[]) => scene.add(line(coordinates.map(world), 0xc3dac8, 0.72))
  fieldLine([[x, -6.8, 0.01], [x, 6.8, 0.01]])
  fieldLine([[x, -5.5, 0.01], [5.5, -5.5, 0.01], [5.5, 5.5, 0.01], [x, 5.5, 0.01]])
  const spot = new Mesh(new PlaneGeometry(0.19, 0.19), new MeshBasicMaterial({ color: 0xc3dac8 }))
  spot.rotation.x = -Math.PI / 2
  spot.position.y = 0.012
  scene.add(spot)

  const frameMaterial = new MeshBasicMaterial({ color: 0xf3f9ef })
  for (const y of [-h, h]) {
    const post = new Mesh(new BoxGeometry(0.05, goal.height, 0.05), frameMaterial)
    post.position.set(y, goal.height / 2, x)
    scene.add(post)
  }
  const bar = new Mesh(new BoxGeometry(h * 2 + 0.05, 0.05, 0.05), frameMaterial)
  bar.position.set(0, goal.height, x)
  scene.add(bar)

  const depth = 0.65
  const netLine = (coordinates: Vec3[]) => scene.add(line(coordinates.map(world), 0xc3dac8, 0.38))
  for (let index = 0; index <= 24; index++) {
    const y = -h + 2 * h * index / 24
    netLine([[x, y, goal.height], [x + depth, y, goal.height], [x + depth, y, 0], [x, y, 0]])
  }
  for (let index = 0; index <= 8; index++) {
    const z = goal.height * index / 8
    netLine([[x, -h, z], [x + depth, -h, z], [x + depth, h, z], [x, h, z]])
  }
}

export function mountScene(container: HTMLElement, reset: HTMLButtonElement) {
  let renderer: WebGLRenderer
  try {
    renderer = new WebGLRenderer({ antialias: true })
  } catch {
    const fallback = document.createElement('p')
    fallback.className = 'scene-fallback'
    fallback.textContent = '3D view unavailable'
    container.append(fallback)
    reset.disabled = true
    return () => {}
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.domElement.setAttribute('aria-label', 'Interactive three-dimensional flight path')
  container.prepend(renderer.domElement)

  const scene = new Scene()
  scene.background = new Color(0x102d25)
  field(scene)
  const shot = new Group()
  scene.add(shot)

  const camera = new PerspectiveCamera(39, 1, 0.1, 150)
  camera.position.set(16, 8, -2)
  const controls = new OrbitControls(camera, renderer.domElement)
  controls.target.set(0, 0.65, 5.4)
  controls.minDistance = 4
  controls.maxDistance = 60
  controls.maxPolarAngle = Math.PI * 0.49
  controls.update()
  controls.saveState()

  let scheduled = false
  function draw() {
    if (scheduled) return
    scheduled = true
    requestAnimationFrame(() => {
      renderer.render(scene, camera)
      scheduled = false
    })
  }

  controls.addEventListener('change', draw)
  reset.addEventListener('click', () => controls.reset())
  new ResizeObserver(() => {
    const { clientWidth, clientHeight } = container
    if (!clientWidth || !clientHeight) return
    camera.aspect = clientWidth / clientHeight
    camera.updateProjectionMatrix()
    renderer.setSize(clientWidth, clientHeight, false)
    draw()
  }).observe(container)

  return (flight: Flight | undefined, wind: Vec3, observations?: readonly Vec3[]) => {
    shot.traverse((object) => {
      if (object instanceof Mesh || object instanceof Line) {
        object.geometry.dispose()
        const materials = Array.isArray(object.material) ? object.material : [object.material]
        materials.forEach((material) => material.dispose())
      }
    })
    shot.clear()
    const speed = Math.hypot(...wind)
    if (speed > 0 && Number.isFinite(speed)) {
      shot.add(new ArrowHelper(
        world(wind).normalize(), new Vector3(-4.8, 0.14, 2.2),
        Math.min(3.4, 1 + 0.25 * speed), 0x7cddf2, 0.35, 0.22,
      ))
    }
    if (observations) {
      for (const point of [release, ...observations]) {
        shot.add(marker(point, 0xffa86b, 0.077))
      }
    }

    if (flight && flight.points.length > 1) {
      const interval = Math.ceil(flight.points.length / 110)
      const positions = flight.points
        .filter((_, index) => index % interval === 0 || index === flight.points.length - 1)
        .map((point) => world(point.position))
      const curve = new CatmullRomCurve3(positions)
      shot.add(new Mesh(
        new TubeGeometry(curve, Math.max(24, positions.length * 2), 0.041, 8),
        new MeshBasicMaterial({ color: 0xe1fb99 }),
      ))
      const last = flight.points[flight.points.length - 1]
      shot.add(marker(last.position, 0xe1fb99, 0.069))
    }
    draw()
  }
}
