export type Vec3 = readonly [number, number, number]
export type State = readonly number[]

export const release: Vec3 = [0, 0, 0.11]
export const goal = { x: 11, halfWidth: 3.66, height: 2.44 }
export const radius = 0.11
export const maxTime = 2
export const gravity = 9.81
const density = 1.2
const mass = 0.43
export const airFactor = density * Math.PI * radius ** 2 / (2 * mass)
export const effectiveGravity = gravity * (1 - density * (4 * Math.PI * radius ** 3 / 3) / mass)
const maxStep = 0.002
const eventTolerance = 1e-12

export interface Motion {
  velocity: Vec3
  spin: Vec3
  cd: number
}

export type FlightEvent = 'goal_inside' | 'goal_outside' | 'ground_first' | 'timeout' | 'numerical_failure'
export interface Flight {
  event: FlightEvent
  time: number
  points: { time: number; position: Vec3 }[]
  crossing?: { y: number; z: number }
}

export function vectorIsFinite(values: Vec3): boolean {
  return values.length === 3 && values.every(Number.isFinite)
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

export function insideGoal(y: number, z: number): boolean {
  return Math.abs(y) < goal.halfWidth - eventTolerance &&
    z > eventTolerance && z < goal.height - eventTolerance
}

export function position(state: State): Vec3 {
  return [state[0], state[1], state[2]]
}

export function integrator(motion: Motion, wind: Vec3) {
  const drag = airFactor * motion.cd
  const lift = airFactor * radius
  const [sx, sy, sz] = motion.spin
  const stages = Array.from({ length: 4 }, () => Array<number>(6).fill(0))
  const scratch = Array<number>(6).fill(0)

  function derivative(state: State, out: number[]) {
    const vx = state[3]
    const vy = state[4]
    const vz = state[5]
    const ux = vx - wind[0]
    const uy = vy - wind[1]
    const uz = vz - wind[2]
    const scale = -drag * Math.hypot(ux, uy, uz)
    out[0] = vx
    out[1] = vy
    out[2] = vz
    out[3] = scale * ux + lift * (sy * uz - sz * uy)
    out[4] = scale * uy + lift * (sz * ux - sx * uz)
    out[5] = scale * uz + lift * (sx * uy - sy * ux) - effectiveGravity
  }

  return function step(state: State, dt: number, out: number[]) {
    const [a, b, c, d] = stages
    derivative(state, a)
    for (let i = 0; i < 6; i++) scratch[i] = state[i] + dt * a[i] / 2
    derivative(scratch, b)
    for (let i = 0; i < 6; i++) scratch[i] = state[i] + dt * b[i] / 2
    derivative(scratch, c)
    for (let i = 0; i < 6; i++) scratch[i] = state[i] + dt * c[i]
    derivative(scratch, d)
    for (let i = 0; i < 6; i++) out[i] = state[i] + dt * (a[i] + 2 * b[i] + 2 * c[i] + d[i]) / 6
  }
}

function interpolate(a: State, b: State, dt: number, axis: number, fraction: number): number {
  const s = fraction
  const cubic = s * s * (3 - 2 * s)
  return a[axis] + cubic * (b[axis] - a[axis]) +
    dt * s * (1 - s) * ((1 - s) * a[axis + 3] - s * b[axis + 3])
}

function bisectCrossing(value: (fraction: number) => number, low: number, high: number) {
  for (let i = 0; i < 36; i++) {
    const mid = (low + high) / 2
    if (value(mid) < 0) low = mid
    else high = mid
  }
  return (low + high) / 2
}

function turningPoints(a: number, b: number, c: number): number[] {
  if (a === 0) return b === 0 ? [] : [-c / b]
  const discriminant = b * b - 4 * a * c
  if (discriminant <= 0) return []
  const first = -(b + (b < 0 ? -1 : 1) * Math.sqrt(discriminant)) / 2
  return [first / a, c / first]
}

function crossingFraction(a: State, b: State, dt: number, axis: number, threshold: number, direction: number) {
  const start = direction * (a[axis] - threshold)
  const finish = direction * (b[axis] - threshold)
  const entry = direction * dt * a[axis + 3]
  const exit = direction * dt * b[axis + 3]
  const control1 = start + entry / 3
  const control2 = finish - exit / 3
  if (Math.max(start, finish, control1, control2) < 0 || Math.min(start, finish, control1, control2) > 0) {
    return Infinity
  }

  // A turning point may hide an outward crossing between endpoints on the same side.
  const delta = finish - start
  const quadratic = 3 * delta - 2 * entry - exit
  const cubic = entry + exit - 2 * delta
  const value = (s: number) => s === 1 ? finish : start + s * (entry + s * (quadratic + s * cubic))
  const cuts = turningPoints(3 * cubic, 2 * quadratic, entry)
    .filter((s) => s > 0 && s < 1)
    .sort((first, second) => first - second)
  cuts.push(1)

  let low = 0
  let before = start
  for (const high of cuts) {
    const after = value(high)
    if (before < 0 && (after > 0 || (high === 1 && after === 0 && exit > 0))) {
      return bisectCrossing(value, low, high)
    }
    low = high
    before = after
  }
  return Infinity
}

export function simulate(motion: Motion, wind: Vec3, end = maxTime, record = true): Flight {
  if (!vectorIsFinite(motion.velocity) || !vectorIsFinite(motion.spin) || !vectorIsFinite(wind)) {
    throw new Error('Velocity, spin and wind must be finite.')
  }
  if (!Number.isFinite(motion.cd) || motion.cd < 0) {
    throw new Error('Drag coefficient must be finite and nonnegative.')
  }

  const points: Flight['points'] = record ? [{ time: 0, position: release }] : []
  if (motion.velocity[2] <= 0) return { event: 'ground_first', time: 0, points }

  const step = integrator(motion, wind)
  let current = [...release, ...motion.velocity]
  let next = Array<number>(6).fill(0)
  const steps = Math.ceil(end / maxStep)
  const dt = end / steps

  for (let index = 0; index < steps; index++) {
    const time = index * dt
    step(current, dt, next)
    if (!next.every(Number.isFinite)) return { event: 'numerical_failure', time, points }
    const goalRoot = crossingFraction(current, next, dt, 0, goal.x, 1)
    const groundRoot = crossingFraction(current, next, dt, 2, radius, -1)

    const fraction = Math.min(goalRoot, groundRoot)
    if (Number.isFinite(fraction)) {
      const stop = time + fraction * dt
      const groundFirst = groundRoot <= goalRoot + eventTolerance
      const y = interpolate(current, next, dt, 1, fraction)
      const z = groundFirst ? radius : interpolate(current, next, dt, 2, fraction)
      const x = groundFirst ? interpolate(current, next, dt, 0, fraction) : goal.x
      if (record) points.push({ time: stop, position: [x, y, z] })
      if (groundFirst) return { event: 'ground_first', time: stop, points }
      return { event: insideGoal(y, z) ? 'goal_inside' : 'goal_outside', time: stop, points, crossing: { y, z } }
    }

    if (record) points.push({ time: (index + 1) * dt, position: position(next) })
    ;[current, next] = [next, current]
  }
  return { event: 'timeout', time: end, points }
}
