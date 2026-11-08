import {
  airFactor, cross, dot, effectiveGravity, gravity, integrator, radius, release, simulate, vectorIsFinite,
  type Motion, type Vec3,
} from './flight.ts'

export const times = [0.05, 0.1] as const
export type Observations = readonly [Vec3, Vec3]

const scales = [30, 10, 10, 0.2, 50, 50]
const nominalBounds = [[10, -20, 0.1, 0.05, -100, -100], [45, 20, 20, 0.6, 100, 100]]
const sampleBounds = [[5, -30, 0.01, 0.01, -150, -150], [55, 30, 30, 1, 150, 150]]
const difference = 2e-5

export interface FitResult {
  motion: Motion
  residuals: number[]
  rms: number
}

interface FitOptions {
  expanded?: boolean
  initial?: Motion
}

function chordBasis(last: Vec3): [Vec3, Vec3] {
  const chord: Vec3 = [last[0] - release[0], last[1] - release[1], last[2] - release[2]]
  const length = Math.hypot(...chord)
  if (!(length > 0) || !Number.isFinite(length)) throw new Error('Observation chord must have a finite nonzero length.')
  const e1: Vec3 = [chord[0] / length, chord[1] / length, chord[2] / length]
  const transverse = cross(Math.abs(e1[2]) > 0.99 ? [1, 0, 0] : [0, 0, 1], e1)
  const norm = Math.hypot(...transverse)
  const e2: Vec3 = [transverse[0] / norm, transverse[1] / norm, transverse[2] / norm]
  return [e2, cross(e1, e2)]
}

function starts(observations: Observations, wind: Vec3, e2: Vec3, e3: Vec3): number[][] {
  const [a, b] = observations
  const h = times[0]
  const displacement = a.map((value, i) => value - release[i])
  const second = b.map((value, i) => value - release[i])
  const acceleration = second.map((value, i) => (value - 2 * displacement[i]) / h ** 2)
  const velocity = displacement.map((value, i) => value / h - h * acceleration[i] / 2)
  const ballistic = displacement.map((value, i) => (
    h * (value + (i === 2 ? gravity * h ** 2 / 2 : 0)) +
    2 * h * (second[i] + (i === 2 ? gravity * (2 * h) ** 2 / 2 : 0))
  ) / (5 * h ** 2))
  const relative: Vec3 = [
    velocity[0] + h * acceleration[0] - wind[0],
    velocity[1] + h * acceleration[1] - wind[1],
    velocity[2] + h * acceleration[2] - wind[2],
  ]
  const air: Vec3 = [acceleration[0], acceleration[1], acceleration[2] + effectiveGravity]
  const speed = Math.hypot(...relative)
  const spin = cross(relative, air)
  const liftScale = airFactor * radius * speed ** 2
  const drag = -dot(air, relative) / (airFactor * speed ** 3)
  return [
    [...velocity, drag, dot(spin, e2) / liftScale, dot(spin, e3) / liftScale],
    [...velocity, 0.2, 0, 0],
    [...ballistic, 0.2, 0, 0],
  ]
}

function squaredNorm(values: number[]): number {
  return values.reduce((total, value) => total + value * value, 0)
}

function solveLinear(matrix: number[][], rhs: number[]): number[] | undefined {
  const rows = matrix.map((row, i) => [...row, rhs[i]])
  const size = rhs.length
  for (let column = 0; column < size; column++) {
    let pivot = column
    for (let i = column + 1; i < size; i++) {
      if (Math.abs(rows[i][column]) > Math.abs(rows[pivot][column])) pivot = i
    }
    if (!(Math.abs(rows[pivot][column]) > 1e-20)) return undefined
    const pivotRow = rows[pivot]
    rows[pivot] = rows[column]
    rows[column] = pivotRow
    for (let i = column + 1; i < size; i++) {
      const factor = rows[i][column] / pivotRow[column]
      for (let j = column + 1; j <= size; j++) rows[i][j] -= factor * pivotRow[j]
    }
  }
  const solution = Array<number>(size).fill(0)
  for (let i = size - 1; i >= 0; i--) {
    let value = rows[i][size]
    for (let j = i + 1; j < size; j++) value -= rows[i][j] * solution[j]
    solution[i] = value / rows[i][i]
  }
  return solution.every(Number.isFinite) ? solution : undefined
}

function minimize(
  initial: number[],
  residual: (value: number[]) => number[],
  low: number[],
  high: number[],
): { value: number[]; errors: number[]; score: number } | undefined {
  const clip = (value: number, i: number) => Math.max(low[i], Math.min(high[i], value))
  let value = initial.map((component, i) => clip(component / scales[i], i))
  let errors = residual(value)
  let score = squaredNorm(errors)
  let damping = 1e-8
  if (!Number.isFinite(score)) return undefined

  for (let iteration = 0; iteration < 70; iteration++) {
    if (score < 1e-23) return { value, errors, score }
    const columns = value.map((_, i) => {
      const shifted = value.slice()
      shifted[i] += difference
      return residual(shifted).map((error, j) => (error - errors[j]) / difference)
    })
    const gradient = columns.map((column) => column.reduce((sum, entry, j) => sum + entry * errors[j], 0))
    const free = gradient.map((g, i) => !(
      (value[i] <= low[i] + 1e-10 && g > 0) || (value[i] >= high[i] - 1e-10 && g < 0)
    ))
    const norm = Math.max(...gradient.map((g, i) => free[i] ? Math.abs(g) : 0))
    if (!Number.isFinite(norm)) return undefined
    if (norm < 1e-11) return { value, errors, score }
    const normal = columns.map((column, i) => columns.map((other, j) => (
      free[i] && free[j] ? column.reduce((sum, entry, k) => sum + entry * other[k], 0) : 0
    )))
    let improved = false
    for (let attempt = 0; attempt < 12; attempt++) {
      const regularized = normal.map((row, i) => row.map((entry, j) => (
        entry + (i === j ? (free[i] ? damping : 1) : 0)
      )))
      const delta = solveLinear(regularized, gradient.map((g, i) => free[i] ? -g : 0))
      if (!delta) return undefined
      const trial = value.map((component, i) => clip(component + delta[i], i))
      const trialErrors = residual(trial)
      const trialScore = squaredNorm(trialErrors)
      if (trialScore < score) {
        const change = score - trialScore
        value = trial
        errors = trialErrors
        score = trialScore
        damping = Math.max(1e-14, damping / 3)
        if (change < 1e-16 && norm < 2e-8) return { value, errors, score }
        improved = true
        break
      }
      damping *= 10
    }
    if (!improved) return norm < 2e-8 ? { value, errors, score } : undefined
  }
  return undefined
}

export function fit(observations: Observations, wind: Vec3, options: FitOptions = {}): FitResult {
  if (!observations.every(vectorIsFinite) || !vectorIsFinite(wind)) {
    throw new Error('Observation coordinates and wind must be finite.')
  }
  const [e2, e3] = chordBasis(observations[1])
  const [lower, upper] = options.expanded ? sampleBounds : nominalBounds
  const low = lower.map((value, i) => value / scales[i])
  const high = upper.map((value, i) => value / scales[i])

  function unpack(value: number[]): Motion {
    const [vx, vy, vz, cd, w2, w3] = value.map((component, i) => component * scales[i])
    return {
      velocity: [vx, vy, vz], cd,
      spin: [w2 * e2[0] + w3 * e3[0], w2 * e2[1] + w3 * e3[1], w2 * e2[2] + w3 * e3[2]],
    }
  }

  function residual(value: number[]): number[] {
    const motion = unpack(value)
    const state = [...release, ...motion.velocity]
    const step = integrator(motion, wind)
    const errors: number[] = []
    let previous = 0
    times.forEach((time, index) => {
      const dt = (time - previous) / 10
      for (let i = 0; i < 10; i++) step(state, dt, state)
      for (let axis = 0; axis < 3; axis++) errors.push(state[axis] - observations[index][axis])
      previous = time
    })
    return errors
  }

  function result(candidate: ReturnType<typeof minimize>): FitResult | undefined {
    if (!candidate) return undefined
    const motion = unpack(candidate.value)
    const observedFlight = simulate(motion, wind, times[1], false)
    if (observedFlight.event !== 'timeout') return undefined
    return { motion, residuals: candidate.errors, rms: Math.sqrt(candidate.score / 6) }
  }

  if (options.initial) {
    const { velocity, cd, spin } = options.initial
    const warmed = result(minimize([...velocity, cd, dot(spin, e2), dot(spin, e3)], residual, low, high))
    if (warmed) return warmed
  }

  let best: FitResult | undefined
  for (const guess of starts(observations, wind, e2, e3)) {
    if (!guess.every(Number.isFinite)) continue
    const candidate = result(minimize(guess, residual, low, high))
    if (candidate && (!best || candidate.rms < best.rms)) best = candidate
    if (best && best.rms < 1e-10) break
  }
  if (!best) throw new Error('No valid fit within the observation window.')
  return best
}
