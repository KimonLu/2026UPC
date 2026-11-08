import { fit, type Observations } from './fit.ts'
import { simulate, type FlightEvent, type Motion, type Vec3 } from './flight.ts'

type Vec2 = readonly [number, number]
type SampleEvent = FlightEvent | 'fit_failure'
export const sampleCount = 2000

export interface Region {
  center: Vec2
  major: number
  minor: number
  angle: number
}

export interface SampleSummary {
  counts: Record<SampleEvent, number>
  region?: Region
}

function normalGenerator() {
  let seed = 20261107

  // Mulberry32, mapped strictly inside (0, 1) for the Box–Muller transform.
  function uniform() {
    seed = (seed + 0x6D2B79F5) >>> 0
    let value = Math.imul(seed ^ (seed >>> 15), seed | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return (((value ^ (value >>> 14)) >>> 0) + 0.5) / 4294967296
  }

  let spare: number | undefined
  return () => {
    if (spare !== undefined) {
      const value = spare
      spare = undefined
      return value
    }
    const radial = Math.sqrt(-2 * Math.log(uniform()))
    const angle = 2 * Math.PI * uniform()
    spare = radial * Math.sin(angle)
    return radial * Math.cos(angle)
  }
}

export function predictionRegion(crossings: readonly Vec2[]): Region | undefined {
  const count = crossings.length
  if (count < 7) return undefined

  const center: Vec2 = [
    crossings.reduce((sum, [y]) => sum + y, 0) / count,
    crossings.reduce((sum, [, z]) => sum + z, 0) / count,
  ]
  const centered = crossings.map(([y, z]): Vec2 => [y - center[0], z - center[1]])
  let yy = 0
  let yz = 0
  let zz = 0
  for (const [y, z] of centered) {
    yy += y * y
    yz += y * z
    zz += z * z
  }
  yy /= count - 1
  yz /= count - 1
  zz /= count - 1
  if (Math.max(yy, zz) <= 1e-18) return undefined

  const spread = Math.hypot(yy - zz, 2 * yz)
  const majorVariance = (yy + zz + spread) / 2
  const minorVariance = (yy + zz - spread) / 2
  const rankTwo = minorVariance > Math.max(1e-24, majorVariance * 1e-12)
  const angle = Math.atan2(2 * yz, yy - zz) / 2
  const cosine = Math.cos(angle)
  const sine = Math.sin(angle)
  const distances = centered.map(([y, z]) => {
    const along = y * cosine + z * sine
    const across = -y * sine + z * cosine
    return along ** 2 / majorVariance + (rankTwo ? across ** 2 / minorVariance : 0)
  }).sort((a, b) => a - b)
  const index = 0.95 * (count - 1)
  const low = Math.floor(index)
  const cutoff = distances[low] + (index - low) * (distances[low + 1] - distances[low])
  return {
    center, angle,
    major: Math.sqrt(majorVariance * cutoff),
    minor: rankTwo ? Math.sqrt(minorVariance * cutoff) : 0,
  }
}

export function samplePrediction(
  observations: Observations,
  wind: Vec3,
  noiseMillimeters: number,
  initial: Motion,
): SampleSummary | undefined {
  if (noiseMillimeters === 0) return undefined
  if (!Number.isFinite(noiseMillimeters) || noiseMillimeters < 0) {
    throw new Error('Position noise must be finite and nonnegative.')
  }

  const counts: SampleSummary['counts'] = {
    goal_inside: 0, goal_outside: 0, ground_first: 0, timeout: 0, fit_failure: 0, numerical_failure: 0,
  }
  const crossings: Vec2[] = []
  const normal = normalGenerator()
  const sigma = noiseMillimeters / 1000
  const perturb = ([x, y, z]: Vec3): Vec3 => [
    x + sigma * normal(), y + sigma * normal(), z + sigma * normal(),
  ]

  for (let i = 0; i < sampleCount; i++) {
    const sampled: Observations = [perturb(observations[0]), perturb(observations[1])]
    let motion: Motion
    try {
      motion = fit(sampled, wind, { expanded: true, initial }).motion
    } catch {
      counts.fit_failure++
      continue
    }
    const flight = simulate(motion, wind, undefined, false)
    counts[flight.event]++
    if (flight.crossing) crossings.push([flight.crossing.y, flight.crossing.z])
  }

  return { counts, region: predictionRegion(crossings) }
}
