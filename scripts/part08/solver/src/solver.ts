import { fit, type FitResult, type Observations } from './fit.ts'
import { simulate, type Flight, type Motion, type Vec3 } from './flight.ts'
import type { SampleSummary } from './uncertainty.ts'

export type SolveRequest =
  | { mode: 'inverse'; wind: Vec3; observations: Observations; noise: number }
  | { mode: 'forward'; wind: Vec3; motion: Motion }

type NominalResult =
  | { kind: 'flight'; flight: Flight; fit?: FitResult }
  | { kind: 'error'; message: string }

export type SolveMessage = NominalResult
  | { kind: 'samples'; result: SampleSummary }
  | { kind: 'sample_error' }

export function solve(request: SolveRequest): NominalResult {
  try {
    if (request.mode === 'inverse') {
      const inverse = fit(request.observations, request.wind)
      return { kind: 'flight', flight: simulate(inverse.motion, request.wind), fit: inverse }
    }
    return { kind: 'flight', flight: simulate(request.motion, request.wind) }
  } catch (error) {
    return { kind: 'error', message: error instanceof Error ? error.message : 'Calculation failed.' }
  }
}
