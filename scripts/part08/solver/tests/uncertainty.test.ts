import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { fit, type Observations } from '../src/fit.ts'
import type { Vec3 } from '../src/flight.ts'
import { predictionRegion, samplePrediction, type SampleSummary } from '../src/uncertainty.ts'

interface Reference {
  noise: number
  wind: Vec3
  counts: SampleSummary['counts']
  center: [number, number]
  area: number
}

const observations: Observations = [[1.5, 0.45, 0.44], [2.96, 0.86, 0.74]]
const references = JSON.parse(readFileSync(
  new URL('../../../../data/part08/uncertainty_reference.json', import.meta.url), 'utf8',
)) as Reference[]

for (const reference of references) {
  test(`matches 2,000 Python samples at ${reference.noise} mm and wind ${reference.wind.join(', ')}`, () => {
    const nominal = fit(observations, reference.wind)
    const result = samplePrediction(observations, reference.wind, reference.noise, nominal.motion)
    assert.ok(result)
    assert.deepEqual(result.counts, reference.counts)
    assert.ok(result.region)
    const { center, major, minor } = result.region
    assert.ok(Math.hypot(center[0] - reference.center[0], center[1] - reference.center[1]) < 0.001)
    assert.ok(Math.abs(Math.PI * major * minor - reference.area) < 0.005)
  })
}

test('skips sampling when observations are exact', () => {
  const wind: Vec3 = [0, 0, 0]
  assert.equal(samplePrediction(observations, wind, 0, fit(observations, wind).motion), undefined)
})

test('omits a region with too few crossings or zero spread', () => {
  assert.equal(predictionRegion([[0, 0], [1, 1]]), undefined)
  assert.equal(predictionRegion(Array.from({ length: 12 }, () => [1, 2])), undefined)
})

test('represents a rank-one cloud as a segment along the principal axis', () => {
  const region = predictionRegion(Array.from({ length: 20 }, (_, i) => [i, i * 2]))
  assert.ok(region)
  assert.ok(region.major > 0)
  assert.equal(region.minor, 0)
  assert.ok(Math.abs(Math.tan(region.angle) - 2) < 1e-12)
})
