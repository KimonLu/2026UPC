import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { fit, type Observations } from '../src/fit.ts'
import { insideGoal, simulate, type Motion, type Vec3 } from '../src/flight.ts'

interface ReferenceCase {
  name: string
  wind: Vec3
  motion: Motion
  event: string
  time: number
  y: number
  z: number
}

const { observations, cases } = JSON.parse(
  readFileSync(new URL('../../../../data/part08/wind_reference.json', import.meta.url), 'utf8'),
) as { observations: Observations; cases: ReferenceCase[] }

let maxDistance = 0
let maxTime = 0
for (const reference of cases) {
  test(reference.name, () => {
    const inverse = reference.name.startsWith('W1/')
      ? fit(observations, reference.wind)
      : undefined
    if (inverse) assert.ok(inverse.rms < 1e-7)
    const result = simulate(inverse?.motion ?? reference.motion, reference.wind)
    assert.equal(result.event, reference.event)
    assert.ok(result.crossing)
    const distance = Math.hypot(result.crossing.y - reference.y, result.crossing.z - reference.z)
    const time = Math.abs(result.time - reference.time)
    maxDistance = Math.max(maxDistance, distance)
    maxTime = Math.max(maxTime, time)
    assert.ok(distance < 0.001, `Goal plane deviation: ${distance} m`)
    assert.ok(time < 0.0001, `Event time deviation: ${time} s`)
  })
}

test.after(() => {
  console.log(`Maximum reference deviations: ${maxDistance * 1000} mm, ${maxTime * 1000} ms`)
})

test('only the open goal rectangle scores', () => {
  assert.equal(insideGoal(0, 0), false)
  assert.equal(insideGoal(3.66, 1), false)
  assert.equal(insideGoal(-3.66, 1), false)
  assert.equal(insideGoal(0, 2.44), false)
  assert.equal(insideGoal(3.65999, 2.43999), true)
})

test('stops at downward ground contact before reaching the goal', () => {
  const result = simulate({ velocity: [20, 0, 0.4], spin: [0, 0, 0], cd: 0.2 }, [0, 3, 0])
  assert.equal(result.event, 'ground_first')
  assert.ok(result.time > 0.05 && result.time < 0.1)
  assert.equal(result.crossing, undefined)
  assert.equal(result.points.at(-1)?.position[2], 0.11)
})

test('distinguishes a door-plane miss from a goal', () => {
  const result = simulate({ velocity: [30, 15, 5], spin: [0, 0, 0], cd: 0 }, [0, 0, 0])
  assert.equal(result.event, 'goal_outside')
  assert.ok(result.crossing && result.crossing.y > 3.66)
})

test('finds an outward crossing and return within the same integration step', () => {
  // With no drag and vertical spin, the horizontal motion is a circular arc.
  const result = simulate({
    velocity: [17.572081704558162, -34.524917226993004, 5],
    spin: [0, 0, 1141.525650452297],
    cd: 0,
  }, [0, 0, 0])
  assert.equal(result.event, 'goal_inside')
  assert.ok(Math.abs(result.time - 0.4001195788) < 2e-6)
  assert.ok(result.crossing && Math.abs(result.crossing.y - 2.60420916) < 0.0001)
})

test('reports time limit and immediate contact separately', () => {
  const motion = { velocity: [0, 0, 20], spin: [0, 0, 0], cd: 0 } as const
  assert.equal(simulate(motion, [0, 0, 0]).event, 'timeout')
  assert.equal(simulate({ ...motion, velocity: [20, 0, 0] }, [0, 0, 0]).time, 0)
})

test('rejects non-finite inputs, negative drag and a zero observation chord', () => {
  assert.throws(() => simulate({ velocity: [20, 0, 4], spin: [0, 0, 0], cd: -1 }, [0, 0, 0]), /drag/i)
  assert.throws(() => fit(observations, [Infinity, 0, 0]), /finite/i)
  assert.throws(() => fit([observations[0], [0, 0, 0.11]], [0, 0, 0]), /chord/i)
})

test('reports a trajectory that cannot fit within the observation window', () => {
  assert.throws(() => fit(observations, [0, 0, Number.MAX_VALUE]), /fit/i)
})
