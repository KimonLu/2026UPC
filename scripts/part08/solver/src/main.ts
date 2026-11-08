import './style.css'
import { release, type Flight, type Motion, type Vec3 } from './flight.ts'
import { times, type FitResult, type Observations } from './fit.ts'
import { renderGoal } from './goal-plot.ts'
import { mountScene } from './scene.ts'
import type { SolveMessage, SolveRequest } from './solver.ts'
import { sampleCount, type SampleSummary } from './uncertainty.ts'

const titles = { inverse: 'Three-Point Prediction', forward: 'Forward Simulation' }
type Mode = keyof typeof titles

const observations: Observations = [[1.5, 0.45, 0.44], [2.96, 0.86, 0.74]]
const calm: Vec3 = [0, 0, 0]
const defaultMotion: Motion = {
  velocity: [30.41543629298755, 9.417331506222395, 6.907303300994369],
  cd: 0.37897788891169476,
  spin: [13.969170172225922, -5.452907986349557, -58.189274351631916],
}

function byId<T extends HTMLElement | SVGElement = HTMLElement>(id: string) {
  const element = document.getElementById(id)
  if (!element) throw new Error(`Missing element: ${id}`)
  return element as unknown as T
}

const form = byId<HTMLFormElement>('solver-form')
const chart = byId<SVGSVGElement>('goal-plot')
const status = byId('status')
const time = byId('event-time')
const hit = byId('goal-result')
const fitted = byId('fit-panel')
const showFlight = mountScene(byId('scene'), byId<HTMLButtonElement>('reset-view'))

const axes = ['x', 'y', 'z']
let active: Worker | undefined
let currentMode: Mode = 'inverse'

function field(name: string, label: string, value: number, min?: number) {
  return `<input name="${name}" aria-label="${label}" type="number" step="any" value="${value}"${min === undefined ? '' : ` min="${min}"`} required>`
}

function vector(name: string, label: string, value: Vec3, suffix = '') {
  return axes.map((axis, i) => field(`${name}-${axis}`, `${label}${axis}${suffix}`, value[i])).join('')
}

function vectorField(name: string, label: string, unit: string, value: Vec3) {
  return `<fieldset><legend>${label} <span>${unit}</span></legend>
    <div class="vector-grid">${axes.map((axis) => `<span>${axis}</span>`).join('')}
    ${vector(name, `${label} `, value)}</div></fieldset>`
}

function formMarkup(mode: Mode): string {
  const entry = mode === 'inverse'
    ? `<fieldset>
        <legend>Observations <span>m</span></legend>
        <div class="observations-grid">
          <span>Time</span><span>x</span><span>y</span><span>z</span>
          <span>0.00 s</span>${release.map((value) => `<span class="fixed-value">${value.toFixed(2)}</span>`).join('')}
          ${observations.map((row, index) => (
            `<span>${times[index].toFixed(2)} s</span>${vector(`obs-${index}`, '', row, ` at ${times[index].toFixed(2)} s`)}`
          )).join('')}
        </div>
      </fieldset>
      <label class="single-field"><span>Position noise <i>σ</i></span>
        <select name="noise" aria-label="Position noise">
          <option value="0">0 mm</option>
          <option value="1">1 mm</option>
          <option value="5" selected>5 mm</option>
          <option value="10">10 mm</option>
        </select>
      </label>`
    : `${vectorField('velocity', 'Initial velocity', 'm/s', defaultMotion.velocity)}
      ${vectorField('spin', 'Angular velocity', 'rad/s', defaultMotion.spin)}
      <label class="single-field"><span>Drag coefficient <i>C<sub>D</sub></i></span>
        ${field('cd', 'Drag coefficient', defaultMotion.cd, 0)}</label>`

  return `<p class="eyebrow">INPUTS</p>
    ${entry}${vectorField('wind', 'Wind', 'm/s', calm)}
    <div class="form-actions"><button type="submit" class="calculate">Calculate</button>
    <button type="button" class="quiet-button" id="reset-inputs">Reset</button></div>`
}

function number(name: string) {
  return (form.elements.namedItem(name) as HTMLInputElement).valueAsNumber
}

function readVector(name: string): Vec3 {
  return [number(`${name}-x`), number(`${name}-y`), number(`${name}-z`)]
}

function readForm(): SolveRequest {
  const wind = readVector('wind')
  return currentMode === 'inverse'
    ? {
      mode: 'inverse', wind, observations: [readVector('obs-0'), readVector('obs-1')],
      noise: Number((form.elements.namedItem('noise') as HTMLSelectElement).value),
    }
    : { mode: 'forward', wind, motion: { velocity: readVector('velocity'), spin: readVector('spin'), cd: number('cd') } }
}

function format(value: number, digits = 3) {
  return (Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value).toFixed(digits)
}

function showFit(result?: FitResult) {
  fitted.hidden = !result
  if (!result) return
  const { motion, rms } = result
  const tuple = (values: Vec3) => `(${values.map((value) => format(value)).join(', ')})`
  const metrics = [
    ['Initial velocity', `${tuple(motion.velocity)} m/s`],
    ['Angular velocity', `${tuple(motion.spin)} rad/s`],
    ['Drag coefficient', format(motion.cd, 4)],
    ['Coordinate RMS', `${format(rms * 1000)} mm`],
  ]
  byId('fit-metrics').innerHTML = metrics.map(([label, value]) => (
    `<div><dt>${label}</dt><dd>${value}</dd></div>`
  )).join('')
}

function setStatus(text: string, kind = '') {
  status.textContent = text
  status.dataset.kind = kind
}

function showResult(flight: Flight, request: SolveRequest, inverse?: FitResult) {
  const labels: Record<Flight['event'], string> = {
    goal_inside: 'Goal',
    goal_outside: 'Outside goal',
    ground_first: 'Ground contact',
    timeout: 'Time limit',
    numerical_failure: 'Numerical failure',
  }
  setStatus(labels[flight.event], flight.event)
  const crossed = flight.event === 'goal_inside' || flight.event === 'goal_outside'
  byId('time-label').textContent = crossed ? 'Time to goal plane' : 'Flight time'
  time.textContent = flight.event === 'numerical_failure' ? '—' : `${format(flight.time, 4)} s`
  const point = flight.event === 'goal_inside' ? flight.crossing : undefined
  hit.textContent = point ? `(${format(point.y)}, ${format(point.z)}) m` : '—'
  renderGoal(chart, point)
  showFlight(flight, request.wind, request.mode === 'inverse' ? request.observations : undefined)
  showFit(inverse)
}

function sampleStatus(text = '') {
  const element = byId('sample-status')
  element.hidden = !text
  element.textContent = text
}

function showSamples(summary: SampleSummary, point?: Flight['crossing']) {
  sampleStatus()
  renderGoal(chart, point, summary.region)
  byId('region-key').hidden = !summary.region
  const counts = byId('sample-counts')
  const labels: Record<keyof SampleSummary['counts'], string> = {
    goal_inside: 'Goal',
    goal_outside: 'Outside goal',
    ground_first: 'Ground contact',
    timeout: 'Time limit',
    fit_failure: 'Fit failure',
    numerical_failure: 'Numerical failure',
  }
  counts.innerHTML = (Object.keys(labels) as (keyof typeof labels)[]).map((event) => (
    `<div><dt>${labels[event]}</dt><dd>${summary.counts[event]} / ${sampleCount}</dd></div>`
  )).join('')
  counts.hidden = false
}

function clearResults(message = 'Ready') {
  active?.terminate()
  active = undefined
  setStatus(message)
  byId('time-label').textContent = 'Time'
  time.textContent = '—'
  hit.textContent = '—'
  showFit()
  sampleStatus()
  byId('region-key').hidden = true
  byId('sample-counts').hidden = true
  renderGoal(chart)
  showFlight(undefined, calm)
}

function calculate() {
  clearResults('Calculating…')
  const request = readForm()
  const sampling = request.mode === 'inverse' && request.noise > 0
  let point: Flight['crossing']
  const worker = new Worker(new URL('./solver.worker.ts', import.meta.url), { type: 'module' })
  active = worker
  const stop = () => {
    worker.terminate()
    active = undefined
  }
  worker.addEventListener('message', (event: MessageEvent<SolveMessage>) => {
    if (worker !== active) return
    const result = event.data
    switch (result.kind) {
      case 'flight':
        showResult(result.flight, request, result.fit)
        point = result.flight.event === 'goal_inside' ? result.flight.crossing : undefined
        if (sampling) sampleStatus('Sampling 2,000 shots…')
        else stop()
        break
      case 'error':
        stop()
        setStatus(result.message, 'error')
        break
      case 'samples':
        stop()
        showSamples(result.result, point)
        break
      case 'sample_error':
        stop()
        sampleStatus('Sampling failed')
        break
    }
  })
  worker.addEventListener('error', () => {
    if (worker !== active) return
    stop()
    sampleStatus()
    setStatus('Calculation failed', 'error')
  })
  worker.postMessage(request)
}

function openMode() {
  currentMode = location.hash === '#/forward' ? 'forward' : 'inverse'
  byId('mode-title').textContent = titles[currentMode]
  document.title = `${titles[currentMode]} · Penalty Trajectory`
  byId('observation-key').hidden = currentMode !== 'inverse'
  document.querySelectorAll<HTMLAnchorElement>('[data-mode]').forEach((link) => {
    if (link.dataset.mode === currentMode) link.setAttribute('aria-current', 'page')
    else link.removeAttribute('aria-current')
  })
  form.innerHTML = formMarkup(currentMode)
  byId('reset-inputs').addEventListener('click', () => {
    form.reset()
    calculate()
  })
  calculate()
}

form.addEventListener('input', () => clearResults())
form.addEventListener('submit', (event) => {
  event.preventDefault()
  calculate()
})
window.addEventListener('hashchange', openMode)
openMode()
