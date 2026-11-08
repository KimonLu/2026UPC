import { goal } from './flight.ts'
import type { Region } from './uncertainty.ts'

const width = 440
const height = 272

function add(parent: SVGElement, tag: string, attributes: Record<string, string | number>, text?: string) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', tag)
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value))
  if (text !== undefined) element.textContent = text
  parent.append(element)
  return element
}

export function renderGoal(svg: SVGSVGElement, crossing?: { y: number; z: number }, region?: Region) {
  svg.replaceChildren()

  let minY = -4.25
  let maxY = 4.25
  let minZ = 0
  let maxZ = 2.9
  if (region) {
    const { center: [y, z], angle, major, minor } = region
    const across = Math.hypot(major * Math.cos(angle), minor * Math.sin(angle))
    const up = Math.hypot(major * Math.sin(angle), minor * Math.cos(angle))
    minY = Math.min(minY, y - across - 0.2)
    maxY = Math.max(maxY, y + across + 0.2)
    minZ = Math.min(minZ, z - up - 0.2)
    maxZ = Math.max(maxZ, z + up + 0.2)
  }
  const unit = Math.min((width - 88) / (maxY - minY), (height - 81) / (maxZ - minZ))
  const mapY = (y: number) => width / 2 + unit * (y - (minY + maxY) / 2)
  const mapZ = (z: number) => (height - 24) / 2 - unit * (z - (minZ + maxZ) / 2)
  const left = mapY(-goal.halfWidth)
  const right = mapY(goal.halfWidth)
  const top = mapZ(goal.height)
  const bottom = mapZ(0)

  add(svg, 'rect', { x: left, y: top, width: right - left, height: bottom - top, class: 'net-fill' })
  for (let y = -3.5; y <= 3.5; y += 0.5) {
    add(svg, 'line', { x1: mapY(y), x2: mapY(y), y1: top, y2: bottom, class: 'net-line' })
  }
  for (let z = 0.25; z < goal.height; z += 0.25) {
    add(svg, 'line', { x1: left, x2: right, y1: mapZ(z), y2: mapZ(z), class: 'net-line' })
  }
  if (region) {
    const cx = mapY(region.center[0])
    const cy = mapZ(region.center[1])
    const group = add(svg, 'g', {
      class: 'prediction-region', 'data-testid': 'prediction-region',
      transform: `translate(${cx} ${cy}) rotate(${-region.angle * 180 / Math.PI})`,
    })
    add(group, 'title', {}, 'Empirical 95% region among goal-plane crossings')
    if (region.minor > 0) {
      add(group, 'ellipse', { cx: 0, cy: 0, rx: unit * region.major, ry: unit * region.minor })
    } else {
      add(group, 'line', { x1: -unit * region.major, x2: unit * region.major, y1: 0, y2: 0 })
    }
  }
  add(svg, 'path', { d: `M ${left} ${bottom} V ${top} H ${right} V ${bottom}`, class: 'goal-frame' })
  add(svg, 'line', { x1: left - 12, x2: right + 12, y1: bottom, y2: bottom, class: 'chart-axis' })
  for (const y of [-3.66, -2, 0, 2, 3.66]) {
    add(svg, 'line', { x1: mapY(y), x2: mapY(y), y1: bottom, y2: bottom + 5, class: 'chart-axis' })
    add(svg, 'text', { x: mapY(y), y: bottom + 19, 'text-anchor': 'middle', class: 'chart-tick' }, String(y))
  }
  for (const z of [0, 1, 2.44]) {
    add(svg, 'text', { x: left - 8, y: mapZ(z) + 4, 'text-anchor': 'end', class: 'chart-tick' }, String(z))
  }
  add(svg, 'text', { x: width / 2, y: bottom + 38, 'text-anchor': 'middle', class: 'chart-label' }, 'y (m)')
  add(svg, 'text', { x: left - 7, y: top - 14, 'text-anchor': 'end', class: 'chart-label' }, 'z (m)')

  if (crossing) {
    const x = mapY(crossing.y)
    const y = mapZ(crossing.z)
    const mark = add(svg, 'g', { 'data-testid': 'goal-marker' })
    add(mark, 'title', {}, `y = ${crossing.y.toFixed(3)} m, z = ${crossing.z.toFixed(3)} m`)
    add(mark, 'circle', { cx: x, cy: y, r: 9, class: 'goal-halo' })
    add(mark, 'circle', { cx: x, cy: y, r: 4.5, class: 'goal-dot' })
  }
}
