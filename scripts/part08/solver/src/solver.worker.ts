import { solve, type SolveMessage, type SolveRequest } from './solver.ts'
import { samplePrediction } from './uncertainty.ts'

const worker = globalThis as unknown as DedicatedWorkerGlobalScope
const send = (message: SolveMessage) => worker.postMessage(message)

worker.addEventListener('message', (event: MessageEvent<SolveRequest>) => {
  const request = event.data
  const nominal = solve(request)
  send(nominal)
  if (request.mode !== 'inverse' || !request.noise || nominal.kind !== 'flight' || !nominal.fit) return

  try {
    const result = samplePrediction(request.observations, request.wind, request.noise, nominal.fit.motion)
    if (result) send({ kind: 'samples', result })
  } catch {
    send({ kind: 'sample_error' })
  }
})
