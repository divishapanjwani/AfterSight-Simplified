/** Fetch wrapper for the AfterSight API (proxied by Vite from /api to http://localhost:8020). */

import type { Comparison, DatasetStatus, Health, ModelList, SceneList } from '../types/api'

export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly fix: string | null

  constructor(status: number, code: string, message: string, fix: string | null = null) {
    super(message)
    this.status = status
    this.code = code
    this.fix = fix
  }

  /** Data or artifacts not built yet (503) or a model not trained (409 model_not_trained). */
  get notReady(): boolean {
    return this.status === 503 || this.code === 'model_not_trained' || this.code === 'results_not_found'
  }
}

export async function getJson<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
  const query = new URLSearchParams()
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== undefined && value !== '') query.set(key, String(value))
  }
  const qs = query.toString()
  const url = `/api${path}${qs ? `?${qs}` : ''}`
  let response: Response
  try {
    response = await fetch(url)
  } catch {
    throw new ApiError(0, 'backend_unreachable', 'The AfterSight API is not reachable.', 'cd backend && uvicorn app.main:app --port 8020')
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    const error = body?.error
    throw new ApiError(response.status, error?.code ?? 'http_error', error?.message ?? `Request failed (${response.status})`, error?.fix ?? null)
  }
  return (await response.json()) as T
}

export const api = {
  health: () => getJson<Health>('/health'),
  datasetStatus: () => getJson<DatasetStatus>('/dataset/status'),
  models: () => getJson<ModelList>('/models'),
  scenes: (params: Record<string, string | number | boolean | undefined>) => getJson<SceneList>('/scenes', params),
  comparison: (design: string, scheme = 'four_class') => getJson<Comparison>('/results/comparison', { design, scheme }),
}

export const sceneImageUrl = (sceneId: string, phase: 'pre' | 'post', size: 'thumb' | 'view' = 'thumb') =>
  `/api/scenes/${encodeURIComponent(sceneId)}/image/${phase}?size=${size}`
