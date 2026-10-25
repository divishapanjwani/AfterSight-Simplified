/** Fetch wrapper for the AfterSight API (proxied by Vite from /api to http://localhost:8020). */

import type {
  AnalysisResult, AnalysisSummary, Clusters, Comparison, Confounding, DatasetStatus, Explanation, FeatureSummary, Health, ModelList,
  ModelMetrics, ReviewQueue, Scatter, SceneDetail, SceneList, TreeView,
} from '../types/api'

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

async function toApiError(response: Response): Promise<ApiError> {
  const body = await response.json().catch(() => null)
  const error = body?.error
  return new ApiError(response.status, error?.code ?? 'http_error', error?.message ?? `Request failed (${response.status})`, error?.fix ?? null)
}

const unreachable = () =>
  new ApiError(0, 'backend_unreachable', 'The AfterSight API is not reachable.', 'source .venv/bin/activate && cd backend && uvicorn app.main:app --port 8020')

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  let response: Response
  try {
    response = await fetch(`/api${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  } catch {
    throw unreachable()
  }
  if (!response.ok) throw await toApiError(response)
  return (await response.json()) as T
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
    throw unreachable()
  }
  if (!response.ok) throw await toApiError(response)
  return (await response.json()) as T
}

export const api = {
  health: () => getJson<Health>('/health'),
  datasetStatus: () => getJson<DatasetStatus>('/dataset/status'),
  models: () => getJson<ModelList>('/models'),
  scenes: (params: Record<string, string | number | boolean | undefined>) => getJson<SceneList>('/scenes', params),
  comparison: (design: string, scheme = 'four_class') => getJson<Comparison>('/results/comparison', { design, scheme }),
  scene: (sceneId: string) => getJson<SceneDetail>(`/scenes/${encodeURIComponent(sceneId)}`),
  analyse: (sceneId: string, modelName: string) => postJson<AnalysisResult>('/analysis', { scene_id: sceneId, model_name: modelName }),
  analyses: (limit = 50) => getJson<{ total: number; items: AnalysisSummary[] }>('/analyses', { limit }),
  analysis: (analysisId: string) => getJson<AnalysisResult>(`/analysis/${encodeURIComponent(analysisId)}`),
  modelMetrics: (name: string, design: string, scheme = 'four_class') => getJson<ModelMetrics>(`/models/${name}/metrics`, { design, scheme }),
  tree: (maxDepth: number, design = 'within_event') => getJson<TreeView>('/models/decision_tree/tree', { max_depth: maxDepth, design }),
  confounding: () => getJson<Confounding>('/results/confounding'),
  featureSummary: () => getJson<FeatureSummary>('/features/summary'),
  scatter: (x: string, y: string, n = 1500) => getJson<Scatter>('/features/scatter', { x, y, n }),
  clusters: (k: number, design = 'within_event') => getJson<Clusters>('/clusters', { k, design }),
  reviewQueue: (model: string, offset: number, limit: number, split = 'test') => getJson<ReviewQueue>('/review-queue', { model, split, offset, limit }),
  explain: (analysisId: string, buildingUid: string) =>
    getJson<Explanation>(`/analysis/${encodeURIComponent(analysisId)}/buildings/${encodeURIComponent(buildingUid)}/explain`),
}

export const cropUrl = (buildingUid: string, phase: 'pre' | 'post', size: 96 | 192 | 384 = 192) =>
  `/api/buildings/${encodeURIComponent(buildingUid)}/crop/${phase}?size=${size}`
export const exportCsvUrl = (analysisId: string) => `/api/analysis/${encodeURIComponent(analysisId)}/export.csv`
export const reportPdfUrl = (analysisId: string) => `/api/analysis/${encodeURIComponent(analysisId)}/report.pdf`

export const sceneImageUrl = (sceneId: string, phase: 'pre' | 'post', size: 'thumb' | 'view' = 'thumb') =>
  `/api/scenes/${encodeURIComponent(sceneId)}/image/${phase}?size=${size}`
