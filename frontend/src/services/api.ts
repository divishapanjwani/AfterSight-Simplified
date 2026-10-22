/** Thin fetch wrapper. All requests go to /api, proxied to the FastAPI backend by Vite. */

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export async function getJson<T>(path: string): Promise<T> {
  let response: Response
  try {
    response = await fetch(`/api${path}`)
  } catch {
    throw new ApiError(0, 'Backend unreachable — is the API running on port 8020?')
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new ApiError(response.status, body?.error?.message ?? `Request failed (${response.status})`)
  }
  return (await response.json()) as T
}

export interface Health {
  status: string
  version: string
  scope: string
}

export const fetchHealth = () => getJson<Health>('/health')
