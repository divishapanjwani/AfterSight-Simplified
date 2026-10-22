import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render } from '@testing-library/react'
import type { ReactElement } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import { ThemeProvider } from '../theme/ThemeProvider'

type Route = { status?: number; body: unknown } | ((url: URL) => { status?: number; body: unknown })

/** Mock fetch by path prefix; records every requested URL. Unmatched paths -> 404 structured error. */
export function mockApi(routes: Record<string, Route>) {
  const calls: URL[] = []
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), 'http://localhost')
    calls.push(url)
    const key = Object.keys(routes).sort((a, b) => b.length - a.length).find((k) => url.pathname === k || url.pathname.startsWith(`${k}/`))
    const route = key ? routes[key] : { status: 404, body: { error: { code: 'not_found', message: 'No such endpoint or resource.' } } }
    const { status = 200, body } = typeof route === 'function' ? route(url) : route
    return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
  })
  vi.stubGlobal('fetch', fetchMock)
  return calls
}

export function renderAt(ui: ReactElement, path = '/') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <ThemeProvider>
        <MemoryRouter initialEntries={[path]}>{ui}</MemoryRouter>
      </ThemeProvider>
    </QueryClientProvider>,
  )
}

export const health = { status: 'ok', version: '0.1.0', scope: 'Damage classification on building footprints derived from xBD annotations.' }
