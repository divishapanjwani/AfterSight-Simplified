import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('App shell', () => {
  it('shows backend status when the API responds', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ status: 'ok', version: '0.1.0', scope: 'x' }), { status: 200 }),
      ),
    )
    renderApp()
    expect(await screen.findByText('API ok · v0.1.0')).toBeInTheDocument()
  })

  it('shows an error when the backend is unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network')))
    renderApp()
    expect(await screen.findByRole('alert')).toHaveTextContent('Backend unreachable')
  })
})
