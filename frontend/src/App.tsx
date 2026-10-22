import { useQuery } from '@tanstack/react-query'
import { fetchHealth } from './services/api'

/** Phase 0 shell: proves the frontend can reach the backend. Real pages arrive in Phase 5. */
export default function App() {
  const health = useQuery({ queryKey: ['health'], queryFn: fetchHealth, retry: false })

  return (
    <main className="mx-auto max-w-3xl px-6 py-16">
      <h1 className="text-3xl font-semibold tracking-tight">AfterSight</h1>
      <p className="mt-2 text-sm">Damage classification on building footprints derived from xBD annotations.</p>

      <section className="mt-8 border border-ink/20 bg-white/60 p-4 font-mono text-sm" aria-live="polite">
        {health.isPending && <span>Checking backend…</span>}
        {health.isError && <span role="alert">Backend not reachable: {health.error.message}</span>}
        {health.isSuccess && (
          <span>
            API {health.data.status} · v{health.data.version}
          </span>
        )}
      </section>
    </main>
  )
}
