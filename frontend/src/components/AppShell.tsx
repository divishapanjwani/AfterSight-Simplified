import { useQuery } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { api } from '../services/api'
import { useTheme } from '../theme/ThemeProvider'
import { useWorkflow } from './Workflow'

const SCOPE = 'Damage classification on building footprints derived from xBD annotations.'

/** Workflow steps (left to right); links follow the scene/analysis/building the user is working on. */
function useSteps(): { to: string; label: string; ready: boolean; match: string }[] {
  const { state } = useWorkflow()
  const triage = state.sceneId ? `/triage/${state.sceneId}?model=${state.model ?? ''}${state.analysisId ? '&run=1' : ''}` : '/triage'
  const caseFile = state.analysisId && state.buildingUid ? `/case/${state.analysisId}/${state.buildingUid}` : '/case'
  return [
    { to: '/library?split=test', label: 'Library', ready: true, match: '/library' },
    { to: triage, label: 'Triage', ready: true, match: '/triage' },
    { to: caseFile, label: 'Case file', ready: true, match: '/case' },
    { to: '/arena', label: 'Arena', ready: true, match: '/arena' },
    { to: '/reports', label: 'Report', ready: true, match: '/reports' },
  ]
}
const TOOLS: { to: string; label: string; ready: boolean }[] = [
  { to: '/', label: 'Readiness', ready: true },
  { to: '/features', label: 'Feature Lab', ready: true },
  { to: '/clusters', label: 'Cluster Atlas', ready: true },
  { to: '/review', label: 'Review Queue', ready: true },
]

function ApiDot() {
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, retry: false, refetchInterval: 30_000 })
  const ok = health.isSuccess
  return (
    <span className="flex items-center gap-1.5 font-mono text-[11px] text-ink-3" title={ok ? `API v${health.data.version}` : 'API not reachable'}>
      <span aria-hidden="true" className={`inline-block h-2 w-2 rounded-full ${ok ? 'bg-ok' : health.isPending ? 'bg-ink-3' : 'bg-danger'}`} />
      {ok ? 'API online' : health.isPending ? 'API …' : 'API offline'}
    </span>
  )
}

function StepLink({ to, label, ready, index, match }: { to: string; label: string; ready: boolean; index?: number; match?: string }) {
  const { pathname } = useLocation()
  const prefix = index !== undefined ? <span className="num mr-1.5 text-ink-3">{String(index + 1).padStart(2, '0')}</span> : null
  if (!ready) {
    return (
      <span aria-disabled="true" title="Arrives in Phase 5 Part B" className="flex items-center px-3 py-2 text-sm text-ink-3/70">
        {prefix}{label}
      </span>
    )
  }
  return (
    <NavLink to={to} end={!match}
      className={({ isActive }) => {
        const active = match ? pathname.startsWith(match) : isActive
        return `flex items-center border-b-2 px-3 py-2 text-sm ${active ? 'border-ink font-medium text-ink' : 'border-transparent text-ink-2 hover:text-ink'}`
      }}>
      {prefix}{label}
    </NavLink>
  )
}

export function AppShell({ children }: { children: ReactNode }) {
  const { theme, toggle } = useTheme()
  const steps = useSteps()
  return (
    <div className="min-h-screen bg-paper">
      <header className="border-b border-rule bg-sheet">
        <div className="mx-auto flex max-w-[1240px] items-center justify-between gap-4 px-5 py-2.5">
          <div className="flex items-baseline gap-3">
            <span className="text-[17px] font-semibold tracking-tight text-ink">AfterSight</span>
            <span className="hidden text-xs text-ink-3 md:inline">{SCOPE}</span>
          </div>
          <div className="flex items-center gap-4">
            <ApiDot />
            <button type="button" onClick={toggle} aria-pressed={theme === 'dark'}
              className="rounded-[3px] border border-rule-strong px-2 py-1 text-xs text-ink-2 hover:bg-sheet-2">
              {theme === 'dark' ? '☾ Dark' : '☀ Light'}
            </button>
          </div>
        </div>
        <nav aria-label="Workflow" className="mx-auto flex max-w-[1240px] items-center justify-between px-5">
          <ol className="flex items-center">
            {steps.map((s, i) => (
              <li key={s.to} className="flex items-center">
                <StepLink {...s} index={i} />
                {i < steps.length - 1 && <span aria-hidden="true" className="text-ink-3/60">→</span>}
              </li>
            ))}
          </ol>
          <ul className="flex items-center">
            {TOOLS.map((t) => <li key={t.to}><StepLink {...t} /></li>)}
          </ul>
        </nav>
      </header>
      <main className="mx-auto max-w-[1240px] px-5 pt-6 pb-16">{children}</main>
    </div>
  )
}

export { SCOPE }
