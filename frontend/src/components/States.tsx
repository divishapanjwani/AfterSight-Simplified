/** Designed loading / empty / error / not-ready states, used by every page. */
import { useState, type ReactNode } from 'react'
import { ApiError } from '../services/api'

export function CommandBlock({ command }: { command: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false) // clipboard blocked: the command stays selectable
    }
  }
  return (
    <div className="mt-2 flex items-center gap-2 rounded-[3px] border border-rule bg-sheet-2 px-2 py-1.5">
      <code className="num flex-1 truncate text-xs text-ink" title={command}>$ {command}</code>
      <button type="button" onClick={copy} className="rounded-[3px] border border-rule-strong px-1.5 py-0.5 text-[11px] text-ink-2 hover:bg-sheet">
        {copied ? 'Copied' : 'Copy'}
      </button>
    </div>
  )
}

export function LoadingState({ label, rows = 3 }: { label: string; rows?: number }) {
  return (
    <div role="status" aria-live="polite" className="py-2">
      <p className="mb-3 font-mono text-xs text-ink-3">{label}</p>
      <div className="space-y-2" aria-hidden="true">
        {Array.from({ length: rows }, (_, i) => (
          <div key={i} className="h-3 animate-pulse rounded-sm bg-sheet-2" style={{ width: `${90 - i * 15}%` }} />
        ))}
      </div>
    </div>
  )
}

export function EmptyState({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-start gap-2 rounded-[4px] border border-dashed border-rule-strong px-4 py-6">
      <p className="text-sm font-medium text-ink">{title}</p>
      {children && <div className="text-sm text-ink-2">{children}</div>}
      {action}
    </div>
  )
}

/** Not built / not trained: neutral tone (it is a setup step, not a failure) + the command that fixes it. */
export function NotReadyState({ title, message, fix }: { title: string; message?: string; fix?: string | null }) {
  return (
    <div role="status" className="rounded-[4px] border border-dashed border-ink-3 bg-sheet-2/50 px-4 py-4">
      <p className="flex items-center gap-2 text-sm font-medium text-ink"><span aria-hidden="true" className="font-mono text-ink-3">○</span>{title}</p>
      {message && <p className="mt-1 text-sm text-ink-2">{message}</p>}
      {fix && <CommandBlock command={fix} />}
    </div>
  )
}

export function ErrorState({ error, what }: { error: unknown; what: string }) {
  const e = error instanceof ApiError ? error : new ApiError(0, 'unknown', String(error))
  if (e.notReady) {
    return <NotReadyState title={`Not available yet: ${what}`} message={e.message} fix={e.fix} />
  }
  return (
    <div role="alert" className="rounded-[4px] border border-danger bg-danger-soft px-4 py-4">
      <p className="flex items-center gap-2 text-sm font-medium text-danger"><span aria-hidden="true">✕</span>Could not load {what}</p>
      <p className="mt-1 text-sm text-ink">{e.message}</p>
      <p className="num mt-1 text-[11px] text-ink-3">code: {e.code}{e.status ? ` · HTTP ${e.status}` : ''}</p>
      {e.fix && <CommandBlock command={e.fix} />}
    </div>
  )
}
