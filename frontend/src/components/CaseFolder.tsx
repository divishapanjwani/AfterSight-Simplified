import type { ReactNode } from 'react'

/** A card styled as a file folder: a thin tab label sits on the top edge, e.g. "CASE · B-0142". */
export function CaseFolder({ tab, title, aside, children, tone = 'default', className = '' }: {
  tab: string
  title?: ReactNode
  aside?: ReactNode
  children: ReactNode
  tone?: 'default' | 'caution' | 'danger'
  className?: string
}) {
  const rule = tone === 'caution' ? 'border-caution' : tone === 'danger' ? 'border-danger' : 'border-rule'
  return (
    <section className={`relative mt-3 ${className}`}>
      <div className={`absolute -top-[17px] left-4 rounded-t-[4px] border border-b-0 ${rule} bg-sheet-2 px-2 py-0.5 font-mono text-[10px] tracking-[0.12em] text-ink-3 uppercase`}>
        {tab}
      </div>
      <div className={`rounded-[4px] border ${rule} bg-sheet p-4 shadow-folder`}>
        {(title || aside) && (
          <header className="mb-3 flex items-start justify-between gap-3">
            {title && <h2 className="text-[15px] font-semibold tracking-tight text-ink">{title}</h2>}
            {aside}
          </header>
        )}
        {children}
      </div>
    </section>
  )
}
