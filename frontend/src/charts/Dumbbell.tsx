/** Dumbbell: two values per row on one 0-1 axis (e.g. within- vs cross-event macro-F1). Direct labels. */
export interface DumbbellRow { label: string; a: number | null; b: number | null; note?: string }

export function Dumbbell({ rows, aLabel, bLabel, width = 640, axisLabel }: { rows: DumbbellRow[]; aLabel: string; bLabel: string; width?: number; axisLabel: string }) {
  const left = 190
  const right = 104
  const rowH = 30
  const plotW = width - left - right
  const x = (v: number) => left + v * plotW
  const height = rows.length * rowH + 36
  return (
    <figure>
      <div className="mb-2 flex gap-4 text-xs text-ink-2">
        <span className="flex items-center gap-1.5"><svg width="10" height="10" aria-hidden="true"><circle cx="5" cy="5" r="4.5" fill="var(--cat-1)" /></svg>{aLabel}</span>
        <span className="flex items-center gap-1.5"><svg width="10" height="10" aria-hidden="true"><rect x="0.5" y="0.5" width="9" height="9" fill="var(--cat-2)" /></svg>{bLabel}</span>
      </div>
      <svg width={width} height={height} role="img" aria-label={`${aLabel} vs ${bLabel}`}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={4} y2={height - 26} stroke="var(--rule)" />
            <text x={x(t)} y={height - 12} textAnchor="middle" fontSize="10" fill="var(--ink-3)" className="num">{t.toFixed(2)}</text>
          </g>
        ))}
        <text x={left + plotW / 2} y={height} textAnchor="middle" fontSize="10" fill="var(--ink-3)">{axisLabel}</text>
        {rows.map((r, i) => {
          const y = 16 + i * rowH
          return (
            <g key={r.label}>
              <text x={left - 10} y={y + 4} textAnchor="end" fontSize="12" fill="var(--ink)">{r.label}</text>
              {r.a !== null && r.b !== null && <line x1={x(r.a)} x2={x(r.b)} y1={y} y2={y} stroke="var(--ink-3)" strokeWidth="2" />}
              {r.b !== null && (
                <g><title>{`${r.label} · ${bLabel}: ${r.b.toFixed(3)}`}</title>
                  <rect x={x(r.b) - 5} y={y - 5} width="10" height="10" fill="var(--cat-2)" stroke="var(--sheet)" strokeWidth="2" /></g>
              )}
              {r.a !== null && (
                <g><title>{`${r.label} · ${aLabel}: ${r.a.toFixed(3)}`}</title>
                  <circle cx={x(r.a)} cy={y} r="5.5" fill="var(--cat-1)" stroke="var(--sheet)" strokeWidth="2" /></g>
              )}
              <text x={width - right + 8} y={y + 4} fontSize="11" fill="var(--ink-2)" className="num">
                {r.a?.toFixed(2) ?? '—'} → {r.b?.toFixed(2) ?? '—'}
              </text>
            </g>
          )
        })}
      </svg>
    </figure>
  )
}
