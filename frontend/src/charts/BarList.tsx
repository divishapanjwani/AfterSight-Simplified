/** Horizontal value bars with the number written next to each bar (recall, importance, sizes). */
export function BarList({ items, max = 1, digits = 3, color = 'var(--ink-2)', width = 160 }: {
  items: { label: string; value: number | null; hint?: string }[]; max?: number; digits?: number; color?: string; width?: number
}) {
  return (
    <ul className="space-y-1">
      {items.map((it) => (
        <li key={it.label} className="grid items-center gap-2 text-xs" style={{ gridTemplateColumns: `minmax(0,1fr) ${width}px 48px` }} title={it.hint}>
          <span className="truncate text-ink-2">{it.label}</span>
          <span className="h-2.5 rounded-[1px] bg-sheet-2">
            {it.value !== null && <span className="block h-2.5 rounded-[1px]" style={{ width: `${Math.max(0, Math.min(1, it.value / max)) * 100}%`, background: color }} />}
          </span>
          <span className="num text-right text-ink">{it.value === null ? '—' : digits === 0 ? it.value.toLocaleString('en-GB') : it.value.toFixed(digits)}</span>
        </li>
      ))}
    </ul>
  )
}
