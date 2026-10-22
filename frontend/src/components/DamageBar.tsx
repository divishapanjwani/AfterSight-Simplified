import type { DamageClass } from '../types/api'
import { DAMAGE, DAMAGE_CLASSES, DamagePatternDefs } from '../theme/damage'

/** Stacked bar of class counts (pattern + colour) with the counts written out for screen readers and eyes. */
export function DamageBar({ counts, width = 132, id }: { counts: Record<DamageClass, number>; width?: number; id: string }) {
  const total = DAMAGE_CLASSES.reduce((sum, c) => sum + (counts[c] ?? 0), 0)
  const text = DAMAGE_CLASSES.map((c) => `${DAMAGE[c].label} ${counts[c] ?? 0}`).join(', ')
  if (total === 0) return <span className="num text-xs text-ink-3">no buildings</span>
  let x = 0
  return (
    <div className="flex items-center gap-2" title={text}>
      <svg width={width} height="10" role="img" aria-label={`Ground truth: ${text}`}>
        <DamagePatternDefs idPrefix={`bar-${id}-`} />
        {DAMAGE_CLASSES.map((c) => {
          const w = ((counts[c] ?? 0) / total) * width
          const rect = w > 0 ? (
            <rect key={c} x={x} y={0} width={Math.max(w - 1, 0.5)} height={10}
              fill={`url(#bar-${id}-${DAMAGE[c].pattern})`} stroke={DAMAGE[c].color} strokeWidth="1" />
          ) : null
          x += w
          return rect
        })}
      </svg>
      <span className="num text-[11px] whitespace-nowrap text-ink-2" aria-hidden="true">
        {DAMAGE_CLASSES.map((c) => counts[c] ?? 0).join(' · ')}
      </span>
    </div>
  )
}
