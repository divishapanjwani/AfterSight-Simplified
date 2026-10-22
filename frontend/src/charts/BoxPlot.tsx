/** One horizontal box per damage class: whiskers p5–p95, box q1–q3, median tick. Pattern + label per class. */
import { DAMAGE, DAMAGE_CLASSES, DamagePatternDefs } from '../theme/damage'
import { fmtValue } from '../theme/features'
import type { DamageClass } from '../types/api'

type Stats = { n: number; p5: number | null; q1: number | null; median: number | null; q3: number | null; p95: number | null }

export function BoxPlot({ perClass, width = 620, label }: { perClass: Record<string, Stats>; width?: number; label: string }) {
  const left = 120
  const right = 20
  const rowH = 34
  const vals = DAMAGE_CLASSES.flatMap((c) => [perClass[c]?.p5, perClass[c]?.p95]).filter((v): v is number => v !== null && v !== undefined)
  const lo = Math.min(...vals)
  const hi = Math.max(...vals)
  const span = hi - lo || 1
  const x = (v: number) => left + ((v - lo) / span) * (width - left - right)
  const height = DAMAGE_CLASSES.length * rowH + 30
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => lo + t * span)
  return (
    <svg width={width} height={height} role="img" aria-label={`Distribution of ${label} per damage class (training split)`}>
      <DamagePatternDefs idPrefix="bx-" />
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={4} y2={height - 24} stroke="var(--rule)" />
          <text x={x(t)} y={height - 10} fontSize="10" textAnchor="middle" fill="var(--ink-3)" className="num">{fmtValue(t)}</text>
        </g>
      ))}
      {DAMAGE_CLASSES.map((c, i) => {
        const s = perClass[c]
        const y = 10 + i * rowH
        if (!s || s.q1 === null || s.q3 === null || s.median === null || s.p5 === null || s.p95 === null) {
          return <text key={c} x={left} y={y + 12} fontSize="11" fill="var(--ink-3)">no {DAMAGE[c as DamageClass].label.toLowerCase()} buildings</text>
        }
        return (
          <g key={c}>
            <title>{`${DAMAGE[c].label}: median ${fmtValue(s.median)}, middle half ${fmtValue(s.q1)}–${fmtValue(s.q3)}, 5–95% ${fmtValue(s.p5)}–${fmtValue(s.p95)} (n=${s.n})`}</title>
            <text x={left - 10} y={y + 13} fontSize="12" textAnchor="end" fill="var(--ink)">{DAMAGE[c].label}</text>
            <line x1={x(s.p5)} x2={x(s.p95)} y1={y + 9} y2={y + 9} stroke={DAMAGE[c].color} strokeWidth="1.5" />
            <rect x={x(s.q1)} y={y} width={Math.max(1, x(s.q3) - x(s.q1))} height={18} fill={`url(#bx-${DAMAGE[c].pattern})`} stroke={DAMAGE[c].color} strokeWidth="1.5" />
            <line x1={x(s.median)} x2={x(s.median)} y1={y - 2} y2={y + 20} stroke="var(--ink)" strokeWidth="2" />
          </g>
        )
      })}
    </svg>
  )
}
