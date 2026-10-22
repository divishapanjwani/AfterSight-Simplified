/** Scatter in a fixed box. Each group has its own colour AND marker shape, so identity never relies on colour. */
import { fmtValue } from '../theme/features'

export type Marker = 'circle' | 'square' | 'triangle' | 'diamond'
export interface ScatterGroup { key: string; label: string; color: string; marker: Marker }

function Mark({ marker, x, y, color }: { marker: Marker; x: number; y: number; color: string }) {
  const r = 2.6
  if (marker === 'square') return <rect x={x - r} y={y - r} width={2 * r} height={2 * r} fill={color} />
  if (marker === 'triangle') return <path d={`M${x},${y - r * 1.2}L${x + r * 1.1},${y + r}L${x - r * 1.1},${y + r}Z`} fill={color} />
  if (marker === 'diamond') return <path d={`M${x},${y - r * 1.3}L${x + r * 1.3},${y}L${x},${y + r * 1.3}L${x - r * 1.3},${y}Z`} fill={color} />
  return <circle cx={x} cy={y} r={r} fill={color} />
}

export function ScatterPlot({ points, groups, xLabel, yLabel, width = 560, height = 400, logX = false, logY = false }: {
  points: { x: number; y: number; group: string }[]; groups: ScatterGroup[]; xLabel: string; yLabel: string
  width?: number; height?: number; logX?: boolean; logY?: boolean
}) {
  const pad = { l: 56, r: 26, t: 10, b: 40 }
  const tx = (v: number) => (logX ? Math.log10(Math.max(v, 1e-9)) : v)
  const ty = (v: number) => (logY ? Math.log10(Math.max(v, 1e-9)) : v)
  const xs = points.map((p) => tx(p.x))
  const ys = points.map((p) => ty(p.y))
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)]
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)]
  const sx = (v: number) => pad.l + ((tx(v) - x0) / (x1 - x0 || 1)) * (width - pad.l - pad.r)
  const sy = (v: number) => height - pad.b - ((ty(v) - y0) / (y1 - y0 || 1)) * (height - pad.t - pad.b)
  const byKey = new Map(groups.map((g) => [g.key, g]))
  const inv = (t: number, lo: number, hi: number, log: boolean) => { const v = lo + t * (hi - lo); return log ? 10 ** v : v }
  return (
    <figure>
      <ul className="mb-2 flex flex-wrap gap-4 text-xs text-ink-2">
        {groups.map((g) => (
          <li key={g.key} className="flex items-center gap-1.5"><svg width="10" height="10" aria-hidden="true"><Mark marker={g.marker} x={5} y={5} color={g.color} /></svg>{g.label}</li>
        ))}
      </ul>
      <svg width={width} height={height} role="img" aria-label={`Scatter of ${yLabel} against ${xLabel}`}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={pad.l + t * (width - pad.l - pad.r)} x2={pad.l + t * (width - pad.l - pad.r)} y1={pad.t} y2={height - pad.b} stroke="var(--rule)" />
            <text x={pad.l + t * (width - pad.l - pad.r)} y={height - pad.b + 14} fontSize="10" textAnchor="middle" fill="var(--ink-3)" className="num">{fmtValue(inv(t, x0, x1, logX))}</text>
            <line x1={pad.l} x2={width - pad.r} y1={height - pad.b - t * (height - pad.t - pad.b)} y2={height - pad.b - t * (height - pad.t - pad.b)} stroke="var(--rule)" />
            <text x={pad.l - 6} y={height - pad.b - t * (height - pad.t - pad.b) + 3} fontSize="10" textAnchor="end" fill="var(--ink-3)" className="num">{fmtValue(inv(t, y0, y1, logY))}</text>
          </g>
        ))}
        {points.map((p, i) => {
          const g = byKey.get(p.group)
          return g ? <Mark key={i} marker={g.marker} x={sx(p.x)} y={sy(p.y)} color={g.color} /> : null
        })}
        <text x={pad.l + (width - pad.l - pad.r) / 2} y={height - 6} fontSize="11" textAnchor="middle" fill="var(--ink-2)">{xLabel}{logX ? ' (log)' : ''}</text>
        <text x={12} y={pad.t + (height - pad.t - pad.b) / 2} fontSize="11" textAnchor="middle" fill="var(--ink-2)" transform={`rotate(-90 12 ${pad.t + (height - pad.t - pad.b) / 2})`}>{yLabel}{logY ? ' (log)' : ''}</text>
      </svg>
    </figure>
  )
}
