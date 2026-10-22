/** Small single-series line (elbow / silhouette curves) with a marked x position. */
import { fmtValue } from '../theme/features'

const compact = (v: number) => (Math.abs(v) >= 10_000 ? `${(v / 1000).toLocaleString('en-GB', { maximumFractionDigits: 0 })}k` : fmtValue(v))

export function LineChart({ points, mark, label, width = 300, height = 150 }: { points: { x: number; y: number }[]; mark?: number; label: string; width?: number; height?: number }) {
  const pad = { l: 46, r: 10, t: 8, b: 28 }
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
  const sx = (v: number) => pad.l + ((v - x0) / (x1 - x0 || 1)) * (width - pad.l - pad.r)
  const sy = (v: number) => height - pad.b - ((v - y0) / (y1 - y0 || 1)) * (height - pad.t - pad.b)
  return (
    <figure>
      <figcaption className="mb-1 text-xs text-ink-2">{label}</figcaption>
      <svg width={width} height={height} role="img" aria-label={label}>
        {[y0, (y0 + y1) / 2, y1].map((v) => (
          <g key={v}><line x1={pad.l} x2={width - pad.r} y1={sy(v)} y2={sy(v)} stroke="var(--rule)" />
            <text x={pad.l - 6} y={sy(v) + 3} fontSize="10" textAnchor="end" fill="var(--ink-3)" className="num">{compact(v)}</text></g>
        ))}
        {mark !== undefined && <line x1={sx(mark)} x2={sx(mark)} y1={pad.t} y2={height - pad.b} stroke="var(--ink)" strokeDasharray="3 3" />}
        <path d={points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x)},${sy(p.y)}`).join('')} fill="none" stroke="var(--cat-1)" strokeWidth="2" />
        {points.map((p) => (
          <g key={p.x}><title>{`k=${p.x}: ${fmtValue(p.y)}`}</title><circle cx={sx(p.x)} cy={sy(p.y)} r="4" fill="var(--cat-1)" stroke="var(--sheet)" strokeWidth="2" />
            <text x={sx(p.x)} y={height - pad.b + 14} fontSize="10" textAnchor="middle" fill="var(--ink-3)" className="num">{p.x}</text></g>
        ))}
      </svg>
    </figure>
  )
}
