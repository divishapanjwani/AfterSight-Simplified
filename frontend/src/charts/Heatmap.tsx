/** Row-normalised heatmap (confusion matrix, cluster contingency). Counts + row % written in every cell. */
const STEPS = ['var(--mag-0)', 'var(--mag-1)', 'var(--mag-2)', 'var(--mag-3)', 'var(--mag-4)']

export function Heatmap({ rows, cols, matrix, rowTitle, colTitle, caption, cell = 64 }: {
  rows: string[]; cols: string[]; matrix: number[][]; rowTitle: string; colTitle: string; caption?: string; cell?: number
}) {
  const shares = matrix.map((r) => {
    const total = r.reduce((a, b) => a + b, 0)
    return r.map((v) => (total ? v / total : 0))
  })
  return (
    <figure>
      <table className="border-separate border-spacing-[2px] text-xs" aria-label={caption}>
        <thead>
          <tr>
            <th className="pr-2 text-right font-normal text-ink-3">{rowTitle} ↓ · {colTitle} →</th>
            {cols.map((c) => <th key={c} className="px-1 pb-1 text-center font-normal text-ink-2" style={{ width: cell }}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((r, i) => (
            <tr key={rows[i]}>
              <th scope="row" className="pr-2 text-right font-normal whitespace-nowrap text-ink-2">{rows[i]}</th>
              {r.map((v, j) => {
                const share = shares[i][j]
                const step = Math.min(4, Math.floor(share * 5))
                return (
                  <td key={j} title={`${rows[i]} → ${cols[j]}: ${v} (${(share * 100).toFixed(0)}% of row)`}
                    className="num h-11 rounded-[2px] text-center leading-tight" style={{ background: STEPS[step], color: step >= 3 ? 'var(--paper)' : 'var(--ink)' }}>
                    {v.toLocaleString('en-GB')}<br /><span className="text-[10px] opacity-80">{(share * 100).toFixed(0)}%</span>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {caption && <figcaption className="mt-1 text-[11px] text-ink-3">{caption}</figcaption>}
    </figure>
  )
}
