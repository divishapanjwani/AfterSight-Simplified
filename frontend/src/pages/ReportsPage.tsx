/** Reports: every saved analysis with its CSV and PDF exports. Provenance shown per row. */
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { CaseFolder } from '../components/CaseFolder'
import { SplitChip } from '../components/Chip'
import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { api } from '../services/api'

export function ReportsPage() {
  const q = useQuery({ queryKey: ['analyses'], queryFn: () => api.analyses(100) })
  return (
    <div className="space-y-5">
      <div>
        <p className="font-mono text-[11px] tracking-[0.14em] text-ink-3 uppercase">Reports</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink">Exports for each analysis</h1>
        <p className="mt-1 text-xs text-ink-3">Each PDF contains the scene, model, its split and any warning, ground-truth vs predicted counts, scene metrics,
          the confusion matrix, review suggestions and the limitations. The CSV has one row per building.</p>
      </div>
      {q.isPending && <LoadingState label="Loading saved analyses…" rows={4} />}
      {q.isError && <ErrorState error={q.error} what="saved analyses" />}
      {q.data && q.data.items.length === 0 && (
        <EmptyState title="No analyses yet.">
          Run an analysis on the <Link to="/triage" className="text-ink underline underline-offset-2">Triage Desk</Link> — its reports appear here.
        </EmptyState>
      )}
      {q.data && q.data.items.length > 0 && (
        <CaseFolder tab={`ANALYSES · ${q.data.total}`}>
          <table className="w-full text-xs">
            <thead><tr className="border-b border-rule text-left text-ink-3">
              <th className="py-1.5 font-normal">Scene</th><th className="py-1.5 font-normal">Split</th><th className="py-1.5 font-normal">Model</th>
              <th className="py-1.5 text-right font-normal">buildings</th><th className="py-1.5 text-right font-normal">scene macro-F1</th>
              <th className="py-1.5 text-right font-normal">suggested for human review</th><th className="py-1.5 pl-4 font-normal">Created</th><th className="py-1.5 text-right font-normal">Exports</th></tr></thead>
            <tbody>
              {q.data.items.map((a) => (
                <tr key={a.analysis_id} className="border-b border-rule/60 last:border-0">
                  <td className="py-1.5"><Link to={`/triage/${a.scene_id}?model=${a.model}&run=1`} className="num text-ink hover:underline">{a.scene_id}</Link></td>
                  <td className="py-1.5"><SplitChip split={a.split} /></td>
                  <td className="py-1.5 text-ink-2">{a.model_label}</td>
                  <td className="num py-1.5 text-right">{a.n_buildings}</td>
                  <td className="num py-1.5 text-right">{a.macro_f1?.toFixed(2) ?? 'not available'}</td>
                  <td className="num py-1.5 text-right">{a.n_review}</td>
                  <td className="num py-1.5 pl-4 text-ink-3">{a.created_at.replace('T', ' ').slice(0, 16)}</td>
                  <td className="py-1.5 text-right whitespace-nowrap">
                    <a href={a.csv_url} className="text-ink underline underline-offset-2">CSV</a>
                    <a href={a.pdf_url} className="ml-3 text-ink underline underline-offset-2">PDF</a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-[11px] text-ink-3">Scene-level scores are small samples; rows on training or validation scenes are optimistic (their PDFs carry the warning).</p>
        </CaseFolder>
      )}
    </div>
  )
}
