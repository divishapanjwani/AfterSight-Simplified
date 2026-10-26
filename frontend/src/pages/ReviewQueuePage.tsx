/** Review Queue: buildings SUGGESTED FOR HUMAN REVIEW (low confidence or models disagree). Not "errors". */
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { Link, useSearchParams } from 'react-router-dom'
import { CaseFolder } from '../components/CaseFolder'
import { Chip } from '../components/Chip'
import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { api } from '../services/api'
import { DAMAGE, DamageSwatch } from '../theme/damage'

const PAGE = 25

export function ReviewQueuePage() {
  const [params, setParams] = useSearchParams()
  const models = useQuery({ queryKey: ['models'], queryFn: api.models })
  const model = params.get('model') ?? models.data?.default_model ?? 'knn'
  const page = Math.max(1, Number(params.get('page') ?? 1) || 1)
  const q = useQuery({ queryKey: ['review', model, page], queryFn: () => api.reviewQueue(model, (page - 1) * PAGE, PAGE), placeholderData: keepPreviousData })
  const set = (k: string, v: string) => { const n = new URLSearchParams(params); n.set(k, v); if (k !== 'page') n.delete('page'); setParams(n) }
  const d = q.data
  const pages = d ? Math.max(1, Math.ceil(d.total_flagged / PAGE)) : 1
  return (
    <div className="space-y-5">
      <div className="flex items-end justify-between">
        <div>
          <p className="font-mono text-[11px] tracking-[0.14em] text-ink-3 uppercase">Review queue</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink">Buildings suggested for human review</h1>
          <p className="mt-1 flex items-center gap-2 text-xs text-ink-3">Held-out test scenes <Chip kind="test" /></p>
        </div>
        <div role="radiogroup" aria-label="Model" className="flex overflow-hidden rounded-[3px] border border-rule-strong">
          {(models.data?.models ?? []).map((m) => (
            <button key={m.name} type="button" role="radio" aria-checked={model === m.name} disabled={!m.trained} onClick={() => set('model', m.name)}
              className={`border-r border-rule-strong px-2.5 py-1 text-xs last:border-r-0 disabled:opacity-40 ${model === m.name ? 'bg-ink text-paper' : 'bg-sheet text-ink-2 hover:bg-sheet-2'}`}>{m.label}</button>
          ))}
        </div>
      </div>
      {q.isPending && <LoadingState label="Building the review queue…" rows={5} />}
      {q.isError && <ErrorState error={q.error} what="the review queue" />}
      {d && (
        <>
          <CaseFolder tab="HOW THIS LIST IS MADE">
            <p className="text-sm text-ink-2">{d.framing}</p>
            <p className="mt-1 text-sm text-ink-2">
              <span className="num text-ink">{d.total_flagged.toLocaleString('en-GB')}</span> of <span className="num text-ink">{d.total_buildings.toLocaleString('en-GB')}</span> test buildings
              (<span className="num">{((d.total_flagged / Math.max(d.total_buildings, 1)) * 100).toFixed(0)}%</span>) are suggested for human review with this model:{' '}
              {d.threshold !== null ? (
                <>confidence below <span className="num">{d.threshold.toFixed(2)}</span> (chosen on validation{d.target_accuracy != null && <>: the lowest threshold at which confident buildings were at least <span className="num">{(d.target_accuracy * 100).toFixed(0)}%</span> correct</>}), or another model disagrees.</>
              ) : (
                <>another model disagrees with it.</>
              )}
              {' '}A suggestion is a prompt for a human to check, not a confirmed mistake.
            </p>
            {d.threshold === null && (
              <p role="note" className="mt-2 rounded-[3px] border border-dashed border-ink-3 px-2 py-1.5 text-xs text-ink-2">
                <b className="text-ink">No low-confidence flags for this model.</b> It has no usable review threshold
                {d.threshold_note ? <>: {d.threshold_note}</> : null}. Only model-disagreement flags are listed.
              </p>
            )}
            <p className="mt-1 text-xs text-ink-3">
              Breakdown: <span className="num">{d.flagged_low_confidence.toLocaleString('en-GB')}</span> low confidence ·{' '}
              <span className="num">{d.flagged_disagreement_only.toLocaleString('en-GB')}</span>{' '}
              {d.threshold !== null ? 'confident, but another model disagrees.' : 'where another model disagrees (no threshold, so none are called confident).'}
            </p>
          </CaseFolder>
          {d.total_flagged === 0 && <EmptyState title="Nothing is suggested for human review with this model." />}
          {d.total_flagged > 0 && d.items.length === 0 && (
            // e.g. ?page=99: the queue is not empty, this page is past its end
            <EmptyState title={`Page ${page} is past the end of the queue (${pages} ${pages === 1 ? 'page' : 'pages'}).`}
              action={<button type="button" onClick={() => set('page', '1')} className="text-xs text-ink underline underline-offset-2">Go to page 1</button>} />
          )}
          {d.items.length > 0 && (
            <CaseFolder tab={`QUEUE · PAGE ${page} / ${pages}`}>
              <table className="w-full text-xs">
                <thead><tr className="border-b border-rule text-left text-ink-3">
                  <th className="py-1.5 pr-3 font-normal">Building</th><th className="py-1.5 pr-3 font-normal">Scene</th>
                  <th className="py-1.5 pr-3 font-normal"><Chip kind="predicted" /></th><th className="py-1.5 pr-3 font-normal"><Chip kind="ground-truth" /></th>
                  <th className="py-1.5 text-right font-normal">confidence</th><th className="py-1.5 pl-4 font-normal">Why suggested</th><th /></tr></thead>
                <tbody>
                  {d.items.map((it) => (
                    <tr key={it.building_uid} className="border-b border-rule/60 align-top last:border-0">
                      <td className="num py-1.5 pr-3 text-ink">b{it.building_uid.split('_b').at(-1)}</td>
                      <td className="num py-1.5 pr-3 whitespace-nowrap text-ink-2">{it.scene_id}</td>
                      <td className="py-1.5"><span className="flex items-center gap-1 text-ink"><DamageSwatch cls={it.predicted} size={11} />{DAMAGE[it.predicted].short}</span></td>
                      <td className="py-1.5"><span className="flex items-center gap-1 text-ink-2"><DamageSwatch cls={it.true_label} size={11} />{DAMAGE[it.true_label].short}</span></td>
                      <td className="num py-1.5 text-right text-ink">{it.confidence.toFixed(2)}</td>
                      <td className="py-1.5 pl-4 text-ink-2">{it.reasons.join(' · ')}</td>
                      <td className="py-1.5 text-right"><Link to={`/triage/${it.scene_id}?model=${model}&run=1&b=${it.building_uid}`} className="whitespace-nowrap text-ink underline underline-offset-2">Open →</Link></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <nav aria-label="Pages" className="mt-3 flex justify-end gap-2 text-xs">
                <button type="button" disabled={page <= 1} onClick={() => set('page', String(page - 1))} className="rounded-[3px] border border-rule-strong px-2 py-1 text-ink-2 disabled:opacity-40">← Previous</button>
                <button type="button" disabled={page >= pages} onClick={() => set('page', String(page + 1))} className="rounded-[3px] border border-rule-strong px-2 py-1 text-ink-2 disabled:opacity-40">Next →</button>
              </nav>
            </CaseFolder>
          )}
        </>
      )}
    </div>
  )
}
