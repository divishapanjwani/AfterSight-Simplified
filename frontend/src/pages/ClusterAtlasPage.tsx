/**
 * Cluster Atlas: K-Means structure in the features, compared with the reference labels. Never used as predictions.
 * Shows the PRECOMPUTED results for one k at a time (nothing is refitted here). The API still returns the inertia
 * curve and PCA points; they are not displayed (see docs/API.md).
 */
import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { BarList } from '../charts/BarList'
import { Heatmap } from '../charts/Heatmap'
import { CaseFolder } from '../components/CaseFolder'
import { Chip } from '../components/Chip'
import { ErrorState, LoadingState } from '../components/States'
import { api } from '../services/api'
import { DAMAGE } from '../theme/damage'

export function ClusterAtlasPage() {
  const [params, setParams] = useSearchParams()
  const k = Number(params.get('k') ?? 4)
  const q = useQuery({ queryKey: ['clusters', k], queryFn: () => api.clusters(k) })
  const set = (key: string, v: string) => { const n = new URLSearchParams(params); n.set(key, v); setParams(n, { replace: true }) }
  const d = q.data
  return (
    <div className="space-y-5">
      <div className="flex items-end justify-between">
        <div>
          <p className="font-mono text-[11px] tracking-[0.14em] text-ink-3 uppercase">Cluster atlas</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink">Do the features form damage groups on their own?</h1>
          <p className="mt-1 flex items-center gap-2 text-xs text-ink-3">K-Means on training buildings <Chip kind="training" label="TRAINING SPLIT" />
            <span className="font-mono text-[10px] tracking-wider">ANALYSIS ONLY · CLUSTERS ARE NEVER PREDICTIONS</span></p>
        </div>
        {d && (
          <div role="radiogroup" aria-label="Number of clusters k" className="flex overflow-hidden rounded-[3px] border border-rule-strong">
            {d.k_values.map((v) => (
              <button key={v} type="button" role="radio" aria-checked={v === k} onClick={() => set('k', String(v))}
                className={`num border-r border-rule-strong px-2.5 py-1 text-xs last:border-r-0 ${v === k ? 'bg-ink text-paper' : 'bg-sheet text-ink-2 hover:bg-sheet-2'}`}>k={v}</button>
            ))}
          </div>
        )}
      </div>
      {q.isPending && <LoadingState label="Loading clusters…" rows={4} />}
      {q.isError && <ErrorState error={q.error} what="the cluster analysis" />}
      {d && (
        <>
          <div className="grid grid-cols-3 gap-5">
            <CaseFolder tab={`k = ${k}`} title="Cluster sizes">
              <BarList digits={0} max={Math.max(...d.cluster_sizes)} items={d.cluster_sizes.map((n, i) => ({ label: `cluster ${i}`, value: n }))} />
              <p className="mt-2 text-[11px] text-ink-3">Training buildings in each cluster. A cluster number is not a damage class.</p>
            </CaseFolder>
            <CaseFolder tab="SCORES" title="Separation and agreement">
              <dl className="space-y-3 text-xs">
                <div>
                  <dt className="text-ink-3">silhouette (sampled)</dt>
                  <dd className="num text-lg text-ink">{d.silhouette_sampled.toFixed(3)}</dd>
                  <dd className="text-[11px] text-ink-3">How well separated the clusters are, from −1 to 1.</dd>
                </div>
                <div>
                  <dt className="text-ink-3">adjusted Rand index vs reference labels</dt>
                  <dd className="num text-lg text-ink">{d.adjusted_rand_index.toFixed(3)}</dd>
                  <dd className="text-[11px] text-ink-3">Agreement between clusters and the reference damage labels: 0 = random, 1 = identical. It is not prediction accuracy.</dd>
                </div>
              </dl>
            </CaseFolder>
            <CaseFolder tab="READING IT" title="What this can and can’t tell us">
              <ul className="space-y-2 text-xs leading-5 text-ink-2">
                <li>K-Means groups buildings by their features (colour, texture, change, shape). A cluster ID is not a damage class.</li>
                <li><b className="text-ink">Can:</b> show whether buildings fall into natural groups, and which damage classes each group contains.</li>
                <li><b className="text-ink">Can’t:</b> label damage. Clusters have no names; a low ARI means the strongest structure in the features is something else (roof type, scene look, disaster).</li>
                <li>Results exist only for the precomputed values <span className="num">k = {d.k_values.join(', ')}</span>; nothing is refitted on this page.</li>
                <li>{d.interpretation_note}</li>
              </ul>
            </CaseFolder>
          </div>
          <CaseFolder tab="CONTINGENCY" title="Which damage classes fall into each cluster">
            <Heatmap rows={d.cluster_sizes.map((_, i) => `cluster ${i}`)} cols={d.classes.map((c) => DAMAGE[c as keyof typeof DAMAGE]?.short ?? c)}
              matrix={d.contingency_rows_cluster_cols_class} rowTitle="cluster" colTitle="reference label" caption="row % = share of the cluster’s buildings" />
          </CaseFolder>
        </>
      )}
    </div>
  )
}
