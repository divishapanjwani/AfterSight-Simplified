/** Cluster Atlas: K-Means structure in the features, compared with the labels. Never used as predictions. */
import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { BarList } from '../charts/BarList'
import { Heatmap } from '../charts/Heatmap'
import { LineChart } from '../charts/LineChart'
import { ScatterPlot, type Marker } from '../charts/ScatterPlot'
import { CaseFolder } from '../components/CaseFolder'
import { Chip } from '../components/Chip'
import { ErrorState, LoadingState } from '../components/States'
import { api } from '../services/api'
import { DAMAGE, DAMAGE_CLASSES } from '../theme/damage'

const CAT = ['var(--cat-1)', 'var(--cat-2)', 'var(--cat-3)', 'var(--cat-4)', 'var(--cat-5)', 'var(--cat-6)', 'var(--cat-7)']
const SHAPES: Marker[] = ['circle', 'square', 'triangle', 'diamond']

export function ClusterAtlasPage() {
  const [params, setParams] = useSearchParams()
  const k = Number(params.get('k') ?? 4)
  const colourBy = params.get('by') ?? 'cluster'
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
            <CaseFolder tab="CHOOSING K" title="Elbow and silhouette">
              <div className="space-y-3">
                <LineChart label="Inertia (lower = tighter clusters)" mark={d.elbow_k_heuristic} points={Object.entries(d.curves).map(([kk, v]) => ({ x: Number(kk), y: v.inertia }))} />
                <LineChart label="Silhouette, sampled (higher = better separated)" mark={d.chosen_k_by_silhouette} points={Object.entries(d.curves).map(([kk, v]) => ({ x: Number(kk), y: v.silhouette }))} />
                <p className="text-[11px] text-ink-3">Best silhouette at <span className="num">k={d.chosen_k_by_silhouette}</span>; elbow heuristic suggests <span className="num">k={d.elbow_k_heuristic}</span>.</p>
              </div>
            </CaseFolder>
            <CaseFolder tab={`k = ${k}`} title="Cluster sizes and agreement">
              <BarList digits={0} max={Math.max(...d.cluster_sizes)} items={d.cluster_sizes.map((n, i) => ({ label: `cluster ${i}`, value: n }))} />
              <dl className="mt-3 grid grid-cols-2 gap-2 text-xs">
                <div><dt className="text-ink-3">adjusted Rand index vs labels</dt><dd className="num text-lg text-ink">{d.adjusted_rand_index.toFixed(3)}</dd></div>
                <div><dt className="text-ink-3">silhouette (sampled)</dt><dd className="num text-lg text-ink">{d.silhouette_sampled.toFixed(3)}</dd></div>
              </dl>
              <p className="mt-2 text-[11px] text-ink-3">ARI 1 = clusters match the damage classes exactly; 0 = no better than random grouping.</p>
            </CaseFolder>
            <CaseFolder tab="READING IT" title="What this can and can’t tell us">
              <ul className="space-y-2 text-xs leading-5 text-ink-2">
                <li><b className="text-ink">Can:</b> show whether buildings fall into natural groups by colour, texture and change, and which damage classes each group contains.</li>
                <li><b className="text-ink">Can’t:</b> label damage. Clusters have no names; a low ARI means the strongest structure in the features is something else (roof type, scene look, disaster).</li>
                <li>{d.interpretation_note}</li>
              </ul>
            </CaseFolder>
          </div>
          <div className="grid grid-cols-[1fr_1fr] gap-5">
            <CaseFolder tab="PCA · VISUALISATION ONLY" title="2-D projection of training buildings"
              aside={<div role="radiogroup" aria-label="Colour by" className="flex overflow-hidden rounded-[3px] border border-rule-strong text-xs">
                {['cluster', 'label'].map((v) => (
                  <button key={v} type="button" role="radio" aria-checked={colourBy === v} onClick={() => set('by', v)}
                    className={`border-r border-rule-strong px-2 py-0.5 last:border-r-0 ${colourBy === v ? 'bg-ink text-paper' : 'bg-sheet text-ink-2'}`}>{v === 'cluster' ? 'by cluster' : 'by ground truth'}</button>
                ))}</div>}>
              <ScatterPlot width={560} height={360} xLabel="principal component 1" yLabel="principal component 2"
                groups={colourBy === 'cluster'
                  ? d.cluster_sizes.map((_, i) => ({ key: String(i), label: `cluster ${i}`, color: CAT[i % CAT.length], marker: SHAPES[Math.floor(i / CAT.length) % SHAPES.length] }))
                  : DAMAGE_CLASSES.map((c, i) => ({ key: c, label: DAMAGE[c].label, color: DAMAGE[c].color, marker: SHAPES[i] }))}
                points={d.pca_points.map((p) => ({ x: p.x, y: p.y, group: colourBy === 'cluster' ? String(p.cluster) : p.label }))} />
              <p className="text-[11px] text-ink-3">{d.pca_note} · components explain {(d.pca_explained_variance[0] * 100).toFixed(0)}% and {(d.pca_explained_variance[1] * 100).toFixed(0)}% of the variance.</p>
            </CaseFolder>
            <CaseFolder tab="CONTINGENCY" title="Which damage classes fall into each cluster">
              <Heatmap rows={d.cluster_sizes.map((_, i) => `cluster ${i}`)} cols={d.classes.map((c) => DAMAGE[c as keyof typeof DAMAGE]?.short ?? c)}
                matrix={d.contingency_rows_cluster_cols_class} rowTitle="cluster" colTitle="ground truth" caption="row % = share of the cluster’s buildings" />
            </CaseFolder>
          </div>
        </>
      )}
    </div>
  )
}
