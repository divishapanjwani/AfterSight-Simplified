/** Feature Lab: why classical features work (or fail) — distributions per damage class on the TRAINING split. */
import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { BoxPlot } from '../charts/BoxPlot'
import { ScatterPlot, type Marker } from '../charts/ScatterPlot'
import { CaseFolder } from '../components/CaseFolder'
import { Chip } from '../components/Chip'
import { ErrorState, LoadingState } from '../components/States'
import { api } from '../services/api'
import { DAMAGE, DAMAGE_CLASSES } from '../theme/damage'
import { featureLabel } from '../theme/features'

const MARKERS: Marker[] = ['circle', 'square', 'triangle', 'diamond']

// Features drawn on a log axis in the scatter plot. These are the 7 features the training pipeline
// log1p-transforms (|skew| > 2 on the training split, see preprocessing.log1p_features in the model metadata).
// This only changes how the chart is drawn; the API values are raw.
const LOG_AXIS_FEATURES = new Set(['area_px', 'glcm_contrast_pre', 'glcm_contrast_post', 'abs_dmean_r', 'abs_dmean_g', 'abs_dmean_b', 'gray_absdiff_mean'])

function FeatureSelect({ axis, value, features, onChange }: {
  axis: 'x' | 'y'; value: string | undefined; features: { feature: string }[]; onChange: (feature: string) => void
}) {
  return (
    <label className="flex items-center gap-1 text-xs text-ink-3">{axis}
      <select className="rounded-[3px] border border-rule-strong bg-sheet px-2 py-1 text-xs text-ink" value={value} onChange={(e) => onChange(e.target.value)}>
        {features.map((f) => <option key={f.feature} value={f.feature}>{featureLabel(f.feature)}</option>)}
      </select>
    </label>
  )
}

export function FeatureLabPage() {
  // the selected feature and the scatter axes live in the URL (?f=…&x=…&y=…), so a view can be shared
  const [params, setParams] = useSearchParams()
  const summary = useQuery({ queryKey: ['feature-summary'], queryFn: api.featureSummary })
  const features = summary.data?.features ?? [] // already sorted by separation score, best first
  const selected = params.get('f') ?? features[0]?.feature
  const xName = params.get('x') ?? features[0]?.feature
  const yName = params.get('y') ?? features[1]?.feature
  const scatter = useQuery({ queryKey: ['scatter', xName, yName], queryFn: () => api.scatter(xName!, yName!), enabled: Boolean(xName && yName) })

  function setParam(key: string, value: string) {
    const next = new URLSearchParams(params)
    next.set(key, value)
    setParams(next, { replace: true })
  }

  const current = features.find((f) => f.feature === selected)
  const maxEta = Math.max(...features.map((f) => f.separation_eta2), 1e-9) // bar lengths are relative to the best feature

  return (
    <div className="space-y-5">
      <div className="flex items-end justify-between">
        <div>
          <p className="font-mono text-[11px] tracking-[0.14em] text-ink-3 uppercase">Feature lab</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink">What separates the damage classes?</h1>
          <p className="mt-1 flex items-center gap-2 text-xs text-ink-3">Computed on the training split only <Chip kind="training" label="TRAINING SPLIT" />
            {summary.data && <span className="num">n = {summary.data.n.toLocaleString('en-GB')} buildings</span>}</p>
        </div>
      </div>
      {summary.isPending && <LoadingState label="Computing feature distributions…" rows={5} />}
      {summary.isError && <ErrorState error={summary.error} what="feature statistics" />}
      {summary.data && current && (
        <div className="grid grid-cols-[340px_1fr] gap-5">
          <CaseFolder tab={`FEATURES · ${features.length}`} title="Ranked by separation score">
            <p className="mb-2 text-[11px] text-ink-3">η² = share of the feature’s variance explained by damage class (0 = no separation, 1 = perfect).</p>
            <ul className="max-h-[620px] overflow-y-auto" aria-label="Features">
              {features.map((f) => (
                <li key={f.feature}>
                  <button type="button" onClick={() => setParam('f', f.feature)} aria-current={f.feature === selected}
                    className={`grid w-full grid-cols-[1fr_70px_40px] items-center gap-2 border-b border-rule/50 px-1 py-1 text-left text-xs hover:bg-paper ${f.feature === selected ? 'bg-sheet-2' : ''}`}>
                    <span className="truncate text-ink" title={f.feature}>{featureLabel(f.feature)}<span className="ml-1 text-[10px] text-ink-3">{f.group}</span></span>
                    <span className="h-2 rounded-[1px] bg-sheet-2"><span className="block h-2 rounded-[1px] bg-ink-2" style={{ width: `${(f.separation_eta2 / maxEta) * 100}%` }} /></span>
                    <span className="num text-right text-ink-2">{f.separation_eta2.toFixed(3)}</span>
                  </button>
                </li>
              ))}
            </ul>
          </CaseFolder>
          <div className="space-y-5">
            <CaseFolder tab="DISTRIBUTION" title={<>{featureLabel(current.feature)} <span className="font-mono text-xs font-normal text-ink-3">{current.feature}</span></>}
              aside={<span className="num text-xs text-ink-2">η² = {current.separation_eta2.toFixed(3)}</span>}>
              <BoxPlot perClass={current.per_class} label={current.feature} width={660} />
              <p className="mt-2 text-[11px] text-ink-3">Box = middle half of buildings, line = median, whiskers = 5th–95th percentile. Overlapping boxes mean this feature alone
                cannot tell the classes apart — which is why the models combine all {features.length} of them.</p>
            </CaseFolder>
            <CaseFolder tab="SCATTER" title="Two features together" aside={
              <div className="flex gap-2">
                <FeatureSelect axis="x" value={xName} features={features} onChange={(f) => setParam('x', f)} />
                <FeatureSelect axis="y" value={yName} features={features} onChange={(f) => setParam('y', f)} />
              </div>}>
              {scatter.isPending && <LoadingState label="Sampling training buildings…" />}
              {scatter.isError && <ErrorState error={scatter.error} what="the scatter sample" />}
              {scatter.data && (
                <>
                  <ScatterPlot width={660} height={380} xLabel={featureLabel(scatter.data.x)} yLabel={featureLabel(scatter.data.y)}
                    logX={LOG_AXIS_FEATURES.has(scatter.data.x)} logY={LOG_AXIS_FEATURES.has(scatter.data.y)}
                    groups={DAMAGE_CLASSES.map((c, i) => ({ key: c, label: DAMAGE[c].label, color: DAMAGE[c].color, marker: MARKERS[i] }))}
                    points={scatter.data.points.map((p) => ({ x: p.x, y: p.y, group: p.label }))} />
                  <p className="text-[11px] text-ink-3">Stratified sample of {scatter.data.n.toLocaleString('en-GB')} training buildings (each class in proportion). Skewed features use a log axis.</p>
                </>
              )}
            </CaseFolder>
          </div>
        </div>
      )}
    </div>
  )
}
