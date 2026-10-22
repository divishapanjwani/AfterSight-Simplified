/**
 * Building Case File: crops, predicted vs ground truth, confidence (with its meaning), and the model's
 * explanation — LR contributions, the decision-tree path, or the KNN neighbours. All values from the API.
 */
import { useQuery } from '@tanstack/react-query'
import { api, sceneImageUrl } from '../services/api'
import { DAMAGE, DamageSwatch } from '../theme/damage'
import { featureLabel, fmtValue } from '../theme/features'
import type { AnalysisResult, BuildingPrediction, DamageClass, Explanation } from '../types/api'
import { CaseFolder } from './CaseFolder'
import { Chip } from './Chip'
import { ErrorState, LoadingState } from './States'

export const caseTab = (uid: string) => `CASE · B-${uid.split('_b').at(-1)}`

function ClassBadge({ cls }: { cls: DamageClass }) {
  return <span className="inline-flex items-center gap-1.5 text-sm font-medium text-ink"><DamageSwatch cls={cls} size={16} />{DAMAGE[cls].label}</span>
}

/** Crop of the view JPEG around the footprint, with the footprint outlined (same pixel coordinates). */
function Crops({ building, sceneId }: { building: BuildingPrediction; sceneId: string }) {
  const pts = building.polygon.flat(2)
  const xs = pts.map((p) => p[0])
  const ys = pts.map((p) => p[1])
  const side = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys))
  const half = side / 2 + Math.max(16, 0.35 * side)
  // keep the crop window inside the 1024 px image so edge buildings don't show an empty band
  const inside = (c: number) => Math.min(Math.max(c, Math.min(half, 512)), 1024 - Math.min(half, 512))
  const cx = inside((Math.max(...xs) + Math.min(...xs)) / 2)
  const cy = inside((Math.max(...ys) + Math.min(...ys)) / 2)
  const d = building.polygon.map((rings) => rings.map((r) => `M${r.map(([x, y]) => `${x},${y}`).join('L')}Z`).join('')).join('')
  return (
    <div className="grid grid-cols-2 gap-2">
      {(['pre', 'post'] as const).map((phase) => (
        <figure key={phase}>
          <svg viewBox={`${cx - half} ${cy - half} ${2 * half} ${2 * half}`} role="img" aria-label={`${phase}-disaster crop of ${building.building_uid}`}
            className="aspect-square w-full rounded-[2px] border border-rule bg-ink">
            <image href={sceneImageUrl(sceneId, phase, 'view')} x={0} y={0} width={1024} height={1024} preserveAspectRatio="none" />
            <path d={d} fill="none" stroke="#000" strokeOpacity={0.6} strokeWidth={half / 30} fillRule="evenodd" />
            <path d={d} fill="none" stroke="var(--paper)" strokeWidth={half / 70} strokeDasharray={`${half / 25} ${half / 50}`} fillRule="evenodd" />
          </svg>
          <figcaption className="mt-0.5 font-mono text-[10px] tracking-wider text-ink-3 uppercase">{phase}-disaster · footprint outlined</figcaption>
        </figure>
      ))}
    </div>
  )
}

function Scores({ building, kind }: { building: BuildingPrediction; kind: string }) {
  const entries = Object.entries(building.scores).sort((a, b) => b[1] - a[1])
  return (
    <div>
      <p className="mb-1 text-xs text-ink-3">Class scores · <span className="text-ink-2">{kind}</span></p>
      <ul className="space-y-1">
        {entries.map(([cls, score]) => (
          <li key={cls} className="grid grid-cols-[110px_1fr_44px] items-center gap-2 text-xs">
            <span className="flex items-center gap-1 text-ink-2"><DamageSwatch cls={cls as DamageClass} size={11} />{DAMAGE[cls as DamageClass]?.short ?? cls}</span>
            <span className="h-2 rounded-[1px] bg-sheet-2"><span className="block h-2 rounded-[1px] bg-ink-2" style={{ width: `${score * 100}%` }} /></span>
            <span className="num text-right text-ink">{score.toFixed(2)}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function LrView({ e }: { e: Explanation }) {
  const rows = e.detail.top_features ?? []
  const max = Math.max(...rows.map((r) => Math.abs(r.contribution)), 1e-9)
  return (
    <div>
      <p className="mb-2 text-xs text-ink-2">Top features pushing the score for <b>{DAMAGE[e.predicted].label.toLowerCase()}</b> up (right) or down (left).
        Contribution = coefficient × standardised value, in log-odds.</p>
      <ul className="space-y-1.5">
        {rows.map((r) => (
          <li key={r.feature} className="grid grid-cols-[1fr_110px_48px] items-center gap-2 text-xs">
            <span className="truncate text-ink" title={r.feature}>{featureLabel(r.feature)} <span className="num text-ink-3">= {fmtValue(r.value_raw)}</span></span>
            <span className="relative h-3" aria-hidden="true">
              <span className="absolute inset-y-0 left-1/2 w-px bg-rule-strong" />
              <span className={`absolute inset-y-0.5 ${r.contribution >= 0 ? 'left-1/2 bg-ink-2' : 'right-1/2 bg-ink-3/60'}`}
                style={{ width: `${(Math.abs(r.contribution) / max) * 50}%` }} />
            </span>
            <span className="num text-right text-ink">{r.contribution >= 0 ? '+' : ''}{r.contribution.toFixed(2)}</span>
          </li>
        ))}
      </ul>
      <p className="num mt-2 text-[11px] text-ink-3">intercept {e.detail.intercept?.toFixed(2)} · total score {e.detail.score?.toFixed(2)}</p>
    </div>
  )
}

function TreeView({ e }: { e: Explanation }) {
  const steps = e.detail.steps ?? []
  return (
    <div>
      <p className="mb-2 text-xs text-ink-2">The questions the decision tree asked, from the root to the leaf (values in raw feature units).</p>
      <ol className="space-y-1">
        {steps.map((s, i) => (
          <li key={i} className="grid grid-cols-[20px_1fr] gap-2 text-xs">
            <span className="num text-ink-3">{i + 1}</span>
            <span className="text-ink">
              {featureLabel(s.feature)} <span className="num text-ink-2">{fmtValue(s.value_raw)} {s.op === '<=' ? '≤' : '>'} {fmtValue(s.threshold_raw)}</span>
              <span className="ml-1 font-mono text-[10px] text-ink-3">{s.feature}</span>
            </span>
          </li>
        ))}
        <li className="grid grid-cols-[20px_1fr] gap-2 text-xs">
          <span aria-hidden="true" className="text-ink-3">→</span>
          <span className="text-ink">Leaf of <span className="num">{e.detail.leaf_samples}</span> training buildings:{' '}
            <span className="num text-ink-2">{Object.entries(e.detail.leaf_class_fractions ?? {}).map(([c, p]) => `${DAMAGE[c as DamageClass]?.short ?? c} ${(p * 100).toFixed(0)}%`).join(' · ')}</span>
          </span>
        </li>
      </ol>
      <p className="mt-2 text-[11px] text-ink-3">Leaf shares are class-weighted training fractions, not calibrated probabilities.</p>
    </div>
  )
}

function KnnView({ e }: { e: Explanation }) {
  const n = e.detail.neighbours ?? []
  const votes = n.reduce<Record<string, number>>((acc, x) => ({ ...acc, [x.label]: (acc[x.label] ?? 0) + 1 }), {})
  return (
    <div>
      <p className="mb-2 text-xs text-ink-2">The {e.detail.k} most similar <b>training</b> buildings (standardised feature space, {e.detail.weights} vote).{' '}
        Votes: <span className="num">{Object.entries(votes).map(([c, v]) => `${DAMAGE[c as DamageClass]?.short ?? c} ${v}`).join(' · ')}</span></p>
      <ul className="grid grid-cols-5 gap-2">
        {n.map((x) => (
          <li key={x.sample_id} className="text-[10px]">
            <img src={x.crop_urls.post} alt={`Post-disaster crop of neighbour ${x.sample_id}`} width={96} height={96} loading="lazy"
              className="aspect-square w-full rounded-[2px] border border-rule bg-ink object-cover" />
            <span className="mt-0.5 flex items-center gap-1 text-ink-2"><DamageSwatch cls={x.label} size={10} />{DAMAGE[x.label].short}</span>
            <span className="num block text-ink-3">d {x.distance.toFixed(2)}</span>
          </li>
        ))}
      </ul>
      <p className="mt-2 flex items-center gap-1.5 text-[11px] text-ink-3">Neighbour labels are <Chip kind="ground-truth" /> from <Chip kind="training" /> buildings.</p>
    </div>
  )
}

export function CaseFile({ analysis, uid }: { analysis: AnalysisResult; uid: string }) {
  const building = analysis.buildings.find((b) => b.building_uid === uid)
  const explain = useQuery({
    queryKey: ['explain', analysis.analysis_id, uid],
    queryFn: () => api.explain(analysis.analysis_id, uid),
    enabled: Boolean(building),
  })
  if (!building) return <ErrorState error={new Error('That building is not part of this analysis.')} what="the case file" />

  return (
    <CaseFolder tab={caseTab(uid)} title={<span className="num text-[13px]">{uid}</span>}
      aside={<span className="text-xs text-ink-3">{analysis.model.label}</span>}>
      <div className="space-y-4">
        <Crops building={building} sceneId={analysis.scene_id} />

        <div className="grid grid-cols-2 gap-3 rounded-[3px] border border-rule bg-paper/50 p-3">
          <div>
            <Chip kind="predicted" />
            <div className="mt-1"><ClassBadge cls={building.predicted} /></div>
            <p className="num mt-1 text-xs text-ink-2">confidence {building.confidence.toFixed(2)}</p>
          </div>
          <div>
            <Chip kind="ground-truth" />
            <div className="mt-1"><ClassBadge cls={building.true_label} /></div>
            <p className={`mt-1 text-xs ${building.correct ? 'text-ok' : 'text-danger'}`}>{building.correct ? '✓ agrees with ground truth' : '✗ differs from ground truth'}</p>
          </div>
        </div>

        {analysis.provenance.warning && (
          <p role="note" className="rounded-[3px] border border-caution bg-caution-soft px-2 py-1.5 text-xs text-ink">! {analysis.provenance.warning}</p>
        )}

        {building.needs_review && (
          <div className="rounded-[3px] border border-dashed border-ink-3 px-2 py-1.5 text-xs text-ink-2">
            <span className="font-mono text-[10px] tracking-wider text-ink">SUGGESTED FOR HUMAN REVIEW</span>
            <ul className="mt-0.5">{building.review_reasons.map((r) => <li key={r}>{r}</li>)}</ul>
          </div>
        )}

        <Scores building={building} kind={analysis.confidence_kind} />

        <div className="border-t border-rule pt-3">
          <p className="mb-2 font-mono text-[10px] tracking-[0.12em] text-ink-3 uppercase">Why the model decided this</p>
          {explain.isPending && <LoadingState label="Loading explanation…" rows={3} />}
          {explain.isError && <ErrorState error={explain.error} what="the explanation" />}
          {explain.data?.kind === 'lr_contributions' && <LrView e={explain.data} />}
          {explain.data?.kind === 'decision_path' && <TreeView e={explain.data} />}
          {explain.data?.kind === 'nearest_neighbours' && <KnnView e={explain.data} />}
          {explain.data?.kind === 'none' && <p className="text-xs text-ink-2">{explain.data.detail.note}</p>}
          {explain.data && explain.data.global_top_features.length > 0 && (
            <p className="mt-3 text-[11px] text-ink-3">
              Most relied-on features overall (permutation importance, <Chip kind="validation" />):{' '}
              {explain.data.global_top_features.map((f) => featureLabel(f.feature)).join(' · ')}
            </p>
          )}
        </div>

        <dl className="grid grid-cols-3 gap-2 border-t border-rule pt-3 text-xs">
          <div><dt className="text-ink-3">footprint area</dt><dd className="num text-ink">{building.area_px.toLocaleString('en-GB')} px</dd></div>
          <div><dt className="text-ink-3">mixed labels</dt><dd className="text-ink">{building.mixed_label ? 'yes — possibly merged' : 'no'}</dd></div>
          <div><dt className="text-ink-3">at image edge</dt><dd className="text-ink">{building.touches_border ? 'yes — may be cut off' : 'no'}</dd></div>
        </dl>
      </div>
    </CaseFolder>
  )
}
