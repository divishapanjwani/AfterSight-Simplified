/**
 * Building Case File: crops, predicted vs ground truth, confidence (with its meaning), and the model's
 * explanation — LR contributions or the KNN neighbours.
 * All values come from the API.
 */
import { useQuery } from '@tanstack/react-query'
import { api, sceneImageUrl } from '../services/api'
import { DAMAGE, DamageSwatch } from '../theme/damage'
import { featureLabel, fmtValue } from '../theme/features'
import type { AnalysisResult, BuildingPrediction, DamageClass, Explanation } from '../types/api'
import { CaseFolder } from './CaseFolder'
import { Chip } from './Chip'
import { ErrorState, LoadingState } from './States'

const IMAGE_SIZE = 1024 // every xBD scene image is 1024 x 1024 pixels

export const caseTab = (uid: string) => `CASE · B-${uid.split('_b').at(-1)}`

/** Short class name ("minor", "destroyed"…), or the raw key if the API ever sends an unknown class. */
function shortName(cls: string): string {
  return DAMAGE[cls as DamageClass]?.short ?? cls
}

function ClassBadge({ cls }: { cls: DamageClass }) {
  return <span className="inline-flex items-center gap-1.5 text-sm font-medium text-ink"><DamageSwatch cls={cls} size={16} />{DAMAGE[cls].label}</span>
}

/**
 * Square window (centre + half-width, in image pixels) around a footprint, with a margin so the
 * surroundings are visible.
 */
function cropWindow(polygon: BuildingPrediction['polygon']) {
  const points = polygon.flat(2)
  const xs = points.map((p) => p[0])
  const ys = points.map((p) => p[1])
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)

  const side = Math.max(maxX - minX, maxY - minY)
  const half = side / 2 + Math.max(16, 0.35 * side)

  // Keep the window inside the image so buildings at the edge don't show an empty band.
  // (A window wider than the image is centred on the image instead.)
  const limit = Math.min(half, IMAGE_SIZE / 2)
  const keepInside = (centre: number) => Math.min(Math.max(centre, limit), IMAGE_SIZE - limit)

  return { cx: keepInside((maxX + minX) / 2), cy: keepInside((maxY + minY) / 2), half }
}

/** SVG path for a (multi)polygon: one "M…L…Z" segment per ring, so holes are cut out with evenodd. */
function polygonPath(polygon: BuildingPrediction['polygon']): string {
  let d = ''
  for (const part of polygon) {
    for (const ring of part) {
      d += `M${ring.map(([x, y]) => `${x},${y}`).join('L')}Z`
    }
  }
  return d
}

/** Crop of the view JPEG around the footprint, with the footprint outlined (same pixel coordinates). */
function Crops({ building, sceneId }: { building: BuildingPrediction; sceneId: string }) {
  const { cx, cy, half } = cropWindow(building.polygon)
  const d = polygonPath(building.polygon)
  return (
    <div className="grid grid-cols-2 gap-2">
      {(['pre', 'post'] as const).map((phase) => (
        <figure key={phase}>
          <svg viewBox={`${cx - half} ${cy - half} ${2 * half} ${2 * half}`} role="img" aria-label={`${phase}-disaster crop of ${building.building_uid}`}
            className="aspect-square w-full rounded-[2px] border border-rule bg-ink">
            <image href={sceneImageUrl(sceneId, phase, 'view')} x={0} y={0} width={IMAGE_SIZE} height={IMAGE_SIZE} preserveAspectRatio="none" />
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
            <span className="flex items-center gap-1 text-ink-2"><DamageSwatch cls={cls as DamageClass} size={11} />{shortName(cls)}</span>
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

/** How many neighbours carry each label, in the order the labels first appear (nearest first). */
function countVotes(neighbours: { label: string }[]): Record<string, number> {
  const votes: Record<string, number> = {}
  for (const neighbour of neighbours) {
    votes[neighbour.label] = (votes[neighbour.label] ?? 0) + 1
  }
  return votes
}

function KnnView({ e }: { e: Explanation }) {
  const neighbours = e.detail.neighbours ?? []
  const votes = countVotes(neighbours)
  return (
    <div>
      <p className="mb-2 text-xs text-ink-2">The {e.detail.k} most similar <b>training</b> buildings (standardised feature space, {e.detail.weights} vote).{' '}
        Votes: <span className="num">{Object.entries(votes).map(([c, v]) => `${shortName(c)} ${v}`).join(' · ')}</span></p>
      <ul className="grid grid-cols-5 gap-2">
        {neighbours.map((x) => (
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

/** Picks the explanation view that matches the model (the API's `kind` field). */
function ExplanationView({ e }: { e: Explanation }) {
  switch (e.kind) {
    case 'lr_contributions':
      return <LrView e={e} />
    case 'nearest_neighbours':
      return <KnnView e={e} />
    default:
      return null
  }
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
          {explain.data && <ExplanationView e={explain.data} />}
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
