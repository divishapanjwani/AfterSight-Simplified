/**
 * Triage Desk: pick a model, run the analysis on one scene, compare the before/after imagery with
 * footprints coloured by ground truth, prediction or agreement, and open any building's case file.
 * State lives in the URL: /triage/:sceneId?model=&overlay=&b=
 */
import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { CaseFile } from '../components/CaseFile'
import { CaseFolder } from '../components/CaseFolder'
import { Chip, SplitChip } from '../components/Chip'
import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { useWorkflow } from '../components/Workflow'
import { api, exportCsvUrl, reportPdfUrl, sceneImageUrl } from '../services/api'
import { DAMAGE, DAMAGE_CLASSES, DamageLegend, DamageSwatch } from '../theme/damage'
import type { AnalysisResult, BuildingPrediction, DamageClass, SceneDetail } from '../types/api'
import { SceneViewer, type Footprint, type OverlayMode } from '../viewer/SceneViewer'

const OVERLAYS: { value: OverlayMode; label: string }[] = [
  { value: 'ground_truth', label: 'Ground truth' },
  { value: 'prediction', label: 'Prediction' },
  { value: 'agreement', label: 'Correct vs wrong' },
  { value: 'none', label: 'Off' },
]
const SORTS = { severity: 'Most severe first', review: 'Suggested for review first', confidence: 'Least confident first' } as const
const SEVERITY: Record<DamageClass, number> = { 'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, destroyed: 3 }

export function ProvenanceBanner({ warning, note }: { warning: string | null; note: string }) {
  if (!warning) return <p className="flex items-center gap-2 text-xs text-ink-3"><Chip kind="test" />{note}</p>
  return (
    <div role="alert" className="flex items-start gap-2 rounded-[4px] border-2 border-caution bg-caution-soft px-3 py-2 text-sm text-ink">
      <span aria-hidden="true" className="font-mono font-semibold text-caution">!</span><span><b>{warning}</b> {note}</span>
    </div>
  )
}

function SceneHeader({ scene }: { scene: SceneDetail }) {
  return (
    <div className="flex items-end justify-between gap-6">
      <div>
        <p className="font-mono text-[11px] tracking-[0.14em] text-ink-3 uppercase">Triage desk</p>
        <h1 className="num mt-1 text-xl font-medium text-ink">{scene.scene_id}</h1>
        <p className="mt-0.5 flex items-center gap-2 text-xs text-ink-3">
          {scene.event} · {scene.disaster_type} <Chip kind="config" /> <SplitChip split={scene.split} />
          <span>{scene.n_buildings} buildings</span>
        </p>
      </div>
      <DamageLegend compact />
    </div>
  )
}

function CountsTable({ analysis }: { analysis: AnalysisResult }) {
  const m = analysis.scene_metrics
  return (
    <div className="space-y-2">
      <table className="w-full text-xs">
        <thead><tr className="text-left text-ink-3"><th className="font-normal">Class</th>
          <th className="text-right font-normal"><Chip kind="predicted" /></th><th className="text-right font-normal"><Chip kind="ground-truth" /></th></tr></thead>
        <tbody>
          {DAMAGE_CLASSES.map((c) => (
            <tr key={c} className="border-t border-rule/60">
              <td className="py-1"><span className="flex items-center gap-1.5 text-ink-2"><DamageSwatch cls={c} size={12} />{DAMAGE[c].label}</span></td>
              <td className="num py-1 text-right text-ink">{analysis.counts.predicted[c]}</td>
              <td className="num py-1 text-right text-ink">{analysis.counts.ground_truth[c]}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {m && (
        <p className="text-xs text-ink-2">
          On this scene: accuracy <span className="num text-ink">{m.accuracy?.toFixed(2)}</span> · macro-F1 <span className="num text-ink">{m.macro_f1?.toFixed(2)}</span>{' '}
          <span className="text-ink-3">(n = {m.n}, a {analysis.provenance.split === 'test' ? 'held-out test' : analysis.provenance.split} scene; one scene is a small sample)</span>
        </p>
      )}
    </div>
  )
}

function BuildingList({ buildings, selected, onSelect, sort, onSort }: {
  buildings: BuildingPrediction[]; selected: string | null; onSelect: (uid: string) => void
  sort: keyof typeof SORTS; onSort: (s: keyof typeof SORTS) => void
}) {
  const sorted = useMemo(() => [...buildings].sort((a, b) => {
    if (sort === 'review') return Number(b.needs_review) - Number(a.needs_review) || a.confidence - b.confidence
    if (sort === 'confidence') return a.confidence - b.confidence
    return SEVERITY[b.predicted] - SEVERITY[a.predicted] || b.confidence - a.confidence
  }), [buildings, sort])
  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <p className="text-xs text-ink-3">{buildings.length} buildings</p>
        <label className="flex items-center gap-1.5 text-xs text-ink-3">Sort
          <select value={sort} onChange={(e) => onSort(e.target.value as keyof typeof SORTS)} className="rounded-[3px] border border-rule-strong bg-sheet px-1.5 py-0.5 text-xs text-ink">
            {Object.entries(SORTS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
        </label>
      </div>
      <ul className="max-h-[430px] overflow-y-auto rounded-[3px] border border-rule" aria-label="Buildings">
        {sorted.map((b) => (
          <li key={b.building_uid}>
            <button type="button" onClick={() => onSelect(b.building_uid)} aria-current={b.building_uid === selected}
              className={`grid w-full grid-cols-[56px_1fr_1fr_40px_16px] items-center gap-2 border-b border-rule/60 px-2 py-1.5 text-left text-xs hover:bg-paper ${b.building_uid === selected ? 'bg-sheet-2' : ''}`}>
              <span className="num text-ink-2">b{b.building_uid.split('_b').at(-1)}</span>
              <span className="flex items-center gap-1 text-ink" title="PREDICTED"><DamageSwatch cls={b.predicted} size={11} />{DAMAGE[b.predicted].short}</span>
              <span className="flex items-center gap-1 text-ink-3" title="GROUND TRUTH">● {DAMAGE[b.true_label].short}</span>
              <span className="num text-right text-ink-2">{b.confidence.toFixed(2)}</span>
              <span aria-label={b.needs_review ? 'suggested for review' : undefined} title={b.needs_review ? 'Suggested for human review' : ''}
                className="text-center font-mono text-ink-3">{b.needs_review ? '◌' : ''}</span>
            </button>
          </li>
        ))}
      </ul>
      <p className="mt-1.5 text-[11px] text-ink-3">Columns: id · <b>predicted</b> · ground truth · confidence · ◌ suggested for human review</p>
    </div>
  )
}

export function TriageDeskPage() {
  const { sceneId } = useParams()
  const [params, setParams] = useSearchParams()
  const navigate = useNavigate()
  const { update } = useWorkflow()
  const overlay = (params.get('overlay') as OverlayMode) ?? 'prediction'
  const selected = params.get('b')
  const sort = (params.get('sort') as keyof typeof SORTS) ?? 'severity'
  const set = (k: string, v: string | null) => {
    const next = new URLSearchParams(params)
    if (v === null) next.delete(k)
    else next.set(k, v)
    setParams(next, { replace: true })
  }

  const models = useQuery({ queryKey: ['models'], queryFn: api.models })
  const model = params.get('model') ?? models.data?.default_model ?? 'logistic_regression'
  const scene = useQuery({ queryKey: ['scene', sceneId], queryFn: () => api.scene(sceneId!), enabled: Boolean(sceneId) })
  const runRequested = params.get('run') === '1' // set by "Run analysis"; kept in the URL so reloads restore it
  const run = useQuery({ queryKey: ['analysis', sceneId, model], queryFn: () => api.analyse(sceneId!, model),
    enabled: Boolean(sceneId) && runRequested, staleTime: Infinity })
  const analysis = run.data

  useEffect(() => { if (sceneId) update({ sceneId, model }) }, [sceneId, model, update])
  useEffect(() => { if (analysis) update({ analysisId: analysis.analysis_id, buildingUid: selected ?? undefined }) }, [analysis, selected, update])

  const firstTest = useQuery({ queryKey: ['scenes', 'first-test'], queryFn: () => api.scenes({ split: 'test', has_buildings: true, page_size: 1 }), enabled: !sceneId })
  if (!sceneId) {
    return (
      <EmptyState title="Choose a scene to triage.">
        <p>The Triage Desk works on one scene at a time. Held-out test scenes give honest results.</p>
        <div className="mt-3 flex gap-3">
          {firstTest.data?.items[0] && (
            <button type="button" onClick={() => navigate(`/triage/${firstTest.data!.items[0].scene_id}`)}
              className="rounded-[3px] bg-signal px-3 py-1.5 text-sm font-medium text-signal-ink">Open the first test scene</button>
          )}
          <Link to="/library?split=test" className="self-center text-sm text-ink underline underline-offset-2">Browse the Scene Library</Link>
        </div>
      </EmptyState>
    )
  }
  if (scene.isPending) return <LoadingState label="Loading scene…" rows={4} />
  if (scene.isError) return <ErrorState error={scene.error} what="this scene" />
  const s = scene.data

  const byUid = new Map(analysis?.buildings.map((b) => [b.building_uid, b]))
  const footprints: Footprint[] = s.buildings.map((g) => {
    const p = byUid.get(g.building_uid)
    return { uid: g.building_uid, polygon: g.polygon, trueLabel: g.true_label, predicted: p?.predicted, correct: p?.correct }
  })
  const effectiveOverlay: OverlayMode = !analysis && (overlay === 'prediction' || overlay === 'agreement') ? 'ground_truth' : overlay
  const reviewCount = analysis?.buildings.filter((b) => b.needs_review).length ?? 0

  return (
    <div className="space-y-4">
      <SceneHeader scene={s} />
      <ProvenanceBanner warning={s.provenance.warning} note={s.provenance.note} />

      <div role="toolbar" aria-label="Analysis controls" className="flex flex-wrap items-center gap-4 rounded-[4px] border border-rule bg-sheet px-3 py-2.5">
        <div role="radiogroup" aria-label="Model" className="flex overflow-hidden rounded-[3px] border border-rule-strong">
          {(models.data?.models ?? []).map((m) => (
            <button key={m.name} type="button" role="radio" aria-checked={model === m.name} disabled={!m.trained}
              title={m.trained ? m.confidence : `Not trained — ${m.fix}`} onClick={() => set('model', m.name)}
              className={`border-r border-rule-strong px-2.5 py-1 text-sm last:border-r-0 disabled:cursor-not-allowed disabled:opacity-40 ${model === m.name ? 'bg-ink text-paper' : 'bg-sheet text-ink-2 hover:bg-sheet-2'}`}>
              {m.label.replace(' (optional)', '')}{m.optional && <span className="ml-1 text-[10px] opacity-70">optional</span>}
            </button>
          ))}
        </div>
        <button type="button" onClick={() => (runRequested ? run.refetch() : set('run', '1'))} disabled={run.isFetching}
          className="rounded-[3px] bg-signal px-3 py-1.5 text-sm font-medium text-signal-ink hover:brightness-110 disabled:opacity-60">
          {run.isFetching ? 'Analysing…' : analysis ? 'Re-run analysis' : 'Run analysis'}
        </button>
        <div role="radiogroup" aria-label="Overlay" className="ml-auto flex overflow-hidden rounded-[3px] border border-rule-strong">
          {OVERLAYS.map((o) => {
            const needsAnalysis = o.value === 'prediction' || o.value === 'agreement'
            return (
              <button key={o.value} type="button" role="radio" aria-checked={effectiveOverlay === o.value} disabled={needsAnalysis && !analysis}
                onClick={() => set('overlay', o.value)} title={needsAnalysis && !analysis ? 'Run the analysis first' : ''}
                className={`border-r border-rule-strong px-2.5 py-1 text-sm last:border-r-0 disabled:opacity-40 ${effectiveOverlay === o.value ? 'bg-ink text-paper' : 'bg-sheet text-ink-2 hover:bg-sheet-2'}`}>
                {o.label}
              </button>
            )
          })}
        </div>
      </div>
      {run.isError && <ErrorState error={run.error} what="the analysis" />}

      <div className="grid grid-cols-[720px_1fr] gap-5">
        <div className="space-y-2">
          <SceneViewer preUrl={sceneImageUrl(s.scene_id, 'pre', 'view')} postUrl={sceneImageUrl(s.scene_id, 'post', 'view')}
            width={s.width} height={s.height} footprints={footprints} mode={effectiveOverlay} selected={selected}
            onSelect={(uid) => set('b', uid)} />
          <p className="text-[11px] text-ink-3">
            {effectiveOverlay === 'agreement'
              ? <>Green outline = prediction agrees with ground truth · red dashed = differs. </>
              : <>Fill pattern + colour = {effectiveOverlay === 'prediction' ? 'PREDICTED' : 'GROUND TRUTH'} class. </>}
            Footprints derived from xBD annotation masks · pixel coordinates. Drag to pan, scroll to zoom, click a footprint.
          </p>
        </div>

        <div className="min-w-0 space-y-4">
          {!analysis && !run.isFetching && (
            <CaseFolder tab="ANALYSIS · NOT RUN">
              <p className="text-sm text-ink-2">Showing <b>ground-truth</b> footprints. Run the analysis to see what the selected model predicts for each building.</p>
            </CaseFolder>
          )}
          {run.isFetching && !analysis && <LoadingState label="Running the analysis…" rows={4} />}
          {analysis && selected && (
            <div>
              <button type="button" onClick={() => set('b', null)} className="mb-1 text-xs text-ink-2 underline underline-offset-2">← all buildings</button>
              <CaseFile analysis={analysis} uid={selected} />
              <Link to={`/case/${analysis.analysis_id}/${selected}`} className="mt-2 inline-block text-xs text-ink underline underline-offset-2">Open as a full page →</Link>
            </div>
          )}
          {analysis && !selected && (
            <>
              <CaseFolder tab={`ANALYSIS · ${analysis.analysis_id.slice(0, 8)}`} title={analysis.model.label}
                aside={<span className="num text-[11px] text-ink-3">{(analysis.seconds * 1000).toFixed(0)} ms{analysis.cached ? ' · cached' : ''}</span>}>
                <CountsTable analysis={analysis} />
                <p className="mt-2 text-xs text-ink-2">
                  <span className="num text-ink">{reviewCount}</span> of {analysis.buildings.length} buildings are <b>suggested for human review</b>
                  {analysis.review_threshold !== null && <> (confidence below <span className="num">{analysis.review_threshold.toFixed(2)}</span>, chosen on validation)</>}.
                </p>
                <p className="mt-1 text-[11px] text-ink-3">Confidence means: {analysis.confidence_kind}.</p>
                <div className="mt-3 flex gap-3 text-xs">
                  <a href={exportCsvUrl(analysis.analysis_id)} className="text-ink underline underline-offset-2">Export CSV</a>
                  <a href={reportPdfUrl(analysis.analysis_id)} className="text-ink underline underline-offset-2">PDF report</a>
                </div>
              </CaseFolder>
              <BuildingList buildings={analysis.buildings} selected={selected} onSelect={(uid) => set('b', uid)} sort={sort} onSort={(v) => set('sort', v)} />
            </>
          )}
        </div>
      </div>
    </div>
  )
}
