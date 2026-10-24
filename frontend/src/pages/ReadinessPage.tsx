/** Home / Readiness board: what is built, what is trained, and the honest headline numbers — all from the API. */
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { CaseFolder } from '../components/CaseFolder'
import { Chip } from '../components/Chip'
import { CommandBlock, ErrorState, LoadingState } from '../components/States'
import { api } from '../services/api'
import type { Comparison, StatusTile } from '../types/api'
import { DamageLegend } from '../theme/damage'

const TILE_ORDER: { key: string; title: string }[] = [
  { key: 'scene_cache', title: 'Scene cache' },
  { key: 'features', title: 'Feature table' },
  { key: 'splits', title: 'Split manifests' },
  { key: 'models', title: 'Models' },
  { key: 'review_thresholds', title: 'Review thresholds' },
  { key: 'last_analysis', title: 'Last analysis' },
]

const fmt = (v: number | null | undefined, digits = 3) => (v === null || v === undefined ? 'not available' : v.toFixed(digits))
const count = (v: unknown) => (typeof v === 'number' ? v.toLocaleString('en-GB') : String(v ?? '—'))

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-rule/70 py-1 last:border-0">
      <dt className="text-xs text-ink-3">{label}</dt>
      <dd className="num text-right text-[13px] text-ink">{children}</dd>
    </div>
  )
}

function TileBody({ name, tile }: { name: string; tile: StatusTile }) {
  const d = tile.detail as Record<string, unknown>
  if (tile.state === 'not_ready' || (!tile.ready && !tile.state)) {
    return (
      <>
        <p className="text-sm text-ink-2">{String(tile.reason ?? d.reason ?? 'Not built yet.')}</p>
        {tile.fix && <CommandBlock command={tile.fix} />}
      </>
    )
  }
  switch (name) {
    case 'scene_cache':
      return (
        <dl>
          <Fact label="scenes">{count(d.scenes)}</Fact>
          <Fact label="buildings (ground truth)">{count(d.buildings)}</Fact>
          <Fact label="events">{(d.events as string[]).length}</Fact>
          <p className="mt-2 text-xs leading-5 text-ink-3">{(d.events as string[]).join(' · ')}</p>
        </dl>
      )
    case 'features':
      return (
        <dl>
          <Fact label="buildings">{count(d.buildings)}</Fact>
          <Fact label="features per building">{count(d.features)}</Fact>
          <Fact label="schema version">v{count(d.schema_version)}</Fact>
        </dl>
      )
    case 'splits':
      return (
        <dl>
          {Object.entries(d).map(([design, hash]) => <Fact key={design} label={design.replace('_', '-')}>{String(hash)}</Fact>)}
          <p className="mt-2 text-xs text-ink-3">Manifest hashes · scene-grouped, leakage-checked</p>
        </dl>
      )
    case 'models': {
      const trained = d.trained as Record<string, boolean>
      return (
        <dl>
          {Object.entries(trained).map(([m, ok]) => (
            <Fact key={m} label={m.replaceAll('_', ' ')}>{ok ? 'trained' : <span className="text-ink-3">not trained</span>}</Fact>
          ))}
          {tile.reason && <p className="mt-2 text-xs text-ink-3">{tile.reason}</p>}
          {tile.fix && <CommandBlock command={tile.fix} />}
        </dl>
      )
    }
    case 'review_thresholds': {
      // per served model: the validation-chosen threshold, or "not available" when there is no usable one
      const thresholds = (d.thresholds ?? {}) as Record<string, number | null>
      return (
        <dl>
          {Object.entries(thresholds).map(([m, t]) => (
            <Fact key={m} label={m.replaceAll('_', ' ')}>{t === null ? <span className="text-ink-3">not available</span> : t.toFixed(2)}</Fact>
          ))}
          {d.rule != null && <p className="mt-2 text-xs text-ink-3">Chosen on validation: <span className="num">{String(d.rule)}</span></p>}
          {tile.reason && <p className="mt-1 text-xs text-ink-2">{tile.reason}</p>}
          {tile.fix && <CommandBlock command={tile.fix} />}
        </dl>
      )
    }
    case 'last_analysis':
      return <dl><Fact label="analysis id">{String(d.analysis_id)}</Fact><Fact label="cached analyses">{count(d.count)}</Fact></dl>
    default:
      return null
  }
}

/** The tile's own three-way state when the API sends one; otherwise ready / not ready. */
function ReadyBadge({ tile }: { tile: StatusTile }) {
  const state = tile.state ?? (tile.ready ? 'ready' : 'not_ready')
  if (state === 'ready') {
    return <span className="inline-flex items-center gap-1 rounded-[3px] bg-ok-soft px-1.5 py-px font-mono text-[10px] tracking-wider text-ok">✓ READY</span>
  }
  if (state === 'partial') {
    return <span title={tile.reason ?? undefined} className="inline-flex items-center gap-1 rounded-[3px] border border-caution bg-caution-soft px-1.5 py-px font-mono text-[10px] tracking-wider text-caution">◐ PARTIAL</span>
  }
  return <span className="inline-flex items-center gap-1 rounded-[3px] border border-dashed border-ink-3 px-1.5 py-px font-mono text-[10px] tracking-wider text-ink-3">○ NOT READY</span>
}

function Headline({ within, cross }: { within: Comparison; cross: Comparison }) {
  const lr = (c: Comparison) => c.models.find((m) => m.model === 'logistic_regression')
  const best = (c: Comparison) => c.models.reduce((a, b) => (b.test_macro_f1 > a.test_macro_f1 ? b : a))
  const rows = [
    { c: within, title: 'Within-event', text: `New scenes from the same ${within.test_events.length} disasters the models were trained on.` },
    { c: cross, title: 'Cross-event', text: `A disaster never seen in training (${cross.holdout_event ?? cross.test_events.join(', ')}).` },
  ]
  return (
    <div className="grid grid-cols-2 gap-4">
      {rows.map(({ c, title, text }) => (
        <div key={c.design} className="rounded-[3px] border border-rule bg-paper/50 p-3">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-ink">{title} <span className="font-normal text-ink-3">· {c.design_label}</span></p>
            <Chip kind="test" />
          </div>
          <p className="mt-0.5 text-xs text-ink-3">{text}</p>
          <p className="mt-2 flex items-baseline gap-2">
            <span className="num text-3xl font-medium text-ink">{fmt(lr(c)?.test_macro_f1)}</span>
            <span className="text-xs text-ink-2">macro-F1 · Logistic Regression (default)</span>
          </p>
          <p className="num mt-1 text-xs text-ink-3">best: {best(c).model.replaceAll('_', ' ')} {fmt(best(c).test_macro_f1)}</p>
        </div>
      ))}
    </div>
  )
}

export function ReadinessPage() {
  const status = useQuery({ queryKey: ['status'], queryFn: api.datasetStatus })
  const models = useQuery({ queryKey: ['models'], queryFn: api.models })
  const within = useQuery({ queryKey: ['comparison', 'within_event'], queryFn: () => api.comparison('within_event'), retry: false })
  const cross = useQuery({ queryKey: ['comparison', 'cross_event'], queryFn: () => api.comparison('cross_event'), retry: false })

  return (
    <div className="space-y-8">
      <div className="flex items-end justify-between gap-6">
        <div>
          <p className="font-mono text-[11px] tracking-[0.14em] text-ink-3 uppercase">Readiness board</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink">Is everything built, and how good is it honestly?</h1>
        </div>
        <Link to="/library?split=test" className="rounded-[3px] bg-signal px-3 py-2 text-sm font-medium text-signal-ink hover:brightness-110">
          Open held-out test scenes →
        </Link>
      </div>

      {status.isPending && <LoadingState label="Checking dataset, features, splits and models…" rows={4} />}
      {status.isError && <ErrorState error={status.error} what="readiness status" />}
      {status.isSuccess && (
        <div className="grid grid-cols-3 gap-x-5 gap-y-6">
          {TILE_ORDER.map(({ key, title }, i) => {
            const tile = status.data.tiles[key]
            if (!tile) return null
            return (
              <CaseFolder key={key} tab={`STATUS · ${String(i + 1).padStart(2, '0')}`} title={title} aside={<ReadyBadge tile={tile} />}>
                <TileBody name={key} tile={tile} />
              </CaseFolder>
            )
          })}
        </div>
      )}

      <CaseFolder tab="RESULTS · HEADLINE" title="How well does it work?"
        aside={<span className="text-xs text-ink-3">4-class damage · each model evaluated on its test set once</span>}>
        {(within.isPending || cross.isPending) && <LoadingState label="Loading saved results…" rows={2} />}
        {within.isError && <ErrorState error={within.error} what="within-event results" />}
        {cross.isError && !within.isError && <ErrorState error={cross.error} what="cross-event results" />}
        {within.isSuccess && cross.isSuccess && (
          <>
            <Headline within={within.data} cross={cross.data} />
            <p className="mt-3 text-xs leading-5 text-ink-2">
              The gap is the honest story: on scenes from disasters it has seen, the model does much better than on a new
              disaster. The full comparison, per-event scores and the confounding check live in the Model Arena.
            </p>
          </>
        )}
      </CaseFolder>

      <CaseFolder tab="MODELS · TRIAGE DESK" title="Models served to the Triage Desk"
        aside={models.isSuccess && <span className="text-xs text-ink-3">{models.data.serving_design_label} · 4-class</span>}>
        {models.isPending && <LoadingState label="Loading models…" />}
        {models.isError && <ErrorState error={models.error} what="models" />}
        {models.isSuccess && (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-rule text-left text-xs text-ink-3">
                <th className="py-1.5 font-normal">Model</th>
                <th className="py-1.5 font-normal">State</th>
                <th className="py-1.5 text-right font-normal">macro-F1 <Chip kind="test" /></th>
                <th className="py-1.5 pl-6 font-normal">Confidence means</th>
                <th className="py-1.5 text-right font-normal">Review below</th>
              </tr>
            </thead>
            <tbody>
              {models.data.models.map((m) => (
                <tr key={m.name} className="border-b border-rule/60 last:border-0">
                  <td className="py-2 text-ink">
                    {m.label}
                    {m.default && <span className="ml-2 rounded-[3px] border border-ink-3 px-1 font-mono text-[10px] tracking-wider text-ink-2">DEFAULT</span>}
                  </td>
                  <td className="py-2">{m.trained ? <span className="text-ink-2">trained</span> : <Chip kind="not-available" label="NOT TRAINED" />}</td>
                  <td className="num py-2 text-right">{fmt(m.test_macro_f1)}</td>
                  <td className="py-2 pl-6 text-xs text-ink-2">{m.confidence}</td>
                  <td className="num py-2 text-right">{fmt(m.review_threshold, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </CaseFolder>

      <div className="flex items-center justify-between border-t border-rule pt-4">
        <DamageLegend />
        <p className="text-xs text-ink-3">Footprints derived from rasterised xBD masks · pixel coordinates only</p>
      </div>
    </div>
  )
}
