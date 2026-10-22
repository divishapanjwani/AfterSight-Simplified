/** Scene Library: browse cached scenes, defaulting to held-out TEST scenes. Filters live in the URL. */
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { Link, useSearchParams } from 'react-router-dom'
import { CaseFolder } from '../components/CaseFolder'
import { Chip, SplitChip } from '../components/Chip'
import { DamageBar } from '../components/DamageBar'
import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { api, sceneImageUrl } from '../services/api'
import type { SceneSummary } from '../types/api'
import { DamageLegend } from '../theme/damage'

const PAGE_SIZE = 20
const SPLITS = [
  { value: 'test', label: 'Test (held out)' },
  { value: 'val', label: 'Validation' },
  { value: 'train', label: 'Training' },
  { value: 'all', label: 'All scenes' },
] as const
const SORTS = [
  { value: 'scene_id', label: 'Scene id' },
  { value: '-n_buildings', label: 'Most buildings' },
  { value: '-destroyed', label: 'Most destroyed' },
] as const

const SPLIT_WARNING: Record<string, string> = {
  train: 'Training scenes were used to fit the models. Any accuracy you see on them is optimistic.',
  val: 'Validation scenes were used to choose model settings. Accuracy on them is somewhat optimistic.',
  all: 'This list mixes training, validation and held-out test scenes — check each row’s split chip.',
}

function SceneRow({ scene }: { scene: SceneSummary }) {
  const flagged = scene.quality.alignment_flag
  return (
    <tr className="border-b border-rule/60 align-middle last:border-0 hover:bg-paper/60">
      <td className="py-2 pr-3">
        <div className="flex gap-1">
          {(['pre', 'post'] as const).map((phase) => (
            <figure key={phase} className="relative">
              <img src={sceneImageUrl(scene.scene_id, phase)} alt={`${phase}-disaster thumbnail of ${scene.scene_id}`}
                width={56} height={56} loading="lazy" className="h-14 w-14 rounded-[2px] border border-rule object-cover" />
              <figcaption className="absolute bottom-0.5 left-0.5 rounded-[2px] bg-ink/70 px-1 font-mono text-[9px] text-paper">{phase}</figcaption>
            </figure>
          ))}
        </div>
      </td>
      <td className="py-2 pr-3">
        <p className="num text-[13px] whitespace-nowrap text-ink">{scene.scene_id}</p>
        <p className="text-xs text-ink-3">{scene.event} · {scene.disaster_type ?? 'unknown type'} <span className="text-[10px]">(config)</span></p>
      </td>
      <td className="py-2 pr-3"><SplitChip split={scene.split} /></td>
      <td className="num py-2 pr-3 text-right text-[13px]">{scene.n_buildings}</td>
      <td className="py-2 pr-3 pl-2"><DamageBar counts={scene.ground_truth_counts} id={scene.scene_id} /></td>
      <td className="py-2 pr-3 text-xs">
        {scene.warnings.length === 0 ? <span className="text-ink-3">—</span> : (
          <ul className="space-y-0.5">
            {scene.warnings.map((w) => (
              <li key={w} className={`flex gap-1 ${flagged && w.startsWith('Pre/post') ? 'text-caution' : 'text-ink-2'}`}>
                <span aria-hidden="true">⚠</span><span className="line-clamp-1" title={w}>{w}</span>
              </li>
            ))}
          </ul>
        )}
      </td>
      <td className="py-2 text-right">
        {scene.n_buildings > 0 ? (
          <Link to={`/triage/${scene.scene_id}`} className="rounded-[3px] border border-ink px-2 py-1 text-xs whitespace-nowrap text-ink hover:bg-sheet-2">Triage →</Link>
        ) : <span className="text-xs text-ink-3">no buildings</span>}
      </td>
    </tr>
  )
}

export function SceneLibraryPage() {
  const [params, setParams] = useSearchParams()
  const split = params.get('split') ?? 'test' // the Library opens on held-out scenes
  const event = params.get('event') ?? ''
  const disasterType = params.get('type') ?? ''
  const sort = params.get('sort') ?? 'scene_id'
  const page = Math.max(1, Number(params.get('page') ?? 1) || 1)

  const update = (changes: Record<string, string | null>) => {
    const next = new URLSearchParams(params)
    for (const [k, v] of Object.entries(changes)) {
      if (v === null || v === '') next.delete(k)
      else next.set(k, v)
    }
    if (!('page' in changes)) next.delete('page')
    setParams(next)
  }

  const query = useQuery({
    queryKey: ['scenes', split, event, disasterType, sort, page],
    queryFn: () => api.scenes({
      split: split === 'all' ? undefined : split, event: event || undefined, disaster_type: disasterType || undefined,
      sort, page, page_size: PAGE_SIZE,
    }),
    placeholderData: keepPreviousData,
  })
  const data = query.data
  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1

  const select = 'rounded-[3px] border border-rule-strong bg-sheet px-2 py-1 text-sm text-ink'
  return (
    <div className="space-y-5">
      <div className="flex items-end justify-between gap-6">
        <div>
          <p className="font-mono text-[11px] tracking-[0.14em] text-ink-3 uppercase">Scene library</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink">Choose a scene to triage</h1>
          {data && <p className="mt-1 text-xs text-ink-3">Splits shown are from the <span className="text-ink-2">{data.split_design_label}</span> design used by the Triage Desk models.</p>}
        </div>
        <DamageLegend compact />
      </div>

      <div role="toolbar" aria-label="Scene filters" className="flex flex-wrap items-center gap-4 rounded-[4px] border border-rule bg-sheet px-3 py-2.5">
        <div role="radiogroup" aria-label="Split" className="flex overflow-hidden rounded-[3px] border border-rule-strong">
          {SPLITS.map((s) => (
            <button key={s.value} type="button" role="radio" aria-checked={split === s.value} onClick={() => update({ split: s.value })}
              className={`border-r border-rule-strong px-2.5 py-1 text-sm last:border-r-0 ${split === s.value ? 'bg-ink text-paper' : 'bg-sheet text-ink-2 hover:bg-sheet-2'}`}>
              {s.label}
              {data?.facets.split[s.value] !== undefined && <span className="num ml-1 text-[11px] opacity-70">{data.facets.split[s.value]}</span>}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-1.5 text-xs text-ink-3">Event
          <select className={select} value={event} onChange={(e) => update({ event: e.target.value })}>
            <option value="">All events</option>
            {Object.entries(data?.facets.event ?? {}).sort().map(([e, n]) => <option key={e} value={e}>{e} ({n})</option>)}
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-xs text-ink-3">Disaster type
          <select className={select} value={disasterType} onChange={(e) => update({ type: e.target.value })}>
            <option value="">All types</option>
            {Object.keys(data?.facets.disaster_type ?? {}).sort().map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
          <Chip kind="config" />
        </label>
        <label className="ml-auto flex items-center gap-1.5 text-xs text-ink-3">Sort
          <select className={select} value={sort} onChange={(e) => update({ sort: e.target.value })}>
            {SORTS.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
          </select>
        </label>
      </div>

      {SPLIT_WARNING[split] && (
        <div role="note" className="flex items-start gap-2 rounded-[4px] border border-caution bg-caution-soft px-3 py-2 text-sm text-ink">
          <span aria-hidden="true" className="font-mono text-caution">!</span>{SPLIT_WARNING[split]}
        </div>
      )}

      <CaseFolder tab={data ? `SCENES · ${data.total} MATCH` : 'SCENES'}>
        {query.isPending && <LoadingState label="Loading scenes…" rows={5} />}
        {query.isError && <ErrorState error={query.error} what="scenes" />}
        {data && data.items.length === 0 && (
          <EmptyState title="No scenes match these filters."
            action={<button type="button" onClick={() => setParams(new URLSearchParams())} className="text-sm text-ink underline underline-offset-2">Reset to test scenes</button>} />
        )}
        {data && data.items.length > 0 && (
          <>
            <table className="w-full text-sm" aria-busy={query.isFetching}>
              <thead>
                <tr className="border-b border-rule text-left text-xs text-ink-3">
                  <th className="py-1.5 font-normal">Pre / post</th>
                  <th className="py-1.5 font-normal">Scene</th>
                  <th className="py-1.5 font-normal">Split</th>
                  <th className="py-1.5 pr-3 text-right font-normal">Buildings</th>
                  <th className="py-1.5 pl-2 font-normal"><span className="mr-1">Damage</span><Chip kind="ground-truth" /></th>
                  <th className="py-1.5 font-normal">Data quality</th>
                  <th className="py-1.5" />
                </tr>
              </thead>
              <tbody>{data.items.map((s) => <SceneRow key={s.scene_id} scene={s} />)}</tbody>
            </table>
            <nav aria-label="Pages" className="mt-3 flex items-center justify-between text-xs text-ink-3">
              <span className="num">{(page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, data.total)} of {data.total}</span>
              <div className="flex gap-2">
                <button type="button" disabled={page <= 1} onClick={() => update({ page: String(page - 1) })}
                  className="rounded-[3px] border border-rule-strong px-2 py-1 text-ink-2 disabled:opacity-40">← Previous</button>
                <span className="num self-center">page {page} / {pages}</span>
                <button type="button" disabled={page >= pages} onClick={() => update({ page: String(page + 1) })}
                  className="rounded-[3px] border border-rule-strong px-2 py-1 text-ink-2 disabled:opacity-40">Next →</button>
              </div>
            </nav>
          </>
        )}
      </CaseFolder>
    </div>
  )
}
