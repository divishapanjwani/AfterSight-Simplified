/**
 * Model Arena: every saved result, read from the API. The first chart tells the honest story — each model
 * on new scenes of known disasters (within-event) vs an unseen disaster (cross-event).
 */
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { BarList } from '../charts/BarList'
import { Dumbbell } from '../charts/Dumbbell'
import { Heatmap } from '../charts/Heatmap'
import { CaseFolder } from '../components/CaseFolder'
import { Chip } from '../components/Chip'
import { ErrorState, LoadingState, NotReadyState } from '../components/States'
import { ApiError, api } from '../services/api'
import { DAMAGE, DamageSwatch } from '../theme/damage'
import { featureLabel, fmtValue } from '../theme/features'
import type { Comparison, DamageClass, TreeNode } from '../types/api'

const NAMES: Record<string, string> = {
  baseline: 'Threshold baseline', logistic_regression: 'Logistic Regression', knn: 'KNN', decision_tree: 'Decision Tree', random_forest: 'Random Forest',
}
const name = (m: string, optional?: boolean) => `${NAMES[m] ?? m}${optional ? ' (optional)' : ''}`
const f3 = (v: number | null | undefined) => (v === null || v === undefined ? 'not available' : v.toFixed(3))
const SECTIONS = ['story', 'table', 'confusion', 'events', 'tree', 'confounding', 'three', 'experiments'] as const
const SECTION_LABELS: Record<(typeof SECTIONS)[number], string> = {
  story: 'Honest headline', table: 'Comparison', confusion: 'Confusion', events: 'Per event', tree: 'Decision tree',
  confounding: 'Confounding', three: '3-class', experiments: 'Experiments',
}

function useComparison(design: string, scheme: string) {
  return useQuery({ queryKey: ['comparison', design, scheme], queryFn: () => api.comparison(design, scheme), retry: false })
}

function Segmented<T extends string>({ value, options, onChange, label }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="flex overflow-hidden rounded-[3px] border border-rule-strong">
      {options.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={value === o.value} onClick={() => onChange(o.value)}
          className={`border-r border-rule-strong px-2.5 py-1 text-xs last:border-r-0 ${value === o.value ? 'bg-ink text-paper' : 'bg-sheet text-ink-2 hover:bg-sheet-2'}`}>{o.label}</button>
      ))}
    </div>
  )
}

function Story({ within, cross, title }: { within: Comparison; cross: Comparison; title: string }) {
  const rows = within.models.map((m) => ({
    label: name(m.model, m.optional), a: m.test_macro_f1, b: cross.models.find((c) => c.model === m.model)?.test_macro_f1 ?? null,
  }))
  return (
    <div className="grid grid-cols-[1fr_300px] gap-6">
      <Dumbbell rows={rows} aLabel="within-event · new scenes, known disasters" bLabel={`cross-event · unseen disaster (${cross.holdout_event ?? cross.test_events.join(', ')})`}
        axisLabel={`${title} · test macro-F1 (each model evaluated on its test set once)`} />
      <div className="space-y-3 text-xs leading-5 text-ink-2">
        <p><b className="text-ink">Within-event.</b> {within.split_note}</p>
        <p><b className="text-ink">Cross-event.</b> {cross.split_note}</p>
        <p>Every model loses a large part of its score on the unseen disaster. Read the within-event numbers as an optimistic upper bound;
          the confounding check below shows why.</p>
      </div>
    </div>
  )
}

function ComparisonTable({ c }: { c: Comparison }) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="border-b border-rule text-left text-ink-3">
          <th className="py-1.5 font-normal">Model</th>
          <th className="py-1.5 text-right font-normal">macro-F1 <Chip kind="validation" /></th>
          <th className="py-1.5 text-right font-normal">macro-F1 <Chip kind="test" /></th>
          <th className="py-1.5 text-right font-normal">accuracy</th>
          {c.classes.map((cls) => (
            <th key={cls} className="py-1.5 text-right font-normal">
              <span className="inline-flex items-center gap-1">recall {DAMAGE[cls as DamageClass] ? <DamageSwatch cls={cls as DamageClass} size={10} /> : null}{DAMAGE[cls as DamageClass]?.short ?? cls}</span>
            </th>
          ))}
          <th className="py-1.5 text-right font-normal">train s</th>
          <th className="py-1.5 text-right font-normal">ms / 1k</th>
        </tr>
      </thead>
      <tbody>
        {c.models.map((m) => (
          <tr key={m.model} className="border-b border-rule/60 last:border-0">
            <td className="py-1.5 text-ink">{name(m.model, m.optional)}{m.model === 'baseline' && <span className="ml-1 text-ink-3">· non-ML</span>}</td>
            <td className="num py-1.5 text-right">{f3(m.val_macro_f1)}</td>
            <td className="num py-1.5 text-right font-medium text-ink">{f3(m.test_macro_f1)}</td>
            <td className="num py-1.5 text-right">{f3(m.test_accuracy)}</td>
            {c.classes.map((cls) => <td key={cls} className="num py-1.5 text-right">{f3(m.test_recall[cls])}</td>)}
            <td className="num py-1.5 text-right">{m.train_seconds.toFixed(1)}</td>
            <td className="num py-1.5 text-right">{(m.inference_seconds_per_1000 * 1000).toFixed(1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function ConfusionSection() {
  const [model, setModel] = useState('logistic_regression')
  const [design, setDesign] = useState<'within_event' | 'cross_event'>('within_event')
  const [split, setSplit] = useState<'test' | 'val'>('test')
  const q = useQuery({ queryKey: ['metrics', model, design], queryFn: () => api.modelMetrics(model, design) })
  const m = q.data?.metrics[split]
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-3">
        <Segmented label="Model" value={model} onChange={setModel}
          options={Object.keys(NAMES).map((k) => ({ value: k, label: NAMES[k] }))} />
        <Segmented label="Design" value={design} onChange={setDesign} options={[{ value: 'within_event', label: 'within-event' }, { value: 'cross_event', label: 'cross-event' }]} />
        <Segmented label="Split" value={split} onChange={setSplit} options={[{ value: 'test', label: 'test' }, { value: 'val', label: 'validation' }]} />
      </div>
      {q.isPending && <LoadingState label="Loading metrics…" />}
      {q.isError && <ErrorState error={q.error} what={`${NAMES[model]} metrics`} />}
      {m && (
        <div className="flex items-start gap-8">
          <Heatmap rows={m.confusion_matrix.labels} cols={m.confusion_matrix.labels} matrix={m.confusion_matrix.rows_true_cols_pred}
            rowTitle="ground truth" colTitle="predicted" caption={`${split === 'test' ? 'TEST SET' : 'VALIDATION'} · n = ${m.n.toLocaleString('en-GB')} · row % = recall`} />
          <dl className="space-y-1 text-xs">
            <div><dt className="text-ink-3">macro-F1</dt><dd className="num text-lg text-ink">{f3(m.macro_f1)}</dd></div>
            <div><dt className="text-ink-3">weighted-F1</dt><dd className="num text-ink">{f3(m.weighted_f1)}</dd></div>
            <div><dt className="text-ink-3">accuracy</dt><dd className="num text-ink">{f3(m.accuracy)}</dd></div>
            {q.data && <div><dt className="text-ink-3">hyperparameters (tuned on validation)</dt><dd className="num text-ink-2">{JSON.stringify(q.data.hyperparameters)}</dd></div>}
            {m.notes.map((n) => <p key={n} className="text-ink-3">{n}</p>)}
          </dl>
        </div>
      )}
    </div>
  )
}

function EventsSection() {
  const [model, setModel] = useState('logistic_regression')
  const q = useQuery({ queryKey: ['metrics', model, 'within_event'], queryFn: () => api.modelMetrics(model, 'within_event') })
  const events = q.data ? Object.entries(q.data.metrics.test_per_event) : []
  return (
    <div className="space-y-3">
      <Segmented label="Model" value={model} onChange={setModel} options={Object.keys(NAMES).filter((k) => k !== 'baseline').map((k) => ({ value: k, label: NAMES[k] }))} />
      {q.isPending && <LoadingState label="Loading per-event metrics…" />}
      {q.isError && <ErrorState error={q.error} what="per-event metrics" />}
      {q.data && (
        <div className="grid grid-cols-[1fr_320px] gap-6">
          <table className="w-full text-xs">
            <thead><tr className="border-b border-rule text-left text-ink-3"><th className="py-1.5 font-normal">Event (within-event test)</th>
              <th className="py-1.5 text-right font-normal">n</th><th className="py-1.5 text-right font-normal">macro-F1</th>
              <th className="py-1.5 text-right font-normal">recall major</th><th className="py-1.5 text-right font-normal">recall destroyed</th></tr></thead>
            <tbody>
              {events.map(([e, m]) => (
                <tr key={e} className="border-b border-rule/60 last:border-0">
                  <td className="py-1.5 text-ink">{e}</td><td className="num py-1.5 text-right">{m.n.toLocaleString('en-GB')}</td>
                  <td className="num py-1.5 text-right font-medium">{f3(m.macro_f1)}</td>
                  <td className="num py-1.5 text-right">{f3(m.per_class['major-damage']?.recall)}</td>
                  <td className="num py-1.5 text-right">{f3(m.per_class.destroyed?.recall)}</td>
                </tr>
              ))}
              <tr className="border-t border-rule-strong"><td className="py-1.5 text-ink">All events pooled</td><td className="num py-1.5 text-right">{q.data.metrics.test.n.toLocaleString('en-GB')}</td>
                <td className="num py-1.5 text-right font-medium">{f3(q.data.metrics.test.macro_f1)}</td><td /><td /></tr>
            </tbody>
          </table>
          <p className="text-xs leading-5 text-ink-2">If every single event scores <b>below</b> the pooled number, part of the pooled score comes from telling the
            disasters apart (each has its own typical damage mix), not from reading damage on each roof.</p>
        </div>
      )}
    </div>
  )
}

function TreeNodeView({ node, classes }: { node: TreeNode; classes: string[] }) {
  const summary = (
    <span className="inline-flex items-center gap-2 text-xs">
      {node.feature ? <span className="text-ink">{featureLabel(node.feature)} ≤ <span className="num">{fmtValue(node.threshold_raw ?? 0)}</span></span>
        : <span className="text-ink-2">{node.truncated ? 'deeper levels hidden' : 'leaf'}</span>}
      <span className="num text-ink-3">n={node.samples.toLocaleString('en-GB')}</span>
      <span className="inline-flex items-center gap-1 text-ink-2"><DamageSwatch cls={node.majority as DamageClass} size={10} />{DAMAGE[node.majority as DamageClass]?.short}</span>
      <span className="num text-[10px] text-ink-3">{classes.map((c) => `${((node.class_fractions[c] ?? 0) * 100).toFixed(0)}`).join('/')}%</span>
    </span>
  )
  if (!node.left || !node.right) return <div className="py-0.5 pl-4">{summary}</div>
  return (
    <details open={node.depth < 2} className="pl-4">
      <summary className="cursor-pointer py-0.5">{summary}</summary>
      <div className="border-l border-rule">
        <p className="pl-4 font-mono text-[10px] text-ink-3">yes ↓</p>
        <TreeNodeView node={node.left} classes={classes} />
        <p className="pl-4 font-mono text-[10px] text-ink-3">no ↓</p>
        <TreeNodeView node={node.right} classes={classes} />
      </div>
    </details>
  )
}

function TreeSection() {
  const [depth, setDepth] = useState(3)
  const q = useQuery({ queryKey: ['tree', depth], queryFn: () => api.tree(depth) })
  return (
    <div className="space-y-2">
      <label className="flex items-center gap-2 text-xs text-ink-3">Show depth
        <input type="range" min={1} max={8} value={depth} onChange={(e) => setDepth(Number(e.target.value))} aria-label="Tree depth shown" />
        <span className="num text-ink">{depth}</span>
      </label>
      {q.isPending && <LoadingState label="Loading the tree…" />}
      {q.isError && <ErrorState error={q.error} what="the decision tree" />}
      {q.data && (
        <>
          <p className="text-xs text-ink-3">Within-event 4-class tree: {q.data.n_nodes} nodes, depth {q.data.max_depth}, {q.data.n_leaves} leaves. Thresholds in raw feature units.
            Leaf shares (no/minor/major/destroyed) are class-weighted training fractions.</p>
          <div className="max-h-[420px] overflow-auto rounded-[3px] border border-rule bg-paper/40 py-2"><TreeNodeView node={q.data.root} classes={q.data.classes} /></div>
        </>
      )}
    </div>
  )
}

function ConfoundingSection() {
  const c = useQuery({ queryKey: ['confounding'], queryFn: api.confounding, retry: false })
  const x = useQuery({ queryKey: ['experiments'], queryFn: api.experiments, retry: false })
  if (c.isPending) return <LoadingState label="Loading the confounding check…" />
  if (c.isError) return <ErrorState error={c.error} what="the confounding check" />
  const m = c.data.test_metrics
  const variants = x.data ? Object.keys(x.data.variants).map((v) => ({ label: `Feature-set experiment · ${v}`, value: x.data!.rows.find((r) => r.variant === v)?.event_prediction_accuracy_within_event ?? null })) : []
  return (
    <div className="grid grid-cols-[1fr_1fr] gap-8">
      <div className="space-y-3">
        <p className="text-sm text-ink-2">A classifier given the <b>same features as the damage models</b> was asked which <b>disaster</b> each building comes from. Event is never an input to any damage model.</p>
        <BarList items={[
          { label: 'Event prediction accuracy (same features)', value: m.accuracy },
          { label: `Always guess the most common event (${c.data.majority_class})`, value: c.data.majority_class_accuracy },
          { label: `Uniform chance (${m.confusion_matrix.labels.length} events)`, value: c.data.chance_accuracy_uniform },
          ...variants,
        ]} />
        <p className="text-xs leading-5 text-ink-2">The features carry a strong signature of <i>which</i> disaster a building is from (sensor, region, vegetation, roofs).
          {x.data
            ? ' The feature-set experiments reduced that signature, but cross-event scores did not improve — the gap is a genuine shift between disasters.'
            : ' Feature-set experiment results are not available, so they are not compared here (see Experiments below).'}</p>
      </div>
      <Heatmap rows={m.confusion_matrix.labels} cols={m.confusion_matrix.labels.map((e) => e.replace('hurricane-', 'h-'))} matrix={m.confusion_matrix.rows_true_cols_pred}
        rowTitle="true event" colTitle="predicted" cell={56} caption={`TEST SET (${c.data.design_label}) · predicting EVENT, analysis only`} />
    </div>
  )
}

function ExperimentsSection() {
  const x = useQuery({ queryKey: ['experiments'], queryFn: api.experiments, retry: false })
  if (x.isPending) return <LoadingState label="Loading experiments…" />
  // The saved summary exists but is broken: say so plainly, never show partial numbers as results.
  if (x.error instanceof ApiError && x.error.code === 'results_invalid') {
    return <NotReadyState title="Saved feature-set experiment results are malformed or incomplete" message={x.error.message} fix={x.error.fix} />
  }
  if (x.isError) return <ErrorState error={x.error} what="the experiments" />  // never run: "not available yet" + command
  return (
    <div className="space-y-3">
      <p className="text-xs text-ink-2">{x.data.note}</p>
      <ul className="text-xs text-ink-2">{Object.entries(x.data.variants).map(([k, v]) => <li key={k}><span className="num text-ink">{k}</span> — {v}</li>)}</ul>
      <table className="w-full text-xs">
        <thead><tr className="border-b border-rule text-left text-ink-3">
          <th className="py-1.5 font-normal">Variant</th><th className="py-1.5 font-normal">Design</th><th className="py-1.5 font-normal">Model</th>
          <th className="py-1.5 text-right font-normal">macro-F1 <Chip kind="test" /></th><th className="py-1.5 text-right font-normal">recall major</th>
          <th className="py-1.5 text-right font-normal">recall destroyed</th><th className="py-1.5 text-right font-normal">event-pred. acc.</th></tr></thead>
        <tbody>
          {x.data.rows.map((r) => (
            <tr key={`${r.variant}-${r.design}-${r.model}`} className="border-b border-rule/60 last:border-0">
              <td className="num py-1.5 text-ink">{r.variant} <span className="text-ink-3">({r.n_features})</span></td><td className="py-1.5 text-ink-2">{r.design.replace('_', '-')}</td>
              <td className="py-1.5 text-ink-2">{NAMES[r.model]}</td><td className="num py-1.5 text-right font-medium">{f3(r.test_macro_f1)}</td>
              <td className="num py-1.5 text-right">{f3(r.test_recall_major)}</td><td className="num py-1.5 text-right">{f3(r.test_recall_destroyed)}</td>
              <td className="num py-1.5 text-right">{r.design === 'within_event' ? f3(r.event_prediction_accuracy_within_event) : ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function ModelArenaPage() {
  const [design, setDesign] = useState<'within_event' | 'cross_event'>('within_event')
  const within = useComparison('within_event', 'four_class')
  const cross = useComparison('cross_event', 'four_class')
  const within3 = useComparison('within_event', 'three_class')
  const cross3 = useComparison('cross_event', 'three_class')
  const table = design === 'within_event' ? within : cross

  return (
    <div className="space-y-7">
      <div>
        <p className="font-mono text-[11px] tracking-[0.14em] text-ink-3 uppercase">Model arena</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink">How good is each model — honestly?</h1>
        <nav aria-label="Arena sections" className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs">
          {SECTIONS.map((s) => <a key={s} href={`#${s}`} className="text-ink-2 underline-offset-2 hover:underline">{SECTION_LABELS[s]}</a>)}
        </nav>
      </div>

      <div id="story"><CaseFolder tab="ARENA · 01" title="The same models on known vs unseen disasters" aside={<Chip kind="test" />}>
        {(within.isPending || cross.isPending) && <LoadingState label="Loading comparisons…" />}
        {within.isError && <ErrorState error={within.error} what="within-event results" />}
        {within.isSuccess && cross.isSuccess && <Story within={within.data} cross={cross.data} title="4-class damage" />}
      </CaseFolder></div>

      <div id="table"><CaseFolder tab="ARENA · 02" title="Model comparison (4-class)"
        aside={<Segmented label="Design" value={design} onChange={setDesign} options={[{ value: 'within_event', label: 'within-event' }, { value: 'cross_event', label: 'cross-event' }]} />}>
        {table.isPending && <LoadingState label="Loading…" />}
        {table.isError && <ErrorState error={table.error} what="the comparison" />}
        {table.data && <><ComparisonTable c={table.data} /><p className="mt-2 text-[11px] text-ink-3">{table.data.split_note} Training times are fits on the training split; inference = test-set prediction time per 1,000 buildings (KNN and RF use all cores).</p></>}
      </CaseFolder></div>

      <div id="confusion"><CaseFolder tab="ARENA · 03" title="Confusion matrices"><ConfusionSection /></CaseFolder></div>
      <div id="events"><CaseFolder tab="ARENA · 04" title="Per-event scores (within-event test)"><EventsSection /></CaseFolder></div>
      <div id="tree"><CaseFolder tab="ARENA · 05" title="Decision tree"><TreeSection /></CaseFolder></div>
      <div id="confounding"><CaseFolder tab="ARENA · 06" title="Event-confounding check" aside={<span className="font-mono text-[10px] tracking-wider text-ink-3">ANALYSIS ONLY</span>}><ConfoundingSection /></CaseFolder></div>

      <div id="three"><CaseFolder tab="ARENA · 07 · EXPERIMENT" title="3-class experiment: no-damage / damaged / destroyed"
        aside={<span className="font-mono text-[10px] tracking-wider text-ink-3">SEPARATE EXPERIMENT · 4-CLASS STAYS PRIMARY</span>}>
        {within3.isSuccess && cross3.isSuccess ? <Story within={within3.data} cross={cross3.data} title="3-class (experiment)" />
          : within3.isError ? <ErrorState error={within3.error} what="the 3-class results" /> : <LoadingState label="Loading…" />}
      </CaseFolder></div>

      <div id="experiments"><CaseFolder tab="EXPERIMENTS · FEATURE SETS" title="Experiments: confounding-reduced feature sets"
        aside={<span className="font-mono text-[10px] tracking-wider text-ink-3">NOT SELECTABLE IN THE TRIAGE DESK</span>}>
        <ExperimentsSection />
      </CaseFolder></div>
    </div>
  )
}
