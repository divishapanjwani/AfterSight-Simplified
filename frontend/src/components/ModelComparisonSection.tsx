/**
 * Model Arena · "Model Comparison & Ensemble Analysis": KNN, Decision Tree, SVM, Random Forest, the soft-voting
 * ensemble and the GridSearchCV-optimised versions. Every number comes from /api/results/model-comparison
 * (scripts/run_model_comparison.py). Only accuracy, precision, recall, F1, macro-F1 and confusion matrices.
 */
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { BarList } from '../charts/BarList'
import { Heatmap } from '../charts/Heatmap'
import { api } from '../services/api'
import { DAMAGE, DamageSwatch } from '../theme/damage'
import type { ComparedModel, DamageClass, ModelComparison } from '../types/api'
import { Chip } from './Chip'
import { ErrorState, LoadingState } from './States'

type Design = 'within_event' | 'cross_event'
const DESIGNS: { value: Design; label: string }[] = [{ value: 'within_event', label: 'within-event' }, { value: 'cross_event', label: 'cross-event' }]
const f3 = (v: number | null | undefined) => (v === null || v === undefined ? '—' : v.toFixed(3))
const short = (c: string) => DAMAGE[c as DamageClass]?.short ?? c

function Toggle<T extends string>({ value, options, onChange, label }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="flex overflow-hidden rounded-[3px] border border-rule-strong">
      {options.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={value === o.value} onClick={() => onChange(o.value)}
          className={`border-r border-rule-strong px-2.5 py-1 text-xs last:border-r-0 ${value === o.value ? 'bg-ink text-paper' : 'bg-sheet text-ink-2 hover:bg-sheet-2'}`}>{o.label}</button>
      ))}
    </div>
  )
}

const kindLabel = (m: ComparedModel) =>
  m.kind !== 'ensemble' ? m.kind : m.ensemble_type === 'serial' ? 'serial ensemble (boosting)' : 'parallel ensemble'

function KindTag({ m }: { m: ComparedModel }) {
  return <span className="ml-2 rounded-[3px] border border-rule-strong px-1 font-mono text-[10px] tracking-wider text-ink-2 uppercase">{kindLabel(m)}</span>
}

/** Every model that appears in either design, in the within-event order (optimised ones exist within-event only). */
function allModels(doc: ModelComparison): ComparedModel[] {
  const seen = new Map<string, ComparedModel>()
  for (const d of Object.values(doc.designs)) for (const m of d.models) if (!seen.has(m.model)) seen.set(m.model, m)
  return [...seen.values()]
}

const find = (doc: ModelComparison, design: Design, model: string) => doc.designs[design]?.models.find((m) => m.model === model)

function ComparisonTable({ doc }: { doc: ModelComparison }) {
  const cols = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1'] as const
  const head = ['accuracy', 'precision (macro)', 'recall (macro)', 'macro-F1']
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-left text-ink-3"><th className="py-1 font-normal" />
          {DESIGNS.map((d) => <th key={d.value} colSpan={4} className="border-b border-rule py-1 text-center font-normal">{d.label} · <Chip kind="test" /></th>)}</tr>
        <tr className="border-b border-rule text-left text-ink-3">
          <th className="py-1.5 font-normal">Model</th>
          {DESIGNS.flatMap((d) => head.map((h) => <th key={d.value + h} className="py-1.5 text-right font-normal">{h}</th>))}
        </tr>
      </thead>
      <tbody>
        {allModels(doc).map((m) => (
          <tr key={m.model} className="border-b border-rule/60 last:border-0">
            <td className="py-1.5 text-ink">{m.label}<KindTag m={m} /></td>
            {DESIGNS.map((d) => {
              const row = find(doc, d.value, m.model)
              return row ? cols.map((c) => <td key={d.value + c} className={`num py-1.5 text-right ${c === 'macro_f1' ? 'font-medium text-ink' : ''}`}>{f3(row.metrics[c])}</td>)
                : <td key={d.value} colSpan={4} className="py-1.5 text-center text-ink-3">not evaluated</td>
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function MacroF1Bars({ doc }: { doc: ModelComparison }) {
  const [design, setDesign] = useState<Design>('within_event')
  const models = doc.designs[design]?.models ?? []
  return (
    <div className="space-y-2">
      <Toggle label="Bar chart design" value={design} onChange={setDesign} options={DESIGNS} />
      <BarList width={260} items={models.map((m) => ({ label: `${m.label} · ${kindLabel(m)}`, value: m.metrics.macro_f1 }))} />
      <p className="text-[11px] text-ink-3">Test macro-F1, {doc.designs[design]?.design_label}.{design === 'cross_event' && ' Optimised versions were tuned and tested within-event only, so they are not shown here.'}</p>
    </div>
  )
}

function GridSearchTable({ doc }: { doc: ModelComparison }) {
  const fmt = (o: Record<string, unknown>) => Object.entries(o).map(([k, v]) => `${k}=${Array.isArray(v) ? `[${v.join(', ')}]` : String(v)}`).join(' · ')
  return (
    <div className="space-y-1">
      <table className="w-full text-xs">
        <thead><tr className="border-b border-rule text-left text-ink-3">
          <th className="py-1.5 font-normal">Model</th><th className="py-1.5 font-normal">Grid</th><th className="py-1.5 font-normal">Best params</th>
          <th className="py-1.5 text-right font-normal">best CV macro-F1</th><th className="py-1.5 pl-6 text-right font-normal">test macro-F1 default → optimised</th>
        </tr></thead>
        <tbody>
          {Object.entries(doc.grid_search).map(([k, g]) => (
            <tr key={k} className="border-b border-rule/60 align-top last:border-0">
              <td className="py-1.5 text-ink">{g.label}<span className="block text-[10px] text-ink-3">{g.trained_on}</span></td>
              <td className="num py-1.5 text-ink-2">{fmt(g.grid)}{Object.keys(g.fixed).length > 0 && <span className="block text-ink-3">fixed: {fmt(g.fixed)}</span>}</td>
              <td className="num py-1.5 text-ink">{fmt(g.best_params)}</td>
              <td className="num py-1.5 text-right">{f3(g.best_cv_macro_f1)}</td>
              <td className="num py-1.5 pl-6 text-right text-ink">{f3(g.test_macro_f1_default)} → {f3(g.test_macro_f1_optimised)}
                <span className="ml-1 text-ink-3">({g.difference >= 0 ? '+' : ''}{g.difference.toFixed(3)})</span></td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-[11px] text-ink-3">{Object.values(doc.grid_search)[0]?.cv}. The test set was never used for selection; each optimised model was evaluated on the within-event test set once.</p>
    </div>
  )
}

function SelectedModel({ doc }: { doc: ModelComparison }) {
  const [design, setDesign] = useState<Design>('within_event')
  const [model, setModel] = useState('soft_voting')
  const models = doc.designs[design]?.models ?? []
  const selected = models.find((m) => m.model === model) ?? models[0]
  if (!selected) return null
  const cm = selected.metrics.confusion_matrix
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-3">
        <Toggle label="Compared model" value={selected.model} onChange={setModel} options={models.map((m) => ({ value: m.model, label: m.label }))} />
        <Toggle label="Confusion design" value={design} onChange={setDesign} options={DESIGNS} />
      </div>
      <div className="flex flex-wrap items-start gap-8">
        <Heatmap rows={cm.labels} cols={cm.labels} matrix={cm.rows_true_cols_pred} rowTitle="ground truth" colTitle="predicted"
          caption={`TEST SET · ${selected.label} · ${doc.designs[design].design_label} · n = ${selected.metrics.n.toLocaleString('en-GB')} · row % = recall`} />
        <table className="text-xs" aria-label="Per-class metrics">
          <thead><tr className="border-b border-rule text-left text-ink-3"><th className="py-1 pr-4 font-normal">Class</th>
            <th className="py-1 pr-3 text-right font-normal">precision</th><th className="py-1 pr-3 text-right font-normal">recall</th><th className="py-1 pr-3 text-right font-normal">F1</th>
            <th className="py-1 text-right font-normal">support</th></tr></thead>
          <tbody>
            {cm.labels.map((c) => {
              const v = selected.metrics.per_class[c]
              return (
                <tr key={c} className="border-b border-rule/60 last:border-0">
                  <td className="py-1 pr-4 text-ink"><span className="inline-flex items-center gap-1"><DamageSwatch cls={c as DamageClass} size={10} />{short(c)}</span></td>
                  <td className="num py-1 pr-3 text-right">{f3(v?.precision)}</td><td className="num py-1 pr-3 text-right">{f3(v?.recall)}</td>
                  <td className="num py-1 pr-3 text-right font-medium">{f3(v?.f1)}</td><td className="num py-1 text-right">{v?.support.toLocaleString('en-GB')}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function EnsembleExplanation({ doc }: { doc: ModelComparison }) {
  const within = doc.designs.within_event?.models ?? []
  const ens = within.find((m) => m.model === 'soft_voting')
  const serial = within.find((m) => m.ensemble_type === 'serial')
  if (!ens) return null
  const members = within.filter((m) => ens.members?.includes(m.model))
  const best = members.reduce((a, b) => (b.metrics.macro_f1 > a.metrics.macro_f1 ? b : a))
  const crossOf = (id: string) => find(doc, 'cross_event', id)?.metrics.macro_f1
  const crossMembers = members.map((m) => ({ m, v: crossOf(m.model) }))
  const crossBest = crossMembers.reduce((a, b) => ((b.v ?? -1) > (a.v ?? -1) ? b : a))
  const verdict = (e: number, b: number) => (e > b ? 'higher than' : e < b ? 'lower than' : 'equal to')
  return (
    <div className="grid grid-cols-[1fr_1fr] gap-8 text-xs leading-5 text-ink-2">
      <div className="space-y-2">
        <p><b className="text-ink">Which models.</b> {members.map((m) => m.label).join(', ')}, combined with equal weights ({ens.implementation}).</p>
        <p><b className="text-ink">Soft voting.</b> Each model gives a probability-like score for each of the four classes. The ensemble averages the
          three scores per class and predicts the class with the highest average, so a confident model can outvote two unsure ones.</p>
        <p><b className="text-ink">Parallel vs serial.</b> A <i>parallel</i> ensemble trains its members independently and combines them (soft voting here;
          the Random Forest is itself a parallel ensemble of 200 trees, by bagging). A <i>serial</i> ensemble (boosting, e.g. AdaBoost) trains
          each new model to fix the previous ones' mistakes.{serial
            ? <> Here: {serial.label}, macro-F1 <span className="num text-ink">{f3(serial.metrics.macro_f1)}</span> within-event and <span className="num text-ink">{f3(crossOf(serial.model))}</span> cross-event.</>
            : ' No serial ensemble is shown here.'}</p>
      </div>
      <div className="space-y-2">
        <p><b className="text-ink">Real result, within-event.</b> Soft voting reaches macro-F1 <span className="num text-ink">{f3(ens.metrics.macro_f1)}</span>, {verdict(ens.metrics.macro_f1, best.metrics.macro_f1)} its
          best member ({best.label}, <span className="num">{f3(best.metrics.macro_f1)}</span>). Members: {members.map((m) => `${m.label} ${f3(m.metrics.macro_f1)}`).join(' · ')}.</p>
        <p><b className="text-ink">Cross-event.</b> Soft voting <span className="num text-ink">{f3(crossOf(ens.model))}</span> vs best member {crossBest.m.label} <span className="num">{f3(crossBest.v)}</span>
          {' '}({members.map((m) => `${m.label} ${f3(crossOf(m.model))}`).join(' · ')}).</p>
        <p className="text-ink-3">The SVM in the ensemble was trained on a scene-grouped sample of {doc.designs.within_event.svm_sample.buildings.toLocaleString('en-GB')} of {doc.designs.within_event.svm_sample.of_training_buildings.toLocaleString('en-GB')} training buildings.</p>
      </div>
    </div>
  )
}

export function ModelComparisonSection() {
  const q = useQuery({ queryKey: ['model-comparison'], queryFn: api.modelComparison, retry: false })
  if (q.isPending) return <LoadingState label="Loading the model comparison…" />
  if (q.isError) return <ErrorState error={q.error} what="the model comparison" />
  const doc = q.data
  return (
    <div className="space-y-6">
      <section aria-label="A. Comparison table"><h3 className="mb-2 text-sm font-medium text-ink">A. All models on the test sets</h3><ComparisonTable doc={doc} /></section>
      <section aria-label="B. Macro-F1 chart"><h3 className="mb-2 text-sm font-medium text-ink">B. Macro-F1</h3><MacroF1Bars doc={doc} /></section>
      <section aria-label="C. GridSearchCV"><h3 className="mb-2 text-sm font-medium text-ink">C. GridSearchCV (grouped by scene)</h3><GridSearchTable doc={doc} /></section>
      <section aria-label="D and E. Selected model"><h3 className="mb-2 text-sm font-medium text-ink">D–E. Confusion matrix and per-class metrics</h3><SelectedModel doc={doc} /></section>
      <section aria-label="F. Ensemble"><h3 className="mb-2 text-sm font-medium text-ink">F. How the ensemble works</h3><EnsembleExplanation doc={doc} /></section>
      <p className="text-[11px] text-ink-3">{doc.notes.join(' ')}</p>
    </div>
  )
}
