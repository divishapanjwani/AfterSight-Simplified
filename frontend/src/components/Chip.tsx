/** Provenance chips: every number on screen says where it comes from. Glyph + word, never colour alone. */

export type ChipKind = 'predicted' | 'ground-truth' | 'test' | 'validation' | 'training' | 'not-available' | 'config' | 'demo'

const CHIPS: Record<ChipKind, { text: string; glyph: string; className: string; title: string }> = {
  predicted: { text: 'PREDICTED', glyph: '◆', className: 'border-ink text-ink', title: 'Output of a model' },
  'ground-truth': { text: 'GROUND TRUTH', glyph: '●', className: 'border-ink-2 text-ink-2 border-double border-[3px]', title: 'Reference labels from xBD annotations (mask-derived), used for research evaluation' },
  test: { text: 'TEST SET', glyph: '▲', className: 'border-ink-2 text-ink-2', title: 'Held-out test data: never used for training or tuning' },
  validation: { text: 'VALIDATION', glyph: '△', className: 'border-caution text-caution bg-caution-soft', title: 'Used to tune model settings — somewhat optimistic' },
  training: { text: 'TRAINING SCENE', glyph: '!', className: 'border-caution text-caution bg-caution-soft', title: 'Used to fit the models — results here are optimistic' },
  'not-available': { text: 'NOT AVAILABLE', glyph: '–', className: 'border-dashed border-ink-3 text-ink-3', title: 'Not computed or not built yet' },
  config: { text: 'FROM CONFIG', glyph: '⚙', className: 'border-dashed border-ink-3 text-ink-3', title: 'Set in config/events.yaml, not stored in the dataset' },
  demo: { text: 'DEMO', glyph: '◇', className: 'border-danger text-danger', title: 'Demo data — not real results' },
}

export function Chip({ kind, label }: { kind: ChipKind; label?: string }) {
  const c = CHIPS[kind]
  return (
    <span title={c.title}
      className={`inline-flex items-center gap-1 rounded-[3px] border px-1.5 py-px font-mono text-[10px] leading-4 font-medium tracking-wider whitespace-nowrap ${c.className}`}>
      <span aria-hidden="true">{c.glyph}</span>
      {label ?? c.text}
    </span>
  )
}

/** Split -> chip. not_in_manifest gets NOT AVAILABLE with an explanatory label. */
export function SplitChip({ split }: { split: 'train' | 'val' | 'test' | 'not_in_manifest' }) {
  if (split === 'test') return <Chip kind="test" />
  if (split === 'val') return <Chip kind="validation" />
  if (split === 'train') return <Chip kind="training" />
  return <Chip kind="not-available" label="NO SPLIT" />
}
