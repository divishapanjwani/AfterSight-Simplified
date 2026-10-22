/**
 * The single damage scale used everywhere: one ordinal plum ramp (light -> dark = more severe) PLUS a
 * pattern per class, so classes never rely on colour alone. Minor and major are close in lightness on
 * imagery, so their patterns differ most: dots vs diagonal hatch.
 */
import type { DamageClass } from '../types/api'

export const DAMAGE_CLASSES: DamageClass[] = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

export const DAMAGE: Record<DamageClass, { label: string; short: string; color: string; pattern: string; patternName: string }> = {
  'no-damage': { label: 'No damage', short: 'none', color: 'var(--dmg-0)', pattern: 'dmg-p0', patternName: 'outline' },
  'minor-damage': { label: 'Minor damage', short: 'minor', color: 'var(--dmg-1)', pattern: 'dmg-p1', patternName: 'dots' },
  'major-damage': { label: 'Major damage', short: 'major', color: 'var(--dmg-2)', pattern: 'dmg-p2', patternName: 'hatch' },
  destroyed: { label: 'Destroyed', short: 'destroyed', color: 'var(--dmg-3)', pattern: 'dmg-p3', patternName: 'solid' },
}

/** SVG <defs> for the four damage fills. Render once per SVG that uses them. */
export function DamagePatternDefs({ idPrefix = '' }: { idPrefix?: string }) {
  return (
    <defs>
      <pattern id={`${idPrefix}dmg-p0`} width="6" height="6" patternUnits="userSpaceOnUse">
        <rect width="6" height="6" fill="var(--dmg-0)" opacity="0.18" />
      </pattern>
      <pattern id={`${idPrefix}dmg-p1`} width="5" height="5" patternUnits="userSpaceOnUse">
        <rect width="5" height="5" fill="var(--dmg-1)" opacity="0.15" />
        <circle cx="2.5" cy="2.5" r="1.1" fill="var(--dmg-1)" />
      </pattern>
      <pattern id={`${idPrefix}dmg-p2`} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="5" height="5" fill="var(--dmg-2)" opacity="0.2" />
        <rect width="2" height="5" fill="var(--dmg-2)" />
      </pattern>
      <pattern id={`${idPrefix}dmg-p3`} width="4" height="4" patternUnits="userSpaceOnUse">
        <rect width="4" height="4" fill="var(--dmg-3)" />
      </pattern>
    </defs>
  )
}

/** Small swatch (pattern + outline) always shown next to the class name. */
export function DamageSwatch({ cls, size = 14 }: { cls: DamageClass; size?: number }) {
  const id = `sw-${cls}-`
  return (
    <svg width={size} height={size} viewBox="0 0 14 14" aria-hidden="true" className="shrink-0">
      <DamagePatternDefs idPrefix={id} />
      <rect x="1" y="1" width="12" height="12" fill={`url(#${id}${DAMAGE[cls].pattern})`} stroke={DAMAGE[cls].color} strokeWidth="1.5" />
    </svg>
  )
}

export function DamageLegend({ compact = false }: { compact?: boolean }) {
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-2" aria-label="Damage scale">
      {DAMAGE_CLASSES.map((cls) => (
        <li key={cls} className="flex items-center gap-1.5">
          <DamageSwatch cls={cls} />
          <span>{DAMAGE[cls].label}</span>
          {!compact && <span className="text-ink-3">· {DAMAGE[cls].patternName}</span>}
        </li>
      ))}
    </ul>
  )
}
