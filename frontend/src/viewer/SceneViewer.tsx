/**
 * Before/after viewer: pre image on the left of the swipe line, post on the right; wheel/buttons zoom,
 * drag pans; footprints are SVG polygons in IMAGE PIXEL coordinates (no map, no geography).
 *
 * Three stacked layers share one transform: pre image, post image (clipped in screen space at the swipe
 * position) and the footprint overlay. A click that moved less than 4 px selects a footprint.
 */
import { useCallback, useRef, useState, type KeyboardEvent, type PointerEvent as ReactPointerEvent, type WheelEvent } from 'react'
import { DAMAGE, DamagePatternDefs } from '../theme/damage'
import type { DamageClass, PolygonParts } from '../types/api'

export type OverlayMode = 'ground_truth' | 'prediction' | 'agreement' | 'none'

export interface Footprint {
  uid: string
  polygon: PolygonParts
  trueLabel: DamageClass
  predicted?: DamageClass
  correct?: boolean
}

const MIN_SCALE = 1
const MAX_SCALE = 8

function pathOf(parts: PolygonParts): string {
  return parts.map((rings) => rings.map((ring) => `M${ring.map(([x, y]) => `${x},${y}`).join('L')}Z`).join('')).join('')
}

export function SceneViewer({ preUrl, postUrl, width, height, footprints, mode, selected, onSelect, size = 720 }: {
  preUrl: string
  postUrl: string
  width: number
  height: number
  footprints: Footprint[]
  mode: OverlayMode
  selected: string | null
  onSelect: (uid: string | null) => void
  size?: number
}) {
  const [swipe, setSwipe] = useState(0.5)
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 })
  const drag = useRef<{ x: number; y: number; vx: number; vy: number; moved: boolean } | null>(null)
  const fit = size / Math.max(width, height)

  const clamp = useCallback((scale: number, x: number, y: number) => {
    const max = size * (scale - 1)
    return { scale, x: Math.min(0, Math.max(-max, x)), y: Math.min(0, Math.max(-max, y)) }
  }, [size])

  const zoomAt = (factor: number, cx = size / 2, cy = size / 2) => {
    setView((v) => {
      const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, v.scale * factor))
      const k = scale / v.scale
      return clamp(scale, cx - (cx - v.x) * k, cy - (cy - v.y) * k)
    })
  }

  const onWheel = (e: WheelEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    zoomAt(e.deltaY < 0 ? 1.2 : 1 / 1.2, e.clientX - rect.left, e.clientY - rect.top)
  }
  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: false }
  }
  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    const d = drag.current
    if (!d) return
    const dx = e.clientX - d.x
    const dy = e.clientY - d.y
    if (Math.abs(dx) + Math.abs(dy) > 4) d.moved = true
    if (d.moved) setView((v) => clamp(v.scale, d.vx + dx, d.vy + dy))
  }
  const endDrag = () => { drag.current = null }
  // keyboard: arrows pan, + / - zoom, 0 resets (the building list offers another keyboard route)
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.target !== e.currentTarget) return
    const step = 60
    const moves: Record<string, [number, number]> = { ArrowLeft: [step, 0], ArrowRight: [-step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] }
    if (moves[e.key]) {
      e.preventDefault()
      const [dx, dy] = moves[e.key]
      setView((v) => clamp(v.scale, v.x + dx, v.y + dy))
    } else if (e.key === '+' || e.key === '=') zoomAt(1.5)
    else if (e.key === '-') zoomAt(1 / 1.5)
    else if (e.key === '0') setView({ scale: 1, x: 0, y: 0 })
  }
  const clickFootprint = (uid: string) => {
    if (!drag.current?.moved) onSelect(uid === selected ? null : uid)
  }

  const transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale * fit})`
  const layer = 'absolute top-0 left-0 origin-top-left'
  const stroke = 1.6 / (view.scale * fit) // constant on-screen line width

  const styleFor = (f: Footprint) => {
    if (mode === 'agreement') {
      if (f.correct === undefined) return { fill: 'none', stroke: 'var(--ink-3)', dash: undefined }
      // solid vs dashed carries the meaning without colour (red/green alone fails for colour-blind viewers)
      return f.correct
        ? { fill: 'none', stroke: 'var(--ok)', dash: undefined, width: 1.3 }
        : { fill: 'none', stroke: 'var(--danger)', dash: `${4 * stroke} ${2 * stroke}`, width: 2.2 }
    }
    const cls = mode === 'prediction' ? f.predicted : f.trueLabel
    if (!cls) return { fill: 'none', stroke: 'var(--ink-3)', dash: undefined }
    return { fill: `url(#ov-${DAMAGE[cls].pattern})`, stroke: DAMAGE[cls].color, dash: undefined }
  }

  return (
    <div className="space-y-2">
      <div
        className="relative cursor-grab touch-none overflow-hidden rounded-[3px] border border-rule bg-ink select-none active:cursor-grabbing"
        style={{ width: size, height: size }}
        onWheel={onWheel} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={endDrag} onPointerLeave={endDrag}
        onKeyDown={onKeyDown} tabIndex={0} role="group" aria-label="Scene viewer: arrow keys pan, plus and minus zoom, 0 resets"
        data-testid="scene-viewer"
      >
        <div className={layer} style={{ transform, width, height }}>
          <img src={preUrl} alt="Pre-disaster image" width={width} height={height} draggable={false} />
        </div>
        <div className="absolute inset-0" style={{ clipPath: `inset(0 0 0 ${swipe * 100}%)` }}>
          <div className={layer} style={{ transform, width, height }}>
            <img src={postUrl} alt="Post-disaster image" width={width} height={height} draggable={false} />
          </div>
        </div>
        {mode !== 'none' && (
          <svg className={layer} style={{ transform, width, height }} viewBox={`0 0 ${width} ${height}`} width={width} height={height}
            aria-label="Building footprints">
            <DamagePatternDefs idPrefix="ov-" />
            {footprints.map((f) => {
              const s = styleFor(f)
              const isSelected = f.uid === selected
              return (
                <path key={f.uid} d={pathOf(f.polygon)} fill={s.fill} stroke={isSelected ? 'var(--focus)' : s.stroke}
                  strokeWidth={isSelected ? stroke * 2.4 : stroke * ('width' in s ? (s.width as number) : 1)} strokeDasharray={s.dash} fillRule="evenodd"
                  opacity={selected && !isSelected ? 0.55 : 1} className="cursor-pointer"
                  role="button" tabIndex={0} aria-label={`Building ${f.uid}`} aria-pressed={isSelected}
                  onPointerUp={() => clickFootprint(f.uid)}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(isSelected ? null : f.uid) } }} />
              )
            })}
          </svg>
        )}
        {/* swipe divider */}
        <div className="pointer-events-none absolute inset-y-0 w-px bg-paper shadow-[0_0_0_1px_rgba(0,0,0,0.4)]" style={{ left: `${swipe * 100}%` }} />
        <span className="pointer-events-none absolute top-2 left-2 rounded-[2px] bg-ink/75 px-1.5 py-0.5 font-mono text-[10px] tracking-wider text-paper">PRE</span>
        <span className="pointer-events-none absolute top-2 right-2 rounded-[2px] bg-ink/75 px-1.5 py-0.5 font-mono text-[10px] tracking-wider text-paper">POST</span>
      </div>
      <div className="flex items-center gap-3" style={{ width: size }}>
        <label className="flex flex-1 items-center gap-2 text-xs text-ink-3">
          <span className="font-mono">pre</span>
          <input type="range" min={0} max={1} step={0.005} value={swipe} onChange={(e) => setSwipe(Number(e.target.value))}
            aria-label="Swipe between pre- and post-disaster image" className="flex-1 accent-[var(--ink)]" />
          <span className="font-mono">post</span>
        </label>
        <div className="flex items-center gap-1" role="group" aria-label="Zoom">
          <button type="button" onClick={() => zoomAt(1 / 1.5)} aria-label="Zoom out" className="h-7 w-7 rounded-[3px] border border-rule-strong text-sm text-ink-2 hover:bg-sheet-2">−</button>
          <span className="num w-12 text-center text-xs text-ink-2">{Math.round(view.scale * 100)}%</span>
          <button type="button" onClick={() => zoomAt(1.5)} aria-label="Zoom in" className="h-7 w-7 rounded-[3px] border border-rule-strong text-sm text-ink-2 hover:bg-sheet-2">+</button>
          <button type="button" onClick={() => setView({ scale: 1, x: 0, y: 0 })} className="h-7 rounded-[3px] border border-rule-strong px-2 text-xs text-ink-2 hover:bg-sheet-2">Reset</button>
        </div>
      </div>
    </div>
  )
}
