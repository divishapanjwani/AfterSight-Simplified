/** Plain-English labels for the 37 features (definitions: docs/FEATURES.md). The raw name is always shown too. */
const C: Record<string, string> = { r: 'red', g: 'green', b: 'blue' }

const LABELS: Record<string, string> = {
  gray_absdiff_mean: 'Mean pixel change (grey)',
  brightness_ratio: 'Brightness ratio post ÷ pre',
  gray_std_pre: 'Roof roughness, pre', gray_std_post: 'Roof roughness, post', gray_std_delta: 'Roughness change',
  edge_density_pre: 'Edge density, pre', edge_density_post: 'Edge density, post', edge_density_delta: 'Edge density change',
  glcm_contrast_pre: 'Texture contrast, pre', glcm_contrast_post: 'Texture contrast, post', glcm_contrast_delta: 'Texture contrast change',
  glcm_homogeneity_pre: 'Texture smoothness, pre', glcm_homogeneity_post: 'Texture smoothness, post',
  glcm_homogeneity_delta: 'Texture smoothness change',
  ring_dmean: 'Surroundings brightness change', ring_absdiff_mean: 'Surroundings pixel change',
  contrast_change: 'Building-vs-surroundings contrast change',
  area_px: 'Footprint area (px)', compactness: 'Footprint compactness',
}

export function featureLabel(name: string): string {
  const base = name.replace(/_vs_scene$/, '')
  let label = LABELS[base]
  let m: RegExpMatchArray | null
  if (!label && (m = base.match(/^(pre|post)_(mean|std)_([rgb])$/))) {
    label = `${m[2] === 'mean' ? 'Mean' : 'Spread of'} ${C[m[3]]}, ${m[1]}`
  } else if (!label && (m = base.match(/^abs_dmean_([rgb])$/))) {
    label = `Change in mean ${C[m[1]]}`
  } else if (!label && (m = base.match(/^dstd_([rgb])$/))) {
    label = `Change in ${C[m[1]]} spread`
  }
  return (label ?? base) + (name.endsWith('_vs_scene') ? ' (vs scene)' : '')
}

/** Short number formatting that keeps small feature values readable. */
export function fmtValue(v: number): string {
  const a = Math.abs(v)
  if (a >= 1000) return v.toLocaleString('en-GB', { maximumFractionDigits: 0 })
  if (a >= 10) return v.toFixed(1)
  if (a >= 1) return v.toFixed(2)
  return v.toFixed(3)
}
