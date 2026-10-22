/** Types mirroring backend/app/schemas/api.py. Only fields the UI uses are listed. */

export type Split = 'train' | 'val' | 'test' | 'not_in_manifest'
export type DamageClass = 'no-damage' | 'minor-damage' | 'major-damage' | 'destroyed'

export interface Health {
  status: string
  version: string
  scope: string
}

export interface StatusTile {
  ready: boolean
  detail: Record<string, unknown>
  fix: string | null
}

export interface DatasetStatus {
  scope: string
  source: string | null
  demo_mode: boolean
  tiles: Record<string, StatusTile>
}

export interface SceneQuality {
  alignment_shift_px: number | null
  alignment_flag: boolean | null
  n_mixed_label: number
  n_touching_border: number
}

export interface SceneSummary {
  scene_id: string
  event: string
  disaster_type: string | null
  disaster_type_source: 'config'
  xbd_split: string
  split: Split
  n_buildings: number
  ground_truth_counts: Record<DamageClass, number>
  quality: SceneQuality
  warnings: string[]
}

export interface SceneList {
  items: SceneSummary[]
  total: number
  page: number
  page_size: number
  split_design: string
  split_design_label: string
  default_split: 'test'
  facets: Record<'event' | 'split' | 'disaster_type', Record<string, number>>
}

export interface ModelInfo {
  name: string
  label: string
  selectable: boolean
  optional: boolean
  default: boolean
  trained: boolean
  design: string
  scheme: string
  confidence: string
  hyperparameters: Record<string, unknown> | null
  val_macro_f1: number | null
  test_macro_f1: number | null
  trained_at: string | null
  review_threshold: number | null
  fix: string | null
}

export interface ModelList {
  serving_design: string
  serving_design_label: string
  serving_scheme: string
  default_model: string
  models: ModelInfo[]
}

export interface ComparisonRow {
  model: string
  optional: boolean
  val_macro_f1: number
  test_macro_f1: number
  test_accuracy: number
  test_recall: Record<string, number | null>
  train_seconds: number
  inference_seconds_per_1000: number
}

export interface Comparison {
  design: string
  design_label: string
  scheme: string
  scheme_label: string
  classes: string[]
  models: ComparisonRow[]
  split_note: string
  experiment: boolean
  provenance: string
}
