/** Types mirroring backend/app/schemas/api.py. Only fields the UI uses are listed. */

export type Split = 'train' | 'val' | 'test' | 'not_in_manifest'
export type DamageClass = 'no-damage' | 'minor-damage' | 'major-damage' | 'destroyed'

export interface Health {
  status: string
  version: string
  scope: string
}

export type TileState = 'ready' | 'partial' | 'not_ready'

export interface StatusTile {
  ready: boolean
  /** Three-way state for tiles that can be partly ready (models, review thresholds); absent on the others. */
  state?: TileState | null
  /** One line naming the affected models. */
  reason?: string | null
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
  holdout_event: string | null
  test_events: string[]
  experiment: boolean
  provenance: string
}

export type Ring = [number, number][]
export type PolygonParts = Ring[][]

export interface Provenance {
  split: Split
  design: string
  design_label: string
  warning: string | null
  note: string
}

export interface GroundTruthBuilding {
  building_uid: string
  true_label: DamageClass
  area_px: number
  label_purity: number
  mixed_label: boolean
  touches_border: boolean
  polygon: PolygonParts
}

export interface SceneDetail extends SceneSummary {
  width: number
  height: number
  provenance: Provenance
  n_dropped: number
  dropped_reasons: Record<string, number>
  footprint_source: string
  image_urls: Record<string, string>
  buildings: GroundTruthBuilding[]
}

export interface ClassificationMetrics {
  n: number
  accuracy: number | null
  macro_f1: number | null
  weighted_f1: number | null
  per_class: Record<string, { precision: number | null; recall: number | null; f1: number | null; support: number }>
  confusion_matrix: { labels: string[]; rows_true_cols_pred: number[][] }
  notes: string[]
}

export interface BuildingPrediction {
  building_uid: string
  true_label: DamageClass
  predicted: DamageClass
  correct: boolean
  confidence: number
  scores: Record<string, number>
  needs_review: boolean
  review_reasons: string[]
  area_px: number
  mixed_label: boolean
  touches_border: boolean
  polygon: PolygonParts
}

export interface AnalysisModel {
  name: string
  label: string
  design: string
  design_label: string
  scheme: string
  trained_at: string
  hyperparameters: Record<string, unknown>
  test_macro_f1: number
  optional: boolean
}

export interface AnalysisResult {
  analysis_id: string
  cached: boolean
  created_at: string
  scene_id: string
  event: string
  disaster_type: string | null
  model: AnalysisModel
  confidence_kind: string
  review_threshold: number | null
  review_rule: string | null
  provenance: Provenance
  counts: { ground_truth: Record<DamageClass, number>; predicted: Record<DamageClass, number> }
  scene_metrics: ClassificationMetrics | null
  seconds: number
  buildings: BuildingPrediction[]
}

export interface LrContribution {
  feature: string
  value_raw: number
  standardised: number
  coefficient: number
  contribution: number
}

export interface Neighbour {
  sample_id: string
  label: DamageClass
  distance: number
  split: 'train'
  crop_urls: { pre: string; post: string }
}

export interface Explanation {
  analysis_id: string
  building_uid: string
  model: string
  kind: 'lr_contributions' | 'nearest_neighbours'
  predicted: DamageClass
  true_label: DamageClass
  confidence: number
  confidence_kind: string
  detail: {
    // lr_contributions
    predicted_class?: string
    intercept?: number
    score?: number
    top_features?: LrContribution[]
    note?: string
    // nearest_neighbours
    k?: number
    weights?: string
    neighbours?: Neighbour[]
  }
  global_top_features: { feature: string; mean: number; std: number }[]
  crop_urls: { pre: string; post: string }
}

// ---------------------------------------------------------------- arena / results
export interface ModelMetrics {
  model: string
  label: string
  design: string
  design_label: string
  scheme: string
  scheme_label: string
  classes: string[]
  optional: boolean
  hyperparameters: Record<string, unknown>
  train_seconds: number
  inference_seconds_per_1000_test: number
  confidence: string
  metrics: { val: ClassificationMetrics; test: ClassificationMetrics; test_per_event: Record<string, ClassificationMetrics> }
  preprocessing: { log1p_features?: string[] }
}

export interface TreeNode {
  id: number
  depth: number
  samples: number
  class_fractions: Record<string, number>
  majority: string
  feature?: string
  threshold_raw?: number
  left?: TreeNode
  right?: TreeNode
  leaf?: boolean
  truncated?: boolean
}

export interface TreeView {
  n_nodes: number
  max_depth: number
  n_leaves: number
  classes: string[]
  view_max_depth: number
  root: TreeNode
}

export interface Confounding {
  design_label: string
  target: string
  test_metrics: ClassificationMetrics
  majority_class: string
  majority_class_accuracy: number
  chance_accuracy_uniform: number
}

export interface FeatureSummary {
  split: 'train'
  n: number
  separation_score: string
  features: { feature: string; group: string; separation_eta2: number;
    per_class: Record<string, { n: number; mean: number | null; p5: number | null; q1: number | null; median: number | null; q3: number | null; p95: number | null }> }[]
}

export interface Scatter {
  split: 'train'
  x: string
  y: string
  n: number
  points: { x: number; y: number; label: DamageClass }[]
}

/**
 * The fields of GET /clusters that the Cluster Atlas shows. The API also returns chosen_k_by_silhouette,
 * elbow_k_heuristic, inertia, curves, pca_points, pca_explained_variance and pca_note; since v1.3 they are not
 * displayed (see docs/API.md), so they are not typed here.
 */
export interface Clusters {
  design_label: string
  classes: string[]
  k_values: number[]
  interpretation_note: string
  k: number
  silhouette_sampled: number
  adjusted_rand_index: number
  cluster_sizes: number[]
  contingency_rows_cluster_cols_class: number[][]
  never_predictions: true
}

export interface ReviewItem {
  building_uid: string
  scene_id: string
  event: string
  predicted: DamageClass
  true_label: DamageClass
  confidence: number
  reasons: string[]
  other_models: Record<string, string>
}

export interface ReviewQueue {
  model: string
  split: Split
  framing: string
  threshold: number | null
  threshold_rule: string | null
  target_accuracy: number | null
  /** Why this model has no usable review threshold (null when it has one). */
  threshold_note?: string | null
  total_buildings: number
  total_flagged: number
  flagged_low_confidence: number
  flagged_disagreement_only: number
  items: ReviewItem[]
  offset: number
  limit: number
}

export interface AnalysisSummary {
  analysis_id: string
  scene_id: string
  event: string
  created_at: string
  model: string
  model_label: string
  split: Split
  warning: string | null
  n_buildings: number
  n_review: number
  accuracy: number | null
  macro_f1: number | null
  csv_url: string
  pdf_url: string
}
