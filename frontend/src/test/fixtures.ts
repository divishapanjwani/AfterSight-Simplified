/** Hand-written API responses for UI tests (shapes copied from the real API; values are test values, not xBD). */
import type { Comparison, DatasetStatus, ModelList, SceneList, SceneSummary } from '../types/api'

export const readyStatus: DatasetStatus = {
  scope: 'Damage classification on building footprints derived from xBD annotations.',
  source: 'xbd',
  demo_mode: false,
  tiles: {
    scene_cache: { ready: true, detail: { scenes: 12, events: ['ev-a', 'ev-b'], buildings: 345 }, fix: null },
    features: { ready: true, detail: { buildings: 340, features: 37, schema_version: 1 }, fix: null },
    splits: { ready: true, detail: { within_event: 'aaaa1111', cross_event: 'bbbb2222' }, fix: null },
    models: { ready: true, detail: { trained: { logistic_regression: true, random_forest: false } }, fix: 'python scripts/train.py' },
    review_thresholds: { ready: true, detail: { rule: 'lowest t' }, fix: null },
    last_analysis: { ready: false, detail: {}, fix: 'POST /api/analysis' },
  },
}

export const missingStatus: DatasetStatus = {
  ...readyStatus,
  tiles: {
    scene_cache: { ready: false, detail: { reason: 'The scene cache has not been built yet.' }, fix: 'python scripts/inspect_dataset.py --alignment' },
    features: { ready: false, detail: { reason: 'Feature table not found' }, fix: 'python scripts/build_features.py' },
  },
}

export const models: ModelList = {
  serving_design: 'within_event', serving_design_label: 'within-event generalisation', serving_scheme: 'four_class',
  default_model: 'logistic_regression',
  models: [
    { name: 'logistic_regression', label: 'Logistic Regression', selectable: true, optional: false, default: true, trained: true,
      design: 'within_event', scheme: 'four_class', confidence: 'probability (not calibrated)', hyperparameters: { C: 10 },
      val_macro_f1: 0.611, test_macro_f1: 0.622, trained_at: 't', review_threshold: 0.44, fix: null },
    { name: 'random_forest', label: 'Random Forest (optional)', selectable: true, optional: true, default: false, trained: false,
      design: 'within_event', scheme: 'four_class', confidence: 'mean of tree leaf fractions', hyperparameters: null,
      val_macro_f1: null, test_macro_f1: null, trained_at: null, review_threshold: null, fix: 'python scripts/train.py' },
  ],
}

const row = (model: string, f1: number) => ({ model, optional: false, val_macro_f1: f1, test_macro_f1: f1, test_accuracy: f1,
  test_recall: {}, train_seconds: 1, inference_seconds_per_1000: 0.001 })

export const comparison = (design: string, lr: number, other: number): Comparison => ({
  design, design_label: design === 'within_event' ? 'within-event generalisation' : 'cross-event generalisation',
  scheme: 'four_class', scheme_label: '4-class', classes: [], split_note: '', experiment: false, provenance: 'TEST SET',
  models: [row('logistic_regression', lr), row('decision_tree', other)],
})

export const scene = (id: string, split: SceneSummary['split']): SceneSummary => ({
  scene_id: id, event: 'ev-a', disaster_type: 'flooding', disaster_type_source: 'config', xbd_split: 'train', split,
  n_buildings: 7, ground_truth_counts: { 'no-damage': 4, 'minor-damage': 1, 'major-damage': 0, destroyed: 2 },
  quality: { alignment_shift_px: 12.5, alignment_flag: true, n_mixed_label: 0, n_touching_border: 1 },
  warnings: ['Pre/post images may be misaligned (estimated shift 12.5 px).'],
})

export const sceneList = (split: SceneSummary['split'], ids: string[]): SceneList => ({
  items: ids.map((id) => scene(id, split)), total: ids.length, page: 1, page_size: 20,
  split_design: 'within_event', split_design_label: 'within-event generalisation', default_split: 'test',
  facets: { event: { 'ev-a': 9 }, split: { test: 3, val: 2, train: 4 }, disaster_type: { flooding: 9 } },
})

// ---------------------------------------------------------------- triage / case file
import type { AnalysisResult, BuildingPrediction, Explanation, SceneDetail } from '../types/api'

const square = (x: number): [number, number][][][] => [[[[x, 10], [x + 20, 10], [x + 20, 30], [x, 30], [x, 10]]]]

export const sceneDetail = (id: string, split: SceneSummary['split'], warning: string | null): SceneDetail => ({
  ...scene(id, split), n_buildings: 2, width: 1024, height: 1024,
  ground_truth_counts: { 'no-damage': 1, 'minor-damage': 0, 'major-damage': 0, destroyed: 1 },
  provenance: { split, design: 'within_event', design_label: 'within-event generalisation', warning, note: split === 'test' ? 'Held-out test scene: never used for training or tuning.' : 'Training scene.' },
  n_dropped: 0, dropped_reasons: {}, footprint_source: 'mask', image_urls: {},
  buildings: [
    { building_uid: `${id}_b0001`, true_label: 'no-damage', area_px: 400, label_purity: 1, mixed_label: false, touches_border: false, polygon: square(10) },
    { building_uid: `${id}_b0002`, true_label: 'destroyed', area_px: 400, label_purity: 1, mixed_label: true, touches_border: false, polygon: square(60) },
  ],
})

const prediction = (uid: string, true_label: BuildingPrediction['true_label'], predicted: BuildingPrediction['predicted'], confidence: number, x: number): BuildingPrediction => ({
  building_uid: uid, true_label, predicted, correct: true_label === predicted, confidence,
  scores: { 'no-damage': predicted === 'no-damage' ? confidence : 0.1, destroyed: predicted === 'destroyed' ? confidence : 0.1 },
  needs_review: confidence < 0.5, review_reasons: confidence < 0.5 ? [`confidence ${confidence.toFixed(2)} is below the review threshold 0.48 (chosen on validation)`] : [],
  area_px: 400, mixed_label: false, touches_border: false, polygon: square(x),
})

export const analysisFor = (id: string, model = 'logistic_regression', split: SceneSummary['split'] = 'test', warning: string | null = null): AnalysisResult => ({
  analysis_id: 'abcdef0123456789', cached: false, created_at: 't', scene_id: id, event: 'ev-a', disaster_type: 'flooding',
  model: { name: model, label: model === 'knn' ? 'K-Nearest Neighbours' : model === 'decision_tree' ? 'Decision Tree' : 'Logistic Regression',
    design: 'within_event', design_label: 'within-event generalisation', scheme: 'four_class', trained_at: 't', hyperparameters: {}, test_macro_f1: 0.5, optional: false },
  confidence_kind: model === 'knn' ? 'vote fraction of neighbours' : 'probability (not calibrated)',
  review_threshold: 0.48, review_rule: 'rule',
  provenance: { split, design: 'within_event', design_label: 'within-event generalisation', warning, note: 'note' },
  counts: { ground_truth: { 'no-damage': 1, 'minor-damage': 0, 'major-damage': 0, destroyed: 1 }, predicted: { 'no-damage': 0, 'minor-damage': 0, 'major-damage': 0, destroyed: 2 } },
  scene_metrics: { n: 2, accuracy: 0.5, macro_f1: 0.333, weighted_f1: 0.333, per_class: {}, confusion_matrix: { labels: [], rows_true_cols_pred: [] }, notes: [] },
  seconds: 0.004,
  buildings: [prediction(`${id}_b0001`, 'no-damage', 'destroyed', 0.41, 10), prediction(`${id}_b0002`, 'destroyed', 'destroyed', 0.93, 60)],
})

const base = { analysis_id: 'abcdef0123456789', model: 'x', predicted: 'destroyed' as const, true_label: 'destroyed' as const, confidence: 0.93,
  confidence_kind: 'k', global_top_features: [{ feature: 'gray_std_post', mean: 0.27, std: 0.01 }], crop_urls: { pre: '', post: '' } }

export const explanations: Record<string, Explanation> = {
  logistic_regression: { ...base, building_uid: 'x', kind: 'lr_contributions', detail: { intercept: -0.85, score: 2.44, top_features: [
    { feature: 'post_mean_b', value_raw: 155.9, standardised: 1.2, coefficient: -3.6, contribution: -4.37 },
    { feature: 'ring_dmean', value_raw: 4.2, standardised: 0.9, coefficient: 3.4, contribution: 3.07 },
  ] } },
  decision_tree: { ...base, building_uid: 'x', kind: 'decision_path', detail: { prediction: 'destroyed', leaf_samples: 3677,
    leaf_class_fractions: { destroyed: 0.6, 'no-damage': 0.4 },
    steps: [{ feature: 'glcm_homogeneity_delta', value_raw: -0.022, op: '>', threshold_raw: -0.146 }] } },
  knn: { ...base, building_uid: 'x', kind: 'nearest_neighbours', detail: { k: 2, weights: 'uniform', neighbours: [
    { sample_id: 'ev-b_00000001_b0003', label: 'destroyed', distance: 1.5, split: 'train', crop_urls: { pre: '/p', post: '/q' } },
    { sample_id: 'ev-b_00000002_b0004', label: 'minor-damage', distance: 1.9, split: 'train', crop_urls: { pre: '/p', post: '/q' } },
  ] } },
}
