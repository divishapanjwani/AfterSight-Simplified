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
    models: { ready: true, detail: { trained: { knn: true, decision_tree: true } }, fix: null },
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
  default_model: 'knn',
  models: [
    { name: 'knn', label: 'K-Nearest Neighbours', selectable: true, optional: false, default: true, trained: true,
      design: 'within_event', scheme: 'four_class', confidence: 'vote fraction of neighbours', hyperparameters: { n_neighbors: 31 },
      val_macro_f1: 0.611, test_macro_f1: 0.622, trained_at: 't', review_threshold: 0.44, fix: null },
    { name: 'decision_tree', label: 'Decision Tree', selectable: true, optional: false, default: false, trained: false,
      design: 'within_event', scheme: 'four_class', confidence: 'class fraction in the leaf', hyperparameters: null,
      val_macro_f1: null, test_macro_f1: null, trained_at: null, review_threshold: null, fix: 'python scripts/train.py' },
  ],
}

const row = (model: string, f1: number) => ({ model, optional: false, val_macro_f1: f1, test_macro_f1: f1, test_accuracy: f1,
  test_recall: {}, train_seconds: 1, inference_seconds_per_1000: 0.001 })

/** All five served models (KNN default, all trained), as the real API lists them. */
export const fiveModels: ModelList = {
  ...models,
  models: [
    { ...models.models[0] },
    { ...models.models[1], trained: true, fix: null, test_macro_f1: 0.47 },
    { ...models.models[1], name: 'random_forest', label: 'Random Forest', optional: true, trained: true, fix: null, confidence: 'mean of tree leaf fractions', test_macro_f1: 0.546 },
    { ...models.models[1], name: 'svm', label: 'SVM', optional: true, trained: true, fix: null, confidence: 'Platt-scaled SVM score (not calibrated on validation)', test_macro_f1: 0.514 },
    { ...models.models[1], name: 'soft_voting', label: 'Soft Voting (SVM + RF + KNN)', optional: true, trained: true, fix: null, confidence: 'mean of the SVM, Random Forest and KNN scores', test_macro_f1: 0.542 },
  ],
}

export const comparison = (design: string, knn: number, tree: number): Comparison => ({
  design, design_label: design === 'within_event' ? 'within-event generalisation' : 'cross-event generalisation',
  scheme: 'four_class', scheme_label: '4-class', classes: [], split_note: '', experiment: false, provenance: 'TEST SET',
  holdout_event: design === 'cross_event' ? 'ev-c' : null, test_events: design === 'cross_event' ? ['ev-c'] : ['ev-a', 'ev-b', 'ev-c'],
  models: [row('knn', knn), row('decision_tree', tree)],
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

export const analysisFor = (id: string, model = 'knn', split: SceneSummary['split'] = 'test', warning: string | null = null): AnalysisResult => ({
  analysis_id: 'abcdef0123456789', cached: false, created_at: 't', scene_id: id, event: 'ev-a', disaster_type: 'flooding',
  model: { name: model, label: ({ knn: 'K-Nearest Neighbours', random_forest: 'Random Forest', svm: 'SVM', soft_voting: 'Soft Voting (SVM + RF + KNN)' } as Record<string, string>)[model] ?? 'Decision Tree',
    design: 'within_event', design_label: 'within-event generalisation', scheme: 'four_class', trained_at: 't', hyperparameters: {}, test_macro_f1: 0.5, optional: false },
  confidence_kind: model === 'knn' ? 'vote fraction of neighbours' : model === 'random_forest' ? 'mean of tree leaf fractions' : 'class fraction in the leaf',
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
  decision_tree: { ...base, building_uid: 'x', kind: 'decision_path', detail: { prediction: 'destroyed', leaf_samples: 3677,
    leaf_class_fractions: { destroyed: 0.6, 'no-damage': 0.4 },
    steps: [{ feature: 'glcm_homogeneity_delta', value_raw: -0.022, op: '>', threshold_raw: -0.146 }] } },
  random_forest: { ...base, building_uid: 'x', kind: 'none', detail: { note: 'No per-building explanation for this model.' } },
  svm: { ...base, building_uid: 'x', kind: 'none', detail: { note: 'No per-building explanation for this model.' } },
  soft_voting: { ...base, building_uid: 'x', kind: 'none', detail: { note: 'No per-building explanation for this model.' } },
  knn: { ...base, building_uid: 'x', kind: 'nearest_neighbours', detail: { k: 2, weights: 'uniform', neighbours: [
    { sample_id: 'ev-b_00000001_b0003', label: 'destroyed', distance: 1.5, split: 'train', crop_urls: { pre: '/p', post: '/q' } },
    { sample_id: 'ev-b_00000002_b0004', label: 'minor-damage', distance: 1.9, split: 'train', crop_urls: { pre: '/p', post: '/q' } },
  ] } },
}

// ---------------------------------------------------------------- model comparison (test values, not xBD)
import type { ComparedModel, ModelComparison } from '../types/api'

const LABELS4 = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
const cmp = (model: string, label: string, kind: ComparedModel['kind'], f1: number, extra: Partial<ComparedModel> = {}): ComparedModel => ({
  model, label, kind, source: 's', ...extra,
  metrics: { n: 10, accuracy: f1 + 0.1, macro_precision: f1 + 0.01, macro_recall: f1 - 0.01, macro_f1: f1,
    per_class: Object.fromEntries(LABELS4.map((c, i) => [c, { precision: 0.5 + i / 100, recall: 0.4, f1: 0.45, support: 10 + i }])),
    confusion_matrix: { labels: LABELS4, rows_true_cols_pred: [[5, 1, 0, 0], [1, 4, 1, 0], [0, 1, 3, 1], [0, 0, 1, 6]] } },
})

export const modelComparison: ModelComparison = {
  label: 'Model comparison', created_at: 't', seed: 42, grid_search: {
    decision_tree: { label: 'Decision Tree', grid: { max_depth: [4, 8], min_samples_leaf: [20, 100] }, fixed: {}, cv: 'GroupKFold(n_splits=3) grouped by scene',
      trained_on: 'full within-event training split', best_params: { max_depth: 8, min_samples_leaf: 100 }, best_cv_macro_f1: 0.4811,
      test_macro_f1_default: 0.46, test_macro_f1_optimised: 0.472, difference: 0.012 },
  },
  notes: ['Test values.'],
  designs: {
    within_event: { design_label: 'within-event generalisation', classes: LABELS4, n_test: 10,
      svm_sample: { buildings: 10128, scenes: 168, of_training_buildings: 49929, seed: 42 },
      models: [cmp('knn', 'KNN', 'default', 0.51), cmp('decision_tree', 'Decision Tree', 'default', 0.46), cmp('random_forest', 'Random Forest', 'default', 0.55),
        cmp('svm', 'SVM', 'default', 0.505), cmp('soft_voting', 'Soft Voting (SVM + RF + KNN)', 'ensemble', 0.561, { members: ['svm', 'random_forest', 'knn'], implementation: 'VotingClassifier' }),
        cmp('decision_tree_optimised', 'Decision Tree (optimised)', 'optimised', 0.472)] },
    cross_event: { design_label: 'cross-event generalisation', classes: LABELS4, n_test: 10,
      svm_sample: { buildings: 10061, scenes: 175, of_training_buildings: 42425, seed: 42 },
      models: [cmp('knn', 'KNN', 'default', 0.27), cmp('decision_tree', 'Decision Tree', 'default', 0.29), cmp('random_forest', 'Random Forest', 'default', 0.291),
        cmp('svm', 'SVM', 'default', 0.274), cmp('soft_voting', 'Soft Voting (SVM + RF + KNN)', 'ensemble', 0.282, { members: ['svm', 'random_forest', 'knn'], implementation: 'VotingClassifier' })] },
  },
}
