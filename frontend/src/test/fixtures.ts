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
