import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { comparison, models } from '../test/fixtures'
import { mockApi, renderAt } from '../test/utils'
import { ClusterAtlasPage } from './ClusterAtlasPage'
import { FeatureLabPage } from './FeatureLabPage'
import { ModelArenaPage } from './ModelArenaPage'
import { ReportsPage } from './ReportsPage'
import { ReviewQueuePage } from './ReviewQueuePage'

afterEach(() => vi.unstubAllGlobals())

const cm = { labels: ['no-damage', 'destroyed'], rows_true_cols_pred: [[8, 2], [1, 9]] }
const metrics = { n: 20, accuracy: 0.85, macro_f1: 0.851, weighted_f1: 0.85, per_class: { destroyed: { precision: 1, recall: 0.9, f1: 0.9, support: 10 } }, confusion_matrix: cm, notes: [] }

describe('Model Arena', () => {
  it('tells the within- vs cross-event story with API numbers, and separates experiments', async () => {
    mockApi({
      '/api/results/comparison': (url) => ({ body: url.searchParams.get('design') === 'cross_event' ? comparison('cross_event', 0.291, 0.301) : comparison('within_event', 0.532, 0.481) }),
      '/api/models': (url) => url.pathname.endsWith('/metrics')
        ? { body: { model: 'logistic_regression', hyperparameters: { C: 10 }, metrics: { val: metrics, test: metrics, test_per_event: { 'ev-a': metrics } } } }
        : { body: { root: { id: 0, depth: 0, samples: 50, class_fractions: {}, majority: 'destroyed', leaf: true }, n_nodes: 1, max_depth: 0, n_leaves: 1, classes: [], view_max_depth: 3 } },
      '/api/results/confounding': { body: { design_label: 'within-event generalisation', target: 'event', test_metrics: { ...metrics, accuracy: 0.829, confusion_matrix: { labels: ['ev-a', 'ev-b'], rows_true_cols_pred: [[5, 1], [2, 4]] } }, majority_class: 'ev-a', majority_class_accuracy: 0.252, chance_accuracy_uniform: 0.2 } },
      '/api/results/experiments': { body: { label: 'Phase 3b experiment', section: 'Experiments', selectable: false, note: 'Separate models for analysis only.', variants: { change_only: '17 features' },
        rows: [{ variant: 'change_only', n_features: 17, design: 'within_event', model: 'logistic_regression', val_macro_f1: 0.4, test_macro_f1: 0.45, test_recall_major: 0.5, test_recall_destroyed: 0.76, event_prediction_accuracy_within_event: 0.632 }] } },
    })
    renderAt(<ModelArenaPage />, '/arena')
    expect(await screen.findAllByText('0.53 → 0.29')).not.toHaveLength(0)
    expect(screen.getByText('NOT SELECTABLE IN THE TRIAGE DESK')).toBeInTheDocument()
    expect(await screen.findByText('0.829')).toBeInTheDocument() // event-prediction accuracy
    expect(screen.getByText('0.252')).toBeInTheDocument()
    expect((await screen.findAllByText('0.632')).length).toBeGreaterThan(0) // Phase 3b event accuracy
    expect(screen.getByText('SEPARATE EXPERIMENT · 4-CLASS STAYS PRIMARY')).toBeInTheDocument()
  })
})

describe('Review Queue', () => {
  it('frames flags neutrally and reconciles the totals', async () => {
    const calls = mockApi({
      '/api/models': { body: models },
      '/api/review-queue': { body: { model: 'logistic_regression', split: 'test', framing: 'Suggested for human review — low confidence or model disagreement, not confirmed errors.',
        threshold: 0.48, threshold_rule: 'r', target_accuracy: 0.7, total_buildings: 100, total_flagged: 70, flagged_low_confidence: 40, flagged_disagreement_only: 30, offset: 0, limit: 25,
        items: [{ building_uid: 'ev-a_00000001_b0003', scene_id: 'ev-a_00000001', event: 'ev-a', predicted: 'minor-damage', true_label: 'no-damage', confidence: 0.31,
          reasons: ['low confidence (0.31 < 0.48)'], other_models: {} }] } },
    })
    renderAt(<ReviewQueuePage />, '/review')
    expect(await screen.findByText(/not confirmed errors/)).toBeInTheDocument()
    expect(screen.getByText(/test buildings/).closest('p')).toHaveTextContent('70 of 100 test buildings (70%) are suggested for human review')
    expect(screen.getByText(/low confidence ·/).closest('p')).toHaveTextContent('40 low confidence · 30 confident, but another model disagrees')
    expect(screen.getByRole('link', { name: 'Open →' })).toHaveAttribute('href', '/triage/ev-a_00000001?model=logistic_regression&run=1&b=ev-a_00000001_b0003')
    expect(screen.getByText(/test buildings/).closest('p')).toHaveTextContent('at least 70% correct')
    expect(screen.queryByText(/error rate/i)).not.toBeInTheDocument()
    expect(calls.find((u) => u.pathname === '/api/review-queue')!.searchParams.get('split')).toBe('test')
  })
})

describe('Review Queue without a usable threshold', () => {
  it('says why there are no low-confidence flags and never quotes a validation threshold', async () => {
    mockApi({
      '/api/models': { body: models },
      '/api/review-queue': { body: { model: 'logistic_regression', split: 'test', framing: 'Suggested for human review — low confidence or model disagreement, not confirmed errors.',
        threshold: null, threshold_rule: null, target_accuracy: null,
        threshold_note: 'calibration found no threshold for logistic_regression that reaches 70% accuracy on validation',
        total_buildings: 100, total_flagged: 30, flagged_low_confidence: 0, flagged_disagreement_only: 30, offset: 0, limit: 25,
        items: [{ building_uid: 'ev-a_00000001_b0003', scene_id: 'ev-a_00000001', event: 'ev-a', predicted: 'minor-damage', true_label: 'no-damage', confidence: 0.31,
          reasons: ['other models disagree: knn says no-damage'], other_models: { knn: 'no-damage' } }] } },
    })
    renderAt(<ReviewQueuePage />, '/review')
    const note = await screen.findByRole('note')
    expect(note).toHaveTextContent('No low-confidence flags for this model.')
    expect(note).toHaveTextContent('calibration found no threshold for logistic_regression that reaches 70% accuracy on validation')
    expect(note).toHaveTextContent('Only model-disagreement flags are listed.')
    const summary = screen.getByText(/test buildings/).closest('p')!
    expect(summary).toHaveTextContent('30 of 100 test buildings (30%) are suggested for human review with this model: another model disagrees with it.')
    expect(summary).not.toHaveTextContent('chosen on validation')
    expect(screen.getByText(/low confidence ·/).closest('p')).toHaveTextContent('0 low confidence · 30 confident, but another model disagrees')
  })
})

describe('Cluster Atlas', () => {
  it('is labelled analysis-only and refetches when k changes', async () => {
    const body = (k: number) => ({ design_label: 'w', classes: ['no-damage', 'destroyed'], k_values: [2, 3], chosen_k_by_silhouette: 2, elbow_k_heuristic: 3,
      pca_explained_variance: [0.32, 0.24], pca_note: '2-D PCA projection — for visualisation only', interpretation_note: 'Clusters are never used as damage predictions.',
      k, inertia: 10, silhouette_sampled: 0.25, adjusted_rand_index: 0.041, cluster_sizes: Array.from({ length: k }, () => 5),
      contingency_rows_cluster_cols_class: Array.from({ length: k }, () => [3, 2]), pca_points: [{ x: 0, y: 0, cluster: 0, label: 'no-damage' }, { x: 1, y: 1, cluster: 1, label: 'destroyed' }],
      curves: { 2: { inertia: 10, silhouette: 0.25 }, 3: { inertia: 8, silhouette: 0.2 } }, never_predictions: true })
    const calls = mockApi({ '/api/clusters': (url) => ({ body: body(Number(url.searchParams.get('k'))) }) })
    renderAt(<ClusterAtlasPage />, '/clusters?k=2')
    expect(await screen.findByText('ANALYSIS ONLY · CLUSTERS ARE NEVER PREDICTIONS')).toBeInTheDocument()
    expect(await screen.findByText('0.041')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('radio', { name: 'k=3' }))
    expect(await screen.findByText('cluster 2', { selector: 'span' })).toBeInTheDocument()
    expect(calls.at(-1)!.searchParams.get('k')).toBe('3')
  })
})

describe('Feature Lab', () => {
  it('lists features by separation and shows the training-split box plot', async () => {
    const stats = { n: 5, mean: 1, p5: 0, q1: 0.5, median: 1, q3: 1.5, p95: 2 }
    mockApi({
      '/api/features/summary': { body: { split: 'train', n: 1234, separation_score: 'eta', features: [
        { feature: 'glcm_homogeneity_post', group: 'texture', separation_eta2: 0.185, per_class: { 'no-damage': stats, 'minor-damage': stats, 'major-damage': stats, destroyed: stats } },
        { feature: 'area_px', group: 'shape', separation_eta2: 0.01, per_class: { 'no-damage': stats, 'minor-damage': stats, 'major-damage': stats, destroyed: stats } },
      ] } },
      '/api/features/scatter': { body: { split: 'train', x: 'glcm_homogeneity_post', y: 'area_px', n: 2, points: [{ x: 0.5, y: 100, label: 'no-damage' }, { x: 0.7, y: 900, label: 'destroyed' }] } },
    })
    renderAt(<FeatureLabPage />, '/features')
    expect(await screen.findByText('η² = 0.185')).toBeInTheDocument()
    expect(screen.getByText('n = 1,234 buildings')).toBeInTheDocument()
    expect(screen.getByText('TRAINING SPLIT')).toBeInTheDocument()
    await userEvent.click(within(screen.getByRole('list', { name: 'Features' })).getByText('Footprint area (px)'))
    expect(await screen.findByText('η² = 0.010')).toBeInTheDocument()
  })
})

describe('Reports', () => {
  it('lists saved analyses with split chips and export links', async () => {
    mockApi({ '/api/analyses': { body: { total: 1, items: [{ analysis_id: 'a1', scene_id: 'ev-a_00000001', event: 'ev-a', created_at: '2026-10-22T13:00:39+00:00',
      model: 'knn', model_label: 'K-Nearest Neighbours', split: 'train', warning: 'This scene was used for training. Accuracy here is optimistic.',
      n_buildings: 9, n_review: 2, accuracy: 0.5, macro_f1: 0.41, csv_url: '/api/analysis/a1/export.csv', pdf_url: '/api/analysis/a1/report.pdf' }] } } })
    renderAt(<ReportsPage />, '/reports')
    expect(await screen.findByText('ev-a_00000001')).toBeInTheDocument()
    expect(screen.getByText('TRAINING SCENE')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'PDF' })).toHaveAttribute('href', '/api/analysis/a1/report.pdf')
  })

  it('empty state points to the Triage Desk', async () => {
    mockApi({ '/api/analyses': { body: { total: 0, items: [] } } })
    renderAt(<ReportsPage />, '/reports')
    expect(await screen.findByText('No analyses yet.')).toBeInTheDocument()
  })
})
