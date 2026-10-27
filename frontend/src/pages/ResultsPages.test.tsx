import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ModelComparisonSection } from '../components/ModelComparisonSection'
import { comparison, modelComparison, models } from '../test/fixtures'
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
  const arenaRoutes = () => ({
    '/api/results/comparison': (url: URL) => ({ body: url.searchParams.get('design') === 'cross_event' ? comparison('cross_event', 0.291, 0.301) : comparison('within_event', 0.532, 0.481) }),
    '/api/models': (url: URL) => url.pathname === '/api/models' ? { body: models }
      : url.pathname.endsWith('/metrics')
        ? { body: { model: 'knn', hyperparameters: { n_neighbors: 31 }, metrics: { val: metrics, test: metrics, test_per_event: { 'ev-a': metrics } } } }
        : { body: { root: { id: 0, depth: 0, samples: 50, class_fractions: {}, majority: 'destroyed', leaf: true }, n_nodes: 1, max_depth: 0, n_leaves: 1, classes: [], view_max_depth: 3 } },
    '/api/results/model-comparison': { body: modelComparison },
  })

  it('tells the within- vs cross-event story with API numbers', async () => {
    mockApi(arenaRoutes())
    renderAt(<ModelArenaPage />, '/arena')
    expect(await screen.findAllByText('0.53 → 0.29')).not.toHaveLength(0)
    expect(screen.queryByText(/Phase 3b/i)).not.toBeInTheDocument()
  })

  it('has six cards in order, without the confounding check or the 3-class experiment, and never requests their results', async () => {
    const calls = mockApi(arenaRoutes())
    const { container } = renderAt(<ModelArenaPage />, '/arena')
    await screen.findByText('A. All models on the test sets')
    expect([...container.querySelectorAll('section.relative h2')].map((h) => h.textContent)).toEqual([
      'The same models on known vs unseen disasters', 'Model comparison (4-class)', 'Confusion matrices',
      'Per-event scores (within-event test)', 'Decision tree', 'Model Comparison & Ensemble Analysis'])
    for (const gone of ['Event-confounding check', '3-class experiment: no-damage / damaged / destroyed', 'SEPARATE EXPERIMENT · 4-CLASS STAYS PRIMARY']) {
      expect(screen.queryByText(gone)).not.toBeInTheDocument()
    }
    expect(screen.queryByRole('link', { name: 'Confounding' })).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: '3-class' })).not.toBeInTheDocument()
    expect(calls.some((u) => u.pathname === '/api/results/confounding')).toBe(false)
    expect(calls.some((u) => u.searchParams.get('scheme') === 'three_class')).toBe(false)
  })

  it('has no feature-set Experiments section and never requests its results', async () => {
    const calls = mockApi(arenaRoutes())
    renderAt(<ModelArenaPage />, '/arena')
    expect(await screen.findAllByText('0.53 → 0.29')).not.toHaveLength(0)
    for (const gone of ['EXPERIMENTS · FEATURE SETS', 'Experiments: confounding-reduced feature sets', 'NOT SELECTABLE IN THE TRIAGE DESK']) {
      expect(screen.queryByText(gone)).not.toBeInTheDocument()
    }
    expect(screen.queryByText(/Feature-set experiment/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/reduced that signature/)).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Experiments' })).not.toBeInTheDocument()
    expect(document.getElementById('experiments')).toBeNull()
    expect(calls.some((u) => u.pathname === '/api/results/experiments')).toBe(false)
  })

  it('shows KNN and the Decision Tree as served models (no research labels), never LR or Random Forest', async () => {
    const routes = arenaRoutes()
    // even if a saved Logistic Regression row reached the page, the Arena lists only the baseline and the served models
    const withLr = (url: URL) => {
      const body = routes['/api/results/comparison'](url).body
      return { body: { ...body, models: [...body.models, { ...body.models[0], model: 'logistic_regression', test_macro_f1: 0.999 }] } }
    }
    const calls = mockApi({ ...routes, '/api/results/comparison': withLr })
    renderAt(<ModelArenaPage />, '/arena')
    expect(await screen.findAllByText('0.48 → 0.30')).not.toHaveLength(0) // Decision Tree, within → cross
    expect(screen.getAllByText('Decision Tree').length).toBeGreaterThan(0)
    expect(screen.queryByText(/research/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/not served/i)).not.toBeInTheDocument()
    expect(screen.queryByText('Logistic Regression')).not.toBeInTheDocument()
    expect(screen.queryByText('0.999')).not.toBeInTheDocument()
    // Random Forest is not served in this fixture: it appears only inside the model-comparison card
    expect(screen.queryAllByText(/Random Forest/).every((el) => el.closest('#ensembles'))).toBe(true)
    const confusionModels = screen.getAllByRole('radiogroup', { name: 'Model' })[0]
    expect(within(confusionModels).getAllByRole('radio').map((r) => r.textContent)).toEqual(['KNN', 'Decision Tree'])
    expect(await within(confusionModels).findByRole('radio', { name: 'KNN' })).toHaveAttribute('aria-checked', 'true') // the default
    expect(calls.some((u) => u.pathname.includes('logistic_regression'))).toBe(false)
  })

})

describe('Review Queue', () => {
  it('frames flags neutrally and reconciles the totals', async () => {
    const calls = mockApi({
      '/api/models': { body: models },
      '/api/review-queue': { body: { model: 'knn', split: 'test', framing: 'Suggested for human review — low confidence or model disagreement, not confirmed errors.',
        threshold: 0.48, threshold_rule: 'r', target_accuracy: 0.7, total_buildings: 100, total_flagged: 70, flagged_low_confidence: 40, flagged_disagreement_only: 30, offset: 0, limit: 25,
        items: [{ building_uid: 'ev-a_00000001_b0003', scene_id: 'ev-a_00000001', event: 'ev-a', predicted: 'minor-damage', true_label: 'no-damage', confidence: 0.31,
          reasons: ['low confidence (0.31 < 0.48)'], other_models: {} }] } },
    })
    renderAt(<ReviewQueuePage />, '/review')
    expect(await screen.findByText(/not confirmed errors/)).toBeInTheDocument()
    expect(screen.getByText(/test buildings/).closest('p')).toHaveTextContent('70 of 100 test buildings (70%) are suggested for human review')
    expect(screen.getByText(/low confidence ·/).closest('p')).toHaveTextContent('40 low confidence · 30 confident, but another model disagrees')
    expect(screen.getByRole('link', { name: 'Open →' })).toHaveAttribute('href', '/triage/ev-a_00000001?model=knn&run=1&b=ev-a_00000001_b0003')
    expect(screen.getByText(/test buildings/).closest('p')).toHaveTextContent('at least 70% correct')
    expect(screen.queryByText(/error rate/i)).not.toBeInTheDocument()
    expect(calls.find((u) => u.pathname === '/api/review-queue')!.searchParams.get('split')).toBe('test')
    expect(calls.find((u) => u.pathname === '/api/review-queue')!.searchParams.get('model')).toBe('knn') // the default
    const picker = screen.getByRole('radiogroup', { name: 'Model' })
    expect(within(picker).getAllByRole('radio').map((r) => r.textContent)).toEqual(['K-Nearest Neighbours', 'Decision Tree'])
  })
})

describe('Review Queue without a usable threshold', () => {
  it('says why there are no low-confidence flags and never quotes a validation threshold', async () => {
    mockApi({
      '/api/models': { body: models },
      '/api/review-queue': { body: { model: 'knn', split: 'test', framing: 'Suggested for human review — low confidence or model disagreement, not confirmed errors.',
        threshold: null, threshold_rule: null, target_accuracy: null,
        threshold_note: 'calibration found no threshold for knn that reaches 70% accuracy on validation',
        total_buildings: 100, total_flagged: 30, flagged_low_confidence: 0, flagged_disagreement_only: 30, offset: 0, limit: 25,
        items: [{ building_uid: 'ev-a_00000001_b0003', scene_id: 'ev-a_00000001', event: 'ev-a', predicted: 'minor-damage', true_label: 'no-damage', confidence: 0.31,
          reasons: ['other models disagree: decision_tree says no-damage'], other_models: { decision_tree: 'no-damage' } }] } },
    })
    renderAt(<ReviewQueuePage />, '/review')
    const note = await screen.findByRole('note')
    expect(note).toHaveTextContent('No low-confidence flags for this model.')
    expect(note).toHaveTextContent('calibration found no threshold for knn that reaches 70% accuracy on validation')
    expect(note).toHaveTextContent('Only model-disagreement flags are listed.')
    const summary = screen.getByText(/test buildings/).closest('p')!
    expect(summary).toHaveTextContent('30 of 100 test buildings (30%) are suggested for human review with this model: another model disagrees with it.')
    expect(summary).not.toHaveTextContent('chosen on validation')
    const breakdown = screen.getByText(/low confidence ·/).closest('p')!
    expect(breakdown).toHaveTextContent('0 low confidence · 30 where another model disagrees (no threshold, so none are called confident).')
    expect(breakdown).not.toHaveTextContent('30 confident')
  })
})

describe('Review Queue paging', () => {
  it('a page past the end is not reported as an empty queue', async () => {
    mockApi({
      '/api/models': { body: models },
      '/api/review-queue': { body: { model: 'knn', split: 'test', framing: 'Suggested for human review — low confidence or model disagreement, not confirmed errors.',
        threshold: 0.48, threshold_rule: 'r', target_accuracy: 0.7, threshold_note: null, total_buildings: 100, total_flagged: 30,
        flagged_low_confidence: 20, flagged_disagreement_only: 10, offset: 2450, limit: 25, items: [] } },
    })
    renderAt(<ReviewQueuePage />, '/review?page=99')
    expect(await screen.findByText('Page 99 is past the end of the queue (2 pages).')).toBeInTheDocument()
    expect(screen.queryByText('Nothing is suggested for human review with this model.')).not.toBeInTheDocument()
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

  it('shows sizes, silhouette and ARI with their meaning, the contingency heatmap, and no PCA or elbow charts', async () => {
    mockApi({ '/api/clusters': { body: { design_label: 'w', classes: ['no-damage', 'destroyed'], k_values: [2, 3, 4], chosen_k_by_silhouette: 2, elbow_k_heuristic: 3,
      pca_explained_variance: [0.32, 0.24], pca_note: '2-D PCA projection — for visualisation only', interpretation_note: 'Clusters are never used as damage predictions.',
      k: 2, inertia: 10, silhouette_sampled: 0.253, adjusted_rand_index: 0.041, cluster_sizes: [7, 5], contingency_rows_cluster_cols_class: [[4, 3], [1, 4]],
      pca_points: [{ x: 0, y: 0, cluster: 0, label: 'no-damage' }], curves: { 2: { inertia: 10, silhouette: 0.25 } }, never_predictions: true } } })
    renderAt(<ClusterAtlasPage />, '/clusters?k=2')
    expect(await screen.findByText('0.253')).toBeInTheDocument()
    expect(screen.getByText('How well separated the clusters are, from −1 to 1.')).toBeInTheDocument()
    expect(screen.getByText(/0 = random, 1 = identical\. It is not prediction accuracy\./)).toBeInTheDocument()
    expect(screen.getByText(/A cluster ID is not a damage class/)).toBeInTheDocument()
    expect(screen.getByText(/Results exist only for the precomputed values/)).toHaveTextContent('k = 2, 3, 4')
    expect(screen.getByText('Which damage classes fall into each cluster')).toBeInTheDocument()
    expect(screen.getByText('ANALYSIS ONLY · CLUSTERS ARE NEVER PREDICTIONS')).toBeInTheDocument()
    for (const removed of [/principal component/i, /Elbow/i, /Inertia/i, /2-D projection/i]) {
      expect(screen.queryByText(removed)).not.toBeInTheDocument()
    }
  })

  it('missing artifact: says K-Means has not been run, with the command', async () => {
    mockApi({ '/api/clusters': { status: 404, body: { error: { code: 'results_not_found', message: 'K-Means has not been run.', fix: 'python scripts/run_kmeans.py' } } } })
    renderAt(<ClusterAtlasPage />, '/clusters')
    expect(await screen.findByText('Not available yet: the cluster analysis')).toBeInTheDocument()
    expect(screen.getByText('$ python scripts/run_kmeans.py')).toBeInTheDocument()
  })

  it('a k that was not precomputed says which k values exist', async () => {
    mockApi({ '/api/clusters': { status: 404, body: { error: { code: 'k_not_available', message: 'k=11 was not computed; available: [2, 3, 4].' } } } })
    renderAt(<ClusterAtlasPage />, '/clusters?k=11')
    expect(await screen.findByText('k=11 was not computed; available: [2, 3, 4].')).toBeInTheDocument()
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

describe('Model Comparison & Ensemble Analysis', () => {
  const routes = () => ({
    '/api/models': (url: URL) => url.pathname === '/api/models' ? { body: models }
      : url.pathname.endsWith('/metrics')
        ? { body: { model: 'knn', hyperparameters: {}, metrics: { val: metrics, test: metrics, test_per_event: { 'ev-a': metrics } } } }
        : { body: { root: { id: 0, depth: 0, samples: 50, class_fractions: {}, majority: 'destroyed', leaf: true }, n_nodes: 1, max_depth: 0, n_leaves: 1, classes: [], view_max_depth: 3 } },
    '/api/results/comparison': (url: URL) => {
      const c = comparison(url.searchParams.get('design') ?? 'within_event', 0.5, 0.4)
      return { body: { ...c, models: [{ ...c.models[0], model: 'baseline', test_macro_f1: 0.383 }, ...c.models] } }
    },
    '/api/results/model-comparison': { body: modelComparison },
  })

  it('shows the comparison table, GridSearchCV, confusion matrix, per-class metrics and the ensemble explanation from the API', async () => {
    mockApi(routes())
    renderAt(<ModelComparisonSection />)
    const table = (await screen.findByText('A. All models on the test sets')).closest('section')!
    expect(within(table).getByText('Soft Voting (SVM + RF + KNN)')).toBeInTheDocument()
    expect(within(table).getAllByText('parallel ensemble')).toHaveLength(1)
    expect(within(table).getByText('not evaluated')).toBeInTheDocument() // the optimised DT has no cross-event result
    expect(within(table).getByText('0.561')).toBeInTheDocument()
    const grid = screen.getByText('C. GridSearchCV (grouped by scene)').closest('section')!
    expect(grid).toHaveTextContent('max_depth=8 · min_samples_leaf=100')
    expect(grid).toHaveTextContent('0.460 → 0.472(+0.012)')
    expect(screen.getByRole('radio', { name: 'Soft Voting (SVM + RF + KNN)' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByRole('table', { name: 'Per-class metrics' })).toHaveTextContent('destroyed0.5300.4000.45013')
    const ens = screen.getByText('F. How the ensemble works').closest('section')!
    expect(ens).toHaveTextContent('Soft voting reaches macro-F1 0.561, higher than its best member (Random Forest, 0.550)')
    expect(ens).toHaveTextContent('Soft voting 0.282 vs best member Random Forest 0.291')
    expect(ens).toHaveTextContent('Random Forest is itself a parallel ensemble')
    await userEvent.click(within(screen.getByRole('radiogroup', { name: 'Bar chart design' })).getByRole('radio', { name: 'cross-event' }))
    expect(screen.getByText(/Optimised versions were tuned and tested within-event only/)).toBeInTheDocument()
  })

  it('the Arena has the new section and no threshold baseline anywhere', async () => {
    mockApi(routes())
    renderAt(<ModelArenaPage />, '/arena')
    expect(await screen.findByText('Model Comparison & Ensemble Analysis')).toBeInTheDocument()
    await screen.findByText('A. All models on the test sets')
    expect(screen.queryByText(/baseline/i)).not.toBeInTheDocument()
    expect(screen.queryByText('0.383')).not.toBeInTheDocument()
  })

  it('missing results show the command that builds them', async () => {
    mockApi({ '/api/results/model-comparison': { status: 404, body: { error: { code: 'results_not_found', message: 'The model comparison has not been run.', fix: 'python scripts/run_model_comparison.py' } } } })
    renderAt(<ModelComparisonSection />)
    expect(await screen.findByText('$ python scripts/run_model_comparison.py')).toBeInTheDocument()
  })
})
