import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AppShell } from '../components/AppShell'
import { comparison, missingStatus, models, readyStatus } from '../test/fixtures'
import { health, mockApi, renderAt } from '../test/utils'
import { ReadinessPage } from './ReadinessPage'

afterEach(() => vi.unstubAllGlobals())

const page = () => renderAt(<AppShell><ReadinessPage /></AppShell>)

describe('Readiness board', () => {
  it('shows tiles and headline numbers exactly as the API reports them', async () => {
    mockApi({
      '/api/health': { body: health }, '/api/dataset/status': { body: readyStatus }, '/api/models': { body: models },
      '/api/results/comparison': (url) => ({ body: url.searchParams.get('design') === 'cross_event' ? comparison('cross_event', 0.111, 0.222) : comparison('within_event', 0.876, 0.5) }),
    })
    page()
    const cache = (await screen.findByText('Scene cache')).closest('section')!
    expect(within(cache).getByText('345')).toBeInTheDocument()
    expect(within(cache).getByText('✓ READY')).toBeInTheDocument()
    expect(await screen.findByText('0.876')).toBeInTheDocument() // within-event KNN, from the API
    expect(screen.getByText('0.500')).toBeInTheDocument() // within-event Decision Tree
    expect(screen.getByText('0.111')).toBeInTheDocument() // cross-event KNN
    expect(screen.getByText('0.222')).toBeInTheDocument() // cross-event Decision Tree
    expect(screen.getAllByText('macro-F1 · K-Nearest Neighbours (default)')).toHaveLength(2)
    expect(screen.getAllByText('macro-F1 · Decision Tree')).toHaveLength(2)
    expect(screen.getByText('Damage classification on building footprints derived from xBD annotations.')).toBeInTheDocument()
  })

  it('the headline depends only on the served models: a Logistic Regression row is never shown', async () => {
    const withLr = (design: string, knn: number, tree: number) => {
      const c = comparison(design, knn, tree)
      return { ...c, models: [...c.models, { ...c.models[0], model: 'logistic_regression', test_macro_f1: 0.999 }] }
    }
    mockApi({
      '/api/health': { body: health }, '/api/dataset/status': { body: readyStatus }, '/api/models': { body: models },
      '/api/results/comparison': (url) => ({ body: url.searchParams.get('design') === 'cross_event' ? withLr('cross_event', 0.271, 0.294) : withLr('within_event', 0.52, 0.47) }),
    })
    page()
    const headline = (await screen.findByText('How well does it work?')).closest('section')!
    expect(await within(headline).findByText('0.520')).toBeInTheDocument()
    for (const v of ['0.470', '0.271', '0.294']) expect(within(headline).getByText(v)).toBeInTheDocument()
    expect(within(headline).queryByText('0.999')).not.toBeInTheDocument()
    expect(screen.queryByText(/logistic/i)).not.toBeInTheDocument()
  })

  it('marks untrained models and shows the command that fixes them', async () => {
    mockApi({ '/api/health': { body: health }, '/api/dataset/status': { body: readyStatus }, '/api/models': { body: models },
      '/api/results/comparison': { body: comparison('within_event', 0.5, 0.5) } })
    page()
    const table = (await screen.findByText('Models served to the Triage Desk')).closest('section')!
    expect(await within(table).findByText('NOT TRAINED')).toBeInTheDocument()
    expect(within(table).getByText('DEFAULT')).toBeInTheDocument()
  })

  it('shows PARTIAL review thresholds with the reason and "not available" for the missing model', async () => {
    const partial = { ...readyStatus, tiles: { ...readyStatus.tiles,
      review_thresholds: { ready: false, state: 'partial' as const, reason: 'usable for knn; missing for decision_tree',
        detail: { rule: 'lowest t', thresholds: { knn: 0.5333, decision_tree: null } }, fix: 'python scripts/calibrate_review.py' } } }
    mockApi({ '/api/health': { body: health }, '/api/dataset/status': { body: partial }, '/api/models': { body: models },
      '/api/results/comparison': { body: comparison('within_event', 0.5, 0.5) } })
    page()
    const tile = (await screen.findByText('Review thresholds')).closest('section')!
    expect(within(tile).getByText('◐ PARTIAL')).toBeInTheDocument()
    expect(within(tile).getByText('0.53')).toBeInTheDocument()
    expect(within(tile).getByText('not available')).toBeInTheDocument()
    expect(within(tile).getByText('usable for knn; missing for decision_tree')).toBeInTheDocument()
    expect(within(tile).getByText('$ python scripts/calibrate_review.py')).toBeInTheDocument()
    expect(within(tile).queryByText('✓ READY')).not.toBeInTheDocument()
  })

  it('review thresholds are READY with only the served models (KNN and Decision Tree), and the tile lists exactly those', async () => {
    const status = { ...readyStatus, tiles: { ...readyStatus.tiles,
      review_thresholds: { ready: true, state: 'ready' as const,
        reason: 'usable for every required served model (knn, decision_tree)',
        detail: { rule: 'lowest t', thresholds: { knn: 0.5333, decision_tree: 0.5257 } }, fix: null } } }
    mockApi({ '/api/health': { body: health }, '/api/dataset/status': { body: status }, '/api/models': { body: models },
      '/api/results/comparison': { body: comparison('within_event', 0.5, 0.5) } })
    page()
    const tile = (await screen.findByText('Review thresholds')).closest('section')!
    expect(within(tile).getByText('✓ READY')).toBeInTheDocument()
    expect(within(tile).getAllByText('0.53')).toHaveLength(2) // 0.5333 and 0.5257, both shown to 2 decimals
    expect(within(tile).getByText('usable for every required served model (knn, decision_tree)')).toBeInTheDocument()
    expect(within(tile).queryByText(/logistic_regression|random_forest/)).not.toBeInTheDocument()
  })

  it('models tile is READY when the served models (KNN and Decision Tree) are trained', async () => {
    const status = { ...readyStatus, tiles: { ...readyStatus.tiles,
      models: { ready: true, state: 'ready' as const, reason: 'all required served models are trained',
        detail: { trained: { knn: true, decision_tree: true } }, fix: null } } }
    mockApi({ '/api/health': { body: health }, '/api/dataset/status': { body: status }, '/api/models': { body: models },
      '/api/results/comparison': { body: comparison('within_event', 0.5, 0.5) } })
    page()
    const tile = (await screen.findByText('Models')).closest('section')!
    expect(within(tile).getByText('✓ READY')).toBeInTheDocument()
    expect(within(tile).getByText('all required served models are trained')).toBeInTheDocument()
    expect(within(tile).queryByText(/logistic_regression|random_forest|optional/)).not.toBeInTheDocument()
  })

  it('models table shows test macro-F1, confidence meaning and review threshold, but no validation column', async () => {
    mockApi({ '/api/health': { body: health }, '/api/dataset/status': { body: readyStatus }, '/api/models': { body: models },
      '/api/results/comparison': { body: comparison('within_event', 0.5, 0.5) } })
    page()
    const table = (await screen.findByText('Models served to the Triage Desk')).closest('section')!
    expect(await within(table).findByText('0.622')).toBeInTheDocument() // test macro-F1 from /api/models
    expect(within(table).getByText('vote fraction of neighbours')).toBeInTheDocument()
    expect(within(table).getByText('0.44')).toBeInTheDocument()
    // val_macro_f1 is still in the API response (0.611) but is no longer displayed here
    expect(within(table).queryByText('0.611')).not.toBeInTheDocument()
    expect(within(table).queryByText('VALIDATION')).not.toBeInTheDocument()
  })

  it('dataset-missing mode: every missing tile says NOT READY with its fix command', async () => {
    mockApi({
      '/api/health': { body: health }, '/api/dataset/status': { body: missingStatus }, '/api/models': { body: models },
      '/api/results/comparison': { status: 404, body: { error: { code: 'results_not_found', message: 'No saved comparison.', fix: 'python scripts/train.py' } } },
    })
    page()
    expect(await screen.findAllByText('○ NOT READY')).toHaveLength(2)
    expect(screen.getByText('$ python scripts/inspect_dataset.py --alignment')).toBeInTheDocument()
    expect(await screen.findByText('Not available yet: within-event results')).toBeInTheDocument()
  })

  it('shows an error with the start command when the backend is unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network')))
    page()
    expect(await screen.findByText('Could not load readiness status')).toBeInTheDocument()
    expect(screen.getAllByText(/uvicorn app.main:app --port 8020/).length).toBeGreaterThan(0)
  })

  it('theme toggle switches to dark and back', async () => {
    mockApi({ '/api/health': { body: health }, '/api/dataset/status': { body: readyStatus }, '/api/models': { body: models },
      '/api/results/comparison': { body: comparison('within_event', 0.5, 0.5) } })
    page()
    const toggle = screen.getByRole('button', { name: /light/i })
    await userEvent.click(toggle)
    expect(document.documentElement.dataset.theme).toBe('dark')
    await userEvent.click(screen.getByRole('button', { name: /dark/i }))
    expect(document.documentElement.dataset.theme).toBe('light')
  })
})
