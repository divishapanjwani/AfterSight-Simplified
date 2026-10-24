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
    expect(await screen.findByText('0.876')).toBeInTheDocument() // within-event LR, from the API
    expect(screen.getByText('0.111')).toBeInTheDocument() // cross-event LR
    expect(screen.getByText('best: decision tree 0.222')).toBeInTheDocument()
    expect(screen.getByText('Damage classification on building footprints derived from xBD annotations.')).toBeInTheDocument()
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
      review_thresholds: { ready: false, state: 'partial' as const, reason: 'usable for logistic_regression; missing for knn',
        detail: { rule: 'lowest t', thresholds: { logistic_regression: 0.4836, knn: null } }, fix: 'python scripts/calibrate_review.py' } } }
    mockApi({ '/api/health': { body: health }, '/api/dataset/status': { body: partial }, '/api/models': { body: models },
      '/api/results/comparison': { body: comparison('within_event', 0.5, 0.5) } })
    page()
    const tile = (await screen.findByText('Review thresholds')).closest('section')!
    expect(within(tile).getByText('◐ PARTIAL')).toBeInTheDocument()
    expect(within(tile).getByText('0.48')).toBeInTheDocument()
    expect(within(tile).getByText('not available')).toBeInTheDocument()
    expect(within(tile).getByText('usable for logistic_regression; missing for knn')).toBeInTheDocument()
    expect(within(tile).getByText('$ python scripts/calibrate_review.py')).toBeInTheDocument()
    expect(within(tile).queryByText('✓ READY')).not.toBeInTheDocument()
  })

  it('review thresholds stay READY when only the optional model lacks one, and the tile says so', async () => {
    const status = { ...readyStatus, tiles: { ...readyStatus.tiles,
      review_thresholds: { ready: true, state: 'ready' as const,
        reason: 'usable for logistic_regression, knn, decision_tree; random_forest (optional) has no usable threshold',
        detail: { rule: 'lowest t', thresholds: { logistic_regression: 0.4836, knn: 0.5333, decision_tree: 0.5257, random_forest: null } }, fix: null } } }
    mockApi({ '/api/health': { body: health }, '/api/dataset/status': { body: status }, '/api/models': { body: models },
      '/api/results/comparison': { body: comparison('within_event', 0.5, 0.5) } })
    page()
    const tile = (await screen.findByText('Review thresholds')).closest('section')!
    expect(within(tile).getByText('✓ READY')).toBeInTheDocument()
    expect(within(tile).getByText('not available')).toBeInTheDocument()
    expect(within(tile).getByText(/random_forest \(optional\) has no usable threshold/)).toBeInTheDocument()
  })

  it('models tile stays READY when only an optional model is untrained, and says so', async () => {
    const status = { ...readyStatus, tiles: { ...readyStatus.tiles,
      models: { ready: true, state: 'ready' as const, reason: 'all required served models are trained; random_forest (optional) not trained',
        detail: { trained: { logistic_regression: true, knn: true, random_forest: false } }, fix: 'python scripts/train.py' } } }
    mockApi({ '/api/health': { body: health }, '/api/dataset/status': { body: status }, '/api/models': { body: models },
      '/api/results/comparison': { body: comparison('within_event', 0.5, 0.5) } })
    page()
    const tile = (await screen.findByText('Models')).closest('section')!
    expect(within(tile).getByText('✓ READY')).toBeInTheDocument()
    expect(within(tile).getByText(/random_forest \(optional\) not trained/)).toBeInTheDocument()
  })

  it('models table shows test macro-F1, confidence meaning and review threshold, but no validation column', async () => {
    mockApi({ '/api/health': { body: health }, '/api/dataset/status': { body: readyStatus }, '/api/models': { body: models },
      '/api/results/comparison': { body: comparison('within_event', 0.5, 0.5) } })
    page()
    const table = (await screen.findByText('Models served to the Triage Desk')).closest('section')!
    expect(await within(table).findByText('0.622')).toBeInTheDocument() // test macro-F1 from /api/models
    expect(within(table).getByText('probability (not calibrated)')).toBeInTheDocument()
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
