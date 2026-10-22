import { fireEvent, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Route, Routes } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { analysisFor, explanations, models, sceneDetail } from '../test/fixtures'
import { mockApi, renderAt } from '../test/utils'
import { CasePage } from './CasePage'
import { TriageDeskPage } from './TriageDeskPage'

afterEach(() => vi.unstubAllGlobals())

const ID = 'ev-a_00000007'
const TRAIN_WARNING = 'This scene was used for training. Accuracy here is optimistic.'

function api(split: 'test' | 'train' = 'test') {
  const calls: { url: URL; body?: unknown }[] = []
  const urls = mockApi({
    '/api/models': { body: models },
    [`/api/scenes/${ID}`]: { body: sceneDetail(ID, split, split === 'train' ? TRAIN_WARNING : null) },
    '/api/analysis': (url) => {
      if (url.pathname.endsWith('/explain')) return { body: explanations.logistic_regression }
      if (url.pathname.startsWith('/api/analysis/abcdef')) return { body: analysisFor(ID) }
      return { body: analysisFor(ID, 'logistic_regression', split, split === 'train' ? TRAIN_WARNING : null) }
    },
  })
  return { urls, calls }
}

const page = (path = `/triage/${ID}`) =>
  renderAt(<Routes><Route path="/triage/:sceneId" element={<TriageDeskPage />} /><Route path="/case/:analysisId/:buildingUid" element={<CasePage />} /></Routes>, path)

describe('Triage Desk', () => {
  it('a held-out scene shows the test note; prediction overlays wait for an analysis', async () => {
    api()
    page()
    expect(await screen.findByText(ID)).toBeInTheDocument()
    expect(screen.getByText('Held-out test scene: never used for training or tuning.')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Prediction' })).toBeDisabled()
    expect(screen.getByRole('radio', { name: 'Ground truth' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getAllByRole('button', { name: /^Building / })).toHaveLength(2) // ground-truth footprints drawn
  })

  it('a training scene shows the optimism warning prominently', async () => {
    api('train')
    page()
    expect(await screen.findByRole('alert')).toHaveTextContent(TRAIN_WARNING)
  })

  it('untrained models cannot be selected; the default model is preselected', async () => {
    api()
    page()
    await screen.findByText(ID)
    expect(await screen.findByRole('radio', { name: /Random Forest/ })).toBeDisabled()
    expect(screen.getByRole('radio', { name: 'Logistic Regression' })).toHaveAttribute('aria-checked', 'true')
  })

  it('running the analysis posts scene + model and shows counts and review suggestions from the API', async () => {
    const { urls } = api()
    page()
    await userEvent.click(await screen.findByRole('button', { name: 'Run analysis' }))
    const summary = (await screen.findByText('ANALYSIS · abcdef01')).closest('section')!
    expect(summary).toHaveTextContent('1 of 2 buildings are suggested for human review') // 1 of 2 below 0.48
    expect(within(summary).getByText('0.48')).toBeInTheDocument()
    expect(within(summary).getByText('probability (not calibrated)', { exact: false })).toBeInTheDocument()
    expect(urls.some((u) => u.pathname === '/api/analysis')).toBe(true)
    expect(screen.getByRole('radio', { name: 'Prediction' })).not.toBeDisabled()
  })

  it('clicking a footprint opens that building’s case file with its explanation', async () => {
    api()
    page(`/triage/${ID}?run=1`)
    await screen.findByText('ANALYSIS · abcdef01')
    fireEvent.pointerUp(screen.getByRole('button', { name: `Building ${ID}_b0002` }))
    expect(await screen.findByText('CASE · B-0002')).toBeInTheDocument()
    expect(screen.getByText('✓ agrees with ground truth')).toBeInTheDocument()
    expect(await screen.findByText('Mean blue, post', { exact: false })).toBeInTheDocument()
    expect(screen.getByText('-4.37')).toBeInTheDocument()
  })

  it('selecting a low-confidence building from the list shows it as suggested for review, neutrally', async () => {
    api()
    page(`/triage/${ID}?run=1`)
    const list = await screen.findByRole('list', { name: 'Buildings' })
    await userEvent.click(within(list).getByText('b0001'))
    expect(await screen.findByText('SUGGESTED FOR HUMAN REVIEW')).toBeInTheDocument()
    expect(screen.getByText('✗ differs from ground truth')).toBeInTheDocument()
  })
})

it('keyboard: Enter on a focused footprint opens it; arrow keys pan without errors', async () => {
  api()
  page(`/triage/${ID}?run=1`)
  await screen.findByText('ANALYSIS · abcdef01')
  const footprint = screen.getByRole('button', { name: `Building ${ID}_b0001` })
  footprint.focus()
  await userEvent.keyboard('{Enter}')
  expect(await screen.findByText('CASE · B-0001')).toBeInTheDocument()
  const viewer = screen.getByRole('group', { name: /Scene viewer/ })
  viewer.focus()
  await userEvent.keyboard('{ArrowLeft}+')
  expect(screen.getByText('150%')).toBeInTheDocument()
})

describe('Case file page', () => {
  it.each([
    ['decision_tree', 'Leaf of', 'Texture smoothness change'],
    ['knn', 'Votes:', 'minor'],
  ])('renders the %s explanation', async (model, text, detail) => {
    mockApi({
      '/api/analysis': (url) => url.pathname.endsWith('/explain') ? { body: explanations[model] } : { body: analysisFor(ID, model) },
    })
    renderAt(<Routes><Route path="/case/:analysisId/:buildingUid" element={<CasePage />} /></Routes>, `/case/abcdef0123456789/${ID}_b0002`)
    expect(await screen.findByText(text, { exact: false })).toBeInTheDocument()
    expect(screen.getAllByText(detail, { exact: false }).length).toBeGreaterThan(0)
  })

  it('unknown analysis shows a clean error', async () => {
    mockApi({ '/api/analysis': { status: 404, body: { error: { code: 'analysis_not_found', message: 'Unknown analysis id.' } } } })
    renderAt(<Routes><Route path="/case/:analysisId/:buildingUid" element={<CasePage />} /></Routes>, '/case/0000000000000000/x_b0001')
    expect(await screen.findByText('Could not load this analysis')).toBeInTheDocument()
  })
})
