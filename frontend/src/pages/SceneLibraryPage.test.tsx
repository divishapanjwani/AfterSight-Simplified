import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { sceneList } from '../test/fixtures'
import { mockApi, renderAt } from '../test/utils'
import { SceneLibraryPage } from './SceneLibraryPage'

afterEach(() => vi.unstubAllGlobals())

function scenesApi() {
  return mockApi({
    '/api/scenes': (url) => {
      const split = url.searchParams.get('split')
      if (split === 'train') return { body: sceneList('train', ['ev-a_00000001', 'ev-a_00000002']) }
      if (url.searchParams.get('event') === 'ev-empty') return { body: { ...sceneList('test', []), total: 0 } }
      return { body: sceneList('test', ['ev-a_00000009']) }
    },
  })
}

describe('Scene Library', () => {
  it('defaults to held-out test scenes', async () => {
    const calls = scenesApi()
    renderAt(<SceneLibraryPage />, '/library')
    expect(await screen.findByText('ev-a_00000009')).toBeInTheDocument()
    expect(calls[0].searchParams.get('split')).toBe('test')
    expect(screen.getByRole('radio', { name: /Test \(held out\)/ })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByText('TEST SET')).toBeInTheDocument()
    expect(screen.queryByRole('note')).not.toBeInTheDocument() // no warning on test scenes
  })

  it('switching to training scenes shows the optimism warning on the page and on every row', async () => {
    const calls = scenesApi()
    renderAt(<SceneLibraryPage />, '/library')
    await screen.findByText('ev-a_00000009')
    await userEvent.click(screen.getByRole('radio', { name: /Training/ }))
    expect(await screen.findByText('ev-a_00000001')).toBeInTheDocument()
    expect(calls.at(-1)!.searchParams.get('split')).toBe('train')
    expect(screen.getByRole('note')).toHaveTextContent('Training scenes were used to fit the models')
    expect(screen.getAllByText('TRAINING SCENE')).toHaveLength(2)
  })

  it('shows ground-truth counts and data-quality warnings from the API', async () => {
    scenesApi()
    renderAt(<SceneLibraryPage />, '/library')
    expect(await screen.findByRole('img', { name: /Ground truth: No damage 4, Minor damage 1, Major damage 0, Destroyed 2/ })).toBeInTheDocument()
    expect(screen.getByText('Pre/post images may be misaligned (estimated shift 12.5 px).')).toBeInTheDocument()
  })

  it('empty filter result offers a reset', async () => {
    scenesApi()
    renderAt(<SceneLibraryPage />, '/library?event=ev-empty')
    expect(await screen.findByText('No scenes match these filters.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reset to test scenes' })).toBeInTheDocument()
  })

  it('dataset not built: shows the command instead of an error', async () => {
    mockApi({ '/api/scenes': { status: 503, body: { error: { code: 'dataset_not_ready', message: 'The scene cache has not been built yet.', fix: 'python scripts/inspect_dataset.py --alignment' } } } })
    renderAt(<SceneLibraryPage />, '/library')
    expect(await screen.findByText('scenes is not available yet')).toBeInTheDocument()
    expect(screen.getByText('$ python scripts/inspect_dataset.py --alignment')).toBeInTheDocument()
  })
})
