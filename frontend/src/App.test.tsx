import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { App } from './App'
import { EDITS, jsonResponse, MATCH_RESULT, PHOTO_INFO, photo } from './test/fixtures'

/** A fake server: answers each API address the way the real one would. */
function fakeServer(options: { rejectNames?: string[] } = {}) {
  return vi.fn(async (path: string, init?: RequestInit) => {
    if (path === 'api/v1/edits') return jsonResponse(EDITS)
    if (path === 'api/v1/inspect') {
      const file = (init!.body as FormData).get('photo') as File
      if (options.rejectNames?.includes(file.name)) {
        return jsonResponse({ error: 'invalid_image', detail: 'the file could not be read as an image' }, 400)
      }
      return jsonResponse(PHOTO_INFO)
    }
    if (path === 'api/v1/match') return jsonResponse(MATCH_RESULT)
    return jsonResponse({ error: 'http_error', detail: 'Not Found' }, 404)
  })
}

beforeEach(() => {
  vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: () => 'blob:photo', revokeObjectURL: () => {} }))
})
afterEach(() => vi.unstubAllGlobals())

describe('App', () => {
  it('only allows matching once both photos have passed the check', async () => {
    vi.stubGlobal('fetch', fakeServer())
    const user = userEvent.setup()
    render(<App />)
    const button = screen.getByRole('button', { name: 'Match the look' })
    expect(button).toBeDisabled()

    await user.upload(screen.getByLabelText('Your photo'), photo('mine.jpg'))
    expect(await screen.findByText('mine.jpg, 1600 × 1067 px')).toBeInTheDocument()
    expect(button).toBeDisabled()

    await user.upload(screen.getByLabelText('The look you want'), photo('ref.jpg'))
    expect(await screen.findByText('ref.jpg, 1600 × 1067 px')).toBeInTheDocument()
    expect(button).toBeEnabled()
  })

  it('shows a rejected photo’s error under that photo, not the other one', async () => {
    vi.stubGlobal('fetch', fakeServer({ rejectNames: ['broken.jpg'] }))
    const user = userEvent.setup()
    render(<App />)

    await user.upload(screen.getByLabelText('Your photo'), photo('mine.jpg'))
    await user.upload(screen.getByLabelText('The look you want'), photo('broken.jpg'))

    const lookSlot = screen.getByLabelText('The look you want').closest('figure')!
    const photoSlot = screen.getByLabelText('Your photo').closest('figure')!
    expect(await within(lookSlot).findByText(/This photo can’t be used/)).toBeInTheDocument()
    expect(within(photoSlot).queryByText(/can’t be used/)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Match the look' })).toBeDisabled()
  })

  it('shows the new look, the edits and the downloads after matching', async () => {
    vi.stubGlobal('fetch', fakeServer())
    const user = userEvent.setup()
    render(<App />)
    await user.upload(screen.getByLabelText('Your photo'), photo('mine.jpg'))
    await user.upload(screen.getByLabelText('The look you want'), photo('ref.jpg'))
    await screen.findByText('ref.jpg, 1600 × 1067 px')

    await user.click(screen.getByRole('button', { name: 'Match the look' }))

    expect(await screen.findByRole('img', { name: 'Your photo with the new look' })).toBeInTheDocument()
    expect(screen.getByText('88%')).toBeInTheDocument()
    expect(screen.getByText('+0.35 EV')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /Tone curve/ })).toBeInTheDocument()
    expect(screen.getByText(/skipped for blue/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Download photo' })).toBeInTheDocument()
    // One-slider edits do not repeat their name as a heading.
    expect(screen.queryByRole('heading', { name: 'Exposure' })).not.toBeInTheDocument()
  })

  it('moves the comparison divider through its accessible slider', async () => {
    vi.stubGlobal('fetch', fakeServer())
    const user = userEvent.setup()
    render(<App />)
    await user.upload(screen.getByLabelText('Your photo'), photo('mine.jpg'))
    await user.upload(screen.getByLabelText('The look you want'), photo('ref.jpg'))
    await screen.findByText('ref.jpg, 1600 × 1067 px')
    await user.click(screen.getByRole('button', { name: 'Match the look' }))

    const slider = await screen.findByRole('slider', { name: 'Compare the original with the new look' })
    // jsdom cannot move a range input with arrow keys; a real browser does,
    // and the browser check covers that. Here the value change is simulated.
    fireEvent.change(slider, { target: { value: '80' } })
    expect(slider).toHaveValue('80')
    const frame = screen.getByRole('img', { name: 'Your photo with the new look' }).parentElement!
    expect(frame.style.getPropertyValue('--split')).toBe('80%')
  })
})
