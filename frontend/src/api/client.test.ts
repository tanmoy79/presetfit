import { afterEach, describe, expect, it, vi } from 'vitest'

import { jsonResponse, PHOTO_INFO, photo } from '../test/fixtures'
import { ApiError, exportLook, inspectPhoto } from './client'

afterEach(() => vi.unstubAllGlobals())

describe('api client', () => {
  it('returns the photo information when the check succeeds', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(PHOTO_INFO)))
    await expect(inspectPhoto(photo('a.jpg'))).resolves.toEqual(PHOTO_INFO)
  })

  it('turns an error answer into an ApiError with the server’s explanation', async () => {
    const answer = jsonResponse({ error: 'unsupported_image', detail: "unsupported image format 'GIF'" }, 415)
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(answer))
    const failure = inspectPhoto(photo('a.gif'))
    await expect(failure).rejects.toBeInstanceOf(ApiError)
    await expect(failure).rejects.toMatchObject({ code: 'unsupported_image', message: "unsupported image format 'GIF'" })
  })

  it('explains when the server cannot be reached at all', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    await expect(inspectPhoto(photo('a.jpg'))).rejects.toMatchObject({ code: 'unreachable' })
  })

  it('sends slider values as JSON when exporting a look', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('TITLE "PresetFit"'))
    vi.stubGlobal('fetch', fetchMock)
    const blob = await exportLook('lut', { exposure: { exposure: 0.5 } })
    expect(await blob.text()).toContain('TITLE')
    const [path, init] = fetchMock.mock.calls[0]
    expect(path).toBe('api/v1/lut')
    expect(JSON.parse(init.body)).toEqual({ settings: { exposure: { exposure: 0.5 } }, title: 'PresetFit' })
  })
})
