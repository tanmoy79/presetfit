// The only module that talks to the server. Components call these functions
// and never use fetch() themselves, so the API's addresses live in one place.

import type { EditInfo, ErrorBody, MatchResult, PhotoInfo, SliderValues } from './types'

/** A request the server answered with an error, or could not be reached. */
export class ApiError extends Error {
  readonly code: string

  constructor(code: string, message: string) {
    super(message)
    this.name = 'ApiError'
    this.code = code
  }
}

const UNREACHABLE = 'the app is not responding; check that it is still running'

async function request(path: string, init: RequestInit): Promise<Response> {
  let response: Response
  try {
    response = await fetch(path, init)
  } catch {
    // fetch() only throws when no answer arrived at all.
    throw new ApiError('unreachable', UNREACHABLE)
  }
  if (!response.ok) throw await toError(response)
  return response
}

async function toError(response: Response): Promise<ApiError> {
  try {
    const body = (await response.json()) as Partial<ErrorBody>
    return new ApiError(body.error ?? 'http_error', body.detail ?? response.statusText)
  } catch {
    return new ApiError('http_error', response.statusText || 'the server sent an unreadable answer')
  }
}

/** Check one photo the moment it is chosen and get a small preview of it. */
export async function inspectPhoto(photo: File, signal?: AbortSignal): Promise<PhotoInfo> {
  const body = new FormData()
  body.append('photo', photo)
  const response = await request('api/v1/inspect', { method: 'POST', body, signal })
  return (await response.json()) as PhotoInfo
}

/** Give `source` the look of `reference`. `strength` is 0-1. */
export async function matchLook(source: File, reference: File, strength: number): Promise<MatchResult> {
  const body = new FormData()
  body.append('source', source)
  body.append('reference', reference)
  body.append('strength', String(strength))
  const response = await request('api/v1/match', { method: 'POST', body })
  return (await response.json()) as MatchResult
}

/** The edits and their slider ranges, used to draw slider positions. */
export async function listEdits(): Promise<EditInfo[]> {
  const response = await request('api/v1/edits', { method: 'GET' })
  return (await response.json()) as EditInfo[]
}

/** Turn slider values into a downloadable file: a preset or a .cube LUT. */
export async function exportLook(kind: 'preset' | 'lut', settings: SliderValues): Promise<Blob> {
  const response = await request(`api/v1/${kind}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ settings, title: 'PresetFit' }),
  })
  return response.blob()
}
