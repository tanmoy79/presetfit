import type { EditInfo, MatchResult, PhotoInfo } from '../api/types'

export const PHOTO_INFO: PhotoInfo = {
  width: 1600,
  height: 1067,
  megapixels: 1.7,
  preview: 'data:image/jpeg;base64,cHJldmlldw==',
}

export const EDITS: EditInfo[] = [
  {
    name: 'exposure',
    label: 'Exposure',
    params: [{ name: 'exposure', label: 'Exposure', minimum: -3, maximum: 3, default: 0, step: 0.01, unit: 'EV' }],
  },
]

export const MATCH_RESULT: MatchResult = {
  image: 'data:image/jpeg;base64,cmVzdWx0',
  settings: {
    exposure: { exposure: 0.35 },
    tone_curve: { point_0: 30, point_64: 61, point_128: 131, point_191: 191, point_255: 255 },
  },
  report: {
    sections: [
      { name: 'exposure', label: 'Exposure', lines: [{ label: 'Exposure', value: '+0.35 EV' }], curve: null },
      {
        name: 'tone_curve',
        label: 'Tone curve',
        lines: [{ label: 'Point 0', value: '0 → 30' }],
        curve: {
          points: [
            [0, 30],
            [255, 255],
          ],
          handles: [
            [0, 30],
            [255, 255],
          ],
          axis_max: 255,
        },
      },
    ],
    notes: ['Per-colour (HSL) edits were skipped for blue.'],
    distance_before: 8,
    distance_after: 1,
    improvement: 0.875,
  },
  report_text: 'EDITS APPLIED',
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

export function photo(name: string): File {
  return new File(['pixels'], name, { type: 'image/jpeg' })
}
