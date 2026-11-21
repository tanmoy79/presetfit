import { useState } from 'react'

import { ApiError, exportLook } from '../api/client'
import type { MatchResult } from '../api/types'

interface Props {
  result: MatchResult
  sourceName: string
}

/** Ways to take the result away: the photo, the look as a preset or LUT, or the edits as text. */
export function Downloads({ result, sourceName }: Props) {
  const [message, setMessage] = useState<{ text: string; isError: boolean } | null>(null)

  const save = (href: string, filename: string) => {
    const link = document.createElement('a')
    link.href = href
    link.download = filename
    document.body.append(link)
    link.click()
    link.remove()
  }

  const saveExport = async (kind: 'preset' | 'lut', filename: string) => {
    try {
      const url = URL.createObjectURL(await exportLook(kind, result.settings))
      save(url, filename)
      URL.revokeObjectURL(url)
      setMessage(null)
    } catch (error) {
      const reason = error instanceof ApiError ? error.message : 'unknown error'
      setMessage({ text: `Could not download the file: ${reason}.`, isError: true })
    }
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(result.report_text)
      setMessage({ text: 'Edits copied.', isError: false })
    } catch {
      setMessage({ text: 'Could not copy: your browser blocked clipboard access.', isError: true })
    }
  }

  const stem = sourceName.replace(/\.[^.]+$/, '')
  return (
    <div className="downloads">
      <button type="button" className="button" onClick={() => save(result.image, `${stem}_matched.jpg`)}>
        Download photo
      </button>
      <button type="button" className="button quiet" onClick={() => saveExport('preset', 'look.presetfit.json')}>
        Download preset
      </button>
      <button type="button" className="button quiet" onClick={() => saveExport('lut', 'look.cube')}>
        Download LUT
      </button>
      <button type="button" className="button quiet" onClick={copy}>
        Copy edits as text
      </button>
      <p className={message?.isError ? 'caption error' : 'caption'} role="status" aria-live="polite">
        {message?.text}
      </p>
    </div>
  )
}
