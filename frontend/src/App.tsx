import { useEffect, useState } from 'react'

import { ApiError, listEdits, matchLook } from './api/client'
import type { EditInfo, MatchResult } from './api/types'
import { CompareView } from './components/CompareView'
import { Downloads } from './components/Downloads'
import { EditsPanel } from './components/EditsPanel'
import { PhotoSlot } from './components/PhotoSlot'
import { usePhotoSlot } from './hooks/usePhotoSlot'

/** What the person's own photo looked like when the current result was made. */
interface Matched {
  result: MatchResult
  name: string
  original: string
  preview: string
}

export function App() {
  const source = usePhotoSlot()
  const reference = usePhotoSlot()
  const [strength, setStrength] = useState(100)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [matched, setMatched] = useState<Matched | null>(null)
  const [edits, setEdits] = useState<EditInfo[]>([])

  useEffect(() => {
    // Only used to place slider markers; the page works without it.
    listEdits().then(setEdits, () => setEdits([]))
  }, [])

  const ready = source.state.status === 'ready' && reference.state.status === 'ready'

  const match = async () => {
    if (source.state.status !== 'ready' || reference.state.status !== 'ready') return
    const photo = source.state
    setBusy(true)
    setError(null)
    try {
      const result = await matchLook(photo.file, reference.state.file, strength / 100)
      setMatched({ result, name: photo.file.name, original: photo.objectUrl, preview: photo.info.preview })
    } catch (failure) {
      const reason = failure instanceof ApiError ? failure.message : 'unknown error'
      setError(`Could not match the look: ${reason}.`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="wall">
      <header className="masthead">
        <h1>PresetFit</h1>
        <p className="lede">Put the look of a photo you love onto your own.</p>
      </header>

      <section className="pair" aria-label="Photos">
        <PhotoSlot
          step={1}
          title="Your photo"
          hint="Drop your photo here, or click to choose one"
          state={source.state}
          onChoose={source.choose}
        />
        <PhotoSlot
          step={2}
          title="The look you want"
          hint="Drop the edited photo whose look you want"
          state={reference.state}
          onChoose={reference.choose}
        />
      </section>

      <section className="action" aria-label="Match">
        <span className="step" aria-hidden="true">
          3
        </span>
        <div className="strength">
          <label htmlFor="strength">Strength</label>
          <input
            id="strength"
            type="range"
            min={0}
            max={100}
            step={5}
            value={strength}
            onChange={(event) => setStrength(Number(event.target.value))}
          />
          <output htmlFor="strength">{strength}%</output>
        </div>
        <button type="button" className="button primary" disabled={!ready || busy} onClick={match}>
          {busy ? 'Matching the look…' : 'Match the look'}
        </button>
        <p className={error ? 'caption error' : 'caption'} role="status" aria-live="polite">
          {error ?? (busy ? 'This takes a few seconds; larger photos take longer.' : '')}
        </p>
      </section>

      {matched && (
        <section className="result" aria-label="Result">
          <div className="result-photo">
            <CompareView
              key={matched.result.image}
              before={matched.original}
              beforeFallback={matched.preview}
              after={matched.result.image}
            />
            <p className="caption">Drag across the photo to compare your original with the new look.</p>
            <Downloads result={matched.result} sourceName={matched.name} />
          </div>
          <EditsPanel result={matched.result} edits={edits} />
        </section>
      )}

      <footer className="colophon">
        <p>Photos are processed to make the result and are not stored.</p>
      </footer>
    </main>
  )
}
