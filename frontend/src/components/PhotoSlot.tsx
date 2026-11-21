import { useId, useState, type DragEvent } from 'react'

import type { PhotoSlotState } from '../hooks/usePhotoSlot'

const ACCEPTED = 'image/jpeg,image/png,image/webp,image/tiff,image/bmp,.jpg,.jpeg,.png,.webp,.tif,.tiff,.bmp'

interface Props {
  step: number
  title: string
  hint: string
  state: PhotoSlotState
  onChoose: (file: File) => void
}

/** One print on the wall: drop or choose a photo, then see it or why it can't be used. */
export function PhotoSlot({ step, title, hint, state, onChoose }: Props) {
  const inputId = useId()
  const noteId = useId()
  const [dragging, setDragging] = useState(false)

  const take = (files: FileList | null) => {
    const file = files?.[0]
    if (file) onChoose(file)
  }

  const onDrop = (event: DragEvent) => {
    event.preventDefault()
    setDragging(false)
    take(event.dataTransfer.files)
  }

  const preview = state.status === 'ready' ? state.info.preview : null

  return (
    <figure className="slot" data-status={state.status}>
      <figcaption className="slot-title">
        <span className="step" aria-hidden="true">
          {step}
        </span>
        <label htmlFor={inputId}>{title}</label>
      </figcaption>
      <div
        className={`print drop${dragging ? ' is-dragging' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        {preview ? (
          <img src={preview} alt={state.status === 'ready' ? state.file.name : ''} />
        ) : (
          <span className="drop-hint">{state.status === 'empty' ? hint : state.file.name}</span>
        )}
        <input
          id={inputId}
          type="file"
          accept={ACCEPTED}
          aria-describedby={noteId}
          onChange={(event) => {
            take(event.target.files)
            event.target.value = '' // so choosing the same file again is noticed
          }}
        />
      </div>
      <p className="caption" id={noteId} role="status" aria-live="polite">
        <SlotNote state={state} />
      </p>
    </figure>
  )
}

function SlotNote({ state }: { state: PhotoSlotState }) {
  switch (state.status) {
    case 'empty':
      return <>JPEG, PNG, WebP, TIFF or BMP</>
    case 'checking':
      return <>Checking this photo…</>
    case 'ready':
      return (
        <>
          {state.file.name}, {state.info.width} × {state.info.height} px
        </>
      )
    case 'invalid':
      return <span className="error">This photo can’t be used: {state.message}. Choose a different file.</span>
  }
}
