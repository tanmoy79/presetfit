import { useRef, useState, type PointerEvent } from 'react'

interface Props {
  before: string
  beforeFallback: string
  after: string
}

/**
 * The matched photo with a divider the person drags across it: the original
 * shows to the left of the divider, the new look to the right. A real range
 * input sits underneath, so the comparison also works from the keyboard.
 */
export function CompareView({ before, beforeFallback, after }: Props) {
  const [split, setSplit] = useState(50)
  const [beforeSrc, setBeforeSrc] = useState(before)
  const frame = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)

  const moveTo = (clientX: number) => {
    const box = frame.current?.getBoundingClientRect()
    if (!box || box.width === 0) return
    setSplit(Math.min(100, Math.max(0, ((clientX - box.left) / box.width) * 100)))
  }

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    dragging.current = true
    event.currentTarget.setPointerCapture(event.pointerId)
    moveTo(event.clientX)
  }

  return (
    <div className="compare">
      <div
        ref={frame}
        className="print compare-frame"
        style={{ '--split': `${split}%` } as React.CSSProperties}
        onPointerDown={onPointerDown}
        onPointerMove={(event) => dragging.current && moveTo(event.clientX)}
        onPointerUp={() => (dragging.current = false)}
        onPointerCancel={() => (dragging.current = false)}
      >
        <img className="compare-after" src={after} alt="Your photo with the new look" draggable={false} />
        <img
          className="compare-before"
          src={beforeSrc}
          alt="Your original photo"
          draggable={false}
          // Browsers cannot show every format the app reads (TIFF, for one).
          onError={() => setBeforeSrc(beforeFallback)}
        />
        <div className="compare-divider" aria-hidden="true" />
        <span className="compare-tag compare-tag-before" aria-hidden="true">
          Original
        </span>
        <span className="compare-tag compare-tag-after" aria-hidden="true">
          New look
        </span>
      </div>
      <label className="visually-hidden" htmlFor="compare-range">
        Compare the original with the new look
      </label>
      <input
        id="compare-range"
        className="visually-hidden"
        type="range"
        min={0}
        max={100}
        value={Math.round(split)}
        aria-valuetext={`${Math.round(split)}% original`}
        onChange={(event) => setSplit(Number(event.target.value))}
      />
    </div>
  )
}
