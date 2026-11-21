import type { Curve } from '../api/types'

interface Props {
  curve: Curve
  description: string
}

/**
 * The tone curve as a photo editor draws it: input tones run left to right
 * from black to white, output tones bottom to top.
 */
export function ToneCurve({ curve, description }: Props) {
  const top = curve.axis_max
  const pad = top * 0.04 // room so handles in the corners are not cut off
  const flip = (y: number) => top - y // SVG counts y downwards; tones count upwards
  const quarters = [0.25, 0.5, 0.75].map((fraction) => top * fraction)

  return (
    <figure className="curve">
      <svg viewBox={`${-pad} ${-pad} ${top + 2 * pad} ${top + 2 * pad}`} role="img" aria-label={description}>
        {quarters.map((at) => (
          <g key={at} className="curve-grid">
            <line x1={at} y1={0} x2={at} y2={top} />
            <line x1={0} y1={at} x2={top} y2={at} />
          </g>
        ))}
        <line className="curve-unchanged" x1={0} y1={top} x2={top} y2={0} />
        <polyline className="curve-shape" points={curve.points.map(([x, y]) => `${x},${flip(y)}`).join(' ')} />
        {curve.handles.map(([x, y]) => (
          <circle key={x} className="curve-handle" cx={x} cy={flip(y)} r={top * 0.018} />
        ))}
      </svg>
      <figcaption className="curve-axis">
        <span>Shadows</span>
        <span>Highlights</span>
      </figcaption>
    </figure>
  )
}
