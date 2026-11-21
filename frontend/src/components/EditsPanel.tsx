import type { EditInfo, MatchResult, Param, ReportSection } from '../api/types'
import { ToneCurve } from './ToneCurve'

interface Props {
  result: MatchResult
  edits: EditInfo[]
}

/** The list of edits that produced the new look, with the match score and any notes. */
export function EditsPanel({ result, edits }: Props) {
  const { report, settings } = result
  return (
    <section className="edits" aria-labelledby="edits-title">
      <h2 id="edits-title">How it was done</h2>
      <p className="score">
        Look matched <strong>{Math.round(report.improvement * 100)}%</strong>
      </p>
      {report.sections.map((section) => (
        <EditSection
          key={section.name}
          section={section}
          edit={edits.find((item) => item.name === section.name)}
          values={settings[section.name] ?? {}}
        />
      ))}
      {report.notes.length > 0 && (
        <div className="notes">
          {report.notes.map((note) => (
            <p key={note}>{note}</p>
          ))}
        </div>
      )}
    </section>
  )
}

interface SectionProps {
  section: ReportSection
  edit: EditInfo | undefined
  values: Record<string, number>
}

function EditSection({ section, edit, values }: SectionProps) {
  // A one-slider edit such as Exposure needs no heading repeating its name.
  const selfTitled = section.lines.length === 1 && section.lines[0].label === section.label
  // Edits with one or two sliders are drawn as sliders; the rest as text lines.
  const asSliders = edit !== undefined && edit.params.length <= 2 && edit.params.length === section.lines.length
  const summary = section.lines.map((line) => `${line.label}: ${line.value}`).join(', ')

  return (
    <div className="edit">
      {!selfTitled && <h3>{section.label}</h3>}
      {section.curve && <ToneCurve curve={section.curve} description={`${section.label}. ${summary}`} />}
      <dl>
        {section.lines.map((line, index) => (
          <div className="edit-line" key={line.label}>
            <dt>{line.label}</dt>
            <dd>{line.value}</dd>
            {asSliders && <SliderTrack param={edit.params[index]} value={values[edit.params[index].name] ?? 0} />}
          </div>
        ))}
      </dl>
    </div>
  )
}

function SliderTrack({ param, value }: { param: Param; value: number }) {
  const span = param.maximum - param.minimum
  const at = (x: number) => `${((x - param.minimum) / span) * 100}%`
  return (
    <div
      className="track"
      data-param={param.name}
      style={{ '--zero': at(param.default), '--at': at(value) } as React.CSSProperties}
      aria-hidden="true"
    >
      <i />
    </div>
  )
}
