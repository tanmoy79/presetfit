// Shapes of the data the Python API sends and receives.
// They mirror src/presetfit/api/schemas.py; change both together.

export type SliderValues = Record<string, Record<string, number>>

export interface ReportLine {
  label: string
  value: string
}

export interface Curve {
  points: [number, number][]
  handles: [number, number][]
  axis_max: number
}

export interface ReportSection {
  name: string
  label: string
  lines: ReportLine[]
  curve: Curve | null
}

export interface Report {
  sections: ReportSection[]
  notes: string[]
  distance_before: number
  distance_after: number
  improvement: number
}

export interface MatchResult {
  image: string
  settings: SliderValues
  report: Report
  report_text: string
}

export interface PhotoInfo {
  width: number
  height: number
  megapixels: number
  preview: string
}

export interface Param {
  name: string
  label: string
  minimum: number
  maximum: number
  default: number
  step: number
  unit: string
}

export interface EditInfo {
  name: string
  label: string
  params: Param[]
}

export interface ErrorBody {
  error: string
  detail: string
}
