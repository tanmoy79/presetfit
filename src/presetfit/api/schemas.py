"""Request and response shapes of the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReportLineSchema(BaseModel):
    label: str
    value: str


class CurveSchema(BaseModel):
    """A curve to draw, such as the tone curve."""

    points: list[tuple[float, float]] = Field(description="Samples tracing the curve.")
    handles: list[tuple[float, float]] = Field(description="The control points.")
    axis_max: float = Field(description="Value at the top of both axes.")


class ReportSectionSchema(BaseModel):
    name: str
    label: str
    lines: list[ReportLineSchema]
    curve: CurveSchema | None = None


class ReportSchema(BaseModel):
    sections: list[ReportSectionSchema]
    notes: list[str]
    distance_before: float
    distance_after: float
    improvement: float = Field(description="Fraction of the look gap closed, 0-1.")


class MatchResponse(BaseModel):
    """Result of a match: the edited picture and the edits that produced it."""

    image: str = Field(description="The edited photo as a data URL (JPEG).")
    settings: dict[str, dict[str, float]] = Field(
        description="Slider values by edit, reusable with /api/v1/apply and /api/v1/lut."
    )
    report: ReportSchema
    report_text: str = Field(description="The same report as plain text.")


class InspectResponse(BaseModel):
    """What the server found when it checked one uploaded photo."""

    width: int
    height: int
    megapixels: float
    preview: str = Field(description="A small version of the photo as a data URL (JPEG).")


class ApplyResponse(BaseModel):
    image: str = Field(description="The edited photo as a data URL (JPEG).")


class SettingsRequest(BaseModel):
    settings: dict[str, dict[str, float]]
    title: str = Field(default="PresetFit", max_length=80)


class ParamSchema(BaseModel):
    name: str
    label: str
    minimum: float
    maximum: float
    default: float
    step: float
    unit: str


class EditSchema(BaseModel):
    """One edit and its sliders, so a client can draw controls without hard-coding them."""

    name: str
    label: str
    params: list[ParamSchema]


class HealthResponse(BaseModel):
    status: str
    version: str


class ErrorResponse(BaseModel):
    """Body of every error response."""

    error: str = Field(description="Stable machine-readable code.")
    detail: str = Field(description="Explanation suitable for showing to a person.")
