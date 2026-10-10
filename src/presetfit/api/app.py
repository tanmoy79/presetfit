"""The FastAPI application.

Like the CLI, this layer only translates: HTTP in, a call to
:class:`presetfit.service.PresetFitService`, HTTP out.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Annotated, BinaryIO

from fastapi import APIRouter, Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from presetfit._version import __version__
from presetfit.api.config import ApiSettings
from presetfit.api.schemas import (
    ApplyResponse,
    EditSchema,
    ErrorResponse,
    HealthResponse,
    InspectResponse,
    MatchResponse,
    ParamSchema,
    ReportSchema,
    SettingsRequest,
)
from presetfit.color.pixels import Pixels
from presetfit.edits.settings import EditSettings
from presetfit.exceptions import (
    FittingError,
    ImageDecodeError,
    ImageTooLargeError,
    InvalidSettingsError,
    PresetFitError,
    UnsupportedImageError,
)
from presetfit.imaging import ImageReader, ImageWriter
from presetfit.report import TextReportFormatter
from presetfit.service import PresetFitService

logger = logging.getLogger(__name__)

_INDEX_PAGE = Path(__file__).parent / "static" / "index.html"
_READ_CHUNK_BYTES = 1024 * 1024
_PREVIEW_EDGE = 640

# Each expected failure maps to one HTTP status and one stable error code.
_ERROR_STATUS: dict[type[PresetFitError], tuple[int, str]] = {
    ImageTooLargeError: (413, "image_too_large"),
    UnsupportedImageError: (415, "unsupported_image"),
    ImageDecodeError: (400, "invalid_image"),
    InvalidSettingsError: (422, "invalid_settings"),
    FittingError: (422, "fit_failed"),
}
_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    status: {"model": ErrorResponse} for status in (400, 413, 415, 422)
}


class _Context:
    """The long-lived objects every request handler needs."""

    def __init__(self, service: PresetFitService, settings: ApiSettings) -> None:
        self.service = service
        self.settings = settings
        self.reader = ImageReader(max_pixels=settings.max_pixels)
        self.writer = ImageWriter(jpeg_quality=settings.jpeg_quality)
        self.formatter = TextReportFormatter()

    def read_upload(self, upload: UploadFile) -> Pixels:
        """Decode an uploaded file, refusing anything over the size limit."""
        return self.reader.read(_read_limited(upload.file, self.settings.max_upload_bytes))

    def to_data_url(self, image: Pixels, max_edge: int | None = None) -> str:
        encoded = base64.b64encode(self.writer.encode(image, "JPEG", max_edge)).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"


def _context(request: Request) -> _Context:
    context: _Context = request.app.state.context
    return context


ContextDep = Annotated[_Context, Depends(_context)]
router = APIRouter()


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page web interface."""
    return FileResponse(_INDEX_PAGE, media_type="text/html")


@router.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    """Liveness probe for load balancers and container orchestrators."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/api/v1/edits", response_model=list[EditSchema], tags=["look"])
def edits(context: ContextDep) -> list[EditSchema]:
    """Describe every edit and its sliders, in the order they are applied."""
    return [
        EditSchema(
            name=edit.name,
            label=edit.label,
            params=[
                ParamSchema(
                    name=spec.name,
                    label=spec.label,
                    minimum=spec.minimum,
                    maximum=spec.maximum,
                    default=spec.default,
                    step=spec.step,
                    unit=spec.unit,
                )
                for spec in edit.params
            ],
        )
        for edit in context.service.pipeline.edits
    ]


@router.post(
    "/api/v1/inspect", response_model=InspectResponse, responses=_ERROR_RESPONSES, tags=["look"]
)
def inspect(
    context: ContextDep,
    photo: Annotated[UploadFile, File(description="The photo to check.")],
) -> InspectResponse:
    """Check one photo before it is used, and return a small preview of it.

    The interface calls this the moment a photo is chosen, so a file that
    cannot be used is reported next to that photo rather than later.
    """
    image = context.read_upload(photo)
    height, width, _ = image.data.shape
    return InspectResponse(
        width=width,
        height=height,
        megapixels=round(width * height / 1e6, 1),
        preview=context.to_data_url(image, max_edge=_PREVIEW_EDGE),
    )


@router.post(
    "/api/v1/match", response_model=MatchResponse, responses=_ERROR_RESPONSES, tags=["look"]
)
def match(
    context: ContextDep,
    source: Annotated[UploadFile, File(description="The photo to edit.")],
    reference: Annotated[UploadFile, File(description="The photo whose look to copy.")],
    strength: Annotated[float, Form(ge=0.0, le=1.0)] = 1.0,
) -> MatchResponse:
    """Give ``source`` the look of ``reference`` and report the edits used."""
    result = context.service.match(
        context.read_upload(source), context.read_upload(reference), strength=strength
    )
    return MatchResponse(
        image=context.to_data_url(result.image),
        settings=result.settings.to_dict(),
        report=ReportSchema.model_validate(result.report.to_dict()),
        report_text=context.formatter.format(result.report),
    )


@router.post(
    "/api/v1/apply", response_model=ApplyResponse, responses=_ERROR_RESPONSES, tags=["look"]
)
def apply(
    context: ContextDep,
    source: Annotated[UploadFile, File(description="The photo to edit.")],
    preset: Annotated[str, Form(description="A preset document, as saved by the app.")],
) -> ApplyResponse:
    """Apply a previously fitted preset to another photo."""
    settings = context.service.presets.loads(preset)
    image = context.service.apply(context.read_upload(source), settings)
    return ApplyResponse(image=context.to_data_url(image))


@router.post("/api/v1/preset", responses=_ERROR_RESPONSES, tags=["look"])
def preset_file(context: ContextDep, body: SettingsRequest) -> Response:
    """Return slider settings as a downloadable preset file."""
    document = context.service.presets.dumps(EditSettings.from_dict(body.settings))
    return _download(document, "application/json", "look.presetfit.json")


@router.post("/api/v1/lut", responses=_ERROR_RESPONSES, tags=["look"])
def lut_file(context: ContextDep, body: SettingsRequest) -> Response:
    """Return slider settings baked into a ``.cube`` LUT for other editors."""
    cube = context.service.export_lut(EditSettings.from_dict(body.settings), title=body.title)
    return _download(cube, "text/plain; charset=us-ascii", "look.cube")


def create_app(
    service: PresetFitService | None = None, settings: ApiSettings | None = None
) -> FastAPI:
    """Build the web application.

    This is an *application factory*: nothing is created at import time, so
    tests can build an app with their own service and limits.
    """
    settings = settings or ApiSettings.from_env()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)

    app = FastAPI(
        title="PresetFit",
        version=__version__,
        description="Give one photo the look of another and list the edits that did it.",
    )
    app.state.context = _Context(service or PresetFitService.create_default(), settings)
    app.include_router(router)
    _install_error_handlers(app)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    return app


def _install_error_handlers(app: FastAPI) -> None:
    """Make every failure, expected or not, return the same JSON error shape."""

    @app.exception_handler(PresetFitError)
    async def handle_expected(_: Request, error: PresetFitError) -> JSONResponse:
        status, code = _ERROR_STATUS.get(type(error), (400, "bad_request"))
        return _error(status, code, str(error))

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_: Request, error: RequestValidationError) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'][1:]) or 'request'}: {item['msg']}"
            for item in error.errors()
        )
        return _error(422, "invalid_request", problems)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(_: Request, error: StarletteHTTPException) -> JSONResponse:
        return _error(error.status_code, "http_error", str(error.detail))

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, error: Exception) -> JSONResponse:
        # Log the details for operators; never send internals to the client.
        logger.exception("unhandled error", exc_info=error)
        return _error(500, "internal_error", "Something went wrong on the server.")


def _error(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content=ErrorResponse(error=code, detail=detail).model_dump()
    )


def _download(content: str, media_type: str, filename: str) -> Response:
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _read_limited(stream: BinaryIO, limit: int) -> bytes:
    """Read ``stream`` fully, stopping with an error as soon as it exceeds ``limit``."""
    chunks: list[bytes] = []
    total = 0
    while chunk := stream.read(_READ_CHUNK_BYTES):
        total += len(chunk)
        if total > limit:
            raise ImageTooLargeError(
                f"file is larger than the {limit / (1024 * 1024):.0f} MB upload limit"
            )
        chunks.append(chunk)
    if not chunks:
        raise ImageDecodeError("the uploaded file is empty")
    return b"".join(chunks)
