from __future__ import annotations

import base64
import json
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from presetfit import ColorSpace, ImageReader, Pixels, PresetFitService
from presetfit import __version__ as package_version
from presetfit.api import ApiSettings, create_app

from .helpers import encode, lab_difference, settings_with


@pytest.fixture(scope="module")
def client(service: PresetFitService) -> Iterator[TestClient]:
    with TestClient(create_app(service, ApiSettings())) as test_client:
        yield test_client


def scene_space() -> ColorSpace:
    return ColorSpace.SRGB


def upload(image: Pixels, name: str = "photo.png") -> tuple[str, bytes, str]:
    return (name, encode(image), "image/png")


def decode_data_url(data_url: str) -> Pixels:
    prefix = "data:image/jpeg;base64,"
    assert data_url.startswith(prefix)
    return ImageReader().read(base64.b64decode(data_url[len(prefix) :]))


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": package_version}


def test_index_page_is_served_with_security_headers(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Match look" in response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_edits_endpoint_describes_every_slider(
    client: TestClient, service: PresetFitService
) -> None:
    body = client.get("/api/v1/edits").json()
    assert [edit["name"] for edit in body] == [edit.name for edit in service.pipeline.edits]
    exposure = next(edit for edit in body if edit["name"] == "exposure")["params"][0]
    assert (exposure["minimum"], exposure["maximum"], exposure["unit"]) == (-3.0, 3.0, "EV")


def test_match_returns_picture_settings_and_report(
    client: TestClient, service: PresetFitService, scene: Pixels
) -> None:
    reference = service.apply(scene, settings_with(service.pipeline, exposure={"exposure": 0.6}))
    response = client.post(
        "/api/v1/match", files={"source": upload(scene), "reference": upload(reference)}
    )
    assert response.status_code == 200
    body = response.json()
    image = decode_data_url(body["image"])
    assert image.data.shape == scene.data.shape
    assert lab_difference(image, reference) < 2.5  # includes JPEG loss
    assert body["settings"]["exposure"]["exposure"] == pytest.approx(0.6, abs=0.1)
    assert body["report"]["improvement"] > 0.9
    assert body["report_text"].startswith("EDITS APPLIED")
    assert len(body["report"]["sections"]) == len(service.pipeline.edits)


def test_inspect_reports_size_and_a_small_preview(client: TestClient, scene: Pixels) -> None:
    response = client.post("/api/v1/inspect", files={"photo": upload(scene)})
    assert response.status_code == 200
    body = response.json()
    assert (body["width"], body["height"], body["megapixels"]) == (128, 96, 0.0)
    assert decode_data_url(body["preview"]).data.shape == scene.data.shape  # already small


def test_inspect_shrinks_large_photos_for_the_preview(client: TestClient) -> None:
    large = Pixels(np.full((900, 1600, 3), 0.5, dtype=np.float32), scene_space())
    body = client.post("/api/v1/inspect", files={"photo": upload(large)}).json()
    assert (body["width"], body["height"], body["megapixels"]) == (1600, 900, 1.4)
    assert decode_data_url(body["preview"]).data.shape == (360, 640, 3)


@pytest.mark.parametrize(
    ("name", "content", "status", "code"),
    [
        ("notes.txt", b"not an image", 400, "invalid_image"),
        ("empty.jpg", b"", 400, "invalid_image"),
        ("animation.gif", None, 415, "unsupported_image"),
    ],
)
def test_inspect_explains_why_a_file_cannot_be_used(
    client: TestClient, scene: Pixels, name: str, content: bytes | None, status: int, code: str
) -> None:
    data = encode(scene, "GIF") if content is None else content
    response = client.post("/api/v1/inspect", files={"photo": (name, data, "image/gif")})
    assert response.status_code == status
    assert response.json()["error"] == code


def test_match_report_includes_the_tone_curve(client: TestClient, scene: Pixels) -> None:
    files = {"source": upload(scene), "reference": upload(scene)}
    sections = client.post("/api/v1/match", files=files).json()["report"]["sections"]
    curve = next(section for section in sections if section["name"] == "tone_curve")["curve"]
    assert curve["handles"] == [[0, 0], [64, 64], [128, 128], [191, 191], [255, 255]]
    assert all(section["curve"] is None for section in sections if section["name"] != "tone_curve")


def test_match_honours_strength(client: TestClient, scene: Pixels, other_scene: Pixels) -> None:
    files = {"source": upload(scene), "reference": upload(other_scene)}
    response = client.post("/api/v1/match", files=files, data={"strength": "0"})
    assert response.status_code == 200
    settings = response.json()["settings"]
    assert all(value == 0 for edit in settings.values() for value in edit.values())


def test_preset_and_lut_downloads_then_apply(
    client: TestClient, service: PresetFitService, scene: Pixels
) -> None:
    settings = settings_with(service.pipeline, saturation={"saturation": -100.0}).to_dict()

    preset = client.post("/api/v1/preset", json={"settings": settings})
    assert preset.status_code == 200
    assert "attachment" in preset.headers["content-disposition"]
    assert json.loads(preset.text)["format"] == "presetfit-preset"

    lut = client.post("/api/v1/lut", json={"settings": settings, "title": "mono"})
    assert lut.status_code == 200
    assert lut.text.startswith('TITLE "mono"')

    applied = client.post(
        "/api/v1/apply", files={"source": upload(scene)}, data={"preset": preset.text}
    )
    assert applied.status_code == 200
    grey = decode_data_url(applied.json()["image"]).data
    assert np.abs(grey[..., 0] - grey[..., 2]).max() < 0.03  # black and white


@pytest.mark.parametrize(
    ("files", "data", "status", "code"),
    [
        ({"source": ("a.png", b"not an image", "image/png")}, {}, 400, "invalid_image"),
        ({"source": ("a.png", b"", "image/png")}, {}, 400, "invalid_image"),
        ({}, {}, 422, "invalid_request"),
        (None, {"strength": "7"}, 422, "invalid_request"),
    ],
    ids=["garbage", "empty", "missing-file", "bad-strength"],
)
def test_match_rejects_bad_requests(
    client: TestClient,
    scene: Pixels,
    files: dict[str, tuple[str, bytes, str]] | None,
    data: dict[str, str],
    status: int,
    code: str,
) -> None:
    good = {"source": upload(scene), "reference": upload(scene)}
    sent = good if files is None else ({**good, **files} if files else {"reference": upload(scene)})
    response = client.post("/api/v1/match", files=sent, data=data)
    assert response.status_code == status
    assert response.json()["error"] == code
    assert response.json()["detail"]


def test_unsupported_format_is_415(client: TestClient, scene: Pixels) -> None:
    gif = ("a.gif", encode(scene, "GIF"), "image/gif")
    response = client.post("/api/v1/match", files={"source": gif, "reference": upload(scene)})
    assert response.status_code == 415
    assert response.json()["error"] == "unsupported_image"


def test_upload_limits_are_enforced(service: PresetFitService, scene: Pixels) -> None:
    files = {"source": upload(scene), "reference": upload(scene)}
    tiny_bytes = TestClient(create_app(service, ApiSettings(max_upload_bytes=1000)))
    tiny_pixels = TestClient(create_app(service, ApiSettings(max_pixels=1000)))
    for limited in (tiny_bytes, tiny_pixels):
        response = limited.post("/api/v1/match", files=files)
        assert response.status_code == 413
        assert response.json()["error"] == "image_too_large"
        checked = limited.post("/api/v1/inspect", files={"photo": upload(scene)})
        assert checked.status_code == 413
        assert "limit" in checked.json()["detail"]


@pytest.mark.parametrize(
    "settings",
    [{"exposure": {"exposure": 99.0}}, {"sharpen": {"amount": 1.0}}],
    ids=["out-of-range", "unknown-edit"],
)
def test_invalid_settings_are_422(
    client: TestClient, settings: dict[str, dict[str, float]]
) -> None:
    for path in ("/api/v1/lut", "/api/v1/preset"):
        response = client.post(path, json={"settings": settings})
        assert response.status_code == 422
        assert response.json()["error"] == "invalid_settings"


def test_apply_rejects_a_bad_preset(client: TestClient, scene: Pixels) -> None:
    response = client.post("/api/v1/apply", files={"source": upload(scene)}, data={"preset": "{}"})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_settings"


def test_unknown_route_uses_the_standard_error_shape(client: TestClient) -> None:
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json() == {"error": "http_error", "detail": "Not Found"}


def test_unexpected_errors_do_not_leak_details(scene: Pixels) -> None:
    class BrokenService(PresetFitService):
        def match(self, source: Pixels, reference: Pixels, strength: float = 1.0):  # type: ignore[no-untyped-def]
            raise RuntimeError("secret internal detail")

    broken = BrokenService.create_default()
    client = TestClient(create_app(broken, ApiSettings()), raise_server_exceptions=False)
    response = client.post(
        "/api/v1/match", files={"source": upload(scene), "reference": upload(scene)}
    )
    assert response.status_code == 500
    assert response.json()["error"] == "internal_error"
    assert "secret" not in response.text


class TestApiSettings:
    def test_defaults(self) -> None:
        settings = ApiSettings.from_env({})
        assert settings.max_upload_bytes == 25 * 1024 * 1024
        assert settings.max_pixels == 40_000_000

    def test_reads_environment(self) -> None:
        settings = ApiSettings.from_env(
            {
                "PRESETFIT_MAX_UPLOAD_MB": "5",
                "PRESETFIT_MAX_MEGAPIXELS": "12",
                "PRESETFIT_JPEG_QUALITY": "80",
                "PRESETFIT_LOG_LEVEL": "debug",
            }
        )
        assert settings == ApiSettings(5 * 1024 * 1024, 12_000_000, 80, "DEBUG")

    @pytest.mark.parametrize("value", ["abc", "0", "-3"])
    def test_rejects_bad_numbers(self, value: str) -> None:
        with pytest.raises(ValueError, match="PRESETFIT_MAX_UPLOAD_MB"):
            ApiSettings.from_env({"PRESETFIT_MAX_UPLOAD_MB": value})

    def test_blank_values_fall_back_to_defaults(self) -> None:
        assert ApiSettings.from_env({"PRESETFIT_MAX_UPLOAD_MB": " "}) == ApiSettings()
