"""Command-line interface.

Run ``presetfit --help`` for usage. The CLI only parses arguments, calls
:class:`presetfit.service.PresetFitService` and prints; it contains no image
logic of its own.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from presetfit._version import __version__
from presetfit.exceptions import PresetFitError
from presetfit.export import CubeLutExporter
from presetfit.imaging import ImageReader, ImageWriter
from presetfit.report import TextReportFormatter
from presetfit.service import PresetFitService

EXIT_OK = 0
EXIT_FAILURE = 1


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="presetfit",
        description="Give one photo the look of another and list the edits that did it.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show progress details")
    commands = parser.add_subparsers(dest="command", required=True)

    match = commands.add_parser("match", help="copy the look of a reference photo onto a photo")
    match.add_argument("source", type=Path, help="the photo to edit")
    match.add_argument("reference", type=Path, help="the photo whose look to copy")
    match.add_argument("-o", "--output", type=Path, help="where to save the edited photo")
    match.add_argument("--strength", type=_strength, default=1.0, help="0-1, default 1")
    match.add_argument("--preset", type=Path, help="also save the edits as a JSON preset")
    match.add_argument("--lut", type=Path, help="also save the edits as a .cube LUT")
    match.add_argument("--report", type=Path, help="also save the text report to a file")
    match.add_argument("--json", action="store_true", help="print the result as JSON")
    match.set_defaults(handler=_run_match)

    apply = commands.add_parser("apply", help="apply a saved preset to a photo")
    apply.add_argument("source", type=Path, help="the photo to edit")
    apply.add_argument("--preset", type=Path, required=True, help="a preset saved by 'match'")
    apply.add_argument("-o", "--output", type=Path, help="where to save the edited photo")
    apply.set_defaults(handler=_run_apply)

    serve = commands.add_parser("serve", help="start the web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_run_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    try:
        return int(args.handler(args))
    except PresetFitError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_FAILURE
    except OSError as error:
        print(
            f"error: {error.strerror or error}: {error.filename or ''}".rstrip(": "),
            file=sys.stderr,
        )
        return EXIT_FAILURE


def _run_match(args: argparse.Namespace) -> int:
    service = PresetFitService.create_default()
    reader = ImageReader()
    result = service.match(
        reader.read(args.source), reader.read(args.reference), strength=args.strength
    )

    output = args.output or _default_output(args.source)
    ImageWriter().write(result.image, output)
    if args.preset:
        service.presets.save(result.settings, args.preset)
    if args.lut:
        CubeLutExporter(service.pipeline).save(result.settings, args.lut)
    report_text = TextReportFormatter().format(result.report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_text + "\n", encoding="utf-8")

    if args.json:
        payload = {
            "output": str(output),
            "settings": result.settings.to_dict(),
            "report": result.report.to_dict(),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(report_text)
        print(f"\nSaved {output}")
    return EXIT_OK


def _run_apply(args: argparse.Namespace) -> int:
    service = PresetFitService.create_default()
    settings = service.presets.load(args.preset)
    output = args.output or _default_output(args.source)
    ImageWriter().write(service.apply(ImageReader().read(args.source), settings), output)
    print(f"Saved {output}")
    return EXIT_OK


def _run_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "error: the web app needs extra packages; install them with "
            "'pip install \"presetfit[api]\"'",
            file=sys.stderr,
        )
        return EXIT_FAILURE
    uvicorn.run("presetfit.api.app:create_app", factory=True, host=args.host, port=args.port)
    return EXIT_OK


def _default_output(source: Path) -> Path:
    return source.with_name(f"{source.stem}_matched.jpg")


def _strength(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("strength must be a number") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("strength must be between 0 and 1")
    return value


if __name__ == "__main__":
    sys.exit(main())
