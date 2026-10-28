# -*- coding: utf-8 -*-
"""Flask backend that exposes the helmet detector to the web frontend.

Run it with::

    python app.py

The model is loaded once by the shared detector singleton and reused for
every request, so uploading images never triggers a reload or a retrain.
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from helmet_detection import (
    HelmetDetectionError,
    InvalidImageError,
    Settings,
    WeightsNotAvailableError,
    get_detector,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = int(os.environ.get("HELMET_MAX_UPLOAD_MB", "12")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SAMPLES_DIR = PROJECT_ROOT / "samples"


def _allowed(filename: str) -> bool:
    return Path(filename or "").suffix.lower() in ALLOWED_EXTENSIONS


def _asset_version() -> str:
    """Cache-buster for static assets, so edits are picked up without a hard refresh."""

    stamp = 0
    for name in ("css/style.css", "js/app.js"):
        path = PROJECT_ROOT / "static" / name
        if path.is_file():
            stamp = max(stamp, int(path.stat().st_mtime))
    return str(stamp)


def create_app(settings: Settings | None = None) -> Flask:
    """Build the Flask application."""

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["JSON_SORT_KEYS"] = False

    detector = get_detector(settings)
    asset_version = _asset_version()

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
            samples=_list_samples(),
            asset_version=asset_version,
        )

    @app.get("/api/health")
    def health():
        weights, source = detector.resolve_weights()
        error = None
        try:
            detector.load()
        except Exception as exc:
            error = str(exc)
        return jsonify(
            {
                "status": "ok" if error is None else "degraded",
                "model": detector.info.to_dict(),
                "weights_available": weights is not None,
                "weights_path": str(weights) if weights else None,
                "weights_source": source,
                "dataset_available": detector.settings.data_yaml.is_file(),
                "dataset_path": str(detector.settings.data_yaml),
                "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
                "auto_train": detector.settings.auto_train,
                "confidence": detector.settings.conf,
                "error": error,
                "samples": _list_samples(),
            }
        )

    @app.post("/api/detect")
    def detect():
        upload = request.files.get("image") or request.files.get("file")
        if upload is None or not upload.filename:
            return jsonify({"ok": False, "error": "No image was uploaded."}), 400
        if not _allowed(upload.filename):
            return jsonify(
                {
                    "ok": False,
                    "error": "Unsupported file type. Upload a JPG, PNG, WEBP or BMP image.",
                }
            ), 415

        payload = upload.read()
        if not payload:
            return jsonify({"ok": False, "error": "The uploaded file is empty."}), 400

        try:
            result = detector.predict(payload)
        except InvalidImageError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except WeightsNotAvailableError as exc:
            return jsonify({"ok": False, "error": str(exc), "code": "no_weights"}), 503
        except HelmetDetectionError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500
        except Exception as exc:  # pragma: no cover - defensive
            app.logger.exception("Detection failed")
            return jsonify({"ok": False, "error": f"Detection failed: {exc}"}), 500

        result["ok"] = True
        result["filename"] = upload.filename
        return jsonify(result)

    @app.get("/api/samples/<name>")
    def sample_file(name: str):
        samples_root = SAMPLES_DIR.resolve()
        candidate = (samples_root / name).resolve()
        try:
            candidate.relative_to(samples_root)
        except ValueError:
            return jsonify({"ok": False, "error": "Sample not found."}), 404
        if not candidate.is_file():
            return jsonify({"ok": False, "error": "Sample not found."}), 404
        return send_file(candidate)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        limit = MAX_UPLOAD_BYTES // (1024 * 1024)
        return jsonify({"ok": False, "error": f"Image is too large. The limit is {limit} MB."}), 413

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException):
        return jsonify({"ok": False, "error": error.description}), error.code or 500

    @app.errorhandler(Exception)
    def unhandled(error: Exception):  # pragma: no cover - defensive
        app.logger.exception("Unhandled error")
        return jsonify({"ok": False, "error": str(error)}), 500

    detector.warmup()
    return app


def _list_samples() -> list[dict[str, str]]:
    if not SAMPLES_DIR.is_dir():
        return []
    samples = []
    for path in sorted(SAMPLES_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            samples.append({"name": path.name, "url": f"/api/samples/{path.name}"})
    return samples


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("HELMET_HOST", "127.0.0.1")
    port = int(os.environ.get("HELMET_PORT", "8000"))
    debug = os.environ.get("HELMET_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)
