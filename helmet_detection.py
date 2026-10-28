# -*- coding: utf-8 -*-
"""Helmet detection core: model lifecycle, training and inference.

The module is import-safe. Importing it never trains a model, never reads
the network and never loads weights into memory. The YOLO weights are
resolved and loaded lazily on the first prediction and then cached for the
lifetime of the process, so a long running web server pays the load cost
exactly once instead of once per request.

Command line::

    python helmet_detection.py train   --data "<dataset>.yaml" [--force]
    python helmet_detection.py predict --image photo.jpg [--save annotated.jpg]
    python helmet_detection.py serve   --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = PROJECT_ROOT / "archive" / "coco128.yaml"
DEFAULT_PROJECT_DIR = PROJECT_ROOT / "runs"
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "best.pt"
FALLBACK_WEIGHTS = "yolov8n.pt"

HELMET_LABELS = ("with helmet", "helmet", "hard hat", "wearing helmet", "helmet on")
NO_HELMET_LABELS = (
    "without helmet",
    "no helmet",
    "not wearing helmet",
    "bare head",
    "helmet off",
)
PLATE_LABELS = ("number plate", "license plate", "numberplate", "plate")
PERSON_LABELS = ("person", "rider", "human", "worker", "man", "woman", "pedestrian")

VERDICT_PRESENT = "helmet_present"
VERDICT_MISSING = "helmet_missing"
VERDICT_NO_PERSON = "no_person"
VERDICT_UNDETERMINED = "undetermined"

VERDICT_COPY = {
    VERDICT_PRESENT: (
        "Helmet detected",
        "A helmet was found on at least one subject in this image.",
        "ok",
    ),
    VERDICT_MISSING: (
        "Helmet missing",
        "At least one subject in this image was detected without a helmet.",
        "alert",
    ),
    VERDICT_NO_PERSON: (
        "No subject found",
        "No rider or person was detected, so there is no head to judge.",
        "neutral",
    ),
    VERDICT_UNDETERMINED: (
        "Undetermined",
        "A person was detected but no helmet verdict could be derived.",
        "warn",
    ),
}


class HelmetDetectionError(Exception):
    """Base class for every recoverable error raised by this module."""


class InvalidImageError(HelmetDetectionError):
    """Raised when the uploaded payload cannot be decoded as an image."""


class WeightsNotAvailableError(HelmetDetectionError):
    """Raised when no usable weights file could be located."""


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, overridable through ``HELMET_*`` env vars."""

    weights_path: Path = DEFAULT_WEIGHTS_PATH
    data_yaml: Path = DEFAULT_DATA_YAML
    project_dir: Path = DEFAULT_PROJECT_DIR
    base_weights: str = FALLBACK_WEIGHTS
    epochs: int = 50
    imgsz: int = 640
    batch: int = 8
    conf: float = 0.25
    iou: float = 0.45
    device: str = ""
    auto_train: bool = True
    ocr_enabled: bool = True
    workers: int = 0

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> "Settings":
            """Build settings from environment variables, falling back to defaults."""

            root = Path(base_dir).resolve() if base_dir else PROJECT_ROOT
            env = os.environ

            def _path(name: str, default: Path) -> Path:
                raw = env.get(name, "").strip()
                if not raw:
                    return default
                candidate = Path(raw).expanduser()
                return candidate if candidate.is_absolute() else (root / candidate)

            def _int(name: str, default: int) -> int:
                raw = env.get(name, "").strip()
                try:
                    return int(raw) if raw else default
                except ValueError:
                    return default

            def _float(name: str, default: float) -> float:
                raw = env.get(name, "").strip()
                try:
                    return float(raw) if raw else default
                except ValueError:
                    return default

            def _bool(name: str, default: bool) -> bool:
                raw = env.get(name, "").strip().lower()
                if not raw:
                    return default
                return raw in {"1", "true", "yes", "on"}

            return cls(
                weights_path=_path("HELMET_WEIGHTS", root / "weights" / "best.pt"),
                data_yaml=_path("HELMET_DATA", root / "archive" / "coco128.yaml"),
                project_dir=_path("HELMET_PROJECT", root / "runs"),
                base_weights=env.get("HELMET_BASE_WEIGHTS", FALLBACK_WEIGHTS).strip() or FALLBACK_WEIGHTS,
                epochs=_int("HELMET_EPOCHS", 50),
                imgsz=_int("HELMET_IMGSZ", 640),
                batch=_int("HELMET_BATCH", 8),
                conf=_float("HELMET_CONF", 0.25),
                iou=_float("HELMET_IOU", 0.45),
                device=env.get("HELMET_DEVICE", "").strip(),
                auto_train=_bool("HELMET_AUTO_TRAIN", True),
                ocr_enabled=_bool("HELMET_OCR", True),
                workers=_int("HELMET_WORKERS", 0),
            )


@dataclass
class ModelInfo:
    """Everything the frontend needs to know about the active model."""

    source: str = "unloaded"
    weights: str = ""
    classes: list[str] = field(default_factory=list)
    supports_helmet: bool = False
    state: str = "unloaded"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "weights": self.weights,
            "classes": list(self.classes),
            "supports_helmet": self.supports_helmet,
            "state": self.state,
            "note": self.note,
        }


def decode_image(payload: bytes) -> np.ndarray | None:
    """Decode raw bytes into a BGR image, tolerating non-ASCII file names."""

    if not payload:
        return None
    buffer = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image


def encode_jpeg(image: np.ndarray, quality: int = 88) -> bytes:
    """Encode a BGR image as JPEG bytes."""

    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise HelmetDetectionError("Failed to encode the annotated image.")
    return buffer.tobytes()


def to_data_url(image: np.ndarray, quality: int = 88) -> str:
    """Encode a BGR image as an inline JPEG data URL."""

    encoded = base64.b64encode(encode_jpeg(image, quality)).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _normalize(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


def is_helmet_label(label: str) -> bool:
    text = _normalize(label)
    return any(text == item or item in text for item in HELMET_LABELS)


def is_no_helmet_label(label: str) -> bool:
    text = _normalize(label)
    return any(text == item or item in text for item in NO_HELMET_LABELS)


def is_plate_label(label: str) -> bool:
    text = _normalize(label)
    return any(text == item or item in text for item in PLATE_LABELS)


def is_person_label(label: str) -> bool:
    text = _normalize(label)
    return any(text == item or item in text for item in PERSON_LABELS)


def categorize(label: str) -> str:
    """Bucket a raw class name into a stable category used by the UI."""

    if is_no_helmet_label(label):
        return "without_helmet"
    if is_helmet_label(label):
        return "with_helmet"
    if is_plate_label(label):
        return "number_plate"
    if is_person_label(label):
        return "person"
    return "other"


class HelmetDetector:
    """Loads the model once and serves predictions from the cached instance."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self._model: Any | None = None
        self._lock = threading.Lock()
        self._ocr_reader: Any | None = None
        self._ocr_failed = False
        self.info = ModelInfo()

    # -- model lifecycle ------------------------------------------------

    def _candidate_weights(self) -> list[Path]:
        candidates = [
            self.settings.weights_path,
            PROJECT_ROOT / "weights" / "best.pt",
            PROJECT_ROOT / "weights" / "last.pt",
            PROJECT_ROOT / "runs" / "detect" / "custom_model" / "weights" / "best.pt",
            PROJECT_ROOT / "runs" / "custom_model_local" / "weights" / "best.pt",
        ]
        candidates.extend(sorted(PROJECT_ROOT.glob("runs/**/weights/best.pt")))
        unique: list[Path] = []
        for path in candidates:
            resolved = path.expanduser()
            if resolved not in unique:
                unique.append(resolved)
        return unique

    def resolve_weights(self) -> tuple[Path | None, str]:
        """Return ``(weights_path, source)`` for the best available model."""

        for candidate in self._candidate_weights():
            if candidate.is_file():
                return candidate, "trained"
        return None, "missing"

    def load(self) -> Any:
        """Load (and cache) the YOLO model. Safe to call from many threads."""

        with self._lock:
            if self._model is not None:
                return self._model

            from ultralytics import YOLO

            weights, source = self.resolve_weights()
            if weights is None:
                if self.settings.auto_train and self._dataset_available():
                    weights = self.train(force=False)
                    source = "trained"
                else:
                    weights = self._resolve_base_weights()
                    source = "base"

            self._model = YOLO(str(weights))
            names = self._model.names
            classes = [str(names[key]) for key in sorted(names)] if isinstance(names, dict) else [str(n) for n in names]
            supports_helmet = any(is_helmet_label(c) or is_no_helmet_label(c) for c in classes)
            note = ""
            if source == "base":
                note = (
                    "Stock detector in use - it has no helmet classes. Train a helmet "
                    "model with `python helmet_detection.py train --data <dataset>.yaml` "
                    "for real verdicts."
                )
            elif not supports_helmet:
                note = "The loaded weights do not expose helmet classes."
            self.info = ModelInfo(
                source=source,
                weights=str(weights),
                classes=classes,
                supports_helmet=supports_helmet,
                state="ready",
                note=note,
            )
            return self._model

    def warmup(self) -> None:
        """Load the model in the background so the first request is fast."""

        def _run() -> None:
            try:
                self.load()
            except Exception:
                pass

        threading.Thread(target=_run, name="helmet-model-warmup", daemon=True).start()

    def _dataset_available(self) -> bool:
        return self.settings.data_yaml.is_file()

    def _resolve_base_weights(self) -> Path:
        """Locate the stock detector used when no trained weights exist yet."""

        candidate = Path(self.settings.base_weights).expanduser()
        if candidate.is_file():
            return candidate
        local = PROJECT_ROOT / self.settings.base_weights
        if local.is_file():
            return local
        return candidate

    def train(self, force: bool = False) -> Path:
        """Train the detector and persist the best weights for later reuse."""

        weights = self.settings.weights_path.expanduser()
        if weights.is_file() and not force:
            print(f"Weights already present at {weights} - skipping training.")
            return weights
        if not self._dataset_available():
            raise HelmetDetectionError(
                f"Dataset yaml not found at {self.settings.data_yaml}. "
                "Pass --data with the path to your coco128.yaml."
            )

        from ultralytics import YOLO

        weights.parent.mkdir(parents=True, exist_ok=True)
        self.settings.project_dir.mkdir(parents=True, exist_ok=True)
        print(f"Training from {self.settings.base_weights} for {self.settings.epochs} epochs...")
        model = YOLO(self.settings.base_weights)
        model.train(
            data=str(self.settings.data_yaml),
            epochs=self.settings.epochs,
            imgsz=self.settings.imgsz,
            batch=self.settings.batch,
            project=str(self.settings.project_dir),
            name="custom_model",
            exist_ok=True,
        )
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        if not best.is_file():
            raise HelmetDetectionError(f"Training finished but {best} was not produced.")
        shutil.copy2(best, weights)
        print(f"Weights saved to {weights}")
        with self._lock:
            self._model = None
            self.info = ModelInfo()
        return weights

    # -- inference ------------------------------------------------------

    def _get_ocr_reader(self) -> Any | None:
        if self._ocr_failed or not self.settings.ocr_enabled:
            return None
        if self._ocr_reader is not None:
            return self._ocr_reader
        try:
            import easyocr

            self._ocr_reader = easyocr.Reader(["en"], verbose=False)
        except Exception:
            self._ocr_failed = True
            self._ocr_reader = None
        return self._ocr_reader

    def _read_plate(self, crop: np.ndarray) -> tuple[str, float]:
        reader = self._get_ocr_reader()
        if reader is None or crop.size == 0:
            return "", 0.0
        try:
            readings = reader.readtext(crop)
        except Exception:
            return "", 0.0
        if not readings:
            return "", 0.0
        text = " ".join(str(item[1]) for item in readings).strip()
        confidence = sum(float(item[2]) for item in readings) / len(readings)
        return text, round(confidence, 4)

    def predict(self, payload: bytes, annotate: bool = True) -> dict[str, Any]:
        """Run detection on raw image bytes and return a JSON-safe payload."""

        image = decode_image(payload)
        if image is None:
            raise InvalidImageError("The uploaded file could not be decoded as an image.")

        model = self.load()
        predict_kwargs: dict[str, Any] = {
            "source": image,
            "conf": self.settings.conf,
            "iou": self.settings.iou,
            "imgsz": self.settings.imgsz,
            "verbose": False,
        }
        if self.settings.device:
            predict_kwargs["device"] = self.settings.device
        if self.settings.workers:
            predict_kwargs["workers"] = self.settings.workers

        started = time.perf_counter()
        results = model.predict(**predict_kwargs)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        result = results[0] if results else None
        detections: list[dict[str, Any]] = []
        plates: list[dict[str, Any]] = []

        if result is not None and getattr(result, "boxes", None) is not None and len(result.boxes):
            names = result.names
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            for index in range(len(classes)):
                label = str(names[int(classes[index])])
                x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy[index][:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = max(x1, x2), max(y1, y2)
                detection = {
                    "label": label,
                    "category": categorize(label),
                    "confidence": round(float(confidences[index]), 4),
                    "bbox": [x1, y1, x2, y2],
                    "width": x2 - x1,
                    "height": y2 - y1,
                }
                detections.append(detection)
                if is_plate_label(label):
                    text, text_confidence = self._read_plate(image[y1:y2, x1:x2])
                    plates.append({"text": text, "confidence": text_confidence, "bbox": [x1, y1, x2, y2]})

        summary = {
            "total": len(detections),
            "with_helmet": sum(1 for d in detections if d["category"] == "with_helmet"),
            "without_helmet": sum(1 for d in detections if d["category"] == "without_helmet"),
            "persons": sum(1 for d in detections if d["category"] in {"person", "without_helmet", "with_helmet"}),
            "plates": sum(1 for d in detections if d["category"] == "number_plate"),
        }

        verdict, label, detail, severity = self._verdict(summary, detections)
        payload_out: dict[str, Any] = {
            "verdict": verdict,
            "verdict_label": label,
            "verdict_detail": detail,
            "severity": severity,
            "summary": summary,
            "detections": detections,
            "plates": plates,
            "model": self.info.to_dict(),
            "elapsed_ms": elapsed_ms,
            "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        }
        if annotate:
            payload_out["annotated_image"] = to_data_url(result.plot() if result is not None else image)
        return payload_out

    def _verdict(self, summary: dict[str, int], detections: list[dict[str, Any]]) -> tuple[str, str, str, str]:
        if summary["without_helmet"] > 0:
            verdict = VERDICT_MISSING
        elif summary["with_helmet"] > 0:
            verdict = VERDICT_PRESENT
        elif summary["persons"] > 0:
            verdict = VERDICT_UNDETERMINED
        else:
            verdict = VERDICT_NO_PERSON

        label, detail, severity = VERDICT_COPY[verdict]
        if verdict == VERDICT_UNDETERMINED and not self.info.supports_helmet:
            detail = (
                "The active model does not expose helmet classes, so no headwear "
                "verdict can be produced. Train or load a helmet-specific model."
            )
            severity = "warn"
        return verdict, label, detail, severity


_detector: HelmetDetector | None = None
_detector_lock = threading.Lock()


def get_detector(settings: Settings | None = None) -> HelmetDetector:
    """Return the process-wide detector singleton."""

    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = HelmetDetector(settings)
        elif settings is not None:
            _detector.settings = settings
        return _detector


def predict(payload: bytes, annotate: bool = True, settings: Settings | None = None) -> dict[str, Any]:
    """Convenience wrapper around the cached detector."""

    return get_detector(settings).predict(payload, annotate=annotate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Helmet detection core")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the detector and cache the weights")
    train_parser.add_argument("--data", default=None, help="Path to the dataset yaml")
    train_parser.add_argument("--weights", default=None, help="Destination for best.pt")
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--imgsz", type=int, default=None)
    train_parser.add_argument("--batch", type=int, default=None)
    train_parser.add_argument("--force", action="store_true", help="Retrain even if weights exist")

    predict_parser = subparsers.add_parser("predict", help="Run detection on a local image")
    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument("--save", default=None, help="Write the annotated image here")
    predict_parser.add_argument("--json", dest="as_json", action="store_true")

    serve_parser = subparsers.add_parser("serve", help="Start the web application")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--debug", action="store_true")

    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if args.command == "train":
        overrides: dict[str, Any] = {}
        if args.data:
            overrides["data_yaml"] = Path(args.data).expanduser()
        if args.weights:
            overrides["weights_path"] = Path(args.weights).expanduser()
        if args.epochs:
            overrides["epochs"] = args.epochs
        if args.imgsz:
            overrides["imgsz"] = args.imgsz
        if args.batch:
            overrides["batch"] = args.batch
        detector = HelmetDetector(replace(settings, **overrides))
        detector.train(force=args.force)
        return 0

    if args.command == "predict":
        image_path = Path(args.image).expanduser()
        payload = image_path.read_bytes()
        result = predict(payload)
        if args.save:
            if "annotated_image" in result:
                encoded = result["annotated_image"].split(",", 1)[1]
                Path(args.save).write_bytes(base64.b64decode(encoded))
                print(f"Annotated image written to {args.save}")
        if args.as_json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Verdict: {result['verdict_label']} - {result['verdict_detail']}")
            for detection in result["detections"]:
                print(f"  {detection['label']:<16} {detection['confidence']:.2f}  {detection['bbox']}")
        return 0

    if args.command == "serve":
        from app import create_app

        create_app(settings).run(host=args.host, port=args.port, debug=args.debug)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
