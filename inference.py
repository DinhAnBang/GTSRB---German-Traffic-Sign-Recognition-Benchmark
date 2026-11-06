"""Reusable ONNX inference helpers for the GTSRB traffic-sign classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


DEFAULT_IMAGE_SIZE = (128, 128)
DEFAULT_PADDING_COLOR = (114, 114, 114)
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_labels(labels_path: str | Path) -> dict[int, dict[str, str]]:
    """Load the class mapping stored in ``labels.json``."""

    import json

    raw_labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    labels: dict[int, dict[str, str]] = {}

    for raw_index, raw_value in raw_labels.items():
        index = int(raw_index)
        if isinstance(raw_value, str):
            labels[index] = {"en": raw_value, "vi": raw_value}
        else:
            labels[index] = {
                "en": str(raw_value.get("en", f"Class {index}")),
                "vi": str(raw_value.get("vi", raw_value.get("en", f"Lớp {index}"))),
            }

    return labels


def resize_with_letterbox(
    image: Image.Image,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    padding_color: tuple[int, int, int] = DEFAULT_PADDING_COLOR,
) -> Image.Image:
    """Resize an image without stretching it, then pad it to the model size."""

    target_width, target_height = size
    image = ImageOps.exif_transpose(image).convert("RGB")
    source_width, source_height = image.size

    if source_width <= 0 or source_height <= 0:
        raise ValueError("Ảnh không có kích thước hợp lệ.")

    scale = min(target_width / source_width, target_height / source_height)
    resized_size = (
        max(1, round(source_width * scale)),
        max(1, round(source_height * scale)),
    )
    resized = image.resize(resized_size, Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", size, padding_color)
    offset = (
        (target_width - resized.width) // 2,
        (target_height - resized.height) // 2,
    )
    canvas.paste(resized, offset)
    return canvas


def preprocess_image(
    image: Image.Image,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Convert a PIL image to an ONNX tensor with NCHW float32 layout."""

    prepared = resize_with_letterbox(image, size=size)
    array = np.asarray(prepared, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def _ensure_probabilities(scores: np.ndarray) -> np.ndarray:
    """Return a probability vector even if an export returns logits."""

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        raise ValueError("Model không trả về kết quả dự đoán.")

    score_sum = float(scores.sum())
    if np.all(scores >= 0) and np.isclose(score_sum, 1.0, atol=1e-3):
        return scores

    stable_scores = scores - np.max(scores)
    exponentials = np.exp(stable_scores)
    return exponentials / exponentials.sum()


def predict_image(
    session: ort.InferenceSession,
    image: Image.Image,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Run inference and return the highest-probability classes first."""

    input_meta = session.get_inputs()[0]
    tensor = preprocess_image(image, size=size)
    raw_output = session.run(None, {input_meta.name: tensor})[0]
    probabilities = _ensure_probabilities(raw_output)

    top_k = max(1, min(top_k, probabilities.size))
    indices = np.argsort(probabilities)[::-1][:top_k]
    return [
        {
            "class_id": int(index),
            "confidence": float(probabilities[index]),
        }
        for index in indices
    ]


def split_into_quadrants(image: Image.Image) -> list[dict[str, Any]]:
    """Split an image into four equal regions for classifier-based scanning."""

    prepared = ImageOps.exif_transpose(image).convert("RGB")
    width, height = prepared.size
    if width < 2 or height < 2:
        raise ValueError("Ảnh cần rộng và cao ít nhất 2 pixel để chia thành 4 vùng.")

    middle_x = width // 2
    middle_y = height // 2
    boxes = [
        ("Q1", (0, 0, middle_x, middle_y)),
        ("Q2", (middle_x, 0, width, middle_y)),
        ("Q3", (0, middle_y, middle_x, height)),
        ("Q4", (middle_x, middle_y, width, height)),
    ]
    return [
        {
            "name": name,
            "box": box,
            "image": prepared.crop(box),
        }
        for name, box in boxes
    ]


def iter_image_files(dataset_path: str | Path):
    """Yield ``(class_id, image_path)`` pairs from a numeric ImageFolder tree."""

    root = Path(dataset_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {root}")

    for class_dir in sorted(root.iterdir(), key=lambda item: item.name):
        if not class_dir.is_dir() or not class_dir.name.isdigit():
            continue

        class_id = int(class_dir.name)
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                yield class_id, image_path

