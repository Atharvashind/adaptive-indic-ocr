"""Verified adapter for the recovered multilingual CRNN-CTC checkpoint."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageEnhance
from torch import nn
from torch.nn import functional as torch_functional

from src.models.base_adapter import BaseOCRAdapter, ImageInput

EXPECTED_ARCHITECTURE = "multilingual_crnn_v1"
EXPECTED_CHECKPOINT_SHA256 = "9a63c6ce29b37fdbc1b9181e5fb494d501351ebc916a1ed3189f0819c63fea10"
EXPECTED_VOCABULARY_SHA256 = "5220a159757d92bccb69e447dd1f2e3ecb30d6bdcca55538c7fe429f68822d5f"
EXPECTED_PARAMETER_COUNT = 6_095_128
EXPECTED_OUTPUT_CLASSES = 280
BLANK_TOKEN = "<BLANK>"


class BLSTM(nn.Module):
    def __init__(self, nIn: int, nH: int, nOut: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(nIn, nH, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(nH * 2, nOut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.rnn(x)
        return self.fc(x)


class RecoveredCRNN(nn.Module):
    """CRNN architecture reproduced from original notebook cell 94."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.ReLU(),
        )
        self.rnn = nn.Sequential(
            BLSTM(512, 256, 256),
            BLSTM(256, 256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.rnn(x)
        return x.permute(1, 0, 2)


@dataclass(frozen=True)
class PreparedCRNNBatch:
    """Model-native batch and its sort/padding metadata."""

    images: torch.Tensor
    content_widths: tuple[int, ...]
    input_lengths: tuple[int, ...]
    original_indices: tuple[int, ...]
    padded_width: int


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_artifact(path: Path, member: str | None) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"Configured model artifact does not exist: {path}")
    if member is None:
        return path.read_bytes()
    try:
        with zipfile.ZipFile(path, "r") as archive:
            return archive.read(member)
    except KeyError as error:
        raise FileNotFoundError(
            f"Artifact member {member!r} is missing from {path}"
        ) from error
    except zipfile.BadZipFile as error:
        raise ValueError(f"Configured artifact is not a valid ZIP file: {path}") from error


def _verify_artifact_hash(data: bytes, expected_hash: str, label: str) -> None:
    actual_hash = _sha256(data)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_hash}, found {actual_hash}"
        )


class CRNNAdapter(BaseOCRAdapter):
    """CPU prediction adapter preserving the recovered CRNN inference behavior."""

    def __init__(
        self,
        configuration: Mapping[str, Any],
        project_root: Path,
    ) -> None:
        self.configuration = dict(configuration)
        self.project_root = project_root.resolve()
        self._validate_configuration()

        self.input_height = int(self.configuration["input_height"])
        self.contrast_factor = float(self.configuration["contrast_factor"])
        self.blank_index = int(self.configuration["blank_index"])
        self.batch_size = int(self.configuration["batch_size"])
        self.device = torch.device("cpu")

        vocabulary_data = self._load_configured_artifact("vocabulary")
        _verify_artifact_hash(
            vocabulary_data,
            str(self.configuration["vocabulary_sha256"]),
            "Vocabulary",
        )
        self.vocabulary = self._validate_vocabulary(
            json.loads(vocabulary_data.decode("utf-8"))
        )
        self.index_to_character = {
            index: character for character, index in self.vocabulary.items()
        }

        checkpoint_data = self._load_configured_artifact("checkpoint")
        _verify_artifact_hash(
            checkpoint_data,
            str(self.configuration["checkpoint_sha256"]),
            "Checkpoint",
        )
        state_dict = torch.load(
            io.BytesIO(checkpoint_data),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state_dict, Mapping) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state_dict.items()
        ):
            raise TypeError("Recovered checkpoint must contain a raw tensor state dictionary")

        self.model = RecoveredCRNN(num_classes=len(self.vocabulary)).to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        if parameter_count != EXPECTED_PARAMETER_COUNT:
            raise ValueError(
                f"Unexpected CRNN parameter count {parameter_count}; "
                f"expected {EXPECTED_PARAMETER_COUNT}"
            )
        if self.model.training:
            raise RuntimeError("CRNN failed to enter evaluation mode")
        self.checkpoint_sha256 = _sha256(checkpoint_data)
        self.vocabulary_sha256 = _sha256(vocabulary_data)

    @classmethod
    def from_config_file(
        cls,
        name: str = "crnn",
        config_path: Path | None = None,
        project_root: Path | None = None,
    ) -> "CRNNAdapter":
        root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        selected_config = config_path or root / "configs" / "models.yaml"
        if not selected_config.is_absolute():
            selected_config = root / selected_config
        if not selected_config.is_file():
            raise FileNotFoundError(f"Model configuration not found: {selected_config}")
        loaded = yaml.safe_load(selected_config.read_text(encoding="utf-8"))
        model_config = loaded.get("models", {}).get(name) if isinstance(loaded, dict) else None
        if not isinstance(model_config, dict):
            raise ValueError(f"Model {name!r} is not configured in {selected_config}")
        return cls(model_config, root)

    def _validate_configuration(self) -> None:
        expected = {
            "architecture": EXPECTED_ARCHITECTURE,
            "input_height": 32,
            "contrast_factor": 2.0,
            "blank_index": 0,
            "output_classes": EXPECTED_OUTPUT_CLASSES,
            "device": "cpu",
            "decoder": "greedy_ctc",
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "vocabulary_sha256": EXPECTED_VOCABULARY_SHA256,
        }
        for key, expected_value in expected.items():
            if self.configuration.get(key) != expected_value:
                raise ValueError(
                    f"Unsupported or unverified CRNN configuration {key!r}: "
                    f"{self.configuration.get(key)!r}; expected {expected_value!r}"
                )
        batch_size = self.configuration.get("batch_size")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("CRNN batch_size must be a positive integer")
        if not self.configuration.get("checkpoint_path"):
            raise ValueError("CRNN checkpoint_path is required")
        if not self.configuration.get("vocabulary_path"):
            raise ValueError("CRNN vocabulary_path is required")

    def _load_configured_artifact(self, key: str) -> bytes:
        path = Path(str(self.configuration[f"{key}_path"])).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        member = self.configuration.get(f"{key}_member")
        return _read_artifact(path.resolve(), str(member) if member else None)

    @staticmethod
    def _validate_vocabulary(value: Any) -> dict[str, int]:
        if not isinstance(value, dict) or not value:
            raise TypeError("Vocabulary must be a non-empty character-to-index object")
        if any(
            not isinstance(character, str)
            or not isinstance(index, int)
            or isinstance(index, bool)
            for character, index in value.items()
        ):
            raise TypeError("Vocabulary keys must be strings and values must be integer indices")
        if value.get(BLANK_TOKEN) != 0:
            raise ValueError("Vocabulary must map <BLANK> to index 0")
        indices = list(value.values())
        if len(set(indices)) != len(indices) or sorted(indices) != list(range(len(value))):
            raise ValueError("Vocabulary indices must be unique and contiguous from zero")
        if len(value) != EXPECTED_OUTPUT_CLASSES:
            raise ValueError(
                f"Vocabulary has {len(value)} classes; expected {EXPECTED_OUTPUT_CLASSES}"
            )
        return dict(value)

    def preprocess_image(self, image: ImageInput) -> tuple[torch.Tensor, int]:
        image_path = Path(image)
        try:
            with Image.open(image_path) as source:
                grayscale = source.convert("L")
        except (OSError, ValueError) as error:
            raise OSError(f"Unable to read CRNN input image {image_path}: {error}") from error

        original_width, original_height = grayscale.size
        if original_width <= 0 or original_height <= 0:
            raise ValueError(f"Invalid image dimensions for {image_path}")
        content_width = int(round(original_width * self.input_height / original_height))
        if content_width < 1:
            raise ValueError(f"Resized width is invalid for {image_path}")
        resized = grayscale.resize(
            (content_width, self.input_height),
            Image.Resampling.BILINEAR,
        )
        enhanced = ImageEnhance.Contrast(resized).enhance(self.contrast_factor)
        image_tensor = torch.tensor(
            np.array(enhanced),
            dtype=torch.float32,
            device=self.device,
        ) / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5
        return image_tensor.unsqueeze(0), content_width

    def prepare_batch(self, images: list[ImageInput]) -> PreparedCRNNBatch:
        if not images:
            raise ValueError("Cannot prepare an empty CRNN batch")
        if len(images) > self.batch_size:
            raise ValueError(
                f"CRNN batch contains {len(images)} images; configured maximum is {self.batch_size}"
            )
        prepared = [
            (index, *self.preprocess_image(image))
            for index, image in enumerate(images)
        ]
        prepared.sort(key=lambda item: item[2], reverse=True)
        max_width = prepared[0][2]
        padded_width = ((max_width + 3) // 4) * 4
        padded_images = [
            torch_functional.pad(
                tensor,
                (0, padded_width - width),
                value=0.0,
            )
            for _, tensor, width in prepared
        ]
        widths = tuple(width for _, _, width in prepared)
        return PreparedCRNNBatch(
            images=torch.stack(padded_images),
            content_widths=widths,
            input_lengths=tuple((width // 4) - 1 for width in widths),
            original_indices=tuple(index for index, _, _ in prepared),
            padded_width=padded_width,
        )

    def decode(self, predictions: torch.Tensor) -> list[str]:
        if predictions.ndim != 3 or predictions.shape[-1] != len(self.vocabulary):
            raise ValueError(
                f"Expected CRNN logits of shape (time, batch, {len(self.vocabulary)}), "
                f"received {tuple(predictions.shape)}"
            )
        if not bool(torch.isfinite(predictions).all().item()):
            raise ValueError("CRNN output contains NaN or infinite values")
        predicted_indices = predictions.permute(1, 0, 2).argmax(2).cpu().tolist()
        decoded: list[str] = []
        for sequence in predicted_indices:
            characters: list[str] = []
            previous = -1
            for class_index in sequence:
                if class_index != self.blank_index and class_index != previous:
                    try:
                        characters.append(self.index_to_character[class_index])
                    except KeyError as error:
                        raise ValueError(
                            f"Model emitted class absent from vocabulary: {class_index}"
                        ) from error
                previous = class_index
            decoded.append("".join(characters))
        return decoded

    def predict(self, images: list[ImageInput]) -> list[str]:
        if not images:
            return []
        decoded_in_input_order: list[str | None] = [None] * len(images)
        for start in range(0, len(images), self.batch_size):
            batch_paths = images[start : start + self.batch_size]
            prepared = self.prepare_batch(batch_paths)
            with torch.inference_mode():
                logits = self.model(prepared.images)
            expected_time = (prepared.padded_width // 4) - 1
            expected_shape = (expected_time, len(batch_paths), len(self.vocabulary))
            if tuple(logits.shape) != expected_shape:
                raise ValueError(
                    f"Unexpected CRNN output shape {tuple(logits.shape)}; "
                    f"expected {expected_shape}"
                )
            batch_predictions = self.decode(logits)
            for original_index, prediction in zip(
                prepared.original_indices,
                batch_predictions,
                strict=True,
            ):
                decoded_in_input_order[start + original_index] = prediction
        if any(prediction is None for prediction in decoded_in_input_order):
            raise RuntimeError("CRNN failed to produce a prediction for every input image")
        return [prediction for prediction in decoded_in_input_order if prediction is not None]
