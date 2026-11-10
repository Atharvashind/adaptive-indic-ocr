from __future__ import annotations

import hashlib
import json
import platform
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
RECOVERY_DIR = ROOT / "experiments" / "ocr_evaluation" / "recovery"
ARCHIVE_PATH = (
    ROOT
    / "experiments"
    / "recovered_baseline"
    / "Multilingual_CRNN_OCR-20261105T161607Z-1-001.zip"
)
ARCHIVE_SHA256 = "a5b9568a0bad28b120186b33211451c23013658d268e7c2c95a25029b1d88194"
CHECKPOINTS = {
    "best_model_epoch4.pt": {
        "member": "Multilingual_CRNN_OCR/Model_Weights/best_model_epoch4.pt",
        "sha256": "9a63c6ce29b37fdbc1b9181e5fb494d501351ebc916a1ed3189f0819c63fea10",
        "size_bytes": 24398525,
    },
    "final_model_epoch5.pt": {
        "member": "Multilingual_CRNN_OCR/Model_Weights/final_model_epoch5.pt",
        "sha256": "9ff4b4ae2143621893f165df41c44bbe9fbee6b81fd57297797e5ebe3d58dac8",
        "size_bytes": 24398661,
    },
}
JSON_PATH = RECOVERY_DIR / "checkpoint_load_test.json"
MARKDOWN_PATH = RECOVERY_DIR / "checkpoint_load_test.md"
INSTALL_COMMAND = (
    r".\.venv\Scripts\python.exe -m pip install torch==2.15.0+cpu "
    r"--index-url https://download.pytorch.org/whl/cpu "
    r"--extra-index-url https://pypi.org/simple --only-binary=:all:"
)
PIP_UPGRADE_COMMAND = r".\.venv\Scripts\python.exe -m pip install --upgrade pip"
INITIAL_INSTALL_COMMAND = (
    r".\.venv\Scripts\python.exe -m pip install torch==2.15.0+cpu "
    r"--index-url https://download.pytorch.org/whl/cpu"
)
RUN_COMMAND = (
    r".\.venv\Scripts\python.exe "
    r".\experiments\ocr_evaluation\recovery\run_checkpoint_load_test.py"
)


class BLSTM(nn.Module):
    def __init__(self, nIn: int, nH: int, nOut: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(nIn, nH, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(nH * 2, nOut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.rnn(x)
        return self.fc(x)


class CRNN(nn.Module):
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_state_dict(loaded: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Checkpoint root must be a mapping, got {type(loaded).__name__}")

    if all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in loaded.items()):
        return loaded

    for key in ("state_dict", "model_state_dict"):
        candidate = loaded.get(key)
        if isinstance(candidate, Mapping) and all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in candidate.items()
        ):
            return candidate

    raise TypeError("Checkpoint mapping contains no raw or wrapped tensor state dictionary")


def load_and_test_checkpoint(name: str, checkpoint_path: Path) -> dict[str, Any]:
    expected = CHECKPOINTS[name]
    size_bytes = checkpoint_path.stat().st_size
    checkpoint_hash_before = sha256_file(checkpoint_path)
    if size_bytes != expected["size_bytes"] or checkpoint_hash_before != expected["sha256"]:
        raise ValueError(f"{name} bytes do not match the previously audited checkpoint")

    loaded = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    state_dict = get_state_dict(loaded)
    model = CRNN(num_classes=280).to("cpu")
    incompatible = model.load_state_dict(state_dict, strict=True)
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 6_095_128:
        raise ValueError(f"Unexpected model parameter count: {parameter_count}")
    with torch.inference_mode():
        output = model(torch.zeros((1, 1, 32, 128), dtype=torch.float32, device="cpu"))
    expected_shape = (31, 1, 280)
    if tuple(output.shape) != expected_shape:
        raise ValueError(f"{name} smoke output shape {tuple(output.shape)} != {expected_shape}")
    if output.shape[-1] != 280:
        raise ValueError(f"{name} smoke output class dimension is {output.shape[-1]}, expected 280")
    finite_output = bool(torch.isfinite(output).all().item())
    if not finite_output:
        raise ValueError(f"{name} smoke output contains NaN or infinite values")

    checkpoint_hash_after = sha256_file(checkpoint_path)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise ValueError(f"{name} checkpoint bytes changed during the test")

    return {
        "name": name,
        "archive_member": expected["member"],
        "file_exists": True,
        "file_size_bytes": size_bytes,
        "sha256_before": checkpoint_hash_before,
        "sha256_after": checkpoint_hash_after,
        "cpu_deserialization": "passed (torch.load weights_only=True, map_location='cpu')",
        "state_dict_container": type(loaded).__name__,
        "state_dict_key_count": len(state_dict),
        "strict_load": "passed",
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "parameter_count": parameter_count,
        "output_dimension": int(output.shape[-1]),
        "model_device": str(next(model.parameters()).device),
        "model_training_mode": model.training,
        "forward_smoke_test": {
            "input_shape": [1, 1, 32, 128],
            "output_shape": list(output.shape),
            "output_class_dimension": int(output.shape[-1]),
            "finite_output": finite_output,
        },
    }


def markdown_report(report: dict[str, Any]) -> str:
    env = report["environment"]
    lines = [
        "# Safe CPU checkpoint loading test",
        "",
        f"- **Status:** {report['status']}",
        f"- **Tested at (UTC):** {report['tested_at_utc']}",
        f"- **Python:** {env['python_version']} (`{env['python_executable']}`)",
        f"- **Platform:** {env['platform']}",
        f"- **PyTorch:** {env['torch_version']}",
        f"- **CUDA available:** {env['cuda_available']}",
        f"- **Installation command:** `{report['installation']['command']}`",
        f"- **pip compatibility step:** `{report['installation']['pip_upgrade_command']}`",
        f"- **Initial install attempt:** `{report['installation']['initial_attempt_command']}` "
        f"({report['installation']['initial_attempt_outcome']})",
        f"- **Execution command:** `{report['execution_command']}`",
        "",
        "## Source and archive integrity",
        "",
        f"- Archive: `{report['archive']['path']}`",
        f"- Archive size: {report['archive']['size_bytes']:,} bytes",
        f"- Archive SHA-256 before: `{report['archive']['sha256_before']}`",
        f"- Archive SHA-256 after: `{report['archive']['sha256_after']}`",
        f"- Archive unchanged: {report['archive']['unchanged']}",
        "- Checkpoints were read from their exact ZIP members into a temporary directory; the archive was not extracted or modified.",
        "",
        "## Results",
        "",
    ]
    for checkpoint in report["checkpoints"]:
        lines.extend(
            [
                f"### {checkpoint['name']}",
                "",
                f"- Archive member: `{checkpoint['archive_member']}`",
                f"- Exists / size: {checkpoint['file_exists']} / {checkpoint['file_size_bytes']:,} bytes",
                f"- SHA-256 before / after: `{checkpoint['sha256_before']}` / `{checkpoint['sha256_after']}`",
                f"- CPU deserialization: {checkpoint['cpu_deserialization']}",
                f"- State dictionary: {checkpoint['state_dict_container']}, {checkpoint['state_dict_key_count']} keys",
                f"- Strict load: {checkpoint['strict_load']}",
                f"- Missing / unexpected keys: {checkpoint['missing_keys']} / {checkpoint['unexpected_keys']}",
                f"- Parameters / output classes: {checkpoint['parameter_count']:,} / {checkpoint['output_dimension']}",
                f"- Device / evaluation mode: {checkpoint['model_device']} / {not checkpoint['model_training_mode']}",
                f"- Forward smoke test: input `{checkpoint['forward_smoke_test']['input_shape']}`, output `{checkpoint['forward_smoke_test']['output_shape']}`, finite={checkpoint['forward_smoke_test']['finite_output']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope",
            "",
            "This validates safe CPU deserialization, strict state-dictionary compatibility, and a synthetic architecture forward pass only. It does not initialize an optimizer, train, access the test set, decode text, or compute OCR metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    if torch.version.cuda is not None:
        raise RuntimeError(f"Expected a CPU-only PyTorch build, got CUDA build {torch.version.cuda}")
    if JSON_PATH.exists() or MARKDOWN_PATH.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing checkpoint-load report; archive or move it first."
        )
    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint archive not found: {ARCHIVE_PATH}")

    archive_hash_before = sha256_file(ARCHIVE_PATH)
    if archive_hash_before != ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 differs from the previously audited archive")

    report: dict[str, Any] = {
        "status": "in_progress",
        "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        "installation": {
            "command": INSTALL_COMMAND,
            "pip_upgrade_command": PIP_UPGRADE_COMMAND,
            "initial_attempt_command": INITIAL_INSTALL_COMMAND,
            "initial_attempt_outcome": (
                "failed before installing torch: pip 22.3.1 rejected normalized dependency "
                "metadata and attempted an unavailable source build; no torch package was installed"
            ),
            "index": "https://download.pytorch.org/whl/cpu",
            "cuda_wheel": False,
        },
        "execution_command": RUN_COMMAND,
        "environment": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "archive": {
            "path": str(ARCHIVE_PATH),
            "size_bytes": ARCHIVE_PATH.stat().st_size,
            "sha256_before": archive_hash_before,
        },
        "checkpoints": [],
    }

    with tempfile.TemporaryDirectory(prefix="crnn_cpu_checkpoint_test_") as temp_dir:
        temp_path = Path(temp_dir)
        with zipfile.ZipFile(ARCHIVE_PATH, "r") as archive:
            for name, expected in CHECKPOINTS.items():
                member_info = archive.getinfo(expected["member"])
                if member_info.file_size != expected["size_bytes"]:
                    raise ValueError(f"Unexpected size for archive member {expected['member']}")
                checkpoint_path = temp_path / name
                checkpoint_path.write_bytes(archive.read(expected["member"]))
                result = load_and_test_checkpoint(name, checkpoint_path)
                report["checkpoints"].append(result)
                del result

    archive_hash_after = sha256_file(ARCHIVE_PATH)
    report["archive"]["sha256_after"] = archive_hash_after
    report["archive"]["unchanged"] = archive_hash_after == archive_hash_before
    if not report["archive"]["unchanged"]:
        raise ValueError("Original checkpoint archive changed during the test")

    report["status"] = (
        "passed"
        if len(report["checkpoints"]) == len(CHECKPOINTS)
        and all(item["strict_load"] == "passed" for item in report["checkpoints"])
        else "failed"
    )
    JSON_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(markdown_report(report), encoding="utf-8")
    print(f"Status: {report['status']}")
    print(f"JSON: {JSON_PATH}")
    print(f"Markdown: {MARKDOWN_PATH}")
    for checkpoint in report["checkpoints"]:
        print(
            f"{checkpoint['name']}: strict={checkpoint['strict_load']}, "
            f"keys={checkpoint['state_dict_key_count']}, "
            f"output={checkpoint['forward_smoke_test']['output_shape']}"
        )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
