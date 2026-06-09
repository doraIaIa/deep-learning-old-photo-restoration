from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from old_photo_restoration.config import load_config
from old_photo_restoration.pipeline import RestorationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chạy pipeline submission Phase 1C với mask bypass.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--checkpoint-config", type=Path, default=Path("configs/checkpoints.yaml"))
    parser.add_argument("--external-config", type=Path, default=Path("configs/external_paths.yaml"))
    parser.add_argument("--face-mode", choices=["off", "auto"], default="off")
    parser.add_argument("--reference", type=Path, default=None)
    return parser


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def compare_images(output_path: Path, reference_path: Path) -> dict[str, Any]:
    output = read_rgb(output_path)
    reference = read_rgb(reference_path)
    same_size = output.shape == reference.shape
    report: dict[str, Any] = {
        "same_size": bool(same_size),
        "output_shape": list(output.shape),
        "reference_shape": list(reference.shape),
        "mae": None,
        "max_absolute_error": None,
        "psnr": None,
    }
    if not same_size:
        return report

    diff = np.abs(output.astype(np.float32) - reference.astype(np.float32))
    mae = float(diff.mean())
    mse = float(np.mean(np.square(output.astype(np.float32) - reference.astype(np.float32))))
    psnr = float("inf") if mse == 0.0 else float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))
    report["mae"] = mae
    report["max_absolute_error"] = int(diff.max())
    report["psnr"] = psnr
    return report


def main() -> int:
    args = build_parser().parse_args()
    image_path = resolve_path(args.image)
    mask_path = resolve_path(args.mask)
    output_dir = resolve_path(args.output_dir)
    reference_path = resolve_path(args.reference)

    config = load_config(
        inference_path=resolve_path(args.config),
        checkpoint_path=resolve_path(args.checkpoint_config),
        external_path=resolve_path(args.external_config),
    )
    pipeline = RestorationPipeline(config)

    result = pipeline.run(
        image_path=image_path,
        output_dir=output_dir,
        mask_path=mask_path,
        face_mode=args.face_mode,
        golden_reference=reference_path,
    )

    metadata_path = result.output_dir / "metadata.json"
    metadata = dict(result.metadata)
    comparison: dict[str, Any] | None = None
    if reference_path is not None:
        comparison = compare_images(result.restored_path, reference_path)
        metadata["comparison"] = comparison
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"restored_before_face: {result.restored_path}")
    print(f"final_mask: {result.mask_path}")
    print(f"metadata: {metadata_path}")
    if comparison is not None:
        print(f"same_size: {comparison['same_size']}")
        print(f"mae: {comparison['mae']}")
        print(f"max_absolute_error: {comparison['max_absolute_error']}")
        print(f"psnr: {comparison['psnr']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
