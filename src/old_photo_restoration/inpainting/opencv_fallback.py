from __future__ import annotations

from pathlib import Path


def run_opencv_fallback(image_path: Path, mask_path: Path, output_dir: Path) -> Path:
    raise NotImplementedError("The OpenCV fallback path is unavailable in the current runtime configuration.")
