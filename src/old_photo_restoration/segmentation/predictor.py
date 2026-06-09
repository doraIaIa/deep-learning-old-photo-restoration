from __future__ import annotations

from pathlib import Path


class SegmentationPredictor:
    def predict(self, image_path: Path) -> None:
        raise NotImplementedError("Segmentation predictor chưa được migrate trong phase này.")
