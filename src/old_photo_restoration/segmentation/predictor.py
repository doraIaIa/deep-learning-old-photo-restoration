from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from old_photo_restoration.config import ProjectConfig
from old_photo_restoration.segmentation.model import CrackSegmenter
from old_photo_restoration.utils.checkpoints import sha256_file
from old_photo_restoration.utils.device import get_best_device


class SegmentationPredictor:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.checkpoint_path = config.checkpoint.expected_path
        self.expected_sha256 = config.checkpoint.sha256.lower()
        self.device = torch.device("cuda" if get_best_device(prefer=("cuda", "cpu")) == "cuda" else "cpu")
        self._model: CrackSegmenter | None = None
        self._checkpoint_payload: dict | None = None
        self._checkpoint_sha256: str | None = None

    def verify_checkpoint(self) -> str:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint segmentation: {self.checkpoint_path}")
        actual_sha256 = sha256_file(self.checkpoint_path).lower()
        if actual_sha256 != self.expected_sha256:
            raise ValueError(
                "SHA256 checkpoint r013 không khớp. "
                f"expected={self.expected_sha256}, actual={actual_sha256}"
            )
        self._checkpoint_sha256 = actual_sha256
        return actual_sha256

    def load_model(self) -> CrackSegmenter:
        if self._model is not None:
            return self._model

        self.verify_checkpoint()
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise KeyError(f"Checkpoint thiếu `model_state_dict`: {self.checkpoint_path}")

        model_config = checkpoint.get("model_config") or {}
        model = CrackSegmenter(
            in_channels=int(model_config.get("in_channels", 3)),
            out_channels=int(model_config.get("out_channels", 1)),
            base_channels=int(model_config.get("base_channels", 8)),
        ).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()

        self._checkpoint_payload = checkpoint
        self._model = model
        return model

    @staticmethod
    def read_image_rgb(image_path: Path) -> np.ndarray:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Không đọc được ảnh đầu vào: {image_path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def build_inference_tensor(image_rgb: np.ndarray, image_size: int = 512) -> torch.Tensor:
        resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))).float()
        return tensor.unsqueeze(0)

    @staticmethod
    def resize_probability_mask(probability_mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        target_height, target_width = target_hw
        return cv2.resize(probability_mask, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def binary_mask_from_probability(probability_mask: np.ndarray, threshold: float) -> np.ndarray:
        return (probability_mask >= threshold).astype(np.uint8) * 255

    @torch.no_grad()
    def predict_probability(self, image_path: Path) -> np.ndarray:
        image_rgb = self.read_image_rgb(image_path)
        model = self.load_model()
        input_tensor = self.build_inference_tensor(image_rgb).to(self.device)
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
        if probability.ndim != 2:
            raise ValueError(f"Probability mask phải có 2 chiều, nhận được shape {probability.shape}")
        probability = self.resize_probability_mask(probability, image_rgb.shape[:2])
        return np.clip(probability, 0.0, 1.0)

    def predict_dl_mask(self, image_path: Path, threshold: float = 0.5) -> np.ndarray:
        probability = self.predict_probability(image_path)
        return self.binary_mask_from_probability(probability, threshold)

    @property
    def checkpoint_sha256(self) -> str | None:
        return self._checkpoint_sha256

    @property
    def checkpoint_payload(self) -> dict | None:
        return self._checkpoint_payload
