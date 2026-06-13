from __future__ import annotations

from pathlib import Path

from old_photo_restoration.utils.image_io import read_image_rgb, write_image_rgb

from .codeformer_backend import run_codeformer_restoration
from .config import FaceRestorationConfig


class CodeFormerWrapper:
    """Path-based compatibility facade for the in-memory CodeFormer backend."""

    def __init__(self, repo_root: Path, checkpoint: Path, conda_env: str = "codeformer") -> None:
        self.repo_root = repo_root
        self.checkpoint = checkpoint
        self.conda_env = conda_env

    def run(self, image_path: Path, output_dir: Path, fidelity: float = 0.7) -> Path:
        restored, _ = run_codeformer_restoration(
            read_image_rgb(image_path),
            FaceRestorationConfig(
                enabled=True,
                required=True,
                repo_path=str(self.repo_root),
                env_name=self.conda_env,
                fidelity_weight=fidelity,
            ),
            work_dir=output_dir / "_runtime",
        )
        output_path = output_dir / "codeformer_output.png"
        write_image_rgb(output_path, restored)
        return output_path
