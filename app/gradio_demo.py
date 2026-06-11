from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from old_photo_restoration.config import ProjectConfig, load_config
from old_photo_restoration.inpainting.lama_wrapper import LamaInpainter
from old_photo_restoration.pipeline import RestorationPipeline
from old_photo_restoration.segmentation.predictor import SegmentationPredictor


DEFAULT_OUTPUT_DIR = Path("examples/outputs/gradio_runs")
AUTO_MASK_MODE = "auto-mask"


def resolve_path(path: Path | str) -> Path:
    raw_path = Path(path)
    return raw_path if raw_path.is_absolute() else (PROJECT_ROOT / raw_path).resolve()


def write_uploaded_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Không ghi được ảnh upload: {path}")


def load_runtime_config() -> ProjectConfig:
    return load_config(
        inference_path=resolve_path("configs/inference.yaml"),
        checkpoint_path=resolve_path("configs/checkpoints.yaml"),
        external_path=resolve_path("configs/external_paths.yaml"),
    )


def validate_runtime_readiness(config: ProjectConfig) -> None:
    predictor = SegmentationPredictor(config)
    predictor.verify_checkpoint()

    lama_readiness = LamaInpainter(config.lama).readiness()
    if not lama_readiness.get("repo_exists"):
        raise RuntimeError(f"Thiếu LaMa repo_root: {config.lama.repo_root}")
    if not lama_readiness.get("predict_script_exists"):
        raise RuntimeError(f"Thiếu LaMa predict.py: {config.lama.predict_script}")
    if not lama_readiness.get("checkpoint_exists"):
        raise RuntimeError(f"Thiếu LaMa checkpoint: {config.lama.checkpoint}")
    if not lama_readiness.get("available"):
        raise RuntimeError(
            "LaMa chưa sẵn sàng để chạy local demo. "
            f"reason={lama_readiness.get('reason')}"
        )


def build_status_text(output_dir: Path, metadata: dict[str, Any]) -> str:
    lines = [
        "Run thành công.",
        f"output_dir: {output_dir}",
        f"segmentation_model_version: {metadata.get('segmentation_model_version', 'r013')}",
        f"segmentation_threshold: {metadata.get('segmentation_threshold', '')}",
        f"mask_source: {metadata.get('mask_source', '')}",
        f"mask_refine: {metadata.get('mask_refine', '')}",
        f"final_mask_ratio: {metadata.get('final_mask_ratio', '')}",
        f"inpainting_backend: {metadata.get('inpainting_backend', '')}",
    ]
    return "\n".join(lines)


def run_auto_mask_pipeline(
    image: np.ndarray | None,
    output_dir_text: str = str(DEFAULT_OUTPUT_DIR),
    mode: str = AUTO_MASK_MODE,
    face_mode: str = "off",
    segmenter_choice: str = "R013 Custom Attention U-Net",
) -> tuple[np.ndarray | None, str | None, str | None, dict[str, Any] | None, str]:
    if image is None:
        return None, None, None, None, "Thiếu ảnh đầu vào."
    if mode != AUTO_MASK_MODE:
        return image, None, None, None, f"Mode không hỗ trợ trong phase hiện tại: {mode}"
    if face_mode != "off":
        return image, None, None, None, "Phase hiện tại chỉ hỗ trợ face_mode=off."

    try:
        config = load_runtime_config()
        validate_runtime_readiness(config)
        pipeline = RestorationPipeline(config)

        base_output_dir = resolve_path(output_dir_text or str(DEFAULT_OUTPUT_DIR))
        run_dir = base_output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        input_path = run_dir / "input.png"
        write_uploaded_image(input_path, image)

        import os
        
        segmenter_arch = "r013_custom_attnunet"
        segmenter_checkpoint = None
        if "R014" in segmenter_choice:
            segmenter_arch = "r014_resnet34"
            env_ckpt = os.environ.get("R014_SEGMENTER_CHECKPOINT")
            if not env_ckpt or not Path(env_ckpt).exists():
                return image, None, None, {"error": "Missing checkpoint"}, "R014 checkpoint not found. Set R014_SEGMENTER_CHECKPOINT or provide checkpoint path."
            segmenter_checkpoint = Path(env_ckpt)

        result = pipeline.run(
            image_path=input_path,
            output_dir=run_dir,
            mask_path=None,
            face_mode="off",
            segmenter_arch=segmenter_arch,
            segmenter_checkpoint=segmenter_checkpoint,
        )
        metadata = dict(result.metadata)
        metadata["ui_mode"] = mode
        metadata["ui_face_mode"] = face_mode
        status = build_status_text(run_dir, metadata)
        return image, str(result.mask_path), str(result.restored_path), metadata, status
    except Exception as exc:
        return image, None, None, {"error": str(exc)}, f"Lỗi readiness/runtime: {exc}"


def run_auto_mask_from_path(image_path: Path, output_dir: Path, face_mode: str = "off", segmenter_arch: str = "r013_custom_attnunet", segmenter_checkpoint: Path | None = None) -> dict[str, Path]:
    config = load_runtime_config()
    validate_runtime_readiness(config)
    pipeline = RestorationPipeline(config)
    result = pipeline.run(
        image_path=resolve_path(image_path),
        output_dir=resolve_path(output_dir),
        mask_path=None,
        face_mode=face_mode,
        segmenter_arch=segmenter_arch,
        segmenter_checkpoint=segmenter_checkpoint,
    )
    metadata_path = result.output_dir / "metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps(result.metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_dir": result.output_dir,
        "final_mask": result.mask_path,
        "restored_before_face": result.restored_path,
        "metadata": metadata_path,
    }


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Old Photo Restoration Demo") as demo:
        gr.Markdown("# Old Photo Restoration Demo")
        gr.Markdown(
            "Pipeline local hiện tại chạy auto-mask. Hỗ trợ segmenter R013 (default) và R014 (optional)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload ảnh cũ", type="numpy")
                mode = gr.Dropdown(
                    label="Mode",
                    choices=[AUTO_MASK_MODE],
                    value=AUTO_MASK_MODE,
                    interactive=False,
                )
                face_mode = gr.Dropdown(
                    label="Face mode",
                    choices=["off"],
                    value="off",
                    interactive=False,
                )
                segmenter_choice = gr.Dropdown(
                    label="Segmenter",
                    choices=["R013 Custom Attention U-Net", "R014 ResNet-34 (3x3 Dilation)"],
                    value="R013 Custom Attention U-Net",
                    interactive=True,
                )
                output_dir = gr.Textbox(
                    label="Output directory",
                    value=str(DEFAULT_OUTPUT_DIR),
                )
                run_button = gr.Button("Run")
            with gr.Column(scale=1):
                input_preview = gr.Image(label="Input preview", type="numpy")
                final_mask = gr.Image(label="Final mask", type="filepath")
                restored_before_face = gr.Image(label="Restored before face", type="filepath")
        metadata = gr.JSON(label="Metadata")
        status = gr.Textbox(label="Status", lines=8)

        run_button.click(
            run_auto_mask_pipeline,
            inputs=[input_image, output_dir, mode, face_mode, segmenter_choice],
            outputs=[input_preview, final_mask, restored_before_face, metadata, status],
        )
    demo.queue(default_concurrency_limit=1)
    return demo


def launch_app(server_name: str = "127.0.0.1", server_port: int = 7860, share: bool = False) -> None:
    demo = create_app()
    demo.launch(server_name=server_name, server_port=server_port, share=share)


if __name__ == "__main__":
    launch_app()
