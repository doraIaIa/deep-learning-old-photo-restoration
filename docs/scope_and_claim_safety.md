# Scope and Claim-Safety Notes

This document clarifies the exact implemented scope, artifact policies, and explicit limitations of the Old Photo Restoration project. It serves to establish unambiguous boundaries on what the repository provides.

## Implemented Scope
- **Module 1**: Crack Segmentation. Utilizes an Attention U-Net architecture. Operational segmenter: R013.
- **Module 1.5**: Hybrid Mask Refinement. Combines deep learning probability masks with classical computer vision morphology to recover thin scratches.
- **Module 2**: Inpainting Backend. Utilizes a pre-trained LaMa (Large Mask Inpainting) model wrapper.
- **Module 3**: Optional Face Restoration. Provides CodeFormer + RetinaFace integration.

## Artifact and Checkpoint Policy
- **Strict Isolation**: Checkpoint binaries and datasets are kept strictly local. They are ignored by version control to prevent repository bloat. Only metadata manifests and folder skeletons are committed.
- **Historical Evidence**: Earlier iterative development runs (e.g., R006-R008) are retained as evidence-only skeletons.
- **Current Binaries**: Current checkpoints (e.g., R009-R013) remain local ignored binaries. R013 is designated as the current operational segmenter.

## Explicit Limitations and Safe Claims
To maintain project integrity, the following caveats apply:
- **LaMa Integration**: The LaMa inpainting backend is used entirely as a pretrained subprocess wrapper. **No fine-tuning** of LaMa is performed within this pipeline.
- **Quantitative Metrics**: The repository does not claim completed LPIPS, FID, or masked-region LPIPS measurements. These remain future evaluation protocols.
- **Face Restoration**: CodeFormer is an optional dependency and **does not guarantee identity preservation**.
- **Historical Originals**: Reproduction checkpoints are not presented as historical originals without verifiable evidence.
