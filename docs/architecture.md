# Architecture

The repository is organized around configuration, segmentation, inpainting, face restoration, evaluation, and utility layers. The current architecture prioritizes clear module boundaries so that inference, artifact verification, and minimal reproducibility can be inspected independently.


## Module 1: Crack Segmentation

- **Model Family**: Attention U-Net. The architecture incorporates Attention Gates to filter skip connections and focus the decoder on relevant crack-like regions while suppressing background noise.
- **Input/Output**: Expects an RGB image and outputs a probability/binary mask representing defect locations.
- **Loss Strategy History**: The training sequence experimented with varying losses. Early iterations used BCE+Dice. Later iterations (starting R008) introduced Tversky-style loss (e.g., alpha=0.3, beta=0.7) and increased recall penalties (beta=0.8 in R011) to capture faint scratches.
- **Threshold Calibration**: Operational thresholds were historically calibrated based on validation distributions, finalizing at 0.50 for the current R013 model.

## Module 1.5: Hybrid Mask Refinement

To mitigate false negatives from the deep learning segmenter on extremely thin scratches, a hybrid refinement strategy is used:
- Combines the deep learning mask with a classical computer vision branch (utilizing CLAHE, morphological Blackhat, and Canny edge detection).
- The 
epair_wide_v1 policy ensures the union mask undergoes morphological closing, connecting, and dilation to fully encompass the defect area before inpainting.
- This subsystem provides conservative context but does not claim perfect pixel-level accuracy.
- The repair_wide_v1 policy widens and connects narrow scratch regions before inpainting, using conservative morphology rather than claiming perfect pixel-level masks.

## Module 2: Inpainting Backend

The pipeline leverages LaMa (Resolution-robust Large Mask Inpainting) as the backend restoration engine.
- **Implementation**: LaMa is used purely as a pretrained external wrapper or subprocess.
- **Design Rationale**: No further fine-tuning of the LaMa model is performed in the this project. This dependency isolation avoids generative training instability.

## Module 3: Optional Face Restoration

For photos containing degraded faces, an optional face enhancement module is available:
- **Components**: Combines RetinaFace for face detection and CodeFormer for face restoration.
- **Usage**: This module is dependency-gated and strictly optional. It does not provide an absolute identity preservation guarantee and is not considered part of the core scratch-removal evaluation.
