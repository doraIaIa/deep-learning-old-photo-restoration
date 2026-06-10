# Scripts

This directory contains the public utility commands used to prepare data manifests, run the restoration pipeline, launch the demo, evaluate segmentation, verify artifacts, and reproduce the documented training/evaluation workflow.

The scripts are intentionally kept at the top level of this directory so they can be invoked consistently from the repository root.

## Main commands

- 
un_pipeline.py: runs the restoration pipeline on an input image using the configured segmenter, mask refinement, and inpainting backend.
- 
un_gradio_demo.py: launches the Gradio demo wrapper.
- check_readiness.py: checks whether required local artifacts and configuration entries are available.
- erify_artifacts.py: validates expected artifact metadata and local checkpoint availability.
- download_checkpoints.py: documents or assists external checkpoint retrieval where applicable.
- uild_dataset.py: prepares or validates dataset metadata according to the manifest-driven layout.
- 	rain_segmentation.py: training entrypoint for the segmentation model.
- 	rain_r013_finetune.py: R013-specific fine-tuning/reproduction entrypoint.
- evaluate_segmentation.py: segmentation evaluation utility.
- 
un_ablation.py: ablation/status utility for documented experiment variants.
- uild_demo_assets.py: prepares demo assets used by documentation.
- smoke_lama_inpainting.py: smoke test for the LaMa inpainting wrapper.

## Notes

Checkpoint binaries, datasets, generated outputs, and local machine paths are not committed to Git. The repository tracks scripts, manifests, README files, and directory structure needed to reproduce or inspect the project workflow.
