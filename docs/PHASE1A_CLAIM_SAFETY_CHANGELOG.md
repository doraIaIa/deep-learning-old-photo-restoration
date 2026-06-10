# PHASE1A Claim Safety Changelog

- Ngày thực hiện: `2026-06-10`
- Branch: `harden/submission-claim-safety`

## File đã sửa

- `README.md`
- `docs/demo_script.md`
- `docs/deployment.md`
- `docs/experiment_summary.md`
- `docs/external_dependencies.md`
- `docs/reproducibility.md`
- `docs/limitations.md`
- `docs/evaluation_protocol.md`

## Lý do sửa

- Hạ claim về checkpoint, evaluation và Module 3 về đúng mức evidence hiện có.
- Khóa lại wording quanh `R013_REPRO`, `118 valid pairs`, split `83/18/17`, threshold `0.50`.
- Nêu rõ LaMa hiện là official/pretrained wrapper, không phải LaMa fine-tune.
- Chuyển `LPIPS`, `FID`, `masked-region LPIPS`, full end-to-end evaluation và identity preservation sang caveat hoặc future work.

## File cố ý không sửa

- `scripts/evaluate_segmentation.py`
- `scripts/run_ablation.py`
- `app/gradio_demo.py`
- `configs/checkpoints.yaml`
- `configs/external_paths.yaml`
- `configs/inference.yaml`
- Mọi file ảnh trong `docs/assets/` và `examples/`
- Mọi file trong repo cũ

## Claim đã hạ cấp thành caveat hoặc future work

- LaMa fine-tune
- `LPIPS`
- `FID`
- `masked-region LPIPS`
- full quantitative end-to-end evaluation
- CodeFormer identity preservation
- Module 3 face restoration hoàn chỉnh

## Xác nhận phạm vi

- Không train
- Không eval
- Không inference
- Không copy checkpoint
- Không sửa scripts train/eval/ablation
- Không sửa các config meaningful ở Phase 1A
