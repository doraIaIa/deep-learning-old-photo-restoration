# Limitations

## Current Scope

Pipeline hiện tại của submission tập trung vào:

- segmentation cho vùng hư hại;
- hybrid mask;
- `repair_wide_v1`;
- official/pretrained LaMa wrapper.

## What Is Not A Safe Claim Yet

- LaMa fine-tune.
- `LPIPS`, `FID`, `masked-region LPIPS`.
- Full quantitative end-to-end evaluation.
- CodeFormer identity preservation.
- Module 3 face restoration hoàn chỉnh trong repo submission.
- Phong illumination như một phần implementation hiện có của repo submission.

## Dataset And Experiment Caveats

- `R013` phải được mô tả là `120` ảnh ban đầu nhưng chỉ `118` valid pairs.
- `R012` chỉ là nhánh thực nghiệm với `15` manual samples.
- `demo3` là golden regression case, không phải benchmark đại diện toàn bộ tập ảnh cũ thực.

## Future Work

- Fine-tune LaMa với artifact đầy đủ.
- LPIPS/FID/masked-region LPIPS.
- Hoàn thiện Module 3 ở mức end-to-end.
- Bổ sung evaluation protocol đầy đủ hơn ngoài smoke/regression.
