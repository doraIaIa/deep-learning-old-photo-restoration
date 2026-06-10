import os
import argparse
import csv
import json
import random
import cv2
import numpy as np

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_mae(img1, img2):
    return np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))

def calculate_mse(img1, img2):
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

def calculate_ssim(img1, img2):
    if not HAS_SKIMAGE:
        return 0.0
    return ssim(img1, img2, data_range=255, channel_axis=2 if len(img1.shape)==3 else None)

def evaluate_metrics(clean, degraded, mask):
    # Masked metrics
    mask_bool = mask > 127
    if len(mask.shape) == 2 and len(clean.shape) == 3:
        mask_bool = np.stack([mask_bool]*3, axis=-1)
    
    clean_masked = clean[mask_bool]
    degraded_masked = degraded[mask_bool]
    
    if len(clean_masked) == 0:
        masked_mae = 0.0
        masked_mse = 0.0
        masked_psnr = float('inf')
    else:
        masked_mae = np.mean(np.abs(clean_masked.astype(np.float32) - degraded_masked.astype(np.float32)))
        masked_mse = np.mean((clean_masked.astype(np.float32) - degraded_masked.astype(np.float32)) ** 2)
        masked_psnr = 20 * np.log10(255.0 / np.sqrt(masked_mse)) if masked_mse > 0 else float('inf')
        
    return {
        "psnr": calculate_psnr(clean, degraded),
        "ssim": calculate_ssim(clean, degraded) if HAS_SKIMAGE else 0.0,
        "mae": calculate_mae(clean, degraded),
        "mse": calculate_mse(clean, degraded),
        "masked_mae": masked_mae,
        "masked_mse": masked_mse,
        "masked_psnr": masked_psnr
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate restoration on paired synthetic data.")
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--degraded-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["metrics-only", "generate-and-evaluate"], default="metrics-only")
    parser.add_argument("--restored-dir", type=str)
    parser.add_argument("--run-pipeline", action="store_true")
    parser.add_argument("--pipeline-mode", type=str, default="oracle-mask")
    parser.add_argument("--external-paths", type=str)
    parser.add_argument("--limit-size", type=int, default=512)
    parser.add_argument("--write-preview-grid", action="store_true", default=False)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    random.seed(args.seed)
    
    clean_files = sorted(os.listdir(args.clean_dir))
    random.shuffle(clean_files)
    
    selected_files = clean_files[:args.max_samples]
    
    results = []
    failed = 0
    
    warnings = []
    if not HAS_SKIMAGE:
        warnings.append("skimage is not available. SSIM will be reported as 0.0.")
    
    for filename in selected_files:
        clean_path = os.path.join(args.clean_dir, filename)
        degraded_path = os.path.join(args.degraded_dir, filename)
        mask_path = os.path.join(args.mask_dir, filename)
        
        if not os.path.exists(degraded_path) or not os.path.exists(mask_path):
            failed += 1
            continue
            
        clean_img = cv2.imread(clean_path)
        degraded_img = cv2.imread(degraded_path)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if clean_img is None or degraded_img is None or mask_img is None:
            failed += 1
            continue
            
        if clean_img.shape != degraded_img.shape:
            degraded_img = cv2.resize(degraded_img, (clean_img.shape[1], clean_img.shape[0]))
        if mask_img.shape != clean_img.shape[:2]:
            mask_img = cv2.resize(mask_img, (clean_img.shape[1], clean_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        # Ensure mask is binary
        _, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        
        # Calculate baseline metrics
        baseline_metrics = evaluate_metrics(clean_img, degraded_img, mask_img)
        
        restored_metrics = None
        if args.mode == "metrics-only" and args.restored_dir:
            restored_path = os.path.join(args.restored_dir, filename)
            if os.path.exists(restored_path):
                restored_img = cv2.imread(restored_path)
                if restored_img is not None:
                    if restored_img.shape != clean_img.shape:
                        restored_img = cv2.resize(restored_img, (clean_img.shape[1], clean_img.shape[0]))
                    restored_metrics = evaluate_metrics(clean_img, restored_img, mask_img)
        
        elif args.mode == "generate-and-evaluate":
            # For this task, if LaMa is configured, we can run it.
            # But the prompt said "Do not import heavy external model unless generate-and-evaluate is requested."
            # Since the user requested evaluating WITHOUT training and using LaMa to generate outputs.
            # We will use the existing pipeline if available.
            # Here we just implement a stub or try importing the pipeline.
            try:
                import sys
                sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                from src.pipeline.restoration_pipeline import OldPhotoRestorationPipeline
                
                # To be safe and minimal:
                if not hasattr(OldPhotoRestorationPipeline, '_initialized'):
                    os.environ["LAMA_SKIP_CUDA"] = "1" # Just in case
                    pipe = OldPhotoRestorationPipeline(external_paths_config=args.external_paths)
                    # Force disable codeformer
                    pipe.module3 = None 
                    OldPhotoRestorationPipeline._initialized = True
                    OldPhotoRestorationPipeline._pipe = pipe
                
                pipe = OldPhotoRestorationPipeline._pipe
                
                # Run restoration.
                # In oracle mode, we use GT mask.
                if args.pipeline_mode == "oracle-mask":
                    # We might need to bypass module1
                    mask_to_use = mask_img
                    # LaMa wrapper might be accessible via pipe.module2
                    restored_img = pipe.module2.infer(degraded_img, mask_to_use)
                else:
                    restored_img, _ = pipe.run(degraded_img)
                
                if restored_img is not None:
                    # Save output
                    cv2.imwrite(os.path.join(args.output_dir, filename), restored_img)
                    if restored_img.shape != clean_img.shape:
                        restored_img = cv2.resize(restored_img, (clean_img.shape[1], clean_img.shape[0]))
                    restored_metrics = evaluate_metrics(clean_img, restored_img, mask_img)
            except Exception as e:
                warnings.append(f"Failed to generate output for {filename}: {str(e)}")
        
        row = {
            "filename": filename,
            "baseline_psnr": baseline_metrics["psnr"],
            "baseline_ssim": baseline_metrics["ssim"],
            "baseline_mae": baseline_metrics["mae"],
            "baseline_mse": baseline_metrics["mse"],
            "baseline_masked_mae": baseline_metrics["masked_mae"],
            "baseline_masked_mse": baseline_metrics["masked_mse"],
            "baseline_masked_psnr": baseline_metrics["masked_psnr"],
        }
        
        if restored_metrics:
            row.update({
                "restored_psnr": restored_metrics["psnr"],
                "restored_ssim": restored_metrics["ssim"],
                "restored_mae": restored_metrics["mae"],
                "restored_mse": restored_metrics["mse"],
                "restored_masked_mae": restored_metrics["masked_mae"],
                "restored_masked_mse": restored_metrics["masked_mse"],
                "restored_masked_psnr": restored_metrics["masked_psnr"],
            })
            
        results.append(row)
        
    with open(os.path.join(args.output_dir, "per_image_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        if results:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
            
    with open(os.path.join(args.output_dir, "selected_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename"])
        for r in results:
            w.writerow([r["filename"]])
            
    with open(os.path.join(args.output_dir, "warnings.md"), "w", encoding="utf-8") as f:
        for w in warnings:
            f.write(f"- {w}\n")
            
    # Aggregate
    if results:
        agg = {
            "evaluated_count": len(results),
            "failed_count": failed,
            "baseline": {
                "mean_psnr": np.mean([r["baseline_psnr"] for r in results]),
                "mean_ssim": np.mean([r["baseline_ssim"] for r in results]),
                "mean_mae": np.mean([r["baseline_mae"] for r in results]),
                "mean_mse": np.mean([r["baseline_mse"] for r in results]),
                "mean_masked_mae": np.mean([r["baseline_masked_mae"] for r in results]),
                "mean_masked_mse": np.mean([r["baseline_masked_mse"] for r in results]),
                "mean_masked_psnr": np.mean([r["baseline_masked_psnr"] for r in results if r["baseline_masked_psnr"] != float('inf')]),
            }
        }
        
        if "restored_psnr" in results[0]:
            agg["restored"] = {
                "mean_psnr": np.mean([r["restored_psnr"] for r in results]),
                "mean_ssim": np.mean([r["restored_ssim"] for r in results]),
                "mean_mae": np.mean([r["restored_mae"] for r in results]),
                "mean_mse": np.mean([r["restored_mse"] for r in results]),
                "mean_masked_mae": np.mean([r["restored_masked_mae"] for r in results]),
                "mean_masked_mse": np.mean([r["restored_masked_mse"] for r in results]),
                "mean_masked_psnr": np.mean([r["restored_masked_psnr"] for r in results if r["restored_masked_psnr"] != float('inf')]),
            }
            
            agg["improvement"] = {
                "psnr_diff": agg["restored"]["mean_psnr"] - agg["baseline"]["mean_psnr"],
                "ssim_diff": agg["restored"]["mean_ssim"] - agg["baseline"]["mean_ssim"],
                "mae_diff": agg["baseline"]["mean_mae"] - agg["restored"]["mean_mae"],
                "mse_diff": agg["baseline"]["mean_mse"] - agg["restored"]["mean_mse"],
            }
            
        with open(os.path.join(args.output_dir, "aggregate_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
            
        with open(os.path.join(args.output_dir, "aggregate_metrics.md"), "w", encoding="utf-8") as f:
            f.write("# Aggregate Metrics\n")
            for k, v in agg.items():
                f.write(f"- {k}: {v}\n")

if __name__ == "__main__":
    main()
