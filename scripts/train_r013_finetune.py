from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_RUNS_MANIFEST = Path("artifacts/manifests/reproduction_runs_manifest.csv")
DEFAULT_CHECKPOINTS_MANIFEST = Path("artifacts/manifests/checkpoints_manifest.csv")
DEFAULT_DATASETS_MANIFEST = Path("artifacts/manifests/datasets_manifest.csv")


def configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safe R013 fine-tune evidence/status CLI cho submission repo. "
            "Script này không chạy fine-tune thật và không tạo output mặc định."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser(
        "status",
        help="In R013 fine-tune evidence/status từ manifests hiện có.",
        description=(
            "Hiển thị R013_REPRO, dataset facts và checkpoint policy để root script "
            "không còn là stub chết."
        ),
    )
    status_parser.add_argument(
        "--runs-manifest",
        type=Path,
        default=DEFAULT_RUNS_MANIFEST,
        help="Path tới reproduction runs manifest CSV.",
    )
    status_parser.add_argument(
        "--checkpoints-manifest",
        type=Path,
        default=DEFAULT_CHECKPOINTS_MANIFEST,
        help="Path tới checkpoints manifest CSV.",
    )
    status_parser.add_argument(
        "--datasets-manifest",
        type=Path,
        default=DEFAULT_DATASETS_MANIFEST,
        help="Path tới datasets manifest CSV.",
    )
    status_parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repo root để resolve path tương đối.",
    )
    status_parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Nếu truyền, ghi JSON summary ra path chỉ định.",
    )
    status_parser.set_defaults(handler=run_status)
    return parser


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    warnings: list[str] = []
    if not path.exists():
        warnings.append(f"Thiếu manifest: {path}")
        return [], warnings
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        warnings.append(f"Manifest rỗng: {path}")
    return rows, warnings


def find_first(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str] | None:
    for row in rows:
        if row.get(key) == value:
            return row
    return None


def write_json_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_dataset_fact(row: dict[str, str] | None) -> str:
    if row is None:
        return "unavailable_from_manifest"

    parts: list[str] = []
    image_count = row.get("image_count", "").strip()
    mask_count = row.get("mask_count", "").strip()
    notes = row.get("notes", "").strip()

    if image_count:
        parts.append(f"image_count={image_count}")
    if mask_count:
        parts.append(f"mask_count={mask_count}")
    if notes:
        parts.append(f"notes={notes}")

    return " | ".join(parts) if parts else "unavailable_from_manifest"


def run_status(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    runs_path = (repo_root / args.runs_manifest).resolve()
    checkpoints_path = (repo_root / args.checkpoints_manifest).resolve()
    datasets_path = (repo_root / args.datasets_manifest).resolve()

    run_rows, run_warnings = read_manifest(runs_path)
    checkpoint_rows, checkpoint_warnings = read_manifest(checkpoints_path)
    dataset_rows, dataset_warnings = read_manifest(datasets_path)
    warnings = run_warnings + checkpoint_warnings + dataset_warnings

    r013_run = find_first(run_rows, "run_id", "R013_REPRO")
    r013_dataset = find_first(dataset_rows, "dataset_id", "r013_finetune_set")
    r013_checkpoint = None
    for row in checkpoint_rows:
        if row.get("run_id") == "R013_REPRO" and "final Module 1 local operational checkpoint" in row.get("role", ""):
            r013_checkpoint = row
            break
    dataset_fact = summarize_dataset_fact(r013_dataset)

    print("r013_finetune_evidence_status_entrypoint")
    print("this script does not run fine-tuning inside the submission build")
    if r013_checkpoint is not None:
        print("R013_REPRO is the current final Module 1 checkpoint according to the checkpoints manifest")
    else:
        print("[WARNING] R013 checkpoint claim unavailable_from_manifest")
    if r013_dataset is not None:
        print("dataset_fact:", dataset_fact)
        print("dataset_manifest_status:", r013_dataset.get("status", ""))
    else:
        print("[WARNING] Không tìm thấy dataset row cho R013 trong datasets manifest.")
        print("dataset_fact: unavailable_from_manifest")
        print("dataset_manifest_status: unavailable_from_manifest")
    if r013_checkpoint is not None:
        print("checkpoint_policy:", r013_checkpoint.get("git_policy", ""))
        print("checkpoint_local_path:", r013_checkpoint.get("repo_relative_path", ""))
    if r013_run is not None:
        print("run_summary:", f"best_epoch={r013_run.get('best_epoch','')} | val_iou={r013_run.get('val_iou','')} | val_f1={r013_run.get('val_f1','')}")
    print("phase_note: this is an evidence/status entrypoint to avoid a dead root script")
    if warnings:
        for item in warnings:
            print(f"[WARNING] {item}")

    payload = {
        "entrypoint": "r013_finetune_evidence_status",
        "runs_manifest": str(runs_path),
        "checkpoints_manifest": str(checkpoints_path),
        "datasets_manifest": str(datasets_path),
        "warnings": warnings,
        "r013_run": r013_run,
        "r013_checkpoint": r013_checkpoint,
        "r013_dataset": r013_dataset,
        "dataset_fact": dataset_fact,
        "notes": [
            "This script does not run fine-tuning inside the submission build.",
            "R013_REPRO is scoped to Module 1 only.",
            "Checkpoint policy is local ignored do not commit.",
        ],
    }
    if args.json_output is not None:
        write_json_output(args.json_output.resolve(), payload)
    return 0


def main() -> int:
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "handler"):
        parser.error("Thiếu subcommand. Dùng `status` để xem R013 fine-tune evidence/status.")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
