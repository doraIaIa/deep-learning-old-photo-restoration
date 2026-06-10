from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_DATASETS_MANIFEST = Path("artifacts/manifests/datasets_manifest.csv")


def configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safe CLI cho dataset status trong submission repo. "
            "Script này không build dataset, không copy data và không tạo output mặc định."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser(
        "status",
        help="In dataset status theo manifest hiện có.",
        description=(
            "Hiển thị trạng thái dataset manifests và nhắc rằng full datasets là external, "
            "không nằm trong Git của repo submission."
        ),
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
        warnings.append(f"Thiếu datasets manifest: {path}")
        return [], warnings
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        warnings.append(f"Datasets manifest rỗng: {path}")
    return rows, warnings


def write_json_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "dataset_id": row.get("dataset_id", ""),
            "role": row.get("role", ""),
            "external_source_path": row.get("external_source_path", ""),
            "status": row.get("status", ""),
            "notes": row.get("notes", ""),
        }
        for row in rows
    ]


def collect_dataset_sources(rows: list[dict[str, str]]) -> list[str]:
    sources: list[str] = []
    for row in rows:
        dataset_id = row.get("dataset_id", "").strip()
        if dataset_id:
            sources.append(dataset_id)
    return sources


def run_status(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    manifest_path = (repo_root / args.datasets_manifest).resolve()
    rows, warnings = read_manifest(manifest_path)
    dataset_sources = collect_dataset_sources(rows)

    print("dataset_status_entrypoint")
    print("full datasets are external and should not be committed to Git")
    print("manifest_path:", manifest_path)
    print("dataset_sources:")
    if warnings:
        print("[WARNING] dataset_sources unavailable_from_manifest")
    elif dataset_sources:
        for dataset_id in dataset_sources:
            print(f"- {dataset_id}")
    else:
        print("[WARNING] dataset_sources unavailable_from_manifest")
    print("see_also:")
    print("- data/README.md")
    print("- docs/artifacts.md")

    if warnings:
        for item in warnings:
            print(f"[WARNING] {item}")
    else:
        for row in summarize_rows(rows):
            print(
                f"- {row['dataset_id']}: role={row['role']} | "
                f"status={row['status']} | source={row['external_source_path']}"
            )

    payload = {
        "entrypoint": "dataset_status",
        "manifest_path": str(manifest_path),
        "warnings": warnings,
        "rows": summarize_rows(rows),
        "notes": [
            "Full datasets are external and are not committed to the submission Git repo.",
            "Use data/README.md and docs/artifacts.md for policy details.",
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
        parser.error("Thiếu subcommand. Dùng `status` để xem dataset status.")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
