from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_RUNS_MANIFEST = Path("artifacts/manifests/reproduction_runs_manifest.csv")


def configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safe training evidence/status CLI cho root submission flow. "
            "Script này không chạy training segmentation thật."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser(
        "status",
        help="In training evidence/status từ reproduction manifest.",
        description=(
            "Hiển thị reproduction runs để cho biết root script này là entrypoint mô tả evidence, "
            "không phải training runner."
        ),
    )
    status_parser.add_argument(
        "--runs-manifest",
        type=Path,
        default=DEFAULT_RUNS_MANIFEST,
        help="Path tới reproduction runs manifest CSV.",
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
        warnings.append(f"Thiếu reproduction runs manifest: {path}")
        return [], warnings
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        warnings.append(f"Reproduction runs manifest rỗng: {path}")
    return rows, warnings


def select_repro_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keep = {"R010_REPRO", "R011_REPRO", "R012_REPRO", "R013_REPRO"}
    return [row for row in rows if row.get("run_id") in keep]


def write_json_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_status(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    manifest_path = (repo_root / args.runs_manifest).resolve()
    rows, warnings = read_manifest(manifest_path)
    selected = select_repro_rows(rows)

    print("training_evidence_status_entrypoint")
    print("this root submission script is not a training runner")
    print("training and fine-tuning evidence is described via reproduction manifest")
    print("manifest_path:", manifest_path)
    print("phase_note: if real training code is ported later, it should live under scripts/research/")
    for row in selected:
        print(
            f"- {row.get('run_id','')}: dataset={row.get('dataset','')} | "
            f"purpose={row.get('purpose','')} | report_status={row.get('report_status','')}"
        )
    if warnings:
        for item in warnings:
            print(f"[WARNING] {item}")

    payload = {
        "entrypoint": "training_evidence_status",
        "manifest_path": str(manifest_path),
        "warnings": warnings,
        "runs": selected,
        "notes": [
            "This root submission script is a training evidence/status entrypoint.",
            "It is not a real training runner.",
            "Any future real training code should live under scripts/research/.",
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
        parser.error("Thiếu subcommand. Dùng `status` để xem training evidence/status.")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
