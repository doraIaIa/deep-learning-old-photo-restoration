from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Protocol/status runner cho ablation trong repo submission. "
            "Script này không phải full LPIPS/FID ablation runner và không tự chạy pipeline/eval thật trong Phase 1B."
        )
    )
    parser.add_argument(
        "--mode",
        default="protocol_status",
        choices=["protocol_status", "smoke_manifest"],
        help="Mode hiện tại chỉ mô tả protocol và trạng thái artifact; không chạy pipeline thật.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Nếu truyền, script sẽ ghi protocol/status ra thư mục này.",
    )
    parser.add_argument(
        "--write-protocol-status",
        action="store_true",
        help="Bật cờ này hoặc truyền --output-dir để ghi JSON/Markdown status ra ngoài repo.",
    )
    parser.add_argument(
        "--available-artifacts-root",
        type=Path,
        default=None,
        help="Chỉ kiểm tra artifact tồn tại/không tồn tại dưới root này; không chạy pipeline.",
    )
    return parser


def resolve_optional_path(path: Path | None) -> Path | None:
    return None if path is None else path.resolve()


def build_protocol_definition() -> list[dict[str, str]]:
    return [
        {
            "case_id": "opencv_baseline",
            "description": "OpenCV baseline nếu repo thực sự có backend và artifact tương ứng.",
            "status": "protocol_only",
        },
        {
            "case_id": "unet_mask_plus_classical_inpaint",
            "description": "Mask từ segmentation rồi ghép backend cổ điển nếu có implementation thật.",
            "status": "protocol_only",
        },
        {
            "case_id": "ground_truth_mask_plus_pretrained_lama",
            "description": "Oracle/ground-truth mask + official/pretrained LaMa.",
            "status": "partially evidenced by smoke artifacts",
        },
        {
            "case_id": "predicted_mask_plus_pretrained_lama",
            "description": "Predicted/hybrid mask + official/pretrained LaMa.",
            "status": "partially evidenced by smoke artifacts",
        },
        {
            "case_id": "full_pipeline_plus_optional_face_module",
            "description": "Full pipeline + face module nếu sau này có evidence đủ mạnh.",
            "status": "future_work_or_optional",
        },
    ]


def build_known_limitations() -> list[str]:
    return [
        "Full quantitative ablation chưa được đóng gói trong repo submission.",
        "LPIPS/FID/masked-region LPIPS chưa có artifact đủ để claim đã hoàn tất.",
        "LaMa hiện là official/pretrained wrapper, không phải LaMa fine-tune.",
        "Module 3/CodeFormer không được claim identity preservation.",
        "Script này chỉ là protocol/status runner trong Phase 1B, không phải full experimental orchestrator.",
    ]


def inspect_artifacts(root: Path | None) -> list[dict[str, str]]:
    if root is None:
        return []
    targets = [
        "examples/outputs/seg_smoke_demo3/metadata.json",
        "examples/outputs/pipeline_smoke_demo3/metadata.json",
        "examples/outputs/gradio_smoke_demo3/metadata.json",
        "examples/outputs/readiness_mask_bypass_regression/metadata.json",
    ]
    rows: list[dict[str, str]] = []
    for target in targets:
        candidate = root / target
        rows.append(
            {
                "path": str(candidate),
                "exists": "yes" if candidate.exists() else "no",
            }
        )
    return rows


def build_status_payload(mode: str, artifacts_root: Path | None) -> dict[str, object]:
    return {
        "mode": mode,
        "runner_scope": "protocol_status_only",
        "protocol_cases": build_protocol_definition(),
        "known_limitations": build_known_limitations(),
        "artifacts_root_checked": str(artifacts_root) if artifacts_root is not None else None,
        "artifact_checks": inspect_artifacts(artifacts_root),
        "phase1b_note": (
            "Script này hiện mô tả protocol/status cho ablation và không chạy full LPIPS/FID/full ablation. "
            "Các runner đánh giá sâu hơn chỉ nên bổ sung ở phase sau khi có evidence tương ứng."
        ),
    }


def render_markdown(payload: dict[str, object]) -> str:
    protocol_cases = payload["protocol_cases"]
    artifact_checks = payload["artifact_checks"]
    lines = [
        "# Ablation Protocol Status",
        "",
        f"- mode: `{payload['mode']}`",
        f"- runner_scope: `{payload['runner_scope']}`",
        "",
        "## Protocol Cases",
    ]
    for item in protocol_cases:
        assert isinstance(item, dict)
        lines.append(f"- `{item['case_id']}`: {item['description']} | status={item['status']}")
    lines.extend(["", "## Known Limitations"])
    for item in payload["known_limitations"]:
        lines.append(f"- {item}")
    if artifact_checks:
        lines.extend(["", "## Artifact Checks"])
        for item in artifact_checks:
            assert isinstance(item, dict)
            lines.append(f"- `{item['path']}` -> exists={item['exists']}")
    return "\n".join(lines)


def write_protocol_status(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_protocol_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "ablation_protocol_status.md").write_text(
        render_markdown(payload),
        encoding="utf-8",
    )


def main() -> int:
    configure_utf8_stdio()
    args = build_parser().parse_args()
    output_dir = resolve_optional_path(args.output_dir)
    artifacts_root = resolve_optional_path(args.available_artifacts_root)

    if args.write_protocol_status and output_dir is None:
        raise SystemExit("Thiếu --output-dir khi dùng --write-protocol-status.")

    payload = build_status_payload(args.mode, artifacts_root)
    should_write = args.write_protocol_status or output_dir is not None
    if should_write:
        assert output_dir is not None
        write_protocol_status(output_dir, payload)
        print(f"ablation_protocol_status_json: {output_dir / 'ablation_protocol_status.json'}")
        print(f"ablation_protocol_status_md: {output_dir / 'ablation_protocol_status.md'}")
    else:
        print("protocol_status_only")
        print("Script này không chạy full ablation trong Phase 1B.")
        print("Dùng --output-dir để ghi protocol/status ra file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
