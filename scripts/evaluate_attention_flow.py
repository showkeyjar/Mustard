from __future__ import annotations

import json
import sys
from pathlib import Path

from carm.attention_flow import build_attention_report, nodes_from_payloads


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payloads: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                payloads.append(payload)
    return payloads


def write_attention_flow_report(
    input_path: Path = Path("data/attention/attention_flow.jsonl"),
    output_path: Path = Path("artifacts/attention_flow_latest.json"),
) -> dict[str, object]:
    nodes = nodes_from_payloads(_load_jsonl(input_path))
    report = build_attention_report(nodes)
    report["sources"] = {"attention_flow": str(input_path)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    input_path = Path(args[0]) if len(args) >= 1 else Path("data/attention/attention_flow.jsonl")
    output_path = Path(args[1]) if len(args) >= 2 else Path("artifacts/attention_flow_latest.json")
    report = write_attention_flow_report(input_path, output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
