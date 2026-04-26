from __future__ import annotations

import json
import sys
from pathlib import Path

from carm.attention_flow import build_training_views, nodes_from_payloads, training_views_to_jsonl


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


def export_attention_training_views(
    input_path: Path = Path("data/attention/attention_flow.jsonl"),
    output_path: Path = Path("data/attention/training_views.jsonl"),
) -> list[dict[str, object]]:
    nodes = nodes_from_payloads(_load_jsonl(input_path))
    views = build_training_views(nodes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = training_views_to_jsonl(views)
    output_path.write_text((payload + "\n") if payload else "", encoding="utf-8")
    return [view.to_dict() for view in views]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    input_path = Path(args[0]) if len(args) >= 1 else Path("data/attention/attention_flow.jsonl")
    output_path = Path(args[1]) if len(args) >= 2 else Path("data/attention/training_views.jsonl")
    views = export_attention_training_views(input_path, output_path)
    print(json.dumps({"view_count": len(views), "output_path": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
