from __future__ import annotations

import json
import sys
from pathlib import Path

from carm.attention_flow import AttentionTrainingView, build_training_view_report


def _load_jsonl(path: Path) -> list[AttentionTrainingView]:
    if not path.exists():
        return []
    views: list[AttentionTrainingView] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            views.append(
                AttentionTrainingView(
                    episode_id=str(payload.get("episode_id", "")),
                    step_idx=int(payload.get("step_idx", 0) or 0),
                    current_focus=str(payload.get("current_focus", "")),
                    next_focus=str(payload.get("next_focus", "")),
                    residual_pressure=[str(item) for item in payload.get("residual_pressure", []) if str(item)],
                    evidence_need=[str(item) for item in payload.get("evidence_need", []) if str(item)],
                    recommended_transition=str(payload.get("recommended_transition", "")),
                    recommended_action=str(payload.get("recommended_action", "")),
                    release_allowed=bool(payload.get("release_allowed", False)),
                    release_condition=str(payload.get("release_condition", "")),
                    supervision_note=str(payload.get("supervision_note", "")),
                )
            )
    return views


def write_attention_training_view_report(
    input_path: Path = Path("data/attention/training_views.jsonl"),
    output_path: Path = Path("artifacts/attention_training_views_latest.json"),
) -> dict[str, object]:
    report = build_training_view_report(_load_jsonl(input_path))
    report["sources"] = {"training_views": str(input_path)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    input_path = Path(args[0]) if len(args) >= 1 else Path("data/attention/training_views.jsonl")
    output_path = Path(args[1]) if len(args) >= 2 else Path("artifacts/attention_training_views_latest.json")
    report = write_attention_training_view_report(input_path, output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
