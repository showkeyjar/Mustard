from __future__ import annotations

import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def build_learning_focus_eval(
    input_path: str | Path = "data/learning/learning_intake_samples.jsonl",
    output_path: str | Path = "data/eval/learning_focus_eval.json",
    *,
    limit: int = 12,
) -> dict[str, object]:
    source = Path(input_path)
    output = Path(output_path)
    rows = _load_jsonl(source)

    prompts: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        prompt = str(row.get("user_input", "")).strip()
        expected_tool = str(row.get("expected_tool", "")).strip()
        logic_skill = str(row.get("logic_skill", "")).strip()
        source_type = str(row.get("source_type", "")).strip()
        if not (prompt and expected_tool and logic_skill):
            continue
        key = "|".join([prompt, expected_tool, logic_skill])
        if key in seen:
            continue
        seen.add(key)
        prompts.append(
            {
                "id": f"learning-focus-{index:03d}",
                "prompt": prompt,
                "expected_tool": expected_tool,
                "logic_skill": logic_skill,
                "source_type": source_type,
            }
        )
        if len(prompts) >= limit:
            break

    payload = {"prompts": prompts}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    payload = build_learning_focus_eval()
    print(json.dumps({"prompt_count": len(payload.get("prompts", [])), "output_path": "data/eval/learning_focus_eval.json"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
