from __future__ import annotations

import json
import sys
from pathlib import Path

from carm.reasoning_codec import build_pattern_report


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_reasoning_pattern_report(
    eval_path: Path = Path("data/eval/real_prompt_eval_latest.json"),
    prompt_path: Path = Path("configs/real_prompt_eval.json"),
    output_path: Path = Path("artifacts/reasoning_pattern_codec_latest.json"),
    hard_eval_path: Path = Path("configs/hard_logic_eval.json"),
) -> dict[str, object]:
    report = build_pattern_report(
        _read_json(eval_path),
        _read_json(prompt_path),
        _read_json(hard_eval_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    eval_path = Path(args[0]) if len(args) >= 1 else Path("data/eval/real_prompt_eval_latest.json")
    prompt_path = Path(args[1]) if len(args) >= 2 else Path("configs/real_prompt_eval.json")
    output_path = Path(args[2]) if len(args) >= 3 else Path("artifacts/reasoning_pattern_codec_latest.json")
    hard_eval_path = Path(args[3]) if len(args) >= 4 else Path("configs/hard_logic_eval.json")
    report = write_reasoning_pattern_report(eval_path, prompt_path, output_path, hard_eval_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
