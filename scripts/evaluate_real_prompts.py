from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.training import load_training_config
from scripts.evaluate_pretraining import build_runner_from_state_dir, load_eval_prompts


def evaluate_isolated_prompts(
    prompts: list[dict[str, str]],
    *,
    artifact_dir: Path,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []

    for item in prompts:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_runner = build_runner_from_state_dir(None, root / "baseline")
            pretrained_runner = build_runner_from_state_dir(artifact_dir, root / "pretrained")

            _, baseline_trace = baseline_runner.run(str(item.get("prompt", "")))
            _, pretrained_trace = pretrained_runner.run(str(item.get("prompt", "")))

        expected_tool = str(item.get("expected_tool", ""))
        baseline_used_tool = next((step.selected_tool for step in baseline_trace.steps if step.selected_tool), "")
        pretrained_used_tool = next((step.selected_tool for step in pretrained_trace.steps if step.selected_tool), "")

        rows.append(
            {
                "id": str(item.get("id", "")),
                "logic_skill": str(item.get("logic_skill", "")),
                "expected_tool": expected_tool,
                "baseline_used_tool": baseline_used_tool,
                "pretrained_used_tool": pretrained_used_tool,
                "baseline_actions": list(baseline_trace.actions),
                "pretrained_actions": list(pretrained_trace.actions),
                "baseline_match": baseline_used_tool == expected_tool if expected_tool else False,
                "pretrained_match": pretrained_used_tool == expected_tool if expected_tool else False,
            }
        )

    total = max(1, len(rows))
    baseline_matches = sum(1 for row in rows if row["baseline_match"])
    pretrained_matches = sum(1 for row in rows if row["pretrained_match"])
    return {
        "summary": {
            "prompt_count": len(rows),
            "baseline_match_rate": round(baseline_matches / total, 4),
            "pretrained_match_rate": round(pretrained_matches / total, 4),
            "baseline_avg_steps": round(sum(len(row["baseline_actions"]) for row in rows) / total, 4),
            "pretrained_avg_steps": round(sum(len(row["pretrained_actions"]) for row in rows) / total, 4),
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    prompt_path = Path(args[0]) if args else Path("configs/real_prompt_eval.json")
    prompts = load_eval_prompts(prompt_path)
    training = load_training_config("configs/training.yaml")
    artifact_dir = Path(str(training.get("training", {}).get("pretraining", {}).get("artifact_dir", "data/pretrain")))

    result = evaluate_isolated_prompts(prompts, artifact_dir=artifact_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
