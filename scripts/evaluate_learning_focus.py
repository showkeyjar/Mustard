from __future__ import annotations

import json
from pathlib import Path

from carm.training import load_training_config
from scripts.evaluate_pretraining import load_eval_prompts
from scripts.evaluate_real_prompts import evaluate_isolated_prompts


def write_learning_focus_report(
    eval_path: str | Path = "data/eval/learning_focus_eval.json",
    output_path: str | Path = "artifacts/learning_focus_eval_latest.json",
) -> dict[str, object]:
    prompts = load_eval_prompts(eval_path)
    training = load_training_config("configs/training.yaml")
    artifact_dir = Path(str(training.get("training", {}).get("pretraining", {}).get("artifact_dir", "data/pretrain")))
    payload = evaluate_isolated_prompts(prompts, artifact_dir=artifact_dir)
    result = {
        "summary": payload.get("summary", {}),
        "rows": payload.get("rows", []),
        "sources": {"learning_focus_eval": str(eval_path)},
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    result = write_learning_focus_report()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
