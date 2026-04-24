from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.reasoning_codec import build_pattern_report
from carm.training import load_training_config
from scripts.evaluate_pretraining import build_runner_from_state_dir, load_eval_prompts


CONTROLS = {
    "policy": {
        "prefer_calculator_for_mixed_numeric_code": 1,
        "prefer_search_for_comparison_evidence": 1,
    }
}


def evaluate_candidate(output_path: Path = Path("artifacts/combined_tool_policy_candidate_latest.json")) -> dict[str, object]:
    training = load_training_config("configs/training.yaml")
    artifact_dir = Path(str(training.get("training", {}).get("pretraining", {}).get("artifact_dir", "data/pretrain")))
    prompts = load_eval_prompts("configs/real_prompt_eval.json")
    prompt_payload = {"prompts": prompts}
    hard_eval_payload = _read_json(Path("configs/hard_logic_eval.json"))

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        candidate_workspace = root / "candidate"
        candidate_workspace.mkdir(parents=True, exist_ok=True)
        (candidate_workspace / "runtime_controls.json").write_text(
            json.dumps(CONTROLS, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        runner = build_runner_from_state_dir(artifact_dir, candidate_workspace)
        real_prompt_report = _evaluate_prompts(runner, prompts)
        codec_report = build_pattern_report(real_prompt_report, prompt_payload, hard_eval_payload)

    real_summary = real_prompt_report["summary"]
    hard_eval = codec_report.get("hard_eval", {})
    result = {
        "control": {
            "policy.prefer_calculator_for_mixed_numeric_code": 1,
            "policy.prefer_search_for_comparison_evidence": 1,
        },
        "artifact_dir": str(artifact_dir),
        "real_prompt_summary": real_summary,
        "hard_eval_summary": {
            "pass_rate": hard_eval.get("pass_rate", 0.0) if isinstance(hard_eval, dict) else 0.0,
            "failed_case_ids": hard_eval.get("failed_case_ids", []) if isinstance(hard_eval, dict) else [],
        },
        "key_rows": [
            row
            for row in real_prompt_report["rows"]
            if row["id"] in {"real-mixed", "repair-comparison-005"}
        ],
        "real_prompt_rows": real_prompt_report["rows"],
        "decision": (
            "candidate_pass"
            if (
                float(real_summary.get("pretrained_match_rate", 0.0)) >= 1.0
                and isinstance(hard_eval, dict)
                and float(hard_eval.get("pass_rate", 0.0)) >= 1.0
            )
            else "candidate_fail"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _evaluate_prompts(runner: object, prompts: list[dict[str, str]]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for item in prompts:
        _, trace = runner.run(str(item.get("prompt", "")))
        expected_tool = str(item.get("expected_tool", ""))
        used_tool = _first_tool(trace.steps)
        rows.append(
            {
                "id": str(item.get("id", "")),
                "logic_skill": str(item.get("logic_skill", "")),
                "expected_tool": expected_tool,
                "pretrained_used_tool": used_tool,
                "pretrained_actions": list(trace.actions),
                "pretrained_match": used_tool == expected_tool if expected_tool else False,
            }
        )

    total = max(1, len(rows))
    matches = sum(1 for row in rows if row["pretrained_match"])
    return {
        "summary": {
            "prompt_count": len(rows),
            "pretrained_match_rate": round(matches / total, 4),
            "pretrained_avg_steps": round(sum(len(row["pretrained_actions"]) for row in rows) / total, 4),
        },
        "rows": rows,
    }


def _first_tool(steps: object) -> str:
    return next((str(step.selected_tool) for step in steps if getattr(step, "selected_tool", "")), "")


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    result = evaluate_candidate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

