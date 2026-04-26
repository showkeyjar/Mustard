from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.training import load_training_config
from scripts.evaluate_pretraining import build_runner_from_state_dir, load_eval_prompts
from carm.reasoning_codec import build_pattern_report


REAL_MIXED_PROMPT = "这个问题里既有一段 Python 代码，又问到一次性迁移 48000 条数据预计分几批处理，每批 6000 条。你会先走哪类工具？"
GUARD_PROMPTS = [
    {
        "id": "mixed-numeric-code",
        "prompt": REAL_MIXED_PROMPT,
        "expected_tool": "calculator",
    },
    {
        "id": "plain-code",
        "prompt": "帮我理解这段 Python 脚本为什么会报空指针错误",
        "expected_tool": "code_executor",
    },
    {
        "id": "plain-calc",
        "prompt": "请计算 12 + 30 / 3",
        "expected_tool": "calculator",
    },
]


def evaluate_candidate(output_path: Path = Path("artifacts/tool_boundary_candidate_latest.json")) -> dict[str, object]:
    training = load_training_config("configs/training.yaml")
    artifact_dir = Path(str(training.get("training", {}).get("pretraining", {}).get("artifact_dir", "data/pretrain")))
    prompts = load_eval_prompts("configs/real_prompt_eval.json")
    prompt_payload = {"prompts": prompts}
    hard_eval_payload = _read_json(Path("configs/hard_logic_eval.json"))
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        baseline_runner = build_runner_from_state_dir(artifact_dir, root / "baseline")
        candidate_workspace = root / "candidate"
        candidate_workspace.mkdir(parents=True, exist_ok=True)
        (candidate_workspace / "runtime_controls.json").write_text(
            json.dumps(
                {
                    "policy": {
                        "prefer_calculator_for_mixed_numeric_code": 1,
                        "require_conflict_verify_before_answer": 1,
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        candidate_runner = build_runner_from_state_dir(artifact_dir, candidate_workspace)

        rows = []
        for item in GUARD_PROMPTS:
            _, baseline_trace = baseline_runner.run(str(item["prompt"]))
            _, candidate_trace = candidate_runner.run(str(item["prompt"]))
            baseline_tool = _first_tool(baseline_trace.steps)
            candidate_tool = _first_tool(candidate_trace.steps)
            rows.append(
                {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "expected_tool": item["expected_tool"],
                    "baseline_tool": baseline_tool,
                    "candidate_tool": candidate_tool,
                    "baseline_match": baseline_tool == item["expected_tool"],
                    "candidate_match": candidate_tool == item["expected_tool"],
                    "baseline_actions": list(baseline_trace.actions),
                    "candidate_actions": list(candidate_trace.actions),
                    "candidate_changes_tool": baseline_tool != candidate_tool,
                }
            )
        real_prompt_report = _evaluate_prompts(candidate_runner, prompts)
        codec_report = build_pattern_report(real_prompt_report, prompt_payload, hard_eval_payload)

    target_row = next(row for row in rows if row["id"] == "mixed-numeric-code")
    candidate_matches = sum(1 for row in rows if row["candidate_match"])
    real_summary = real_prompt_report["summary"]
    hard_eval = codec_report.get("hard_eval", {})
    result = {
        "prompt": REAL_MIXED_PROMPT,
        "control": {
            "policy.prefer_calculator_for_mixed_numeric_code": 1,
            "policy.require_conflict_verify_before_answer": 1,
        },
        "artifact_dir": str(artifact_dir),
        "expected_tool": target_row["expected_tool"],
        "baseline_tool": target_row["baseline_tool"],
        "candidate_tool": target_row["candidate_tool"],
        "baseline_actions": target_row["baseline_actions"],
        "candidate_actions": target_row["candidate_actions"],
        "candidate_changes_tool": target_row["candidate_changes_tool"],
        "guard_summary": {
            "case_count": len(rows),
            "candidate_match_rate": round(candidate_matches / max(len(rows), 1), 4),
            "failed_case_ids": [str(row["id"]) for row in rows if not row["candidate_match"]],
        },
        "real_prompt_summary": real_summary,
        "hard_eval_summary": {
            "pass_rate": hard_eval.get("pass_rate", 0.0) if isinstance(hard_eval, dict) else 0.0,
            "failed_case_ids": hard_eval.get("failed_case_ids", []) if isinstance(hard_eval, dict) else [],
        },
        "rows": rows,
        "real_prompt_rows": real_prompt_report["rows"],
        "decision": (
            "candidate_pass"
            if (
                target_row["candidate_tool"] == "calculator"
                and candidate_matches == len(rows)
                and float(real_summary.get("pretrained_match_rate", 0.0)) >= 0.95
                and isinstance(hard_eval, dict)
                and float(hard_eval.get("pass_rate", 0.0)) >= 1.0
            )
            else "candidate_fail"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _first_tool(steps: object) -> str:
    return next((str(step.selected_tool) for step in steps if getattr(step, "selected_tool", "")), "")


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
