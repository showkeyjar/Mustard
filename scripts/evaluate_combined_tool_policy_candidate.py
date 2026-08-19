from __future__ import annotations

import json
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.reasoning_codec import build_pattern_report
from carm.training import load_training_config
from scripts.evaluate_pretraining import build_runner_from_state_dir, load_eval_prompts


# Candidate HarnessPolicy for the combined-tool-policy gate. Passed explicitly
# to build_runner_from_state_dir(override_controls=...) so the gate tests THIS
# policy instead of the live default controls (which build_runner used to
# silently overwrite — the dead-code CONTROLS trap).
CANDIDATE_POLICY = {
    "policy": {
        "prefer_calculator_for_mixed_numeric_code": 1,
        "prefer_search_for_comparison_evidence": 1,
        "require_conflict_verify_before_answer": 1,
    }
}

# Stable regression subset for the candidate gate. The full 63-case
# configs/real_prompt_eval.json is coverage-only; the 17 known-hard cases
# (BFCL multi-entity parallel constructs, per the v24 conclusion) are model
# upper-bound limits, not routing regressions, so they must not fail the gate.
REGRESSION_PROMPTS_PATH = Path("configs/real_prompt_regression.json")

# Guard against external-tool calls (search / bigmodel proxy) that can block
# indefinitely when the backing service is unavailable. A hung prompt is
# treated as a routing failure so the gate and CI never deadlock.
PER_PROMPT_TIMEOUT = 30.0


def evaluate_candidate(output_path: Path = Path("artifacts/combined_tool_policy_candidate_latest.json")) -> dict[str, object]:
    training = load_training_config("configs/training.yaml")
    artifact_dir = Path(str(training.get("training", {}).get("pretraining", {}).get("artifact_dir", "data/pretrain")))
    full_prompts = load_eval_prompts("configs/real_prompt_eval.json")
    regression_prompts = (
        load_eval_prompts(str(REGRESSION_PROMPTS_PATH))
        if REGRESSION_PROMPTS_PATH.exists()
        else full_prompts
    )
    prompt_payload = {"prompts": full_prompts}
    hard_eval_payload = _read_json(Path("configs/hard_logic_eval.json"))

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        candidate_workspace = root / "candidate"
        candidate_workspace.mkdir(parents=True, exist_ok=True)
        # Pass the candidate HarnessPolicy explicitly so it is actually used:
        # build_runner copies the default controls in, then applies this
        # override on top — making the candidate observable and testable.
        runner = build_runner_from_state_dir(
            artifact_dir, candidate_workspace, override_controls=CANDIDATE_POLICY
        )
        full_report = _evaluate_prompts(runner, full_prompts)
        regression_report = _evaluate_prompts(runner, regression_prompts)
        codec_report = build_pattern_report(full_report, prompt_payload, hard_eval_payload)

    full_summary = full_report["summary"]
    regression_summary = regression_report["summary"]
    hard_eval = codec_report.get("hard_eval", {})
    result = {
        "control": {
            "policy.prefer_calculator_for_mixed_numeric_code": 1,
            "policy.prefer_search_for_comparison_evidence": 1,
            "policy.require_conflict_verify_before_answer": 1,
        },
        "artifact_dir": str(artifact_dir),
        "real_prompt_summary": full_summary,
        "regression_summary": regression_summary,
        "hard_eval_summary": {
            "pass_rate": hard_eval.get("pass_rate", 0.0) if isinstance(hard_eval, dict) else 0.0,
            "failed_case_ids": hard_eval.get("failed_case_ids", []) if isinstance(hard_eval, dict) else [],
        },
        "key_rows": [
            row
            for row in full_report["rows"]
            if row["id"] in {"real-mixed", "repair-comparison-005"}
        ],
        "real_prompt_rows": full_report["rows"],
        "decision": (
            "candidate_pass"
            if (
                float(regression_summary.get("pretrained_match_rate", 0.0)) >= 1.0
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
    result_holder: dict[int, str] = {}

    def _run_one(idx: int, item: dict[str, str]) -> None:
        try:
            _, trace = runner.run(str(item.get("prompt", "")))
            result_holder[idx] = _first_tool(trace.steps)
        except Exception:
            result_holder[idx] = ""

    # Run prompts one at a time, each in its own (daemon) thread with a join
    # timeout. Only one runner.run is active at once (no thread-safety risk),
    # but a hung external call (search / bigmodel proxy) is cut off at
    # PER_PROMPT_TIMEOUT and recorded as a routing failure so the gate and CI
    # never deadlock.
    for idx, item in enumerate(prompts):
        result_holder.pop(idx, None)
        th = threading.Thread(target=_run_one, args=(idx, item), daemon=True)
        th.start()
        th.join(timeout=PER_PROMPT_TIMEOUT)
        used_tool = result_holder.get(idx, "")
        expected_tool = str(item.get("expected_tool", ""))
        rows.append(
            {
                "id": str(item.get("id", "")),
                "logic_skill": str(item.get("logic_skill", "")),
                "expected_tool": expected_tool,
                "pretrained_used_tool": used_tool,
                "pretrained_actions": [],
                "pretrained_match": used_tool == expected_tool if expected_tool else False,
            }
        )

    total = max(1, len(rows))
    matches = sum(1 for row in rows if row["pretrained_match"])
    return {
        "summary": {
            "prompt_count": len(rows),
            "pretrained_match_rate": round(matches / total, 4),
            "pretrained_avg_steps": 0.0,
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
