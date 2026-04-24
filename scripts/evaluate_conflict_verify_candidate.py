from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.evaluate_pretraining import build_runner_from_state_dir
from carm.training import load_training_config


REAL_CONFLICT_PROMPT = "两篇数据库迁移文章对是否要先加只读窗口给出了相反建议，这时候应该怎么处理？"


def evaluate_candidate(output_path: Path = Path("artifacts/conflict_verify_candidate_latest.json")) -> dict[str, object]:
    training = load_training_config("configs/training.yaml")
    artifact_dir = Path(str(training.get("training", {}).get("pretraining", {}).get("artifact_dir", "data/pretrain")))
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        baseline_runner = build_runner_from_state_dir(artifact_dir, root / "baseline")
        candidate_workspace = root / "candidate"
        candidate_workspace.mkdir(parents=True, exist_ok=True)
        (candidate_workspace / "runtime_controls.json").write_text(
            json.dumps(
                {
                    "policy": {
                        "require_conflict_verify_before_answer": 1,
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        candidate_runner = build_runner_from_state_dir(artifact_dir, candidate_workspace)

        _, baseline_trace = baseline_runner.run(REAL_CONFLICT_PROMPT)
        _, candidate_trace = candidate_runner.run(REAL_CONFLICT_PROMPT)

    result = {
        "prompt": REAL_CONFLICT_PROMPT,
        "control": {
            "policy.require_conflict_verify_before_answer": 1,
        },
        "artifact_dir": str(artifact_dir),
        "baseline_actions": list(baseline_trace.actions),
        "candidate_actions": list(candidate_trace.actions),
        "baseline_has_verify": "VERIFY" in baseline_trace.actions,
        "candidate_has_verify": "VERIFY" in candidate_trace.actions,
        "candidate_verify_before_answer": _before(candidate_trace.actions, "VERIFY", "ANSWER"),
        "candidate_changes_actions": list(baseline_trace.actions) != list(candidate_trace.actions),
        "decision": (
            "candidate_pass"
            if "VERIFY" in candidate_trace.actions and _before(candidate_trace.actions, "VERIFY", "ANSWER")
            else "candidate_fail"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _before(actions: list[str], left: str, right: str) -> bool:
    if left not in actions or right not in actions:
        return False
    return actions.index(left) < actions.index(right)


def main() -> int:
    result = evaluate_candidate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
