from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from carm.runner import AgentRunner
from carm.training import load_training_config
from tools.base import ToolManager
from tools.bigmodel_tool import BigModelProxyTool
from tools.calc_tool import CalculatorTool
from tools.code_tool import CodeExecutorTool
from tools.search_tool import SearchTool


def load_eval_prompts(path: str | Path = "configs/pretrain_eval.json") -> list[dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = payload.get("prompts", []) if isinstance(payload, dict) else []
    return [item for item in prompts if isinstance(item, dict) and str(item.get("prompt", "")).strip()]


def build_tool_manager() -> ToolManager:
    return ToolManager(
        [
            SearchTool(),
            CalculatorTool(),
            CodeExecutorTool(),
            BigModelProxyTool(),
        ]
    )


def write_eval_training_config(path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "training": {
                    "mode": "two_stage",
                    "online_evolution": {
                        "enabled": True,
                        "allow_episode_learning": False,
                        "allow_user_signals": False,
                        "signal_state_path": str(Path(path).with_name("eval_evolution_state.json")),
                        "signal_log_path": str(Path(path).with_name("eval_signals.jsonl")),
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def build_runner_from_state_dir(source_dir: Path | None, workspace: Path) -> AgentRunner:
    workspace.mkdir(parents=True, exist_ok=True)
    training_config_path = workspace / "eval_training.json"
    write_eval_training_config(training_config_path)

    policy_state = workspace / "policy_state.json"
    concept_state = workspace / "concept_state.json"
    core_state = workspace / "core_state.json"
    evolution_state = workspace / "eval_evolution_state.json"
    if source_dir is not None:
        for name in ("policy_state.json", "concept_state.json", "core_state.json", "evolution_state.json"):
            source = source_dir / name
            if source.exists():
                target = workspace / ("eval_evolution_state.json" if name == "evolution_state.json" else name)
                shutil.copyfile(source, target)

    return AgentRunner(
        build_tool_manager(),
        experience_path=workspace / "episodes.jsonl",
        policy_state_path=policy_state,
        concept_state_path=concept_state,
        core_state_path=core_state,
        review_path=workspace / "reviews.jsonl",
        controls_path=workspace / "runtime_controls.json",
        training_config_path=training_config_path,
    )


def evaluate_runner(runner: AgentRunner, prompts: list[dict[str, str]]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    tool_matches = 0
    structured_writes = 0
    by_logic_skill: dict[str, dict[str, float]] = {}

    for item in prompts:
        prompt = str(item.get("prompt", ""))
        expected_tool = str(item.get("expected_tool", ""))
        logic_skill = str(item.get("logic_skill", ""))
        answer, trace = runner.run(prompt)
        used_tool = next((step.selected_tool for step in trace.steps if step.selected_tool), "")
        if expected_tool and used_tool == expected_tool:
            tool_matches += 1
        if any(step.target_slot in {"PLAN", "HYP", "DRAFT"} for step in trace.steps):
            structured_writes += 1
        bucket = by_logic_skill.setdefault(logic_skill or "unknown", {"count": 0.0, "tool_match": 0.0, "step_sum": 0.0})
        bucket["count"] += 1
        bucket["tool_match"] += 1 if (used_tool == expected_tool and expected_tool) else 0
        bucket["step_sum"] += len(trace.actions)
        rows.append(
            {
                "id": str(item.get("id", "")),
                "prompt": prompt,
                "logic_skill": logic_skill,
                "expected_tool": expected_tool,
                "used_tool": used_tool,
                "tool_match": used_tool == expected_tool if expected_tool else False,
                "actions": list(trace.actions),
                "step_count": len(trace.actions),
                "structured_write": any(step.target_slot in {"PLAN", "HYP", "DRAFT"} for step in trace.steps),
                "answer_preview": answer.splitlines()[0] if answer else "",
            }
        )

    total = max(1, len(prompts))
    return {
        "rows": rows,
        "summary": {
            "prompt_count": len(prompts),
            "tool_match_rate": round(tool_matches / total, 4),
            "structured_write_rate": round(structured_writes / total, 4),
            "avg_step_count": round(sum(row["step_count"] for row in rows) / total, 4),
            "by_logic_skill": {
                key: {
                    "count": int(value["count"]),
                    "tool_match_rate": round(value["tool_match"] / max(value["count"], 1.0), 4),
                    "avg_step_count": round(value["step_sum"] / max(value["count"], 1.0), 4),
                }
                for key, value in by_logic_skill.items()
            },
        },
    }


def compare_results(baseline: dict[str, object], pretrained: dict[str, object]) -> dict[str, object]:
    baseline_summary = baseline["summary"]
    pretrained_summary = pretrained["summary"]
    by_id = {str(row["id"]): row for row in baseline["rows"]}
    deltas: list[dict[str, object]] = []
    for row in pretrained["rows"]:
        baseline_row = by_id.get(str(row["id"]), {})
        deltas.append(
            {
                "id": row["id"],
                "logic_skill": row.get("logic_skill", ""),
                "expected_tool": row["expected_tool"],
                "baseline_used_tool": baseline_row.get("used_tool", ""),
                "pretrained_used_tool": row["used_tool"],
                "baseline_actions": baseline_row.get("actions", []),
                "pretrained_actions": row["actions"],
                "baseline_tool_match": baseline_row.get("tool_match", False),
                "pretrained_tool_match": row["tool_match"],
            }
        )
    return {
        "baseline": baseline_summary,
        "pretrained": pretrained_summary,
        "delta": {
            "tool_match_rate": round(float(pretrained_summary["tool_match_rate"]) - float(baseline_summary["tool_match_rate"]), 4),
            "structured_write_rate": round(float(pretrained_summary["structured_write_rate"]) - float(baseline_summary["structured_write_rate"]), 4),
            "avg_step_count": round(float(pretrained_summary["avg_step_count"]) - float(baseline_summary["avg_step_count"]), 4),
        },
        "rows": deltas,
    }


def main() -> int:
    prompts = load_eval_prompts()
    training = load_training_config("configs/training.yaml")
    artifact_dir = Path(str(training.get("training", {}).get("pretraining", {}).get("artifact_dir", "data/pretrain")))

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        baseline_runner = build_runner_from_state_dir(None, root / "baseline")
        pretrained_runner = build_runner_from_state_dir(artifact_dir, root / "pretrained")
        baseline = evaluate_runner(baseline_runner, prompts)
        pretrained = evaluate_runner(pretrained_runner, prompts)
        comparison = compare_results(baseline, pretrained)

    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
