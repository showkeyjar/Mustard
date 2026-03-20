from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from carm.runner import AgentRunner
from carm.runtime_controls import load_control_state
from scripts.apply_slow_path_actions import main as apply_main
from scripts.consolidate_reviews import main as consolidate_main
from scripts.evaluate_control_versions import main as evaluate_main
from scripts.judge_control_rollout import main as judge_main
from scripts.system_status import main as status_main
from tools.base import ToolManager
from tools.bigmodel_tool import BigModelProxyTool
from tools.calc_tool import CalculatorTool
from tools.code_tool import CodeExecutorTool
from tools.search_tool import SearchTool


DEFAULT_CONFIG_PATH = Path("configs/control_cycle.json")
DEFAULT_PROMPT_SET = "default"


def build_runner() -> AgentRunner:
    return AgentRunner(
        ToolManager(
            [
                SearchTool(),
                CalculatorTool(),
                CodeExecutorTool(),
                BigModelProxyTool(),
            ]
        ),
        experience_path=Path(os.environ.get("CARM_EXPERIENCE_PATH", "data/experience/episodes.jsonl")),
        policy_state_path=Path(os.environ.get("CARM_POLICY_STATE_PATH", "data/experience/policy_state.json")),
        concept_state_path=Path(os.environ.get("CARM_CONCEPT_STATE_PATH", "data/experience/concept_state.json")),
        core_state_path=Path(os.environ.get("CARM_CORE_STATE_PATH", "data/experience/core_state.json")),
        review_path=Path(os.environ.get("CARM_REVIEW_PATH", "data/review/reviews.jsonl")),
        controls_path=Path(os.environ.get("CARM_CONTROLS_PATH", "data/control/runtime_controls.json")),
    )


def load_sampling_prompts(
    config_path: Path | None = None,
    prompt_set: str = DEFAULT_PROMPT_SET,
) -> list[dict[str, str]]:
    config_path = config_path or Path(os.environ.get("CARM_CONTROL_CYCLE_CONFIG", str(DEFAULT_CONFIG_PATH)))
    if not config_path.exists():
        return []

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []

    prompt_sets = payload.get("prompt_sets", {})
    if not isinstance(prompt_sets, dict):
        return []

    prompts = prompt_sets.get(prompt_set, [])
    if not isinstance(prompts, list):
        return []
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(prompts):
        if isinstance(item, str):
            prompt = item.strip()
            if not prompt:
                continue
            normalized.append(
                {
                    "id": f"{prompt_set}-{index + 1}",
                    "prompt": prompt,
                    "tag": "",
                    "expected_tool": "",
                }
            )
            continue
        if isinstance(item, dict):
            prompt = str(item.get("prompt", "")).strip()
            if not prompt:
                continue
            normalized.append(
                {
                    "id": str(item.get("id", f"{prompt_set}-{index + 1}")).strip() or f"{prompt_set}-{index + 1}",
                    "prompt": prompt,
                    "tag": str(item.get("tag", "")).strip(),
                    "expected_tool": str(item.get("expected_tool", "")).strip(),
                }
            )
    return normalized


def append_sample_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    prompt_set = os.environ.get("CARM_CONTROL_PROMPT_SET", DEFAULT_PROMPT_SET)
    prompts = (
        [{"id": f"cli-{index + 1}", "prompt": prompt.strip(), "tag": "", "expected_tool": ""} for index, prompt in enumerate(argv) if prompt.strip()]
        or load_sampling_prompts(prompt_set=prompt_set)
    )
    control_state_path = Path(os.environ.get("CARM_CONTROL_STATE_PATH", "data/control/control_state.json"))
    samples_path = Path(os.environ.get("CARM_CONTROL_SAMPLES_PATH", "data/control/control_cycle_samples.jsonl"))

    print("[1/5] consolidate reviews")
    consolidate_main()

    print("[2/5] apply slow-path actions")
    apply_main()

    control_state = load_control_state(control_state_path)
    if str(control_state.get("rollout_status", "stable")) == "candidate":
        print("[3/5] run candidate sampling prompts")
        runner = build_runner()
        budget = int(control_state.get("candidate_episode_budget", 0) or 0)
        active_prompts = prompts or load_sampling_prompts(prompt_set=DEFAULT_PROMPT_SET)
        for task in active_prompts[: max(1, budget)]:
            prompt = task["prompt"]
            answer, trace = runner.run(prompt)
            used_tool = next((step.selected_tool for step in trace.steps if step.selected_tool), "")
            sample_record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "control_version": control_state.get("candidate_version", control_state.get("current_version", "")),
                "task_id": task["id"],
                "prompt": prompt,
                "tag": task.get("tag", ""),
                "expected_tool": task.get("expected_tool", ""),
                "used_tool": used_tool,
                "tool_match": bool(task.get("expected_tool")) and used_tool == task.get("expected_tool"),
                "answer_preview": answer.splitlines()[0] if answer else "",
                "step_count": len(trace.steps),
                "final_action": trace.actions[-1] if trace.actions else "",
            }
            append_sample_record(samples_path, sample_record)
            print(f"sampled> {prompt}")
            print(answer.splitlines()[0] if answer else "(empty answer)")
    else:
        print("[3/5] no candidate rollout; skip sampling")

    print("[4/5] evaluate control versions")
    evaluate_main()

    print("[5/5] judge rollout and print status")
    judge_main()
    status_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
