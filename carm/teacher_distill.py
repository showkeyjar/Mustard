from __future__ import annotations

import json
from pathlib import Path

from carm.pretrain_data import PretrainSample, annotate_sample, infer_logic_skill, infer_task_type
from tools.base import ToolManager
from tools.bigmodel_tool import BigModelProxyTool


def build_teacher_manager() -> ToolManager:
    return ToolManager([BigModelProxyTool()])


def distill_prompt_with_teacher(prompt: str, tool_manager: ToolManager | None = None) -> PretrainSample:
    manager = tool_manager or build_teacher_manager()
    result = manager.execute("bigmodel_proxy", prompt, {"mode": "distill"})
    payload = json.loads(result.result) if result.ok and result.result.strip().startswith("{") else {}

    task_type = str(payload.get("task_type") or infer_task_type(prompt))
    logic_skill = str(payload.get("logic_skill") or infer_logic_skill(prompt, task_type))
    sample = PretrainSample(
        user_input=prompt.strip(),
        task_type=task_type,
        source_type="teacher_distill",
        expected_tool=str(payload.get("expected_tool", "search")),
        target_slot=str(payload.get("target_slot", "PLAN")),
        logic_skill=logic_skill,
        plan_summary=str(payload.get("plan_summary", "")),
        plan_action_items=[str(item) for item in payload.get("plan_action_items", []) if str(item).strip()],
        plan_unknowns=[str(item) for item in payload.get("plan_unknowns", []) if str(item).strip()],
        evidence_targets=[str(item) for item in payload.get("evidence_targets", []) if str(item).strip()],
        draft_summary=str(payload.get("draft_summary", "")),
        quality_score=float(payload.get("quality_score", result.confidence or 0.9)),
        metadata={"teacher_source": result.source, "teacher_confidence": result.confidence},
    )
    return annotate_sample(sample)


def distill_prompts_with_teacher(
    prompts: list[str],
    *,
    limit: int | None = None,
    tool_manager: ToolManager | None = None,
) -> list[PretrainSample]:
    manager = tool_manager or build_teacher_manager()
    unique_prompts: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        text = prompt.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique_prompts.append(text)

    selected = unique_prompts[:limit] if limit is not None else unique_prompts
    return [distill_prompt_with_teacher(prompt, manager) for prompt in selected]


def export_teacher_samples(path: str | Path, samples: list[PretrainSample]) -> None:
    from carm.pretrain_data import save_pretrain_samples

    save_pretrain_samples(path, samples)
