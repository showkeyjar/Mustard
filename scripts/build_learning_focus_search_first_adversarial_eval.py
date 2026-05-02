from __future__ import annotations

import json
from pathlib import Path


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_learning_focus_search_first_adversarial_eval(
    root: Path = Path("."),
    *,
    output_path: str | Path = "data/eval/learning_focus_search_first_adversarial_eval.json",
) -> dict[str, object]:
    eval_payload = _read_json(root / "data" / "eval" / "learning_focus_eval.json")
    latest_payload = _read_json(root / "artifacts" / "learning_focus_eval_latest.json")

    prompt_rows = eval_payload.get("prompts", []) if isinstance(eval_payload, dict) else []
    latest_rows = latest_payload.get("rows", []) if isinstance(latest_payload, dict) else []
    if not isinstance(prompt_rows, list):
        prompt_rows = []
    if not isinstance(latest_rows, list):
        latest_rows = []

    prompt_by_id = {
        str(row.get("id", "")).strip(): row
        for row in prompt_rows
        if isinstance(row, dict) and str(row.get("id", "")).strip()
    }

    target_rows = [
        row
        for row in latest_rows
        if isinstance(row, dict)
        and str(row.get("logic_skill", "")).strip() == "evidence_judgment"
        and str(row.get("expected_tool", "")).strip() == "search"
        and not bool(row.get("pretrained_match", True))
    ]

    templates = {
        "tool-use stability under ambiguity": [
            (
                "source_comparison_first",
                "你要整理公开 agent 设计里关于 tool-use stability under ambiguity 的做法。请先去检索公开资料，对比至少两份来源后再给 Mustard 的低风险实验，不能只凭记忆或抽象推断直接作答。",
            ),
            (
                "evidence_gate_before_takeaway",
                "围绕 tool-use stability under ambiguity，用户要你总结可借鉴/不建议/待观察三类要点，并指出哪条来自公开资料、哪条只是待验证假设。当前第一步该调用什么工具，为什么不能直接归纳？",
            ),
        ],
        "conflict-aware answer suppression": [
            (
                "public_pattern_reconciliation",
                "你需要比较公开 agent 设计资料里对 conflict-aware answer suppression 的不同表述，再提炼 Mustard 可以离线验证的 suppression 规则。没有检索到外部依据前，不允许直接下结论。",
            ),
            (
                "citation_needed_before_policy",
                "用户问 conflict-aware answer suppression 值不值得进 Mustard。请先检索公开设计思想或论文证据，再给出实验建议；不要把这个问题误当成一个可以直接算出答案的题。",
            ),
        ],
        "small reasoning model routing": [
            (
                "routing_evidence_before_rule",
                "你要总结公开资料里 small reasoning model routing 的适用条件、失败边界和低风险接入方式。先找证据，再决定 Mustard 是否该学这套路由，不要直接凭常识给规则。",
            ),
            (
                "compare_public_guidance",
                "请基于公开 agent 设计资料，对比 small reasoning model routing 在不同系统里的约束，再提出一个 Mustard 可验证实验。这个任务第一步为什么必须先检索而不是直接推理？",
            ),
        ],
    }

    prompts: list[dict[str, object]] = []
    for row in target_rows:
        source_id = str(row.get("id", "")).strip()
        source_prompt = prompt_by_id.get(source_id, {})
        base_prompt = str(source_prompt.get("prompt", "")).strip()
        theme = ""
        if "主题=" in base_prompt:
            after = base_prompt.split("主题=", 1)[1]
            theme = after.split("。", 1)[0].strip()
        variants = templates.get(theme)
        if not variants:
            variants = [
                (
                    "search_before_synthesis",
                    f"下面这类公开 agent 设计学习题目前仍容易误路由到 calculator：{base_prompt}。请把它改写成一个更强调“先检索公开证据，再综合输出实验建议”的任务。",
                ),
                (
                    "search_before_experiment",
                    f"围绕这个主题先检索公开资料、再输出 Mustard 的低风险实验，不能直接归纳：{base_prompt}",
                ),
            ]
        for mutation, prompt in variants:
            prompts.append(
                {
                    "id": f"search-first-adversarial-{len(prompts) + 1:03d}",
                    "prompt": prompt,
                    "expected_tool": "search",
                    "logic_skill": "evidence_judgment",
                    "source_type": "learning_focus_search_first_adversarial",
                    "mutation": mutation,
                    "source_learning_focus_id": source_id,
                    "source_prompt": base_prompt,
                }
            )

    payload = {
        "summary": {
            "target_failure_count": len(target_rows),
            "prompt_count": len(prompts),
            "target_ids": [str(row.get("id", "")).strip() for row in target_rows],
        },
        "prompts": prompts,
    }

    target = root / Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    payload = build_learning_focus_search_first_adversarial_eval()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
