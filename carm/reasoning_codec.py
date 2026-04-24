from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


HARD_LOGIC_SKILLS = {
    "comparison",
    "conflict_detection",
    "result_integration",
    "termination_judgment",
    "tool_selection",
}


PATTERN_BY_SKILL = {
    "comparison": "compare_with_evidence",
    "conflict_detection": "conflict_first",
    "result_integration": "integrate_for_audience",
    "termination_judgment": "stop_or_continue",
    "tool_selection": "tool_boundary",
    "evidence_judgment": "evidence_check",
    "step_planning": "plan_then_verify",
    "constraint_planning": "constraint_plan",
}


@dataclass(frozen=True)
class ReasoningPatternRecord:
    case_id: str
    logic_skill: str
    pattern_id: str
    expected_tool: str
    used_tool: str
    fit_score: float
    residual_features: list[str]
    reconstruction_notes: list[str]
    is_hard_logic: bool
    action_count: int
    used_verify: bool
    tool_match: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def encode_eval_row(row: dict[str, Any], prompt: str = "") -> ReasoningPatternRecord:
    logic_skill = str(row.get("logic_skill", ""))
    pattern_id = PATTERN_BY_SKILL.get(logic_skill, "generic_reasoning")
    expected_tool = str(row.get("expected_tool", ""))
    used_tool = _used_tool(row)
    actions = _actions(row)
    used_verify = "VERIFY" in actions
    tool_match = bool(row.get("pretrained_match", row.get("pretrained_tool_match", False)))
    residual_features = _residual_features(
        row=row,
        prompt=prompt,
        pattern_id=pattern_id,
        actions=actions,
        expected_tool=expected_tool,
        used_tool=used_tool,
        tool_match=tool_match,
        used_verify=used_verify,
    )
    fit_score = _fit_score(
        pattern_id=pattern_id,
        tool_match=tool_match,
        residual_features=residual_features,
        used_verify=used_verify,
        action_count=len(actions),
    )
    return ReasoningPatternRecord(
        case_id=str(row.get("id", "")),
        logic_skill=logic_skill,
        pattern_id=pattern_id,
        expected_tool=expected_tool,
        used_tool=used_tool,
        fit_score=fit_score,
        residual_features=residual_features,
        reconstruction_notes=_reconstruction_notes(pattern_id, residual_features),
        is_hard_logic=logic_skill in HARD_LOGIC_SKILLS,
        action_count=len(actions),
        used_verify=used_verify,
        tool_match=tool_match,
    )


def build_pattern_report(
    eval_payload: dict[str, Any],
    prompt_payload: dict[str, Any] | None = None,
    hard_eval_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompts = _prompt_lookup(prompt_payload or {})
    rows = eval_payload.get("rows", [])
    if not isinstance(rows, list):
        rows = []

    records = [
        encode_eval_row(row, prompts.get(str(row.get("id", "")), ""))
        for row in rows
        if isinstance(row, dict)
    ]
    report = {
        "summary": _summary(records),
        "pattern_counts": _counts(record.pattern_id for record in records),
        "residual_counts": _counts(feature for record in records for feature in record.residual_features),
        "records": [record.to_dict() for record in records],
    }
    if hard_eval_payload:
        report["hard_eval"] = validate_hard_eval(records, hard_eval_payload)
    return report


def validate_hard_eval(
    records: list[ReasoningPatternRecord],
    hard_eval_payload: dict[str, Any],
) -> dict[str, Any]:
    record_by_id = {record.case_id: record for record in records}
    cases = hard_eval_payload.get("cases", [])
    if not isinstance(cases, list):
        cases = []

    rows: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id", ""))
        record = record_by_id.get(case_id)
        if record is None:
            rows.append(
                {
                    "id": case_id,
                    "pass": False,
                    "failure_reasons": ["missing_record"],
                    "expected_pattern": str(case.get("expected_pattern", "")),
                    "actual_pattern": "",
                }
            )
            continue

        required_residuals = _string_list(case.get("required_residuals", []))
        unacceptable_failures = _string_list(case.get("unacceptable_failures", []))
        failure_reasons: list[str] = []
        expected_pattern = str(case.get("expected_pattern", ""))
        if expected_pattern and record.pattern_id != expected_pattern:
            failure_reasons.append("pattern_mismatch")
        missing_residuals = [
            residual
            for residual in required_residuals
            if residual not in record.residual_features
        ]
        if missing_residuals:
            failure_reasons.append("missing_residuals:" + ",".join(missing_residuals))
        triggered_failures = [
            failure
            for failure in unacceptable_failures
            if failure in record.residual_features
        ]
        if triggered_failures:
            failure_reasons.append("unacceptable_failures:" + ",".join(triggered_failures))
        rows.append(
            {
                "id": case_id,
                "pass": not failure_reasons,
                "failure_reasons": failure_reasons,
                "expected_pattern": expected_pattern,
                "actual_pattern": record.pattern_id,
                "required_residuals": required_residuals,
                "actual_residuals": record.residual_features,
                "expected_decision": str(case.get("expected_decision", "")),
                "fit_score": record.fit_score,
            }
        )

    total = len(rows)
    passed = sum(1 for row in rows if row["pass"])
    return {
        "case_count": total,
        "pass_count": passed,
        "pass_rate": round(passed / max(total, 1), 4),
        "failed_case_ids": [str(row["id"]) for row in rows if not row["pass"]],
        "rows": rows,
    }


def _summary(records: list[ReasoningPatternRecord]) -> dict[str, Any]:
    total = len(records)
    hard = [record for record in records if record.is_hard_logic]
    residual_records = [record for record in records if record.residual_features]
    low_fit = [record for record in records if record.fit_score < 0.65]
    verify_expected = [
        record
        for record in records
        if any(
            feature in record.residual_features
            for feature in ("conflict_unresolved", "needs_evidence", "boundary_ambiguous")
        )
    ]
    return {
        "record_count": total,
        "hard_logic_count": len(hard),
        "avg_fit_score": _avg(record.fit_score for record in records),
        "hard_logic_avg_fit_score": _avg(record.fit_score for record in hard),
        "residual_explanation_rate": round(len(residual_records) / max(total, 1), 4),
        "hard_logic_residual_rate": round(
            sum(1 for record in hard if record.residual_features) / max(len(hard), 1),
            4,
        ),
        "verify_when_residual_risky_rate": round(
            sum(1 for record in verify_expected if record.used_verify) / max(len(verify_expected), 1),
            4,
        ),
        "low_fit_count": len(low_fit),
        "low_fit_case_ids": [record.case_id for record in low_fit[:10]],
    }


def _residual_features(
    *,
    row: dict[str, Any],
    prompt: str,
    pattern_id: str,
    actions: list[str],
    expected_tool: str,
    used_tool: str,
    tool_match: bool,
    used_verify: bool,
) -> list[str]:
    features: list[str] = []
    text = prompt.lower()

    if not tool_match:
        features.append("tool_mismatch")
    if expected_tool != used_tool and used_tool:
        features.append("tool_boundary_shift")
    if pattern_id == "conflict_first":
        features.append("conflict_unresolved")
        if not used_verify:
            features.append("missing_verify")
    if pattern_id == "compare_with_evidence":
        if _contains_any(prompt, ("证据", "资料", "来源", "比较", "对比")):
            features.append("needs_evidence")
        if used_verify:
            features.append("explicit_verification")
    if pattern_id == "integrate_for_audience":
        features.append("audience_compression")
        if used_tool == "bigmodel_proxy":
            features.append("generation_delegated")
    if pattern_id == "tool_boundary":
        if _contains_any(prompt, ("既有", "又问", "先走", "哪类工具", "代码")):
            features.append("boundary_ambiguous")
        elif expected_tool == "calculator":
            features.append("direct_numeric")
    if pattern_id == "stop_or_continue":
        features.append("termination_risk")
    if len(actions) <= 3 and pattern_id in {"conflict_first", "compare_with_evidence"}:
        features.append("compressed_trace")
    if "repair-" in str(row.get("id", "")):
        features.append("repair_case")
    if "conflict" in text and "conflict_unresolved" not in features:
        features.append("conflict_unresolved")

    return list(dict.fromkeys(features))


def _fit_score(
    *,
    pattern_id: str,
    tool_match: bool,
    residual_features: list[str],
    used_verify: bool,
    action_count: int,
) -> float:
    score = 0.52
    if tool_match:
        score += 0.22
    if pattern_id != "generic_reasoning":
        score += 0.12
    if residual_features:
        score += 0.08
    if used_verify:
        score += 0.04
    if action_count <= 4:
        score += 0.02
    if "tool_mismatch" in residual_features:
        score -= 0.28
    if "missing_verify" in residual_features:
        score -= 0.12
    return round(max(0.0, min(score, 1.0)), 4)


def _reconstruction_notes(pattern_id: str, residual_features: list[str]) -> list[str]:
    notes = [f"base_pattern={pattern_id}"]
    if residual_features:
        notes.append("residuals=" + ",".join(residual_features))
    else:
        notes.append("residuals=none")
    if "missing_verify" in residual_features:
        notes.append("risky_residual_without_verify")
    if "tool_mismatch" in residual_features:
        notes.append("tool_choice_not_reconstructed")
    return notes


def _actions(row: dict[str, Any]) -> list[str]:
    actions = row.get("pretrained_actions", row.get("actions", []))
    if not isinstance(actions, list):
        return []
    return [str(action) for action in actions]


def _used_tool(row: dict[str, Any]) -> str:
    return str(row.get("pretrained_used_tool", row.get("used_tool", "")))


def _prompt_lookup(payload: dict[str, Any]) -> dict[str, str]:
    prompts = payload.get("prompts", [])
    if not isinstance(prompts, list):
        return {}
    lookup: dict[str, str] = {}
    for item in prompts:
        if not isinstance(item, dict):
            continue
        lookup[str(item.get("id", ""))] = str(item.get("prompt", ""))
    return lookup


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _avg(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return round(sum(items) / len(items), 4)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
