from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _git_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _select_real_prompt_eval(
    train_report: dict[str, object],
    latest_real_prompt_eval: dict[str, object],
) -> tuple[dict[str, object], str]:
    evaluation = train_report.get("evaluation", {})
    if not isinstance(evaluation, dict):
        evaluation = {}
    report_eval = evaluation.get("real_prompt_eval", {})
    if not isinstance(report_eval, dict):
        report_eval = {}

    latest_summary = latest_real_prompt_eval.get("summary", {})
    report_summary = report_eval.get("summary", {})
    if isinstance(latest_summary, dict) and latest_summary:
        return latest_real_prompt_eval, "data/eval/real_prompt_eval_latest.json"
    if isinstance(report_summary, dict) and report_summary:
        return report_eval, "data/train_runs/auto_train_latest.json:evaluation.real_prompt_eval"
    return {}, ""


def _summarize_failures(real_prompt_eval: dict[str, object]) -> list[dict[str, object]]:
    rows = real_prompt_eval.get("rows", [])
    if not isinstance(rows, list):
        return []

    failures: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict) or bool(row.get("pretrained_match", False)):
            continue
        failures.append(
            {
                "id": str(row.get("id", "")),
                "logic_skill": str(row.get("logic_skill", "")),
                "expected_tool": str(row.get("expected_tool", "")),
                "used_tool": str(row.get("pretrained_used_tool", "")),
            }
        )
    return failures[:10]


def build_current_best_payload(root: Path) -> dict[str, object]:
    train_report_path = root / "data" / "train_runs" / "auto_train_latest.json"
    latest_real_prompt_eval_path = root / "data" / "eval" / "real_prompt_eval_latest.json"
    reasoning_pattern_report_path = root / "artifacts" / "reasoning_pattern_codec_latest.json"
    attention_flow_report_path = root / "artifacts" / "attention_flow_latest.json"

    train_report = _read_json(train_report_path)
    latest_real_prompt_eval = _read_json(latest_real_prompt_eval_path)
    reasoning_pattern_report = _read_json(reasoning_pattern_report_path)
    attention_flow_report = _read_json(attention_flow_report_path)
    real_prompt_eval, real_prompt_eval_source = _select_real_prompt_eval(train_report, latest_real_prompt_eval)

    dataset = train_report.get("dataset", {})
    if not isinstance(dataset, dict):
        dataset = {}
    pretraining = train_report.get("pretraining", {})
    if not isinstance(pretraining, dict):
        pretraining = {}
    evaluation = train_report.get("evaluation", {})
    if not isinstance(evaluation, dict):
        evaluation = {}
    pretrain_eval = evaluation.get("pretrain_eval", {})
    if not isinstance(pretrain_eval, dict):
        pretrain_eval = {}

    real_prompt_summary = real_prompt_eval.get("summary", {})
    if not isinstance(real_prompt_summary, dict):
        real_prompt_summary = {}

    pretrained_section = pretrain_eval.get("pretrained", {})
    if not isinstance(pretrained_section, dict):
        pretrained_section = {}

    failures = _summarize_failures(real_prompt_eval)
    pretrained_match_rate = float(real_prompt_summary.get("pretrained_match_rate", 0.0) or 0.0)
    reasoning_summary = reasoning_pattern_report.get("summary", {})
    if not isinstance(reasoning_summary, dict):
        reasoning_summary = {}
    hard_eval = reasoning_pattern_report.get("hard_eval", {})
    if not isinstance(hard_eval, dict):
        hard_eval = {}
    hard_eval_failures = (
        list(hard_eval.get("failed_case_ids", []))
        if isinstance(hard_eval.get("failed_case_ids", []), list)
        else []
    )
    attention_summary = attention_flow_report.get("summary", {})
    if not isinstance(attention_summary, dict):
        attention_summary = {}
    status = (
        "healthy"
        if pretrained_match_rate >= 0.9 and not failures and not hard_eval_failures
        else "needs_attention"
    )

    return {
        "generated_at_utc": _utc_now(),
        "workspace_root": str(root.resolve()),
        "git_commit": _git_head(root),
        "best_run_id": str(train_report.get("run_id", "")),
        "best_variant_name": "pretrained",
        "status": status,
        "decision": (
            "keep_current_best"
            if status == "healthy"
            else "investigate_failures_before_promoting_further_changes"
        ),
        "sources": {
            "train_report": str(train_report_path),
            "real_prompt_eval": real_prompt_eval_source,
            "reasoning_pattern_codec": (
                str(reasoning_pattern_report_path)
                if reasoning_pattern_report
                else ""
            ),
            "attention_flow": (
                str(attention_flow_report_path)
                if attention_flow_report
                else ""
            ),
        },
        "artifacts": {
            "pretrain_artifact_dir": str(pretraining.get("artifact_dir", "")),
            "dataset_path": str(dataset.get("path", "")),
        },
        "dataset": {
            "sample_count": int(dataset.get("sample_count", 0) or 0),
            "teacher_sample_count": int(dataset.get("teacher_sample_count", 0) or 0),
            "real_prompt_candidate_count": int(dataset.get("real_prompt_candidate_count", 0) or 0),
        },
        "evaluation": {
            "pretrain_eval": pretrain_eval,
            "real_prompt_eval": real_prompt_eval,
        },
        "summary": {
            "pretrain_tool_match_rate": float(pretrained_section.get("tool_match_rate", 0.0) or 0.0),
            "real_prompt_count": int(real_prompt_summary.get("prompt_count", 0) or 0),
            "real_prompt_match_rate": pretrained_match_rate,
            "critical_failure_count": len(failures),
            "avg_steps": float(real_prompt_summary.get("pretrained_avg_steps", 0.0) or 0.0),
            "hard_logic_count": int(reasoning_summary.get("hard_logic_count", 0) or 0),
            "hard_logic_avg_fit_score": float(reasoning_summary.get("hard_logic_avg_fit_score", 0.0) or 0.0),
            "residual_explanation_rate": float(reasoning_summary.get("residual_explanation_rate", 0.0) or 0.0),
            "verify_when_residual_risky_rate": float(
                reasoning_summary.get("verify_when_residual_risky_rate", 0.0) or 0.0
            ),
            "hard_eval_pass_rate": float(hard_eval.get("pass_rate", 0.0) or 0.0),
            "attention_focus_continuity": float(attention_summary.get("focus_continuity", 0.0) or 0.0),
            "attention_evidence_grounding": float(attention_summary.get("evidence_grounding", 0.0) or 0.0),
            "attention_residual_resolution": float(attention_summary.get("residual_resolution", 0.0) or 0.0),
            "attention_premature_release_rate": float(attention_summary.get("premature_release_rate", 0.0) or 0.0),
        },
        "hard_eval_failures": hard_eval_failures,
        "key_failures": failures,
    }


def write_current_best(root: Path = Path(".")) -> dict[str, object]:
    payload = build_current_best_payload(root)
    target = root / "artifacts" / "current_best.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    payload = write_current_best(Path("."))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
