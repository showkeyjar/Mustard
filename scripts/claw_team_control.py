from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from carm.pretrain_data import load_review_feedback
from scripts.apply_learning_intake_human_review_sheet import apply_learning_intake_human_review_sheet
from scripts.build_learning_intake_human_gate_packet import build_learning_intake_human_gate_packet
from scripts.build_learning_intake_human_review_draft import build_learning_intake_human_review_draft
from scripts.build_learning_intake_human_review_panel import build_learning_intake_human_review_panel
from scripts.build_learning_intake_review_queue import build_learning_intake_review_queue
from scripts.claw_team_github import automerge_pr, doctor as github_doctor
from scripts.claw_team_github import (
    commit_all_changes,
    commit_selected_paths,
    get_current_branch,
    get_worktree_status,
    load_github_config,
    push_branch,
    push_current_branch,
    review_pr,
    submit_pr,
)
from scripts.compare_learning_intake_suggested_shadow import compare_learning_intake_suggested_shadow
from scripts.export_learning_intake_candidate_import import export_learning_intake_candidate_import
from scripts.preview_learning_intake_human_review_sheet import preview_learning_intake_human_review_sheet
from scripts.simulate_learning_intake_candidate_rebuild import simulate_learning_intake_candidate_rebuild
from scripts.sync_learning_intake_human_review_draft import sync_learning_intake_human_review_draft
from scripts.team_conductor import (
    DEFAULT_CONFIG_PATH,
    bootstrap_workspace,
    collect_signals,
    load_team_config,
    run_cycle,
)


REQUIRED_FILES = [
    Path("configs/team_cycle.json"),
    Path("team/AGENTS.md"),
    Path("team/CONDUCTOR.md"),
    Path("team/OBSERVER.md"),
    Path("team/ARBITER.md"),
    Path("team/RESEARCHER.md"),
    Path("team/BENCHMARK_OWNER.md"),
    Path("team/FAILURE_MINER.md"),
    Path("team/TRAINER.md"),
    Path("team/GUARDIAN.md"),
    Path("memory/MEMORY.md"),
]


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _compact_current_best(payload: dict[str, object], path: Path) -> dict[str, object]:
    if not payload:
        return {}
    summary = payload.get("summary", {})
    return {
        "path": str(path),
        "generated_at_utc": str(payload.get("generated_at_utc", "")),
        "best_run_id": str(payload.get("best_run_id", "")),
        "best_variant_name": str(payload.get("best_variant_name", "")),
        "status": str(payload.get("status", "")),
        "decision": str(payload.get("decision", "")),
        "summary": summary if isinstance(summary, dict) else {},
    }


def _count_markdown_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob("*.md") if item.is_file())


def doctor(root: Path) -> dict[str, object]:
    missing_files = [str(path) for path in REQUIRED_FILES if not (root / path).exists()]
    config = load_team_config(root / DEFAULT_CONFIG_PATH)
    proposals_dir = root / "backlog" / "proposals"
    daily_dir = root / "memory" / "daily"

    return {
        "ok": len(missing_files) == 0 and sys.version_info >= (3, 10),
        "root": str(root.resolve()),
        "python_version": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 10),
        "team_name": config.get("team_name", "mustard-claw"),
        "missing_files": missing_files,
        "proposal_count": _count_markdown_files(proposals_dir),
        "daily_digest_count": _count_markdown_files(daily_dir),
    }


def status(root: Path) -> dict[str, object]:
    config = load_team_config(root / DEFAULT_CONFIG_PATH)
    signals = collect_signals(root)
    current_best_path = root / "artifacts" / "current_best.json"
    current_best = _compact_current_best(_read_json(current_best_path), current_best_path)
    human_review = _summarize_human_review(root)
    return {
        "team_name": config.get("team_name", "mustard-claw"),
        "root": str(root.resolve()),
        "signals": signals,
        "current_best": current_best,
        "human_review": human_review,
        "proposal_count": _count_markdown_files(root / "backlog" / "proposals"),
        "daily_digest_count": _count_markdown_files(root / "memory" / "daily"),
    }


def _summarize_human_review(root: Path) -> dict[str, object]:
    panel_payload = _read_json(root / "artifacts" / "learning_intake_human_review_panel_latest.json")
    draft_payload = _read_json(root / "artifacts" / "learning_intake_human_review_draft_latest.json")
    draft_sync_payload = _read_json(root / "artifacts" / "learning_intake_human_review_draft_sync_latest.json")
    apply_payload = _read_json(root / "artifacts" / "learning_intake_human_review_apply_latest.json")
    import_payload = _read_json(root / "artifacts" / "learning_intake_candidate_import_latest.json")
    preview_payload = _read_json(root / "artifacts" / "learning_intake_human_review_preview_latest.json")
    review_pack = load_review_feedback(root / "data" / "learning" / "candidate_pretrain_review_pack.jsonl")

    panel_summary = panel_payload.get("summary", {}) if isinstance(panel_payload, dict) else {}
    draft_summary = draft_payload.get("summary", {}) if isinstance(draft_payload, dict) else {}
    draft_sync_summary = draft_sync_payload if isinstance(draft_sync_payload, dict) else {}
    apply_summary = apply_payload if isinstance(apply_payload, dict) else {}
    import_summary = import_payload if isinstance(import_payload, dict) else {}
    preview_summary = preview_payload.get("summary", {}) if isinstance(preview_payload, dict) else {}

    return {
        "panel_ready": bool(panel_summary),
        "total_candidates": int(panel_summary.get("total_candidates", len(review_pack)) or 0),
        "recommend_accept": int(panel_summary.get("recommend_accept", 0) or 0),
        "recommend_edit": int(panel_summary.get("recommend_edit", 0) or 0),
        "recommend_defer": int(panel_summary.get("recommend_defer", 0) or 0),
        "ready_to_apply_count": int(preview_summary.get("ready_to_apply_count", 0) or 0),
        "blank_decision_count": int(preview_summary.get("blank_decision_count", len(review_pack)) or 0),
        "invalid_decision_count": int(preview_summary.get("invalid_decision_count", 0) or 0),
        "accepted_count": int(apply_summary.get("accepted_count", 0) or 0),
        "edited_count": int(apply_summary.get("edited_count", 0) or 0),
        "pending_count": int(apply_summary.get("pending_count", len(review_pack)) or 0),
        "import_approved_count": int(import_summary.get("approved_count", 0) or 0),
        "next_action_state": str(preview_payload.get("next_action", {}).get("state", "")) if isinstance(preview_payload, dict) else "",
        "next_action_command": str(preview_payload.get("next_action", {}).get("command", "")) if isinstance(preview_payload, dict) else "",
        "draft_sheet_path": str(draft_summary.get("draft_sheet_path", "")),
        "draft_sync_count": int(draft_sync_summary.get("synced_count", 0) or 0),
        "review_sheet_path": str(panel_summary.get("review_sheet_path", "")),
        "panel_report_path": str(panel_summary.get("report_path", "")),
        "preview_report_path": str(preview_summary.get("report_path", "")),
    }


def build_human_review_session(root: Path) -> dict[str, object]:
    queue_summary = build_learning_intake_review_queue(root)
    suggested_rebuild = simulate_learning_intake_candidate_rebuild(root)
    shadow_delta = compare_learning_intake_suggested_shadow(root)
    packet = build_learning_intake_human_gate_packet(root)
    panel = build_learning_intake_human_review_panel(root)
    draft = build_learning_intake_human_review_draft(root)
    preview = preview_learning_intake_human_review_sheet(root)
    return {
        "mode": "human_review_session",
        "queue_summary": queue_summary,
        "suggested_rebuild_summary": suggested_rebuild,
        "shadow_delta_summary": shadow_delta.get("summary", {}) if isinstance(shadow_delta, dict) else {},
        "packet_summary": packet.get("summary", {}) if isinstance(packet, dict) else {},
        "panel_summary": panel.get("summary", {}) if isinstance(panel, dict) else {},
        "draft_summary": draft.get("summary", {}) if isinstance(draft, dict) else {},
        "preview_summary": preview.get("summary", {}) if isinstance(preview, dict) else {},
        "default_training_admission_changed": False,
    }


def apply_human_review_session(root: Path, *, export_import: bool = False) -> dict[str, object]:
    apply_summary = apply_learning_intake_human_review_sheet(root)
    payload: dict[str, object] = {
        "mode": "apply_human_review_session",
        "apply_summary": apply_summary,
        "default_training_admission_changed": False,
    }
    if export_import:
        payload["import_summary"] = export_learning_intake_candidate_import(root)
    return payload


def _derive_auto_commit_message(root: Path, cycle_payload: dict[str, object]) -> str:
    subject = "Claw Team: update team cycle outputs"

    proposal_paths = cycle_payload.get("proposal_paths", [])
    if isinstance(proposal_paths, list) and proposal_paths:
        first = Path(str(proposal_paths[0]))
        candidate = first if first.is_absolute() else (root / first)
        if candidate.exists():
            first_line = candidate.read_text(encoding="utf-8").splitlines()[0].strip()
            if first_line.startswith("# ") and len(first_line) > 2:
                subject = f"Claw Team: {first_line[2:]}"

    if subject == "Claw Team: update team cycle outputs":
        digest_path = cycle_payload.get("digest_path", "")
        if isinstance(digest_path, str) and digest_path:
            subject = f"Claw Team: update daily digest {Path(digest_path).name}"

    team_actions = cycle_payload.get("team_actions", {})
    if isinstance(team_actions, dict) and team_actions:
        body_lines = ["", "Team actions:"]
        for role in ["conductor", "observer", "failure_miner", "benchmark_owner", "architect", "trainer", "evaluator", "guardian", "arbiter", "researcher"]:
            action = team_actions.get(role)
            if isinstance(action, str) and action.strip():
                body_lines.append(f"- {role}: {action}")
        if len(body_lines) > 2:
            return subject + "\n" + "\n".join(body_lines)

    return subject


def _auto_commit_if_allowed(root: Path, cycle_payload: dict[str, object], *, auto_push: bool = False) -> dict[str, object]:
    direction_review = cycle_payload.get("direction_review", {})
    if not isinstance(direction_review, dict):
        direction_review = {}

    verdict = str(direction_review.get("verdict", "direction_correct"))
    if verdict != "direction_correct":
        return {
            "enabled": True,
            "committed": False,
            "reason": "direction_not_approved",
            "verdict": verdict,
        }

    status = get_worktree_status(root)
    if not status.strip():
        return {
            "enabled": True,
            "committed": False,
            "reason": "clean_worktree",
            "verdict": verdict,
        }

    commit_message = _derive_auto_commit_message(root, cycle_payload)
    commit_result = commit_all_changes(root, commit_message)
    branch = get_current_branch(root)

    payload: dict[str, object] = {
        "enabled": True,
        "verdict": verdict,
        "branch": branch,
        "commit_message": commit_message,
        **commit_result,
    }

    if auto_push and bool(commit_result.get("committed", False)):
        try:
            push_branch(root, branch)
            payload["pushed"] = True
        except Exception as exc:  # pragma: no cover - defensive runtime path
            payload["pushed"] = False
            payload["push_error"] = str(exc)

    return payload


def _paths_for_git_delivery(decision: dict[str, object], *, include_artifacts: bool = False) -> list[str]:
    file_groups = decision.get("file_groups", {})
    if not isinstance(file_groups, dict):
        return []

    paths: list[str] = []
    core = file_groups.get("core", [])
    artifacts = file_groups.get("artifacts", [])
    if isinstance(core, list):
        paths.extend(str(item) for item in core if str(item).strip())
    if include_artifacts and isinstance(artifacts, list):
        for item in artifacts:
            value = str(item).strip()
            if not value:
                continue
            if value.startswith("memory/daily/") or value.startswith("backlog/opportunities/") or value.startswith("team/"):
                paths.append(value)
    return paths


def _build_delivery_summary(cycle_payload: dict[str, object], git_delivery: dict[str, object] | None = None) -> dict[str, object]:
    decision = cycle_payload.get("delivery_decision", {})
    if not isinstance(decision, dict):
        decision = {}
    file_groups = decision.get("file_groups", {})
    if not isinstance(file_groups, dict):
        file_groups = {}

    summary = {
        "delivery_lane": str(decision.get("delivery_lane", "skip")),
        "delivery_reason": str(decision.get("reason", "")),
        "should_commit": bool(decision.get("should_commit", False)),
        "should_push": bool(decision.get("should_push", False)),
        "should_open_pr": bool(decision.get("should_open_pr", False)),
        "core_count": len(file_groups.get("core", [])) if isinstance(file_groups.get("core", []), list) else 0,
        "artifact_count": len(file_groups.get("artifacts", [])) if isinstance(file_groups.get("artifacts", []), list) else 0,
        "volatile_count": len(file_groups.get("volatile", [])) if isinstance(file_groups.get("volatile", []), list) else 0,
        "other_count": len(file_groups.get("other", [])) if isinstance(file_groups.get("other", []), list) else 0,
    }
    if isinstance(git_delivery, dict):
        summary.update(
            {
                "git_committed": bool(git_delivery.get("committed", False)),
                "git_pushed": bool(git_delivery.get("pushed", False)),
                "git_commit_sha": str(git_delivery.get("commit_sha", "")),
                "git_branch": str(git_delivery.get("branch", "")),
            }
        )
    return summary


def _auto_sync_git_from_cycle(root: Path, cycle_payload: dict[str, object], *, include_artifacts: bool = False) -> dict[str, object]:
    decision = cycle_payload.get("delivery_decision", {})
    if not isinstance(decision, dict):
        return {"enabled": True, "committed": False, "pushed": False, "reason": "delivery_decision_missing"}

    lane = str(decision.get("delivery_lane", "skip"))
    if lane == "skip":
        return {"enabled": True, "committed": False, "pushed": False, "reason": str(decision.get("reason", "skip")), "lane": lane}

    paths = _paths_for_git_delivery(decision, include_artifacts=include_artifacts)
    if not paths:
        return {"enabled": True, "committed": False, "pushed": False, "reason": "no_selected_paths", "lane": lane}

    commit_message = _derive_auto_commit_message(root, cycle_payload)
    commit_result = commit_selected_paths(root, paths, commit_message)
    payload: dict[str, object] = {
        "enabled": True,
        "lane": lane,
        "paths": paths,
        "commit_message": commit_message,
        **commit_result,
    }

    if bool(commit_result.get("committed", False)) and bool(decision.get("should_push", False)):
        try:
            push_payload = push_current_branch(root)
            payload.update(push_payload)
        except Exception as exc:
            payload["pushed"] = False
            payload["push_error"] = str(exc)
    else:
        payload["pushed"] = False

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Control the Mustard Claw Team workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap")
    bootstrap_parser.add_argument("--root", default=".")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--root", default=".")
    run_parser.add_argument("--auto-commit", action="store_true")
    run_parser.add_argument("--auto-push", action="store_true")
    run_parser.add_argument("--auto-sync-git", action="store_true")
    run_parser.add_argument("--include-artifacts", action="store_true")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--root", default=".")

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--root", default=".")

    human_review_parser = subparsers.add_parser("human-review")
    human_review_parser.add_argument("--root", default=".")

    draft_human_review_parser = subparsers.add_parser("draft-human-review")
    draft_human_review_parser.add_argument("--root", default=".")

    sync_draft_human_review_parser = subparsers.add_parser("sync-human-review-draft")
    sync_draft_human_review_parser.add_argument("--root", default=".")

    preview_human_review_parser = subparsers.add_parser("preview-human-review")
    preview_human_review_parser.add_argument("--root", default=".")

    apply_human_review_parser = subparsers.add_parser("apply-human-review")
    apply_human_review_parser.add_argument("--root", default=".")
    apply_human_review_parser.add_argument("--export-import", action="store_true")

    github_doctor_parser = subparsers.add_parser("github-doctor")
    github_doctor_parser.add_argument("--root", default=".")

    deliver_parser = subparsers.add_parser("deliver")
    deliver_parser.add_argument("--root", default=".")
    deliver_parser.add_argument("--title", default="")
    deliver_parser.add_argument("--body", default="")
    deliver_parser.add_argument("--commit-message", default="")
    deliver_parser.add_argument("--branch", default="")
    deliver_parser.add_argument("--reviewer", action="append", default=[])
    deliver_parser.add_argument("--draft", action="store_true")
    deliver_parser.add_argument("--force-pr", action="store_true")

    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("--root", default=".")
    review_parser.add_argument("--pr", required=True, type=int)
    review_parser.add_argument("--event", choices=["auto", "APPROVE", "COMMENT", "REQUEST_CHANGES"], default="auto")
    review_parser.add_argument("--body", default="")
    review_parser.add_argument("--check-command", action="append", default=[])

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--root", default=".")
    merge_parser.add_argument("--pr", required=True, type=int)
    merge_parser.add_argument("--method", choices=["merge", "squash", "rebase"], default="")
    merge_parser.add_argument("--commit-title", default="")
    merge_parser.add_argument("--commit-message", default="")

    args = parser.parse_args()
    root = Path(args.root).resolve()

    if args.command == "bootstrap":
        bootstrap_workspace(root)
        payload = {
            "bootstrapped": True,
            "root": str(root),
            "config_path": str((root / DEFAULT_CONFIG_PATH).resolve()),
        }
    elif args.command == "run":
        payload = run_cycle(root=root, config_path=DEFAULT_CONFIG_PATH)
        if bool(getattr(args, "auto_commit", False)):
            payload["auto_commit"] = _auto_commit_if_allowed(
                root,
                payload,
                auto_push=bool(getattr(args, "auto_push", False)),
            )
        if bool(getattr(args, "auto_sync_git", False)):
            payload["git_delivery"] = _auto_sync_git_from_cycle(
                root,
                payload,
                include_artifacts=bool(getattr(args, "include_artifacts", False)),
            )
        payload["delivery_summary"] = _build_delivery_summary(payload, payload.get("git_delivery"))
    elif args.command == "status":
        payload = status(root)
    elif args.command == "human-review":
        payload = build_human_review_session(root)
    elif args.command == "draft-human-review":
        payload = build_learning_intake_human_review_draft(root)
    elif args.command == "sync-human-review-draft":
        payload = sync_learning_intake_human_review_draft(root)
    elif args.command == "preview-human-review":
        payload = preview_learning_intake_human_review_sheet(root)
    elif args.command == "apply-human-review":
        payload = apply_human_review_session(root, export_import=bool(getattr(args, "export_import", False)))
    elif args.command == "github-doctor":
        payload = github_doctor(root)
    elif args.command == "deliver":
        cycle_payload = run_cycle(root=root, config_path=DEFAULT_CONFIG_PATH)
        delivery_decision = cycle_payload.get("delivery_decision", {})
        lane = str(delivery_decision.get("delivery_lane", "skip")) if isinstance(delivery_decision, dict) else "skip"
        if not bool(getattr(args, "force_pr", False)) and lane != "pr_delivery":
            payload = {
                "cycle": cycle_payload,
                "delivery": {
                    "submitted": False,
                    "reason": "pr_lane_not_selected",
                    "delivery_decision": delivery_decision,
                },
            }
        else:
            github_payload = github_doctor(root)
            if not github_payload.get("ok", False):
                payload = {
                    "cycle": cycle_payload,
                    "delivery": {
                        "submitted": False,
                        "reason": "github_doctor_failed",
                        "github": github_payload,
                    },
                }
            else:
                github_config = load_github_config(root)
                base_branch = str(github_config.get("base_branch", "main"))
                title = str(args.title).strip() or f"Claw Team: automated delivery {Path(root).name}"
                body = str(args.body).strip() or "Automated delivery created by Mustard Claw Team."
                commit_message = str(args.commit_message).strip() or title
                delivery_payload = submit_pr(
                    root,
                    title=title,
                    body=body,
                    base_branch=base_branch,
                    commit_message=commit_message,
                    branch_name=str(args.branch).strip() or None,
                    draft=bool(args.draft),
                    reviewers=list(args.reviewer),
                )
                payload = {
                    "cycle": cycle_payload,
                    "delivery": {
                        "submitted": True,
                        "forced": bool(getattr(args, "force_pr", False)),
                        **delivery_payload,
                    },
                }
    elif args.command == "review":
        github_config = load_github_config(root)
        auto_review = github_config.get("auto_review", {})
        if not isinstance(auto_review, dict):
            auto_review = {}
        check_commands = list(args.check_command) or list(auto_review.get("check_commands", []))
        payload = review_pr(
            root,
            pr_number=int(args.pr),
            requested_event=str(args.event),
            check_commands=check_commands,
            body=str(args.body),
        )
    elif args.command == "merge":
        github_payload = github_doctor(root)
        if not github_payload.get("ok", False):
            payload = {
                "merged": False,
                "reason": "github_doctor_failed",
                "github": github_payload,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        github_config = load_github_config(root)
        auto_merge = github_config.get("auto_merge", {})
        if not isinstance(auto_merge, dict):
            auto_merge = {}
        merge_method = str(args.method).strip() or str(auto_merge.get("merge_method", "squash"))
        payload = automerge_pr(
            root,
            pr_number=int(args.pr),
            merge_method=merge_method,
            commit_title=str(args.commit_title),
            commit_message=str(args.commit_message),
        )
    else:
        payload = doctor(root)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
