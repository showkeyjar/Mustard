from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.claw_team_github import automerge_pr, doctor as github_doctor
from scripts.claw_team_github import (
    commit_all_changes,
    get_current_branch,
    get_worktree_status,
    load_github_config,
    push_branch,
    review_pr,
    submit_pr,
)
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
    Path("team/GUARDIAN.md"),
    Path("memory/MEMORY.md"),
]


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
    return {
        "team_name": config.get("team_name", "mustard-claw"),
        "root": str(root.resolve()),
        "signals": signals,
        "proposal_count": _count_markdown_files(root / "backlog" / "proposals"),
        "daily_digest_count": _count_markdown_files(root / "memory" / "daily"),
    }


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
        for role in ["conductor", "observer", "architect", "evaluator", "guardian", "arbiter", "researcher"]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Control the Mustard Claw Team workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap")
    bootstrap_parser.add_argument("--root", default=".")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--root", default=".")
    run_parser.add_argument("--auto-commit", action="store_true")
    run_parser.add_argument("--auto-push", action="store_true")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--root", default=".")

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--root", default=".")

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
    elif args.command == "status":
        payload = status(root)
    elif args.command == "github-doctor":
        payload = github_doctor(root)
    elif args.command == "deliver":
        cycle_payload = run_cycle(root=root, config_path=DEFAULT_CONFIG_PATH)
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
