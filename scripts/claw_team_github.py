from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_GITHUB_CONFIG_PATH = Path("configs/team_github.json")
DEFAULT_API_VERSION = "2026-03-10"


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        capture_output=True,
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_github_config(root: Path = Path(".")) -> dict[str, object]:
    config = _read_json(root / DEFAULT_GITHUB_CONFIG_PATH)
    if config:
        return config
    return {
        "base_branch": "main",
        "default_reviewers": [],
        "auto_review": {
            "enabled": True,
            "same_repo_only": True,
            "check_commands": ["python -m unittest discover -s tests -v"],
        },
    }


def _token() -> str:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or os.environ.get("MUSTARD_GITHUB_TOKEN") or ""
    return token.strip()


def _slug(value: str) -> str:
    lowered = value.lower()
    cleaned = "".join(char if char.isalnum() else "-" for char in lowered)
    compact = "-".join(part for part in cleaned.split("-") if part)
    return compact or "change"


def _git_output(root: Path, *args: str) -> str:
    return _run(["git", *args], cwd=root).stdout.strip()


def get_origin_url(root: Path) -> str:
    return _git_output(root, "remote", "get-url", "origin")


def parse_owner_repo(remote_url: str) -> tuple[str, str]:
    https_match = re.search(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?$", remote_url)
    if not https_match:
        raise ValueError(f"Unsupported GitHub remote URL: {remote_url}")
    return https_match.group("owner"), https_match.group("repo")


def _github_request(
    method: str,
    path: str,
    *,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    token = _token()
    if not token:
        raise RuntimeError("Missing GitHub token. Set GH_TOKEN, GITHUB_TOKEN, or MUSTARD_GITHUB_TOKEN.")

    url = f"https://api.github.com{path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": DEFAULT_API_VERSION,
        "User-Agent": "mustard-claw-team",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {method} {path} failed: {exc.code} {error_body}") from exc
    if not body:
        return {}
    return json.loads(body)


def get_current_branch(root: Path) -> str:
    return _git_output(root, "branch", "--show-current")


def get_worktree_status(root: Path) -> str:
    return _git_output(root, "status", "--short")


def ensure_feature_branch(root: Path, base_branch: str, branch_name: str | None = None) -> str:
    current_branch = get_current_branch(root)
    if current_branch and current_branch != base_branch:
        return current_branch

    target_branch = branch_name or f"claw/{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    _run(["git", "checkout", "-b", target_branch], cwd=root)
    return target_branch


def commit_all_changes(root: Path, commit_message: str) -> dict[str, object]:
    status = get_worktree_status(root)
    if not status:
        return {"committed": False, "status": "clean"}

    _run(["git", "add", "-A"], cwd=root)
    _run(["git", "commit", "-m", commit_message], cwd=root)
    commit_sha = _git_output(root, "rev-parse", "HEAD")
    return {"committed": True, "commit_sha": commit_sha}


def push_branch(root: Path, branch: str) -> None:
    _run(["git", "push", "-u", "origin", branch], cwd=root)


def create_pull_request(
    root: Path,
    *,
    title: str,
    body: str,
    base_branch: str,
    branch: str,
    draft: bool,
    reviewers: list[str] | None = None,
) -> dict[str, object]:
    owner, repo = parse_owner_repo(get_origin_url(root))
    payload = {
        "title": title,
        "body": body,
        "head": branch,
        "base": base_branch,
        "draft": draft,
    }
    pr = _github_request("POST", f"/repos/{owner}/{repo}/pulls", payload=payload)
    pr_number = int(pr["number"])
    reviewers = reviewers or []
    if reviewers:
        _github_request(
            "POST",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/requested_reviewers",
            payload={"reviewers": reviewers},
        )
    return pr


def submit_pr(
    root: Path,
    *,
    title: str,
    body: str,
    base_branch: str,
    commit_message: str,
    branch_name: str | None = None,
    draft: bool = True,
    reviewers: list[str] | None = None,
) -> dict[str, object]:
    branch = ensure_feature_branch(root, base_branch, branch_name)
    commit_result = commit_all_changes(root, commit_message)
    push_branch(root, branch)
    pr = create_pull_request(
        root,
        title=title,
        body=body,
        base_branch=base_branch,
        branch=branch,
        draft=draft,
        reviewers=reviewers,
    )
    return {
        "branch": branch,
        "commit": commit_result,
        "pull_request": {
            "number": pr["number"],
            "url": pr["html_url"],
            "title": pr["title"],
            "draft": pr.get("draft", draft),
        },
    }


def get_pull_request(root: Path, pr_number: int) -> dict[str, object]:
    owner, repo = parse_owner_repo(get_origin_url(root))
    return _github_request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}")


def get_repository(root: Path) -> dict[str, object]:
    owner, repo = parse_owner_repo(get_origin_url(root))
    return _github_request("GET", f"/repos/{owner}/{repo}")


def list_repository_labels(root: Path) -> list[dict[str, object]]:
    owner, repo = parse_owner_repo(get_origin_url(root))
    payload = _github_request("GET", f"/repos/{owner}/{repo}/labels")
    if isinstance(payload, list):
        return payload
    return []


def get_actions_permissions(root: Path) -> dict[str, object]:
    owner, repo = parse_owner_repo(get_origin_url(root))
    return _github_request("GET", f"/repos/{owner}/{repo}/actions/permissions")


def get_actions_workflow_permissions(root: Path) -> dict[str, object]:
    owner, repo = parse_owner_repo(get_origin_url(root))
    return _github_request("GET", f"/repos/{owner}/{repo}/actions/permissions/workflow")


def create_review(root: Path, pr_number: int, event: str, body: str) -> dict[str, object]:
    owner, repo = parse_owner_repo(get_origin_url(root))
    return _github_request(
        "POST",
        f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
        payload={"event": event, "body": body},
    )


def list_reviews(root: Path, pr_number: int) -> list[dict[str, object]]:
    owner, repo = parse_owner_repo(get_origin_url(root))
    payload = _github_request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews")
    if isinstance(payload, list):
        return payload
    return []


def get_commit_status(root: Path, ref: str) -> dict[str, object]:
    owner, repo = parse_owner_repo(get_origin_url(root))
    return _github_request("GET", f"/repos/{owner}/{repo}/commits/{ref}/status")


def merge_pull_request(
    root: Path,
    pr_number: int,
    *,
    merge_method: str,
    commit_title: str = "",
    commit_message: str = "",
) -> dict[str, object]:
    owner, repo = parse_owner_repo(get_origin_url(root))
    payload: dict[str, object] = {"merge_method": merge_method}
    if commit_title.strip():
        payload["commit_title"] = commit_title.strip()
    if commit_message.strip():
        payload["commit_message"] = commit_message.strip()
    return _github_request("PUT", f"/repos/{owner}/{repo}/pulls/{pr_number}/merge", payload=payload)


def _run_check_command(command: str, root: Path) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=str(root),
        shell=True,
        text=True,
        capture_output=True,
        encoding="utf-8",
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "ok": completed.returncode == 0,
    }


def _truncate(text: str, limit: int = 800) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def choose_review_event(
    pr: dict[str, object],
    check_results: list[dict[str, object]],
    requested_event: str,
) -> tuple[str, str]:
    if requested_event != "auto":
        return requested_event, "manual review event selected"

    failures = [item for item in check_results if not item["ok"]]
    if failures:
        return "REQUEST_CHANGES", "local validation failed"

    if bool(pr.get("draft", False)):
        return "COMMENT", "pull request is still draft"

    if pr.get("mergeable") is False:
        return "REQUEST_CHANGES", "pull request is not mergeable"

    return "APPROVE", "all configured checks passed"


def build_review_body(
    pr: dict[str, object],
    review_event: str,
    reason: str,
    check_results: list[dict[str, object]],
    extra_body: str = "",
) -> str:
    lines = [
        f"Claw Team automated review for PR #{pr.get('number')}: {pr.get('title', '')}",
        "",
        f"Verdict: {review_event}",
        f"Reason: {reason}",
    ]
    if extra_body.strip():
        lines.extend(["", extra_body.strip()])
    if check_results:
        lines.extend(["", "Validation summary:"])
        for result in check_results:
            status = "PASS" if result["ok"] else "FAIL"
            lines.append(f"- {status}: `{result['command']}`")
            if not result["ok"]:
                detail = result["stderr"] or result["stdout"]
                if detail:
                    lines.append(f"  {_truncate(detail)}")
    return "\n".join(lines)


def review_pr(
    root: Path,
    *,
    pr_number: int,
    requested_event: str,
    check_commands: list[str],
    body: str = "",
) -> dict[str, object]:
    pr = get_pull_request(root, pr_number)
    check_results = [_run_check_command(command, root) for command in check_commands]
    review_event, reason = choose_review_event(pr, check_results, requested_event)
    review_body = build_review_body(pr, review_event, reason, check_results, extra_body=body)
    review = create_review(root, pr_number, review_event, review_body)
    return {
        "pr_number": pr_number,
        "review_event": review_event,
        "reason": reason,
        "check_results": check_results,
        "review_id": review.get("id"),
        "review_url": review.get("html_url", ""),
    }


def _label_names(pr: dict[str, object]) -> set[str]:
    labels = pr.get("labels", [])
    if not isinstance(labels, list):
        return set()
    names: set[str] = set()
    for label in labels:
        if isinstance(label, dict):
            name = str(label.get("name", "")).strip()
            if name:
                names.add(name)
    return names


def _approved_review_count(reviews: list[dict[str, object]]) -> int:
    return sum(1 for review in reviews if str(review.get("state", "")).upper() == "APPROVED")


def _status_is_clean(status_payload: dict[str, object]) -> bool:
    state = str(status_payload.get("state", "")).lower()
    return state in {"success", ""}


def can_automerge(
    pr: dict[str, object],
    *,
    reviews: list[dict[str, object]],
    status_payload: dict[str, object],
    auto_merge_config: dict[str, object],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not bool(auto_merge_config.get("enabled", True)):
        reasons.append("auto_merge_disabled")
    if bool(pr.get("draft", False)):
        reasons.append("pr_is_draft")
    if pr.get("mergeable") is False:
        reasons.append("pr_not_mergeable")

    if bool(auto_merge_config.get("same_repo_only", True)):
        head_repo = pr.get("head", {}).get("repo", {}) if isinstance(pr.get("head", {}), dict) else {}
        base_repo = pr.get("base", {}).get("repo", {}) if isinstance(pr.get("base", {}), dict) else {}
        head_full_name = str(head_repo.get("full_name", ""))
        base_full_name = str(base_repo.get("full_name", ""))
        if head_full_name and base_full_name and head_full_name != base_full_name:
            reasons.append("cross_repo_pr")

    required_label = str(auto_merge_config.get("required_label", "")).strip()
    if required_label and required_label not in _label_names(pr):
        reasons.append("required_label_missing")

    if bool(auto_merge_config.get("require_approved_review", True)) and _approved_review_count(reviews) <= 0:
        reasons.append("approved_review_missing")

    if bool(auto_merge_config.get("require_clean_status", True)) and not _status_is_clean(status_payload):
        reasons.append("commit_status_not_clean")

    return len(reasons) == 0, reasons


def automerge_pr(
    root: Path,
    *,
    pr_number: int,
    merge_method: str,
    commit_title: str = "",
    commit_message: str = "",
) -> dict[str, object]:
    config = load_github_config(root)
    auto_merge_config = config.get("auto_merge", {})
    if not isinstance(auto_merge_config, dict):
        auto_merge_config = {}

    pr = get_pull_request(root, pr_number)
    reviews = list_reviews(root, pr_number)
    head = pr.get("head", {})
    if not isinstance(head, dict):
        head = {}
    head_sha = str(head.get("sha", "")).strip()
    status_payload = get_commit_status(root, head_sha) if head_sha else {}
    ok, reasons = can_automerge(
        pr,
        reviews=reviews,
        status_payload=status_payload,
        auto_merge_config=auto_merge_config,
    )
    if not ok:
        return {
            "merged": False,
            "pr_number": pr_number,
            "reasons": reasons,
            "approved_review_count": _approved_review_count(reviews),
            "commit_status_state": str(status_payload.get("state", "")),
        }

    merge_response = merge_pull_request(
        root,
        pr_number,
        merge_method=merge_method,
        commit_title=commit_title,
        commit_message=commit_message,
    )
    return {
        "merged": bool(merge_response.get("merged", False)),
        "pr_number": pr_number,
        "merge_method": merge_method,
        "sha": merge_response.get("sha", ""),
        "message": merge_response.get("message", ""),
    }


def doctor(root: Path) -> dict[str, object]:
    remote_url = ""
    owner = ""
    repo = ""
    remote_ok = False
    try:
        remote_url = get_origin_url(root)
        owner, repo = parse_owner_repo(remote_url)
        remote_ok = True
    except Exception:
        remote_ok = False

    token_present = bool(_token())
    git_ok = True
    try:
        _git_output(root, "rev-parse", "--is-inside-work-tree")
    except Exception:
        git_ok = False

    config = load_github_config(root)
    auto_merge = config.get("auto_merge", {})
    if not isinstance(auto_merge, dict):
        auto_merge = {}
    required_label = str(auto_merge.get("required_label", "")).strip()

    repository_info: dict[str, object] = {}
    labels_ok: bool | None = None
    label_names: list[str] = []
    actions_permissions: dict[str, object] | None = None
    workflow_permissions: dict[str, object] | None = None
    diagnostics: list[str] = []

    if token_present and remote_ok:
        try:
            repository_info = get_repository(root)
        except Exception as exc:
            diagnostics.append(f"repository_check_failed:{exc}")
        try:
            labels = list_repository_labels(root)
            label_names = sorted(
                [
                    str(label.get("name", "")).strip()
                    for label in labels
                    if isinstance(label, dict) and str(label.get("name", "")).strip()
                ]
            )
            labels_ok = required_label in label_names if required_label else True
            if required_label and not labels_ok:
                diagnostics.append(f"required_label_missing:{required_label}")
        except Exception as exc:
            diagnostics.append(f"label_check_failed:{exc}")
        try:
            actions_permissions = get_actions_permissions(root)
        except Exception as exc:
            diagnostics.append(f"actions_permissions_check_failed:{exc}")
        try:
            workflow_permissions = get_actions_workflow_permissions(root)
        except Exception as exc:
            diagnostics.append(f"workflow_permissions_check_failed:{exc}")

    actions_enabled = bool(actions_permissions.get("enabled")) if isinstance(actions_permissions, dict) and "enabled" in actions_permissions else None
    workflow_permission_level = (
        str(workflow_permissions.get("default_workflow_permissions", "")).strip()
        if isinstance(workflow_permissions, dict)
        else ""
    )
    can_approve_reviews = (
        bool(workflow_permissions.get("can_approve_pull_request_reviews"))
        if isinstance(workflow_permissions, dict) and "can_approve_pull_request_reviews" in workflow_permissions
        else None
    )

    overall_ok = git_ok and remote_ok and token_present
    if labels_ok is False:
        overall_ok = False
    if actions_enabled is False:
        overall_ok = False
    if can_approve_reviews is False:
        overall_ok = False

    return {
        "ok": overall_ok,
        "git_ok": git_ok,
        "remote_ok": remote_ok,
        "token_present": token_present,
        "remote_url": remote_url,
        "owner": owner,
        "repo": repo,
        "current_branch": get_current_branch(root) if git_ok else "",
        "worktree_dirty": bool(get_worktree_status(root)) if git_ok else False,
        "required_env": "GH_TOKEN or GITHUB_TOKEN or MUSTARD_GITHUB_TOKEN",
        "default_branch": str(repository_info.get("default_branch", "")) if repository_info else "",
        "required_label": required_label,
        "required_label_present": labels_ok,
        "labels": label_names,
        "actions_enabled": actions_enabled,
        "workflow_default_permissions": workflow_permission_level,
        "can_approve_pull_request_reviews": can_approve_reviews,
        "diagnostics": diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="GitHub delivery lane for Mustard Claw Team.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--root", default=".")

    submit_parser = subparsers.add_parser("submit-pr")
    submit_parser.add_argument("--root", default=".")
    submit_parser.add_argument("--title", required=True)
    submit_parser.add_argument("--body", default="")
    submit_parser.add_argument("--base")
    submit_parser.add_argument("--commit-message")
    submit_parser.add_argument("--branch")
    submit_parser.add_argument("--draft", action="store_true")
    submit_parser.add_argument("--reviewer", action="append", default=[])

    review_parser = subparsers.add_parser("review-pr")
    review_parser.add_argument("--root", default=".")
    review_parser.add_argument("--pr", required=True, type=int)
    review_parser.add_argument("--event", choices=["auto", "APPROVE", "COMMENT", "REQUEST_CHANGES"], default="auto")
    review_parser.add_argument("--body", default="")
    review_parser.add_argument("--check-command", action="append", default=[])

    merge_parser = subparsers.add_parser("merge-pr")
    merge_parser.add_argument("--root", default=".")
    merge_parser.add_argument("--pr", required=True, type=int)
    merge_parser.add_argument("--method", choices=["merge", "squash", "rebase"], default="")
    merge_parser.add_argument("--commit-title", default="")
    merge_parser.add_argument("--commit-message", default="")

    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = load_github_config(root)
    base_branch = str(config.get("base_branch", "main"))

    if args.command == "doctor":
        payload = doctor(root)
    elif args.command == "submit-pr":
        title = str(args.title).strip()
        body = str(args.body).strip() or f"Automated Claw Team PR created at {datetime.now(timezone.utc).isoformat()}."
        commit_message = str(args.commit_message).strip() or f"Claw Team: {_slug(title)}"
        payload = submit_pr(
            root,
            title=title,
            body=body,
            base_branch=str(args.base or base_branch),
            commit_message=commit_message,
            branch_name=args.branch,
            draft=bool(args.draft),
            reviewers=list(args.reviewer),
        )
    elif args.command == "review-pr":
        auto_review = config.get("auto_review", {})
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
    else:
        auto_merge = config.get("auto_merge", {})
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

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
