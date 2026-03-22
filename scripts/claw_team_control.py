from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Control the Mustard Claw Team workflow.")
    parser.add_argument("command", choices=["bootstrap", "run", "status", "doctor"])
    parser.add_argument("--root", default=".", help="Repository root. Defaults to current directory.")
    parser.add_argument("--json", action="store_true", dest="as_json")
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
    elif args.command == "status":
        payload = status(root)
    else:
        payload = doctor(root)

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
