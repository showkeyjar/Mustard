from __future__ import annotations

import json
from pathlib import Path

from carm.normalize import normalize_episode_payload


def migrate_file(path: Path) -> int:
    if not path.exists():
        print(f"Skip missing file: {path}")
        return 0

    lines = path.read_text(encoding="utf-8").splitlines()
    migrated: list[str] = []
    changed = 0
    for line in lines:
        if not line.strip():
            continue
        payload = json.loads(line)
        normalized = normalize_episode_payload(payload)
        if normalized != payload:
            changed += 1
        migrated.append(json.dumps(normalized, ensure_ascii=False))

    path.write_text("\n".join(migrated) + ("\n" if migrated else ""), encoding="utf-8")
    print(f"Migrated {changed} episode(s) in {path}")
    return changed


def main() -> int:
    target = Path("data/experience/episodes.jsonl")
    migrate_file(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
