from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from carm.schemas import ReviewRecord


class ReviewStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, review: ReviewRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(review), ensure_ascii=False) + "\n")

    def load_all(self) -> list[ReviewRecord]:
        if not self.path.exists():
            return []
        reviews: list[ReviewRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                reviews.append(ReviewRecord(**payload))
        return reviews
