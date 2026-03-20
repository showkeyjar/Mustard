from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from carm.schemas import EpisodeRecord, StepRecord


class ExperienceStore:
    """Persistent dialogue memory for online learning and recall."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, episode: EpisodeRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(episode), ensure_ascii=False) + "\n")

    def recall(self, query: str, limit: int = 2) -> list[EpisodeRecord]:
        if not self.path.exists():
            return []

        scored: list[tuple[int, EpisodeRecord]] = []
        query_terms = {token for token in query.lower().split() if token}

        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                episode = self._decode_episode(payload)
                score = self._score(query_terms, payload.get("user_input", ""), payload.get("summary", ""))
                if score > 0:
                    scored.append((score, episode))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [episode for _, episode in scored[:limit]]

    def _score(self, query_terms: set[str], user_input: str, summary: str) -> int:
        haystack = f"{user_input} {summary}".lower()
        return sum(1 for term in query_terms if term in haystack)

    def _decode_episode(self, payload: dict) -> EpisodeRecord:
        steps = [StepRecord(**step) for step in payload.get("steps", [])]
        return EpisodeRecord(
            user_input=payload["user_input"],
            answer=payload["answer"],
            summary=payload["summary"],
            success=payload["success"],
            value_score=payload["value_score"],
            steps=steps,
        )
