from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from carm.normalize import normalize_episode_payload
from carm.schemas import EpisodeRecord, StepRecord


class ExperienceStore:
    """Persistent dialogue memory for online learning and recall."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, episode: EpisodeRecord) -> None:
        payload = normalize_episode_payload(asdict(episode))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def recall(self, query: str, limit: int = 2) -> list[EpisodeRecord]:
        if not self.path.exists():
            return []

        scored: list[tuple[int, EpisodeRecord]] = []
        query_terms = set(self._query_terms(query))

        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = normalize_episode_payload(json.loads(line))
                episode = self._decode_episode(payload)
                score = self._score(query_terms, payload)
                if score > 0:
                    scored.append((score, episode))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [episode for _, episode in scored[:limit]]

    def _score(self, query_terms: set[str], payload: dict) -> int:
        features = payload.get("episode_features", {})
        outcome = payload.get("outcome_signature", {})

        score = 0
        haystacks = [
            str(payload.get("user_input", "")).lower(),
            str(payload.get("summary", "")).lower(),
            " ".join(str(item).lower() for item in features.get("keywords", [])),
            " ".join(str(item).lower() for item in features.get("plan_unknowns", [])),
            " ".join(str(item).lower() for item in features.get("evidence_targets", [])),
            str(features.get("draft_summary", "")).lower(),
        ]
        for term in query_terms:
            if any(term in haystack for haystack in haystacks):
                score += 1

        if payload.get("success"):
            score += 1
        if float(outcome.get("value_score", payload.get("value_score", 0.0))) >= 0.7:
            score += 1
        return score

    def _query_terms(self, query: str) -> list[str]:
        lowered = query.lower()
        terms = [token for token in lowered.split() if token]
        if terms:
            return terms
        return [query]

    def _decode_episode(self, payload: dict) -> EpisodeRecord:
        normalized = normalize_episode_payload(payload)
        steps = [StepRecord(**step) for step in normalized.get("steps", [])]
        return EpisodeRecord(
            user_input=normalized["user_input"],
            answer=normalized["answer"],
            summary=normalized["summary"],
            success=normalized["success"],
            value_score=normalized["value_score"],
            episode_features=normalized.get("episode_features", {}),
            outcome_signature=normalized.get("outcome_signature", {}),
            steps=steps,
        )
