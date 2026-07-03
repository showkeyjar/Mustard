"""Short-term session memory for CARM.

Provides conversational context tracking across multiple turns within a single
session. Unlike ExperienceStore (which persists across sessions for learning),
session memory is:

- Ephemeral: cleared when the session ends
- Fast: JSONL append, no DB, no heavy indexing
- Contextual: stores tool outputs and user intents from recent turns

Use cases:
- "上次查的那篇论文的核心结论是什么" -> retrieve last search result
- "它的性能怎么样" -> "它" refers to topic from previous query
- "用刚才的模型再跑一遍" -> "刚才的模型" from previous tool call

"""

from __future__ import annotations

import json
import re
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TurnRecord:
    """A single conversational turn."""

    turn_id: int
    user_input: str
    tool_name: str
    tool_result: str
    confidence: float
    timestamp: str = ""
    # Extracted entities from the user input for anaphora resolution
    entities: list[str] = field(default_factory=list)


@dataclass
class SessionContext:
    """Context window for a single session."""

    session_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    # Entity map: entity_name -> last turn_id where it appeared
    entity_index: dict[str, int] = field(default_factory=dict)

    def add_turn(self, record: TurnRecord) -> None:
        self.turns.append(record)
        # Update entity index
        for entity in record.entities:
            self.entity_index[entity] = record.turn_id

    def last_result_of(self, tool_name: str | None = None) -> str:
        """Return the most recent tool result, optionally filtered by tool."""
        for turn in reversed(self.turns):
            if tool_name is None or turn.tool_name == tool_name:
                return turn.tool_result
        return ""

    def last_turn(self) -> TurnRecord | None:
        return self.turns[-1] if self.turns else None

    def resolve_anaphora(self, query: str) -> str | None:
        """Try to resolve anaphoric references (它/这/刚才/上次) in query.

        Returns the resolved entity string if found, None otherwise.
        """
        anaphora_markers = ("它", "这", "刚才", "上次", "之前", "那个", "那篇", "那个")
        if not any(m in query for m in anaphora_markers):
            return None

        # Strategy 1: "上次/刚才/之前" -> find last tool result
        if any(m in query for m in ("上次", "刚才", "之前")):
            # Look for topic keywords in the query (excluding anaphora markers)
            # e.g., "上次查的那篇论文的核心结论是什么"
            # Try to find what the user is asking about from the last turn
            last = self.last_turn()
            if last and last.tool_result:
                # Simple heuristic: return the last tool result as context
                return last.tool_result

        # Strategy 2: "它/这/那个" -> find the most salient entity from recent turns
        if any(m in query for m in ("它", "这", "那个", "那篇")):
            # Get the last turn's tool result, truncated
            last = self.last_turn()
            if last and last.tool_result:
                return last.tool_result

        return None


def _extract_entities(text: str) -> list[str]:
    """Extract candidate entities from user input for anaphora resolution."""
    # Simple noun phrase extraction: look for sequences of Chinese chars
    # that look like proper nouns (at least 2 chars)
    matches = re.findall(r"[\u4e00-\u9fff]{2,}(?:[的之])?", text)
    # Filter out common stop words
    stops = {
        "什么",
        "怎么",
        "多少",
        "哪里",
        "怎么样",
        "为什么",
        "能不能",
        "可以",
        "帮我",
        "给我",
    }
    return [m.strip("的之") for m in matches if m.strip("的之") not in stops]


class SessionMemoryManager:
    """Manages short-term session memory across multiple turns.

    Thread-safe singleton-like access via get_instance().
    """

    _instance: SessionMemoryManager | None = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, log_path: str | Path | None = None) -> SessionMemoryManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(log_path)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def __init__(self, log_path: str | Path | None = None) -> None:
        self.log_path = Path(log_path or "data/sessions/session_log.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # In-memory sessions: session_id -> SessionContext
        self._sessions: OrderedDict[str, SessionContext] = OrderedDict()
        self._turn_counter: dict[str, int] = {}
        self._file_lock = threading.Lock()

    def get_or_create(self, session_id: str) -> SessionContext:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionContext(session_id=session_id)
            self._turn_counter[session_id] = 0
        # Move to end (LRU)
        self._sessions.move_to_end(session_id)
        return self._sessions[session_id]

    def append_turn(
        self,
        session_id: str,
        user_input: str,
        tool_name: str,
        tool_result: str,
        confidence: float,
        timestamp: str = "",
    ) -> None:
        """Append a new turn to the session and persist to JSONL."""
        ctx = self.get_or_create(session_id)
        self._turn_counter[session_id] += 1
        record = TurnRecord(
            turn_id=self._turn_counter[session_id],
            user_input=user_input,
            tool_name=tool_name,
            tool_result=tool_result,
            confidence=confidence,
            timestamp=timestamp,
            entities=_extract_entities(user_input),
        )
        ctx.add_turn(record)

        # Persist to disk (append-only, thread-safe)
        with self._file_lock:
            with self.log_path.open("a", encoding="utf-8") as f:
                json.dump(
                    {
                        "session_id": session_id,
                        "turn": asdict(record),
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")

    def get_context(self, session_id: str) -> SessionContext | None:
        return self._sessions.get(session_id)

    def resolve_query(self, session_id: str, query: str) -> tuple[str | None, str]:
        """Try to resolve anaphora in a query using session context.

        Returns: (resolved_entity_or_None, resolved_query_with_context)
        """
        ctx = self.get_context(session_id)
        if ctx is None:
            return None, query

        resolved = ctx.resolve_anaphora(query)
        if resolved is None:
            return None, query

        # Enhance the query with context prefix
        enhanced = f"上下文：{resolved[:200]}\n用户问题：{query}"
        return resolved, enhanced

    def get_last_tool_result(
        self, session_id: str, tool_name: str | None = None
    ) -> str:
        """Get the last tool result for anaphora enrichment."""
        ctx = self.get_context(session_id)
        if ctx is None:
            return ""
        return ctx.last_result_of(tool_name)

    def clear_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
            del self._turn_counter[session_id]
