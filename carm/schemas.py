from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from carm.actions import Action


@dataclass
class ToolCall:
    tool_name: str
    query: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class ActionDecision:
    action: Action
    score: float
    reason: str
    target_slot: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    feature_snapshot: dict[str, float] = field(default_factory=dict)


@dataclass
class ToolResult:
    ok: bool
    tool_name: str
    result: str
    confidence: float
    source: str


@dataclass
class StepRecord:
    step_idx: int
    action: str
    reason: str
    score: float
    feature_snapshot: dict[str, float] = field(default_factory=dict)
    reward: float = 0.0
    high_value: bool = False


@dataclass
class EpisodeRecord:
    user_input: str
    answer: str
    summary: str
    success: bool
    value_score: float
    steps: list[StepRecord] = field(default_factory=list)
