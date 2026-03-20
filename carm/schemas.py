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
    state_signature: dict[str, Any] = field(default_factory=dict)
    memory_signature: dict[str, Any] = field(default_factory=dict)
    user_input: str = ""
    selected_tool: str = ""
    target_slot: str = ""
    reward: float = 0.0
    reward_reason: str = ""
    high_value: bool = False
    glance_used: bool = False
    glance_helped: bool = False


@dataclass
class EpisodeRecord:
    user_input: str
    answer: str
    summary: str
    success: bool
    value_score: float
    episode_features: dict[str, Any] = field(default_factory=dict)
    outcome_signature: dict[str, Any] = field(default_factory=dict)
    steps: list[StepRecord] = field(default_factory=list)


@dataclass
class ReviewRecord:
    user_input: str
    success: bool
    value_score: float
    issue_tags: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    target_modules: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
