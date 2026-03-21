from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from carm.actions import Action
from carm.schemas import StepRecord


@dataclass
class EvolutionSignal:
    source: str
    query: str = ""
    goal: str = ""
    preferred_tool: str = ""
    preferred_slot: str = ""
    reward: float = 0.0
    learn: bool = True
    correction: str = ""
    note: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


class OnlineEvolutionManager:
    def __init__(self, state_path: str | Path, signal_log_path: str | Path | None = None) -> None:
        self.state_path = Path(state_path)
        self.signal_log_path = Path(signal_log_path) if signal_log_path is not None else self.state_path.with_name("signals.jsonl")
        self.current_goal = ""
        self.global_tool = ""
        self.global_slot = ""
        self.learning_rate_scale = 1.0
        self.token_tool_overrides: dict[str, str] = {}
        self.token_slot_overrides: dict[str, str] = {}
        self.blocked_terms: set[str] = set()
        self.signal_count = 0
        self._load()

    def guidance_for(self, user_input: str) -> dict[str, object]:
        tokens = self.tokenize(user_input)
        preferred_tool = self.global_tool
        preferred_slot = self.global_slot

        for token in tokens:
            override_tool = self.token_tool_overrides.get(token)
            if override_tool:
                preferred_tool = override_tool
            override_slot = self.token_slot_overrides.get(token)
            if override_slot:
                preferred_slot = override_slot

        should_block_learning = any(token in self.blocked_terms for token in tokens)
        return {
            "current_goal": self.current_goal,
            "preferred_tool": preferred_tool,
            "preferred_slot": preferred_slot,
            "learning_rate_scale": self.learning_rate_scale,
            "block_learning": should_block_learning,
        }

    def apply_signal(self, signal: EvolutionSignal) -> list[StepRecord]:
        self.signal_count += 1
        tokens = self.tokenize(signal.query or signal.goal or signal.correction or signal.note)

        if signal.goal.strip():
            self.current_goal = signal.goal.strip()
        if signal.preferred_tool.strip():
            if tokens:
                for token in tokens:
                    self.token_tool_overrides[token] = signal.preferred_tool.strip()
            else:
                self.global_tool = signal.preferred_tool.strip()
        if signal.preferred_slot.strip():
            if tokens:
                for token in tokens:
                    self.token_slot_overrides[token] = signal.preferred_slot.strip()
            else:
                self.global_slot = signal.preferred_slot.strip()
        if not signal.learn:
            self.blocked_terms.update(tokens)
        if signal.reward != 0.0:
            self.learning_rate_scale = max(0.2, min(2.0, self.learning_rate_scale + signal.reward * 0.15))

        self._append_signal(signal)
        self._save()
        return self.synthetic_steps(signal)

    def synthetic_steps(self, signal: EvolutionSignal) -> list[StepRecord]:
        if not signal.learn:
            return []

        query = signal.query or signal.goal or signal.correction or signal.note
        if not query.strip():
            return []

        reward = signal.reward if signal.reward != 0.0 else 1.0
        steps: list[StepRecord] = []

        if signal.preferred_tool:
            steps.append(
                StepRecord(
                    step_idx=1,
                    action=Action.CALL_TOOL.value,
                    reason=f"User directed tool preference toward {signal.preferred_tool}.",
                    score=abs(reward),
                    feature_snapshot={"bias": 1.0},
                    user_input=query,
                    selected_tool=signal.preferred_tool,
                    reward=reward,
                    reward_reason="user_tool_guidance",
                    high_value=True,
                )
            )

        if signal.preferred_slot:
            steps.append(
                StepRecord(
                    step_idx=1,
                    action=Action.WRITE_MEM.value,
                    reason=f"User directed reasoning toward {signal.preferred_slot}.",
                    score=abs(reward),
                    feature_snapshot={"bias": 1.0, "need_structure": 1.0 if signal.preferred_slot == "PLAN" else 0.0},
                    user_input=query,
                    target_slot=signal.preferred_slot,
                    reward=reward,
                    reward_reason="user_slot_guidance",
                    high_value=True,
                )
            )

        return steps

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        current_ascii: list[str] = []
        for char in text.lower():
            if char.isascii() and (char.isalnum() or char in {"_", "-", "#", "."}):
                current_ascii.append(char)
                continue
            if len(current_ascii) >= 2:
                tokens.append("".join(current_ascii))
            current_ascii = []
        if len(current_ascii) >= 2:
            tokens.append("".join(current_ascii))

        current_han: list[str] = []
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                current_han.append(char)
                continue
            if len(current_han) >= 2:
                tokens.append("".join(current_han))
            current_han = []
        if len(current_han) >= 2:
            tokens.append("".join(current_han))
        return list(dict.fromkeys(tokens))

    def _append_signal(self, signal: EvolutionSignal) -> None:
        self.signal_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.signal_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(signal), ensure_ascii=False) + "\n")

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.current_goal = str(payload.get("current_goal", ""))
        self.global_tool = str(payload.get("global_tool", ""))
        self.global_slot = str(payload.get("global_slot", ""))
        self.learning_rate_scale = float(payload.get("learning_rate_scale", 1.0))
        self.token_tool_overrides = {str(key): str(value) for key, value in payload.get("token_tool_overrides", {}).items()}
        self.token_slot_overrides = {str(key): str(value) for key, value in payload.get("token_slot_overrides", {}).items()}
        self.blocked_terms = {str(item) for item in payload.get("blocked_terms", [])}
        self.signal_count = int(payload.get("signal_count", 0))

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "current_goal": self.current_goal,
            "global_tool": self.global_tool,
            "global_slot": self.global_slot,
            "learning_rate_scale": self.learning_rate_scale,
            "token_tool_overrides": self.token_tool_overrides,
            "token_slot_overrides": self.token_slot_overrides,
            "blocked_terms": sorted(self.blocked_terms),
            "signal_count": self.signal_count,
        }
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
