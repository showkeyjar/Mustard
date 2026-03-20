from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from carm.actions import Action
from carm.memory import MemoryBoard
from carm.schemas import ActionDecision, StepRecord, ToolCall
from carm.state import AgentState


@dataclass
class PolicyContext:
    user_input: str
    state: AgentState
    memory: MemoryBoard


class OnlinePolicy:
    """Hybrid policy with heuristic priors and lightweight online updates."""

    def __init__(self, state_path: str | Path) -> None:
        self.state_path = Path(state_path)
        self.learning_rate = 0.08
        self.action_weights: dict[str, dict[str, float]] = {
            action.value: {} for action in Action
        }
        self.bias: dict[str, float] = {action.value: 0.0 for action in Action}
        self._load()

    def decide(self, state: AgentState, memory: MemoryBoard, user_input: str) -> ActionDecision:
        context = PolicyContext(user_input=user_input, state=state, memory=memory)
        features = self.extract_features(context)
        priors = self.heuristic_priors(context)

        scores: dict[str, float] = {}
        for action in Action:
            score = priors.get(action.value, -0.4) + self.bias[action.value]
            weights = self.action_weights[action.value]
            for name, value in features.items():
                score += weights.get(name, 0.0) * value
            scores[action.value] = score

        chosen = max(scores, key=scores.get)
        return self._build_decision(Action(chosen), scores[chosen], features, context)

    def extract_features(self, context: PolicyContext) -> dict[str, float]:
        user_input = context.user_input
        lower = user_input.lower()
        memory = context.memory
        state = context.state

        return {
            "bias": 1.0,
            "step_idx": min(state.step_idx / 8.0, 1.0),
            "uncertainty": state.uncertainty,
            "answer_ready": state.answer_ready,
            "has_goal": 1.0 if memory.latest("GOAL") else 0.0,
            "has_plan": 1.0 if memory.latest("PLAN") else 0.0,
            "has_result": 1.0 if memory.latest("RESULT") else 0.0,
            "has_draft": 1.0 if memory.latest("DRAFT") else 0.0,
            "has_conflict": 1.0 if memory.latest("CONFLICT") else 0.0,
            "needs_compare": 1.0 if any(token in user_input for token in ("比较", "区别", "优缺点", "vs", "对比")) else 0.0,
            "needs_calc": 1.0 if any(token in user_input for token in ("多少", "计算", "price", "cost", "数字")) else 0.0,
            "needs_code": 1.0 if any(token in lower for token in ("python", "code", "script", "代码")) else 0.0,
            "last_verify": 1.0 if state.last_action == Action.VERIFY.value else 0.0,
            "last_tool": 1.0 if state.last_action in {Action.CALL_TOOL.value, Action.CALL_BIGMODEL.value} else 0.0,
        }

    def heuristic_priors(self, context: PolicyContext) -> dict[str, float]:
        state = context.state
        memory = context.memory
        user_input = context.user_input
        lower = user_input.lower()
        candidate_slot = state.hidden.get("slot_type", "")

        priors = {action.value: -0.5 for action in Action}

        if memory.latest("GOAL") is None:
            priors[Action.WRITE_MEM.value] = 1.4
            return priors

        if candidate_slot and memory.latest(candidate_slot) is None and candidate_slot in {"PLAN", "DRAFT"}:
            priors[Action.WRITE_MEM.value] = 1.2

        if state.uncertainty > 0.7 and memory.latest("RESULT") is None:
            if any(token in user_input for token in ("多少", "计算", "price", "cost", "数字")):
                priors[Action.CALL_TOOL.value] = 1.25
            elif any(token in lower for token in ("python", "code", "script", "代码")):
                priors[Action.CALL_TOOL.value] = 1.15
            elif any(token in user_input for token in ("比较", "区别", "优缺点", "vs", "对比")):
                priors[Action.CALL_TOOL.value] = 1.18
            else:
                priors[Action.CALL_BIGMODEL.value] = 1.05

        draft = memory.latest("DRAFT")
        if state.answer_ready >= 0.8 or (draft is not None and state.uncertainty <= 0.3):
            priors[Action.ANSWER.value] = 1.35

        if memory.latest("CONFLICT") is not None and state.step_idx < 6:
            priors[Action.ROLLBACK.value] = 1.1

        if draft and memory.latest("CONFLICT") is None:
            priors[Action.VERIFY.value] = max(priors[Action.VERIFY.value], 0.95)

        priors[Action.THINK.value] = max(priors[Action.THINK.value], 0.1)
        return priors

    def _build_decision(
        self,
        action: Action,
        score: float,
        features: dict[str, float],
        context: PolicyContext,
    ) -> ActionDecision:
        state = context.state
        memory = context.memory
        user_input = context.user_input
        candidate_slot = state.hidden.get("slot_type", "")

        decision = ActionDecision(
            action=action,
            score=score,
            reason=f"Hybrid policy selected {action.value.lower()} with score {score:.3f}.",
            feature_snapshot=features,
        )

        if action == Action.WRITE_MEM:
            target = "GOAL" if memory.latest("GOAL") is None else candidate_slot or "DRAFT"
            decision.target_slot = target
            decision.reason = f"Persist {target.lower()} into working memory."
            return decision

        if action == Action.CALL_TOOL:
            if features["needs_calc"] > 0.0:
                decision.tool_call = ToolCall(
                    tool_name="calculator",
                    query=user_input,
                    reason="Need precise calculation support.",
                )
                decision.reason = "Use calculator for precision."
            elif features["needs_code"] > 0.0:
                decision.tool_call = ToolCall(
                    tool_name="code_executor",
                    query="print('mock code execution')",
                    reason="Need executable confirmation.",
                )
                decision.reason = "Use code executor for implementation validation."
            else:
                decision.tool_call = ToolCall(
                    tool_name="search",
                    query=user_input,
                    arguments={"top_k": 3},
                    reason="Need external evidence.",
                )
                decision.reason = "Use search tool for external support."
            return decision

        if action == Action.CALL_BIGMODEL:
            decision.tool_call = ToolCall(
                tool_name="bigmodel_proxy",
                query=user_input,
                reason="Need stronger external reasoning support.",
            )
            decision.reason = "Escalate to larger external model."
            return decision

        if action == Action.READ_MEM:
            decision.target_slot = "RESULT" if memory.latest("RESULT") else "PLAN"
            return decision

        if action == Action.VERIFY:
            decision.reason = "Verify draft consistency before answer."
            return decision

        if action == Action.ROLLBACK:
            decision.reason = "Rollback due to detected conflict."
            return decision

        if action == Action.ANSWER:
            decision.reason = "Current draft is ready enough to answer."
            return decision

        decision.reason = "Continue internal reasoning."
        return decision

    def learn(self, steps: list[StepRecord]) -> None:
        for step in steps:
            if not step.high_value:
                continue

            action_name = step.action
            if action_name not in self.action_weights:
                continue

            for feature, value in step.feature_snapshot.items():
                current = self.action_weights[action_name].get(feature, 0.0)
                self.action_weights[action_name][feature] = current + self.learning_rate * step.reward * value
            self.bias[action_name] += self.learning_rate * step.reward

        self._save()

    def export_state(self) -> dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "bias": self.bias,
            "action_weights": self.action_weights,
        }

    def _load(self) -> None:
        if not self.state_path.exists():
            return

        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
        self.bias.update(payload.get("bias", {}))
        stored_weights = payload.get("action_weights", {})
        for action_name, weights in stored_weights.items():
            if action_name in self.action_weights:
                self.action_weights[action_name].update(weights)

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self.export_state(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
