from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from carm.actions import Action
from carm.concepts import AdaptiveConceptModel
from carm.memory import MemoryBoard
from carm.runtime_controls import DEFAULT_CONTROLS
from carm.schemas import ActionDecision, StepRecord, ToolCall
from carm.state import AgentState


@dataclass
class PolicyContext:
    user_input: str
    state: AgentState
    memory: MemoryBoard


class OnlinePolicy:
    """Hybrid policy with heuristic priors and lightweight online updates."""

    def __init__(
        self,
        state_path: str | Path,
        concept_state_path: str | Path | None = None,
        controls: dict[str, float | int] | None = None,
    ) -> None:
        self.state_path = Path(state_path)
        if concept_state_path is None:
            concept_state_path = self.state_path.with_name("concept_state.json")
        self.concepts = AdaptiveConceptModel(concept_state_path)
        base_controls = dict(DEFAULT_CONTROLS["policy"])
        if controls:
            base_controls.update(controls)
        self.controls = base_controls
        self.learning_rate = 0.08
        self.action_weights: dict[str, dict[str, float]] = {
            action.value: {} for action in Action
        }
        self.bias: dict[str, float] = {action.value: 0.0 for action in Action}
        self._load()

    def decide(
        self,
        state: AgentState,
        memory: MemoryBoard,
        user_input: str,
        guidance: dict[str, object] | None = None,
    ) -> ActionDecision:
        context = PolicyContext(user_input=user_input, state=state, memory=memory)
        features = self.extract_features(context, guidance)
        priors = self.heuristic_priors(context, guidance)
        concept_priors = self.concepts.action_priors(user_input)

        scores: dict[str, float] = {}
        for action in Action:
            score = priors.get(action.value, -0.4) + concept_priors.get(action.value, 0.0) + self.bias[action.value]
            weights = self.action_weights[action.value]
            for name, value in features.items():
                score += weights.get(name, 0.0) * value
            scores[action.value] = score

        chosen = max(scores, key=scores.get)
        decision = self._build_decision(Action(chosen), scores[chosen], features, context, guidance)
        return self._enforce_constraints(decision, context)

    def extract_features(self, context: PolicyContext, guidance: dict[str, object] | None = None) -> dict[str, float]:
        user_input = context.user_input
        lower = user_input.lower()
        memory = context.memory
        state = context.state

        semantic_pressure = self.concepts.action_priors(user_input)
        preferred_tool = self.concepts.preferred_tool(user_input)
        guided_tool = str((guidance or {}).get("preferred_tool", ""))

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
            "needs_calc": 1.0
            if any(token in user_input for token in ("多少", "计算", "price", "cost", "数字", "预算", "总价", "每席位", "按年", "每月", "分几批"))
            or ("*" in user_input or "/" in user_input or "+" in user_input or "-" in user_input)
            else 0.0,
            "needs_code": 1.0 if any(token in lower for token in ("python", "code", "script", "代码")) else 0.0,
            "needs_formal_synthesis": 1.0
            if any(token in user_input for token in ("负责人", "管理层", "正式", "简洁", "组织", "结论", "几份资料", "材料"))
            else 0.0,
            "concept_tool_search": 1.0 if preferred_tool == "search" else 0.0,
            "concept_tool_calc": 1.0 if preferred_tool == "calculator" else 0.0,
            "concept_tool_code": 1.0 if preferred_tool == "code_executor" else 0.0,
            "concept_tool_bigmodel": 1.0 if preferred_tool == "bigmodel_proxy" else 0.0,
            "user_tool_search": 1.0 if guided_tool == "search" else 0.0,
            "user_tool_calc": 1.0 if guided_tool == "calculator" else 0.0,
            "user_tool_code": 1.0 if guided_tool == "code_executor" else 0.0,
            "user_tool_bigmodel": 1.0 if guided_tool == "bigmodel_proxy" else 0.0,
            "concept_call_tool": min(max(semantic_pressure.get(Action.CALL_TOOL.value, 0.0), 0.0), 1.0),
            "concept_call_bigmodel": min(max(semantic_pressure.get(Action.CALL_BIGMODEL.value, 0.0), 0.0), 1.0),
            "glance_prefer_tool": 1.0 if state.hidden.get("glance_suggestion") == "prefer_tool" else 0.0,
            "glance_promote_draft": 1.0 if state.hidden.get("glance_suggestion") == "promote_draft" else 0.0,
            "glance_delay_answer": 1.0 if state.hidden.get("glance_suggestion") == "delay_answer" else 0.0,
            "glance_mark_conflict": 1.0 if state.hidden.get("glance_suggestion") == "mark_conflict" else 0.0,
            "last_verify": 1.0 if state.last_action == Action.VERIFY.value else 0.0,
            "last_tool": 1.0 if state.last_action in {Action.CALL_TOOL.value, Action.CALL_BIGMODEL.value} else 0.0,
            "block_learning": 1.0 if bool((guidance or {}).get("block_learning")) else 0.0,
        }

    def heuristic_priors(self, context: PolicyContext, guidance: dict[str, object] | None = None) -> dict[str, float]:
        state = context.state
        memory = context.memory
        candidate_slot = state.hidden.get("slot_type", "")

        priors = {action.value: -0.5 for action in Action}

        if memory.latest("GOAL") is None:
            priors[Action.WRITE_MEM.value] = 1.4
            return priors

        if candidate_slot and memory.latest(candidate_slot) is None and candidate_slot in {"PLAN", "DRAFT"}:
            priors[Action.WRITE_MEM.value] = 1.2

        if state.uncertainty > 0.7 and memory.latest("RESULT") is None:
            priors[Action.CALL_TOOL.value] = 0.55 + float(self.controls.get("call_tool_bonus", 0.0))
            priors[Action.CALL_BIGMODEL.value] = 0.45

        preferred_tool = str((guidance or {}).get("preferred_tool", ""))
        if preferred_tool:
            priors[Action.CALL_TOOL.value] = max(priors[Action.CALL_TOOL.value], 1.0)

        if state.hidden.get("glance_suggestion") == "prefer_tool":
            priors[Action.CALL_TOOL.value] = max(priors[Action.CALL_TOOL.value], 1.1)

        if state.hidden.get("glance_suggestion") == "promote_draft":
            priors[Action.WRITE_MEM.value] = max(priors[Action.WRITE_MEM.value], 1.15)

        if state.hidden.get("glance_suggestion") == "delay_answer":
            priors[Action.ANSWER.value] = min(priors[Action.ANSWER.value], -0.2 - float(self.controls.get("answer_penalty", 0.0)))
            priors[Action.VERIFY.value] = max(priors[Action.VERIFY.value], 0.9 + float(self.controls.get("verify_bonus", 0.0)))

        if state.hidden.get("glance_suggestion") == "mark_conflict":
            priors[Action.ROLLBACK.value] = max(priors[Action.ROLLBACK.value], 1.2)

        draft = memory.latest("DRAFT")
        if draft is not None and (state.answer_ready >= 0.8 or state.uncertainty <= 0.3):
            priors[Action.ANSWER.value] = 1.35

        if memory.latest("CONFLICT") is not None and state.step_idx < 6:
            priors[Action.ROLLBACK.value] = 1.1

        if draft and memory.latest("CONFLICT") is None:
            priors[Action.VERIFY.value] = max(priors[Action.VERIFY.value], 0.95 + float(self.controls.get("verify_bonus", 0.0)))

        priors[Action.THINK.value] = max(priors[Action.THINK.value], 0.1 - float(self.controls.get("think_penalty", 0.0)))
        return priors

    def _build_decision(
        self,
        action: Action,
        score: float,
        features: dict[str, float],
        context: PolicyContext,
        guidance: dict[str, object] | None = None,
    ) -> ActionDecision:
        state = context.state
        memory = context.memory
        candidate_slot = state.hidden.get("slot_type", "")
        user_input = context.user_input
        preferred_tool = str((guidance or {}).get("preferred_tool", "")) or self.concepts.preferred_tool(user_input)

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
            if preferred_tool == "calculator":
                decision.tool_call = ToolCall(
                    tool_name="calculator",
                    query=user_input,
                    reason="Need precise calculation support.",
                )
                decision.reason = "Use calculator for precision."
            elif preferred_tool == "bigmodel_proxy":
                decision.tool_call = ToolCall(
                    tool_name="bigmodel_proxy",
                    query=user_input,
                    reason="Need stronger external synthesis support.",
                )
                decision.reason = "Use larger model for formal synthesis."
            elif preferred_tool == "code_executor":
                decision.tool_call = ToolCall(
                    tool_name="code_executor",
                    query="print('mock code execution')",
                    reason="Need executable confirmation.",
                )
                decision.reason = "Use code executor for implementation validation."
            elif preferred_tool:
                decision.tool_call = ToolCall(
                    tool_name=preferred_tool,
                    query=user_input,
                    arguments={"top_k": 3},
                    reason="Use the learned preferred tool for this task shape.",
                )
                decision.reason = f"Use {preferred_tool} from learned preference."
            elif features["needs_formal_synthesis"] > 0.0:
                decision.tool_call = ToolCall(
                    tool_name="bigmodel_proxy",
                    query=user_input,
                    reason="Need polished synthesis for higher-stakes communication.",
                )
                decision.reason = "Use larger model for formal synthesis."
            elif features["needs_calc"] > 0.0 and features["needs_code"] == 0.0:
                decision.tool_call = ToolCall(
                    tool_name="calculator",
                    query=user_input,
                    reason="Need precise calculation support.",
                )
                decision.reason = "Use calculator for precision."
            elif features["needs_calc"] > 0.0 and any(token in user_input for token in ("分几批", "预算", "总价", "每席位", "按年", "每月")):
                decision.tool_call = ToolCall(
                    tool_name="calculator",
                    query=user_input,
                    reason="Budgeting and batching questions should prefer exact arithmetic.",
                )
                decision.reason = "Use calculator for arithmetic-heavy task."
            elif features["needs_code"] > 0.0:
                decision.tool_call = ToolCall(
                    tool_name="code_executor",
                    query="print('mock code execution')",
                    reason="Need executable confirmation.",
                )
                decision.reason = "Use code executor for implementation validation."
            else:
                decision.tool_call = ToolCall(
                    tool_name=preferred_tool or "search",
                    query=user_input,
                    arguments={"top_k": 3},
                    reason="Need external evidence.",
                )
                decision.reason = f"Use {decision.tool_call.tool_name} for external support."
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

        self.concepts.learn(steps, self.learning_rate)
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

    def _enforce_constraints(self, decision: ActionDecision, context: PolicyContext) -> ActionDecision:
        memory = context.memory
        state = context.state

        if (
            state.last_action == Action.VERIFY.value
            and memory.latest("DRAFT") is not None
            and memory.latest("CONFLICT") is None
            and (state.answer_ready >= 0.8 or state.uncertainty <= 0.3 or state.hidden.get("verified") == "1")
        ):
            return self._build_decision(
                Action.ANSWER,
                decision.score,
                decision.feature_snapshot,
                context,
            )

        if memory.latest("RESULT") is not None and memory.latest("DRAFT") is None:
            return ActionDecision(
                action=Action.WRITE_MEM,
                score=decision.score,
                reason="Materialize a structured draft from the available result.",
                target_slot="DRAFT",
                feature_snapshot=dict(decision.feature_snapshot),
            )

        if decision.action == Action.ANSWER and memory.latest("DRAFT") is None:
            if memory.latest("RESULT") is None and state.uncertainty > 0.5:
                return self._build_decision(
                    Action.CALL_TOOL,
                    decision.score,
                    decision.feature_snapshot,
                    context,
                )
            return self._build_decision(
                Action.THINK,
                decision.score,
                decision.feature_snapshot,
                context,
            )

        if decision.action == Action.CALL_TOOL and memory.latest("RESULT") is not None:
            if memory.latest("DRAFT") is None:
                return self._build_decision(
                    Action.WRITE_MEM,
                    decision.score,
                    decision.feature_snapshot,
                    context,
                )
            return self._build_decision(
                Action.VERIFY,
                decision.score,
                decision.feature_snapshot,
                context,
            )

        if decision.action == Action.WRITE_MEM and decision.target_slot == "GOAL" and memory.latest("GOAL") is not None:
            fallback = Action.THINK if memory.latest("DRAFT") is None else Action.VERIFY
            return self._build_decision(
                fallback,
                decision.score,
                decision.feature_snapshot,
                context,
            )

        return decision
