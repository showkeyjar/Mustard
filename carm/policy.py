from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from carm.actions import Action
from carm.concepts import AdaptiveConceptModel
from carm.memory import MemoryBoard
from carm.runtime_controls import DEFAULT_CONTROLS
from carm.schemas import ActionDecision, StepRecord, ToolCall
from carm.intent import IntentCategory, DEFAULT_TOOL_MAP
from carm.semantic import SemanticEncoder
from carm.signals import (
    is_conflict_task,
    has_compare_signal,
    has_calc_signal,
    has_code_signal,
    has_formal_signal,
    has_comparison_evidence_signal,
    has_explain_signal,
    has_writing_signal,
    has_search_action_signal,
    has_translate_signal,
    has_polish_signal,
    has_consult_signal,
    has_travel_signal,
    has_debug_consult_signal,
    has_deep_reason_signal,
    has_deep_analysis_signal,
    has_anaphora_signal,
    has_multi_intent_signal,
    has_multi_step_signal,
    has_low_intent_signal,
)
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
        tool_manager: object | None = None,
    ) -> None:
        self.state_path = Path(state_path)
        if concept_state_path is None:
            concept_state_path = self.state_path.with_name("concept_state.json")
        self.concepts = AdaptiveConceptModel(concept_state_path)
        self.semantic = SemanticEncoder()
        self._tool_manager = tool_manager  # ToolManager reference for dynamic routing
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

    def _resolve_tool_name(self, category: IntentCategory) -> str:
        """Map an IntentCategory to an actual tool name.

        Priority:
        1. ToolManager.find_by_capability() if a tool manager is registered
        2. DEFAULT_TOOL_MAP fallback
        """
        if self._tool_manager is not None:
            result = self._tool_manager.find_by_capability(category)
            if result is not None:
                return result
        return DEFAULT_TOOL_MAP.get(category, "search")

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
            score = (
                priors.get(action.value, -0.4)
                + concept_priors.get(action.value, 0.0)
                + self.bias[action.value]
            )
            weights = self.action_weights[action.value]
            for name, value in features.items():
                score += weights.get(name, 0.0) * value
            scores[action.value] = score

        chosen = max(scores, key=scores.get)
        decision = self._build_decision(
            Action(chosen), scores[chosen], features, context, guidance
        )
        return self._enforce_constraints(decision, context)

    def extract_features(
        self, context: PolicyContext, guidance: dict[str, object] | None = None
    ) -> dict[str, float]:
        user_input = context.user_input
        lower = user_input.lower()
        memory = context.memory
        state = context.state

        semantic_pressure = self.concepts.action_priors(user_input)
        preferred_tool = self.concepts.preferred_tool(user_input)
        guided_tool = str((guidance or {}).get("preferred_tool", ""))

        # Semantic intent signals from the encoder
        intent_scores = self.semantic.intent_scores(user_input)

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
            "needs_compare": 1.0 if has_compare_signal(user_input) else 0.0,
            "needs_conflict_detection": 1.0 if is_conflict_task(user_input) else 0.0,
            "needs_calc": 1.0 if has_calc_signal(user_input) else 0.0,
            "needs_code": 1.0 if has_code_signal(user_input) else 0.0,
            "needs_formal_synthesis": 1.0 if has_formal_signal(user_input) else 0.0,
            "concept_tool_search": 1.0
            if preferred_tool in ("search", IntentCategory.SEARCH.value)
            else 0.0,
            "concept_tool_calc": 1.0
            if preferred_tool in ("calculator", IntentCategory.CALC.value)
            else 0.0,
            "concept_tool_code": 1.0
            if preferred_tool in ("code_executor", IntentCategory.CODE.value)
            else 0.0,
            "concept_tool_bigmodel": 1.0
            if preferred_tool in ("bigmodel_proxy", IntentCategory.CONSULT.value)
            else 0.0,
            "user_tool_search": 1.0
            if guided_tool in ("search", IntentCategory.SEARCH.value)
            else 0.0,
            "user_tool_calc": 1.0
            if guided_tool in ("calculator", IntentCategory.CALC.value)
            else 0.0,
            "user_tool_code": 1.0
            if guided_tool in ("code_executor", IntentCategory.CODE.value)
            else 0.0,
            "user_tool_bigmodel": 1.0
            if guided_tool in ("bigmodel_proxy", IntentCategory.CONSULT.value)
            else 0.0,
            "semantic_search": intent_scores.get("search", 0.0),
            "semantic_calculator": intent_scores.get("calculator", 0.0),
            "semantic_code": intent_scores.get("code_executor", 0.0),
            "semantic_bigmodel": intent_scores.get("bigmodel_proxy", 0.0),
            "semantic_ambiguous": intent_scores.get("ambiguous", 0.0),
            "concept_call_tool": min(
                max(semantic_pressure.get(Action.CALL_TOOL.value, 0.0), 0.0), 1.0
            ),
            "concept_call_bigmodel": min(
                max(semantic_pressure.get(Action.CALL_BIGMODEL.value, 0.0), 0.0), 1.0
            ),
            "glance_prefer_tool": 1.0
            if state.hidden.get("glance_suggestion") == "prefer_tool"
            else 0.0,
            "glance_promote_draft": 1.0
            if state.hidden.get("glance_suggestion") == "promote_draft"
            else 0.0,
            "glance_delay_answer": 1.0
            if state.hidden.get("glance_suggestion") == "delay_answer"
            else 0.0,
            "glance_mark_conflict": 1.0
            if state.hidden.get("glance_suggestion") == "mark_conflict"
            else 0.0,
            "last_verify": 1.0 if state.last_action == Action.VERIFY.value else 0.0,
            "last_tool": 1.0
            if state.last_action in {Action.CALL_TOOL.value, Action.CALL_BIGMODEL.value}
            else 0.0,
            "block_learning": 1.0
            if bool((guidance or {}).get("block_learning"))
            else 0.0,
        }

    def heuristic_priors(
        self, context: PolicyContext, guidance: dict[str, object] | None = None
    ) -> dict[str, float]:
        state = context.state
        memory = context.memory
        candidate_slot = state.hidden.get("slot_type", "")

        priors = {action.value: -0.5 for action in Action}

        if memory.latest("GOAL") is None:
            priors[Action.WRITE_MEM.value] = 1.4
            return priors

        if (
            candidate_slot
            and memory.latest(candidate_slot) is None
            and candidate_slot in {"PLAN", "DRAFT"}
        ):
            priors[Action.WRITE_MEM.value] = 1.2

        if state.uncertainty > 0.7 and memory.latest("RESULT") is None:
            priors[Action.CALL_TOOL.value] = 0.55 + float(
                self.controls.get("call_tool_bonus", 0.0)
            )
            priors[Action.CALL_BIGMODEL.value] = 0.45

        if is_conflict_task(context.user_input):
            # Conflict tasks need evidence first — boost CALL_TOOL over WRITE_MEM
            # unless we already have external results
            if memory.latest("RESULT") is None:
                priors[Action.CALL_TOOL.value] = max(
                    priors[Action.CALL_TOOL.value],
                    1.3 + float(self.controls.get("call_tool_bonus", 0.0)),
                )
            else:
                priors[Action.CALL_TOOL.value] = max(
                    priors[Action.CALL_TOOL.value],
                    1.05 + float(self.controls.get("call_tool_bonus", 0.0)),
                )
            priors[Action.CALL_BIGMODEL.value] = min(
                priors[Action.CALL_BIGMODEL.value], 0.1
            )
            priors[Action.WRITE_MEM.value] = max(priors[Action.WRITE_MEM.value], 0.95)

        preferred_tool = str((guidance or {}).get("preferred_tool", ""))
        if preferred_tool:
            priors[Action.CALL_TOOL.value] = max(priors[Action.CALL_TOOL.value], 1.0)

        if state.hidden.get("glance_suggestion") == "prefer_tool":
            priors[Action.CALL_TOOL.value] = max(priors[Action.CALL_TOOL.value], 1.1)

        if state.hidden.get("glance_suggestion") == "promote_draft":
            priors[Action.WRITE_MEM.value] = max(priors[Action.WRITE_MEM.value], 1.15)

        if state.hidden.get("glance_suggestion") == "delay_answer":
            priors[Action.ANSWER.value] = min(
                priors[Action.ANSWER.value],
                -0.2 - float(self.controls.get("answer_penalty", 0.0)),
            )
            priors[Action.VERIFY.value] = max(
                priors[Action.VERIFY.value],
                0.9 + float(self.controls.get("verify_bonus", 0.0)),
            )

        if state.hidden.get("glance_suggestion") == "mark_conflict":
            priors[Action.ROLLBACK.value] = max(priors[Action.ROLLBACK.value], 1.2)

        draft = memory.latest("DRAFT")
        if draft is not None and (
            state.answer_ready >= 0.8 or state.uncertainty <= 0.3
        ):
            priors[Action.ANSWER.value] = 1.35

        if memory.latest("CONFLICT") is not None and state.step_idx < 6:
            priors[Action.ROLLBACK.value] = 1.1

        if draft and memory.latest("CONFLICT") is None:
            priors[Action.VERIFY.value] = max(
                priors[Action.VERIFY.value],
                0.95 + float(self.controls.get("verify_bonus", 0.0)),
            )

        priors[Action.THINK.value] = max(
            priors[Action.THINK.value],
            0.1 - float(self.controls.get("think_penalty", 0.0)),
        )

        # Semantic intent boost: use the semantic encoder to nudge priors
        # when keyword signals are absent but intent is still detectable
        intent_scores = self.semantic.intent_scores(context.user_input)
        top_intent = max(
            ((k, v) for k, v in intent_scores.items() if k != "ambiguous"),
            key=lambda x: x[1],
            default=("search", 0.0),
        )
        if top_intent[1] > 0.3:
            # Map semantic intent key to IntentCategory for tool resolution
            _intent_to_category = {
                "calculator": IntentCategory.CALC,
                "code_executor": IntentCategory.CODE,
                "search": IntentCategory.SEARCH,
                "bigmodel_proxy": IntentCategory.CONSULT,
            }
            top_category = _intent_to_category.get(top_intent[0])
            if top_category == IntentCategory.CALC:
                priors[Action.CALL_TOOL.value] = max(
                    priors[Action.CALL_TOOL.value], 0.65
                )
            elif top_category == IntentCategory.CODE:
                priors[Action.CALL_TOOL.value] = max(
                    priors[Action.CALL_TOOL.value], 0.55
                )
            elif top_category == IntentCategory.SEARCH:
                priors[Action.CALL_TOOL.value] = max(
                    priors[Action.CALL_TOOL.value], 0.5
                )
            elif top_category == IntentCategory.CONSULT:
                priors[Action.CALL_BIGMODEL.value] = max(
                    priors[Action.CALL_BIGMODEL.value], 0.55
                )

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
        preferred_tool = str(
            (guidance or {}).get("preferred_tool", "")
        ) or self.concepts.preferred_tool(user_input)

        # --- Low-intent gate: reject queries with no actionable intent ---
        # "嗯", "帮我看看", "太慢了", "不是那个" → no tool, ask user to clarify
        if has_low_intent_signal(user_input):
            return ActionDecision(
                action=Action.ANSWER,
                score=0.9,
                reason="Low/no-intent query — no tool can meaningfully handle this. Prompting user to clarify.",
                tool_call=None,
                feature_snapshot=features,
            )

        # Anti-loop: if THINK was chosen but we've been thinking for too long,
        # force a tool route based on semantic intent. Prevents infinite THINK
        # loops when signals are too weak to trigger CALL_TOOL directly.
        if action == Action.THINK and state.step_idx >= 3:
            # Hard-rule overrides: travel/lifestyle signals force search even
            # when semantic scores are zero (no embedding model available)
            if has_travel_signal(user_input):
                action = Action.CALL_TOOL
                # tool_name="search" will be set in the CALL_TOOL block (Override 0a)
            else:
                intent_scores = self.semantic.intent_scores(user_input)
                # Map semantic intent keys to IntentCategory for resolution
                _tool_intent_keys = [
                    "search",
                    "calculator",
                    "code_executor",
                    "bigmodel_proxy",
                ]
                _key_to_category = {
                    "calculator": IntentCategory.CALC,
                    "code_executor": IntentCategory.CODE,
                    "search": IntentCategory.SEARCH,
                    "bigmodel_proxy": IntentCategory.CONSULT,
                }
                best_intent = max(
                    _tool_intent_keys, key=lambda t: intent_scores.get(t, 0.0)
                )
                best_score = intent_scores.get(best_intent, 0.0)
                if best_score > 0.0:
                    best_category = _key_to_category.get(
                        best_intent, IntentCategory.SEARCH
                    )
                    action = (
                        Action.CALL_TOOL
                        if best_category not in (IntentCategory.CONSULT,)
                        else Action.CALL_BIGMODEL
                    )
                    # We'll set the tool_call below in the CALL_TOOL / CALL_BIGMODEL block

        # Multi-intent override: if the query contains multiple sub-intents,
        # route directly to multi_intent pseudo-tool.  This bypasses the entire
        # single-tool override chain because the runner handles sequential
        # execution of each sub-intent.
        from carm.signals import has_multi_intent_signal, split_multi_intent

        if has_multi_intent_signal(user_input):
            intents = split_multi_intent(user_input)
            if len(intents) >= 2:
                state.hidden["_multi_intent_splits"] = [
                    {"text": i.text, "signal": i.primary_signal} for i in intents
                ]
                return ActionDecision(
                    action=Action.CALL_TOOL,
                    score=0.95,
                    reason=(
                        f"Multi-intent detected ({len(intents)} sub-queries): "
                        + " → ".join(f"{i.text} ({i.primary_signal})" for i in intents)
                    ),
                    tool_call=ToolCall(
                        tool_name="multi_intent",
                        query=user_input,
                        arguments={
                            "intents": [
                                {"text": i.text, "signal": i.primary_signal}
                                for i in intents
                            ]
                        },
                        reason=f"Executing {len(intents)} sub-intents in sequence.",
                    ),
                    feature_snapshot=features,
                )

        # Multi-step override: single intent requiring sequential tool execution
        # "对比分析A和B的差异并给出建议" → search → compare → bigmodel_proxy
        from carm.signals import has_multi_step_signal

        if has_multi_step_signal(user_input):
            state.hidden["_multi_step_plan"] = "search → compare → bigmodel_proxy"
            return ActionDecision(
                action=Action.CALL_TOOL,
                score=0.95,
                reason="Multi-step reasoning chain detected — requires sequential tool execution.",
                tool_call=ToolCall(
                    tool_name="multi_step",
                    query=user_input,
                    arguments={"plan": "search → compare → bigmodel_proxy"},
                    reason="Multi-step: gather evidence, compare, then synthesize.",
                ),
                feature_snapshot=features,
            )

        decision = ActionDecision(
            action=action,
            score=score,
            reason=f"Hybrid policy selected {action.value.lower()} with score {score:.3f}.",
            feature_snapshot=features,
        )

        if action == Action.WRITE_MEM:
            target = (
                "GOAL" if memory.latest("GOAL") is None else candidate_slot or "DRAFT"
            )
            decision.target_slot = target
            decision.reason = f"Persist {target.lower()} into working memory."
            return decision

        if action == Action.CALL_TOOL:
            # --- Tool routing: semantic-first with hard-rule overrides ---
            intent_scores = self.semantic.intent_scores(user_input)
            # Pick the tool with the highest semantic score among the 4 tool intents
            tool_intents = ["search", "calculator", "code_executor", "bigmodel_proxy"]
            semantic_best = max(tool_intents, key=lambda t: intent_scores.get(t, 0.0))
            semantic_best_score = intent_scores.get(semantic_best, 0.0)

            # Hard-rule overrides (highest priority)
            hard_conflict = is_conflict_task(user_input)
            hard_arithmetic = bool(re.search(r"\d+\s*[\*\/+\-]\s*\d+", user_input))
            hard_code_action = has_code_signal(user_input)
            hard_explain = has_explain_signal(user_input)
            hard_search_action = has_search_action_signal(user_input)
            hard_writing = has_writing_signal(user_input)
            _synthesis_verbs = ("总结", "报告", "建议", "归纳", "提炼", "综合")
            hard_synthesis = any(v in user_input for v in _synthesis_verbs)
            hard_formal = has_formal_signal(user_input) and hard_synthesis
            _strong_code_verbs = (
                "运行",
                "写",
                "实现",
                "编写",
                "代码",
                "脚本",
                "执行",
                "跑",
            )
            has_strong_code_verb = any(v in user_input for v in _strong_code_verbs)

            chosen_intent: IntentCategory | None = None  # Set by hard-rule overrides
            chosen_tool = (
                semantic_best  # default: semantic winner (still a string for compat)
            )
            chosen_reason = (
                f"Semantic intent: {semantic_best} ({semantic_best_score:.2f})"
            )
            hard_rule_hit = False  # Track whether a hard rule already chose a tool

            # Override -1: Multi-intent detection → multi_intent router
            # "帮我查一下北京天气，顺便算一下3加5" → split into [search, calculator]
            # This must be first because it overrides ALL single-tool rules.
            from carm.signals import has_multi_intent_signal, split_multi_intent

            if has_multi_intent_signal(user_input):
                intents = split_multi_intent(user_input)
                if len(intents) >= 2:
                    chosen_intent = IntentCategory.MULTI_INTENT
                    chosen_reason = (
                        f"Multi-intent detected ({len(intents)} sub-queries): "
                        + " → ".join(f"{i.text} ({i.primary_signal})" for i in intents)
                    )
                    # Store split intents in state for runner to pick up
                    state.hidden["_multi_intent_splits"] = [
                        {"text": i.text, "signal": i.primary_signal} for i in intents
                    ]
                    hard_rule_hit = True

            # Override 0: Explicit search action → search
            # When both search action and code action are present:
            #   - "搜索一下Python教程" → search wins (explicit "搜索" action verb)
            #   - "写个爬虫抓微博热搜" → code wins ("热搜" is content target, not search action)
            if hard_search_action and not (
                hard_code_action and not has_search_action_signal(user_input)
            ):
                chosen_intent = IntentCategory.SEARCH
                chosen_reason = "Explicit search action detected (搜索/搜一下/查一下)."
                hard_rule_hit = True
            # Override 0a: Travel/lifestyle intent → search
            elif has_travel_signal(user_input) and not hard_writing:
                chosen_intent = IntentCategory.SEARCH
                chosen_reason = "Travel/lifestyle service intent detected."
                hard_rule_hit = True
            # Override 0b: Writing/synthesis intent → consult (bigmodel)
            elif hard_writing or (hard_synthesis and not hard_code_action):
                chosen_intent = IntentCategory.CONSULT
                chosen_reason = (
                    "Writing/synthesis intent detected — routing to consult tool."
                )
                hard_rule_hit = True
            # Override 0c: Translate/polish intent → consult (bigmodel)
            elif has_translate_signal(user_input) or has_polish_signal(user_input):
                chosen_intent = IntentCategory.CONSULT
                chosen_reason = (
                    "Translate/polish intent detected — routing to consult tool."
                )
                hard_rule_hit = True
            # Override 0d: Consultative intent → search or consult
            elif (
                has_consult_signal(user_input)
                and not has_calc_signal(user_input)
                and not has_strong_code_verb
            ):
                if has_deep_analysis_signal(user_input):
                    chosen_intent = IntentCategory.CONSULT
                    chosen_reason = "Consultative + deep analysis intent — routing to consult tool for synthesis."
                else:
                    chosen_intent = IntentCategory.SEARCH
                    chosen_reason = "Consultative/advisory intent without code action — knowledge search."
                hard_rule_hit = True
            # Override 0e: Debug consultative intent → search
            elif has_debug_consult_signal(user_input) and not any(
                v in user_input
                for v in ("运行", "写", "实现", "编写", "执行", "跑一下")
            ):
                chosen_intent = IntentCategory.SEARCH
                chosen_reason = (
                    "Debug consultative intent — seeking help/solutions, not execution."
                )
                hard_rule_hit = True
            # Override 0f: Deep reasoning → consult (bigmodel)
            elif has_deep_reason_signal(user_input):
                chosen_intent = IntentCategory.CONSULT
                chosen_reason = "Deep reasoning/comparative analysis detected — routing to consult tool."
                hard_rule_hit = True
            # Override 1: Conflict tasks → search
            elif hard_conflict:
                chosen_intent = IntentCategory.SEARCH
                chosen_reason = (
                    "Conflict-style questions should gather explicit evidence."
                )
                hard_rule_hit = True
            # Override 2: Explicit arithmetic → calc
            elif hard_arithmetic and not hard_code_action:
                chosen_intent = IntentCategory.CALC
                chosen_reason = (
                    "Hard rule: explicit arithmetic expression requires calculator."
                )
                hard_rule_hit = True
            # Override 2b: Calc signal → calc
            elif (
                has_calc_signal(user_input)
                and not has_code_signal(user_input)
                and not hard_explain
            ):
                chosen_intent = IntentCategory.CALC
                chosen_reason = (
                    "Hard rule: calc intent signal detected (no code intent)."
                )
                hard_rule_hit = True
            # Override 2c: Code + calc → code
            elif (
                has_calc_signal(user_input)
                and has_code_signal(user_input)
                and not hard_explain
            ):
                chosen_intent = IntentCategory.CODE
                chosen_reason = "Hard rule: code+calc co-occurrence, code action wins."
                hard_rule_hit = True
            # Override 3: Clear code action → code
            elif (
                hard_code_action
                and not hard_explain
                and not (has_compare_signal(user_input) and not has_strong_code_verb)
            ):
                chosen_intent = IntentCategory.CODE
                chosen_reason = "Hard rule: code action verb detected."
                hard_rule_hit = True
            # Override 4: Explain intent → search
            elif hard_explain:
                chosen_intent = IntentCategory.SEARCH
                chosen_reason = (
                    "Explain intent detected — user wants knowledge, not execution."
                )
                hard_rule_hit = True
            # Override 5: Formal/synthesis → consult
            elif hard_formal and not hard_conflict:
                chosen_intent = IntentCategory.CONSULT
                chosen_reason = (
                    "Formal/synthesis intent detected — routing to consult tool."
                )
                hard_rule_hit = True
            # Override 4b: Compare intent → search
            elif (
                has_compare_signal(user_input)
                and not hard_arithmetic
                and not has_strong_code_verb
            ):
                chosen_intent = IntentCategory.SEARCH
                chosen_reason = (
                    "Compare intent without explicit code action — knowledge search."
                )
                hard_rule_hit = True

            # L4 Fallback: ultra-low confidence → consult (bigmodel)
            if (
                semantic_best_score < 0.15
                and not hard_rule_hit
                and not hard_conflict
                and not hard_arithmetic
                and not hard_code_action
            ):
                chosen_intent = IntentCategory.CONSULT
                chosen_reason = "L4 catch-all: ultra-low confidence, routing to consult tool for general reasoning."
            elif (
                semantic_best_score < 0.2
                and not hard_rule_hit
                and not hard_conflict
                and not hard_arithmetic
                and not hard_code_action
            ):
                chosen_intent = IntentCategory.SEARCH
                chosen_reason = "Low-confidence semantic routing, defaulting to search."

            # Resolve IntentCategory → actual tool name
            if chosen_intent is not None:
                chosen_tool = self._resolve_tool_name(chosen_intent)
            # else: chosen_tool stays as semantic_best (backward compat)

            # Code executor needs a safe default query
            tool_query = user_input
            if chosen_tool == "code_executor":
                tool_query = (
                    user_input  # CodeExecutorTool extracts code from natural language
                )

            # Build arguments — inject CARM signal analysis for bigmodel_proxy
            tool_args = {}
            if chosen_tool == "search":
                tool_args = {"top_k": 3}
            elif chosen_tool == "bigmodel_proxy":
                signal_summary = self._build_signal_summary(user_input)
                if signal_summary:
                    tool_args = {"carm_signals": signal_summary}
            elif chosen_tool == "multi_intent":
                tool_args = {
                    "split_intents": state.hidden.get("_multi_intent_splits", [])
                }
            elif chosen_tool == "multi_step":
                tool_args = {"plan": "search → compare → bigmodel_proxy"}

            decision.tool_call = ToolCall(
                tool_name=chosen_tool,
                query=tool_query,
                arguments=tool_args,
                reason=chosen_reason,
            )
            decision.reason = f"Use {chosen_tool}: {chosen_reason}"
            return decision

        if action == Action.CALL_BIGMODEL:
            # Path-C: inject CARM signal analysis into LLM prompt
            signal_summary = self._build_signal_summary(user_input)
            consult_tool = self._resolve_tool_name(IntentCategory.CONSULT)
            decision.tool_call = ToolCall(
                tool_name=consult_tool,
                query=user_input,
                arguments={"carm_signals": signal_summary} if signal_summary else {},
                reason="Need stronger external reasoning support.",
            )
            decision.reason = "Escalate to larger external model."
            if signal_summary:
                decision.reason += f" Signals: {signal_summary}"
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
                self.action_weights[action_name][feature] = (
                    current + self.learning_rate * step.reward * value
                )
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

    def _enforce_constraints(
        self, decision: ActionDecision, context: PolicyContext
    ) -> ActionDecision:
        memory = context.memory
        state = context.state
        require_conflict_verify = bool(
            self.controls.get("require_conflict_verify_before_answer", 0)
        )
        prefer_search_for_comparison = bool(
            self.controls.get("prefer_search_for_comparison_evidence", 0)
        )

        if (
            state.last_action == Action.VERIFY.value
            and memory.latest("DRAFT") is not None
            and memory.latest("CONFLICT") is None
            and (
                state.answer_ready >= 0.8
                or state.uncertainty <= 0.3
                or state.hidden.get("verified") == "1"
            )
        ):
            return self._build_decision(
                Action.ANSWER,
                decision.score,
                decision.feature_snapshot,
                context,
            )

        if (
            require_conflict_verify
            and decision.action == Action.ANSWER
            and is_conflict_task(context.user_input)
            and memory.latest("DRAFT") is not None
            and state.hidden.get("verified") != "1"
        ):
            return self._build_decision(
                Action.VERIFY,
                decision.score,
                decision.feature_snapshot,
                context,
            )

        if (
            prefer_search_for_comparison
            and has_comparison_evidence_signal(context.user_input)
            and not has_formal_signal(context.user_input)
            and (
                decision.action == Action.CALL_BIGMODEL
                or (
                    decision.action == Action.CALL_TOOL
                    and decision.tool_call is not None
                    and decision.tool_call.tool_name
                    == self._resolve_tool_name(IntentCategory.CONSULT)
                )
            )
        ):
            return ActionDecision(
                action=Action.CALL_TOOL,
                score=decision.score,
                reason="Candidate gate: use search for comparison/evidence task before synthesis.",
                tool_call=ToolCall(
                    tool_name=self._resolve_tool_name(IntentCategory.SEARCH),
                    query=context.user_input,
                    arguments={"top_k": 3},
                    reason="Comparison/evidence tasks need source grounding before generation.",
                ),
                feature_snapshot=dict(decision.feature_snapshot),
            )

        if (
            is_conflict_task(context.user_input)
            and memory.latest("HYP") is not None
            and memory.latest("DRAFT") is None
            and memory.latest("RESULT") is not None
        ):
            return ActionDecision(
                action=Action.WRITE_MEM,
                score=decision.score,
                reason="Convert conflict-aware hypothesis into a cautious draft (evidence gathered).",
                target_slot="DRAFT",
                feature_snapshot=dict(decision.feature_snapshot),
            )

        # Conflict tasks without evidence must search first
        if (
            is_conflict_task(context.user_input)
            and memory.latest("RESULT") is None
            and decision.action != Action.CALL_TOOL
        ):
            return ActionDecision(
                action=Action.CALL_TOOL,
                score=decision.score,
                reason="Conflict task needs evidence before synthesis — forcing search.",
                tool_call=ToolCall(
                    tool_name=self._resolve_tool_name(IntentCategory.SEARCH),
                    query=context.user_input,
                    arguments={"top_k": 3},
                    reason="Conflict tasks must gather evidence before proceeding.",
                ),
                feature_snapshot=dict(decision.feature_snapshot),
            )

        if memory.latest("RESULT") is not None and memory.latest("DRAFT") is None:
            if is_conflict_task(context.user_input):
                return ActionDecision(
                    action=Action.WRITE_MEM,
                    score=decision.score,
                    reason="Materialize a conflict-aware hypothesis before drafting.",
                    target_slot="HYP",
                    feature_snapshot=dict(decision.feature_snapshot),
                )
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

        if (
            decision.action == Action.WRITE_MEM
            and decision.target_slot == "GOAL"
            and memory.latest("GOAL") is not None
        ):
            fallback = Action.THINK if memory.latest("DRAFT") is None else Action.VERIFY
            return self._build_decision(
                fallback,
                decision.score,
                decision.feature_snapshot,
                context,
            )

        return decision

    # ── Path-C: Signal summary for LLM escalation ──────────────────────

    def _build_signal_summary(self, user_input: str) -> str:
        """Build a compact signal analysis summary for LLM consumption.

        When CARM escalates to bigmodel_proxy, this summary tells the LLM
        what signals CARM detected, so the LLM can use them as priors.
        Format: "signal1, signal2, signal3" — kept short to avoid token waste.
        """
        signals = []
        if has_calc_signal(user_input):
            signals.append("calc")
        if has_code_signal(user_input):
            signals.append("code")
        if has_search_action_signal(user_input):
            signals.append("search")
        if has_writing_signal(user_input):
            signals.append("writing")
        if has_translate_signal(user_input):
            signals.append("translate")
        if has_consult_signal(user_input):
            signals.append("consult")
        if has_travel_signal(user_input):
            signals.append("travel")
        if has_compare_signal(user_input):
            signals.append("compare")
        if has_explain_signal(user_input):
            signals.append("explain")
        if has_formal_signal(user_input):
            signals.append("formal")
        if has_deep_reason_signal(user_input):
            signals.append("deep_reason")
        if has_deep_analysis_signal(user_input):
            signals.append("deep_analysis")
        if has_anaphora_signal(user_input):
            signals.append("anaphora")
        if has_multi_intent_signal(user_input):
            signals.append("multi_intent")
        if has_multi_step_signal(user_input):
            signals.append("multi_step")
        if has_debug_consult_signal(user_input):
            signals.append("debug_consult")
        if is_conflict_task(user_input):
            signals.append("conflict")

        return ",".join(signals)
