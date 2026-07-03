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
    ) -> None:
        self.state_path = Path(state_path)
        if concept_state_path is None:
            concept_state_path = self.state_path.with_name("concept_state.json")
        self.concepts = AdaptiveConceptModel(concept_state_path)
        self.semantic = SemanticEncoder()
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
            "concept_tool_search": 1.0 if preferred_tool == "search" else 0.0,
            "concept_tool_calc": 1.0 if preferred_tool == "calculator" else 0.0,
            "concept_tool_code": 1.0 if preferred_tool == "code_executor" else 0.0,
            "concept_tool_bigmodel": 1.0 if preferred_tool == "bigmodel_proxy" else 0.0,
            "user_tool_search": 1.0 if guided_tool == "search" else 0.0,
            "user_tool_calc": 1.0 if guided_tool == "calculator" else 0.0,
            "user_tool_code": 1.0 if guided_tool == "code_executor" else 0.0,
            "user_tool_bigmodel": 1.0 if guided_tool == "bigmodel_proxy" else 0.0,
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
            if top_intent[0] == "calculator":
                priors[Action.CALL_TOOL.value] = max(
                    priors[Action.CALL_TOOL.value], 0.65
                )
            elif top_intent[0] == "code_executor":
                priors[Action.CALL_TOOL.value] = max(
                    priors[Action.CALL_TOOL.value], 0.55
                )
            elif top_intent[0] == "search":
                priors[Action.CALL_TOOL.value] = max(
                    priors[Action.CALL_TOOL.value], 0.5
                )
            elif top_intent[0] == "bigmodel_proxy":
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

        # Anti-loop: if THINK was chosen but we've been thinking for too long,
        # force a tool route based on semantic intent. Prevents infinite THINK
        # loops when signals are too weak to trigger CALL_TOOL directly.
        if action == Action.THINK and state.step_idx >= 3:
            intent_scores = self.semantic.intent_scores(user_input)
            tool_intents = ["search", "calculator", "code_executor", "bigmodel_proxy"]
            best_intent = max(tool_intents, key=lambda t: intent_scores.get(t, 0.0))
            best_score = intent_scores.get(best_intent, 0.0)
            if best_score > 0.0:
                action = (
                    Action.CALL_TOOL
                    if best_intent != "bigmodel_proxy"
                    else Action.CALL_BIGMODEL
                )
                # We'll set the tool_call below in the CALL_TOOL / CALL_BIGMODEL block

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

            chosen_tool = semantic_best  # default: semantic winner
            chosen_reason = (
                f"Semantic intent: {semantic_best} ({semantic_best_score:.2f})"
            )
            hard_rule_hit = False  # Track whether a hard rule already chose a tool

            # Override 0: Explicit search action → search
            # "搜索一下Python教程" is clearly a search request, not code.
            if hard_search_action:
                chosen_tool = "search"
                chosen_reason = "Explicit search action detected (搜索/搜一下/查一下)."
                hard_rule_hit = True
            # Override 0b: Writing/synthesis intent → bigmodel_proxy
            # "写一篇议论文" / "总结一下趋势" / "帮我归纳"
            elif hard_writing or (hard_synthesis and not hard_code_action):
                chosen_tool = "bigmodel_proxy"
                chosen_reason = (
                    "Writing/synthesis intent detected — routing to big model."
                )
                hard_rule_hit = True
            # Override 0c: Translate/polish intent → bigmodel_proxy
            # "翻译一下这段英文" / "帮我润色一下这段文字"
            elif has_translate_signal(user_input) or has_polish_signal(user_input):
                chosen_tool = "bigmodel_proxy"
                chosen_reason = (
                    "Translate/polish intent detected — routing to big model."
                )
                hard_rule_hit = True
            # Override 0d: Consultative intent without strong code verb → search
            # "如何选择排序算法" / "优化排序性能" / "代码性能瓶颈分析"
            # These are advisory questions, not execution requests.
            # But "写一个排序" still goes to code_executor (strong code verb).
            elif has_consult_signal(user_input) and not any(
                v in user_input
                for v in ("运行", "写", "实现", "编写", "执行", "跑一下")
            ):
                chosen_tool = "search"
                chosen_reason = "Consultative/advisory intent without code action — knowledge search."
                hard_rule_hit = True
            # Override 1: Conflict tasks always search first
            elif hard_conflict:
                chosen_tool = "search"
                chosen_reason = (
                    "Conflict-style questions should gather explicit evidence."
                )
                hard_rule_hit = True
            # Override 2: Explicit arithmetic expression → calculator
            # BUT: if code intent is also present (e.g. "python代码 print(1+1)"),
            # code execution takes priority — user wants to run code.
            elif hard_arithmetic and not hard_code_action:
                chosen_tool = "calculator"
                chosen_reason = (
                    "Hard rule: explicit arithmetic expression requires calculator."
                )
                hard_rule_hit = True
            # Override 2b: Calc signal detected (without code intent) → calculator
            # "1万亿除以14亿" has calc tokens but no explicit operator pattern
            # When code_signal is also present (e.g. "写一个阶乘函数"),
            # code execution takes priority — user wants to run code, not just compute.
            elif (
                has_calc_signal(user_input)
                and not has_code_signal(user_input)
                and not hard_explain
            ):
                chosen_tool = "calculator"
                chosen_reason = (
                    "Hard rule: calc intent signal detected (no code intent)."
                )
                hard_rule_hit = True
            # Override 2c: Code signal + calc signal co-occurrence → code_executor
            # "写一个阶乘函数" has both calc and code signals, but the action
            # verb ("写") makes it a code execution request, not a calculation.
            elif (
                has_calc_signal(user_input)
                and has_code_signal(user_input)
                and not hard_explain
            ):
                chosen_tool = "code_executor"
                chosen_reason = "Hard rule: code+calc co-occurrence, code action wins."
                hard_rule_hit = True
            # Override 3: Clear code action without calc → code_executor
            # BUT: explain intent overrides code — "解释递归" → search, not code
            # AND: compare intent overrides code when no strong code verb —
            # "比较冒泡排序和快速排序的效率" is analysis, not execution.
            elif (
                hard_code_action
                and not hard_explain
                and not (has_compare_signal(user_input) and not has_strong_code_verb)
            ):
                chosen_tool = "code_executor"
                chosen_reason = "Hard rule: code action verb detected."
                hard_rule_hit = True
            # Override 4: Explain intent overrides everything → search
            # "解释冒泡排序的原理" has explain signal; even though
            # semantic encoder gives code_executor high score (algorithm name),
            # the user wants knowledge, not code execution.
            elif hard_explain:
                chosen_tool = "search"
                chosen_reason = (
                    "Explain intent detected — user wants knowledge, not execution."
                )
                hard_rule_hit = True
            # Override 5: Formal/synthesis intent → bigmodel_proxy
            # "管理层报告/正式总结" needs LLM synthesis, not search.
            # This MUST come before compare override — "正式报告：总结..."
            # is a synthesis request, not just a comparison.
            elif hard_formal and not hard_conflict:
                chosen_tool = "bigmodel_proxy"
                chosen_reason = (
                    "Formal/synthesis intent detected — routing to big model."
                )
                hard_rule_hit = True
            # Override 4b: Compare intent without explicit code action → search
            # "比较冒泡排序和快速排序的效率" has compare signal but user
            # wants analysis, not to run code.  Only route to code_executor
            # when there is an explicit code action verb ("运行/写/实现/代码").
            elif (
                has_compare_signal(user_input)
                and not hard_arithmetic
                and not has_strong_code_verb
            ):
                chosen_tool = "search"
                chosen_reason = (
                    "Compare intent without explicit code action — knowledge search."
                )
                hard_rule_hit = True

            # If semantic score is very low (< 0.2) and no hard rule hit,
            # default to search
            if (
                semantic_best_score < 0.2
                and not hard_rule_hit
                and not hard_conflict
                and not hard_arithmetic
                and not hard_code_action
            ):
                chosen_tool = "search"
                chosen_reason = "Low-confidence semantic routing, defaulting to search."

            # Code executor needs a safe default query
            tool_query = user_input
            if chosen_tool == "code_executor":
                tool_query = (
                    user_input  # CodeExecutorTool extracts code from natural language
                )

            decision.tool_call = ToolCall(
                tool_name=chosen_tool,
                query=tool_query,
                arguments={"top_k": 3} if chosen_tool == "search" else {},
                reason=chosen_reason,
            )
            decision.reason = f"Use {chosen_tool}: {chosen_reason}"
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
                    and decision.tool_call.tool_name == "bigmodel_proxy"
                )
            )
        ):
            return ActionDecision(
                action=Action.CALL_TOOL,
                score=decision.score,
                reason="Candidate gate: use search for comparison/evidence task before synthesis.",
                tool_call=ToolCall(
                    tool_name="search",
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
                    tool_name="search",
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
