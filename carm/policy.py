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
    has_evidence_judgment_signal,
    has_explain_signal,
    has_writing_signal,
    has_search_signal,
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
    SEARCH_TOKENS,
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

        # --- Low-intent gate: default vague queries to search ---
        # "嗯", "帮我看看", "太慢了", "不是那个" are too vague for a specific
        # tool, but defaulting to search is more helpful than rejecting.
        _is_low_intent = has_low_intent_signal(user_input)

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
        # BUT: when the query starts with a search action ("查一下...然后..."),
        #   the user is describing a workflow that starts with search — route
        #   to search, not multi-intent.
        from carm.signals import has_multi_intent_signal, split_multi_intent, has_search_action_signal

        # Suppress multi-intent when strong synthesis verbs or "先X再Y" patterns
        # are present — these are sequential workflows, not independent multi-intents.
        _synthesis_override = any(v in user_input for v in ("总结", "分析", "报告", "归纳", "提炼", "综合", "结论", "summarize"))
        _sequential_pattern = "先" in user_input and "再" in user_input
        _code_write_override = any(v in user_input for v in ("写", "实现", "编写", "开发")) and any(
            t in user_input.lower() for t in ("python", "docker", "java", "代码", "脚本", "配置文件", "compose", "go", "rust")
        )
        if has_multi_intent_signal(user_input) and not (
            has_search_action_signal(user_input) and "然后" in user_input
        ) and not _synthesis_override and not _sequential_pattern and not _code_write_override:
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

        if has_multi_step_signal(user_input) and not _synthesis_override and not _sequential_pattern:
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
            hard_writing = has_writing_signal(user_input)
            _synthesis_verbs = ("总结", "报告", "建议", "归纳", "提炼", "综合", "摘要", "结论", "小结", "分析", "summarize", "translate", "write a report", "write a comprehensive", "analyze")
            hard_synthesis = any(v in user_input for v in _synthesis_verbs)
            _is_bare_analyze = user_input.strip() == "分析"
            if _is_bare_analyze:
                hard_synthesis = False
            # "分析" alone is ambiguous — it could be search ("分析一下代码") or
            # synthesis ("分析Kafka高吞吐的原因"). Only treat "分析" as synthesis
            # when it appears at the START of the query (analysis report intent),
            # not when preceded by "帮我"/"请" (which makes it a search request).
            if "分析" in user_input and not user_input.strip().startswith("分析") and not _is_bare_analyze:
                _has_question = any(q in user_input for q in ("为什么", "咋", "吗", "？", "?", "什么原因", "什么"))
                _has_compare = has_compare_signal(user_input)
                _has_specific_object = any(o in user_input for o in ("这段", "这个", "那份", "那个", "这份", "这行", "那个日志"))
                _has_explicit_calc = hard_arithmetic or any(v in user_input for v in ("算下", "算一下", "算算", "计算一下", "估算", "费用"))
                _has_code_write = "写" in user_input and has_code_signal(user_input)
                if (_has_question or _has_compare or _has_specific_object or _has_explicit_calc or _is_bare_analyze or _has_code_write) and not hard_writing:
                    hard_synthesis = False
            _strong_code_verbs = (
                "运行",
                "写",
                "实现",
                "编写",
                "执行",
                "跑",
                "画",
                # English code action verbs
                "write",
                "implement",
                "run ",
                "execute",
                "build",
                "create",
            )
            has_strong_code_verb = any(v in user_input for v in _strong_code_verbs)
            # Extended code detection: "用XX脚本/写XX" patterns where "用" is a
            # code verb only when combined with a code token.
            if not has_strong_code_verb and has_code_signal(user_input):
                _code_verb_patterns = (
                    "用Shell", "用Go", "用Python", "用Java", "用Rust",
                    "用C语言", "用Scala", "用Lua", "用Bash", "用PowerShell",
                    "用Kotlin", "用JavaScript", "用TypeScript", "用Vue",
                    "用React", "用Angular", "用Docker", "用K8s", "用FastAPI",
                    "用Dockerfile", "用Terraform", "用PySpark",
                    "弄个",  # colloquial for "make/create" — code when combined with code token
                )
                has_strong_code_verb = any(p in user_input for p in _code_verb_patterns)
            # Code action requires BOTH a code token AND a code action verb,
            # OR a code action verb without writing/synthesis signal.
            # This prevents "PostgreSQL vs MySQL" (tech term in search query)
            # from triggering code routing.
            hard_code_action = (
                has_code_signal(user_input) and has_strong_code_verb and not hard_writing and not hard_synthesis
            ) or (
                has_strong_code_verb and not hard_writing and not hard_synthesis
            )
            hard_explain = has_explain_signal(user_input)
            hard_search_action = has_search_action_signal(user_input)
            hard_formal = has_formal_signal(user_input) and hard_synthesis

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
            # BUT: when the query starts with a search action ("查一下...然后..."),
            #   the user is describing a workflow that starts with search — route
            #   to search, not multi-intent.
            from carm.signals import has_multi_intent_signal, split_multi_intent

            if has_multi_intent_signal(user_input) and not (
                hard_search_action and "然后" in user_input
            ) and not hard_code_action:
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
            # BUT: when user says they've ALREADY searched and wants to conclude,
            #   don't route to search — route to synthesis (consult).
            #   e.g. "我已经做了两轮搜索...信息足够了，现在应该直接给出结论"
            _past_search_markers = ("已经", "做了", "过了", "完了", "查了", "搜了")
            _sufficient_markers = ("足够", "够了", "充分", "找到了", "收集了", "查到")
            _is_past_search_with_synthesis = (
                any(m in user_input for m in _past_search_markers)
                and any(m in user_input for m in _sufficient_markers)
                and hard_synthesis
            )
            _is_sql_code_action = (
                "sql" in user_input.lower()
                and any(v in user_input for v in ("写", "实现", "编写", "run", "execute", "query"))
            )
            _strong_synthesis_verbs = ("总结", "结论", "摘要", "归纳", "提炼", "综合", "报告", "summarize", "write a report", "write a comprehensive", "analyze")
            _has_strong_synthesis = any(v in user_input for v in _strong_synthesis_verbs)
            # "分析" at start or "请分析" without compare → strong synthesis
            # (prevents search signal from overriding analysis report intent)
            # ALSO: when "分析" survived the guard (hard_synthesis still True)
            #   and is NOT at start, it's a synthesis-worthy analysis request
            if user_input.strip().startswith("分析") and not _is_bare_analyze:
                _has_strong_synthesis = True
            elif "请分析" in user_input and not has_compare_signal(user_input) and not has_calc_signal(user_input):
                _has_strong_synthesis = True
            elif "分析" in user_input and hard_synthesis and not user_input.strip().startswith("分析") and not has_calc_signal(user_input):
                _has_strong_synthesis = True
            if hard_search_action and not (
                hard_code_action and not has_search_action_signal(user_input)
            ) and not _is_past_search_with_synthesis and not _is_sql_code_action:
                chosen_intent = IntentCategory.SEARCH
                chosen_reason = "Explicit search action detected (搜索/搜一下/查一下)."
                hard_rule_hit = True
            # Override 0a: Travel/lifestyle intent → search
            elif has_travel_signal(user_input) and not hard_writing:
                chosen_intent = IntentCategory.SEARCH
                chosen_reason = "Travel/lifestyle service intent detected."
                hard_rule_hit = True
            # Override 0a1: Sequential "先X再Y" pattern — route based on first action
            # "先列出关键步骤再给结论比较..." → search first (compare info needed)
            # "先搜资料再总结..." → search first (explicit search action)
            # "先写代码再跑测试" → code first (explicit code action)
            # The first action in the sequence determines primary routing.
            elif "先" in user_input and ("再" in user_input or "然后" in user_input) and not hard_code_action:
                # Split on whichever connector comes first
                _split_pos = min(
                    user_input.find("再") if "再" in user_input else 999,
                    user_input.find("然后") if "然后" in user_input else 999,
                )
                _first_part = user_input[:_split_pos] if _split_pos < 999 else ""
                _has_first_search = (
                    has_search_action_signal(_first_part)
                    or has_compare_signal(_first_part)
                    or has_search_signal(_first_part)
                )
                _has_first_calc = has_calc_signal(_first_part)
                if _has_first_search:
                    chosen_intent = IntentCategory.SEARCH
                    chosen_reason = "Sequential '先X再Y' pattern — first action is search-related, route to search first."
                    hard_rule_hit = True
                elif _has_first_calc:
                    chosen_intent = IntentCategory.CALC
                    chosen_reason = "Sequential '先X再Y' pattern — first action is calculation, route to calculator first."
                    hard_rule_hit = True
            # Override 0b: Writing/synthesis intent → consult (bigmodel)
            # BUT: evidence_judgment overrides synthesis — "这个建议可靠吗" needs
            # search verification, not LLM synthesis.
            # BUT: compare queries with "结论" should search first, not synthesize.
            # EXCEPT: when user has already gathered evidence (past-search markers),
            #   "对比" in text means they're referencing past comparison, not
            #   requesting a new one — synthesis should win.
            # EXCEPT: strong synthesis verbs ("总结"/"结论"/"摘要") override compare
            #   guard — "总结优缺点" and "给出对比结论" are synthesis requests.
            # EXCEPT: when search signal is present but no strong synthesis verb,
            #   route to search — "分析...给优化建议" has search + weak synthesis.
            elif (
                ((hard_writing or hard_synthesis) and not hard_code_action)
                and not has_evidence_judgment_signal(user_input)
                and not (has_compare_signal(user_input) and not _is_past_search_with_synthesis and not _has_strong_synthesis)
                and not ((has_search_action_signal(user_input) or any(t in user_input for t in SEARCH_TOKENS)) and not _has_strong_synthesis and not hard_writing)
            ):
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
            # v6 fix: evidence_judgment 优先于 consult+deep_analysis 合成。
            # "请基于公开资料总结要点，并给出一个能验证的实验" 有 consult+deep_analysis
            # 信号，但真实意图是先检索证据再下结论 → 应走 search，而不是 bigmodel 合成。
            # 与 Override 0b / 2a 的 evidence_judgment 守卫保持一致。
            # v7 fix: has_search_signal 守卫 — 当查询同时有 search 和 consult 信号时，
            # 优先搜索。"帮我看看这个方案有什么问题" 有 search+consult+deep_analysis，
            # 但真实意图是搜索方案问题，不是合成。
            elif (
                has_consult_signal(user_input)
                and not has_calc_signal(user_input)
                and not has_strong_code_verb
                and not has_search_signal(user_input)
            ):
                if has_deep_analysis_signal(
                    user_input
                ) and not has_evidence_judgment_signal(user_input):
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
            # Override 1: Conflict tasks → search (must come before deep_reason
            # and consult overrides — conflict queries need evidence gathering first).
            # BUT: when the user has already gathered evidence and wants to write
            # a formal conclusion, route to consult (bigmodel_proxy) for synthesis.
            elif hard_conflict and not hard_formal and not (
                "写" in user_input
                and any(w in user_input for w in ("正式", "结论", "总结", "报告"))
            ):
                chosen_intent = IntentCategory.SEARCH
                chosen_reason = (
                    "Conflict-style questions should gather explicit evidence."
                )
                hard_rule_hit = True
            # Override 0f: Deep reasoning → consult (bigmodel)
            elif has_deep_reason_signal(user_input):
                chosen_intent = IntentCategory.CONSULT
                chosen_reason = "Deep reasoning/comparative analysis detected — routing to consult tool."
                hard_rule_hit = True
            # Override 2: Explicit arithmetic → calc
            elif (
                hard_arithmetic
                and not hard_code_action
                and not has_evidence_judgment_signal(user_input)
            ):
                chosen_intent = IntentCategory.CALC
                chosen_reason = (
                    "Hard rule: explicit arithmetic expression requires calculator."
                )
                hard_rule_hit = True
            # Override 2a: Evidence judgment → search (v5 fix for learning_focus 004/005/006)
            # When the query asks to verify/judge/assess reliability of information,
            # it needs search — not calculator — even if it contains numbers.
            # e.g. "判断 2024 年公告是否被 2026 文档推翻" has calc signal but needs search.
            # BUT: when formal/writing intent is also present ("写一份正式结论"), the
            # user wants synthesis (bigmodel_proxy), not just evidence search.
            # Also check for "写" + formal/conclusion pattern since hard_formal requires
            # a synthesis verb which may not always be present.
            elif (
                has_evidence_judgment_signal(user_input)
                and not hard_code_action
                and not (
                    hard_formal
                    or hard_writing
                    or (
                        "写" in user_input
                        and any(
                            w in user_input
                            for w in ("正式", "结论", "总结", "报告", "方案", "组织")
                        )
                    )
                )
            ):
                chosen_intent = IntentCategory.SEARCH
                chosen_reason = "Hard rule: evidence judgment signal detected — search for verification, not calculation."
                hard_rule_hit = True
            # Override 2b: Calc signal → calc
            elif (
                has_calc_signal(user_input)
                and not has_code_signal(user_input)
                and not hard_explain
                and not has_evidence_judgment_signal(user_input)
                and not hard_synthesis
            ):
                chosen_intent = IntentCategory.CALC
                chosen_reason = (
                    "Hard rule: calc intent signal detected (no code intent)."
                )
                hard_rule_hit = True
            # Override 2c: Code + calc → flag-controlled disambiguation
            # When prefer_calculator_for_mixed_numeric_code=1 (default in runtime_controls.json),
            # the system prefers calculator unless a strong code action verb is present.
            # When flag=0, code_executor wins unconditionally (legacy behavior).
            # Strong code action verbs: 运行/写/实现/编写/脚本/执行/跑
            # v5 fix: evidence_judgment 优先于 code+calc 规则
            # BUT: writing intent overrides evidence_judgment (synthesis > verification)
            elif (
                has_evidence_judgment_signal(user_input)
                and not hard_code_action
                and not (
                    hard_formal
                    or hard_writing
                    or (
                        "写" in user_input
                        and any(
                            w in user_input
                            for w in ("正式", "结论", "总结", "报告", "方案", "组织")
                        )
                    )
                )
            ):
                chosen_intent = IntentCategory.SEARCH
                chosen_reason = "Hard rule: evidence judgment overrides code+calc — needs search for verification."
                hard_rule_hit = True
            elif (
                has_calc_signal(user_input)
                and has_code_signal(user_input)
                and not hard_explain
                and not has_evidence_judgment_signal(user_input)
            ):
                _strong_code_action_verbs = (
                    "运行",
                    "写",
                    "实现",
                    "编写",
                    "脚本",
                    "执行",
                    "跑",
                )
                prefer_calc = bool(
                    self.controls.get("prefer_calculator_for_mixed_numeric_code", 0)
                )
                if prefer_calc:
                    if any(v in user_input for v in _strong_code_action_verbs):
                        chosen_intent = IntentCategory.CODE
                        chosen_reason = "Hard rule: code+calc co-occurrence with strong code action verb — code executor wins."
                    else:
                        chosen_intent = IntentCategory.CALC
                        chosen_reason = "Hard rule: code+calc co-occurrence but no strong code action verb — calculator preferred for numeric tasks."
                else:
                    if not any(v in user_input for v in _strong_code_action_verbs):
                        chosen_intent = IntentCategory.CALC
                        chosen_reason = "Hard rule: code+calc co-occurrence but no strong code action verb — calculator preferred for numeric tasks."
                    else:
                        chosen_intent = IntentCategory.CODE
                        chosen_reason = "Hard rule: code+calc co-occurrence with strong code action verb — code executor wins."
                hard_rule_hit = True
            # Override 3: Clear code action → code
            # Explain signal only blocks code when there's no strong code verb
            # ("什么是排序算法" = explain → search, but "写代码分析CSV" = code)
            elif (
                hard_code_action
                and not (hard_explain and not has_strong_code_verb)
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
            # Override 4a: Planning + search signal → search
            # "拆计划" + "查资料" is a planning task that needs search, not code.
            # When the query has deep_analysis signal (计划/方案/规划) AND search
            # signal (资料/哪些/信息) but no actual code action, route to search.
            # Also covers planning queries without explicit search signal — a
            # plan request like "给出发布计划" needs best-practice search, not
            # code execution, even when it mentions dev terms (修复/测试/发布).
            elif (
                has_deep_analysis_signal(user_input)
                and not hard_code_action
                and not hard_arithmetic
                and not hard_writing
            ):
                chosen_intent = IntentCategory.SEARCH
                chosen_reason = "Planning task — user needs information gathering for a plan, not code execution."
                hard_rule_hit = True
            # Override 5: Formal/synthesis → consult (bigmodel)
            # When the user wants to write a formal conclusion/synthesis, route to
            # consult even if conflict was detected — the conflict override (Override 1)
            # already guards against this case, so if we reach here with hard_formal,
            # the user has evidence and wants synthesis.
            elif hard_formal:
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
            # Override 4c: Search signal + no code signal → search
            # When query has search signal (资料/信息/哪些) but no code signal
            # and no strong code action verb, route to search even if semantic
            # code_executor score is slightly higher (e.g. "查官方资料还是跑个脚本").
            elif (
                has_search_signal(user_input)
                and not hard_code_action
                and not has_strong_code_verb
                and not hard_arithmetic
                and semantic_best == "code_executor"
            ):
                chosen_intent = IntentCategory.SEARCH
                chosen_reason = (
                    "Search signal present without code intent — "
                    "prefer information gathering over execution."
                )
                hard_rule_hit = True
            # Override 4d: Choice/alternation query with search signal → search
            # "应该查官方资料还是跑个脚本" — the user is asking which approach
            # to take, not requesting code execution. The "还是" (or/alternatively)
            # pattern with search signal indicates a decision-support query.
            elif (
                "还是" in user_input
                and has_search_signal(user_input)
                and not hard_code_action
                and not hard_arithmetic
            ):
                chosen_intent = IntentCategory.SEARCH
                chosen_reason = (
                    "Choice/alternation query with search signal — "
                    "user needs information to decide, not execute."
                )
                hard_rule_hit = True

            # L4 Fallback: ultra-low confidence → consult (bigmodel)
            # BUT: skip consult fallback when search signal or low-intent flag is present —
            # those queries should default to search, not synthesis.
            # ALSO: English-only queries default to search (knowledge lookup),
            # not consult — "Kafka consumer group rebalance strategy" is a search query.
            _is_english_only = not bool(re.search(r"[\u4e00-\u9fff]", user_input))
            if (
                semantic_best_score < 0.15
                and not hard_rule_hit
                and not hard_conflict
                and not hard_arithmetic
                and not hard_code_action
                and not has_search_signal(user_input)
                and not _is_low_intent
                and not _is_english_only
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

            # Final fallback: no hard rule hit and no low-confidence match.
            # Default to search for general knowledge queries.
            # Also catches low-intent queries ("嗯", "帮我看看") that should
            # default to search rather than being rejected.
            if not hard_rule_hit and chosen_intent is None:
                chosen_intent = IntentCategory.SEARCH
                chosen_reason = "No specific signal detected — defaulting to knowledge search."

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
            and has_compare_signal(context.user_input)
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
                reason="Candidate gate: use search for comparison task before synthesis.",
                tool_call=ToolCall(
                    tool_name=self._resolve_tool_name(IntentCategory.SEARCH),
                    query=context.user_input,
                    arguments={"top_k": 3},
                    reason="Comparison tasks need source grounding before generation.",
                ),
                feature_snapshot=dict(decision.feature_snapshot),
            )

        # Unconditional guard: compare task should never route to bigmodel
        # (fires even when prefer_search_for_comparison_evidence=0)
        if (
            has_compare_signal(context.user_input)
            and not has_formal_signal(context.user_input)
            and decision.action == Action.CALL_BIGMODEL
        ):
            return ActionDecision(
                action=Action.CALL_TOOL,
                score=decision.score,
                reason="Unconditional guard: compare task detected — forcing search over bigmodel.",
                tool_call=ToolCall(
                    tool_name=self._resolve_tool_name(IntentCategory.SEARCH),
                    query=context.user_input,
                    arguments={"top_k": 3},
                    reason="Comparison tasks must gather evidence before any synthesis.",
                ),
                feature_snapshot=dict(decision.feature_snapshot),
            )

        # Guard: explicit search action should never route to bigmodel
        # When concept memory or ambiguous semantics push CALL_BIGMODEL but
        # the query has an explicit search action (检索/搜索/查一下) and no
        # writing/formal intent, force search instead.
        if (
            has_search_action_signal(context.user_input)
            and not has_formal_signal(context.user_input)
            and not has_writing_signal(context.user_input)
            and not (
                "写" in context.user_input
                and any(
                    w in context.user_input
                    for w in ("正式", "结论", "总结", "报告", "方案", "组织")
                )
            )
            and decision.action == Action.CALL_BIGMODEL
        ):
            return ActionDecision(
                action=Action.CALL_TOOL,
                score=decision.score,
                reason="Guard: explicit search action overrides bigmodel escalation.",
                tool_call=ToolCall(
                    tool_name=self._resolve_tool_name(IntentCategory.SEARCH),
                    query=context.user_input,
                    arguments={"top_k": 3},
                    reason="Search action detected — gather evidence before synthesis.",
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
        # BUT: when the user explicitly wants to write a formal conclusion
        # (已经收集了资料 + 写正式结论), they have evidence and want synthesis.
        if (
            is_conflict_task(context.user_input)
            and memory.latest("RESULT") is None
            and decision.action != Action.CALL_TOOL
            and not has_formal_signal(context.user_input)
            and not (
                "写" in context.user_input
                and any(
                    w in context.user_input
                    for w in ("正式", "结论", "总结", "报告")
                )
            )
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
