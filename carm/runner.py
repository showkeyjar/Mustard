from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from carm.actions import Action
from carm.core import AdaptiveReasoningCore
from carm.decoder import SimpleDecoder
from carm.encoder import SimpleEncoder
from carm.experience import ExperienceStore
from carm.glance import InternalGlance
from carm.memory import MemoryBoard, MemorySlot
from carm.policy import OnlinePolicy
from carm.review import ReviewStore
from carm.runtime_controls import load_control_state, load_controls
from carm.schemas import EpisodeRecord, ReviewRecord, StepRecord
from carm.state import AgentState
from carm.verifier import SimpleVerifier
from tools.base import ToolManager


@dataclass
class RollbackCheckpoint:
    state: AgentState
    slots: list[MemorySlot]


@dataclass
class RunTrace:
    actions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)


class AgentRunner:
    def __init__(
        self,
        tool_manager: ToolManager,
        max_steps: int = 8,
        experience_path: str | Path = "data/experience/episodes.jsonl",
        policy_state_path: str | Path = "data/experience/policy_state.json",
        concept_state_path: str | Path = "data/experience/concept_state.json",
        core_state_path: str | Path = "data/experience/core_state.json",
        review_path: str | Path = "data/review/reviews.jsonl",
        controls_path: str | Path = "data/control/runtime_controls.json",
    ) -> None:
        self.controls_path = Path(controls_path)
        self.controls = load_controls(self.controls_path)
        self.control_state = load_control_state(self.controls_path.parent / "control_state.json")
        self.control_version = str(self.control_state.get("current_version", ""))
        self.encoder = SimpleEncoder()
        self.core = AdaptiveReasoningCore(core_state_path, self.controls.get("core", {}))
        self.policy = OnlinePolicy(policy_state_path, concept_state_path, self.controls.get("policy", {}))
        self.verifier = SimpleVerifier()
        self.decoder = SimpleDecoder()
        self.glance = InternalGlance(self.controls.get("glance", {}))
        self.tool_manager = tool_manager
        self.max_steps = max_steps
        self.experience_store = ExperienceStore(experience_path)
        self.review_store = ReviewStore(review_path)

    def run(self, user_input: str) -> tuple[str, RunTrace]:
        state = AgentState(glance_budget=int(self.controls.get("glance", {}).get("budget", 1)))
        memory = MemoryBoard()
        trace = RunTrace()
        rollback_stack: list[RollbackCheckpoint] = []
        self._hydrate_from_experience(user_input, memory)

        for _ in range(self.max_steps):
            observation = self.encoder.encode(user_input, memory)
            state = self.core.step(observation, memory, state)
            state = self._apply_glance(state, memory)
            decision = self.policy.decide(state, memory, user_input)

            trace.actions.append(decision.action.value)
            trace.notes.append(decision.reason)
            trace.steps.append(
                StepRecord(
                    step_idx=state.step_idx,
                    action=decision.action.value,
                    reason=decision.reason,
                    score=decision.score,
                    feature_snapshot=dict(decision.feature_snapshot),
                    state_signature=self._state_signature(state),
                    memory_signature=self._memory_signature(memory),
                    user_input=user_input,
                    selected_tool=decision.tool_call.tool_name if decision.tool_call else "",
                    target_slot=decision.target_slot or "",
                    glance_used=bool(state.hidden.get("glance_trigger")),
                )
            )

            if decision.action == Action.THINK:
                memory.decay()
                continue

            if decision.action == Action.READ_MEM and decision.target_slot:
                memory.focus(decision.target_slot)
                continue

            if decision.action == Action.WRITE_MEM and decision.target_slot:
                rollback_stack.append(self._checkpoint(state, memory))
                memory.write_from_state(state, decision.target_slot, "policy")
                state.last_action = decision.action.value
                continue

            if decision.action in {Action.CALL_TOOL, Action.CALL_BIGMODEL} and decision.tool_call:
                rollback_stack.append(self._checkpoint(state, memory))
                result = self.tool_manager.execute(
                    decision.tool_call.tool_name,
                    decision.tool_call.query,
                    decision.tool_call.arguments,
                )
                memory.store_result(result.result, result.confidence, result.source)
                state.last_action = decision.action.value
                state.uncertainty = max(0.2, state.uncertainty - 0.3)
                continue

            if decision.action == Action.VERIFY:
                ok, message = self.verifier.check(state, memory)
                trace.notes.append(message)
                if ok:
                    state.answer_ready = max(state.answer_ready, 0.95)
                    state.uncertainty = min(state.uncertainty, 0.2)
                state.last_action = decision.action.value
                continue

            if decision.action == Action.ROLLBACK:
                if rollback_stack:
                    checkpoint = rollback_stack.pop()
                    state = checkpoint.state
                    memory.restore(checkpoint.slots)
                    trace.notes.append("Rollback restored previous checkpoint.")
                state.last_action = decision.action.value
                continue

            if decision.action == Action.ANSWER:
                state.last_action = decision.action.value
                answer = self.decoder.render(user_input, state, memory)
                self._finalize_episode(user_input, answer, state, memory, trace)
                return answer, trace

        answer = self.decoder.render(user_input, state, memory)
        self._finalize_episode(user_input, answer, state, memory, trace)
        return answer, trace

    def _checkpoint(self, state: AgentState, memory: MemoryBoard) -> RollbackCheckpoint:
        return RollbackCheckpoint(
            state=state.snapshot(),
            slots=memory.read(),
        )

    def _hydrate_from_experience(self, user_input: str, memory: MemoryBoard) -> None:
        recalled = self.experience_store.recall(user_input, limit=2)
        for episode in recalled:
            memory.write(
                MemorySlot(
                    slot_type="FACT",
                    content=f"经验提示: {episode.summary}",
                    confidence=min(max(episode.value_score, 0.3), 0.95),
                    source="experience",
                    ttl=10,
                )
            )

    def _finalize_episode(
        self,
        user_input: str,
        answer: str,
        state: AgentState,
        memory: MemoryBoard,
        trace: RunTrace,
    ) -> None:
        success = memory.latest("DRAFT") is not None and memory.latest("CONFLICT") is None
        value_score = self._value_score(state, memory, trace)
        self._assign_rewards(trace, state, memory, value_score)
        self.core.learn(user_input, trace.steps, success)
        self.policy.learn(trace.steps)

        summary = self._summarize_episode(memory, trace)
        episode_features = self._episode_features(user_input, memory, trace)
        outcome_signature = self._outcome_signature(state, memory, trace, success, value_score)
        episode = EpisodeRecord(
            user_input=user_input,
            answer=answer,
            summary=summary,
            success=success,
            value_score=value_score,
            episode_features=episode_features,
            outcome_signature=outcome_signature,
            steps=trace.steps,
        )
        self.experience_store.append(episode)
        review = self._review_episode(user_input, trace, success, value_score, episode_features, outcome_signature)
        self.review_store.append(review)

    def _assign_rewards(
        self,
        trace: RunTrace,
        state: AgentState,
        memory: MemoryBoard,
        value_score: float,
    ) -> None:
        final_success = memory.latest("CONFLICT") is None and memory.latest("DRAFT") is not None

        for index, step in enumerate(trace.steps):
            reward = 0.0
            reasons: list[str] = []
            if step.action == Action.WRITE_MEM.value:
                reward += 0.35
                reasons.append("state_written")
            if step.action == Action.CALL_TOOL.value and memory.latest("RESULT") is not None:
                reward += 0.75
                reasons.append("tool_result_obtained")
            if step.action == Action.CALL_BIGMODEL.value and memory.latest("RESULT") is not None:
                reward += 0.5
                reasons.append("external_reasoning_obtained")
            if step.action == Action.VERIFY.value:
                reward += 0.2 if final_success else -0.2
                reasons.append("verification_helped" if final_success else "verification_failed")
            if step.action == Action.THINK.value:
                reward -= 0.1
                reasons.append("idle_reasoning_cost")
            if step.action == Action.ROLLBACK.value:
                reward += 0.15 if memory.latest("CONFLICT") is not None else -0.15
                reasons.append("rollback_needed" if memory.latest("CONFLICT") is not None else "rollback_unnecessary")
            if step.action == Action.ANSWER.value:
                reward += 1.0 if final_success and state.uncertainty <= 0.35 else -0.6
                reasons.append("stable_answer" if final_success and state.uncertainty <= 0.35 else "premature_answer")

            if index > 4:
                reward -= 0.05
                reasons.append("long_horizon_cost")

            step.reward = reward
            step.reward_reason = ",".join(reasons) if reasons else "neutral"
            step.high_value = abs(reward) >= 0.25 and value_score >= 0.45
            step.glance_helped = step.glance_used and reward > 0.0

    def _value_score(self, state: AgentState, memory: MemoryBoard, trace: RunTrace) -> float:
        score = 0.0
        if memory.latest("RESULT") is not None:
            score += 0.35
        if memory.latest("PLAN") is not None:
            score += 0.2
        if memory.latest("DRAFT") is not None:
            score += 0.25
        if memory.latest("CONFLICT") is None:
            score += 0.1
        if trace.actions and trace.actions[-1] == Action.ANSWER.value:
            score += 0.1
        score -= min(state.uncertainty, 1.0) * 0.2
        return max(0.0, min(score, 1.0))

    def _summarize_episode(self, memory: MemoryBoard, trace: RunTrace) -> str:
        parts: list[str] = []
        plan = memory.latest("PLAN")
        result = memory.latest("RESULT")
        draft = memory.latest("DRAFT")
        if plan:
            parts.append(plan.content)
        if result:
            parts.append(result.content)
        if draft:
            parts.append(draft.content)
        parts.append(f"actions={'/'.join(trace.actions)}")
        return " | ".join(parts[:4])

    def _episode_features(self, user_input: str, memory: MemoryBoard, trace: RunTrace) -> dict[str, object]:
        plan = memory.parse_content(memory.latest("PLAN"))
        hyp = memory.parse_content(memory.latest("HYP"))
        draft = memory.parse_content(memory.latest("DRAFT"))
        result = memory.slot_brief(memory.latest("RESULT"))
        tokens = self.core.tokenize(user_input)
        return {
            "control_version": self.control_version,
            "keywords": tokens[:6],
            "action_sequence": list(trace.actions),
            "plan_summary": plan.get("summary", ""),
            "plan_action_items": list(plan.get("action_items", []))[:4] if isinstance(plan.get("action_items"), list) else [],
            "plan_unknowns": list(plan.get("unknowns", []))[:4] if isinstance(plan.get("unknowns"), list) else [],
            "hyp_summary": hyp.get("summary", ""),
            "evidence_targets": list(hyp.get("evidence_targets", []) or plan.get("evidence_targets", []))[:4]
            if isinstance(hyp.get("evidence_targets", []) or plan.get("evidence_targets", []), list)
            else [],
            "draft_summary": draft.get("summary", ""),
            "used_tool": next((step.selected_tool for step in trace.steps if step.selected_tool), ""),
            "has_result": memory.latest("RESULT") is not None,
            "result_brief": result,
        }

    def _outcome_signature(
        self,
        state: AgentState,
        memory: MemoryBoard,
        trace: RunTrace,
        success: bool,
        value_score: float,
    ) -> dict[str, object]:
        draft = memory.parse_content(memory.latest("DRAFT"))
        return {
            "control_version": self.control_version,
            "success": success,
            "value_score": round(value_score, 4),
            "final_action": trace.actions[-1] if trace.actions else "",
            "step_count": len(trace.actions),
            "uncertainty": round(state.uncertainty, 4),
            "confidence_band": draft.get("confidence_band", ""),
            "has_conflict": memory.latest("CONFLICT") is not None,
            "used_external_result": memory.latest("RESULT") is not None,
        }

    def _state_signature(self, state: AgentState) -> dict[str, object]:
        return {
            "phase": state.phase,
            "last_action": state.last_action,
            "step_idx": state.step_idx,
            "uncertainty": round(state.uncertainty, 4),
            "answer_ready": round(state.answer_ready, 4),
            "candidate_slot": state.hidden.get("slot_type", ""),
            "latent_summary": state.hidden.get("latent_summary", ""),
            "glance_trigger": state.hidden.get("glance_trigger", ""),
            "glance_suggestion": state.hidden.get("glance_suggestion", ""),
            "glance_budget": state.glance_budget,
        }

    def _memory_signature(self, memory: MemoryBoard) -> dict[str, object]:
        return {
            "has_goal": memory.latest("GOAL") is not None,
            "has_plan": memory.latest("PLAN") is not None,
            "has_hyp": memory.latest("HYP") is not None,
            "has_result": memory.latest("RESULT") is not None,
            "has_draft": memory.latest("DRAFT") is not None,
            "has_conflict": memory.latest("CONFLICT") is not None,
            "focus_slot": memory.focus_slot or "",
            "plan_brief": memory.slot_brief(memory.latest("PLAN")),
            "result_brief": memory.slot_brief(memory.latest("RESULT")),
        }

    def _apply_glance(self, state: AgentState, memory: MemoryBoard) -> AgentState:
        next_state = state.snapshot()
        signal = self.glance.inspect(next_state, memory)
        next_state.hidden.pop("glance_trigger", None)
        next_state.hidden.pop("glance_suggestion", None)
        next_state.hidden.pop("glance_focus", None)
        next_state.hidden.pop("glance_cooldown", None)

        if signal.active:
            next_state.hidden["glance_trigger"] = signal.trigger
            next_state.hidden["glance_suggestion"] = signal.suggestion
            next_state.hidden["glance_focus"] = " | ".join(signal.focus)
            next_state.hidden["glance_cooldown"] = "1"
            next_state.glance_budget = max(0, next_state.glance_budget - 1)
        return next_state

    def _review_episode(
        self,
        user_input: str,
        trace: RunTrace,
        success: bool,
        value_score: float,
        episode_features: dict[str, object],
        outcome_signature: dict[str, object],
    ) -> ReviewRecord:
        issue_tags: list[str] = []
        strengths: list[str] = []
        weaknesses: list[str] = []
        recommendations: list[str] = []
        target_modules: list[str] = []

        action_sequence = trace.actions
        tool_used = bool(episode_features.get("used_tool"))
        has_result = bool(episode_features.get("has_result"))
        step_count = len(action_sequence)

        if success and has_result:
            strengths.append("外部结果被成功整合进最终结论")
        if success and "WRITE_MEM" in action_sequence:
            strengths.append("中间状态被显式写入工作记忆")
        if step_count <= 5 and success:
            strengths.append("执行路径较短且收敛稳定")

        if not tool_used and any(tag in user_input for tag in ("比较", "计算", "多少", "代码")):
            issue_tags.append("tool_underuse")
            weaknesses.append("需要工具支持的问题未显式调用工具")
            recommendations.append("提高相关表达下 CALL_TOOL 的优先级")
            target_modules.append("policy")

        if any(step.reward_reason == "idle_reasoning_cost" for step in trace.steps):
            issue_tags.append("idle_drift")
            weaknesses.append("出现无效思考步，说明状态到动作映射仍可收紧")
            recommendations.append("对 THINK 动作增加更强的条件约束")
            target_modules.append("policy")

        if not success:
            issue_tags.append("failed_episode")
            weaknesses.append("本轮未形成稳定草稿或存在冲突")
            recommendations.append("优先检查 core 生成的 slot 类型是否与任务阶段匹配")
            target_modules.append("core")

        if has_result and outcome_signature.get("confidence_band") in {"low", "medium"}:
            issue_tags.append("weak_grounding")
            weaknesses.append("已有外部结果但最终置信带仍偏低")
            recommendations.append("强化结果到 DRAFT 的收敛路径")
            target_modules.append("core")

        if not issue_tags:
            issue_tags.append("stable_path")
            recommendations.append("保留当前策略，仅作为 consolidation 的正例")
            target_modules.append("policy")

        evidence = {
            "action_sequence": action_sequence,
            "reward_reasons": [step.reward_reason for step in trace.steps],
            "used_tool": episode_features.get("used_tool", ""),
            "value_score": value_score,
            "glance_triggers": [step.state_signature.get("glance_trigger", "") for step in trace.steps if step.state_signature.get("glance_trigger", "")],
            "glance_help_rate": self._glance_help_rate(trace),
            "controls_snapshot": self.controls,
        }

        return ReviewRecord(
            user_input=user_input,
            success=success,
            value_score=value_score,
            issue_tags=issue_tags,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            target_modules=sorted(set(target_modules)),
            evidence=evidence,
        )

    def _glance_help_rate(self, trace: RunTrace) -> float:
        glance_steps = [step for step in trace.steps if step.glance_used]
        if not glance_steps:
            return 0.0
        helped = sum(1 for step in glance_steps if step.glance_helped)
        return round(helped / len(glance_steps), 4)
