from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from carm.actions import Action
from carm.core import HeuristicReasoningCore
from carm.decoder import SimpleDecoder
from carm.encoder import SimpleEncoder
from carm.experience import ExperienceStore
from carm.memory import MemoryBoard, MemorySlot
from carm.policy import OnlinePolicy
from carm.schemas import EpisodeRecord, StepRecord
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
    ) -> None:
        self.encoder = SimpleEncoder()
        self.core = HeuristicReasoningCore()
        self.policy = OnlinePolicy(policy_state_path)
        self.verifier = SimpleVerifier()
        self.decoder = SimpleDecoder()
        self.tool_manager = tool_manager
        self.max_steps = max_steps
        self.experience_store = ExperienceStore(experience_path)

    def run(self, user_input: str) -> tuple[str, RunTrace]:
        state = AgentState()
        memory = MemoryBoard()
        trace = RunTrace()
        rollback_stack: list[RollbackCheckpoint] = []
        self._hydrate_from_experience(user_input, memory)

        for _ in range(self.max_steps):
            observation = self.encoder.encode(user_input, memory)
            state = self.core.step(observation, memory, state)
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
        self.policy.learn(trace.steps)

        summary = self._summarize_episode(memory, trace)
        episode = EpisodeRecord(
            user_input=user_input,
            answer=answer,
            summary=summary,
            success=success,
            value_score=value_score,
            steps=trace.steps,
        )
        self.experience_store.append(episode)

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
            if step.action == Action.WRITE_MEM.value:
                reward += 0.35
            if step.action == Action.CALL_TOOL.value and memory.latest("RESULT") is not None:
                reward += 0.75
            if step.action == Action.CALL_BIGMODEL.value and memory.latest("RESULT") is not None:
                reward += 0.5
            if step.action == Action.VERIFY.value:
                reward += 0.2 if final_success else -0.2
            if step.action == Action.THINK.value:
                reward -= 0.1
            if step.action == Action.ROLLBACK.value:
                reward += 0.15 if memory.latest("CONFLICT") is not None else -0.15
            if step.action == Action.ANSWER.value:
                reward += 1.0 if final_success and state.uncertainty <= 0.35 else -0.6

            if index > 4:
                reward -= 0.05

            step.reward = reward
            step.high_value = abs(reward) >= 0.25 and value_score >= 0.45

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
