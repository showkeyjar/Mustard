from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from carm.memory import MemoryBoard
from carm.runtime_controls import DEFAULT_CONTROLS
from carm.schemas import StepRecord
from carm.state import AgentState


class AdaptiveReasoningCore:
    """Small online-updated recurrent reasoning core with continuous latent state."""

    SLOT_TYPES = ("PLAN", "HYP", "DRAFT")
    LATENT_DIM = 6

    def __init__(self, state_path: str | Path, controls: dict[str, float | int] | None = None) -> None:
        self.state_path = Path(state_path)
        base_controls = dict(DEFAULT_CONTROLS["core"])
        if controls:
            base_controls.update(controls)
        self.controls = base_controls
        self.learning_rate = 0.06
        self.slot_bias: dict[str, float] = {slot: 0.0 for slot in self.SLOT_TYPES}
        self.feature_weights: dict[str, dict[str, float]] = {
            slot: {} for slot in self.SLOT_TYPES
        }
        self.token_slot_weights: dict[str, dict[str, float]] = {}
        self.recurrent_weights: dict[str, list[float]] = {}
        self.slot_readout_weights: dict[str, list[float]] = {
            slot: [0.0] * self.LATENT_DIM for slot in self.SLOT_TYPES
        }
        self._load()
        if not any(self.feature_weights[slot] for slot in self.SLOT_TYPES):
            self._bootstrap()

    def step(
        self,
        observation: dict[str, str],
        memory: MemoryBoard,
        state: AgentState,
        guidance: dict[str, object] | None = None,
    ) -> AgentState:
        next_state = state.snapshot()
        next_state.step_idx += 1
        next_state.phase = "REASONING"

        user_input = observation["input"]
        if "goal_initialized" not in next_state.hidden:
            next_state.hidden["candidate"] = user_input
            next_state.hidden["slot_type"] = "GOAL"
            next_state.hidden["goal_initialized"] = "1"
            next_state.uncertainty = 0.9
            next_state.answer_ready = 0.0
            return next_state

        features = self.extract_features(user_input, memory, state, guidance)
        next_state.latent = self.update_latent(state.latent, features)
        slot_type = self.choose_slot(user_input, features, guidance)
        candidate = self.render_candidate(slot_type, memory, user_input)

        next_state.hidden["candidate"] = candidate
        next_state.hidden["slot_type"] = slot_type
        next_state.hidden["core_slot_scores"] = json.dumps(self.score_slots(user_input, features), ensure_ascii=False)
        next_state.hidden["latent_summary"] = self.describe_latent(next_state.latent)
        next_state.uncertainty = self.estimate_uncertainty(slot_type, memory)
        next_state.answer_ready = self.estimate_answer_ready(slot_type, memory)
        if state.hidden.get("verified") == "1" and memory.latest("DRAFT") is not None and memory.latest("CONFLICT") is None:
            next_state.hidden["verified"] = "1"
            next_state.uncertainty = min(next_state.uncertainty, 0.2)
            next_state.answer_ready = max(next_state.answer_ready, 0.95)
        return next_state

    def learn(self, user_input: str, steps: list[StepRecord], success: bool) -> None:
        if not success:
            return

        tokens = self.tokenize(user_input)
        for step in steps:
            if step.action != "WRITE_MEM" or not step.high_value or not step.target_slot:
                continue
            slot_type = step.target_slot
            if slot_type not in self.SLOT_TYPES:
                continue

            counts = Counter(tokens)
            for token, count in counts.items():
                bucket = self.token_slot_weights.setdefault(token, {})
                bucket[slot_type] = bucket.get(slot_type, 0.0) + self.learning_rate * step.reward * min(count, 3)

            for feature_name, feature_value in step.feature_snapshot.items():
                current = self.feature_weights[slot_type].get(feature_name, 0.0)
                self.feature_weights[slot_type][feature_name] = current + self.learning_rate * step.reward * feature_value

            self.slot_bias[slot_type] += self.learning_rate * step.reward
            self._update_readout(slot_type, step.feature_snapshot, step.reward)

        self._save()

    def extract_features(
        self,
        user_input: str,
        memory: MemoryBoard,
        state: AgentState,
        guidance: dict[str, object] | None = None,
    ) -> dict[str, float]:
        lower = user_input.lower()
        preferred_slot = str((guidance or {}).get("preferred_slot", ""))
        return {
            "bias": 1.0,
            "step_idx": min(state.step_idx / 8.0, 1.0),
            "uncertainty": state.uncertainty,
            "answer_ready": state.answer_ready,
            "has_plan": 1.0 if memory.latest("PLAN") else 0.0,
            "has_result": 1.0 if memory.latest("RESULT") else 0.0,
            "has_fact": 1.0 if memory.latest("FACT") else 0.0,
            "has_conflict": 1.0 if memory.latest("CONFLICT") else 0.0,
            "compare_signal": 1.0 if any(token in user_input for token in ("比较", "区别", "优缺点", "vs", "对比")) else 0.0,
            "calc_signal": 1.0 if any(token in user_input for token in ("多少", "计算", "cost", "price", "sum", "数字")) else 0.0,
            "code_signal": 1.0 if any(token in lower for token in ("python", "code", "script", "代码")) else 0.0,
            "need_structure": 1.0 if memory.latest("PLAN") is None else 0.0,
            "need_external": 1.0 if memory.latest("RESULT") is None else 0.0,
            "user_prefers_plan": 1.0 if preferred_slot == "PLAN" else 0.0,
            "user_prefers_hyp": 1.0 if preferred_slot == "HYP" else 0.0,
            "user_prefers_draft": 1.0 if preferred_slot == "DRAFT" else 0.0,
        }

    def choose_slot(self, user_input: str, features: dict[str, float], guidance: dict[str, object] | None = None) -> str:
        scores = self.score_slots(user_input, features, guidance)
        return max(scores, key=scores.get)

    def score_slots(
        self,
        user_input: str,
        features: dict[str, float],
        guidance: dict[str, object] | None = None,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        token_counts = Counter(self.tokenize(user_input))
        latent = self.update_latent([0.0] * self.LATENT_DIM, features)
        preferred_slot = str((guidance or {}).get("preferred_slot", ""))
        for slot_type in self.SLOT_TYPES:
            score = self.slot_bias[slot_type]
            for feature_name, feature_value in features.items():
                score += self.feature_weights[slot_type].get(feature_name, 0.0) * feature_value
            for token, count in token_counts.items():
                score += self.token_slot_weights.get(token, {}).get(slot_type, 0.0) * min(count, 3)
            score += self.dot(self.slot_readout_weights[slot_type], latent)
            if preferred_slot and preferred_slot == slot_type:
                score += 0.9
            scores[slot_type] = score
        return scores

    def update_latent(self, previous: list[float], features: dict[str, float]) -> list[float]:
        latent: list[float] = []
        for index in range(self.LATENT_DIM):
            total = 0.55 * previous[index]
            for feature_name, feature_value in features.items():
                weights = self.recurrent_weights.get(feature_name)
                if weights is None:
                    weights = self._default_recurrent_weights(feature_name)
                    self.recurrent_weights[feature_name] = weights
                total += weights[index] * feature_value
            latent.append(math.tanh(total))
        return latent

    def render_candidate(self, slot_type: str, memory: MemoryBoard, user_input: str) -> str:
        result = memory.latest("RESULT")
        plan = memory.latest("PLAN")
        fact = memory.latest("FACT")
        latent_hint = self._latent_hint(memory)
        result_text = memory.slot_brief(result)
        plan_payload = memory.parse_content(plan)
        fact_text = memory.slot_brief(fact)
        keywords = self.select_keywords(user_input)

        if slot_type == "DRAFT":
            payload = {
                "kind": "draft",
                "summary": "",
                "support_items": [],
                "open_risks": [],
                "confidence_band": "low",
            }
            if result is not None:
                payload["summary"] = "基于外部结果形成初步结论"
                payload["support_items"] = [result_text]
                payload["confidence_band"] = "high"
            elif plan is not None:
                payload["summary"] = "基于当前计划形成待验证结论"
                payload["support_items"] = list(plan_payload.get("action_items", []))[:2]
                payload["open_risks"] = list(plan_payload.get("unknowns", []))[:2]
                payload["confidence_band"] = "medium"
            elif fact is not None:
                payload["summary"] = "基于既有经验形成暂定结论"
                payload["support_items"] = [fact_text]
                payload["confidence_band"] = "medium"
            else:
                payload["summary"] = f"围绕任务形成临时草稿: {user_input}"
                payload["open_risks"] = [latent_hint]
                payload["confidence_band"] = "low"
            return json.dumps(payload, ensure_ascii=False)

        if slot_type == "PLAN":
            action_items = self.plan_steps(user_input, keywords)
            evidence_targets = self.plan_needs(memory, user_input)
            unknowns = self.plan_unknowns(user_input, memory, latent_hint)
            payload = {
                "kind": "plan",
                "summary": "建立任务拆解与验证路线",
                "action_items": action_items,
                "unknowns": unknowns,
                "evidence_targets": evidence_targets,
                "keywords": keywords,
                "confidence_band": "medium",
            }
            if fact is not None:
                payload["summary"] = "迁移既有经验并调整计划"
                payload["action_items"].insert(0, f"吸收经验线索: {fact_text}")
            if plan is not None:
                prior_items = plan_payload.get("action_items", [])
                payload["action_items"] = self.merge_steps(prior_items, payload["action_items"])
                prior_unknowns = plan_payload.get("unknowns", [])
                payload["unknowns"] = self.merge_steps(prior_unknowns, payload["unknowns"])
            return json.dumps(payload, ensure_ascii=False)

        payload = {
            "kind": "hypothesis",
            "summary": "",
            "assumptions": [],
            "evidence_targets": [],
            "confidence_band": "low" if result is None else "medium",
        }
        if plan is not None:
            payload["summary"] = "按计划补充关键事实并验证"
            payload["assumptions"] = list(plan_payload.get("unknowns", []))[:2] or ["当前方案缺少关键事实"]
            payload["evidence_targets"] = list(plan_payload.get("evidence_targets", [])) or ["外部支持"]
            return json.dumps(payload, ensure_ascii=False)
        if result is None:
            payload["summary"] = "当前信息不足，需要外部支持验证"
            payload["assumptions"] = [latent_hint]
            payload["evidence_targets"] = [*keywords[:2]] or ["外部支持"]
            return json.dumps(payload, ensure_ascii=False)
        payload["summary"] = "根据已有结果形成待验证结论"
        payload["assumptions"] = ["已有结果可以支撑初步结论"]
        payload["evidence_targets"] = [result_text]
        return json.dumps(payload, ensure_ascii=False)

    def estimate_uncertainty(self, slot_type: str, memory: MemoryBoard) -> float:
        draft_delta = float(self.controls.get("result_draft_uncertainty_delta", 0.0))
        if slot_type == "DRAFT":
            base = 0.3 if memory.latest("RESULT") is not None else 0.48
            return max(0.05, base - draft_delta)
        if slot_type == "PLAN":
            return 0.62 if memory.latest("FACT") is not None else 0.72
        return 0.82 if memory.latest("RESULT") is None else 0.55

    def estimate_answer_ready(self, slot_type: str, memory: MemoryBoard) -> float:
        draft_bonus = float(self.controls.get("result_draft_answer_ready_bonus", 0.0))
        if slot_type == "DRAFT":
            base = 0.6 if memory.latest("RESULT") is not None else 0.35
            return min(0.95, base + draft_bonus)
        if slot_type == "PLAN":
            return 0.18
        return 0.12 if memory.latest("RESULT") is None else 0.28

    def tokenize(self, text: str) -> list[str]:
        ascii_tokens: list[str] = []
        current = []
        for char in text.lower():
            if char.isascii() and (char.isalnum() or char == "_"):
                current.append(char)
            else:
                if len(current) >= 2:
                    ascii_tokens.append("".join(current))
                current = []
        if len(current) >= 2:
            ascii_tokens.append("".join(current))

        chinese_runs: list[str] = []
        current = []
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                current.append(char)
            else:
                if len(current) >= 2:
                    chinese_runs.append("".join(current))
                current = []
        if len(current) >= 2:
            chinese_runs.append("".join(current))

        chinese_tokens: list[str] = []
        for run in chinese_runs:
            chinese_tokens.append(run)
            for size in (2, 3):
                for index in range(0, max(0, len(run) - size + 1)):
                    chinese_tokens.append(run[index : index + size])

        return list(dict.fromkeys(ascii_tokens + chinese_tokens))

    def describe_latent(self, latent: list[float]) -> str:
        labels = [
            "结构化",
            "求证",
            "成稿",
            "外部依赖",
            "经验迁移",
            "约束聚焦",
        ]
        ranked = sorted(enumerate(latent), key=lambda item: abs(item[1]), reverse=True)[:2]
        return ", ".join(f"{labels[index]}={value:.2f}" for index, value in ranked)

    def dot(self, left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def select_keywords(self, user_input: str) -> list[str]:
        tokens = self.tokenize(user_input)
        return tokens[:4]

    def plan_steps(self, user_input: str, keywords: list[str]) -> list[str]:
        steps = [
            "明确目标与约束",
            "识别需要验证的维度",
            "决定是否调用外部工具",
        ]
        if keywords:
            steps.append(f"聚焦关键词: {', '.join(keywords)}")
        return steps

    def plan_needs(self, memory: MemoryBoard, user_input: str) -> list[str]:
        needs: list[str] = []
        if any(token in user_input for token in ("比较", "区别", "优缺点", "vs", "对比")):
            needs.extend(["比较维度", "外部事实"])
        if any(token in user_input for token in ("多少", "计算", "cost", "price", "sum", "数字")):
            needs.append("精确数值")
        if any(token in user_input.lower() for token in ("python", "code", "script")) or "代码" in user_input:
            needs.append("可执行验证")
        if memory.latest("RESULT") is None and not needs:
            needs.append("外部支持")
        return needs

    def plan_unknowns(self, user_input: str, memory: MemoryBoard, latent_hint: str) -> list[str]:
        unknowns: list[str] = []
        if memory.latest("RESULT") is None:
            unknowns.append("缺少外部结果")
        if any(token in user_input for token in ("比较", "区别", "优缺点", "vs", "对比")):
            unknowns.append("比较维度尚未完全确认")
        if any(token in user_input for token in ("多少", "计算", "cost", "price", "sum", "数字")):
            unknowns.append("需要精确数值支撑")
        if not unknowns:
            unknowns.append(latent_hint)
        return unknowns[:3]

    def merge_steps(self, previous: object, current: list[str]) -> list[str]:
        merged: list[str] = []
        for step in list(previous) if isinstance(previous, list) else []:
            text = str(step)
            if text not in merged:
                merged.append(text)
        for step in current:
            if step not in merged:
                merged.append(step)
        return merged[:6]

    def _default_recurrent_weights(self, feature_name: str) -> list[float]:
        presets = {
            "need_structure": [0.8, 0.1, -0.3, 0.2, 0.1, 0.5],
            "need_external": [0.2, 0.75, -0.4, 0.7, 0.1, 0.2],
            "has_result": [-0.4, -0.2, 0.95, -0.5, 0.1, 0.0],
            "has_fact": [0.4, 0.1, 0.0, -0.1, 0.85, 0.2],
            "compare_signal": [0.75, 0.1, -0.1, 0.2, 0.0, 0.45],
            "calc_signal": [0.1, 0.7, -0.2, 0.55, 0.0, 0.1],
            "code_signal": [0.45, 0.35, -0.2, 0.3, 0.0, 0.35],
            "answer_ready": [-0.2, -0.3, 0.9, -0.2, 0.0, -0.1],
            "uncertainty": [0.15, 0.5, -0.5, 0.55, 0.1, 0.2],
            "has_plan": [0.25, 0.35, 0.15, 0.1, 0.0, 0.3],
            "has_conflict": [-0.1, 0.6, -0.4, 0.3, -0.2, 0.55],
            "bias": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            "step_idx": [0.1, 0.1, 0.25, 0.0, 0.0, 0.15],
        }
        return presets.get(feature_name, [0.0] * self.LATENT_DIM)

    def _update_readout(self, slot_type: str, features: dict[str, float], reward: float) -> None:
        latent = self.update_latent([0.0] * self.LATENT_DIM, features)
        for index, value in enumerate(latent):
            self.slot_readout_weights[slot_type][index] += self.learning_rate * reward * value

    def _latent_hint(self, memory: MemoryBoard) -> str:
        if memory.latest("RESULT") is not None:
            return "整合外部结果"
        if memory.latest("FACT") is not None:
            return "迁移既有经验"
        if memory.latest("PLAN") is not None:
            return "补全验证事实"
        return "先建立结构与验证路径"

    def _bootstrap(self) -> None:
        plan_weights = self.feature_weights["PLAN"]
        plan_weights.update(
            {
                "need_structure": 0.95,
                "compare_signal": 0.55,
                "code_signal": 0.45,
                "has_fact": 0.35,
                "has_result": -0.4,
            }
        )
        hyp_weights = self.feature_weights["HYP"]
        hyp_weights.update(
            {
                "need_external": 0.85,
                "uncertainty": 0.4,
                "has_plan": 0.55,
                "calc_signal": 0.45,
                "has_result": -0.55,
            }
        )
        draft_weights = self.feature_weights["DRAFT"]
        draft_weights.update(
            {
                "has_result": 1.15,
                "answer_ready": 0.7,
                "need_external": -0.8,
                "has_plan": 0.2,
                "user_prefers_draft": 0.9,
            }
        )
        plan_weights["user_prefers_plan"] = 0.85
        hyp_weights["user_prefers_hyp"] = 0.85
        self.token_slot_weights = {
            "比较": {"PLAN": 0.4},
            "对比": {"PLAN": 0.4},
            "代码": {"PLAN": 0.35},
            "计算": {"HYP": 0.45},
        }
        self.slot_readout_weights["PLAN"] = [0.75, 0.2, -0.4, 0.15, 0.25, 0.55]
        self.slot_readout_weights["HYP"] = [0.1, 0.85, -0.35, 0.7, 0.1, 0.2]
        self.slot_readout_weights["DRAFT"] = [-0.25, -0.2, 1.05, -0.4, 0.1, -0.1]
        self._save()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
        self.slot_bias.update(payload.get("slot_bias", {}))
        stored_feature_weights = payload.get("feature_weights", {})
        for slot_type, weights in stored_feature_weights.items():
            if slot_type in self.feature_weights:
                self.feature_weights[slot_type].update(weights)
        self.token_slot_weights.update(payload.get("token_slot_weights", {}))
        stored_recurrent = payload.get("recurrent_weights", {})
        for feature_name, weights in stored_recurrent.items():
            self.recurrent_weights[feature_name] = list(weights)
        stored_readouts = payload.get("slot_readout_weights", {})
        for slot_type, weights in stored_readouts.items():
            if slot_type in self.slot_readout_weights:
                self.slot_readout_weights[slot_type] = list(weights)

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "learning_rate": self.learning_rate,
            "slot_bias": self.slot_bias,
            "feature_weights": self.feature_weights,
            "token_slot_weights": self.token_slot_weights,
            "recurrent_weights": self.recurrent_weights,
            "slot_readout_weights": self.slot_readout_weights,
        }
        self.state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
