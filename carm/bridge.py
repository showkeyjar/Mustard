from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from carm.multimodal import MultimodalAdapter
from carm.runner import AgentRunner

if TYPE_CHECKING:
    from carm.desktop import DesktopDigest


@dataclass
class BridgeEvent:
    event_id: str
    timestamp_utc: str
    kind: str
    summary: str
    prompt: str
    source: str
    status: str = "open"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BridgeFeedback:
    timestamp_utc: str
    event_id: str
    feedback_type: str
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BridgeMessage:
    timestamp_utc: str
    role: str
    text: str
    source: str = "chat"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BridgeState:
    current_goal: str = ""
    goal_source_event_id: str = ""
    last_goal_update_utc: str = ""
    pending_suggestion: str = ""
    pending_event_id: str = ""
    pending_question: str = ""
    pending_question_event_id: str = ""
    last_proactive_prompt_utc: str = ""
    proactive_budget_remaining: int = 0
    proactive_status: str = ""


class JsonlStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def load_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        items: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                items.append(payload)
        return items

    def overwrite(self, items: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for payload in items:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class BridgeEventStore:
    def __init__(self, path: str | Path) -> None:
        self._store = JsonlStore(path)

    def append(self, event: BridgeEvent) -> None:
        self._store.append(asdict(event))

    def load_all(self) -> list[BridgeEvent]:
        return [BridgeEvent(**item) for item in self._store.load_all()]

    def update_status(self, event_id: str, status: str) -> None:
        items = self._store.load_all()
        for item in items:
            if str(item.get("event_id", "")) == event_id:
                item["status"] = status
        self._store.overwrite(items)


class BridgeFeedbackStore:
    def __init__(self, path: str | Path) -> None:
        self._store = JsonlStore(path)

    def append(self, feedback: BridgeFeedback) -> None:
        self._store.append(asdict(feedback))

    def load_all(self) -> list[BridgeFeedback]:
        return [BridgeFeedback(**item) for item in self._store.load_all()]


class BridgeMessageStore:
    def __init__(self, path: str | Path) -> None:
        self._store = JsonlStore(path)

    def append(self, message: BridgeMessage) -> None:
        self._store.append(asdict(message))

    def load_recent(self, limit: int = 50) -> list[BridgeMessage]:
        items = self._store.load_all()[-limit:]
        return [BridgeMessage(**item) for item in items]


class BridgeStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> BridgeState:
        if not self.path.exists():
            return BridgeState()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return BridgeState()
        return BridgeState(**payload)

    def save(self, state: BridgeState) -> None:
        self.path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")


class DesktopBridgeController:
    def __init__(
        self,
        runner: AgentRunner,
        event_store: BridgeEventStore,
        feedback_store: BridgeFeedbackStore,
        message_store: BridgeMessageStore,
        state_store: BridgeStateStore | None = None,
        proactive_config: dict[str, Any] | None = None,
    ) -> None:
        self.runner = runner
        self.event_store = event_store
        self.feedback_store = feedback_store
        self.message_store = message_store
        self.state_store = state_store
        self.multimodal_adapter = MultimodalAdapter()
        self.proactive_config = proactive_config or {
            "cooldown_s": 120,
            "min_goal_overlap": 1,
            "initial_budget": 2,
            "max_budget": 3,
            "feedback_window": 6,
            "negative_feedback_limit": 2,
        }

    def ingest_digest(self, digest: DesktopDigest) -> BridgeEvent:
        multimodal_signal = self.multimodal_adapter.from_desktop_digest(digest)
        timestamp = datetime.now(timezone.utc).isoformat()
        event_id = datetime.now(timezone.utc).strftime("bridge-%Y%m%dT%H%M%S%fZ")
        prompt = self._build_prompt(digest)
        display_summary = digest.logic_summary or digest.multimodal_summary or digest.semantic_summary or digest.summary
        event = BridgeEvent(
            event_id=event_id,
            timestamp_utc=timestamp,
            kind="desktop_digest",
            summary=display_summary,
            prompt=prompt,
            source="desktop",
            metadata={
                "raw_summary": digest.summary,
                "logic_summary": digest.logic_summary,
                "top_apps": list(digest.top_apps),
                "focus_window": digest.focus_window,
                "focus_process": digest.focus_process,
                "focus_class_name": digest.focus_class_name,
                "focus_pid": digest.focus_pid,
                "window_sequence": list(digest.window_sequence),
                "window_observations": list(digest.window_observations),
                "event_types": list(digest.event_types),
                "semantic_summary": digest.semantic_summary,
                "semantic_tags": list(multimodal_signal.tags),
                "semantic_confidence": digest.semantic_confidence,
                "modality_hints": list(multimodal_signal.modality_hints),
                "multimodal_summary": digest.multimodal_summary,
                "multimodal_tags": list(digest.multimodal_tags),
                "multimodal_artifact_path": digest.multimodal_artifact_path,
                "suggested_tool": multimodal_signal.suggested_tool,
                "reasoning_clues": list(digest.reasoning_clues),
                "evidence_items": list(digest.evidence_items),
                "clipboard_seen": digest.clipboard_seen,
                "clipboard_preview": digest.clipboard_preview,
                "mouse_active": digest.mouse_active,
                "keyboard_active": digest.keyboard_active,
                "modifier_keys": list(digest.modifier_keys),
            },
        )
        self.event_store.append(event)
        if self.state_store is not None:
            state = self.state_store.load()
            state.pending_suggestion = self.suggest_goal_for_event(event)
            state.pending_event_id = event.event_id
            question, status = self.maybe_create_proactive_question(event, state)
            state.proactive_status = status
            if question:
                state.pending_question = question
                state.pending_question_event_id = event.event_id
                state.last_proactive_prompt_utc = datetime.now(timezone.utc).isoformat()
                state.proactive_budget_remaining = max(0, state.proactive_budget_remaining - 1)
                self.message_store.append(
                    BridgeMessage(
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        role="assistant",
                        text=question,
                        source="bridge_proactive",
                        metadata={"event_id": event.event_id},
                    )
                )
            self.state_store.save(state)
        return event

    def submit_user_message(self, text: str, *, source: str = "chat_window") -> str:
        text = text.strip()
        if not text:
            return ""

        self.message_store.append(
            BridgeMessage(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                role="user",
                text=text,
                source=source,
            )
        )
        answer, _ = self.runner.run(text)
        self.message_store.append(
            BridgeMessage(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                role="assistant",
                text=answer,
                source="carm",
            )
        )
        return answer

    def record_feedback(self, event_id: str, feedback_type: str, note: str = "") -> None:
        feedback = BridgeFeedback(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            event_id=event_id,
            feedback_type=feedback_type,
            note=note,
        )
        self.feedback_store.append(feedback)
        if feedback_type in {"useful", "dismiss", "misread"}:
            self.event_store.update_status(event_id, feedback_type)
            if self.state_store is not None:
                state = self.state_store.load()
                if state.pending_question_event_id == event_id:
                    state.pending_question = ""
                    state.pending_question_event_id = ""
                if feedback_type == "useful":
                    state.proactive_budget_remaining = min(
                        self._max_budget(),
                        max(state.proactive_budget_remaining, 0) + 1,
                    )
                    state.proactive_status = "收到正向反馈，主动追问预算已恢复。"
                elif feedback_type in {"dismiss", "misread"}:
                    state.proactive_budget_remaining = max(0, state.proactive_budget_remaining - 1)
                    state.proactive_status = "收到负向反馈，主动追问会暂时收缩。"
                self.state_store.save(state)

        learning_prompt = (
            f"反馈学习任务: event_id={event_id}，反馈={feedback_type}，说明={note or '无'}。"
            "请据此调整对用户桌面行为的偏好理解。"
        )
        self.submit_user_message(learning_prompt, source="bridge_feedback")

    def load_open_events(self, limit: int = 20) -> list[BridgeEvent]:
        events = self.event_store.load_all()
        return [event for event in events if event.status == "open"][-limit:]

    def event_display_summary(self, event: BridgeEvent) -> str:
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        logic_summary = str(metadata.get("logic_summary", "")).strip()
        multimodal_summary = str(metadata.get("multimodal_summary", "")).strip()
        semantic_summary = str(metadata.get("semantic_summary", "")).strip()
        raw_summary = str(metadata.get("raw_summary", "")).strip()
        return logic_summary or multimodal_summary or semantic_summary or event.summary or raw_summary

    def load_recent_messages(self, limit: int = 50) -> list[BridgeMessage]:
        return self.message_store.load_recent(limit=limit)

    def load_state(self) -> BridgeState:
        if self.state_store is None:
            return BridgeState()
        return self.state_store.load()

    def suggest_goal_for_event(self, event: BridgeEvent) -> str:
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        logic_summary = str(metadata.get("logic_summary", "")).strip("。 ")
        semantic_summary = str(metadata.get("semantic_summary", "")).strip("。 ")
        semantic_tags = [str(item) for item in metadata.get("semantic_tags", []) if str(item).strip()]
        top_apps = metadata.get("top_apps", [])
        if logic_summary:
            return f"当前逻辑线索={logic_summary}"
        if semantic_summary:
            if semantic_tags:
                return f"当前可能在进行{semantic_summary}，标签={','.join(semantic_tags)}"
            return f"当前可能在进行{semantic_summary}"
        app_text = "、".join(str(item) for item in top_apps if str(item).strip())
        if app_text:
            return f"当前可能在处理与 {app_text} 相关的任务"
        return f"当前可能在处理: {event.summary[:48]}"

    def confirm_current_goal(self, goal_text: str, event_id: str = "") -> BridgeState:
        goal_text = goal_text.strip()
        state = self.load_state()
        if not goal_text:
            return state
        state.current_goal = goal_text
        state.goal_source_event_id = event_id
        state.last_goal_update_utc = datetime.now(timezone.utc).isoformat()
        state.proactive_budget_remaining = self._initial_budget()
        state.proactive_status = "当前目标已更新，主动追问预算已重置。"
        if event_id and state.pending_event_id == event_id:
            state.pending_event_id = ""
            state.pending_suggestion = ""
            self.event_store.update_status(event_id, "goal_confirmed")
        if event_id and state.pending_question_event_id == event_id:
            state.pending_question = ""
            state.pending_question_event_id = ""
        if self.state_store is not None:
            self.state_store.save(state)
        self.submit_user_message(
            f"目标确认任务: 当前用户明确目标为 {goal_text}。请更新当前任务偏好并据此协助。",
            source="goal_confirm",
        )
        return state

    def _build_prompt(self, digest: DesktopDigest) -> str:
        app_text = "、".join(digest.top_apps) if digest.top_apps else "当前无明显窗口焦点"
        signals: list[str] = []
        signals.append(f"逻辑摘要={digest.logic_summary}")
        if digest.focus_window:
            signals.append(f"当前焦点={digest.focus_window}")
        if digest.focus_process:
            signals.append(f"焦点进程={digest.focus_process}")
        if digest.focus_class_name:
            signals.append(f"焦点类名={digest.focus_class_name}")
        if digest.focus_pid:
            signals.append(f"焦点PID={digest.focus_pid}")
        if digest.window_sequence:
            signals.append(f"窗口时序={' -> '.join(digest.window_sequence[:4])}")
        if digest.window_observations:
            signals.append(f"窗口观测={'; '.join(digest.window_observations[:3])}")
        if digest.semantic_tags:
            signals.append(f"语义标签={','.join(digest.semantic_tags)}")
        if digest.modality_hints:
            signals.append(f"模态提示={','.join(digest.modality_hints)}")
        signals.append(f"语义置信度={digest.semantic_confidence}")
        if digest.multimodal_summary:
            signals.append(f"视觉摘要={digest.multimodal_summary}")
        if digest.evidence_items:
            signals.append(f"证据清单={'; '.join(digest.evidence_items[:4])}")
        if digest.clipboard_seen:
            signals.append("发生过剪贴板变化")
        if digest.mouse_active:
            signals.append("存在鼠标活动")
        if digest.keyboard_active:
            signals.append("存在键盘修饰键活动")
        signal_text = "；".join(signals) if signals else "无明显交互信号"
        return (
            f"我观察到你刚才主要在 {app_text} 之间活动。"
            f"摘要: {digest.summary}。"
            f"语义理解: {digest.semantic_summary}。"
            f"附加信号: {signal_text}。"
            "我可以先给一个候选任务目标，你也可以直接改写。"
        )

    def maybe_create_proactive_question(self, event: BridgeEvent, state: BridgeState) -> tuple[str, str]:
        current_goal = state.current_goal.strip()
        if not current_goal:
            return "", "未设置当前目标，主动追问关闭。"
        if state.pending_question:
            return "", "已有待处理主动追问，暂不重复发问。"
        if self._recent_negative_feedback_too_high():
            return "", "近期负向反馈较多，主动追问已收缩。"
        if state.proactive_budget_remaining <= 0:
            return "", "主动追问预算已用尽。"

        last_prompt = self._parse_utc(state.last_proactive_prompt_utc)
        cooldown_s = int(self.proactive_config.get("cooldown_s", 120) or 120)
        if last_prompt is not None and datetime.now(timezone.utc) - last_prompt < timedelta(seconds=cooldown_s):
            return "", "主动追问仍在冷却中。"

        overlap = self._goal_overlap(current_goal, event)
        min_overlap = int(self.proactive_config.get("min_goal_overlap", 1) or 1)
        if overlap < min_overlap:
            return "", "这次桌面变化与当前目标相关性不足。"

        suggestion = self.suggest_goal_for_event(event)
        return (
            f"我注意到这次桌面变化可能与当前目标“{current_goal}”相关。"
            f"候选理解: {suggestion}。"
            "要不要按这个方向继续协助？",
            "已生成新的主动追问。",
        )

    def _goal_overlap(self, goal_text: str, event: BridgeEvent) -> int:
        goal_tokens = self._tokenize(goal_text)
        haystacks = [event.summary]
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        haystacks.extend(str(item) for item in metadata.get("top_apps", []) if str(item).strip())
        haystacks.extend(str(item) for item in metadata.get("event_types", []) if str(item).strip())

        overlap = 0
        for token in goal_tokens:
            if any(token and token in haystack.lower() for haystack in haystacks):
                overlap += 1
        return overlap

    def _tokenize(self, text: str) -> list[str]:
        lowered = text.lower()
        parts = re.findall(r"[a-z0-9_+#.\-]+|[\u4e00-\u9fff]{2,}", lowered)
        return [part for part in parts if len(part.strip()) >= 2]

    def _parse_utc(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _initial_budget(self) -> int:
        return int(self.proactive_config.get("initial_budget", 2) or 2)

    def _max_budget(self) -> int:
        return int(self.proactive_config.get("max_budget", 3) or 3)

    def _recent_negative_feedback_too_high(self) -> bool:
        feedback_window = int(self.proactive_config.get("feedback_window", 6) or 6)
        negative_feedback_limit = int(self.proactive_config.get("negative_feedback_limit", 2) or 2)
        recent_feedback = self.feedback_store.load_all()[-feedback_window:]
        negative_count = sum(1 for item in recent_feedback if item.feedback_type in {"dismiss", "misread"})
        useful_count = sum(1 for item in recent_feedback if item.feedback_type == "useful")
        return negative_count >= negative_feedback_limit and negative_count > useful_count
