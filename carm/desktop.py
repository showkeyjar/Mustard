from __future__ import annotations

import ctypes
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from carm.runner import AgentRunner

if TYPE_CHECKING:
    from carm.bridge import DesktopBridgeController


@dataclass
class DesktopEvent:
    timestamp_utc: str
    event_type: str
    source: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class DesktopDigest:
    timestamp_utc: str
    summary: str
    semantic_summary: str
    event_count: int
    event_types: list[str] = field(default_factory=list)
    top_apps: list[str] = field(default_factory=list)
    semantic_tags: list[str] = field(default_factory=list)
    semantic_confidence: str = "low"
    modality_hints: list[str] = field(default_factory=list)
    multimodal_summary: str = ""
    multimodal_tags: list[str] = field(default_factory=list)
    multimodal_artifact_path: str = ""
    clipboard_seen: bool = False
    mouse_active: bool = False
    keyboard_active: bool = False


class DesktopEventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: DesktopEvent) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def load_recent(self, limit: int = 200) -> list[DesktopEvent]:
        if not self.path.exists():
            return []
        lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        recent = lines[-limit:]
        events: list[DesktopEvent] = []
        for line in recent:
            payload = json.loads(line)
            if isinstance(payload, dict):
                events.append(DesktopEvent(**payload))
        return events


class DesktopDigestStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, digest: DesktopDigest) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(digest), ensure_ascii=False) + "\n")


class WindowsDesktopObserver:
    def __init__(
        self,
        clipboard_reader: Callable[[], str] | None = None,
        now_fn: Callable[[], str] | None = None,
    ) -> None:
        self.clipboard_reader = clipboard_reader or self._read_clipboard
        self.now_fn = now_fn or self._utc_now
        self._last_window: str = ""
        self._last_clipboard: str = ""
        self._last_cursor: tuple[int, int] | None = None

    def poll(self) -> list[DesktopEvent]:
        events: list[DesktopEvent] = []
        window_title = self._get_foreground_window_title()
        if window_title and window_title != self._last_window:
            self._last_window = window_title
            events.append(
                DesktopEvent(
                    timestamp_utc=self.now_fn(),
                    event_type="window_focus",
                    source="desktop/window",
                    payload={"title": window_title},
                )
            )

        clipboard_text = self.clipboard_reader()
        if clipboard_text and clipboard_text != self._last_clipboard:
            self._last_clipboard = clipboard_text
            events.append(
                DesktopEvent(
                    timestamp_utc=self.now_fn(),
                    event_type="clipboard",
                    source="desktop/clipboard",
                    payload={
                        "preview": clipboard_text[:160],
                        "length": len(clipboard_text),
                    },
                )
            )

        activity = self._get_input_activity()
        if activity:
            events.append(
                DesktopEvent(
                    timestamp_utc=self.now_fn(),
                    event_type="input_activity",
                    source="desktop/input",
                    payload=activity,
                )
            )
        return events

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _get_foreground_window_title(self) -> str:
        if os.name != "nt":
            return ""
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value.strip()

    def _read_clipboard(self) -> str:
        if os.name != "nt":
            return ""
        CF_UNICODETEXT = 13
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        if not user32.OpenClipboard(None):
            return ""
        try:
            handle = user32.GetClipboardData(CF_UNICODETEXT)
            if not handle:
                return ""
            pointer = kernel32.GlobalLock(handle)
            if not pointer:
                return ""
            try:
                text = ctypes.wstring_at(pointer)
            finally:
                kernel32.GlobalUnlock(handle)
            return text.strip() if text else ""
        except Exception:
            return ""
        finally:
            user32.CloseClipboard()

    def _get_input_activity(self) -> dict[str, Any]:
        if os.name != "nt":
            return {}

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if not user32.GetLastInputInfo(ctypes.byref(info)):
            return {}

        current_tick = kernel32.GetTickCount()
        idle_ms = int(current_tick - info.dwTime)

        point = POINT()
        if not user32.GetCursorPos(ctypes.byref(point)):
            return {}

        cursor = (int(point.x), int(point.y))
        moved = False
        if self._last_cursor is not None:
            moved = cursor != self._last_cursor
        self._last_cursor = cursor

        left_pressed = bool(user32.GetAsyncKeyState(0x01) & 0x8000)
        right_pressed = bool(user32.GetAsyncKeyState(0x02) & 0x8000)
        shift_pressed = bool(user32.GetAsyncKeyState(0x10) & 0x8000)
        ctrl_pressed = bool(user32.GetAsyncKeyState(0x11) & 0x8000)
        alt_pressed = bool(user32.GetAsyncKeyState(0x12) & 0x8000)

        return {
            "idle_ms": idle_ms,
            "cursor": {"x": cursor[0], "y": cursor[1]},
            "cursor_moved": moved,
            "mouse_active": left_pressed or right_pressed or moved,
            "keyboard_active": shift_pressed or ctrl_pressed or alt_pressed,
            "modifiers": {
                "shift": shift_pressed,
                "ctrl": ctrl_pressed,
                "alt": alt_pressed,
            },
        }


class DesktopSummarizer:
    _NOISE_TITLES = (
        "tasklist.exe",
        "系统托盘溢出窗口",
        "program manager",
        "default ime",
        "msctfime ui",
    )

    def summarize(self, events: list[DesktopEvent]) -> DesktopDigest | None:
        if not events:
            return None

        event_types = sorted({event.event_type for event in events})
        top_apps: list[str] = []
        clipboard_seen = False
        mouse_active = False
        keyboard_active = False
        app_counts: dict[str, int] = {}

        for event in events:
            if event.event_type == "window_focus":
                title = self._normalize_window_title(str(event.payload.get("title", "")).strip())
                if title:
                    app_counts[title] = app_counts.get(title, 0) + 1
            elif event.event_type == "clipboard":
                clipboard_seen = True
            elif event.event_type == "input_activity":
                mouse_active = mouse_active or bool(event.payload.get("mouse_active"))
                keyboard_active = keyboard_active or bool(event.payload.get("keyboard_active"))

        top_apps = [title for title, _ in sorted(app_counts.items(), key=lambda item: item[1], reverse=True)[:3]]
        summary_parts = [
            f"事件数={len(events)}",
            f"类型={','.join(event_types)}",
        ]
        if top_apps:
            summary_parts.append(f"主要窗口={'; '.join(top_apps)}")
        if clipboard_seen:
            summary_parts.append("检测到剪贴板变化")
        if mouse_active:
            summary_parts.append("检测到鼠标活动")
        if keyboard_active:
            summary_parts.append("检测到键盘修饰键活动")

        semantic_summary, semantic_tags, semantic_confidence, modality_hints = self._infer_semantic_summary(
            top_apps=top_apps,
            clipboard_seen=clipboard_seen,
            mouse_active=mouse_active,
            keyboard_active=keyboard_active,
            event_types=event_types,
        )

        return DesktopDigest(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            summary="；".join(summary_parts),
            semantic_summary=semantic_summary,
            event_count=len(events),
            event_types=event_types,
            top_apps=top_apps,
            semantic_tags=semantic_tags,
            semantic_confidence=semantic_confidence,
            modality_hints=modality_hints,
            clipboard_seen=clipboard_seen,
            mouse_active=mouse_active,
            keyboard_active=keyboard_active,
        )

    def _normalize_window_title(self, title: str) -> str:
        cleaned = title.strip()
        if not cleaned:
            return ""
        lowered = cleaned.lower()
        if any(noise in lowered for noise in self._NOISE_TITLES):
            return ""
        if lowered.endswith(".exe") and ("\\" in cleaned or "/" in cleaned):
            cleaned = Path(cleaned).name
            lowered = cleaned.lower()
            if any(noise in lowered for noise in self._NOISE_TITLES):
                return ""
        separators = [" - ", " — ", " | "]
        for separator in separators:
            if separator in cleaned:
                parts = [part.strip() for part in cleaned.split(separator) if part.strip()]
                if not parts:
                    return ""
                if any(token in parts[-1].lower() for token in ("chrome", "edge", "firefox", "code", "cursor", "pycharm", "excel", "word")):
                    return parts[-1]
                return parts[0]
        return cleaned

    def _infer_semantic_summary(
        self,
        *,
        top_apps: list[str],
        clipboard_seen: bool,
        mouse_active: bool,
        keyboard_active: bool,
        event_types: list[str],
    ) -> tuple[str, list[str], str, list[str]]:
        lowered_apps = [item.lower() for item in top_apps]
        is_code = any(any(token in app for token in ("vs code", "code", "cursor", "pycharm", "idea", "visual studio")) for app in lowered_apps)
        is_browser = any(any(token in app for token in ("chrome", "edge", "firefox", "browser")) for app in lowered_apps)
        is_doc = any(any(token in app for token in ("word", "docs", "notion", "obsidian")) for app in lowered_apps)
        is_sheet = any(any(token in app for token in ("excel", "sheet", "wps 表格")) for app in lowered_apps)
        is_terminal = any(any(token in app for token in ("terminal", "powershell", "cmd")) for app in lowered_apps)
        is_chat = any(any(token in app for token in ("wechat", "qq", "feishu", "slack", "discord")) for app in lowered_apps)
        semantic_tags: list[str] = []
        modality_hints = ["desktop"]

        if is_code and is_browser:
            action = "查资料并修改代码"
            semantic_tags.extend(["coding", "research"])
        elif is_code and is_terminal:
            action = "编写代码并执行或调试命令"
            semantic_tags.extend(["coding", "terminal"])
        elif is_code:
            action = "编写或阅读代码"
            semantic_tags.append("coding")
        elif is_sheet:
            action = "查看或整理表格数据"
            semantic_tags.append("spreadsheet")
        elif is_doc:
            action = "阅读或编辑文档内容"
            semantic_tags.append("document")
        elif is_browser:
            action = "浏览网页或检索信息"
            semantic_tags.append("research")
        elif is_chat:
            action = "处理聊天消息或协作沟通"
            semantic_tags.append("communication")
        elif top_apps:
            action = f"处理与 {'、'.join(top_apps[:2])} 相关的任务"
            semantic_tags.append("desktop_task")
        else:
            action = "处理当前桌面任务"
            semantic_tags.append("desktop_task")

        detail_parts: list[str] = []
        if clipboard_seen:
            detail_parts.append("期间复制或整理过文本")
            modality_hints.append("text")
        if keyboard_active:
            detail_parts.append("有持续键盘操作")
        if mouse_active:
            detail_parts.append("有交互点击或切换")
        if "window_focus" in event_types and len(top_apps) >= 2:
            detail_parts.append("在多个窗口之间切换")
        if is_browser or is_doc or is_sheet:
            modality_hints.append("image")
        if is_chat:
            modality_hints.append("text")

        detail = "，".join(detail_parts)
        modality_hints = sorted(set(modality_hints))
        semantic_tags = sorted(set(semantic_tags))
        confidence_score = 0
        if top_apps:
            confidence_score += 1
        if semantic_tags and semantic_tags != ["desktop_task"]:
            confidence_score += 1
        if detail_parts:
            confidence_score += 1
        if clipboard_seen:
            confidence_score += 1
        semantic_confidence = "high" if confidence_score >= 3 else "medium" if confidence_score == 2 else "low"
        if detail:
            return f"用户可能正在{action}，{detail}。", semantic_tags, semantic_confidence, modality_hints
        return f"用户可能正在{action}。", semantic_tags, semantic_confidence, modality_hints


class DesktopLearner:
    def __init__(
        self,
        runner: AgentRunner,
        event_store: DesktopEventStore,
        digest_store: DesktopDigestStore,
        summarizer: DesktopSummarizer | None = None,
        bridge_controller: DesktopBridgeController | None = None,
        digest_enricher: Callable[[DesktopDigest], DesktopDigest] | None = None,
    ) -> None:
        self.runner = runner
        self.event_store = event_store
        self.digest_store = digest_store
        self.summarizer = summarizer or DesktopSummarizer()
        self.bridge_controller = bridge_controller
        self.digest_enricher = digest_enricher

    def observe(self, events: list[DesktopEvent]) -> DesktopDigest | None:
        for event in events:
            self.event_store.append(event)
        digest = self.summarizer.summarize(events)
        if digest is None:
            return None
        if self.digest_enricher is not None:
            digest = self.digest_enricher(digest)
        self.digest_store.append(digest)
        if self.bridge_controller is not None:
            self.bridge_controller.ingest_digest(digest)
        learning_prompt = self._build_learning_prompt(digest)
        self.runner.run(learning_prompt)
        return digest

    def _build_learning_prompt(self, digest: DesktopDigest) -> str:
        return (
            "观察学习任务: 基于用户桌面活动摘要更新经验偏好，"
            f"事件摘要={digest.summary}，"
            f"语义摘要={digest.semantic_summary}，"
            f"语义标签={','.join(digest.semantic_tags) if digest.semantic_tags else '无'}，"
            f"语义置信度={digest.semantic_confidence}，"
            f"模态提示={','.join(digest.modality_hints) if digest.modality_hints else 'desktop'}，"
            f"多模态摘要={digest.multimodal_summary or '无'}，"
            f"多模态标签={','.join(digest.multimodal_tags) if digest.multimodal_tags else '无'}，"
            f"窗口={'; '.join(digest.top_apps) if digest.top_apps else '无'}，"
            f"事件类型={','.join(digest.event_types)}"
        )


class DesktopAgentService:
    def __init__(
        self,
        observer: WindowsDesktopObserver,
        learner: DesktopLearner,
        poll_interval_s: float = 2.0,
        digest_interval_s: float = 30.0,
        max_buffer_events: int = 64,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.observer = observer
        self.learner = learner
        self.poll_interval_s = poll_interval_s
        self.digest_interval_s = digest_interval_s
        self.max_buffer_events = max_buffer_events
        self.sleep_fn = sleep_fn or time.sleep

    def run_forever(self) -> None:
        buffer: list[DesktopEvent] = []
        last_digest = time.monotonic()
        while True:
            buffer.extend(self.observer.poll())
            now = time.monotonic()
            if buffer and (now - last_digest >= self.digest_interval_s or len(buffer) >= self.max_buffer_events):
                self.learner.observe(buffer)
                buffer = []
                last_digest = now
            self.sleep_fn(self.poll_interval_s)
