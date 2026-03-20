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
    logic_summary: str = ""
    event_count: int = 0
    event_types: list[str] = field(default_factory=list)
    top_apps: list[str] = field(default_factory=list)
    focus_window: str = ""
    focus_process: str = ""
    focus_class_name: str = ""
    focus_pid: int = 0
    window_sequence: list[str] = field(default_factory=list)
    window_observations: list[str] = field(default_factory=list)
    semantic_tags: list[str] = field(default_factory=list)
    semantic_confidence: str = "low"
    modality_hints: list[str] = field(default_factory=list)
    reasoning_clues: list[str] = field(default_factory=list)
    evidence_items: list[str] = field(default_factory=list)
    multimodal_summary: str = ""
    multimodal_tags: list[str] = field(default_factory=list)
    multimodal_artifact_path: str = ""
    clipboard_seen: bool = False
    clipboard_preview: str = ""
    mouse_active: bool = False
    keyboard_active: bool = False
    modifier_keys: list[str] = field(default_factory=list)


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
        self._last_window_signature: str = ""
        self._last_clipboard: str = ""
        self._last_cursor: tuple[int, int] | None = None

    def poll(self) -> list[DesktopEvent]:
        events: list[DesktopEvent] = []
        window_snapshot = self._get_foreground_window_snapshot()
        window_signature = self._window_signature(window_snapshot)
        if window_snapshot and window_signature and window_signature != self._last_window_signature:
            self._last_window_signature = window_signature
            events.append(
                DesktopEvent(
                    timestamp_utc=self.now_fn(),
                    event_type="window_focus",
                    source="desktop/window",
                    payload=window_snapshot,
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

    def _get_foreground_window_snapshot(self) -> dict[str, Any]:
        if os.name != "nt":
            return {}
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return {}
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return {}
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value.strip()
        if not title:
            return {}

        class_buffer = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buffer, 256)
        class_name = class_buffer.value.strip()

        pid = ctypes.c_ulong(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        pid_value = int(pid.value)
        process_name = ""
        process_path = ""
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid_value)
        if handle:
            try:
                path_buffer_len = ctypes.c_ulong(1024)
                path_buffer = ctypes.create_unicode_buffer(path_buffer_len.value)
                if kernel32.QueryFullProcessImageNameW(handle, 0, path_buffer, ctypes.byref(path_buffer_len)):
                    process_path = path_buffer.value.strip()
                    process_name = Path(process_path).name if process_path else ""
            finally:
                kernel32.CloseHandle(handle)

        return {
            "title": title,
            "class_name": class_name,
            "pid": pid_value,
            "process_name": process_name,
            "process_path": process_path,
        }

    def _window_signature(self, snapshot: dict[str, Any]) -> str:
        if not snapshot:
            return ""
        title = str(snapshot.get("title", "")).strip()
        class_name = str(snapshot.get("class_name", "")).strip()
        process_name = str(snapshot.get("process_name", "")).strip()
        pid = str(snapshot.get("pid", "")).strip()
        return "|".join((title, process_name, class_name, pid))

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
        clipboard_seen = False
        clipboard_preview = ""
        mouse_active = False
        keyboard_active = False
        app_counts: dict[str, int] = {}
        focus_window = ""
        focus_process = ""
        focus_class_name = ""
        focus_pid = 0
        window_sequence: list[str] = []
        window_observations: list[str] = []
        seen_observations: set[str] = set()
        modifier_keys: set[str] = set()

        for event in events:
            if event.event_type == "window_focus":
                title = self._normalize_window_title(str(event.payload.get("title", "")).strip())
                if title:
                    app_counts[title] = app_counts.get(title, 0) + 1
                    focus_window = title
                    focus_process = str(event.payload.get("process_name", "")).strip()
                    focus_class_name = str(event.payload.get("class_name", "")).strip()
                    try:
                        focus_pid = int(event.payload.get("pid", 0) or 0)
                    except Exception:
                        focus_pid = 0
                    if not window_sequence or window_sequence[-1] != title:
                        window_sequence.append(title)
                    observation = self._format_window_observation(title, event.payload)
                    if observation and observation not in seen_observations:
                        seen_observations.add(observation)
                        window_observations.append(observation)
            elif event.event_type == "clipboard":
                clipboard_seen = True
                clipboard_preview = self._sanitize_clipboard_preview(str(event.payload.get("preview", "")))
            elif event.event_type == "input_activity":
                mouse_active = mouse_active or bool(event.payload.get("mouse_active"))
                keyboard_active = keyboard_active or bool(event.payload.get("keyboard_active"))
                raw_modifiers = event.payload.get("modifiers", {})
                if isinstance(raw_modifiers, dict):
                    for key, enabled in raw_modifiers.items():
                        if enabled:
                            modifier_keys.add(str(key))

        top_apps = [title for title, _ in sorted(app_counts.items(), key=lambda item: item[1], reverse=True)[:3]]
        summary_parts = [
            f"事件数={len(events)}",
            f"类型={','.join(event_types)}",
        ]
        if top_apps:
            summary_parts.append(f"主要窗口={'; '.join(top_apps)}")
        if focus_window:
            summary_parts.append(f"当前焦点={focus_window}")
        if focus_process:
            summary_parts.append(f"焦点进程={focus_process}")
        if focus_class_name:
            summary_parts.append(f"焦点类名={focus_class_name}")
        if focus_pid:
            summary_parts.append(f"焦点PID={focus_pid}")
        if window_sequence:
            summary_parts.append(f"窗口序列={' -> '.join(window_sequence[:4])}")
        if window_observations:
            summary_parts.append(f"窗口观测={'; '.join(window_observations[:3])}")
        if clipboard_seen:
            summary_parts.append("检测到剪贴板变化")
        if clipboard_preview:
            summary_parts.append(f"剪贴板预览={clipboard_preview}")
        if mouse_active:
            summary_parts.append("检测到鼠标活动")
        if keyboard_active:
            summary_parts.append("检测到键盘修饰键活动")
        if modifier_keys:
            summary_parts.append(f"修饰键={'+'.join(sorted(modifier_keys))}")

        semantic_summary, semantic_tags, semantic_confidence, modality_hints = self._infer_semantic_summary(
            top_apps=top_apps,
            focus_window=focus_window,
            focus_process=focus_process,
            focus_class_name=focus_class_name,
            focus_pid=focus_pid,
            window_sequence=window_sequence,
            window_observations=window_observations,
            clipboard_seen=clipboard_seen,
            clipboard_preview=clipboard_preview,
            mouse_active=mouse_active,
            keyboard_active=keyboard_active,
            modifier_keys=sorted(modifier_keys),
            event_types=event_types,
        )
        logic_summary, reasoning_clues, evidence_items = self._build_logic_summary(
            top_apps=top_apps,
            focus_window=focus_window,
            focus_process=focus_process,
            focus_class_name=focus_class_name,
            focus_pid=focus_pid,
            window_sequence=window_sequence,
            window_observations=window_observations,
            semantic_tags=semantic_tags,
            event_types=event_types,
            clipboard_seen=clipboard_seen,
            clipboard_preview=clipboard_preview,
            mouse_active=mouse_active,
            keyboard_active=keyboard_active,
            modifier_keys=sorted(modifier_keys),
        )

        return DesktopDigest(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            summary="；".join(summary_parts),
            semantic_summary=semantic_summary,
            logic_summary=logic_summary,
            event_count=len(events),
            event_types=event_types,
            top_apps=top_apps,
            focus_window=focus_window,
            focus_process=focus_process,
            focus_class_name=focus_class_name,
            focus_pid=focus_pid,
            window_sequence=window_sequence,
            window_observations=window_observations,
            semantic_tags=semantic_tags,
            semantic_confidence=semantic_confidence,
            modality_hints=modality_hints,
            reasoning_clues=reasoning_clues,
            evidence_items=evidence_items,
            clipboard_seen=clipboard_seen,
            clipboard_preview=clipboard_preview,
            mouse_active=mouse_active,
            keyboard_active=keyboard_active,
            modifier_keys=sorted(modifier_keys),
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
        return cleaned

    def _format_window_anchor(self, title: str) -> str:
        lowered = title.lower()
        if " - visual studio code" in lowered:
            subject = title.rsplit(" - ", 1)[0].strip()
            return f"Visual Studio Code（{subject}）" if subject else "Visual Studio Code"
        if " - google chrome" in lowered:
            subject = title.rsplit(" - ", 1)[0].strip()
            return f"Google Chrome（{subject}）" if subject else "Google Chrome"
        if " - microsoft edge" in lowered:
            subject = title.rsplit(" - ", 1)[0].strip()
            return f"Microsoft Edge（{subject}）" if subject else "Microsoft Edge"
        if " - 文件资源管理器" in title:
            subject = title.rsplit(" - ", 1)[0].strip()
            return f"文件资源管理器（{subject}）" if subject else "文件资源管理器"
        if title == "CARM Bridge":
            return "CARM Bridge"
        return title

    def _select_semantic_anchors(self, top_apps: list[str]) -> tuple[list[str], list[str]]:
        primary: list[str] = []
        secondary: list[str] = []
        for title in top_apps:
            anchor = self._format_window_anchor(title)
            lowered = anchor.lower()
            if "carm bridge" in lowered or "文件资源管理器" in anchor:
                secondary.append(anchor)
            else:
                primary.append(anchor)
        return primary[:2], secondary[:2]

    def _sanitize_clipboard_preview(self, text: str) -> str:
        preview = " ".join(text.split())
        if len(preview) > 80:
            return f"{preview[:77]}..."
        return preview

    def _format_window_observation(self, title: str, payload: dict[str, Any]) -> str:
        anchor = self._format_window_anchor(title)
        parts: list[str] = [anchor]
        process_name = str(payload.get("process_name", "")).strip()
        class_name = str(payload.get("class_name", "")).strip()
        try:
            pid = int(payload.get("pid", 0) or 0)
        except Exception:
            pid = 0
        if process_name:
            parts.append(f"进程={process_name}")
        if class_name:
            parts.append(f"类名={class_name}")
        if pid:
            parts.append(f"PID={pid}")
        return " | ".join(parts)

    def _infer_semantic_summary(
        self,
        *,
        top_apps: list[str],
        focus_window: str,
        focus_process: str,
        focus_class_name: str,
        focus_pid: int,
        window_sequence: list[str],
        window_observations: list[str],
        clipboard_seen: bool,
        clipboard_preview: str,
        mouse_active: bool,
        keyboard_active: bool,
        modifier_keys: list[str],
        event_types: list[str],
    ) -> tuple[str, list[str], str, list[str]]:
        lowered_apps = [item.lower() for item in top_apps]
        primary_anchors, secondary_anchors = self._select_semantic_anchors(top_apps)
        anchor_text = "、".join(primary_anchors or secondary_anchors)
        focus_anchor = self._format_window_anchor(focus_window) if focus_window else ""
        is_code = any(any(token in app for token in ("vs code", "code", "cursor", "pycharm", "idea", "visual studio")) for app in lowered_apps)
        is_browser = any(any(token in app for token in ("chrome", "edge", "firefox", "browser")) for app in lowered_apps)
        is_doc = any(any(token in app for token in ("word", "docs", "notion", "obsidian")) for app in lowered_apps)
        is_sheet = any(any(token in app for token in ("excel", "sheet", "wps 表格")) for app in lowered_apps)
        is_terminal = any(any(token in app for token in ("terminal", "powershell", "cmd")) for app in lowered_apps)
        is_chat = any(any(token in app for token in ("wechat", "qq", "feishu", "slack", "discord")) for app in lowered_apps)
        semantic_tags: list[str] = []
        modality_hints = ["desktop"]

        if is_code and is_browser:
            action = "代码编辑与网页检索"
            semantic_tags.extend(["coding", "research"])
        elif is_code and is_terminal:
            action = "代码编辑与命令调试"
            semantic_tags.extend(["coding", "terminal"])
        elif is_code:
            action = "代码编辑"
            semantic_tags.append("coding")
        elif is_sheet:
            action = "表格处理"
            semantic_tags.append("spreadsheet")
        elif is_doc:
            action = "文档阅读或编辑"
            semantic_tags.append("document")
        elif is_browser:
            action = "网页检索或浏览"
            semantic_tags.append("research")
        elif is_chat:
            action = "聊天或协作沟通"
            semantic_tags.append("communication")
        elif top_apps:
            action = "桌面任务处理"
            semantic_tags.append("desktop_task")
        else:
            action = "桌面活动"
            semantic_tags.append("desktop_task")

        detail_parts: list[str] = [f"场景标签={action}"]
        if anchor_text:
            detail_parts.append(f"关键窗口={anchor_text}")
        if focus_anchor:
            detail_parts.append(f"当前焦点={focus_anchor}")
        if focus_process:
            detail_parts.append(f"焦点进程={focus_process}")
        if focus_class_name:
            detail_parts.append(f"焦点类名={focus_class_name}")
        if focus_pid:
            detail_parts.append(f"焦点PID={focus_pid}")
        if len(window_sequence) >= 2:
            sequence_text = " -> ".join(self._format_window_anchor(item) for item in window_sequence[:4])
            detail_parts.append(f"窗口时序={sequence_text}")
        if window_observations:
            detail_parts.append(f"窗口观测={'; '.join(window_observations[:2])}")
        if clipboard_seen:
            detail_parts.append("观察到剪贴板变化")
            modality_hints.append("text")
        if clipboard_preview:
            detail_parts.append(f"剪贴板文本={clipboard_preview}")
        if keyboard_active:
            detail_parts.append("观察到键盘修饰键活动")
        if modifier_keys:
            detail_parts.append(f"修饰键={'+'.join(modifier_keys)}")
        if mouse_active:
            detail_parts.append("观察到鼠标活动")
        if "window_focus" in event_types and len(top_apps) >= 2:
            switched = "、".join((primary_anchors + secondary_anchors)[:3])
            if switched:
                detail_parts.append(f"窗口切换={switched}")
            else:
                detail_parts.append("观察到窗口切换")
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
            return f"观察摘要：{detail}。", semantic_tags, semantic_confidence, modality_hints
        return f"观察摘要：场景标签={action}。", semantic_tags, semantic_confidence, modality_hints

    def _build_logic_summary(
        self,
        *,
        top_apps: list[str],
        focus_window: str,
        focus_process: str,
        focus_class_name: str,
        focus_pid: int,
        window_sequence: list[str],
        window_observations: list[str],
        semantic_tags: list[str],
        event_types: list[str],
        clipboard_seen: bool,
        clipboard_preview: str,
        mouse_active: bool,
        keyboard_active: bool,
        modifier_keys: list[str],
    ) -> tuple[str, list[str], list[str]]:
        primary_anchors, secondary_anchors = self._select_semantic_anchors(top_apps)
        anchors = primary_anchors + secondary_anchors
        facts: list[str] = []
        evidence: list[str] = []
        relations: list[str] = []
        clues: list[str] = []
        focus_anchor = self._format_window_anchor(focus_window) if focus_window else ""
        sequence_anchors = [self._format_window_anchor(item) for item in window_sequence]

        if anchors:
            facts.append(f"窗口={'; '.join(anchors)}")
        if window_observations:
            facts.append(f"窗口观测={'; '.join(window_observations[:3])}")
            evidence.extend(window_observations[:3])
        if focus_anchor:
            facts.append(f"当前焦点={focus_anchor}")
            evidence.append(f"当前焦点={focus_anchor}")
        if focus_process:
            facts.append(f"焦点进程={focus_process}")
            evidence.append(f"焦点进程={focus_process}")
        if focus_class_name:
            facts.append(f"焦点类名={focus_class_name}")
            evidence.append(f"焦点类名={focus_class_name}")
        if focus_pid:
            facts.append(f"焦点PID={focus_pid}")
            evidence.append(f"焦点PID={focus_pid}")
        if sequence_anchors:
            facts.append(f"窗口时序={' -> '.join(sequence_anchors[:5])}")
            evidence.append(f"窗口时序={' -> '.join(sequence_anchors[:5])}")
        if event_types:
            facts.append(f"事件={','.join(event_types)}")
        if semantic_tags:
            facts.append(f"标签={','.join(semantic_tags)}")

        signals: list[str] = []
        if clipboard_seen:
            signals.append("clipboard_changed")
            clues.append("存在文本复制或整理行为")
            evidence.append("观察到剪贴板变化")
        if mouse_active:
            signals.append("mouse_active")
            clues.append("存在界面点击或焦点切换")
        if keyboard_active:
            signals.append("modifier_key_active")
            clues.append("存在快捷键或组合键操作")
        if clipboard_preview:
            facts.append(f"剪贴板预览={clipboard_preview}")
            evidence.append(f"剪贴板预览={clipboard_preview}")
        if modifier_keys:
            facts.append(f"修饰键={'+'.join(modifier_keys)}")
            evidence.append(f"修饰键={'+'.join(modifier_keys)}")
        if signals:
            facts.append(f"信号={','.join(signals)}")

        if len(anchors) >= 2:
            relations.append(f"切换关系={anchors[0]} <-> {anchors[1]}")
            relations.append(f"主要工作窗口={anchors[0]}")
            relations.append(f"参考窗口={anchors[1]}")
            clues.append(f"任务可能跨窗口协同，核心窗口为 {anchors[0]} 与 {anchors[1]}")
        elif len(anchors) == 1:
            relations.append(f"焦点集中={anchors[0]}")
            clues.append(f"当前任务主要集中在 {anchors[0]}")
        else:
            relations.append("焦点窗口不足")

        if len(sequence_anchors) >= 2:
            transition = " -> ".join(sequence_anchors[:4])
            relations.append(f"最近切换序列={transition}")
            clues.append(f"最近操作顺序为 {transition}")

        if "coding" in semantic_tags and "research" in semantic_tags:
            clues.append("代码编辑与资料检索同时存在，可视为问题求解链")
        elif "coding" in semantic_tags:
            clues.append("当前观测更接近代码理解或代码修改")
        elif "research" in semantic_tags:
            clues.append("当前观测更接近信息检索或网页阅读")
        elif "document" in semantic_tags:
            clues.append("当前观测更接近文档编辑或内容整理")

        logic_summary = "；".join(
            [
                f"事实: {' | '.join(facts) if facts else '无'}",
                f"证据: {' | '.join(evidence) if evidence else '无'}",
                f"关系: {' | '.join(relations) if relations else '无'}",
                f"推理线索: {' | '.join(clues) if clues else '无'}",
            ]
        )
        return logic_summary, clues, evidence


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
            f"逻辑摘要={digest.logic_summary}，"
            f"当前焦点={digest.focus_window or '无'}，"
            f"焦点进程={digest.focus_process or '无'}，"
            f"焦点类名={digest.focus_class_name or '无'}，"
            f"焦点PID={digest.focus_pid or 0}，"
            f"窗口时序={'; '.join(digest.window_sequence) if digest.window_sequence else '无'}，"
            f"窗口观测={'; '.join(digest.window_observations) if digest.window_observations else '无'}，"
            f"语义标签={','.join(digest.semantic_tags) if digest.semantic_tags else '无'}，"
            f"语义置信度={digest.semantic_confidence}，"
            f"模态提示={','.join(digest.modality_hints) if digest.modality_hints else 'desktop'}，"
            f"推理线索={'; '.join(digest.reasoning_clues) if digest.reasoning_clues else '无'}，"
            f"证据清单={'; '.join(digest.evidence_items) if digest.evidence_items else '无'}，"
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
