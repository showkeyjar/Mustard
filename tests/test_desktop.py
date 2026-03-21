import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from carm.bridge import BridgeEventStore, BridgeFeedbackStore, BridgeMessageStore, BridgeStateStore, DesktopBridgeController
from carm.desktop import (
    DesktopAgentService,
    DesktopDigestStore,
    DesktopDigest,
    DesktopEvent,
    DesktopEventStore,
    DesktopLearner,
    DesktopSummarizer,
    WindowsDesktopObserver,
)
from carm.desktop_runtime import (
    DesktopRuntimeStatus,
    DesktopAgentProcessManager,
    build_bridge_chat_command,
    build_tray_python_command,
    build_startup_shortcut_script,
    format_status_snapshot,
    launch_desktop_bridge,
    load_bridge_state_summary,
    status_payload,
)
from carm.multimodal import MultimodalAdapter, ScreenObservationAdapter


class FakeRunner:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.signals: list[object] = []

    def run(self, prompt: str):
        self.prompts.append(prompt)
        return "ok", None

    def apply_user_signal(self, signal) -> None:
        self.signals.append(signal)


class FakeObserver:
    def __init__(self, batches):
        self.batches = list(batches)

    def poll(self):
        if self.batches:
            return self.batches.pop(0)
        raise KeyboardInterrupt


class DesktopTests(unittest.TestCase):
    def test_summarizer_extracts_basic_signals(self) -> None:
        summarizer = DesktopSummarizer()
        digest = summarizer.summarize(
            [
                DesktopEvent("2026-03-20T00:00:00+00:00", "window_focus", "desktop/window", {"title": "VS Code"}),
                DesktopEvent("2026-03-20T00:00:01+00:00", "clipboard", "desktop/clipboard", {"preview": "abc", "length": 3}),
                DesktopEvent(
                    "2026-03-20T00:00:02+00:00",
                    "input_activity",
                    "desktop/input",
                    {"mouse_active": True, "keyboard_active": True},
                ),
            ]
        )
        self.assertIsNotNone(digest)
        assert digest is not None
        self.assertTrue(digest.clipboard_seen)
        self.assertTrue(digest.mouse_active)
        self.assertTrue(digest.keyboard_active)
        self.assertIn("VS Code", digest.top_apps)
        self.assertEqual(digest.focus_window, "VS Code")
        self.assertEqual(digest.window_observations, ["VS Code"])
        self.assertIn("观察摘要：", digest.semantic_summary)
        self.assertIn("事实:", digest.logic_summary)
        self.assertIn("证据:", digest.logic_summary)
        self.assertIn("推理线索:", digest.logic_summary)
        self.assertIn("coding", digest.semantic_tags)
        self.assertEqual(digest.semantic_confidence, "high")
        self.assertIn("desktop", digest.modality_hints)

    def test_summarizer_filters_noise_windows_and_normalizes_titles(self) -> None:
        summarizer = DesktopSummarizer()
        digest = summarizer.summarize(
            [
                DesktopEvent("2026-03-20T00:00:00+00:00", "window_focus", "desktop/window", {"title": "C:\\WINDOWS\\SYSTEM32\\tasklist.exe"}),
                DesktopEvent("2026-03-20T00:00:01+00:00", "window_focus", "desktop/window", {"title": "main.py - Visual Studio Code"}),
                DesktopEvent("2026-03-20T00:00:02+00:00", "window_focus", "desktop/window", {"title": "系统托盘溢出窗口。"}),
            ]
        )
        assert digest is not None
        self.assertEqual(digest.top_apps, ["main.py - Visual Studio Code"])
        self.assertNotIn("tasklist.exe", digest.summary.lower())

    def test_summarizer_includes_specific_window_anchors(self) -> None:
        summarizer = DesktopSummarizer()
        digest = summarizer.summarize(
            [
                DesktopEvent(
                    "2026-03-20T00:00:00+00:00",
                    "window_focus",
                    "desktop/window",
                    {
                        "title": "README.md - Mustard - Visual Studio Code",
                        "process_name": "Code.exe",
                        "class_name": "Chrome_WidgetWin_1",
                        "pid": 4242,
                    },
                ),
                DesktopEvent(
                    "2026-03-20T00:00:01+00:00",
                    "window_focus",
                    "desktop/window",
                    {
                        "title": "碳寻大模型与涡动通量AI技术交流 - Google Chrome",
                        "process_name": "chrome.exe",
                        "class_name": "Chrome_WidgetWin_1",
                        "pid": 5252,
                    },
                ),
                DesktopEvent(
                    "2026-03-20T00:00:02+00:00",
                    "input_activity",
                    "desktop/input",
                    {"mouse_active": True, "keyboard_active": False},
                ),
            ]
        )
        assert digest is not None
        self.assertEqual(digest.focus_process, "chrome.exe")
        self.assertEqual(digest.focus_class_name, "Chrome_WidgetWin_1")
        self.assertEqual(digest.focus_pid, 5252)
        self.assertIn("Visual Studio Code（README.md - Mustard）", digest.semantic_summary)
        self.assertIn("Google Chrome（碳寻大模型与涡动通量AI技术交流）", digest.semantic_summary)
        self.assertIn("进程=Code.exe", digest.semantic_summary)
        self.assertIn("窗口时序=", digest.semantic_summary)
        self.assertIn("最近切换序列=", digest.logic_summary)

    def test_summarizer_preserves_reasoning_ready_evidence(self) -> None:
        summarizer = DesktopSummarizer()
        digest = summarizer.summarize(
            [
                DesktopEvent(
                    "2026-03-20T00:00:00+00:00",
                    "window_focus",
                    "desktop/window",
                    {
                        "title": "README.md - Mustard - Visual Studio Code",
                        "process_name": "Code.exe",
                        "class_name": "Chrome_WidgetWin_1",
                        "pid": 4242,
                    },
                ),
                DesktopEvent(
                    "2026-03-20T00:00:01+00:00",
                    "window_focus",
                    "desktop/window",
                    {
                        "title": "碳寻大模型与涡动通量AI技术交流 - Google Chrome",
                        "process_name": "chrome.exe",
                        "class_name": "Chrome_WidgetWin_1",
                        "pid": 5252,
                    },
                ),
                DesktopEvent(
                    "2026-03-20T00:00:02+00:00",
                    "clipboard",
                    "desktop/clipboard",
                    {"preview": "PostgreSQL 与 MySQL 对比方案", "length": 23},
                ),
                DesktopEvent(
                    "2026-03-20T00:00:03+00:00",
                    "input_activity",
                    "desktop/input",
                    {
                        "mouse_active": True,
                        "keyboard_active": True,
                        "modifiers": {"ctrl": True, "shift": False, "alt": False},
                    },
                ),
            ]
        )
        assert digest is not None
        self.assertEqual(
            digest.window_sequence,
            [
                "README.md - Mustard - Visual Studio Code",
                "碳寻大模型与涡动通量AI技术交流 - Google Chrome",
            ],
        )
        self.assertEqual(digest.clipboard_preview, "PostgreSQL 与 MySQL 对比方案")
        self.assertEqual(digest.modifier_keys, ["ctrl"])
        self.assertTrue(any("进程=Code.exe" in item for item in digest.window_observations))
        self.assertIn("剪贴板预览=PostgreSQL 与 MySQL 对比方案", digest.logic_summary)
        self.assertIn("修饰键=ctrl", digest.logic_summary)
        self.assertIn("焦点进程=chrome.exe", digest.logic_summary)
        self.assertTrue(any("窗口时序=" in item for item in digest.evidence_items))

    def test_observer_uses_injected_clipboard_reader(self) -> None:
        observer = WindowsDesktopObserver(
            clipboard_reader=lambda: "中文剪贴板",
            now_fn=lambda: "2026-03-20T00:00:00+00:00",
        )
        events = observer.poll()
        clipboard_events = [event for event in events if event.event_type == "clipboard"]
        self.assertEqual(len(clipboard_events), 1)
        self.assertEqual(clipboard_events[0].payload["preview"], "中文剪贴板")

    def test_learner_persists_digest_and_calls_runner(self) -> None:
        with TemporaryDirectory() as temp_dir:
            event_store = DesktopEventStore(Path(temp_dir) / "events.jsonl")
            digest_store = DesktopDigestStore(Path(temp_dir) / "digests.jsonl")
            runner = FakeRunner()
            bridge = DesktopBridgeController(
                runner,
                BridgeEventStore(Path(temp_dir) / "bridge_events.jsonl"),
                BridgeFeedbackStore(Path(temp_dir) / "bridge_feedback.jsonl"),
                BridgeMessageStore(Path(temp_dir) / "bridge_messages.jsonl"),
                BridgeStateStore(Path(temp_dir) / "bridge_state.json"),
            )
            learner = DesktopLearner(runner, event_store, digest_store, bridge_controller=bridge)

            digest = learner.observe(
                [
                    DesktopEvent("2026-03-20T00:00:00+00:00", "window_focus", "desktop/window", {"title": "Chrome"}),
                ]
            )

            self.assertIsNotNone(digest)
            self.assertTrue((Path(temp_dir) / "events.jsonl").exists())
            self.assertTrue((Path(temp_dir) / "digests.jsonl").exists())
            self.assertTrue((Path(temp_dir) / "bridge_events.jsonl").exists())
            self.assertEqual(len(runner.prompts), 1)
            self.assertIn("观察学习任务", runner.prompts[0])
            self.assertIn("逻辑摘要=", runner.prompts[0])
            self.assertIn("焦点进程=", runner.prompts[0])
            self.assertIn("窗口观测=", runner.prompts[0])
            self.assertIn("证据清单=", runner.prompts[0])
            self.assertIn("语义标签=", runner.prompts[0])
            self.assertIn("多模态摘要=", runner.prompts[0])

    def test_service_flushes_buffer_to_learner(self) -> None:
        with TemporaryDirectory() as temp_dir:
            event_store = DesktopEventStore(Path(temp_dir) / "events.jsonl")
            digest_store = DesktopDigestStore(Path(temp_dir) / "digests.jsonl")
            runner = FakeRunner()
            learner = DesktopLearner(runner, event_store, digest_store)
            observer = FakeObserver(
                [
                    [DesktopEvent("2026-03-20T00:00:00+00:00", "window_focus", "desktop/window", {"title": "Explorer"})],
                ]
            )

            service = DesktopAgentService(
                observer=observer,
                learner=learner,
                poll_interval_s=0.0,
                digest_interval_s=0.0,
                max_buffer_events=1,
                sleep_fn=lambda _: None,
            )

            with self.assertRaises(KeyboardInterrupt):
                service.run_forever()

            digests = [json.loads(line) for line in (Path(temp_dir) / "digests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(digests), 1)

    def test_bridge_controller_ingests_digest_and_feedback(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            controller = DesktopBridgeController(
                runner,
                BridgeEventStore(Path(temp_dir) / "bridge_events.jsonl"),
                BridgeFeedbackStore(Path(temp_dir) / "bridge_feedback.jsonl"),
                BridgeMessageStore(Path(temp_dir) / "bridge_messages.jsonl"),
                BridgeStateStore(Path(temp_dir) / "bridge_state.json"),
            )
            digest = DesktopDigest(
                timestamp_utc="2026-03-20T00:00:00+00:00",
                summary="事件数=2；类型=window_focus,clipboard；主要窗口=VS Code",
                semantic_summary="观察摘要：场景标签=代码编辑与网页检索；关键窗口=VS Code；观察到剪贴板变化。",
                logic_summary="事实: 窗口=VS Code | 事件=window_focus,clipboard | 标签=coding,research | 信号=clipboard_changed；关系: 焦点集中=VS Code；推理线索: 存在文本复制或整理行为 | 当前任务主要集中在 VS Code",
                event_count=2,
                event_types=["window_focus", "clipboard"],
                top_apps=["VS Code"],
                clipboard_seen=True,
            )
            event = controller.ingest_digest(digest)
            controller.record_feedback(event.event_id, "useful", "这是在查代码问题")

            events = controller.load_open_events()
            self.assertEqual(events, [])
            feedback_lines = [json.loads(line) for line in (Path(temp_dir) / "bridge_feedback.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(feedback_lines[-1]["feedback_type"], "useful")
            self.assertTrue(any("反馈学习任务" in prompt for prompt in runner.prompts))
            self.assertEqual(len(runner.signals), 1)
            self.assertIn("coding", event.metadata["semantic_tags"])
            self.assertIn("事实:", event.summary)
            self.assertIn("evidence_items", event.metadata)

    def test_bridge_controller_submits_user_message(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            controller = DesktopBridgeController(
                runner,
                BridgeEventStore(Path(temp_dir) / "bridge_events.jsonl"),
                BridgeFeedbackStore(Path(temp_dir) / "bridge_feedback.jsonl"),
                BridgeMessageStore(Path(temp_dir) / "bridge_messages.jsonl"),
                BridgeStateStore(Path(temp_dir) / "bridge_state.json"),
            )
            answer = controller.submit_user_message("帮我总结刚才在做什么")
            self.assertEqual(answer, "ok")
            messages = controller.load_recent_messages()
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0].role, "user")
            self.assertEqual(messages[1].role, "assistant")

    def test_bridge_controller_confirms_current_goal(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            controller = DesktopBridgeController(
                runner,
                BridgeEventStore(Path(temp_dir) / "bridge_events.jsonl"),
                BridgeFeedbackStore(Path(temp_dir) / "bridge_feedback.jsonl"),
                BridgeMessageStore(Path(temp_dir) / "bridge_messages.jsonl"),
                BridgeStateStore(Path(temp_dir) / "bridge_state.json"),
            )
            digest = DesktopDigest(
                timestamp_utc="2026-03-20T00:00:00+00:00",
                summary="事件数=1；主要窗口=VS Code",
                semantic_summary="用户可能正在修改代码。",
                event_count=1,
                top_apps=["VS Code"],
            )
            event = controller.ingest_digest(digest)
            state = controller.load_state()
            self.assertTrue(state.pending_suggestion)

            updated = controller.confirm_current_goal("修复 Python 代码问题", event.event_id)
            self.assertEqual(updated.current_goal, "修复 Python 代码问题")
            self.assertEqual(updated.pending_suggestion, "")
            self.assertTrue(any("目标确认任务" in prompt for prompt in runner.prompts))
            self.assertEqual(len(runner.signals), 1)

    def test_bridge_controller_prefers_semantic_display_summary(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            controller = DesktopBridgeController(
                runner,
                BridgeEventStore(Path(temp_dir) / "bridge_events.jsonl"),
                BridgeFeedbackStore(Path(temp_dir) / "bridge_feedback.jsonl"),
                BridgeMessageStore(Path(temp_dir) / "bridge_messages.jsonl"),
                BridgeStateStore(Path(temp_dir) / "bridge_state.json"),
            )
            event = controller.ingest_digest(
                DesktopDigest(
                    timestamp_utc="2026-03-20T00:00:00+00:00",
                    summary="事件数=1；主要窗口=VS Code",
                    semantic_summary="观察摘要：场景标签=代码编辑；关键窗口=VS Code。",
                    logic_summary="事实: 窗口=VS Code | 标签=coding；关系: 焦点集中=VS Code；推理线索: 当前任务主要集中在 VS Code",
                    event_count=1,
                    top_apps=["VS Code"],
                    semantic_tags=["coding"],
                    semantic_confidence="medium",
                    modality_hints=["desktop"],
                )
            )
            self.assertEqual(controller.event_display_summary(event), "事实: 窗口=VS Code | 标签=coding；关系: 焦点集中=VS Code；推理线索: 当前任务主要集中在 VS Code")

    def test_multimodal_adapter_compresses_desktop_digest(self) -> None:
        adapter = MultimodalAdapter()
        signal = adapter.from_desktop_digest(
            DesktopDigest(
                timestamp_utc="2026-03-20T00:00:00+00:00",
                summary="事件数=1；主要窗口=VS Code",
                semantic_summary="用户可能正在查资料并修改代码。",
                logic_summary="事实: 窗口=VS Code | 标签=coding,research；关系: 焦点集中=VS Code；推理线索: 代码编辑与资料检索同时存在，可视为问题求解链",
                event_count=1,
                top_apps=["VS Code"],
                semantic_tags=["coding", "research"],
                semantic_confidence="high",
                modality_hints=["desktop", "text"],
            )
        )
        self.assertEqual(signal.source, "desktop")
        self.assertEqual(signal.confidence, "high")
        self.assertEqual(signal.suggested_tool, "search")

    def test_screen_observation_adapter_enriches_digest(self) -> None:
        adapter = ScreenObservationAdapter(
            enabled=True,
            capture_fn=lambda: Path("data/desktop/screens/fake.png"),
            describe_fn=lambda path, digest: type("Signal", (), {
                "source": "screen",
                "semantic_text": f"视觉观察: {path.name}",
                "tags": ["screen_context", "ui"],
                "confidence": "high",
                "modality_hints": ["image", "desktop"],
                "suggested_tool": "bigmodel_proxy",
            })(),
        )
        digest = adapter.enrich_digest(
            DesktopDigest(
                timestamp_utc="2026-03-20T00:00:00+00:00",
                summary="事件数=1；主要窗口=VS Code",
                semantic_summary="用户可能正在编写或阅读代码。",
                logic_summary="事实: 窗口=VS Code | 标签=coding；关系: 焦点集中=VS Code；推理线索: 当前任务主要集中在 VS Code",
                event_count=1,
                top_apps=["VS Code"],
                semantic_tags=["coding"],
                semantic_confidence="medium",
                modality_hints=["desktop"],
            )
        )
        self.assertEqual(digest.multimodal_summary, "视觉观察: fake.png")
        self.assertIn("ui", digest.multimodal_tags)
        self.assertEqual(digest.multimodal_artifact_path, "data/desktop/screens/fake.png")
        self.assertIn("image", digest.modality_hints)
        self.assertEqual(digest.semantic_confidence, "high")

    def test_screen_observation_adapter_uses_ocr_signal_when_available(self) -> None:
        adapter = ScreenObservationAdapter(enabled=True, capture_fn=lambda: Path("data/desktop/screens/fake.png"))

        class FakeOcr:
            def __call__(self, image_path: str):
                return (
                    [
                        [[[0, 0], [1, 0], [1, 1], [0, 1]], "VS Code", "0.92"],
                        [[[0, 0], [1, 0], [1, 1], [0, 1]], "PostgreSQL 文档", "0.88"],
                        [[[0, 0], [1, 0], [1, 1], [0, 1]], "比较 MySQL 与 PostgreSQL", "0.81"],
                    ],
                    0.2,
                )

        adapter._ocr_engine = FakeOcr()
        digest = adapter.enrich_digest(
            DesktopDigest(
                timestamp_utc="2026-03-20T00:00:00+00:00",
                summary="事件数=1；主要窗口=VS Code",
                semantic_summary="用户可能正在查资料并修改代码。",
                logic_summary="事实: 窗口=VS Code | 标签=coding,research；关系: 焦点集中=VS Code；推理线索: 代码编辑与资料检索同时存在，可视为问题求解链",
                event_count=1,
                top_apps=["VS Code"],
                semantic_tags=["coding", "research"],
                semantic_confidence="medium",
                modality_hints=["desktop"],
            )
        )
        self.assertIn("屏幕识别到文字", digest.multimodal_summary)
        self.assertIn("视觉证据:", digest.logic_summary)
        self.assertIn("ocr", digest.multimodal_tags)
        self.assertIn("image", digest.modality_hints)

    def test_bridge_controller_creates_proactive_question_for_related_event(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            controller = DesktopBridgeController(
                runner,
                BridgeEventStore(Path(temp_dir) / "bridge_events.jsonl"),
                BridgeFeedbackStore(Path(temp_dir) / "bridge_feedback.jsonl"),
                BridgeMessageStore(Path(temp_dir) / "bridge_messages.jsonl"),
                BridgeStateStore(Path(temp_dir) / "bridge_state.json"),
                proactive_config={"cooldown_s": 0, "min_goal_overlap": 1, "initial_budget": 2, "max_budget": 3},
            )
            controller.confirm_current_goal("修复 VS Code 里的 Python 问题")
            digest = DesktopDigest(
                timestamp_utc="2026-03-20T00:00:00+00:00",
                summary="事件数=1；主要窗口=VS Code",
                semantic_summary="用户可能正在修改代码。",
                event_count=1,
                top_apps=["VS Code"],
            )
            controller.ingest_digest(digest)
            state = controller.load_state()
            self.assertTrue(state.pending_question)
            self.assertEqual(state.proactive_budget_remaining, 1)
            self.assertEqual(state.proactive_status, "已生成新的主动追问。")
            messages = controller.load_recent_messages()
            self.assertTrue(any(message.source == "bridge_proactive" for message in messages))

    def test_bridge_controller_respects_proactive_cooldown(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            controller = DesktopBridgeController(
                runner,
                BridgeEventStore(Path(temp_dir) / "bridge_events.jsonl"),
                BridgeFeedbackStore(Path(temp_dir) / "bridge_feedback.jsonl"),
                BridgeMessageStore(Path(temp_dir) / "bridge_messages.jsonl"),
                BridgeStateStore(Path(temp_dir) / "bridge_state.json"),
                proactive_config={"cooldown_s": 9999, "min_goal_overlap": 1, "initial_budget": 2},
            )
            controller.confirm_current_goal("修复 VS Code 里的 Python 问题")
            digest = DesktopDigest(
                timestamp_utc="2026-03-20T00:00:00+00:00",
                summary="事件数=1；主要窗口=VS Code",
                semantic_summary="用户可能正在修改代码。",
                event_count=1,
                top_apps=["VS Code"],
            )
            controller.ingest_digest(digest)
            first_state = controller.load_state()
            controller.record_feedback(first_state.pending_question_event_id, "useful", "继续")
            controller.ingest_digest(digest)
            second_state = controller.load_state()
            messages = controller.load_recent_messages()
            proactive_count = sum(1 for message in messages if message.source == "bridge_proactive")
            self.assertEqual(proactive_count, 1)
            self.assertEqual(second_state.pending_question, "")
            self.assertEqual(second_state.proactive_status, "主动追问仍在冷却中。")

    def test_bridge_controller_skips_proactive_question_for_unrelated_event(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            controller = DesktopBridgeController(
                runner,
                BridgeEventStore(Path(temp_dir) / "bridge_events.jsonl"),
                BridgeFeedbackStore(Path(temp_dir) / "bridge_feedback.jsonl"),
                BridgeMessageStore(Path(temp_dir) / "bridge_messages.jsonl"),
                BridgeStateStore(Path(temp_dir) / "bridge_state.json"),
                proactive_config={"cooldown_s": 0, "min_goal_overlap": 1, "initial_budget": 2},
            )
            controller.confirm_current_goal("修复 VS Code 里的 Python 问题")
            digest = DesktopDigest(
                timestamp_utc="2026-03-20T00:00:00+00:00",
                summary="事件数=1；主要窗口=Excel",
                semantic_summary="用户可能正在查看或整理表格数据。",
                event_count=1,
                top_apps=["Excel"],
            )
            controller.ingest_digest(digest)
            state = controller.load_state()
            self.assertEqual(state.pending_question, "")
            self.assertEqual(state.proactive_status, "这次桌面变化与当前目标相关性不足。")

    def test_bridge_controller_does_not_stack_pending_proactive_questions(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            controller = DesktopBridgeController(
                runner,
                BridgeEventStore(Path(temp_dir) / "bridge_events.jsonl"),
                BridgeFeedbackStore(Path(temp_dir) / "bridge_feedback.jsonl"),
                BridgeMessageStore(Path(temp_dir) / "bridge_messages.jsonl"),
                BridgeStateStore(Path(temp_dir) / "bridge_state.json"),
                proactive_config={"cooldown_s": 0, "min_goal_overlap": 1, "initial_budget": 2},
            )
            controller.confirm_current_goal("修复 VS Code 里的 Python 问题")
            digest = DesktopDigest(
                timestamp_utc="2026-03-20T00:00:00+00:00",
                summary="事件数=1；主要窗口=VS Code",
                semantic_summary="用户可能正在修改代码。",
                event_count=1,
                top_apps=["VS Code"],
            )
            controller.ingest_digest(digest)
            first_state = controller.load_state()
            self.assertTrue(first_state.pending_question)

            controller.ingest_digest(digest)
            second_state = controller.load_state()
            messages = controller.load_recent_messages()
            proactive_count = sum(1 for message in messages if message.source == "bridge_proactive")

            self.assertEqual(proactive_count, 1)
            self.assertEqual(second_state.pending_question, first_state.pending_question)
            self.assertEqual(second_state.proactive_status, "已有待处理主动追问，暂不重复发问。")

    def test_bridge_controller_clears_pending_question_when_goal_confirmed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            controller = DesktopBridgeController(
                runner,
                BridgeEventStore(Path(temp_dir) / "bridge_events.jsonl"),
                BridgeFeedbackStore(Path(temp_dir) / "bridge_feedback.jsonl"),
                BridgeMessageStore(Path(temp_dir) / "bridge_messages.jsonl"),
                BridgeStateStore(Path(temp_dir) / "bridge_state.json"),
                proactive_config={"cooldown_s": 0, "min_goal_overlap": 1, "initial_budget": 2},
            )
            controller.confirm_current_goal("修复 VS Code 里的 Python 问题")
            digest = DesktopDigest(
                timestamp_utc="2026-03-20T00:00:00+00:00",
                summary="事件数=1；主要窗口=VS Code",
                semantic_summary="用户可能正在修改代码。",
                event_count=1,
                top_apps=["VS Code"],
            )
            event = controller.ingest_digest(digest)
            before = controller.load_state()
            self.assertTrue(before.pending_question)

            after = controller.confirm_current_goal("继续修复 VS Code 里的 Python 问题", event.event_id)
            self.assertEqual(after.pending_question, "")
            self.assertEqual(after.pending_question_event_id, "")
            self.assertEqual(after.proactive_budget_remaining, 2)
            self.assertEqual(after.proactive_status, "当前目标已更新，主动追问预算已重置。")

    def test_bridge_controller_respects_proactive_budget(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            controller = DesktopBridgeController(
                runner,
                BridgeEventStore(Path(temp_dir) / "bridge_events.jsonl"),
                BridgeFeedbackStore(Path(temp_dir) / "bridge_feedback.jsonl"),
                BridgeMessageStore(Path(temp_dir) / "bridge_messages.jsonl"),
                BridgeStateStore(Path(temp_dir) / "bridge_state.json"),
                proactive_config={"cooldown_s": 0, "min_goal_overlap": 1, "initial_budget": 1, "max_budget": 2},
            )
            controller.confirm_current_goal("修复 VS Code 里的 Python 问题")
            digest = DesktopDigest(
                timestamp_utc="2026-03-20T00:00:00+00:00",
                summary="事件数=1；主要窗口=VS Code",
                semantic_summary="用户可能正在修改代码。",
                event_count=1,
                top_apps=["VS Code"],
            )
            first_event = controller.ingest_digest(digest)
            state = controller.load_state()
            self.assertEqual(state.proactive_budget_remaining, 0)

            controller.record_feedback(first_event.event_id, "dismiss", "先别打扰")
            controller.ingest_digest(digest)
            second_state = controller.load_state()
            proactive_count = sum(1 for message in controller.load_recent_messages() if message.source == "bridge_proactive")

            self.assertEqual(proactive_count, 1)
            self.assertEqual(second_state.pending_question, "")
            self.assertEqual(second_state.proactive_status, "主动追问预算已用尽。")

    def test_bridge_controller_recovers_budget_after_useful_feedback(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            controller = DesktopBridgeController(
                runner,
                BridgeEventStore(Path(temp_dir) / "bridge_events.jsonl"),
                BridgeFeedbackStore(Path(temp_dir) / "bridge_feedback.jsonl"),
                BridgeMessageStore(Path(temp_dir) / "bridge_messages.jsonl"),
                BridgeStateStore(Path(temp_dir) / "bridge_state.json"),
                proactive_config={"cooldown_s": 0, "min_goal_overlap": 1, "initial_budget": 1, "max_budget": 2},
            )
            controller.confirm_current_goal("修复 VS Code 里的 Python 问题")
            digest = DesktopDigest(
                timestamp_utc="2026-03-20T00:00:00+00:00",
                summary="事件数=1；主要窗口=VS Code",
                semantic_summary="用户可能正在修改代码。",
                event_count=1,
                top_apps=["VS Code"],
            )
            event = controller.ingest_digest(digest)
            controller.record_feedback(event.event_id, "useful", "这个追问有帮助")

            state = controller.load_state()
            self.assertEqual(state.proactive_budget_remaining, 1)
            self.assertEqual(state.proactive_status, "收到正向反馈，主动追问预算已恢复。")

    def test_bridge_controller_suppresses_proactive_questions_after_recent_negative_feedback(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            controller = DesktopBridgeController(
                runner,
                BridgeEventStore(Path(temp_dir) / "bridge_events.jsonl"),
                BridgeFeedbackStore(Path(temp_dir) / "bridge_feedback.jsonl"),
                BridgeMessageStore(Path(temp_dir) / "bridge_messages.jsonl"),
                BridgeStateStore(Path(temp_dir) / "bridge_state.json"),
                proactive_config={
                    "cooldown_s": 0,
                    "min_goal_overlap": 1,
                    "initial_budget": 3,
                    "max_budget": 3,
                    "feedback_window": 4,
                    "negative_feedback_limit": 2,
                },
            )
            controller.confirm_current_goal("修复 VS Code 里的 Python 问题")
            digest = DesktopDigest(
                timestamp_utc="2026-03-20T00:00:00+00:00",
                summary="事件数=1；主要窗口=VS Code",
                semantic_summary="用户可能正在修改代码。",
                event_count=1,
                top_apps=["VS Code"],
            )
            event = controller.ingest_digest(digest)
            controller.record_feedback(event.event_id, "misread", "理解错了")
            controller.record_feedback(event.event_id, "dismiss", "先不要问")
            controller.ingest_digest(digest)

            state = controller.load_state()
            proactive_count = sum(1 for message in controller.load_recent_messages() if message.source == "bridge_proactive")
            self.assertEqual(proactive_count, 1)
            self.assertEqual(state.pending_question, "")
            self.assertEqual(state.proactive_status, "近期负向反馈较多，主动追问已收缩。")

    def test_runtime_status_is_stopped_without_pid(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manager = DesktopAgentProcessManager(Path(temp_dir) / "runtime.json")
            status = manager.status()
            self.assertFalse(status.running)
            self.assertEqual(status.pid, 0)

    def test_load_bridge_state_summary_reads_goal_and_proactive_state(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "desktop_agent.json"
            bridge_state_path = Path(temp_dir) / "bridge_state.json"
            config_path.write_text(
                json.dumps({"bridge_paths": {"state": str(bridge_state_path)}}, ensure_ascii=False),
                encoding="utf-8",
            )
            bridge_state_path.write_text(
                json.dumps(
                    {
                        "current_goal": "修复 VS Code 里的 Python 问题",
                        "proactive_status": "主动追问仍在冷却中。",
                        "proactive_budget_remaining": 1,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = load_bridge_state_summary(config_path)
            self.assertEqual(summary["current_goal"], "修复 VS Code 里的 Python 问题")
            self.assertEqual(summary["proactive_status"], "主动追问仍在冷却中。")
            self.assertEqual(summary["proactive_budget_remaining"], 1)

    def test_status_payload_includes_bridge_state(self) -> None:
        status = status_payload(
            DesktopRuntimeStatus(
                running=False,
                pid=0,
                started_at_utc="",
                log_path="",
                runtime_path="data/desktop/runtime.json",
                current_goal="整理桌面桥梁交互",
                proactive_status="主动追问预算已用尽。",
                proactive_budget_remaining=0,
            )
        )
        self.assertEqual(status["current_goal"], "整理桌面桥梁交互")
        self.assertEqual(status["proactive_status"], "主动追问预算已用尽。")
        self.assertEqual(status["proactive_budget_remaining"], 0)

    def test_format_status_snapshot_is_human_readable(self) -> None:
        snapshot = format_status_snapshot(
            DesktopRuntimeStatus(
                running=True,
                pid=1234,
                started_at_utc="2026-03-20T00:00:00+00:00",
                log_path="data/desktop/desktop_agent.log",
                runtime_path="data/desktop/runtime.json",
                current_goal="整理 README 和托盘交互",
                proactive_status="主动追问仍在冷却中。",
                proactive_budget_remaining=1,
            )
        )
        self.assertIn("CARM 桌面状态快照", snapshot)
        self.assertIn("运行状态: 运行中", snapshot)
        self.assertIn("当前目标: 整理 README 和托盘交互", snapshot)
        self.assertIn("主动预算: 1", snapshot)

    def test_build_tray_python_command(self) -> None:
        command = build_tray_python_command("python")
        self.assertEqual(command, ["python", "-m", "scripts.desktop_agent_tray"])

    def test_build_bridge_chat_command(self) -> None:
        command = build_bridge_chat_command("python")
        self.assertEqual(command, ["python", "-m", "scripts.desktop_bridge_chat"])

    def test_build_startup_shortcut_script_contains_target(self) -> None:
        script = build_startup_shortcut_script(
            Path("C:/Users/test/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup/CARM.lnk"),
            ["pythonw", "-m", "scripts.desktop_agent_tray"],
            Path("D:/codes/Mustard"),
        )
        self.assertIn("CreateShortcut", script)
        self.assertIn("scripts.desktop_agent_tray", script)
        self.assertIn("D:\\codes\\Mustard", script)

    @patch("carm.desktop_runtime.subprocess.Popen")
    @patch.object(DesktopAgentProcessManager, "start")
    def test_launch_desktop_bridge_starts_tray_and_chat(self, start_mock, popen_mock) -> None:
        start_mock.return_value = DesktopRuntimeStatus(
            running=True,
            pid=1234,
            started_at_utc="2026-03-20T00:00:00+00:00",
            log_path="data/desktop/desktop_agent.log",
            runtime_path="data/desktop/runtime.json",
        )

        payload = launch_desktop_bridge(python_executable="python")

        self.assertTrue(payload["launched"])
        self.assertEqual(start_mock.call_count, 1)
        self.assertEqual(popen_mock.call_count, 2)
        self.assertEqual(payload["tray_command"], ["python", "-m", "scripts.desktop_agent_tray"])
        self.assertEqual(payload["bridge_chat_command"], ["python", "-m", "scripts.desktop_bridge_chat"])


if __name__ == "__main__":
    unittest.main()
