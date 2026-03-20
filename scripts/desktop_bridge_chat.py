from __future__ import annotations

import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext

from carm.bridge import BridgeEventStore, BridgeFeedbackStore, BridgeMessageStore, BridgeStateStore, DesktopBridgeController
from carm.runner import AgentRunner
from tools.base import ToolManager
from tools.bigmodel_tool import BigModelProxyTool
from tools.calc_tool import CalculatorTool
from tools.code_tool import CodeExecutorTool
from tools.search_tool import SearchTool


def load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_runner() -> AgentRunner:
    return AgentRunner(
        ToolManager(
            [
                SearchTool(),
                CalculatorTool(),
                CodeExecutorTool(),
                BigModelProxyTool(),
            ]
        ),
        experience_path=Path(os.environ.get("CARM_EXPERIENCE_PATH", "data/experience/episodes.jsonl")),
        policy_state_path=Path(os.environ.get("CARM_POLICY_STATE_PATH", "data/experience/policy_state.json")),
        concept_state_path=Path(os.environ.get("CARM_CONCEPT_STATE_PATH", "data/experience/concept_state.json")),
        core_state_path=Path(os.environ.get("CARM_CORE_STATE_PATH", "data/experience/core_state.json")),
        review_path=Path(os.environ.get("CARM_REVIEW_PATH", "data/review/reviews.jsonl")),
        controls_path=Path(os.environ.get("CARM_CONTROLS_PATH", "data/control/runtime_controls.json")),
    )


def build_controller() -> DesktopBridgeController:
    config = load_config(Path(os.environ.get("CARM_DESKTOP_CONFIG", "configs/desktop_agent.json")))
    bridge_paths = config.get("bridge_paths", {}) if isinstance(config.get("bridge_paths", {}), dict) else {}
    proactive = config.get("proactive", {}) if isinstance(config.get("proactive", {}), dict) else {}
    return DesktopBridgeController(
        runner=build_runner(),
        event_store=BridgeEventStore(Path(str(bridge_paths.get("events", "data/desktop/bridge_events.jsonl")))),
        feedback_store=BridgeFeedbackStore(Path(str(bridge_paths.get("feedback", "data/desktop/bridge_feedback.jsonl")))),
        message_store=BridgeMessageStore(Path(str(bridge_paths.get("messages", "data/desktop/bridge_messages.jsonl")))),
        state_store=BridgeStateStore(Path(str(bridge_paths.get("state", "data/desktop/bridge_state.json")))),
        proactive_config=proactive,
    )


class BridgeChatApp:
    def __init__(self, root: tk.Tk, controller: DesktopBridgeController) -> None:
        self.root = root
        self.controller = controller
        self.selected_event_id: str = ""
        self.root.title("CARM Bridge")
        self.root.geometry("720x640")

        self.goal_var = tk.StringVar()
        self.suggestion_var = tk.StringVar()
        self.question_var = tk.StringVar()
        self.proactive_status_var = tk.StringVar()
        self.budget_var = tk.StringVar()

        goal_frame = tk.Frame(root)
        goal_frame.pack(fill=tk.X, padx=12, pady=(12, 6))
        tk.Label(goal_frame, text="当前目标:").pack(side=tk.LEFT)
        tk.Label(goal_frame, textvariable=self.goal_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        suggestion_frame = tk.Frame(root)
        suggestion_frame.pack(fill=tk.X, padx=12, pady=(0, 6))
        tk.Label(suggestion_frame, text="候选目标:").pack(side=tk.LEFT)
        tk.Label(suggestion_frame, textvariable=self.suggestion_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        question_frame = tk.Frame(root)
        question_frame.pack(fill=tk.X, padx=12, pady=(0, 6))
        tk.Label(question_frame, text="主动追问:").pack(side=tk.LEFT)
        tk.Label(question_frame, textvariable=self.question_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        proactive_meta_frame = tk.Frame(root)
        proactive_meta_frame.pack(fill=tk.X, padx=12, pady=(0, 6))
        tk.Label(proactive_meta_frame, text="追问预算:").pack(side=tk.LEFT)
        tk.Label(proactive_meta_frame, textvariable=self.budget_var, anchor="w").pack(side=tk.LEFT, padx=(6, 12))
        tk.Label(proactive_meta_frame, text="最近状态:").pack(side=tk.LEFT)
        tk.Label(proactive_meta_frame, textvariable=self.proactive_status_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        self.event_list = tk.Listbox(root, height=8)
        self.event_list.pack(fill=tk.X, padx=12, pady=(0, 6))
        self.event_list.bind("<<ListboxSelect>>", self.on_event_select)

        button_frame = tk.Frame(root)
        button_frame.pack(fill=tk.X, padx=12, pady=(0, 6))
        tk.Button(button_frame, text="采用追问", command=self.use_pending_question).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(button_frame, text="设为当前目标", command=self.confirm_goal).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(button_frame, text="标记有价值", command=lambda: self.feedback("useful")).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(button_frame, text="误判", command=lambda: self.feedback("misread")).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(button_frame, text="不要学习", command=lambda: self.feedback("dismiss")).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(button_frame, text="刷新", command=self.refresh).pack(side=tk.RIGHT)

        self.chat_view = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, height=22)
        self.chat_view.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 6))

        entry_frame = tk.Frame(root)
        entry_frame.pack(fill=tk.X, padx=12, pady=(0, 12))
        self.input_var = tk.StringVar()
        entry = tk.Entry(entry_frame, textvariable=self.input_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        entry.bind("<Return>", lambda _: self.send_message())
        tk.Button(entry_frame, text="发送", command=self.send_message).pack(side=tk.RIGHT)

        self.refresh()

    def refresh(self) -> None:
        events = self.controller.load_open_events()
        self.event_list.delete(0, tk.END)
        for event in events:
            display_summary = self.controller.event_display_summary(event)
            self.event_list.insert(tk.END, f"[{event.kind}] {display_summary}")
        self._events = events
        self._render_messages()

    def on_event_select(self, _event) -> None:
        if not self.event_list.curselection():
            self.selected_event_id = ""
            return
        index = int(self.event_list.curselection()[0])
        if index >= len(self._events):
            self.selected_event_id = ""
            return
        event = self._events[index]
        self.selected_event_id = event.event_id
        self.input_var.set(self.controller.suggest_goal_for_event(event))
        self.suggestion_var.set(self.controller.suggest_goal_for_event(event))

    def confirm_goal(self) -> None:
        goal_text = self.input_var.get().strip()
        if not goal_text:
            messagebox.showinfo("CARM Bridge", "请先输入或选择一个候选目标。")
            return
        self.controller.confirm_current_goal(goal_text, self.selected_event_id)
        self.input_var.set("")
        self.selected_event_id = ""
        self.refresh()

    def use_pending_question(self) -> None:
        state = self.controller.load_state()
        if not state.pending_question:
            messagebox.showinfo("CARM Bridge", "当前没有主动追问。")
            return
        self.input_var.set(state.pending_question)

    def feedback(self, feedback_type: str) -> None:
        if not self.selected_event_id:
            messagebox.showinfo("CARM Bridge", "请先选择一条桌面事件。")
            return
        note = self.input_var.get().strip()
        self.controller.record_feedback(self.selected_event_id, feedback_type, note)
        self.input_var.set("")
        self.selected_event_id = ""
        self.refresh()

    def send_message(self) -> None:
        text = self.input_var.get().strip()
        if not text:
            return
        self.controller.submit_user_message(text)
        self.input_var.set("")
        self.refresh()

    def _render_messages(self) -> None:
        state = self.controller.load_state()
        self.goal_var.set(state.current_goal or "未确认")
        self.suggestion_var.set(state.pending_suggestion or "无")
        self.question_var.set(state.pending_question or "无")
        self.budget_var.set(str(state.proactive_budget_remaining))
        self.proactive_status_var.set(state.proactive_status or "无")
        messages = self.controller.load_recent_messages(limit=30)
        self.chat_view.config(state=tk.NORMAL)
        self.chat_view.delete("1.0", tk.END)
        for message in messages:
            role = "You" if message.role == "user" else "CARM"
            self.chat_view.insert(tk.END, f"{role}: {message.text}\n\n")
        self.chat_view.config(state=tk.DISABLED)
        self.chat_view.see(tk.END)


def main() -> int:
    root = tk.Tk()
    app = BridgeChatApp(root, build_controller())
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
