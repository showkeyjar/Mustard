from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from carm.bridge import BridgeEventStore, BridgeFeedbackStore, BridgeMessageStore, BridgeStateStore, DesktopBridgeController
from carm.desktop import DesktopAgentService, DesktopDigestStore, DesktopEventStore, DesktopLearner, WindowsDesktopObserver
from carm.multimodal import ScreenObservationAdapter
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


def main() -> int:
    service_mode = "--service" in sys.argv[1:]
    config_path = Path(os.environ.get("CARM_DESKTOP_CONFIG", "configs/desktop_agent.json"))
    config = load_config(config_path)
    paths = config.get("paths", {}) if isinstance(config.get("paths", {}), dict) else {}
    bridge_paths = config.get("bridge_paths", {}) if isinstance(config.get("bridge_paths", {}), dict) else {}
    proactive = config.get("proactive", {}) if isinstance(config.get("proactive", {}), dict) else {}
    multimodal = config.get("multimodal", {}) if isinstance(config.get("multimodal", {}), dict) else {}

    event_store = DesktopEventStore(Path(str(paths.get("events", "data/desktop/events.jsonl"))))
    digest_store = DesktopDigestStore(Path(str(paths.get("digests", "data/desktop/digests.jsonl"))))
    bridge_controller = DesktopBridgeController(
        runner=build_runner(),
        event_store=BridgeEventStore(Path(str(bridge_paths.get("events", "data/desktop/bridge_events.jsonl")))),
        feedback_store=BridgeFeedbackStore(Path(str(bridge_paths.get("feedback", "data/desktop/bridge_feedback.jsonl")))),
        message_store=BridgeMessageStore(Path(str(bridge_paths.get("messages", "data/desktop/bridge_messages.jsonl")))),
        state_store=BridgeStateStore(Path(str(bridge_paths.get("state", "data/desktop/bridge_state.json")))),
        proactive_config=proactive,
    )
    observer = WindowsDesktopObserver()
    screen_adapter = ScreenObservationAdapter(
        enabled=bool(multimodal.get("screen_enabled", False)),
        capture_dir=str(multimodal.get("screen_capture_dir", "data/desktop/screens")),
        describe_command=list(multimodal.get("image_describer_command", []))
        if isinstance(multimodal.get("image_describer_command", []), list)
        else [],
    )
    learner = DesktopLearner(
        build_runner(),
        event_store,
        digest_store,
        bridge_controller=bridge_controller,
        digest_enricher=screen_adapter.enrich_digest,
    )
    service = DesktopAgentService(
        observer=observer,
        learner=learner,
        poll_interval_s=float(config.get("poll_interval_s", 2.0) or 2.0),
        digest_interval_s=float(config.get("digest_interval_s", 30.0) or 30.0),
        max_buffer_events=int(config.get("max_buffer_events", 64) or 64),
    )

    if not service_mode:
        print("CARM desktop agent started. Press Ctrl+C to stop.")
    try:
        service.run_forever()
    except KeyboardInterrupt:
        if not service_mode:
            print("CARM desktop agent stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
