from __future__ import annotations

import argparse
import json

from carm.desktop_runtime import (
    DesktopAgentProcessManager,
    format_status_snapshot,
    install_startup_shortcut,
    launch_desktop_bridge,
    remove_startup_shortcut,
    status_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Control the CARM desktop agent runtime.")
    parser.add_argument("command", choices=["launch", "start", "stop", "status", "snapshot", "install-startup", "remove-startup"])
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    manager = DesktopAgentProcessManager()

    if args.command == "launch":
        payload = launch_desktop_bridge()
    elif args.command == "start":
        status = manager.start()
        payload = status_payload(status)
    elif args.command == "stop":
        status = manager.stop()
        payload = status_payload(status)
    elif args.command == "status":
        status = manager.status()
        payload = status_payload(status)
    elif args.command == "snapshot":
        status = manager.status()
        payload = {
            "snapshot": format_status_snapshot(status),
            **status_payload(status),
        }
    elif args.command == "install-startup":
        shortcut = install_startup_shortcut()
        payload = {"installed": True, "shortcut_path": str(shortcut)}
    else:
        shortcut = remove_startup_shortcut()
        payload = {"installed": False, "shortcut_path": str(shortcut)}

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        if args.command == "snapshot":
            print(payload["snapshot"])
        elif args.command == "launch":
            print("CARM 已一键启动。桌面代理、托盘和桥梁窗口正在启动。")
        else:
            print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
