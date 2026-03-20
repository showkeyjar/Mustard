from __future__ import annotations

import json
import os
import subprocess
import sys
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

kernel32 = ctypes.windll.kernel32 if os.name == "nt" else None
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


@dataclass
class DesktopRuntimeStatus:
    running: bool
    pid: int = 0
    started_at_utc: str = ""
    log_path: str = ""
    runtime_path: str = ""
    current_goal: str = ""
    proactive_status: str = ""
    proactive_budget_remaining: int = 0


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_desktop_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path or os.environ.get("CARM_DESKTOP_CONFIG", "configs/desktop_agent.json"))
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_bridge_state_summary(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_desktop_config(config_path)
    bridge_paths = config.get("bridge_paths", {}) if isinstance(config.get("bridge_paths", {}), dict) else {}
    state_path = Path(str(bridge_paths.get("state", "data/desktop/bridge_state.json")))
    if not state_path.exists():
        return {
            "current_goal": "",
            "proactive_status": "",
            "proactive_budget_remaining": 0,
        }
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {
            "current_goal": "",
            "proactive_status": "",
            "proactive_budget_remaining": 0,
        }
    return {
        "current_goal": str(payload.get("current_goal", "")),
        "proactive_status": str(payload.get("proactive_status", "")),
        "proactive_budget_remaining": int(payload.get("proactive_budget_remaining", 0) or 0),
    }


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    process_handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not process_handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(process_handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(process_handle)


class DesktopAgentProcessManager:
    def __init__(self, runtime_path: str | Path = "data/desktop/runtime.json") -> None:
        self.runtime_path = Path(runtime_path)
        self.runtime_path.parent.mkdir(parents=True, exist_ok=True)

    def status(self) -> DesktopRuntimeStatus:
        payload = load_json(self.runtime_path)
        bridge_summary = load_bridge_state_summary()
        pid = int(payload.get("pid", 0) or 0)
        running = is_pid_running(pid)
        if not running and self.runtime_path.exists():
            stale = dict(payload)
            stale["running"] = False
            save_json(self.runtime_path, stale)
        return DesktopRuntimeStatus(
            running=running,
            pid=pid if running else 0,
            started_at_utc=str(payload.get("started_at_utc", "")),
            log_path=str(payload.get("log_path", "")),
            runtime_path=str(self.runtime_path),
            current_goal=str(bridge_summary.get("current_goal", "")),
            proactive_status=str(bridge_summary.get("proactive_status", "")),
            proactive_budget_remaining=int(bridge_summary.get("proactive_budget_remaining", 0) or 0),
        )

    def start(self, *, python_executable: str | None = None) -> DesktopRuntimeStatus:
        current = self.status()
        if current.running:
            return current

        python_executable = python_executable or sys.executable
        log_path = self.runtime_path.parent / "desktop_agent.log"
        command = [python_executable, "-m", "scripts.run_desktop_agent", "--service"]
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

        with log_path.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                stdout=handle,
                stderr=handle,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                cwd=Path.cwd(),
            )

        payload = {
            "pid": process.pid,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "log_path": str(log_path),
            "command": command,
            "running": True,
        }
        save_json(self.runtime_path, payload)
        return self.status()

    def stop(self) -> DesktopRuntimeStatus:
        current = self.status()
        if not current.running:
            return current

        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(current.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            try:
                os.kill(current.pid, 15)
            except OSError:
                pass

        payload = load_json(self.runtime_path)
        payload["running"] = False
        save_json(self.runtime_path, payload)
        return self.status()


def build_tray_python_command(python_executable: str | None = None) -> list[str]:
    return [resolve_gui_python_executable(python_executable), "-m", "scripts.desktop_agent_tray"]


def resolve_gui_python_executable(python_executable: str | None = None) -> str:
    candidate = Path(python_executable or sys.executable)
    if candidate.name.lower() == "python.exe":
        pythonw = candidate.with_name("pythonw.exe")
        if pythonw.exists():
            return str(pythonw)
    return str(candidate)


def build_bridge_chat_command(python_executable: str | None = None) -> list[str]:
    return [resolve_gui_python_executable(python_executable), "-m", "scripts.desktop_bridge_chat"]


def build_startup_shortcut_script(shortcut_path: Path, target_command: list[str], working_directory: Path) -> str:
    powershell_exe = target_command[0]
    arguments = " ".join(f'"{item}"' if " " in item else item for item in target_command[1:])
    return "\n".join(
        [
            "$shell = New-Object -ComObject WScript.Shell",
            f'$shortcut = $shell.CreateShortcut("{shortcut_path}")',
            f'$shortcut.TargetPath = "{powershell_exe}"',
            f'$shortcut.Arguments = \'{arguments}\'',
            f'$shortcut.WorkingDirectory = "{working_directory}"',
            "$shortcut.IconLocation = \"$env:SystemRoot\\System32\\SHELL32.dll,44\"",
            "$shortcut.Save()",
        ]
    )


def install_startup_shortcut(shortcut_name: str = "CARM Desktop Agent Tray.lnk") -> Path:
    startup_dir = Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs/Startup"
    startup_dir.mkdir(parents=True, exist_ok=True)
    shortcut_path = startup_dir / shortcut_name
    command = build_tray_python_command()
    script = build_startup_shortcut_script(shortcut_path, command, Path.cwd())
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return shortcut_path


def remove_startup_shortcut(shortcut_name: str = "CARM Desktop Agent Tray.lnk") -> Path:
    startup_dir = Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs/Startup"
    shortcut_path = startup_dir / shortcut_name
    if shortcut_path.exists():
        shortcut_path.unlink()
    return shortcut_path


def status_payload(status: DesktopRuntimeStatus) -> dict[str, Any]:
    return asdict(status)


def format_status_snapshot(status: DesktopRuntimeStatus) -> str:
    lines = [
        "CARM 桌面状态快照",
        f"- 运行状态: {'运行中' if status.running else '未运行'}",
        f"- PID: {status.pid or '无'}",
        f"- 当前目标: {status.current_goal or '未确认'}",
        f"- 主动状态: {status.proactive_status or '无'}",
        f"- 主动预算: {status.proactive_budget_remaining}",
    ]
    if status.started_at_utc:
        lines.append(f"- 启动时间: {status.started_at_utc}")
    if status.log_path:
        lines.append(f"- 日志路径: {status.log_path}")
    return "\n".join(lines)


def launch_desktop_bridge(
    *,
    python_executable: str | None = None,
    runtime_path: str | Path = "data/desktop/runtime.json",
) -> dict[str, Any]:
    manager = DesktopAgentProcessManager(runtime_path)
    status = manager.start(python_executable=python_executable)

    tray_command = build_tray_python_command(python_executable)
    bridge_chat_command = build_bridge_chat_command(python_executable)

    tray_creationflags = 0
    chat_creationflags = 0
    if os.name == "nt":
        tray_creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        chat_creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    subprocess.Popen(
        tray_command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=tray_creationflags,
        cwd=Path.cwd(),
    )
    subprocess.Popen(
        bridge_chat_command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=chat_creationflags,
        cwd=Path.cwd(),
    )

    return {
        "launched": True,
        "agent": status_payload(status),
        "tray_command": tray_command,
        "bridge_chat_command": bridge_chat_command,
    }
