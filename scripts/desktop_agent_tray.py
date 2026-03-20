from __future__ import annotations

import os
import subprocess
from pathlib import Path

import win32api
import win32con
import win32gui

from carm.desktop_runtime import (
    DesktopAgentProcessManager,
    build_bridge_chat_command,
    format_status_snapshot,
    install_startup_shortcut,
    remove_startup_shortcut,
)


WM_APP_NOTIFY = win32con.WM_USER + 20
TIMER_ID = 1

ID_START = 1001
ID_STOP = 1002
ID_STATUS = 1003
ID_SNAPSHOT = 1004
ID_OPEN_CHAT = 1005
ID_OPEN_DATA = 1006
ID_INSTALL_STARTUP = 1007
ID_REMOVE_STARTUP = 1008
ID_EXIT = 1009


class TrayApp:
    def __init__(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent
        self.manager = DesktopAgentProcessManager()
        self.class_name = "CarmDesktopTrayWindow"
        self.hinst = win32api.GetModuleHandle(None)
        self.hwnd = None
        self.notify_id = None

    def run(self) -> int:
        message_map = {
            WM_APP_NOTIFY: self._on_tray_notify,
            win32con.WM_COMMAND: self._on_command,
            win32con.WM_DESTROY: self._on_destroy,
        }

        window_class = win32gui.WNDCLASS()
        window_class.hInstance = self.hinst
        window_class.lpszClassName = self.class_name
        window_class.lpfnWndProc = message_map
        class_atom = win32gui.RegisterClass(window_class)

        self.hwnd = win32gui.CreateWindow(
            class_atom,
            self.class_name,
            0,
            0,
            0,
            win32con.CW_USEDEFAULT,
            win32con.CW_USEDEFAULT,
            0,
            0,
            self.hinst,
            None,
        )

        self._add_tray_icon()
        win32gui.PumpMessages()
        return 0

    def _add_tray_icon(self) -> None:
        icon = win32gui.LoadIcon(0, win32con.IDI_INFORMATION)
        self.notify_id = (
            self.hwnd,
            1,
            win32gui.NIF_ICON | win32gui.NIF_MESSAGE | win32gui.NIF_TIP,
            WM_APP_NOTIFY,
            icon,
            self._tooltip_text(),
        )
        win32gui.Shell_NotifyIcon(win32gui.NIM_ADD, self.notify_id)

    def _tooltip_text(self) -> str:
        status = self.manager.status()
        if status.current_goal:
            return f"CARM: {status.current_goal}"[:127]
        if status.running:
            return "CARM Desktop Agent (运行中)"
        return "CARM Desktop Agent"

    def _refresh_tooltip(self) -> None:
        if not self.notify_id:
            return
        icon = win32gui.LoadIcon(0, win32con.IDI_INFORMATION)
        self.notify_id = (
            self.hwnd,
            1,
            win32gui.NIF_ICON | win32gui.NIF_MESSAGE | win32gui.NIF_TIP,
            WM_APP_NOTIFY,
            icon,
            self._tooltip_text(),
        )
        win32gui.Shell_NotifyIcon(win32gui.NIM_MODIFY, self.notify_id)

    def _show_menu(self) -> None:
        self._refresh_tooltip()
        menu = win32gui.CreatePopupMenu()
        win32gui.AppendMenu(menu, win32con.MF_STRING, ID_START, "启动代理")
        win32gui.AppendMenu(menu, win32con.MF_STRING, ID_STOP, "停止代理")
        win32gui.AppendMenu(menu, win32con.MF_STRING, ID_STATUS, "查看状态")
        win32gui.AppendMenu(menu, win32con.MF_STRING, ID_SNAPSHOT, "状态快照")
        win32gui.AppendMenu(menu, win32con.MF_STRING, ID_OPEN_CHAT, "打开 Bridge 窗口")
        win32gui.AppendMenu(menu, win32con.MF_STRING, ID_OPEN_DATA, "打开数据目录")
        win32gui.AppendMenu(menu, win32con.MF_STRING, ID_INSTALL_STARTUP, "安装开机自启")
        win32gui.AppendMenu(menu, win32con.MF_STRING, ID_REMOVE_STARTUP, "移除开机自启")
        win32gui.AppendMenu(menu, win32con.MF_SEPARATOR, 0, "")
        win32gui.AppendMenu(menu, win32con.MF_STRING, ID_EXIT, "退出托盘")

        win32gui.SetForegroundWindow(self.hwnd)
        x, y = win32gui.GetCursorPos()
        win32gui.TrackPopupMenu(menu, win32con.TPM_LEFTALIGN, x, y, 0, self.hwnd, None)
        win32gui.PostMessage(self.hwnd, win32con.WM_NULL, 0, 0)

    def _message_box(self, text: str) -> None:
        win32gui.MessageBox(self.hwnd, text, "CARM", win32con.MB_OK)

    def _open_bridge_chat(self) -> None:
        subprocess.Popen(
            build_bridge_chat_command(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            cwd=self.repo_root,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

    def _handle_command(self, command_id: int) -> None:
        if command_id == ID_START:
            self.manager.start()
            self._show_snapshot()
        elif command_id == ID_STOP:
            self.manager.stop()
            self._show_snapshot()
        elif command_id in (ID_STATUS, ID_SNAPSHOT):
            self._show_snapshot()
        elif command_id == ID_OPEN_CHAT:
            self._open_bridge_chat()
        elif command_id == ID_OPEN_DATA:
            os.startfile(str(self.repo_root / "data"))
        elif command_id == ID_INSTALL_STARTUP:
            install_startup_shortcut()
            self._message_box("已安装开机自启。")
        elif command_id == ID_REMOVE_STARTUP:
            remove_startup_shortcut()
            self._message_box("已移除开机自启。")
        elif command_id == ID_EXIT:
            win32gui.DestroyWindow(self.hwnd)

    def _show_snapshot(self) -> None:
        self._message_box(format_status_snapshot(self.manager.status()))

    def _on_tray_notify(self, hwnd, msg, wparam, lparam):
        if lparam == win32con.WM_RBUTTONUP:
            self._show_menu()
        elif lparam == win32con.WM_LBUTTONDBLCLK:
            self._show_snapshot()
        return 0

    def _on_command(self, hwnd, msg, wparam, lparam):
        self._handle_command(win32api.LOWORD(wparam))
        return 0

    def _on_destroy(self, hwnd, msg, wparam, lparam):
        if self.notify_id:
            win32gui.Shell_NotifyIcon(win32gui.NIM_DELETE, self.notify_id)
        win32gui.PostQuitMessage(0)
        return 0


def main() -> int:
    return TrayApp().run()


if __name__ == "__main__":
    raise SystemExit(main())
