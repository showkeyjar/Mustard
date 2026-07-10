"""CARM Interactive CLI — a friendly terminal interface.

Features:
  - Natural language routing with parallel support
  - Multi-turn conversation with anaphora resolution
  - Tool and confidence display
  - Evolution commands (/goal, /prefer, /evolve)
  - Session management (/new, /history)
  - Server mode (/serve)

Usage:
    python -m scripts.interactive_cli
    python -m scripts.interactive_cli --no-parallel
    python -m scripts.interactive_cli --session my-session
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("CARM_NO_EMBEDDING", "1")

from carm.router import CARMRouter
from carm.session_memory import SessionMemoryManager


def _format_result(result, elapsed_ms: float) -> str:
    """Format a RouteResult for terminal display."""
    lines = []

    # Tool badge
    tool_badge = f"[{result.tool_name}]"
    conf_badge = f"conf={result.confidence:.0%}"
    time_badge = f"{elapsed_ms:.0f}ms"
    lines.append(f"\n  {tool_badge} {conf_badge} {time_badge}")

    # Result text
    if result.sub_results:
        for sr in result.sub_results:
            lines.append(f"\n  ┌─ {sr.tool_name}")
            # Indent result text
            for line in sr.result.split("\n"):
                lines.append(f"  │ {line}")
            lines.append(f"  └─")
    else:
        for line in result.result.split("\n"):
            lines.append(f"  {line}")

    if not result.ok:
        lines.append(f"\n  ⚠ 执行未成功: {result.source}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="CARM Interactive CLI")
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel function call"
    )
    parser.add_argument(
        "--session", default=None, help="Session ID for multi-turn conversation"
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Enable semantic embedding (slower but more accurate)",
    )
    args = parser.parse_args()

    # Initialize router
    if args.embedding:
        os.environ.pop("CARM_NO_EMBEDDING", None)

    router = CARMRouter()
    session_id = args.session or f"cli_{int(time.time())}"
    use_parallel = not args.no_parallel

    print("╔══════════════════════════════════════════════════════╗")
    print("║          CARM Interactive CLI  v0.9.0                ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  输入自然语言查询，CARM 自动路由到最佳工具           ║")
    print("║  支持并行调用: '3+5, 7*8' 同时计算                  ║")
    print("║  支持多轮对话: '3+5' → '再加上10'                   ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  /help     查看帮助                                 ║")
    print("║  /tools    查看已注册工具                           ║")
    print("║  /new      开始新会话                               ║")
    print("║  /history  查看当前会话历史                         ║")
    print("║  /serve    启动 REST API 服务                       ║")
    print("║  /quit     退出                                     ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            return 0

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "/quit", "/exit"}:
            print("再见!")
            return 0

        if user_input == "/help":
            _print_help()
            continue

        if user_input == "/tools":
            print(f"\n  已注册工具: {router.tool_names}\n")
            continue

        if user_input == "/new":
            session_id = f"cli_{int(time.time())}"
            print(f"\n  新会话已创建: {session_id}\n")
            continue

        if user_input == "/history":
            _print_history(session_id)
            continue

        if user_input.startswith("/serve"):
            _start_server(router)
            continue

        if user_input.startswith("/goal "):
            print(f"\n  目标已记录: {user_input[6:]}\n")
            continue

        if user_input.startswith("/prefer "):
            parts = user_input[8:].strip().split(maxsplit=1)
            print(f"\n  工具偏好已记录: {parts[0] if parts else '未指定'}\n")
            continue

        # Route the query
        t0 = time.time()
        try:
            if use_parallel:
                result = router.route_parallel(user_input, session_id=session_id)
            else:
                result = router.route(user_input, session_id=session_id)
        except Exception as e:
            print(f"\n  ✗ 执行出错: {e}\n")
            continue
        elapsed_ms = (time.time() - t0) * 1000

        print(_format_result(result, elapsed_ms))
        print()


def _print_help():
    print("""
  CARM 交互式 CLI 帮助
  ─────────────────────
  直接输入查询即可，例如:
    3加5等于多少          → 计算器
    帮我写一个快速排序     → 代码执行
    什么是机器学习         → 搜索/大模型
    3+5, 7*8              → 并行计算

  多轮对话:
    > 3加5等于多少
    [calculator] 计算结果: 3 + 5 = 8
    > 再加上10
    [calculator] 计算结果: 8 + 10 = 18

  并行调用 (逗号分隔):
    > 100的平方根, 2的10次方
    [calculator] [calculator]
    ├ sqrt(100) = 10
    └ 2^10 = 1024

  命令:
    /tools    查看已注册工具
    /new      开始新会话
    /history  查看会话历史
    /serve    启动 REST API 服务
    /quit     退出
""")


def _print_history(session_id: str):
    mgr = SessionMemoryManager.get_instance()
    ctx = mgr.get_or_create(session_id)
    if not ctx.turns:
        print("\n  (无会话历史)\n")
        return
    print(f"\n  会话 {session_id} ({len(ctx.turns)} 轮):")
    for i, turn in enumerate(ctx.turns):
        print(f"  [{i + 1}] You: {turn.user_input[:60]}")
        print(f"      Tool: {turn.tool_name} → {turn.tool_result[:60]}")
    print()


def _start_server(router: CARMRouter):
    print("\n  启动 REST API 服务 (端口 8000)...")
    print("  按 Ctrl+C 停止\n")
    from scripts.carm_server import CARMHandler, HTTPServer
    import logging

    logging.basicConfig(level=logging.INFO)

    global _router
    import scripts.carm_server as srv

    srv._router = router

    server = HTTPServer(("0.0.0.0", 8000), CARMHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  服务已停止\n")
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
