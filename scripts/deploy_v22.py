#!/usr/bin/env python3
"""v22 部署编排：重启服务 → 校验横幅 → 按承诺清单跑受影响类别 → 核对。

为什么要脚本而不是手敲
----------------------
需要跑哪些类别，唯一正确来源是承诺清单里出现过的类别集合。手工列举漏掉一个
类别，就等于把该类别的回退藏起来——这正是"承诺清单驱动"要防的事。

启动后必须确认横幅，因为进程行为不能从 git 提交时间反推（F2 归因事故的教训）。

用法:
    python scripts/deploy_v22.py --plan                # 只看计划，不动服务
    python scripts/deploy_v22.py --restart             # 停旧服务、起新服务、校验横幅
    python scripts/deploy_v22.py --eval                # 按清单跑评测
    python scripts/deploy_v22.py --verify              # 核对承诺
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "data" / "eval" / "diag"
PROMISE = DIAG / "promise_v22.json"
SERVER = ROOT / "scripts" / "carm_bfcl_server_optimized.py"
SERVER_LOG = ROOT / "scripts" / "carm_server_v22.log"
SERVER_ERR = ROOT / "scripts" / "carm_server_v22_err.log"
PORT = 11401

# 跑评测时希望看到的横幅状态。F2 必须是关的——v22 只测 G/H。
EXPECT_BANNER = {
    "G1 param_name_echo query guard": True,
    "G2 empty-dict allowed": True,
    "H  schema vocab snapping": True,
    "F2 recursive re-split": False,
}


def load_promise() -> dict:
    if not PROMISE.exists():
        sys.exit(f"缺少承诺清单 {PROMISE}，先跑 contract_change_gh.py --base v21 --out v22")
    return json.loads(PROMISE.read_text(encoding="utf-8"))


def promised_categories(promise: dict) -> list[str]:
    """承诺清单触及的全部类别。中性样本所在类别也要跑——'预期不变'同样是承诺。"""
    cats = {a["category"] for a in promise["affected"]}
    # 安全不变量所在类别也必须复测，否则无法证明门控没被放开
    for sid in promise.get("invariants_checked", []):
        m = re.match(r"^(.*?)_(?:[\d-]+)$", sid)
        if m:
            cats.add(m.group(1))
    return sorted(cats)


def find_server_pids() -> list[int]:
    try:
        out = subprocess.run(
            ["wmic", "process", "where",
             "name='python.exe'", "get", "ProcessId,CommandLine", "/format:csv"],
            capture_output=True, text=True, errors="ignore", timeout=30,
        ).stdout
    except Exception:
        out = ""
    pids = []
    for line in out.splitlines():
        if "carm_bfcl_server_optimized.py" in line:
            m = re.search(r",(\d+)\s*$", line.strip())
            if m:
                pids.append(int(m.group(1)))
    return pids


def cmd_plan(promise: dict) -> None:
    cats = promised_categories(promise)
    c = promise["counts"]
    print(f"承诺: gain {c['gain']} / loss {c['loss']} / neutral {c['neutral']}")
    print(f"受影响类别 ({len(cats)}): {','.join(cats)}")
    print()
    by_cat: dict[str, list] = {}
    for a in promise["affected"]:
        by_cat.setdefault(a["category"], []).append(a)
    for cat in cats:
        items = by_cat.get(cat, [])
        g = sum(1 for a in items if a["direction"] == "gain")
        n = sum(1 for a in items if a["direction"] == "neutral")
        why = "不变量复测" if not items else f"gain {g} / neutral {n}"
        print(f"  {cat:<26}{why}")
    print()
    print("运行中的服务进程:", find_server_pids() or "无")


def cmd_restart() -> None:
    old = find_server_pids()
    print(f"停止旧服务: {old or '无'}")
    for pid in old:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                       capture_output=True, text=True, errors="ignore")
    time.sleep(2)

    env = dict(os.environ)
    env.pop("CARM_ENABLE_F2", None)  # 显式确保 F2 关闭，不继承外部环境
    with open(SERVER_LOG, "w", encoding="utf-8") as so, \
         open(SERVER_ERR, "w", encoding="utf-8") as se:
        p = subprocess.Popen(
            [sys.executable, "-u", str(SERVER), "--port", str(PORT)],
            cwd=str(ROOT), stdout=so, stderr=se, env=env,
        )
    print(f"启动新服务 pid={p.pid}，等待横幅…")

    banner: dict[str, bool] = {}
    for _ in range(60):
        time.sleep(1)
        for path in (SERVER_LOG, SERVER_ERR, ROOT / "scripts" / "carm_server.log"):
            if not path.exists():
                continue
            txt = path.read_text(encoding="utf-8", errors="ignore")
            for m in re.finditer(r"\[([x ])\] (.+)$", txt, re.MULTILINE):
                banner[m.group(2).strip()] = m.group(1) == "x"
        if len(banner) >= len(EXPECT_BANNER):
            break

    if not banner:
        sys.exit("没读到启动横幅，服务可能启动失败。检查 " + str(SERVER_ERR))

    print("\n横幅读数:")
    bad = []
    for name, want in EXPECT_BANNER.items():
        got = banner.get(name)
        ok = got == want
        print(f"  [{'x' if got else ' '}] {name:<34} 期望 {want}  {'OK' if ok else '不符'}")
        if not ok:
            bad.append(name)
    if bad:
        sys.exit(f"\n横幅与期望不符: {bad}。中止，不要在这个配置下跑 v22。")
    print("\n横幅校验通过，服务口径 = v21 + G + H（F2 关闭）。")


def cmd_eval(promise: dict) -> None:
    cats = promised_categories(promise)
    log = DIAG / "run_v22.log"
    print(f"评测类别: {','.join(cats)}")
    print(f"日志: {log}")
    with open(log, "w", encoding="utf-8") as lf:
        p = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "scripts" / "diag_bfcl_v4.py"),
             "--categories", ",".join(cats), "--tag", "v22", "--workers", "12"],
            cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT,
        )
    print(f"已在后台启动 pid={p.pid}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    promise = load_promise()
    if args.plan or not any((args.restart, args.eval, args.verify)):
        cmd_plan(promise)
    if args.restart:
        cmd_restart()
    if args.eval:
        cmd_eval(promise)
    if args.verify:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "contract_change_gh.py"),
             "--base", "v21", "--verify", "v22"], cwd=str(ROOT))


if __name__ == "__main__":
    main()
