"""契约测试：Change M 的爆炸半径守卫必须是空操作。

为什么需要这个脚本：
  `_requery_value_rejected` 是在 v23 全量评测**跑完之后**加进服务端的。
  这意味着「实测配置」和「部署配置」不是同一份代码 —— 这正是
  「A/B 对比必须锁定基线配置」要防的坑。

  唯一能弥合这个缺口的证据是：守卫在所有已观测数据上都不生效。
  本脚本把这个证据固化成可重跑的断言，而不是一句口头保证。

数据源（两份，缺一不可）：
  1. data/eval/diag/format_requery_v22.json  —— 承诺清单赖以成立的 89 个重问值
  2. data/eval/diag/live_multiple_v23.jsonl  —— v23 实测运行真实产生的重问值

任何一处被拒 => 部署配置与实测配置行为不等价 => 承诺清单失效，必须重跑。

改动守卫阈值前必须先跑本脚本。
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from carm_bfcl_server_optimized import _requery_value_rejected  # noqa: E402

DIAG = Path(__file__).resolve().parent.parent / "data" / "eval" / "diag"
TRACE_LINE = re.compile(r"Format requery: \S+: (.+?) -> (.+)$")


def from_promise() -> list[tuple[str, str]]:
    p = DIAG / "format_requery_v22.json"
    if not p.exists():
        return []
    return [(r["id"], r["requery"])
            for r in json.loads(p.read_text(encoding="utf-8"))
            if r.get("requery") and r["requery"] != r.get("pred")]


def from_run(tag: str = "v23", cat: str = "live_multiple") -> list[tuple[str, str]]:
    p = DIAG / f"{cat}_{tag}.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.open(encoding="utf-8"):
        row = json.loads(line)
        for t in row.get("trace") or []:
            m = TRACE_LINE.search(str(t))
            if not m:
                continue
            try:
                val = ast.literal_eval(m.group(2))
            except Exception:
                val = m.group(2)
            out.append((row.get("id"), str(val)))
    return out


def main() -> int:
    sources = [
        ("承诺清单 (format_requery_v22.json)", from_promise()),
        ("v23 实测运行 (live_multiple_v23.jsonl)", from_run()),
    ]
    print("CONTRACT  Change M 守卫空操作性")
    print("=" * 72)
    failed = 0
    for label, vals in sources:
        if not vals:
            print(f"  [警告] {label}: 数据缺失，无法验证 —— 视为不通过")
            failed += 1
            continue
        rejected = [(i, v) for i, v in vals if _requery_value_rejected(v)]
        status = "空操作 ✓" if not rejected else f"拒绝 {len(rejected)} 个 ✗"
        print(f"  {label}: {len(vals)} 个值 -> {status}")
        for i, v in rejected[:10]:
            print(f"      {i}: {v!r}")
        failed += len(rejected)

    print()
    if failed:
        print(f"  不通过：守卫改变了 {failed} 个已观测值的行为。")
        print("  => 部署配置 != 实测配置，promise_v23_format.json 失效，必须重跑评测。")
        return 1
    print("  通过：守卫在全部已观测数据上均不生效，")
    print("        部署配置与 v23 实测配置在这些数据上行为等价。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
