"""抽查指定样本的 trace，用于锁定 COUNT/ARG_FAIL 的失败机制。

用法：
  python scripts/diag_trace_inspect.py
"""
from __future__ import annotations
import json

TARGETS = {
    "data/eval/diag/parallel_multiple_v22.jsonl": [
        "parallel_multiple_31", "parallel_multiple_24", "parallel_multiple_32",
    ],
    "data/eval/diag/simple_java_v22.jsonl": ["simple_java_1", "simple_java_26"],
    "data/eval/diag/live_parallel_multiple_v21.jsonl": ["live_parallel_multiple_4-3-0"],
    "data/eval/diag/simple_javascript_v22.jsonl": ["simple_javascript_12"],
}

KEYWORDS = ("parallel", "split", "detect", "call", "function", "extract",
            "NAME", "PARAM", "requery", "count", "multi", "predict",
            "arg", "param", "tool_call", "candidate", "route")

def main():
    for path, ids in TARGETS.items():
        rows = {r["id"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}
        for sid in ids:
            r = rows[sid]
            print(f"\n##### {sid}")
            print("  query:", (r.get("query") or "")[:150])
            print("  GT_names:", r.get("gt_names"), " PRED_names:", r.get("pred_names"))
            for l in (r.get("trace") or [])[-16:]:
                s = str(l)
                if any(k in s for k in KEYWORDS):
                    print("   ", s[:220])

if __name__ == "__main__":
    main()
