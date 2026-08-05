#!/usr/bin/env python3
"""量化 Change G1 query 守卫的精度。

背景
----
G1 在 param_name_echo 规则上加了一条守卫：如果值字面出现在 query 里，
就认为是用户给的、不是占位符幻觉。v22 承诺清单里的 2 个 loss 都由它造成：

  live_irrelevance_558-171-0  query 就是单词 "Version"，值 'Version'  -> 守卫放行
  live_irrelevance_598-193-0  query 含 "a movie ticket"，值 'movie'   -> 守卫放行

第一例守卫无从区分（用户字面就打了那个词），第二例是子串误命中
（movie 是领域中心词，不是用户提供的值）。

只有 2 个反例，不足以支撑改守卫规则 —— 那正是 Change J 的过拟合陷阱。
本脚本先回答"这条守卫一共托住了多少样本、其中判对判错各多少"，
把精度算出来，v23 才谈得上要不要动它。

用法:
    python scripts/diag_g1_guard_precision.py v21
"""
from __future__ import annotations

import ast
import glob
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import carm_bfcl_server_optimized as prod  # noqa: E402

DIAG = Path(__file__).resolve().parent.parent / "data" / "eval" / "diag"
PARAMS_LINE = re.compile(r"^\s*(\S+) params: (\{.*\})\s*$")


def strip_suffix(name: str) -> str:
    fn = getattr(prod, "_strip_name_suffix", None)
    return fn(name) if fn else str(name).strip().lower()


def rows(tag: str):
    for path in sorted(glob.glob(str(DIAG / f"*_{tag}.jsonl"))):
        cat = os.path.basename(path)[: -len(f"_{tag}.jsonl")]
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    yield cat, json.loads(line)
                except json.JSONDecodeError:
                    continue


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "v21"
    held = []  # 守卫生效（本来会被判 param_name_echo，因为值在 query 里而放行）
    for cat, r in rows(tag):
        query = str(r.get("query") or "").lower()
        if not query:
            continue
        for ln in r.get("trace") or []:
            m = PARAMS_LINE.match(ln)
            if not m:
                continue
            try:
                args = ast.literal_eval(m.group(2))
            except Exception:
                continue
            if not isinstance(args, dict):
                continue
            for k, v in args.items():
                if not isinstance(v, str) or not v.strip():
                    continue
                nv = v.strip().lower()
                echo = nv == str(k).strip().lower() or nv == strip_suffix(k)
                if echo and nv in query:
                    held.append(
                        {
                            "id": r.get("id"),
                            "cat": cat,
                            "param": k,
                            "value": v,
                            "correct": str(r.get("correct")) == "True",
                            # 值是否作为独立词出现，而不是只作子串
                            "word_bounded": re.search(
                                r"\b" + re.escape(nv) + r"\b", query
                            )
                            is not None,
                            "query_len": len(query),
                        }
                    )
    print(f"tag={tag}")
    print(f"守卫托住的 (样本,参数) 对: {len(held)}")
    uniq = {h["id"]: h for h in held}
    print(f"涉及样本数: {len(uniq)}")
    print()
    ok = sum(1 for h in uniq.values() if h["correct"])
    print(f"其中判对 {ok} / 判错 {len(uniq)-ok}"
          f"  -> 守卫精度 {ok/max(1,len(uniq)):.1%}")
    print()
    print("按类别:")
    c = Counter((h["cat"], h["correct"]) for h in uniq.values())
    for cat in sorted({k[0] for k in c}):
        print(f"  {cat:<24} 对 {c[(cat,True)]:>3} / 错 {c[(cat,False)]:>3}")
    print()
    print("独立词 vs 仅子串（子串命中是误放行的主要嫌疑形态）:")
    c2 = Counter((h["word_bounded"], h["correct"]) for h in uniq.values())
    for wb in (True, False):
        print(f"  word_bounded={str(wb):<6} 对 {c2[(wb,True)]:>3} / 错 {c2[(wb,False)]:>3}")
    print()
    bad = [h for h in uniq.values() if not h["correct"]]
    print(f"判错样本明细（最多 25 条，共 {len(bad)}）:")
    for h in sorted(bad, key=lambda x: x["cat"])[:25]:
        print(f"  {h['cat']:<22} {str(h['id']):<28} "
              f"{h['param']}={h['value']!r} wb={h['word_bounded']}")


if __name__ == "__main__":
    main()
