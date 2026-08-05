#!/usr/bin/env python3
"""暴露类别（无承诺清单）的 Change M 回归守卫。

对于 live_irrelevance / live_relevance 这类「只有结构暴露面、没有逐样本
收益承诺」的类别，Change M 上线后的验证标准是翻转方向而非承诺兑现：

    both_correct    v22 对 且 v23 对
    both_wrong      v22 错 且 v23 错
    adverse         v22 对 -> v23 错   <-- 引入的回归（契约违反）
    favorable       v22 错 -> v23 对   <-- 意外收益

传输错误：error 字段非 None 或 correct 缺失，这类样本不参与翻转判定。

安全不变量 = 无 adverse 翻转（Change M 不引入回归）。
任意 adverse 翻转 -> 进程退出码 1。

用法:
    python scripts/contract_category_flips.py --cat live_irrelevance --base v22 --verify v23
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DIAG = Path(__file__).resolve().parent.parent / "data" / "eval" / "diag"


def load_rows(tag: str, cat: str):
    path = DIAG / f"{cat}_{tag}.jsonl"
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            yield json.loads(line)


def ok_of(row: dict):
    """返回 True/False 表示判定，None 表示传输/执行错误不可比。"""
    if row.get("error"):
        return None
    c = row.get("correct")
    return c if isinstance(c, bool) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", required=True)
    ap.add_argument("--base", default="v22")
    ap.add_argument("--verify", default="v23")
    args = ap.parse_args()

    base = {r["id"]: r for r in load_rows(args.base, args.cat)}
    verify = {r["id"]: r for r in load_rows(args.verify, args.cat)}
    ids = set(base) & set(verify)

    both_c = both_w = adv = fav = 0
    base_err = verify_err = 0
    adv_ids: list[str] = []
    fav_ids: list[str] = []

    for i in sorted(ids):
        b = ok_of(base[i])
        v = ok_of(verify[i])
        if b is None:
            base_err += 1
        if v is None:
            verify_err += 1
        if b is None or v is None:
            continue
        if b and v:
            both_c += 1
        elif not b and not v:
            both_w += 1
        elif b and not v:
            adv += 1
            adv_ids.append(i)
        else:
            fav += 1
            fav_ids.append(i)

    print(f"FLIP  Change M  base={args.base}  verify={args.verify}  cat={args.cat}")
    print("=" * 78)
    print(f"  可比样本          : {len(ids)}")
    print(f"  双对 / 双错       : {both_c} / {both_w}")
    print(f"  favorable(错->对) : {fav}")
    print(f"  adverse(对->错)   : {adv}")
    print(f"  传输错误 base/verify: {base_err} / {verify_err}")
    print(f"  净翻转            : {fav - adv:+d}")
    if fav_ids:
        print(f"  favorable 样本    : {fav_ids}")
    if adv_ids:
        print(f"  !! adverse 样本   : {adv_ids}")

    if adv == 0:
        print("\n  安全不变量保持：无 adverse 翻转（Change M 未引入回归）。")
        raise SystemExit(0)
    else:
        print(f"\n  安全不变量被破坏：{adv} 个 adverse 翻转 —— 契约违反。")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
