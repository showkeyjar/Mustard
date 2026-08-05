#!/usr/bin/env python3
"""Change J（按 query 长度否决 LLM 相关性拒绝）的解冻判据。

背景
----
Change J 的想法：LLM 相关性验证器把 86 个样本判成 irrelevant 并返回 []。
其中 14 个是误拒（GT 非空），如果能识别出来就是白捡的分数。
在 240 条 irrelevance 上扫阈值，发现 char_len > 150 处「可救 6 / 破坏 0」。

为什么冻结
----------
irrelevance 这 240 条的 GT 全为空，"破坏 0" 只是说明这个子集里没有
「长 query + 正确拒绝」的组合——而 query 长度是数据的固有属性，不是模型
行为。live_irrelevance 有 884 条、12.9% 超过 150 字符（最长 10759），
子集上的"零破坏"根本不成立。所以必须在**全量同类数据**上重算。

判据（全部满足才解冻）
  1. 全量 rescue - damage 显著为正（不是个位数噪声）
  2. 存在一个阈值，其 damage 占正确拒绝总数的比例可接受
  3. 阈值不是只在某一个类别上成立

用法:
    python scripts/diag_change_j_threshold.py v21
    python scripts/diag_change_j_threshold.py v21 --thresholds 100,150,200,300,500
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

DIAG = Path(__file__).resolve().parent.parent / "data" / "eval" / "diag"

REJECT_MARK = "as irrelevant"


def load_rows(tag: str):
    for path in sorted(glob.glob(str(DIAG / f"*_{tag}.jsonl"))):
        cat = os.path.basename(path)[: -len(f"_{tag}.jsonl")]
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                yield cat, json.loads(line)
            except json.JSONDecodeError:
                continue


def collect(tag: str):
    """只保留被 LLM 相关性验证器拒掉的样本。

    Change J 只可能改变这些样本的结果——它不新增拒绝，只是撤销拒绝。
    所以其它样本无论 query 多长都与本改动无关，纳入统计只会稀释信号。
    """
    rejected = []
    total_by_cat = defaultdict(int)
    for cat, r in load_rows(tag):
        total_by_cat[cat] += 1
        traces = r.get("trace") or []
        if not any(REJECT_MARK in t for t in traces):
            continue
        gt = r.get("gt") or []
        rejected.append(
            {
                "id": r.get("id"),
                "cat": cat,
                "qlen": len(r.get("query") or ""),
                # GT 为空 = 本该拒绝，撤销拒绝会破坏它
                # GT 非空 = 误拒，撤销拒绝有机会救回（不保证，参数还得对）
                "gt_empty": len(gt) == 0,
                "correct": str(r.get("correct")) == "True",
            }
        )
    return rejected, dict(total_by_cat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--thresholds", default="80,100,120,150,200,250,300,400,500")
    args = ap.parse_args()

    rejected, totals = collect(args.tag)
    if not rejected:
        print(f"tag={args.tag}: 没有找到被相关性验证器拒绝的样本。")
        return

    cats = sorted({r["cat"] for r in rejected})
    n_correct_reject = sum(1 for r in rejected if r["gt_empty"])
    n_wrong_reject = len(rejected) - n_correct_reject

    print(f"tag = {args.tag}   类别 {len(totals)} 个，样本 {sum(totals.values())} 条")
    print(f"相关性验证器拒绝 : {len(rejected)}")
    print(f"  其中正确拒绝   : {n_correct_reject}   (GT 为空，撤销会破坏)")
    print(f"  其中误拒       : {n_wrong_reject}   (GT 非空，撤销才有机会救)")
    print()

    print("拒绝样本按类别分布")
    print(f"  {'category':<24}{'rejected':>10}{'correct':>10}{'wrong':>8}{'类别总量':>10}")
    for c in cats:
        rs = [r for r in rejected if r["cat"] == c]
        print(
            f"  {c:<24}{len(rs):>10}{sum(1 for r in rs if r['gt_empty']):>10}"
            f"{sum(1 for r in rs if not r['gt_empty']):>8}{totals.get(c, 0):>10}"
        )
    print()

    print("阈值扫描：query 字符数 > T 时撤销拒绝")
    print(f"  {'T':>6}{'rescue上界':>12}{'damage':>10}{'净值':>10}   {'damage 落在'}")
    print("  " + "-" * 74)
    for t in [int(x) for x in args.thresholds.split(",")]:
        hits = [r for r in rejected if r["qlen"] > t]
        rescue = sum(1 for r in hits if not r["gt_empty"])
        damage = sum(1 for r in hits if r["gt_empty"])
        dmg_cats = sorted({r["cat"] for r in hits if r["gt_empty"]})
        print(
            f"  {t:>6}{rescue:>12}{damage:>10}{rescue - damage:>+10}   "
            f"{','.join(dmg_cats) if dmg_cats else '-'}"
        )
    print()
    print("  注：rescue 是**上界**。撤销拒绝只是让调用被输出，函数名和参数还得")
    print("      全对才算分。damage 是**确定值**——正确拒绝一旦撤销必然失分。")
    print()

    # query 长度在两类拒绝上的分布，用来判断这个特征本身有没有区分度
    for label, sel in (("正确拒绝", True), ("误拒", False)):
        lens = sorted(r["qlen"] for r in rejected if r["gt_empty"] is sel)
        if not lens:
            print(f"  {label}: 无样本")
            continue
        n = len(lens)
        q = lambda p: lens[min(n - 1, int(p * n))]  # noqa: E731
        print(
            f"  {label:<6} n={n:<4} min={lens[0]:<5} p25={q(.25):<5} "
            f"中位={q(.5):<5} p75={q(.75):<5} p90={q(.9):<5} max={lens[-1]}"
        )


if __name__ == "__main__":
    main()
