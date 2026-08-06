"""Track A COUNT 失败子形态分析。

对每个 COUNT 失败样本，判定：
  direction: under (pred数<gt数) / over (pred数>gt数)
  over 细分:
    redundant : 同一函数名在 pred 中出现次数 > gt（如 invest 同时出负额+withdraw 风格的重复）
    spread    : pred 含 gt 没有的函数，或同一函数被套到多余实体
  under 细分:
    missing_func : gt 的某个函数名在 pred 中完全缺失
    merged      : 数量少但函数名都在（实体被合并）

用法：python scripts/diag_count_subpatterns.py
"""
from __future__ import annotations
import json
from collections import Counter
from diag_weakroot_v24 import load_gt, score, IRRELEVANCE_CATS, WEAK


def analyze(cat, r, gt_map):
    gt = gt_map.get(r["id"])
    if not gt:
        return None
    gt_names = []
    for it in gt:
        if isinstance(it, dict):
            gt_names += list(it.keys())
    pred_names = [c.get("name", "") for c in r["pred"]]
    gn, pn = len(gt_names), len(pred_names)
    if gn == pn:
        return None
    direction = "under" if pn < gn else "over"
    gcount = Counter(gt_names)
    pcount = Counter(pred_names)
    if direction == "over":
        redundant = any(pcount[f] > gcount[f] for f in pcount)
        extra_funcs = [f for f in pcount if f not in gcount]
        return ("over", "redundant" if redundant else ("extra_func" if extra_funcs else "spread"))
    else:
        missing = [f for f in gcount if pcount[f] < gcount[f]]
        return ("under", "missing_func" if missing else "merged")


def main():
    overall = Counter()
    examples = []
    for cat, path in WEAK.items():
        gt = load_gt(cat)
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        for r in rows:
            if r.get("error") or r.get("correct"):
                continue
            if score(r["pred"], gt.get(r["id"]), cat) != "COUNT":
                continue
            a = analyze(cat, r, gt)
            if not a:
                continue
            overall[a] += 1
            if len(examples) < 20:
                examples.append((cat, r["id"], a, r.get("pred_names"), r.get("gt_names")))
    print("=== Track A COUNT 子形态（55 例）===")
    for k, v in overall.most_common():
        print(f"   {k[0]:6s} / {k[1]:12s} : {v}")
    print("\n样例:")
    for cat, sid, a, pn, gn in examples:
        print(f"  {cat} {sid} [{a[0]}/{a[1]}]")
        print(f"     GT={gn}")
        print(f"     PRED={pn}")


if __name__ == "__main__":
    main()
