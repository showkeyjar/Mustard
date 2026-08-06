"""v24 弱项根因分类器。

用与 eval_bfcl_v4_fast.score_response 完全一致的判定逻辑（含 list 解包、
permutation 匹配），把弱项类别的失败样本归因为：
  MISS_ALL  预测为空列表（漏调）
  COUNT     调用数不对（并行/多调用检测失准）
  FUNC_SEL  函数名不对（路由/选择）
  ARG_FAIL  函数名与数量都对，但参数值不匹配（漏参数/值错/类型错）
  ERROR     传输/解析错误

用法：
  python scripts/diag_weakroot_v24.py
"""
from __future__ import annotations
import json, os
from collections import Counter, defaultdict
from itertools import permutations

BFCL_PA = r"D:\tools\miniconda3\envs\BFCL\Lib\site-packages\bfcl_eval\data\possible_answer"
IRRELEVANCE_CATS = {"irrelevance", "live_irrelevance"}

WEAK = {
    "parallel_multiple": "data/eval/diag/parallel_multiple_v22.jsonl",
    "live_parallel": "data/eval/diag/live_parallel_v21.jsonl",
    "live_parallel_multiple": "data/eval/diag/live_parallel_multiple_v21.jsonl",
    "simple_java": "data/eval/diag/simple_java_v22.jsonl",
    "simple_javascript": "data/eval/diag/simple_javascript_v22.jsonl",
}


def load_gt(cat: str) -> dict:
    m = {}
    p = os.path.join(BFCL_PA, f"BFCL_v4_{cat}.json")
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            it = json.loads(line)
            m[it["id"]] = it.get("ground_truth")
    return m


def _values_match(pred, gt):
    if pred is None and gt is None:
        return True
    if pred is None or gt is None:
        return False
    if isinstance(pred, dict) and isinstance(gt, dict):
        for k, gv in gt.items():
            pv = pred.get(k)
            if isinstance(gv, list) and len(gv) >= 1:
                if not any(_values_match(pv, x) for x in gv):
                    return False
            else:
                if not _values_match(pv, gv):
                    return False
        return True
    try:
        return float(pred) == float(gt)
    except (ValueError, TypeError):
        return pred == gt


def _params_match(pred_args: dict, gt_param_dict: dict) -> bool:
    for pn, gv in gt_param_dict.items():
        if not isinstance(gv, list) or not gv:
            continue
        pv = pred_args.get(pn)
        if any(_values_match(pv, x) for x in gv):
            continue
        if isinstance(pv, list) and _values_match(pv, gv):
            continue
        return False
    return True


def score(pc, gt, cat: str) -> str:
    if cat in IRRELEVANCE_CATS:
        return "OK" if (len(pc) > 0 if cat == "live_relevance" else len(pc) == 0) else "ARG_FAIL"
    if not gt:
        return "OK" if len(pc) == 0 else "MISS_ALL"
    gt_names = []
    gt_items = []
    for it in gt:
        if isinstance(it, dict):
            for fn, pr in it.items():
                gt_names.append(fn)
                gt_items.append((fn, pr if isinstance(pr, dict) else {}))
    pn = [c.get("name", "") for c in pc]
    if sorted(gt_names) != sorted(pn):
        return "COUNT" if len(gt_names) != len(pn) else "FUNC_SEL"
    gb = defaultdict(list)
    pb = defaultdict(list)
    for n, p in gt_items:
        gb[n].append(p)
    for c in pc:
        pb[c.get("name", "")].append(c.get("arguments", {}))
    for n, gpl in gb.items():
        ppl = pb.get(n, [])
        if len(gpl) != len(ppl):
            return "COUNT"
        ok = False
        for perm in permutations(range(len(ppl))):
            if all(_params_match(ppl[perm[i]], gp) for i, gp in enumerate(gpl)):
                ok = True
                break
        if not ok:
            return "ARG_FAIL"
    return "OK"


def main():
    totals = Counter()
    for cat, path in WEAK.items():
        gt = load_gt(cat)
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        c = Counter()
        for r in rows:
            if r.get("error"):
                c["ERROR"] += 1
                continue
            # 只在 jsonl 权威标记为 incorrect 的样本上做形态归类，
            # 避免复刻评分器比实际评测更严带来的计数噪声。
            if r.get("correct"):
                continue
            s = score(r["pred"], gt.get(r["id"]), cat)
            # 复刻器若判 OK，说明差异来自实际评测的额外归一化层，
            # 归为 NORM_DIFF 单独计数，不计入形态。
            if s == "OK":
                c["NORM_DIFF"] += 1
                continue
            c[s] += 1
            totals[s] += 1
        n = len(rows)
        inc = sum(v for k, v in c.items() if k not in ("ERROR", "NORM_DIFF"))
        print(f"\n=== {cat}: total={n} incorrect={inc} ===")
        for k, v in c.most_common():
            print(f"   {k:10s} {v}")
    print("\n=== 五类合计（按失败形态）===")
    for k, v in totals.most_common():
        print(f"   {k:10s} {v}")


if __name__ == "__main__":
    main()
