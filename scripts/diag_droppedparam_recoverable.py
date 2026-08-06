"""量化 ARG_FAIL 中「被模型漏掉、但值可从 query 文找回」的必填参数占比。

判定逻辑（保守、可追溯）：
  对每个 GT 必填参数 param，若预测里该 key 缺失或值为 None，则视为「漏参数」。
  再检查 query 原文是否包含该参数的某个候选值（GT 列表里的任一值，转小写做子串匹配）。
  若命中，记为「可从 query 找回」；否则「不可找回（需模型本身补足）」。

用法：python scripts/diag_droppedparam_recoverable.py
"""
from __future__ import annotations
import json, os, re
from collections import Counter
from diag_weakroot_v24 import load_gt, _params_match, score, IRRELEVANCE_CATS, WEAK

def norm(v):
    return str(v).strip().lower()

def find_in_text(text, values):
    t = text.lower()
    for v in values:
        nv = norm(v)
        if nv and nv in t:
            return v
    return None

def main():
    dropped_total = 0
    recoverable = 0
    unrecoverable_examples = []
    for cat, path in WEAK.items():
        gt = load_gt(cat)
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        for r in rows:
            if r.get("error") or r.get("correct"):
                continue
            if score(r["pred"], gt.get(r["id"]), cat) != "ARG_FAIL":
                continue
            g = gt.get(r["id"])
            if not g:
                continue
            # GT params (list of {name: {param: [values]}})
            query = (r.get("query") or "")
            # build per-call GT param dicts
            for item in g:
                if not isinstance(item, dict):
                    continue
                for _fn, params in item.items():
                    if not isinstance(params, dict):
                        continue
                    for pname, gvals in params.items():
                        if not (isinstance(gvals, list) and gvals):
                            continue
                        # predicted value for this param
                        pv = None
                        for call in r["pred"]:
                            if call.get("name") == _fn:
                                pv = call.get("arguments", {}).get(pname)
                        if pv is None or (isinstance(pv, str) and pv == ""):
                            dropped_total += 1
                            hit = find_in_text(query, gvals)
                            if hit is not None:
                                recoverable += 1
                            else:
                                if len(unrecoverable_examples) < 12:
                                    unrecoverable_examples.append((cat, r["id"], pname, gvals))
    print(f"ARG_FAIL 中漏参数总数(观测): {dropped_total}")
    print(f"  其中可从 query 文找回:     {recoverable} ({recoverable/dropped_total:.0%})" if dropped_total else "n/a")
    print(f"  不可找回(需模型补足):      {dropped_total-recoverable}")
    print("\n不可找回样例(前12):")
    for cat, sid, pname, gvals in unrecoverable_examples:
        print(f"  {cat} {sid} param={pname} GT={gvals}")

if __name__ == "__main__":
    main()
